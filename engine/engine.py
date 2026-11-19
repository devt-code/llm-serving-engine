"""The serving engine loop: schedule -> build batch -> forward -> sample -> update.

    engine = LLMEngine(model, kv_cache="paged", kv_cache_gb=2.0)
    engine.add_request(prompt_ids, SamplingParams(max_new_tokens=64))
    while engine.has_unfinished():
        finished = engine.step()

Every call to step() is one forward pass of the model:
  - a PREFILL step packs the sequences of newly admitted requests into one flat sequence, or
  - a DECODE step feeds the last sampled token of every running request (one token each).

KV cache types:
  - "paged" (Phase 3, default): shared pool of 16-token blocks, allocated as requests grow,
    with preemption when the pool runs out. Size it with kv_cache_gb or num_kv_blocks.
  - "slot"  (Phase 2): one max_model_len region per request; max_num_seqs slots.
"""

from typing import Literal

import torch

from engine.batch_cache import SlotKVCache, SlotManager
from engine.events import EventLog
from engine.kv_cache import BlockManager, PagedKVCache
from engine.model import Qwen3ForCausalLM
from engine.request import Request, RequestStatus, SamplingParams
from engine.sampler import sample
from engine.scheduler import Scheduler


class LLMEngine:
    def __init__(
        self,
        model: Qwen3ForCausalLM,
        max_num_seqs: int = 32,
        max_model_len: int = 512,
        max_num_batched_tokens: int = 2048,
        policy: Literal["continuous", "static"] = "continuous",
        kv_cache: Literal["paged", "slot"] = "paged",
        kv_cache_gb: float | None = None,
        num_kv_blocks: int | None = None,
        block_size: int = 16,
        enable_prefix_caching: bool = True,
        event_log: EventLog | None = None,
    ):
        if max_num_batched_tokens < max_model_len:
            raise ValueError("max_num_batched_tokens must be >= max_model_len, or long prompts could never run")
        self.model = model
        self.device = model.lm_head.weight.device
        dtype = model.lm_head.weight.dtype
        self.max_model_len = max_model_len
        self.policy = policy
        self.events = event_log or EventLog()
        config = model.config

        if kv_cache == "slot":
            self.cache = SlotKVCache(config, max_num_seqs, max_model_len, self.device, dtype)
            self.memory = SlotManager(self.cache)
        elif kv_cache == "paged":
            if num_kv_blocks is None:
                gb = kv_cache_gb if kv_cache_gb is not None else 2.0
                num_kv_blocks = int(gb * 1e9) // PagedKVCache.bytes_per_block(config, block_size, dtype)
            self.cache = PagedKVCache(config, num_kv_blocks, block_size, self.device, dtype)
            self.memory = BlockManager(self.cache, self.events, enable_prefix_caching=enable_prefix_caching)
        else:
            raise ValueError(f"unknown kv_cache {kv_cache!r}")

        self.scheduler = Scheduler(self.memory, max_num_seqs, max_num_batched_tokens, policy)
        self.stop_token_ids = set(config.stop_token_ids or (config.eos_token_id,))
        self.step_count = 0
        self.peak_running = 0
        self.events.emit(
            "engine_started", policy=policy, kv_cache=kv_cache, max_num_seqs=max_num_seqs,
            max_model_len=max_model_len, kv_cache_bytes=self.cache.nbytes, device=str(self.device),
            block_size=block_size if kv_cache == "paged" else None,
            **self.memory.usage(),
        )

    # ------------------------------------------------------------------ public API

    def add_request(
        self, prompt_ids: list[int], params: SamplingParams = SamplingParams(), request_id: str | None = None
    ) -> Request:
        if not prompt_ids:
            raise ValueError("empty prompt")
        limit = self.max_request_len
        if len(prompt_ids) + params.max_new_tokens > limit:
            raise ValueError(
                f"prompt ({len(prompt_ids)}) + max_new_tokens ({params.max_new_tokens}) exceeds the "
                f"per-request limit ({limit} tokens)"
            )
        req = Request(prompt_ids=list(prompt_ids), params=params, arrival_time=self.events.now())
        if request_id is not None:
            req.request_id = request_id
        if not params.greedy:
            req.generator = torch.Generator(device=self.device)
            if params.seed is None:
                req.generator.seed()
            else:
                req.generator.manual_seed(params.seed)
        self.scheduler.add(req)
        self.events.emit(
            "request_arrived", id=req.request_id, prompt_len=len(prompt_ids), max_new_tokens=params.max_new_tokens
        )
        return req

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

    @property
    def max_request_len(self) -> int:
        """Longest prompt + max_new_tokens a single request may use."""
        return min(self.max_model_len, self.memory.max_request_tokens)

    def abort_request(self, req: Request) -> None:
        """Cancel a request and free its memory. Call between steps."""
        if self.scheduler.abort(req):
            req.finish_time = self.events.now()
            self.events.emit("request_finished", id=req.request_id, output_len=len(req.output_ids),
                             reason="abort", num_preemptions=req.num_preemptions)

    def generate(
        self, prompts: list[list[int]], params: SamplingParams | list[SamplingParams] = SamplingParams()
    ) -> list[list[int]]:
        """Convenience: submit everything, run to completion, return outputs in input order."""
        if isinstance(params, SamplingParams):
            params = [params] * len(prompts)
        reqs = [self.add_request(p, sp) for p, sp in zip(prompts, params)]
        while self.has_unfinished():
            self.step()
        return [r.output_ids for r in reqs]

    # ------------------------------------------------------------------ one iteration

    @torch.inference_mode()
    def step(self) -> list[Request]:
        """Run one forward pass. Returns the requests that finished during this step."""
        t_start = self.events.now()
        scheduled = self.scheduler.schedule()
        if scheduled is None:
            return []
        for r in scheduled.preempted:
            self.events.emit("request_preempted", id=r.request_id, num_computed_lost=len(r.all_token_ids) - 1)
        reqs = scheduled.requests
        if not reqs:  # everyone got preempted this step; the next step re-admits
            return []

        if scheduled.kind == "prefill":
            computed_before = sum(r.num_computed for r in reqs)
            logits = self._prefill(reqs)
            tokens_computed = sum(r.num_computed for r in reqs) - computed_before
        else:
            logits = self._decode(reqs)
            tokens_computed = len(reqs)

        next_tokens = self._sample(reqs, logits)  # .tolist() inside waits for the GPU
        finished = self._append_tokens(reqs, next_tokens)
        self.peak_running = max(self.peak_running, len(self.scheduler.running))
        self.scheduler.release_finished()

        self.events.emit(
            "step", step=self.step_count, kind=scheduled.kind, num_seqs=len(reqs),
            num_tokens=int(tokens_computed),
            running=[r.request_id for r in self.scheduler.running],
            waiting=len(self.scheduler.waiting),
            num_preempted=len(scheduled.preempted),
            duration_ms=round((self.events.now() - t_start) * 1000, 3),
            **self.memory.usage(),
        )
        self.step_count += 1
        return finished

    def _prefill(self, reqs: list[Request]) -> torch.Tensor:
        # Compute every token that isn't in the cache yet. For a new request that's the prompt
        # minus any cached prefix; for a preempted one it's prompt + already-generated tokens
        # (minus any prefix that survived in the cache).
        seqs = [r.all_token_ids for r in reqs]
        new = [s[r.num_computed :] for r, s in zip(reqs, seqs)]
        for r in reqs:
            kind = "request_resumed" if r.num_preemptions else "request_started"
            blocks = list(r.block_table) if r.block_table is not None else None  # copy: the table grows later
            self.events.emit(kind, id=r.request_id, slot=r.slot, blocks=blocks, cached_tokens=r.num_computed)
        batch = self.memory.prefill_batch(reqs, [len(n) for n in new], [len(s) for s in seqs])
        input_ids = torch.tensor([t for n in new for t in n], device=self.device)
        logits = self.model(input_ids, batch.positions, batch, logits_indices=batch.last_token_idx)  # [B, V]
        for r, s in zip(reqs, seqs):
            r.num_computed = len(s)
            self.memory.commit(r)
        return logits

    def _decode(self, reqs: list[Request]) -> torch.Tensor:
        # The last sampled token isn't in the cache yet; it goes at position num_computed.
        # (Static batching only: a request that already finished keeps rewriting its last
        # position. Its compute is wasted, which is the point of the comparison.)
        positions = [r.num_computed for r in reqs]
        batch = self.memory.decode_batch(reqs, [p + 1 for p in positions])
        input_ids = torch.tensor([r.last_token for r in reqs], device=self.device)
        logits = self.model(input_ids, torch.tensor(positions, device=self.device), batch)  # [B, V]
        for r in reqs:
            if not r.is_finished:
                r.num_computed += 1
                self.memory.commit(r)
        return logits

    def _sample(self, reqs: list[Request], logits: torch.Tensor) -> list[int]:
        tokens = logits.argmax(dim=-1)  # greedy for everyone in one kernel
        for i, r in enumerate(reqs):
            # Non-greedy requests sample individually with their own generator, so a seeded
            # request gives the same tokens no matter what else is in the batch.
            if not r.params.greedy and not r.is_finished:
                tokens[i] = sample(logits[i : i + 1], r.params.temperature, r.params.top_p, r.generator)[0]
        return tokens.tolist()

    def _append_tokens(self, reqs: list[Request], next_tokens: list[int]) -> list[Request]:
        now = self.events.now()
        finished = []
        for r, tok in zip(reqs, next_tokens):
            if r.is_finished:
                continue  # static batching: still computed, output discarded
            r.output_ids.append(tok)
            if r.first_token_time is None:
                r.first_token_time = now
                self.events.emit("first_token", id=r.request_id)
            reason = None
            if (tok in self.stop_token_ids or tok in r.params.stop_token_ids) and not r.params.ignore_eos:
                reason = "stop"
            elif len(r.output_ids) >= r.params.max_new_tokens:
                reason = "length"
            if reason is not None:
                r.status = RequestStatus.FINISHED
                r.finish_time = now
                r.finish_reason = reason
                finished.append(r)
                self.events.emit(
                    "request_finished", id=r.request_id, output_len=len(r.output_ids), reason=reason,
                    num_preemptions=r.num_preemptions,
                )
        return finished
