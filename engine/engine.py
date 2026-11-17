"""The serving engine loop: schedule -> build batch -> forward -> sample -> update.

    engine = LLMEngine(model, max_num_seqs=32, policy="continuous")
    engine.add_request(prompt_ids, SamplingParams(max_new_tokens=64))
    while engine.has_unfinished():
        finished = engine.step()

Every call to step() is one forward pass of the model:
  - a PREFILL step packs the prompts of newly admitted requests into one flat sequence, or
  - a DECODE step feeds the last sampled token of every running request (one token each).
"""

from typing import Literal

import torch

from engine.batch_cache import DecodeBatch, PrefillBatch, SlotKVCache
from engine.events import EventLog
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
        event_log: EventLog | None = None,
    ):
        if max_num_batched_tokens < max_model_len:
            raise ValueError("max_num_batched_tokens must be >= max_model_len, or long prompts could never run")
        self.model = model
        self.device = model.lm_head.weight.device
        dtype = model.lm_head.weight.dtype
        self.max_model_len = max_model_len
        self.policy = policy
        self.cache = SlotKVCache(model.config, max_num_seqs, max_model_len, self.device, dtype)
        self.scheduler = Scheduler(max_num_seqs, max_num_batched_tokens, policy)
        self.events = event_log or EventLog()
        self.eos = model.config.eos_token_id
        self.step_count = 0
        self.events.emit(
            "engine_started", policy=policy, max_num_seqs=max_num_seqs, max_model_len=max_model_len,
            kv_cache_bytes=self.cache.nbytes, device=str(self.device),
        )

    # ------------------------------------------------------------------ public API

    def add_request(
        self, prompt_ids: list[int], params: SamplingParams = SamplingParams(), request_id: str | None = None
    ) -> Request:
        if not prompt_ids:
            raise ValueError("empty prompt")
        if len(prompt_ids) + params.max_new_tokens > self.max_model_len:
            raise ValueError(
                f"prompt ({len(prompt_ids)}) + max_new_tokens ({params.max_new_tokens}) exceeds "
                f"max_model_len ({self.max_model_len})"
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
        reqs = scheduled.requests

        if scheduled.kind == "prefill":
            logits = self._prefill(reqs)
        else:
            logits = self._decode(reqs)

        next_tokens = self._sample(reqs, logits)  # .tolist() inside waits for the GPU
        finished = self._append_tokens(reqs, next_tokens)
        self.scheduler.release_finished()

        self.events.emit(
            "step", step=self.step_count, kind=scheduled.kind, num_seqs=len(reqs),
            num_tokens=sum(len(r.prompt_ids) for r in reqs) if scheduled.kind == "prefill" else len(reqs),
            running=[r.request_id for r in self.scheduler.running],
            slots=[r.slot for r in self.scheduler.running],
            waiting=len(self.scheduler.waiting),
            duration_ms=round((self.events.now() - t_start) * 1000, 3),
        )
        self.step_count += 1
        return finished

    def _prefill(self, reqs: list[Request]) -> torch.Tensor:
        for r in reqs:
            self.events.emit("request_started", id=r.request_id, slot=r.slot)
        batch = PrefillBatch(self.cache, [r.slot for r in reqs], [len(r.prompt_ids) for r in reqs])
        input_ids = torch.tensor([t for r in reqs for t in r.prompt_ids], device=self.device)
        logits = self.model(input_ids, batch.positions, batch, logits_indices=batch.last_token_idx)  # [B, V]
        for r in reqs:
            r.num_computed = len(r.prompt_ids)
        return logits

    def _decode(self, reqs: list[Request]) -> torch.Tensor:
        # The last sampled token isn't in the cache yet; it goes at position num_computed.
        # (Static batching only: a request that already finished keeps rewriting its last
        # position. Its compute is wasted, which is the point of the comparison.)
        positions = [r.num_computed for r in reqs]
        batch = DecodeBatch(self.cache, [r.slot for r in reqs], [p + 1 for p in positions])
        input_ids = torch.tensor([r.last_token for r in reqs], device=self.device)
        logits = self.model(input_ids, torch.tensor(positions, device=self.device), batch)  # [B, V]
        for r in reqs:
            if not r.is_finished:
                r.num_computed += 1
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
                continue  # static batching: slot still computed, output discarded
            r.output_ids.append(tok)
            if r.first_token_time is None:
                r.first_token_time = now
                self.events.emit("first_token", id=r.request_id)
            reason = None
            if tok == self.eos and not r.params.ignore_eos:
                reason = "stop"
            elif len(r.output_ids) >= r.params.max_new_tokens:
                reason = "length"
            if reason is not None:
                r.status = RequestStatus.FINISHED
                r.finish_time = now
                r.finish_reason = reason
                finished.append(r)
                self.events.emit(
                    "request_finished", id=r.request_id, output_len=len(r.output_ids), reason=reason, slot=r.slot
                )
        return finished
