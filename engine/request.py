"""A generation request and its lifecycle: WAITING -> RUNNING -> FINISHED."""

import itertools
from dataclasses import dataclass, field
from enum import Enum

import torch


@dataclass(frozen=True)
class SamplingParams:
    max_new_tokens: int = 128
    temperature: float = 0.0  # 0 = greedy
    top_p: float = 1.0
    seed: int | None = None
    ignore_eos: bool = False  # benchmarks: always generate exactly max_new_tokens
    stop_token_ids: tuple[int, ...] = ()  # extra token ids that end this request

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0


class RequestStatus(Enum):
    WAITING = "waiting"  # queued, no KV cache yet
    RUNNING = "running"  # owns a cache slot, decoding
    FINISHED = "finished"


_ids = itertools.count()


@dataclass
class Request:
    prompt_ids: list[int]
    params: SamplingParams
    arrival_time: float
    request_id: str = field(default_factory=lambda: f"req-{next(_ids)}")

    status: RequestStatus = RequestStatus.WAITING
    output_ids: list[int] = field(default_factory=list)
    slot: int | None = None  # slot cache (Phase 2)
    block_table: list[int] | None = None  # paged cache (Phase 3): logical block -> physical block
    block_hashes: list[int] = field(default_factory=list)  # prefix-cache hashes of its full blocks
    # How many of this request's tokens have their K/V in the cache. After prefill this is
    # len(prompt); each decode step adds one (the previously sampled token). Reset to 0 when
    # the request is preempted (its cache is thrown away and recomputed later).
    num_computed: int = 0
    num_preemptions: int = 0
    first_token_time: float | None = None
    finish_time: float | None = None
    finish_reason: str | None = None  # "stop" (EOS) or "length" (hit max_new_tokens)
    generator: torch.Generator | None = None  # per-request RNG, so seeds don't depend on the batch

    @property
    def last_token(self) -> int:
        return self.output_ids[-1]

    @property
    def all_token_ids(self) -> list[int]:
        """Prompt plus everything generated so far: what a (re)prefill has to process."""
        return self.prompt_ids + self.output_ids

    @property
    def total_len(self) -> int:
        return len(self.prompt_ids) + self.params.max_new_tokens

    @property
    def is_finished(self) -> bool:
        return self.status == RequestStatus.FINISHED
