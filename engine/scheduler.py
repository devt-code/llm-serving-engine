"""Decides, at every step, which requests run.

Two policies, same engine:

  static      The classic approach (and what HF generate() does with a list of prompts).
              Fill a batch, then decode until EVERY request in it is done. Requests that finish
              early keep their slot and keep being computed (their output is discarded), and
              new requests wait for the whole batch to drain.

  continuous  "Iteration-level scheduling" (Orca, OSDI '22; used by vLLM). The batch is
              rebuilt at every step: finished requests leave right away and free their slot,
              waiting requests join as soon as a slot is free.

Each step is either a PREFILL step (process new prompts) or a DECODE step (one token for every
running request). New prompts are prioritized: if anything can be admitted, this step is a
prefill. That keeps time-to-first-token low; the cost is that running requests pause for one
step whenever someone joins. (Mixing prefill and decode tokens in one step, "chunked prefill",
removes that pause; it's a possible later optimization.)
"""

import heapq
from collections import deque
from dataclasses import dataclass
from typing import Literal

from engine.request import Request, RequestStatus


@dataclass
class ScheduledStep:
    kind: Literal["prefill", "decode"]
    requests: list[Request]


class Scheduler:
    def __init__(
        self,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        policy: Literal["continuous", "static"] = "continuous",
    ):
        if policy not in ("continuous", "static"):
            raise ValueError(f"unknown policy {policy!r}")
        self.max_num_seqs = max_num_seqs  # = number of KV cache slots
        self.max_num_batched_tokens = max_num_batched_tokens  # prompt-token budget per prefill step
        self.policy = policy
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.free_slots = list(range(max_num_seqs))  # min-heap: lowest free slot first
        self._static_filling = True  # static policy: still filling the current batch?

    def add(self, req: Request) -> None:
        self.waiting.append(req)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> ScheduledStep | None:
        if self.policy == "static" and not self._static_filling:
            return ScheduledStep("decode", self.running) if self.running else None

        admitted = self._admit()
        if admitted:
            return ScheduledStep("prefill", admitted)
        if self.policy == "static":
            self._static_filling = False  # batch is full (or queue empty): decode it to the end
        if self.running:
            return ScheduledStep("decode", self.running)
        return None

    def _admit(self) -> list[Request]:
        """Move waiting requests into running while slots and the token budget allow (FCFS)."""
        admitted, budget = [], self.max_num_batched_tokens
        while self.waiting and self.free_slots and len(self.waiting[0].prompt_ids) <= budget:
            req = self.waiting.popleft()
            req.slot = heapq.heappop(self.free_slots)
            req.status = RequestStatus.RUNNING
            budget -= len(req.prompt_ids)
            admitted.append(req)
            self.running.append(req)
        return admitted

    def release_finished(self) -> list[Request]:
        """Called after each step. Returns the requests whose slots were freed."""
        if self.policy == "static":
            # Static: nobody leaves until the whole batch is done.
            if not self.running or not all(r.is_finished for r in self.running):
                return []
            done, self.running = self.running, []
            self._static_filling = True
        else:
            done = [r for r in self.running if r.is_finished]
            self.running = [r for r in self.running if not r.is_finished]
        for r in done:
            heapq.heappush(self.free_slots, r.slot)
        return done
