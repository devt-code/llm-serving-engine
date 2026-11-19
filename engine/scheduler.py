"""Decides, at every step, which requests run.

Two policies, same engine:

  static      The classic approach (and what HF generate() does with a list of prompts).
              Fill a batch, then decode until EVERY request in it is done. Requests that finish
              early keep their memory and keep being computed (their output is discarded), and
              new requests wait for the whole batch to drain.

  continuous  "Iteration-level scheduling" (Orca, OSDI '22; used by vLLM). The batch is
              rebuilt at every step: finished requests leave right away and free their memory,
              waiting requests join as soon as there's room.

Each step is either a PREFILL step (process new prompts) or a DECODE step (one token for every
running request). New prompts are prioritized: if anything can be admitted, this step is a
prefill. That keeps time-to-first-token low; the cost is that running requests pause for one
step whenever someone joins. (Mixing prefill and decode tokens in one step, "chunked prefill",
removes that pause; it's a possible later optimization.)

Memory is owned by a memory manager (SlotManager for the Phase 2 slot cache, BlockManager for
the Phase 3 paged cache). Before a decode step, every running request must have room for one
more token. With the paged cache that can fail, and then the scheduler PREEMPTS: it takes the
most recently admitted running request, frees all its memory, and puts it back at the front of
the waiting queue. When it's readmitted, its prompt plus everything it had generated is
prefilled again (recomputation), and decoding resumes where it left off. Newest-first is the
right victim: it has the least work to redo, and the oldest requests keep making progress.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Literal

from engine.request import Request, RequestStatus


@dataclass
class ScheduledStep:
    kind: Literal["prefill", "decode"]
    requests: list[Request]
    preempted: list[Request] = field(default_factory=list)


class Scheduler:
    def __init__(
        self,
        memory,  # SlotManager or BlockManager
        max_num_seqs: int,
        max_num_batched_tokens: int,
        policy: Literal["continuous", "static"] = "continuous",
    ):
        if policy not in ("continuous", "static"):
            raise ValueError(f"unknown policy {policy!r}")
        self.memory = memory
        self.max_num_seqs = max_num_seqs  # cap on running requests
        self.max_num_batched_tokens = max_num_batched_tokens  # prompt-token budget per prefill step
        self.policy = policy
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []  # in admission order: oldest first
        self._static_filling = True  # static policy: still filling the current batch?
        self.num_preemptions = 0

    def add(self, req: Request) -> None:
        self.waiting.append(req)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> ScheduledStep | None:
        if self.policy == "static" and not self._static_filling:
            return self._schedule_decode() if self.running else None

        admitted = self._admit()
        if admitted:
            return ScheduledStep("prefill", admitted)
        if self.policy == "static":
            self._static_filling = False  # batch is full (or queue empty): decode it to the end
        if self.running:
            return self._schedule_decode()
        return None

    def _admit(self) -> list[Request]:
        """Move waiting requests into running while memory, slots and the token budget allow (FCFS).

        Strict FCFS: if the request at the head of the queue doesn't fit, nobody behind it
        jumps ahead. That keeps a preempted request (put back at the head) from starving.
        """
        admitted, budget = [], self.max_num_batched_tokens
        while self.waiting and len(self.running) < self.max_num_seqs:
            req = self.waiting[0]
            n = len(req.all_token_ids)
            # Only tokens that aren't already in the prefix cache cost prefill compute.
            to_compute = n - self.memory.num_cached_tokens(req)
            if to_compute > budget or not self.memory.can_admit(req, n):
                break
            self.waiting.popleft()
            # allocate() returns how many leading tokens are already cached (prefix hit);
            # the prefill step only computes the rest.
            req.num_computed = self.memory.allocate(req, n)
            req.status = RequestStatus.RUNNING
            budget -= n - req.num_computed
            admitted.append(req)
            self.running.append(req)
        return admitted

    def _schedule_decode(self) -> ScheduledStep:
        """Make room for one more token per running request, preempting if memory runs out."""
        preempted = []
        i = 0
        while i < len(self.running):
            req = self.running[i]
            # The next decode writes the last sampled token at position num_computed.
            if req.is_finished or self.memory.ensure_capacity(req, req.num_computed + 1):
                i += 1
                continue
            victim = self.running.pop()  # newest; may be `req` itself, which ends the loop
            self._preempt(victim)
            preempted.append(victim)
        return ScheduledStep("decode", list(self.running), preempted)

    def _preempt(self, req: Request) -> None:
        self.memory.free(req)
        req.status = RequestStatus.WAITING
        req.num_computed = 0
        req.num_preemptions += 1
        self.num_preemptions += 1
        # Front of the queue. Victims are taken newest-first, so after several preemptions
        # in one step the older one ends up first, preserving arrival order.
        self.waiting.appendleft(req)

    def release_finished(self) -> list[Request]:
        """Called after each step. Frees memory of finished requests and returns them."""
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
            self.memory.free(r)
        return done

    def abort(self, req: Request) -> bool:
        """Stop a request immediately (e.g. its client disconnected). Returns False if it was
        already finished. Must be called between steps, from the thread that runs step()."""
        if req.is_finished:
            return False
        if req in self.waiting:
            self.waiting.remove(req)
        elif req in self.running:
            self.running.remove(req)
            self.memory.free(req)
        req.status = RequestStatus.FINISHED
        req.finish_reason = "abort"
        return True
