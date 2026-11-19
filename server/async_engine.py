"""Runs the (synchronous) LLMEngine on a background thread and exposes it to asyncio.

    HTTP handlers (asyncio)                         engine thread
    ───────────────────────                         ─────────────
    async for out in engine.generate(...):   ──►    command queue ──► engine.add_request
        ...                                  ◄──    after each step(): new tokens pushed to
                                                    that request's asyncio.Queue
    (client disconnects -> generator closes) ──►    "abort" command ──► engine.abort_request

Only the engine thread ever touches LLMEngine, so the engine needs no locks. The only
cross-thread handoffs are queue.Queue (handlers -> engine) and loop.call_soon_threadsafe
(engine -> handlers).
"""

import asyncio
import itertools
import queue
import threading
import time
from dataclasses import dataclass
from typing import AsyncIterator

from engine.engine import LLMEngine
from engine.request import Request, SamplingParams


@dataclass
class StepOutput:
    """New tokens for one request since the last update."""

    new_token_ids: list[int]
    finished: bool
    finish_reason: str | None
    num_prompt_tokens: int
    num_output_tokens: int


@dataclass
class _Stream:
    req: Request
    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue
    sent: int = 0


class AsyncEngine:
    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self._commands: queue.Queue = queue.Queue()
        self._streams: dict[str, _Stream] = {}  # engine thread only
        self._ids = itertools.count()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="engine", daemon=True)
        self.started_at = time.time()
        self.total_output_tokens = 0
        self.total_requests = 0
        self.error: BaseException | None = None

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        self._commands.put(("wake",))
        self._thread.join(timeout=10)

    @property
    def max_request_len(self) -> int:
        return self.engine.max_request_len

    # ------------------------------------------------------------------ asyncio side

    async def generate(self, prompt_ids: list[int], params: SamplingParams) -> AsyncIterator[StepOutput]:
        """Submit a request and yield its tokens as they're produced. Closing the generator
        early (client disconnected, stop string hit) aborts the request in the engine."""
        if self.error is not None:
            raise RuntimeError(f"engine thread crashed: {self.error!r}")
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        rid = f"api-{next(self._ids)}"
        self._commands.put(("add", rid, prompt_ids, params, loop, q))
        finished = False
        try:
            while True:
                item = await q.get()
                if isinstance(item, BaseException):
                    raise item
                yield item
                if item.finished:
                    finished = True
                    return
        finally:
            if not finished:
                self._commands.put(("abort", rid))

    # ------------------------------------------------------------------ engine thread

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                # Idle: block until a command arrives. Busy: just drain what's there.
                self._drain_commands(block=not self.engine.has_unfinished())
                if self.engine.has_unfinished():
                    self.engine.step()
                    self._publish()
        except BaseException as e:  # surface crashes to every waiting client
            self.error = e
            for s in self._streams.values():
                s.loop.call_soon_threadsafe(s.queue.put_nowait, RuntimeError(f"engine crashed: {e!r}"))
            raise

    def _drain_commands(self, block: bool) -> None:
        try:
            cmd = self._commands.get(timeout=0.5) if block else self._commands.get_nowait()
        except queue.Empty:
            return
        while True:
            self._handle(cmd)
            try:
                cmd = self._commands.get_nowait()
            except queue.Empty:
                return

    def _handle(self, cmd: tuple) -> None:
        if cmd[0] == "add":
            _, rid, prompt_ids, params, loop, q = cmd
            try:
                req = self.engine.add_request(prompt_ids, params, request_id=rid)
            except ValueError as e:
                loop.call_soon_threadsafe(q.put_nowait, e)
                return
            self._streams[rid] = _Stream(req, loop, q)
            self.total_requests += 1
        elif cmd[0] == "abort":
            stream = self._streams.pop(cmd[1], None)
            if stream is not None:
                self.engine.abort_request(stream.req)

    def _publish(self) -> None:
        for rid, s in list(self._streams.items()):
            req = s.req
            new = req.output_ids[s.sent :]
            if not new and not req.is_finished:
                continue
            s.sent = len(req.output_ids)
            self.total_output_tokens += len(new)
            out = StepOutput(new, req.is_finished, req.finish_reason, len(req.prompt_ids), len(req.output_ids))
            s.loop.call_soon_threadsafe(s.queue.put_nowait, out)
            if req.is_finished:
                del self._streams[rid]

    # ------------------------------------------------------------------ stats (approximate, lock-free reads)

    def stats(self) -> dict:
        sched = self.engine.scheduler
        return {
            "running": len(sched.running),
            "waiting": len(sched.waiting),
            "steps": self.engine.step_count,
            "preemptions": sched.num_preemptions,
            "total_requests": self.total_requests,
            "total_output_tokens": self.total_output_tokens,
            "uptime_s": round(time.time() - self.started_at, 1),
            **self.engine.memory.usage(),
        }
