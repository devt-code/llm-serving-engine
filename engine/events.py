"""Event stream for the dashboard (Phase 5) and for analysis.

One JSON object per line (JSONL). Times are seconds since the engine started.

  {"t": 0.0012, "type": "request_arrived",  "id": "req-3", "prompt_len": 31, "max_new_tokens": 128}
  {"t": 0.0400, "type": "request_started",  "id": "req-3", "slot": 2}
  {"t": 0.0410, "type": "first_token",      "id": "req-3"}
  {"t": 2.9100, "type": "request_finished", "id": "req-3", "output_len": 128, "reason": "length"}
  {"t": 0.0410, "type": "step", "step": 7, "kind": "decode", "num_seqs": 32, "num_tokens": 32,
                "running": ["req-0", ...], "slots": [0, ...], "waiting": 96, "duration_ms": 41.2}

Later phases add KV block events (Phase 3: allocated, freed, preempted).
"""

import json
import time
from pathlib import Path


class EventLog:
    def __init__(self, path: str | Path | None = None, keep_in_memory: bool = False):
        self.t0 = time.perf_counter()
        self.events: list[dict] | None = [] if keep_in_memory else None
        self._file = None
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._file = open(path, "w")

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def emit(self, type: str, **fields) -> None:
        event = {"t": round(self.now(), 6), "type": type, **fields}
        if self.events is not None:
            self.events.append(event)
        if self._file is not None:
            self._file.write(json.dumps(event) + "\n")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
