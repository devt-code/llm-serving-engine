"""Record event traces for the dashboard's replay mode (dashboard/public/traces/).

Each trace is a real engine run on this machine, written as JSONL by engine/events.py.

  python bench/record_demo.py                  # records both demos
  python bench/record_demo.py --only mixed
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.prefix import QUESTIONS, SYSTEM  # noqa: E402
from engine.engine import LLMEngine  # noqa: E402
from engine.events import EventLog  # noqa: E402
from engine.loader import load_model, pick_device  # noqa: E402
from engine.request import SamplingParams  # noqa: E402
from tests.prompts import PROMPTS  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "dashboard" / "public" / "traces"

DEMOS = {
    "mixed": {
        "title": "Steady traffic with a shared system prompt",
        "description": "About 4 requests per second for 40 seconds. Half share a long system prompt, so its blocks "
                       "are computed once and reused by every request that starts with it (aqua squares). Watch "
                       "requests join and leave the batch at every step, and cached blocks stay after their "
                       "requests finish.",
        "kv_cache_gb": 1.0, "rate": 4.0, "duration": 40.0, "shared": 0.5, "max_new": (32, 160),
    },
    "pressure": {
        "title": "Burst into a small KV cache (preemption)",
        "description": "96 requests arrive at once into a cache that can't hold them all. The scheduler admits what "
                       "fits, then preempts the newest requests (orange) when blocks run out, and recomputes them "
                       "later. No request's output changes; only its timing does.",
        "kv_cache_gb": 0.2, "rate": None, "count": 96, "shared": 0.0, "max_new": (64, 224),
    },
}


def make_prompt(tok, rng, shared: bool) -> list[int]:
    if shared:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": rng.choice(QUESTIONS)}]
    else:
        msgs = [{"role": "user", "content": rng.choice(PROMPTS)}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(text)["input_ids"]


def record(model, tok, name: str, cfg: dict) -> dict:
    rng = random.Random(42)
    path = OUT / f"{name}.jsonl"
    log = EventLog(path)
    eng = LLMEngine(model, max_num_seqs=128, max_model_len=1024, max_num_batched_tokens=4096,
                    kv_cache="paged", kv_cache_gb=cfg["kv_cache_gb"], event_log=log)
    arrivals = []
    if cfg["rate"]:
        t = 0.0
        while t < cfg["duration"]:
            arrivals.append(t)
            t += rng.expovariate(cfg["rate"])
    else:
        arrivals = [0.0] * cfg["count"]
    reqs = [(a, make_prompt(tok, rng, rng.random() < cfg["shared"]), rng.randint(*cfg["max_new"])) for a in arrivals]

    t0, nxt = time.perf_counter(), 0
    while nxt < len(reqs) or eng.has_unfinished():
        now = time.perf_counter() - t0
        while nxt < len(reqs) and reqs[nxt][0] <= now:
            eng.add_request(reqs[nxt][1], SamplingParams(max_new_tokens=reqs[nxt][2], ignore_eos=True))
            nxt += 1
        if eng.has_unfinished():
            eng.step()
        elif nxt < len(reqs):
            time.sleep(max(0.0, reqs[nxt][0] - (time.perf_counter() - t0)))
    log.close()
    size = path.stat().st_size
    print(f"{name}: {len(reqs)} requests, {eng.step_count} steps, {eng.scheduler.num_preemptions} preemptions, "
          f"prefix hit rate {eng.memory.usage()['prefix_hit_rate']:.0%}, {size / 1e6:.1f} MB -> {path}")
    return {"file": path.name, "title": cfg["title"], "description": cfg["description"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen3-0.6b")
    ap.add_argument("--only", choices=list(DEMOS), default=None)
    args = ap.parse_args()
    device = pick_device("auto")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, device=device, dtype=torch.bfloat16)
    LLMEngine(model, kv_cache_gb=0.2).generate([tok("warmup")["input_ids"]], SamplingParams(max_new_tokens=4))

    OUT.mkdir(parents=True, exist_ok=True)
    index_path = OUT / "index.json"
    index = {e["file"]: e for e in (json.loads(index_path.read_text()) if index_path.exists() else [])}
    for name, cfg in DEMOS.items():
        if args.only and name != args.only:
            continue
        index[f"{name}.jsonl"] = record(model, tok, name, cfg)
    ordered = [index[f"{n}.jsonl"] for n in DEMOS if f"{n}.jsonl" in index]
    index_path.write_text(json.dumps(ordered, indent=2))


if __name__ == "__main__":
    main()
