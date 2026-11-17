"""Phase 3 benchmark: slot cache (Phase 2) vs paged cache, with IDENTICAL KV memory.

For each memory budget, expressed as "N slots of max_model_len tokens":
  - slot:  N slots, each reserving max_model_len tokens (continuous batching)
  - paged: the same bytes as a pool of 16-token blocks, tried with several concurrency caps
           (--max-num-seqs, default 64,128,256). Too high a cap admits more requests than
           can grow, which shows up as preemptions (recomputed work).

Workload: same as bench/batching.py (all requests at t=0, mixed output lengths, EOS ignored).

Usage:
  python bench/paged.py
  python bench/paged.py --num-requests 256 --slot-budgets 32,64
"""

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.batching import build_workload, pct  # noqa: E402
from engine.engine import LLMEngine  # noqa: E402
from engine.events import EventLog  # noqa: E402
from engine.kv_cache import PagedKVCache  # noqa: E402
from engine.loader import load_model, pick_device  # noqa: E402
from engine.request import SamplingParams  # noqa: E402


def run(model, prompts, lens, kv_cache, engine_kw, trace_path):
    common = dict(max_num_batched_tokens=2048, policy="continuous", kv_cache=kv_cache, **engine_kw)
    LLMEngine(model, **common).generate(prompts[:16], SamplingParams(max_new_tokens=8, ignore_eos=True))  # warmup

    log = EventLog(trace_path)
    eng = LLMEngine(model, event_log=log, **common)
    t0 = time.perf_counter()
    reqs = [eng.add_request(p, SamplingParams(max_new_tokens=n, ignore_eos=True)) for p, n in zip(prompts, lens)]
    while eng.has_unfinished():
        eng.step()
    total = time.perf_counter() - t0
    log.close()

    steps = [json.loads(l) for l in Path(trace_path).read_text().splitlines() if '"type": "step"' in l]
    decode = [s for s in steps if s["kind"] == "decode"]
    ttft = [r.first_token_time - r.arrival_time for r in reqs]
    e2e = [r.finish_time - r.arrival_time for r in reqs]
    # KV memory actually holding tokens vs reserved, averaged over decode steps.
    return {
        "kv_cache": kv_cache,
        "kv_cache_gb": round(eng.cache.nbytes / 1e9, 3),
        "output_tok_per_s": round(sum(lens) / total, 1),
        "total_s": round(total, 2),
        "peak_running": eng.peak_running,
        "mean_running_decode": round(statistics.mean(s["num_seqs"] for s in decode), 1),
        "mean_decode_step_ms": round(statistics.mean(s["duration_ms"] for s in decode), 2),
        "preemptions": eng.scheduler.num_preemptions,
        "ttft_p50_s": round(pct(ttft, 50), 3),
        "ttft_p99_s": round(pct(ttft, 99), 3),
        "e2e_p50_s": round(pct(e2e, 50), 3),
        "e2e_p99_s": round(pct(e2e, 99), 3),
        "steps": len(steps),
        "trace": str(trace_path),
        **{k: v for k, v in engine_kw.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen3-0.6b")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--num-requests", type=int, default=256)
    ap.add_argument("--min-new", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=320)
    ap.add_argument("--slot-budgets", default="32,64", help="memory budgets, in Phase 2 slots of max_model_len")
    ap.add_argument("--max-num-seqs", default="64,128,256", help="paged concurrency caps to try")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--out", default="bench/results/paged.json")
    ap.add_argument("--trace-dir", default="bench/traces")
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, device=device, dtype=dtype)
    prompts, lens = build_workload(tok, args.num_requests, args.min_new, args.max_new)
    print(f"{args.num_requests} requests, outputs {min(lens)}-{max(lens)} tokens ({sum(lens)} total), {device} {args.dtype}")

    results = []
    for slots in [int(s) for s in args.slot_budgets.split(",")]:
        num_blocks = slots * args.max_model_len // args.block_size
        configs = [("slot", dict(max_num_seqs=slots, max_model_len=args.max_model_len))]
        for cap in [int(c) for c in args.max_num_seqs.split(",")]:
            configs.append(("paged", dict(max_num_seqs=cap, max_model_len=args.max_model_len,
                                          num_kv_blocks=num_blocks, block_size=args.block_size)))
        for kind, kw in configs:
            trace = Path(args.trace_dir) / f"phase3_{kind}_budget{slots}_cap{kw['max_num_seqs']}.jsonl"
            r = run(model, prompts, lens, kind, kw, trace)
            r["budget_slots"] = slots
            results.append(r)
            print(f"budget={slots:<3} {kind:>5} cap={kw['max_num_seqs']:<4} {r['kv_cache_gb']:.2f} GB  {r['output_tok_per_s']:7.1f} tok/s  "
                  f"peak running {r['peak_running']:<4} mean {r['mean_running_decode']:<6} "
                  f"step {r['mean_decode_step_ms']:6.1f} ms  preempt {r['preemptions']:<4} "
                  f"TTFT p50/p99 {r['ttft_p50_s']:.2f}/{r['ttft_p99_s']:.2f}s  total {r['total_s']:.1f}s")
            if device.type == "mps":
                torch.mps.empty_cache()

    record = {
        "engine": "ours-phase3", "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(device), "dtype": args.dtype, "machine": platform.platform(), "torch": torch.__version__,
        "workload": {"num_requests": args.num_requests, "min_new": args.min_new, "max_new": args.max_new,
                     "arrival": "all at t=0", "ignore_eos": True},
        "bytes_per_block": PagedKVCache.bytes_per_block(model.config, args.block_size, dtype),
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(record, indent=2))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
