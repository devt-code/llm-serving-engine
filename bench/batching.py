"""Phase 2 benchmark: static vs continuous batching on the same engine.

Workload (offline, all requests arrive at t=0):
  - N chat-templated prompts, cycled from a fixed list
  - output lengths drawn uniformly from [min_new, max_new] with a fixed seed, EOS ignored,
    so every config generates exactly the same tokens. Mixed lengths are the realistic case,
    and the one where static batching wastes the most (short requests wait for long ones).

Metrics per config: output tokens/s, total time, time to first token (p50/p99),
end-to-end latency (p50/p99), steps, KV cache memory. An event trace is written per config.

Usage:
  python bench/batching.py                         # default: 128 requests, batch 32 and 64
  python bench/batching.py --num-requests 64 --batch-sizes 16,32
"""

import argparse
import json
import platform
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.engine import LLMEngine  # noqa: E402
from engine.events import EventLog  # noqa: E402
from engine.loader import load_model, pick_device  # noqa: E402
from engine.request import SamplingParams  # noqa: E402
from tests.prompts import PROMPTS  # noqa: E402


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def build_workload(tok, n, min_new, max_new, seed=0):
    rng = random.Random(seed)
    prompts = []
    for i in range(n):
        text = tok.apply_chat_template(
            [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        prompts.append(tok(text)["input_ids"])
    lens = [rng.randint(min_new, max_new) for _ in range(n)]
    return prompts, lens


def run(model, prompts, lens, policy, batch_size, max_model_len, trace_path):
    log = EventLog(trace_path)
    eng = LLMEngine(model, max_num_seqs=batch_size, max_model_len=max_model_len,
                    max_num_batched_tokens=max(2048, max_model_len), policy=policy, event_log=log)
    # warmup (not timed)
    eng_w = LLMEngine(model, max_num_seqs=batch_size, max_model_len=max_model_len,
                      max_num_batched_tokens=max(2048, max_model_len), policy=policy)
    eng_w.generate(prompts[:batch_size], SamplingParams(max_new_tokens=8, ignore_eos=True))
    del eng_w

    t0 = time.perf_counter()
    reqs = [eng.add_request(p, SamplingParams(max_new_tokens=n, ignore_eos=True)) for p, n in zip(prompts, lens)]
    while eng.has_unfinished():
        eng.step()
    total = time.perf_counter() - t0
    log.close()

    steps = [json.loads(line) for line in Path(trace_path).read_text().splitlines() if '"type": "step"' in line]
    prefill_steps = sum(s["kind"] == "prefill" for s in steps)
    decode_steps = sum(s["kind"] == "decode" for s in steps)
    useful = sum(lens)
    computed_decode_rows = sum(s["num_seqs"] for s in steps if s["kind"] == "decode")
    ttft = [r.first_token_time - r.arrival_time for r in reqs]
    e2e = [r.finish_time - r.arrival_time for r in reqs]
    return {
        "policy": policy,
        "batch_size": batch_size,
        "num_requests": len(reqs),
        "output_tokens": useful,
        "total_s": round(total, 3),
        "output_tok_per_s": round(useful / total, 1),
        "ttft_p50_s": round(pct(ttft, 50), 3),
        "ttft_p99_s": round(pct(ttft, 99), 3),
        "e2e_p50_s": round(pct(e2e, 50), 3),
        "e2e_p99_s": round(pct(e2e, 99), 3),
        "prefill_steps": prefill_steps,
        "decode_steps": decode_steps,
        # Fraction of decode rows that produced a kept token (static batching wastes the rest).
        "decode_utilization": round((useful - len(reqs)) / computed_decode_rows, 3) if computed_decode_rows else None,
        "kv_cache_gb": round(eng.cache.nbytes / 1e9, 2),
        "trace": str(trace_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen3-0.6b")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--num-requests", type=int, default=128)
    ap.add_argument("--min-new", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--batch-sizes", default="32,64")
    ap.add_argument("--max-model-len", type=int, default=320)
    ap.add_argument("--out", default="bench/results/batching.json")
    ap.add_argument("--trace-dir", default="bench/traces")
    args = ap.parse_args()

    device = pick_device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, device=device, dtype=getattr(torch, args.dtype))
    prompts, lens = build_workload(tok, args.num_requests, args.min_new, args.max_new)
    print(f"{args.num_requests} requests, prompts {min(map(len, prompts))}-{max(map(len, prompts))} tokens, "
          f"outputs {min(lens)}-{max(lens)} tokens ({sum(lens)} total), {device} {args.dtype}")

    results = []
    for bs in [int(b) for b in args.batch_sizes.split(",")]:
        for policy in ("static", "continuous"):
            trace = Path(args.trace_dir) / f"phase2_{policy}_bs{bs}.jsonl"
            r = run(model, prompts, lens, policy, bs, args.max_model_len, trace)
            results.append(r)
            print(f"{policy:>10} bs={bs:<3} {r['output_tok_per_s']:7.1f} tok/s  total {r['total_s']:6.1f}s  "
                  f"TTFT p50/p99 {r['ttft_p50_s']:.2f}/{r['ttft_p99_s']:.2f}s  "
                  f"latency p50/p99 {r['e2e_p50_s']:.1f}/{r['e2e_p99_s']:.1f}s  "
                  f"steps {r['prefill_steps']}+{r['decode_steps']}  util {r['decode_utilization']}")
            if device.type == "mps":
                torch.mps.empty_cache()

    record = {
        "engine": "ours-phase2",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(device), "dtype": args.dtype, "machine": platform.platform(), "torch": torch.__version__,
        "workload": {"num_requests": args.num_requests, "min_new": args.min_new, "max_new": args.max_new,
                     "arrival": "all at t=0", "ignore_eos": True},
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(record, indent=2))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
