"""Phase 1 benchmark: our engine, one request at a time (no batching yet).

Same settings as bench/hf_baseline.py at batch size 1, so the numbers are directly comparable:
chat-templated prompt, greedy, exactly --max-new-tokens tokens (EOS ignored).

Usage:
  python bench/engine_single.py
  python bench/engine_single.py --device cpu --dtype float32
"""

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.generate import generate  # noqa: E402
from engine.loader import load_model, pick_device  # noqa: E402

PROMPT = "Explain how a CPU cache works in simple terms."


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen3-0.6b")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="bench/results/engine_single.json")
    args = ap.parse_args()

    device = pick_device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, device=device, dtype=getattr(torch, args.dtype))
    text = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids = tok(text)["input_ids"]

    generate(model, ids, 8, ignore_eos=True)  # warmup

    prefill, total = [], []
    for _ in range(args.repeats):
        sync(device)
        t0 = time.perf_counter()
        generate(model, ids, 1, ignore_eos=True)
        sync(device)
        prefill.append(time.perf_counter() - t0)

        sync(device)
        t0 = time.perf_counter()
        out = generate(model, ids, args.max_new_tokens, ignore_eos=True)
        sync(device)
        total.append(time.perf_counter() - t0)

    p, e2e = sorted(prefill)[len(prefill) // 2], sorted(total)[len(total) // 2]
    n = args.max_new_tokens
    result = {
        "batch_size": 1,
        "prompt_tokens": len(ids),
        "max_new_tokens": n,
        "prefill_s": round(p, 4),
        "e2e_s": round(e2e, 4),
        "e2e_output_tok_per_s": round(n / e2e, 2),
        "decode_tok_per_s": round((n - 1) / (e2e - p), 2),
        "sample_output": tok.decode(out, skip_special_tokens=True)[:200],
    }
    print(
        f"prefill={p * 1000:.1f} ms  e2e={result['e2e_output_tok_per_s']:.1f} tok/s  "
        f"decode={result['decode_tok_per_s']:.1f} tok/s"
    )
    record = {
        "engine": "ours-phase1",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "dtype": args.dtype,
        "machine": platform.platform(),
        "torch": torch.__version__,
        "results": [result],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(record, indent=2))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
