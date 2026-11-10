"""Phase 0 baseline: Hugging Face transformers generate() on Qwen3-0.6B.

Measures, for batch sizes 1 and 8:
  - prefill latency (time to produce the first token, ~TTFT)
  - decode throughput (output tokens/s after the first token)
  - end-to-end output tokens/s

Every request generates exactly --max-new-tokens tokens (EOS ignored) so runs are
comparable across engines. Greedy decoding.

Usage:
  python bench/hf_baseline.py                       # auto device (mps > cuda > cpu)
  python bench/hf_baseline.py --device cpu --dtype float32
"""

import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "Explain how a CPU cache works in simple terms.",
    "Write a short poem about the ocean at night.",
    "What are the main causes of the French Revolution?",
    "Describe the process of photosynthesis step by step.",
    "Give me three tips for writing clean Python code.",
    "Why is the sky blue? Answer like a physics teacher.",
    "Summarize the plot of a heist movie you would like to see.",
    "How does a hash table handle collisions?",
]


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def build_inputs(tok, prompts, device):
    texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for p in prompts
    ]
    return tok(texts, return_tensors="pt", padding=True).to(device)


def timed_generate(model, inputs, new_tokens, device):
    sync(device)
    t0 = time.perf_counter()
    out = model.generate(
        **inputs,
        max_new_tokens=new_tokens,
        min_new_tokens=new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )
    sync(device)
    return time.perf_counter() - t0, out


def run(model, tok, batch_size, max_new_tokens, device, repeats):
    prompts = (PROMPTS * ((batch_size // len(PROMPTS)) + 1))[:batch_size]
    inputs = build_inputs(tok, prompts, device)

    timed_generate(model, inputs, 8, device)  # warmup

    prefill, total = [], []
    for _ in range(repeats):
        t, _ = timed_generate(model, inputs, 1, device)
        prefill.append(t)
        t, out = timed_generate(model, inputs, max_new_tokens, device)
        total.append(t)

    p = sorted(prefill)[len(prefill) // 2]
    e2e = sorted(total)[len(total) // 2]
    out_tokens = batch_size * max_new_tokens
    decode_tokens = batch_size * (max_new_tokens - 1)
    sample = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return {
        "batch_size": batch_size,
        "prompt_tokens_padded": int(inputs["input_ids"].shape[1]),
        "max_new_tokens": max_new_tokens,
        "prefill_s": round(p, 4),
        "e2e_s": round(e2e, 4),
        "e2e_output_tok_per_s": round(out_tokens / e2e, 2),
        "decode_tok_per_s": round(decode_tokens / (e2e - p), 2),
        "per_request_decode_tok_per_s": round((max_new_tokens - 1) / (e2e - p), 2),
        "sample_output": sample[:200],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen3-0.6b")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--batch-sizes", default="1,8")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="bench/results/hf_baseline.json")
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()

    results = []
    with torch.inference_mode():
        for bs in [int(b) for b in args.batch_sizes.split(",")]:
            r = run(model, tok, bs, args.max_new_tokens, device, args.repeats)
            results.append(r)
            print(
                f"bs={bs:>3}  prefill={r['prefill_s']*1000:7.1f} ms  "
                f"e2e={r['e2e_output_tok_per_s']:8.1f} tok/s  "
                f"decode={r['decode_tok_per_s']:8.1f} tok/s  "
                f"({r['per_request_decode_tok_per_s']:.1f} tok/s per request)"
            )

    record = {
        "engine": "hf-transformers",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "dtype": args.dtype,
        "machine": platform.platform(),
        "torch": torch.__version__,
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(record, indent=2))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
