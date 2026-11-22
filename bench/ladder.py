"""The comparison ladder: every optimization measured on ONE workload, same model, same GPU.

    hf          Hugging Face transformers generate(), static batches of --batch (padded)
    static      our engine, static batching (slot cache)
    continuous  our engine, continuous batching (slot cache)
    paged       + paged KV cache (same KV memory as the slot cache)
    prefix      + prefix caching
    vllm        vLLM, same settings (only where vLLM is installed, i.e. Linux + NVIDIA GPU)
    vllm_kv     vLLM with its KV cache capped at the same budget our engine gets

Workload: --num-requests chat requests arriving at once. Half share a ~380-token system prompt
(like an app's built-in instructions), half are short standalone questions. Output lengths
16-256 tokens, fixed seed, EOS ignored, so every engine generates exactly the same tokens.
KV memory for our engine = --slots requests x max_model_len tokens, as the Phase 2 slot cache
needs; the paged runs get the same bytes.

  python bench/ladder.py                       # everything that can run here
  python bench/ladder.py --only hf,prefix      # a subset
  python bench/ladder.py --dtype float16       # T4 GPUs (no fast bf16)
"""

import argparse
import gc
import json
import os
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.batching import pct  # noqa: E402
from bench.prefix import QUESTIONS, SYSTEM  # noqa: E402
from tests.prompts import PROMPTS  # noqa: E402

LADDER = ["hf", "static", "continuous", "paged", "prefix", "vllm", "vllm_kv"]
VLLM = ("vllm", "vllm_kv")
LABELS = {
    "hf": "HF transformers (static batches)",
    "static": "Ours: static batching",
    "continuous": "Ours: continuous batching",
    "paged": "Ours: + paged KV cache",
    "prefix": "Ours: + prefix caching",
    "vllm": "vLLM",
    "vllm_kv": "vLLM, same KV memory as ours",
}


def build_workload(tok, n: int, seed: int = 0):
    rng = random.Random(seed)
    prompts, lens = [], []
    for i in range(n):
        if i % 2 == 0:
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": QUESTIONS[i % len(QUESTIONS)]}]
        else:
            msgs = [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompts.append(tok(text)["input_ids"])
        lens.append(rng.randint(16, 256))
    return prompts, lens


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def free(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def result(name, total, lens, ttft=None, e2e=None, **extra):
    out = {"config": name, "label": LABELS[name], "total_s": round(total, 2),
           "output_tokens": sum(lens), "output_tok_per_s": round(sum(lens) / total, 1), **extra}
    if ttft:
        out |= {"ttft_p50_s": round(pct(ttft, 50), 3), "ttft_p99_s": round(pct(ttft, 99), 3)}
    if e2e:
        out |= {"e2e_p50_s": round(pct(e2e, 50), 2), "e2e_p99_s": round(pct(e2e, 99), 2)}
    return out


# ------------------------------------------------------------------ HF baseline


def run_hf(args, device, dtype, prompts, lens):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    pad = tok.pad_token_id

    def batch_generate(idx):
        ids = [prompts[i] for i in idx]
        width = max(map(len, ids))
        input_ids = torch.tensor([[pad] * (width - len(p)) + p for p in ids], device=device)
        mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in ids], device=device)
        n = max(lens[i] for i in idx)  # static batching: everyone runs as long as the longest
        model.generate(input_ids=input_ids, attention_mask=mask, max_new_tokens=n, min_new_tokens=n,
                       do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=pad)

    with torch.inference_mode():
        batch_generate(list(range(min(4, len(prompts)))))  # warmup
        sync(device)
        t0 = time.perf_counter()
        e2e, ttft = [], []
        for start in range(0, len(prompts), args.batch):
            idx = list(range(start, min(start + args.batch, len(prompts))))
            b0 = time.perf_counter()
            batch_generate(idx)
            sync(device)
            done = time.perf_counter() - t0
            # generate() returns only when the whole batch is done; first tokens come ~one prefill later.
            ttft += [b0 - t0] * len(idx)
            e2e += [done] * len(idx)
        total = time.perf_counter() - t0
    del model
    free(device)
    return result("hf", total, lens, ttft=None, e2e=e2e, note="TTFT not observable inside generate()")


# ------------------------------------------------------------------ our engine


def run_ours(name, args, model, prompts, lens):
    from engine.engine import LLMEngine
    from engine.request import SamplingParams

    max_len = max(len(p) for p in prompts) + max(lens)
    slots = args.slots
    kw = dict(max_model_len=max_len, max_num_batched_tokens=max(4096, max_len))
    if name in ("static", "continuous"):
        kw |= dict(kv_cache="slot", max_num_seqs=slots, policy="static" if name == "static" else "continuous")
    else:
        kw |= dict(kv_cache="paged", max_num_seqs=args.max_num_seqs, block_size=16,
                   num_kv_blocks=slots * max_len // 16, enable_prefix_caching=(name == "prefix"))
    LLMEngine(model, **kw).generate(prompts[:8], SamplingParams(max_new_tokens=8, ignore_eos=True))  # warmup
    eng = LLMEngine(model, **kw)
    t0 = time.perf_counter()
    reqs = [eng.add_request(p, SamplingParams(max_new_tokens=n, ignore_eos=True)) for p, n in zip(prompts, lens)]
    while eng.has_unfinished():
        eng.step()
    total = time.perf_counter() - t0
    ttft = [r.first_token_time - r.arrival_time for r in reqs]
    e2e = [r.finish_time - r.arrival_time for r in reqs]
    extra = dict(kv_cache_gb=round(eng.cache.nbytes / 1e9, 2), peak_running=eng.peak_running,
                 preemptions=eng.scheduler.num_preemptions)
    if name == "prefix":
        extra["prefix_hit_rate"] = eng.memory.usage()["prefix_hit_rate"]
    del eng
    return result(name, total, lens, ttft=ttft, e2e=e2e, **extra)


# ------------------------------------------------------------------ vLLM


def run_vllm(name, args, prompts, lens, kv_bytes):
    """vllm: vLLM sizes its own KV cache from --vllm-gpu-util. vllm_kv: capped at our KV budget."""
    # vLLM runs its engine in a child process. Forking one after CUDA has started fails with
    # "Cannot re-initialize CUDA in forked subprocess" (seen on Colab with vLLM 0.34), so spawn it.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM, SamplingParams as VSP  # noqa: N814
    from vllm.inputs import TokensPrompt

    max_len = max(len(p) for p in prompts) + max(lens)
    kw = dict(model=args.model, dtype=args.dtype, max_model_len=max_len, enable_prefix_caching=True, seed=0)
    if name == "vllm_kv":
        kw["kv_cache_memory_bytes"] = kv_bytes
        note = f"vLLM offline LLM.generate, prefix caching on, KV cache capped at our {kv_bytes / 1e9:.2f} GB"
    else:
        kw["gpu_memory_utilization"] = args.vllm_gpu_util
        note = (f"vLLM offline LLM.generate, prefix caching on, gpu_memory_utilization={args.vllm_gpu_util} "
                f"(vLLM sizes its own KV cache); our KV budget was {kv_bytes / 1e9:.2f} GB")
    llm = LLM(**kw)
    inputs = [TokensPrompt(prompt_token_ids=p) for p in prompts]
    params = [VSP(max_tokens=n, min_tokens=n, ignore_eos=True, temperature=0.0) for n in lens]
    llm.generate(inputs[:8], params[:8], use_tqdm=False)  # warmup
    t0 = time.perf_counter()
    llm.generate(inputs, params, use_tqdm=False)
    total = time.perf_counter() - t0
    return result(name, total, lens, note=note)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen3-0.6b")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--num-requests", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32, help="HF batch size")
    ap.add_argument("--slots", type=int, default=32, help="KV memory budget, in requests of max_model_len")
    ap.add_argument("--max-num-seqs", type=int, default=256, help="paged concurrency cap")
    ap.add_argument("--vllm-gpu-util", type=float, default=0.5)
    ap.add_argument("--only", default=None, help="comma-separated subset of " + ",".join(LADDER))
    ap.add_argument("--out", default="bench/results/ladder.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from engine.kv_cache import PagedKVCache
    from engine.loader import load_model, pick_device

    device = pick_device(args.device)
    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts, lens = build_workload(tok, args.num_requests)
    todo = args.only.split(",") if args.only else LADDER
    if any(t in VLLM for t in todo):
        try:
            import vllm  # noqa: F401
        except ImportError:
            print("vllm not installed here, skipping (it needs Linux + an NVIDIA GPU)")
            todo = [t for t in todo if t not in VLLM]
    max_len = max(len(p) for p in prompts) + max(lens)
    from engine.config import Qwen3Config

    kv_bytes = args.slots * max_len // 16 * PagedKVCache.bytes_per_block(Qwen3Config.from_pretrained(args.model), 16, dtype)
    kv_gb = kv_bytes / 1e9
    print(f"{len(prompts)} requests (half with a {len(prompts[0])}-token shared-system-prompt chat, half short), "
          f"outputs 16-256 ({sum(lens)} tokens), {device} {args.dtype}, KV budget {kv_gb:.2f} GB")

    out_path = Path(args.out)
    existing = json.loads(out_path.read_text())["results"] if out_path.exists() else []
    results = {r["config"]: r for r in existing}
    model = None
    for name in [n for n in LADDER if n in todo]:
        if name == "hf":
            r = run_hf(args, device, dtype, prompts, lens)
        elif name in VLLM:
            if model is not None:
                del model
                model = None
                free(device)
            r = run_vllm(name, args, prompts, lens, kv_bytes)
        else:
            if model is None:
                model = load_model(args.model, device=device, dtype=dtype)
            r = run_ours(name, args, model, prompts, lens)
            free(device)
        results[name] = r
        ttft = f"TTFT p50 {r['ttft_p50_s']:.2f}s" if "ttft_p50_s" in r else ""
        print(f"{r['label']:<36} {r['output_tok_per_s']:8.1f} tok/s   total {r['total_s']:7.1f}s   {ttft}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "timestamp": datetime.now().isoformat(timespec="seconds"), "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor() or platform.machine(),
            "dtype": args.dtype, "machine": platform.platform(), "torch": torch.__version__,
            "workload": {"num_requests": len(prompts), "output_tokens": sum(lens), "kv_budget_gb": round(kv_gb, 2)},
            "results": [results[n] for n in LADDER if n in results],
        }, indent=2))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
