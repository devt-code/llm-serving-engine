"""Prefix caching benchmark: many requests sharing one long system prompt.

Each request = the same system prompt (~500 tokens) + a different short user question, chat
templated, with 32-96 output tokens (EOS ignored). Two arrival patterns:
  - burst:   all requests at t=0 (offline throughput)
  - poisson: requests arrive at --rate per second (what a live server sees; TTFT matters here)
Each runs with prefix caching on and off.

Usage:
  python bench/prefix.py
  python bench/prefix.py --num-requests 64 --rate 4
"""

import argparse
import json
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.batching import pct  # noqa: E402
from engine.engine import LLMEngine  # noqa: E402
from engine.events import EventLog  # noqa: E402
from engine.loader import load_model, pick_device  # noqa: E402
from engine.request import SamplingParams  # noqa: E402

SYSTEM = (
    "You are the support assistant for Northwind Outfitters, an online store for hiking, camping and "
    "climbing gear. Follow these rules in every answer. "
    "1. Be friendly, concise and accurate; prefer short paragraphs or brief lists. "
    "2. Orders ship within two business days from our Denver warehouse. Standard shipping takes three "
    "to seven business days; express shipping takes one to two business days and costs fifteen dollars. "
    "3. Returns are accepted within sixty days for unused items with tags. Used items can be returned "
    "within thirty days only if they are defective. Refunds go back to the original payment method "
    "within five business days after we receive the return. "
    "4. Our warranty covers manufacturing defects for one year on tents, backpacks and sleeping bags, "
    "and for two years on stoves, headlamps and water filters. It never covers normal wear, misuse, "
    "or damage from improper storage. "
    "5. Never promise delivery dates, discounts or exceptions you are not sure about; offer to connect "
    "the customer with a human agent instead. "
    "6. For safety questions about climbing equipment, always recommend that the customer follow the "
    "manufacturer's instructions and get training from a certified instructor. "
    "7. Loyalty members earn one point per dollar spent; five hundred points can be redeemed for a "
    "twenty-five dollar credit. Points expire after eighteen months without a purchase. "
    "8. We price match identical items from major retailers within fourteen days of purchase, "
    "excluding clearance, marketplace sellers and membership-only prices. "
    "9. Gift cards never expire and can be used online only. "
    "10. If the customer is upset, acknowledge their frustration before offering a solution. "
    "11. Do not discuss competitors' products in detail and do not share internal policies beyond "
    "what is written here. "
    "12. Always end with a short question asking whether there is anything else you can help with."
)

QUESTIONS = [
    "When will my order ship?", "Can I return a tent I used once?", "How long does express shipping take?",
    "Is my headlamp still under warranty after 18 months?", "Do you price match Amazon?",
    "How do I redeem loyalty points?", "My stove arrived broken, what now?", "Do gift cards expire?",
    "Can you give me a discount on a backpack?", "How long until I get my refund?",
    "Is this harness safe for lead climbing?", "Do points expire?", "Can I use a gift card in a store?",
    "What does the warranty not cover?", "I'm really annoyed my package is late.", "Do you ship from Denver?",
]


def build(tok, n, seed=0):
    rng = random.Random(seed)
    prompts, lens = [], []
    for i in range(n):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": QUESTIONS[i % len(QUESTIONS)]}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompts.append(tok(text)["input_ids"])
        lens.append(rng.randint(32, 96))
    return prompts, lens


def run(model, prompts, lens, prefix_caching, arrivals, kv_gb, trace):
    kw = dict(max_num_seqs=64, max_model_len=1024, max_num_batched_tokens=4096, kv_cache="paged",
              kv_cache_gb=kv_gb, enable_prefix_caching=prefix_caching)
    LLMEngine(model, **kw).generate(prompts[:4], SamplingParams(max_new_tokens=4, ignore_eos=True))  # warmup
    log = EventLog(trace)
    eng = LLMEngine(model, event_log=log, **kw)
    order = sorted(range(len(prompts)), key=lambda i: arrivals[i])
    reqs, nxt = [None] * len(prompts), 0
    t0 = time.perf_counter()
    while nxt < len(order) or eng.has_unfinished():
        now = time.perf_counter() - t0
        while nxt < len(order) and arrivals[order[nxt]] <= now:
            i = order[nxt]
            reqs[i] = eng.add_request(prompts[i], SamplingParams(max_new_tokens=lens[i], ignore_eos=True))
            nxt += 1
        if eng.has_unfinished():
            eng.step()
        elif nxt < len(order):
            time.sleep(max(0.0, arrivals[order[nxt]] - (time.perf_counter() - t0)))
    total = time.perf_counter() - t0
    log.close()
    steps = [json.loads(l) for l in Path(trace).read_text().splitlines() if '"type": "step"' in l]
    prefill_tokens = sum(s["num_tokens"] for s in steps if s["kind"] == "prefill")
    ttft = [r.first_token_time - r.arrival_time for r in reqs]
    e2e = [r.finish_time - r.arrival_time for r in reqs]
    return {
        "prefix_caching": prefix_caching,
        "output_tok_per_s": round(sum(lens) / total, 1),
        "total_s": round(total, 2),
        "prompt_tokens": sum(map(len, prompts)),
        "prefill_tokens_computed": prefill_tokens,
        "prefix_hit_rate": eng.memory.usage()["prefix_hit_rate"],
        "ttft_p50_ms": round(pct(ttft, 50) * 1000, 1),
        "ttft_p99_ms": round(pct(ttft, 99) * 1000, 1),
        "e2e_p50_s": round(pct(e2e, 50), 2),
        "e2e_p99_s": round(pct(e2e, 99), 2),
        "preemptions": eng.scheduler.num_preemptions,
        "trace": str(trace),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen3-0.6b")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--num-requests", type=int, default=128)
    ap.add_argument("--rate", type=float, default=6.0, help="requests per second for the poisson run")
    ap.add_argument("--kv-cache-gb", type=float, default=2.0)
    ap.add_argument("--out", default="bench/results/prefix.json")
    ap.add_argument("--trace-dir", default="bench/traces")
    args = ap.parse_args()

    device = pick_device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, device=device, dtype=getattr(torch, args.dtype))
    prompts, lens = build(tok, args.num_requests)
    rng = random.Random(1)
    t, poisson = 0.0, []
    for _ in prompts:
        poisson.append(t)
        t += rng.expovariate(args.rate)
    print(f"{len(prompts)} requests, prompt {len(prompts[0])} tokens (shared system prompt + question), "
          f"outputs 32-96 tokens, {device} {args.dtype}")

    results = []
    for pattern, arrivals in [("burst", [0.0] * len(prompts)), (f"poisson@{args.rate:g}/s", poisson)]:
        for pc in (False, True):
            trace = Path(args.trace_dir) / f"prefix_{pattern.split('@')[0]}_{'on' if pc else 'off'}.jsonl"
            r = run(model, prompts, lens, pc, arrivals, args.kv_cache_gb, trace)
            r["arrivals"] = pattern
            results.append(r)
            print(f"{pattern:>14} prefix_caching={'on ' if pc else 'off'}  {r['output_tok_per_s']:7.1f} tok/s  "
                  f"TTFT p50/p99 {r['ttft_p50_ms']:8.1f}/{r['ttft_p99_ms']:8.1f} ms  "
                  f"prefill computed {r['prefill_tokens_computed']:>6}/{r['prompt_tokens']} tokens  "
                  f"hit rate {r['prefix_hit_rate']:.2f}  total {r['total_s']:.1f}s")
            if device.type == "mps":
                torch.mps.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "engine": "ours-prefix-caching", "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(device), "dtype": args.dtype, "machine": platform.platform(), "torch": torch.__version__,
        "workload": {"num_requests": len(prompts), "prompt_tokens": len(prompts[0]), "output_tokens": "32-96",
                     "ignore_eos": True, "poisson_rate": args.rate},
        "results": results,
    }, indent=2))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
