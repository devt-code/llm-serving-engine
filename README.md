# LLM Serving Engine

An LLM inference engine written from scratch in PyTorch: the system that serves a language model
to many users at once. It runs Qwen3-0.6B with continuous batching, a paged KV cache with
preemption, and prefix caching, behind an OpenAI-compatible streaming API. A live dashboard shows
what the engine is doing.

**[▶ Watch a recorded run in the dashboard](https://devt-code.github.io/llm-serving-engine/)**, no GPU needed.

- **3.2× the throughput of Hugging Face `generate()`** on the same model, GPU and workload
  (558 vs 176 tok/s), and **median time to first token cut from 80 s to 2.9 s** under load
  compared with static batching.
- **2.6× more requests running at once in the same GPU memory**, thanks to the paged KV cache.
- **Median time to first token drops from 7.2 s to 29 ms** at 6 requests per second when requests share a
  system prompt, thanks to prefix caching (96% of prompt tokens reused).
- **Outputs identical to Hugging Face** token for token (fp32 greedy, 20 prompts), and identical
  whether a request runs alone, in a batch, or gets preempted and recomputed: 107 tests.

Every number here is measured and reproducible with the scripts in [`bench/`](bench/). Raw results are in
[`bench/results/`](bench/results/).

## Results

All on an Apple M3 Pro GPU (MPS), bf16, Qwen3-0.6B. vLLM comparison on an NVIDIA GPU: see
[Benchmark against vLLM](#benchmark-against-vllm).

### One optimization at a time

256 requests arriving at once: half are chats with a ~400-token system prompt, half are short
questions. Outputs are 16–256 tokens (36,044 in total, EOS ignored, so every configuration generates the
same tokens). Our configurations all get the same 2.4 GB of KV memory.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/ladder-throughput-dark.png">
  <img alt="Throughput: HF 176, static 251, continuous 388, paged 350, prefix caching 558 tok/s" src="docs/ladder-throughput-light.png">
</picture>

| Configuration | Throughput | Median TTFT | Median latency | Peak running |
|---|---:|---:|---:|---:|
| Hugging Face `generate()`, static batches of 32 | 176 tok/s | — | 130.5 s | 32 |
| Ours: static batching | 251 tok/s | 79.9 s | 82.7 s | 32 |
| Ours: continuous batching | 388 tok/s | 40.2 s | 50.2 s | 32 |
| Ours: + paged KV cache | 350 tok/s | 43.0 s | 64.8 s | 94 |
| Ours: + prefix caching | **558 tok/s** | **2.9 s** | **47.1 s** | 256 |

Paged attention on its own is ~10% *slower* than continuous batching on this mixed-length workload.
It runs ~3× more requests at once, but pays for it with 48 preemptions and with gathering every
request's blocks in plain PyTorch (vLLM fuses that gather into a custom kernel). The memory it frees
is what lets prefix caching keep 256 requests in flight. On a workload of short prompts, paging alone
is a clear win; see below.

### Paged KV cache: more requests in the same memory

256 requests, short prompts, mixed output lengths, the same KV memory for both caches (paged runs allow up
to 256 requests at once).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/paged-concurrency-dark.png">
  <img alt="Mean requests running: slot cache 28, paged 73" src="docs/paged-concurrency-light.png">
</picture>

| KV memory | Cache | Mean running | Throughput | Median TTFT |
|---|---|---:|---:|---:|
| 1.17 GB | one fixed slot per request | 28 | 632 tok/s | 23.6 s |
| 1.17 GB | **paged, 16-token blocks** | **73** | **758 tok/s** | **1.8 s** |
| 2.35 GB | one fixed slot per request | 50 | 827 tok/s | 14.4 s |
| 2.35 GB | **paged, 16-token blocks** | **112** | **963 tok/s** | **1.8 s** |

### Prefix caching: a shared system prompt is computed once

128 requests with the same 380-token system prompt and different questions, arriving at 6 per second.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/prefix-ttft-dark.png">
  <img alt="Median time to first token: 7,184 ms without prefix caching, 29 ms with it" src="docs/prefix-ttft-light.png">
</picture>

Without caching the engine can't keep up with 6 requests per second, so a queue builds. With it, each new request only
computes its own question: prefill work drops 27× (52,809 → 1,944 tokens).

## How it works

```
   HTTP clients (OpenAI SDK, Open WebUI, curl)        Dashboard (React)
          │  /v1/chat/completions, SSE streaming           ▲  WebSocket /v1/events
          ▼                                                │
   ┌────────────── server/ (FastAPI) ───────────────┐      │
   │  async handlers  ⇄  queue  ⇄  engine thread    │──────┘
   └────────────────────────┬───────────────────────┘
                            ▼
   ┌──────────────────── engine/ ───────────────────────────────────────┐
   │  Scheduler: who runs this step? admit · preempt · free            │
   │      │                                                             │
   │      ▼                                                             │
   │  Block manager: 16-token blocks, block tables, refcounts,          │
   │                 prefix hash table, LRU of cached blocks            │
   │      │                                                             │
   │      ▼                                                             │
   │  Model runner: Qwen3 in PyTorch (RMSNorm, RoPE, GQA + q/k norm,    │
   │                SwiGLU), attention reads K/V through block tables   │
   │      │                                                             │
   │      ▼                                                             │
   │  Sampler: greedy, temperature, top-p (per-request seeded RNG)      │
   └────────────────────────────────────────────────────────────────────┘
```

Each `step()` is one forward pass: either a **prefill**, which processes the prompts of newly admitted
requests, or a **decode**, which produces one token for every running request.

- **Continuous batching** ([`engine/scheduler.py`](engine/scheduler.py)). The batch is rebuilt every
  step, so finished requests leave immediately and waiting requests join as soon as there's room.
  The forward pass has no batch dimension: requests are concatenated into one flat token list, the
  way vLLM lays them out.
- **Paged KV cache** ([`engine/kv_cache.py`](engine/kv_cache.py)). KV memory is a pool of 16-token
  blocks. Each request has a block table (logical block → physical block), and blocks are handed out
  only as a request grows. When the pool runs dry, the newest request is **preempted**: its blocks are freed and it is
  recomputed later. Outputs don't change.
- **Prefix caching**. A full block's key is a chained hash of its tokens and everything before them.
  New requests reuse the longest cached prefix (reference counted), and finished requests' blocks
  stay cached until memory is needed, evicted least recently used first.
- **One attention layout** for fresh prompts, cached prefixes, and decode: queries in a padded
  `[requests, new tokens]` grid, keys gathered from each request's blocks. Decode batches with very
  different context lengths are attended in length-sorted groups to cut padding.
- **Server** ([`server/`](server/)). The engine runs on its own thread; asyncio handlers talk to it
  through queues. Responses stream as server-sent events, a client that disconnects is aborted
  and its memory freed, and stop strings and usage are reported the way OpenAI does.

The [dev logs](devlog/) walk through every stage, with the numbers, the bugs and the profiling
(one indexing call was 4× slower than another for the same copy on Apple GPUs).

## Run it

Requires Python 3.11 and about 4 GB of disk (PyTorch plus the 1.5 GB model). Works on Apple Silicon (MPS), NVIDIA (CUDA), or CPU.

```bash
git clone https://github.com/devt-code/llm-serving-engine.git && cd llm-serving-engine
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', local_dir='models/qwen3-0.6b')"
```

**Start the server** (OpenAI-compatible, on http://localhost:8000/v1):

```bash
python -m server.api
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
for chunk in client.chat.completions.create(
    model="qwen3-0.6b", messages=[{"role": "user", "content": "Explain a KV cache."}], stream=True
):
    print(chunk.choices[0].delta.content or "", end="")
```

**Open the dashboard** (needs Node 18+), then choose **Live engine**:

```bash
cd dashboard && npm install && npx vite      # http://localhost:5173
```

Click **Send 50 requests** or drag the arrival-rate slider, and watch requests join the batch,
blocks fill up, prefix blocks get shared, and requests get preempted when memory runs out.

**Use it as a library:**

```python
from engine.engine import LLMEngine
from engine.loader import load_model
from engine.request import SamplingParams

engine = LLMEngine(load_model(), kv_cache_gb=2.0)
outputs = engine.generate([[9707, 11, 1879]], SamplingParams(max_new_tokens=32))
```

## Tests

```bash
pytest
```

107 tests (about 3 minutes on a laptop). They:

- compare every model component with Hugging Face, using the real weights;
- check greedy generation on 20 prompts;
- check that batching, preemption, prefix caching and every block size leave outputs unchanged;
- run the block manager's bookkeeping with no leaked blocks;
- drive the server with the official `openai` client.

Two CUDA tests run on NVIDIA GPUs only.

## Benchmarks

| Script | What it measures |
|---|---|
| `bench/ladder.py` | The comparison ladder above (add `--only vllm` on an NVIDIA GPU) |
| `bench/batching.py` | Static vs continuous batching |
| `bench/paged.py` | Slot vs paged cache at equal memory, concurrency sweep |
| `bench/prefix.py` | Shared-system-prompt workload, burst and Poisson arrivals |
| `bench/hf_baseline.py`, `bench/engine_single.py` | Single-request baselines |
| `bench/record_demo.py` | Records the dashboard's replay traces |
| `bench/make_charts.py` | Renders the charts in this README |

### Benchmark against vLLM

[`notebooks/benchmark_vs_vllm.ipynb`](notebooks/benchmark_vs_vllm.ipynb) runs the whole ladder
plus vLLM on one NVIDIA GPU (free on Kaggle or Colab). It needs Linux and CUDA, so those
numbers aren't in this README yet.

## Project layout

```
engine/      model.py (Qwen3), scheduler.py, kv_cache.py (paged + prefix cache), batch_cache.py,
             engine.py (step loop), sampler.py, events.py (event stream), loader.py
server/      api.py (FastAPI, OpenAI-compatible), async_engine.py (engine thread)
dashboard/   React + TypeScript + Vite + Recharts; replay traces in public/traces/
bench/       benchmarks and results/
tests/       pytest suite
notebooks/   NVIDIA GPU checks and the vLLM comparison
devlog/      build notes, results and profiling for each stage
```

## Limits

This engine is for learning and measurement, not production: one GPU, one model family, and attention
written in plain PyTorch (no custom kernels, no CUDA graphs). Very high concurrency on a
small pool causes preemption thrashing, which the dashboard's memory-pressure recording shows.
