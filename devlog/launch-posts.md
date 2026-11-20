# Launch posts (drafts)

Before posting:
1. Turn on GitHub Pages (repo Settings → Pages → Source: **GitHub Actions**) so the demo link works.
2. Ideally, run `notebooks/benchmark_vs_vllm.ipynb` first and add the "X% of vLLM" number. People on
   r/LocalLLaMA will ask for it.
3. Attach a 20–30 s screen recording of the dashboard (memory-pressure replay at 2×, then the
   steady-traffic one). It does more than any paragraph.

---

## r/LocalLLaMA

**Title:** I built an LLM serving engine from scratch (continuous batching, paged KV cache, prefix caching) and measured every step

**Body:**

I wanted to understand what vLLM actually does, so I wrote a serving engine from scratch in PyTorch
and ran Qwen3-0.6B on it. It has continuous batching, a paged KV cache with preemption, prefix
caching, and an OpenAI-compatible streaming API (works with the `openai` client).

Same model, GPU (Apple M3 Pro, bf16) and workload: 256 requests, half sharing a ~400-token system prompt.

| | tok/s | median TTFT |
|---|---:|---:|
| HF `generate()` | 176 | — |
| static batching | 251 | 79.9 s |
| continuous batching | 388 | 40.2 s |
| + paged KV cache | 350 | 43.0 s |
| + prefix caching | 558 | 2.9 s |

A few things I found:
- `pool[tables]` vs `pool.index_select(0, tables.flatten())` for the paged-attention gather: **4×
  difference on MPS** for identical bytes. One line doubled throughput.
- Paging alone was *slower* on mixed-length batches: padding every request to the longest context
  meant gathering about 3.5 GB of mostly padding per step. Sorting by length and attending in groups won back
  most of it.
- Shared system prompt at 6 req/s: median TTFT **7.2 s → 29 ms** with prefix caching.
- In bf16, batching changes a few outputs (near-ties flip); in fp32 they're identical. 107 tests
  check outputs against HF and across batching and preemption.

There's a dashboard that replays recorded runs in the browser (request timeline, KV block map, live
charts): https://devt-code.github.io/llm-serving-engine/

Code and full write-ups: https://github.com/devt-code/llm-serving-engine

Happy to answer questions. Feedback on the scheduler especially welcome.

---

## LinkedIn

I built an LLM serving engine from scratch: the software that runs a language model for many users
at once.

Training a model happens once; serving it happens millions of times a day, and that's where most
inference cost goes. I wanted to understand how engines like vLLM make it efficient, so I implemented the
core ideas myself in PyTorch, ran Qwen3-0.6B on it, and measured each one:

🔹 Continuous batching: 1.34× the throughput of static batching, and half the wait for the first token
🔹 Paged KV cache: 2.6× more users running at once in the same GPU memory
🔹 Prefix caching: time to first token from 7.2 s to 29 ms when requests share a system prompt
🔹 Overall: 3.2× the throughput of Hugging Face's `generate()` on the same hardware

The most useful lesson came from a slowdown. My first paged version was slower than the simple one.
Profiling traced it to a single indexing call that was 4× slower than an equivalent one on Apple
GPUs; changing one line doubled throughput.

It ships with an OpenAI-compatible API and a dashboard that shows every request and every block of
GPU memory as the engine works. You can watch a recorded run in your browser:
https://devt-code.github.io/llm-serving-engine/

Code and write-ups: https://github.com/devt-code/llm-serving-engine

#LLM #MachineLearning #MLSystems #PyTorch #Inference

---

## Resume lines

- Built an LLM inference engine from scratch in PyTorch (continuous batching, paged KV cache with
  preemption, prefix caching, OpenAI-compatible streaming API); **3.2× Hugging Face throughput**
  on the same hardware, **2.6× more concurrent requests** in the same KV memory, **TTFT 7.2 s → 29 ms**
  with prefix caching.
- Built a real-time React/TypeScript dashboard that visualizes batching and KV cache paging from the
  engine's event stream, with a replay mode for GPU-free public demos.
- (After the vLLM run) Reached **X% of vLLM's throughput** on an NVIDIA L4 with the same model and
  workload.
