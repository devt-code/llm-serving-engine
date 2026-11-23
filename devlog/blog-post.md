# I built an LLM serving engine from scratch. Here's what actually made it fast.

*Draft. Edit freely before publishing. Every number comes from `bench/results/` in the repo.*

A language model is trained once and then answers millions of requests. Serving them is where most
of the money goes, and the software that does it (vLLM, SGLang, TensorRT-LLM) has a handful of
ideas at its core. I wanted to understand those ideas well enough to build them, so I wrote a
serving engine from scratch in PyTorch, ran a real model on it (Qwen3-0.6B), and measured every
step.

**Code:** https://github.com/devt-code/llm-serving-engine ·
**Live demo:** https://devt-code--llm-serving-engine.us-east.modal.run ·
**Recorded runs:** https://devt-code.github.io/llm-serving-engine/

## The problem in one paragraph

A model writes one token at a time, and each token needs a full pass through the network. Reading
the prompt (*prefill*) is heavy math; generating tokens one by one (*decode*) is tiny math per step,
so the GPU spends most of its time waiting on memory. To avoid recomputing the past, every layer keeps
the keys and values of every previous token: the **KV cache**. For Qwen3-0.6B that's 112 KB per
token, so one 4,000-token conversation needs about 450 MB. A serving engine is the scheduler and
memory manager that keeps the GPU busy across many users without running out of that memory.

## Step 1: get the model exactly right

Before anything clever, I implemented Qwen3 myself: RMSNorm, rotary position embeddings, attention
with grouped-query heads and Qwen3's per-head query/key normalization, the SwiGLU MLP. I tested every
component against Hugging Face using the real weights. Greedy outputs match token for token on 20
prompts.

Writing the tests first paid off. I checked them by planting six common bugs (wrong RoPE pairing,
a causal mask that ignores the cache offset, the wrong mapping from query heads to KV heads, ...),
and every one made a test fail. Those tests caught real mistakes for the rest of the project.

## Step 2: continuous batching

The obvious way to serve many users is static batching: take 32 requests, run them until all 32
are done, take the next 32. The trouble is that requests finish at different times. I measured that
**42% of decode compute was wasted** on requests that had already finished but were waiting for the
longest one in their batch.

Continuous batching (from the Orca paper) rebuilds the batch at every step: finished requests leave
immediately, waiting ones join. Same engine, same model: **1.34× the throughput, and median time to
first token halved**.

## Step 3: paging the KV cache

Even with continuous batching, every request reserved KV memory for its *maximum* length up front. Memory, not
compute, then limits how many requests can run at once. vLLM's answer, borrowed from operating systems, is
**paging**: one shared pool of 16-token blocks, a block table per request, and blocks handed out only
as a request grows. When the pool runs dry, the newest request is preempted (its blocks freed) and
recomputed later.

In the same 1.17 GB of KV memory, **2.6× more requests ran at once** (73 vs 28), and median time to first
token dropped from 23.6 s to 1.8 s.

### The bug that cost 4×

The first paged version was *slower* than the simple one: 386 vs 622 tok/s. Capping concurrency
showed it wasn't the scheduler: at the same batch size, a paged decode step took 97 ms against 41 ms.
The time was all in gathering each request's blocks for attention. A microbenchmark of one layer found
the culprit: on Apple GPUs, `pool[tables]` (advanced indexing with a 2-D index) took **3.29 ms**, and
`pool.index_select(0, tables.flatten())` took **0.84 ms**, for exactly the same bytes. One line
changed, throughput doubled.

## Step 4: prefix caching

Most real traffic shares a prefix: an app's system prompt, a long document, a few-shot template.
If every request recomputes it, that's wasted work. Prefix caching gives every full block a key made
from a chained hash of its tokens and everything before them. A new request reuses the longest cached
run of blocks (reference counted), and blocks from finished requests stay cached until memory is
needed.

With 128 requests sharing a 380-token system prompt, arriving at 6 per second:

| | Median time to first token | Prompt tokens computed |
|---|---:|---:|
| Prefix caching off | 7,184 ms | 52,809 |
| Prefix caching on | **29 ms** | **1,944** |

Without caching the engine couldn't keep up with 6 requests per second, so a queue built and everyone waited
seven seconds. With caching each request only computes its own question.

## Putting it together

One workload, 256 requests (half with a long shared system prompt, half short), same GPU, same KV
memory for every configuration of mine:

| | Throughput | Median TTFT |
|---|---:|---:|
| Hugging Face `generate()` | 176 tok/s | — |
| Static batching | 251 tok/s | 79.9 s |
| Continuous batching | 388 tok/s | 40.2 s |
| + paged KV cache | 350 tok/s | 43.0 s |
| + prefix caching | **558 tok/s** | **2.9 s** |

Paging on its own went *down* here. In a batch that mixes 400-token and 30-token requests, padding every request
to the longest one meant gathering about 3.5 GB of mostly padding per step. Sorting requests by context
length and attending in a few groups won back most of it (281 → 350 tok/s), but it's still behind,
because paging also brought 48 preemptions. What paging buys is room: it's what lets prefix caching keep 256
requests in flight.

## Against vLLM

The same ladder on an NVIDIA L4, bf16, plus vLLM 0.34 with prefix caching on, run twice: once with
the KV memory it picks for itself, and once capped at the 2.41 GB my engine gets.

| | Throughput |
|---|---:|
| Hugging Face `generate()` | 251 tok/s |
| My engine, with prefix caching | 1,357 tok/s |
| vLLM, same KV memory | 5,914 tok/s |
| vLLM, its default memory | 7,136 tok/s |

That's 5.4× Hugging Face and 23% of vLLM at equal memory. The gap is what I deliberately didn't
build. vLLM captures CUDA graphs and compiles the model, while my engine issues about 1,500 small
kernels per step from Python. Its attention kernels read KV blocks in place, while mine gathers them
into padded tensors first. And it mixes chunks of new prompts into decode steps.

Two surprises. Paging alone was a clear win on the L4 (762 → 941 tok/s) after being slightly slower
on the Mac: there, running 94 requests instead of 32 outweighs the cost of gathering blocks. And getting vLLM to run
on Colab took two fixes: removing a preinstalled torchaudio built for a different CUDA version, and
starting vLLM's engine process with spawn, because forking it after CUDA had started failed.

## Things I didn't expect

- **Batching changes outputs, slightly.** In fp32, a request's output is identical whether it runs
  alone or in a batch of 16. In bf16, 13 of 20 prompts matched over 64 tokens; the others diverged at
  near-ties (two candidate tokens 0.05 apart, while bf16 can only resolve 0.125 at that size). Batching
  changes matrix shapes, which changes the order values are added in. Real engines live with this
  too.
- **At batch size 1, kernel launches are the bottleneck.** One decode step launches about 1,500 tiny GPU
  operations. Fusing five RMSNorm operations into one saved 5 ms per step, about a fifth of the step.
- **Time to first token only means something with realistic arrivals.** "Everything arrives at once" measures
  queueing, not the engine.

## The dashboard

The engine emits an event for every step, request, block allocation, preemption and prefix hit. A
small React app rebuilds the engine's state from that stream: a request timeline, a map of every KV
block, and live charts. It works live against the server, or replays recorded runs in the browser,
with no GPU. The public live demo runs on a small cloud CPU that starts on the first request and
shuts down a minute after the last viewer leaves, which keeps it within a free tier. The
memory-pressure recording shows preemption *thrashing*: some requests are readmitted, run a
few steps and are preempted again, which I would never have noticed in the numbers alone.

## What's next

- CUDA graphs for decode steps, to cut the per-step launch overhead.
- Chunked prefill, so new requests stop pausing everyone else's decoding.
- An upstream contribution in the vLLM ecosystem; the candidates are in the repo's dev log.

If you want to read the code, start with `engine/scheduler.py` and `engine/kv_cache.py`. The dev logs
in the repo have every measurement and every wrong turn.
