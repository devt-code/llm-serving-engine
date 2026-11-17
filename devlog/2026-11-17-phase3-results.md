# Phase 3 results: paged KV cache and preemption (2026-11-17)

## What was built
| File | What it does |
|---|---|
| `engine/kv_cache.py` | `PagedKVCache` (pool of 16-token blocks), `BlockManager` (free list, block tables, events), `PagedPrefillBatch`, `PagedDecodeBatch` |
| `engine/scheduler.py` | Talks to a memory manager instead of owning slots. Preempts the newest running request when the pool runs out |
| `engine/batch_cache.py` | `SlotManager`: the Phase 2 slot cache behind the same interface, kept for comparison |
| `engine/engine.py` | `kv_cache="paged"` (default) or `"slot"`. Preempted requests are re-prefilled with prompt + generated tokens |
| `tests/test_kv_cache.py` | Block manager, paged = slot attention, block sizes 1–64, preemption keeps outputs exact, no leaks |
| `tests/test_cuda.py`, `notebooks/cuda_check.ipynb` | fp16 check on an NVIDIA GPU (Kaggle/Colab); skipped on the Mac |
| `bench/paged.py` | Slot vs paged at identical memory, plus a concurrency-cap sweep |

### How it works
- **Block table:** token at position p of a request lives in physical block `table[p // 16]`,
  offset `p % 16`. Blocks are taken from a free list only when a request grows into them and
  returned the moment it finishes. Waste is at most 15 tokens per request.
- **Admission:** a request is admitted if its current tokens fit with a 1% watermark of
  blocks kept free (room for running requests to grow).
- **Preemption:** before each decode step, every running request needs room for one more
  token. If the pool is empty, the most recently admitted request is preempted: its blocks are
  freed and it goes to the front of the queue. On readmission, prompt + already-generated tokens
  are prefilled again (recomputation) and decoding continues. Newest-first: least work lost,
  and the oldest requests keep making progress.
- **Attention:** prefill writes K/V through a flat slot mapping
  (`block * 16 + offset`). Decode gathers each request's blocks into a dense tensor with
  `index_select`, then runs masked SDPA.

## Correctness
- `pytest`: **82 passed, 2 skipped** (the CUDA tests, which need an NVIDIA GPU).
- Paged greedy outputs are identical to single-request `generate()` in fp32 for block sizes
  1, 4, 16, and 64, **including runs with forced preemption** (pools of 12, 20, and 40 blocks of
  4 tokens). Every allocated block is freed by the end, and preempted == resumed requests.
- Paged decode attention gives the same logits as the slot cache (within 1e-4 in fp32).
- A short block table now raises an error instead of silently writing into block 0 (another
  request's memory). A test hit exactly that while I was writing it.

## Speed: same KV memory, slot vs paged (M3 Pro, MPS, bf16, AC power)
Workload: 256 requests at t=0, outputs 16–255 tokens (36,044 total, EOS ignored), prompts
16–33 tokens, max_model_len 320. "Budget 32" = the memory of 32 Phase 2 slots of 320 tokens.

| Memory | Cache | Cap | Output tok/s | Mean running | Peak running | Decode step | Preemptions | TTFT p50 / p99 (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1.17 GB | slot | 32 | 631.6 | 28.2 | 32 | 40.3 ms | 0 | 23.64 / 49.71 |
| 1.17 GB | paged | 64 | 701.8 | 50.1 | 64 | 64.7 ms | 0 | 17.08 / 40.34 |
| 1.17 GB | paged | 128 | **760.1** | 67.7 | 128 | 77.1 ms | 133 | 2.50 / 27.28 |
| 1.17 GB | paged | 256 | 758.2 | **72.6** | 256 | 79.8 ms | 210 | **1.77 / 2.29** |
| 2.35 GB | slot | 64 | 826.9 | 50.1 | 64 | 53.8 ms | 0 | 14.39 / 32.58 |
| 2.35 GB | paged | 64 | 698.7 | 50.1 | 64 | 64.9 ms | 0 | 17.16 / 40.53 |
| 2.35 GB | paged | 128 | 864.3 | 79.4 | 128 | 83.7 ms | 0 | 2.51 / 26.95 |
| 2.35 GB | paged | 256 | **962.6** | **112.0** | 256 | 102.2 ms | 48 | **1.79 / 2.31** |

Raw data: `bench/results/paged.json`. Reproduce: `python bench/paged.py`.

### What the numbers say
- **Same memory, 2.2–2.6× more requests running on average** (72.6 vs 28.2 at 1.17 GB;
  112.0 vs 50.1 at 2.35 GB), and up to 8× more at peak.
- **Throughput: +20% at 1.17 GB (760 vs 632), +16% at 2.35 GB (963 vs 827).**
- **Half the memory, similar speed:** paged at 1.17 GB (760 tok/s) is within 8% of the slot
  cache at 2.35 GB (827 tok/s). At cap 64, the 1.17 GB pool runs 64 requests with zero
  preemptions, the same concurrency the slot cache needs 2.35 GB for.
- **Time to first token: median 1.8 s vs 14–24 s**, because requests start as soon as their
  prompt fits instead of waiting for a whole 320-token slot.
- **Preemption is a real trade-off:** cap 256 at 1.17 GB preempts 210 times (each one re-runs
  a prefill), yet its throughput matches cap 128, because the extra concurrency pays for the
  recomputation. It does buy a much better p99 TTFT (2.3 s vs 27.3 s).
- **The cost of paging without a kernel:** at the *same* batch size (64 requests, no
  preemption), a paged decode step is 64.9 ms vs 53.8 ms for the slot cache. That 11 ms is the
  gather copying each request's blocks into a dense tensor in every layer, which the slot
  cache avoids by reading a view. vLLM fuses this gather into its PagedAttention kernel; that's
  the Triton stretch goal (NVIDIA only).

### The bug that cost 4× (and how profiling found it)
The first paged run was **slower** than the slot cache (386 vs 622 tok/s at 1.17 GB). A
concurrency sweep showed that even with no preemptions, a paged decode step took 97.5 ms vs
40.9 ms at the same batch, so the problem was attention, not scheduling. A microbenchmark of
one layer's decode attention at B=32, L=288:

| Gather method | ms / layer | × 28 layers |
|---|---:|---:|
| Slot cache view (no gather, Phase 2) | 0.35 | 9.7 ms |
| `pool[tables]` (advanced indexing, first version) | 3.29 | 92.2 ms |
| `pool.index_select(0, tables.flatten())` | **0.84** | **23.6 ms** |
| one fused K+V pool, one `index_select` | 1.37 | 38.4 ms |

On MPS, advanced indexing with a 2-D index is ~4× slower than `index_select` on a flat
index, for the same bytes. Switching took throughput from 386 to 758 tok/s.

## Not done yet
- **CUDA check:** run `notebooks/cuda_check.ipynb` on a Kaggle or Colab T4 (fp16). It runs the
  CUDA tests (logits vs HF, paged engine with preemption) and both benchmarks.

## Interview talking points
- Why memory, not compute, caps the batch, and how block tables turn per-request reservation
  into on-demand allocation (OS virtual-memory analogy, vLLM / SOSP '23).
- Recompute vs swap for preemption: we recompute, which is cheap because prefill is batched
  and compute-bound. Swapping to CPU makes sense for long contexts.
- Newest-first victim selection and strict FCFS admission so preempted requests don't starve.
- Gather cost of paging without a fused kernel, measured, and the indexing-op trap on MPS.
