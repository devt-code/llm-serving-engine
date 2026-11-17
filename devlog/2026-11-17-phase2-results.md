# Phase 2 results: scheduler and continuous batching (2026-11-17)

## What was built
| File | What it does |
|---|---|
| `engine/request.py` | `Request` (WAITING → RUNNING → FINISHED) and `SamplingParams` |
| `engine/scheduler.py` | Picks each step's requests. Two policies: `static` and `continuous` |
| `engine/batch_cache.py` | `SlotKVCache` (one fixed region per request), `PrefillBatch`, `DecodeBatch` |
| `engine/engine.py` | `LLMEngine`: schedule → build batch → forward → sample → update |
| `engine/events.py` | JSONL event stream for the dashboard |
| `engine/model.py` | Attention now calls `kv_cache.attend(...)`, so the cache type decides how keys are read (vLLM's "attention backend" idea). Added `logits_indices` so prefill only computes logits for each prompt's last token |
| `bench/batching.py` | Static vs continuous benchmark; writes traces to `bench/traces/` |

### How a step works
- **Prefill step:** newly admitted prompts are concatenated into one flat sequence. A
  block-diagonal causal mask stops prompts seeing each other. Each token's K/V is written into
  its request's slot.
- **Decode step:** one token per running request. Attention runs directly over the slot array
  (`cache[:n, :L]` is a view, no copy) with a per-slot length mask. Slots are handed out
  lowest-first so `n` stays close to the number of running requests.
- **Scheduling:** FCFS. If anything can be admitted (free slot and prompt-token budget), the
  step is a prefill; otherwise it decodes everyone. Continuous frees a slot the moment a request
  finishes. Static keeps finished requests in the batch (still computed, output discarded) until
  the whole batch is done, which is what `generate()` on a list of prompts does.

## Correctness
- `pytest`: **58/58 pass** (43 from Phase 1 + 15 new).
- fp32 on CPU: batched greedy outputs are **identical** to single-request `generate()` for 12
  requests with mixed lengths, at 1, 4, and 16 slots, under both policies. This also holds with a
  tiny prefill budget and with requests arriving mid-run. Seeded sampling gives the same tokens
  alone or in a batch.
- bf16 on MPS: 13/20 prompts identical to single-request over 64 tokens. The rest diverge
  due to rounding. Example: prompt 11's first token is a near-tie (" Also" 16.995 vs " The" 16.942 in fp32;
  bf16 resolution at 17 is 0.125). Batching changes matmul shapes, which changes the order values
  are summed in, which flips near-ties. This is the known "batch invariance" problem and
  real engines have it too; the fp32 tests show the logic is exact.

## Speed (M3 Pro, MPS, bf16, AC power)
Workload: 128 requests, all arriving at t=0, prompts 16–33 tokens, outputs 16–254 tokens
(uniform, fixed seed, EOS ignored, 18,838 output tokens total). Same tokens for every config.

| Policy | Batch | Output tok/s | Total (s) | TTFT p50 / p99 (s) | Latency p50 / p99 (s) | Decode steps | Decode utilization |
|---|---:|---:|---:|---:|---:|---:|---:|
| static | 32 | 435.4 | 43.3 | 22.17 / 32.78 | 22.8 / 43.2 | 1002 | 58% |
| continuous | 32 | **583.2** | 32.3 | 10.81 / 23.73 | 17.1 / 31.9 | 702 | 100% |
| static | 64 | 619.6 | 30.4 | 15.64 / 15.64 | 16.4 / 30.3 | 503 | 58% |
| continuous | 64 | **744.7** | 25.3 | 1.96 / 14.74 | 15.0 / 25.2 | 429 | 100% |

For reference, one request at a time (Phase 1) is 43.9 tok/s, and HF `generate()` on 32 prompts
is 530 tok/s (Phase 0, fixed 128 output tokens each, so not the same workload).

Raw data: `bench/results/batching.json`. Traces (gitignored): `bench/traces/phase2_*.jsonl`.
Reproduce: `python bench/batching.py`.

### What the numbers say
- **Continuous batching: 1.34× throughput at batch 32, 1.20× at batch 64.**
- **Static wastes 42% of decode compute.** Only 58% of the rows it computes produce a kept
  token; the rest are finished requests riding along until the longest one in the batch is done.
- **Time to first token is where continuous wins most.** At batch 64, median TTFT drops from
  15.6 s to 2.0 s, because a waiting request starts as soon as any slot frees up instead of
  waiting for the whole batch to drain.
- **Why not the full 1/0.58 = 1.7×:** (1) at the end of the run the queue empties and fewer than
  `batch` requests are left running, which static batching doesn't suffer from in the same way;
  (2) every admission is a prefill step that pauses all decoding for one step (86 prefill
  steps at batch 32 versus 4 for static).
- Bigger batches help both policies (one decode step costs about the same for 32 or 64 requests,
  because it's dominated by fixed per-kernel overhead and weight reads). But memory caps the
  batch: this cache reserves `max_model_len` tokens per slot. Batch 64 × 320 tokens = 2.3 GB,
  even though the average request only needs ~160 tokens. **That's the Phase 3 problem.**

## Next optimizations (each needs a measurement before it counts)
- Mix prefill and decode tokens in one step (chunked prefill) to remove the decode pauses.
- Decode attention over the slot array computes garbage rows for unused slots inside `[0, n)`;
  the paged cache replaces this anyway.

## Interview talking points
- Static vs continuous: iteration-level scheduling (Orca, OSDI '22). The batch is rebuilt every
  forward pass, so finished requests leave and new ones join immediately.
- Packing prompts with no batch dimension: concatenate, build a block-diagonal causal mask, and
  compute logits only at each prompt's last token.
- Prefill-first scheduling trades decode smoothness for low TTFT.
- Batch invariance: identical in fp32, near-ties flip in bf16, and why that's expected.
