# Phase 1 results: model runner (2026-11-13)

## Status
- `engine/model.py` and `engine/sampler.py` are implemented: RMSNorm, RoPE, GQA attention with
  Qwen3 q/k norm, SwiGLU MLP, a contiguous per-request KV cache, and greedy/temperature/top-p sampling.
- `pytest`: **43/43 pass** (~50 s). This includes logits matching HF in fp32 on CPU, KV-cache decode
  and chunked prefill matching a full forward pass, greedy outputs identical to HF on 20 prompts,
  and bf16 on MPS within tolerance with ≥95% top-1 agreement.

## Speed: one request, batch 1 (M3 Pro, MPS, bf16, 128 new tokens, EOS ignored)
Both measured back to back on battery power, same prompt and settings.

| Engine | Prefill (ms) | Decode tok/s |
|---|---:|---:|
| HF transformers `generate()` | 37 | 58.8 |
| Ours, first version | 68 | 40.2 |
| Ours + fused RMSNorm + SDPA `enable_gqa` | 69 | 43.9 |

CPU fp32 (64 tokens): 27.7 tok/s decode. Raw data: `bench/results/engine_single*.json`.

## What I learned from profiling the gap
- A decode step processes ONE token, so the GPU does very little math per kernel. The CPU-side
  profile showed only ~4.7 ms of PyTorch overhead per step, but the step took ~23 ms: the time
  goes to the GPU executing ~1,500 tiny kernels, one by one.
- Weight bandwidth isn't the limit yet: one decode step reads ~0.9 GB of weights, which at the
  ~100 GB/s I measured for a big matmul is ~9 ms, well under the 23 ms per step.
- Ablations (each changes one thing, cost per decode step):
  - Fused `F.rms_norm` instead of pow/mean/add/rsqrt/mul: **−5.5 ms** (113 norms per step).
  - `repeat_interleave` for GQA copies the whole KV cache per layer per step; SDPA `enable_gqa=True` avoids it: −0.7 ms.
  - RoPE (cat + neg + 2 mul + add, ×56 per step) costs ~6 ms. Fusing it is a candidate.
  - Dropping the causal mask during decode: no measurable gain.
- Fully fused `F.rms_norm(x, weight)` changes bf16 results (weight multiply before the cast), so
  it fails `test_rmsnorm_bf16_order`. Normalize-only fused, then cast, then multiply is bitwise
  identical to HF.

## Why not chase parity at batch 1
Per-kernel overhead is a fixed cost per step. Phase 2's batching runs many requests in that same
step, spreading the cost across all of them, and that's the real throughput lever. Remaining batch-1 ideas
for later, each needing measurement: build the causal mask once per forward instead of per layer,
fused RoPE, prefill computing logits only for the last token (the 1024×151936 lm_head is wasted on
the other T−1 rows), and CUDA graphs / `torch.compile` on NVIDIA.

## Interview talking points
- No batch dimension: every forward takes a flat `[T]` token list plus positions (vLLM style).
- q/k norm runs before RoPE, per head, over head_dim.
- GQA: 16 query heads share 8 KV heads, which halves KV cache memory (112 KB/token instead of 224 KB).
- The causal mask needs an offset once a cache exists: query i sits at absolute position S−T+i.
  SDPA's `is_causal=True` assumes S == T and is wrong during decode and chunked prefill.
