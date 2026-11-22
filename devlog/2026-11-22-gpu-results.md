# NVIDIA L4 results and the vLLM comparison (2026-11-22)

Ran `notebooks/benchmark_vs_vllm.ipynb` on Colab with an NVIDIA L4 (23 GB, compute capability 8.9):
Python 3.13, PyTorch 2.14.0+cu130, transformers 5.22.2, vLLM 0.34.0, bf16. Both engines ran on the
same PyTorch build. Raw numbers: `bench/results/ladder_gpu.json`.

## Results

Same workload as the Mac ladder: 256 requests at once, half with a 398-token shared system prompt,
36,044 output tokens, EOS ignored.

| Configuration | Throughput | Median TTFT | Peak running | Preemptions |
|---|---:|---:|---:|---:|
| HF `generate()`, static batches of 32 | 251 tok/s | — | 32 | — |
| Ours: static batching | 527 tok/s | 34.4 s | 32 | 0 |
| Ours: continuous batching | 762 tok/s | 19.0 s | 32 | 0 |
| Ours: + paged KV cache | 941 tok/s | 14.0 s | 94 | 48 |
| Ours: + prefix caching | 1,357 tok/s | 0.40 s | 256 | 85 |
| vLLM, same KV memory as ours | 5,914 tok/s | — | — | — |
| vLLM, default memory | 7,136 tok/s | — | — | — |

- Our best is 5.4× HF, 23% of vLLM at equal KV memory, and 19% of vLLM at its default memory.
- KV memory: ours is 21,024 tokens (32 slots × 657). Capped vLLM got 19,590 tokens (2.25 GiB), so
  the equal-memory comparison doesn't favor us. At `gpu_memory_utilization=0.5` vLLM gave itself
  75,495 tokens (8.64 GiB).
- Prefix caching: 88.8% of prompt tokens reused, all 256 requests running at the peak.
- vLLM run to run: the default-memory configuration measured 7,306 and then 7,136 tok/s.
- Both CUDA tests pass: fp16 logits against HF, and the paged engine with preemption.

## What broke on the way

1. **The Hugging Face import failed.** Colab's preinstalled torchaudio (2.11, built for CUDA 13.0)
   didn't match the PyTorch build (CUDA 13.2), and transformers imports torchaudio whenever it is
   installed. Fix: uninstall torchaudio; nothing here uses it.
2. **vLLM's engine process didn't start:** "Cannot re-initialize CUDA in forked subprocess". Fix:
   `VLLM_WORKER_MULTIPROC_METHOD=spawn`, now set by `bench/ladder.py` itself.
3. **The first comparison wasn't fair.** At `gpu_memory_utilization=0.5`, vLLM took 3.6× our KV
   memory. Added the `vllm_kv` configuration, which caps vLLM's KV cache at our budget with
   `kv_cache_memory_bytes`.

## Observations

- Paging alone is a win on CUDA (762 → 941 tok/s), where on the M3 Pro it was slower (388 → 350):
  on the L4, running 94 requests instead of 32 outweighs the cost of gathering blocks.
- The L4 is 1.4× the M3 Pro for HF `generate()` (251 vs 176 tok/s) but 2.4× for our prefix-caching
  configuration (1,357 vs 558).
- What vLLM has that this engine doesn't: CUDA graphs and `torch.compile` (its startup log reports
  27.6 s of compilation), fused attention kernels that read KV blocks in place, and chunked prefill
  (on by default, with an 8,192-token step budget).
