# Upstream contribution plan (vLLM)

Checked 2026-11-20 against vLLM's open `good first issue` list (14 open). None of them is in the
scheduler, block manager, prefix caching, or metrics; the two closest to what this project built
are in the KV cache write path.

## Candidates

| Issue | What | Why it fits | Check first |
|---|---|---|---|
| [#33267](https://github.com/vllm-project/vllm/issues/33267) Remove attention layer name from `unified_kv_cache_update` (opened 2026-01-28) | Small refactor of the KV cache update call | Same idea as our `attend()` writing K/V through a slot mapping; small and self-contained | Is anyone assigned or has anyone claimed it in the comments? |
| [#32335](https://github.com/vllm-project/vllm/issues/32335) Extract KV-cache update from all attention backends (opened 2026-01-14) | Move the K/V write out of each attention backend into one place | Exactly the split this engine has: the cache writes K/V, the backend only attends. 59 comments, so likely split into per-backend sub-tasks; pick one backend | Read the thread for which backends are still unclaimed |
| [#27433](https://github.com/vllm-project/vllm/issues/27433) Batch-invariant feature and performance (opened 2025-10-23) | Make outputs identical regardless of batch composition | We measured this exact effect (13/20 identical in bf16, near-ties flip). Larger and busier (107 comments) | Look for a small, well-defined sub-task |

Ruled out: [#25780](https://github.com/vllm-project/vllm/issues/25780) (preempted recomputation counted as prefix-cache hits), the best thematic fit, was closed as done on 2025-09-29. Our engine counts hits only on a request's first admission, which is the fix it describes.

## How to do it
1. Pick one candidate and read its full thread. Comment that you'd like to take it (or one
   sub-task), and wait for a maintainer to say yes. Don't start a PR on an issue someone claimed.
2. Development install, Python-only (no CUDA build needed for most refactors):
   `VLLM_USE_PRECOMPILED=1 pip install -e .` in a clone of vllm-project/vllm.
3. Testing the KV-cache paths needs Linux + an NVIDIA GPU: use Kaggle/Colab, or a rented GPU.
4. Keep the PR small, follow the PR template, run `pre-commit`, and answer review comments quickly.
5. Join the vLLM developer Slack (#contributors) for questions.

Beyond the `good first issue` label, search open issues for `prefix cache`, `scheduler`, `block`,
`metrics` and `benchmark`. Unlabeled bugs in those areas are often just as approachable.
