# Upstream contribution plan

Updated 2026-11-22. All three vLLM issues picked on 2026-11-20 now have assignees, and every
approachable vLLM issue checked today already had one to four competing PRs. Two smaller repos in
the same ecosystem have open, unclaimed bugs that match what this project built.

## What happened in vLLM

| Issue | Status on 2026-11-22 |
|---|---|
| [#33267](https://github.com/vllm-project/vllm/issues/33267) Remove attention layer name from `unified_kv_cache_update` | 2 assignees |
| [#32335](https://github.com/vllm-project/vllm/issues/32335) Extract KV-cache update from all attention backends | 9 assignees |
| [#27433](https://github.com/vllm-project/vllm/issues/27433) Batch-invariant feature and performance | 4 assignees |
| [#40004](https://github.com/vllm-project/vllm/issues/40004) Priority scheduling preempts running requests | Open PRs #54530, #65282, #65283 |
| [#66698](https://github.com/vllm-project/vllm/issues/66698) `bench serve` warmups prime the prefix cache | Reporter's own PRs #66701, #66707 |
| [#56653](https://github.com/vllm-project/vllm/issues/56653) `bench serve` overcounts peak concurrency | 4 competing PRs |
| [#69139](https://github.com/vllm-project/vllm/issues/69139) Scheduler over-admits with streaming sessions | Reporter has a fix branch |

In vLLM itself, either reply within the first hour of a new issue or find a bug of your own.

## Candidates

| Repo | Issue | Why it fits |
|---|---|---|
| [vllm-project/guidellm](https://github.com/vllm-project/guidellm) (vLLM's benchmarking tool) | [#1565](https://github.com/vllm-project/guidellm/issues/1565) Inter-token latency is biased low when usage counts the end-of-sequence token | The same TTFT and inter-token measurement this project's benchmarks do; pure Python, no GPU |
| vllm-project/guidellm | [#1592](https://github.com/vllm-project/guidellm/issues/1592) Request timings use `time.time()`, so a clock step adds seconds to latency | Latency should come from a monotonic clock; small fix plus a test |
| vllm-project/guidellm | [#1522](https://github.com/vllm-project/guidellm/issues/1522) Metrics with no observations are shown as 0.0 | Report correctness; small |
| [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) (a minimal vLLM) | [#279](https://github.com/GeeeekExplorer/nano-vllm/issues/279) A prompt longer than the whole KV cache blocks the queue and crashes | This engine rejects such requests up front (`max_request_len`) |
| GeeeekExplorer/nano-vllm | [#274](https://github.com/GeeeekExplorer/nano-vllm/issues/274) Crash when decode runs out of KV blocks | This engine preempts the newest request instead |
| GeeeekExplorer/nano-vllm | [#240](https://github.com/GeeeekExplorer/nano-vllm/issues/240) A new block is allocated one token too late | The block-boundary case this engine's paged-cache tests cover |

Before claiming one, read its comments (someone may have taken it since), and for nano-vllm check
that the maintainer is still merging PRs.

## How to do it
1. Read the whole thread, comment that you'd like to take the issue, and wait for a yes where the
   repo assigns work. Don't open a PR on an issue someone else has claimed.
2. Reproduce it first with a failing test. The guidellm issues need no GPU. nano-vllm's scheduler
   and block manager are small Python files, so a test that drives only them can run on a laptop.
3. Keep the PR small, link the issue, follow the repo's contributing guide and PR template, and
   answer review comments quickly.
4. vLLM stays the longer-term goal: watch new `good first issue` and `help wanted` issues, or turn
   one of this project's own findings into a reproducible vLLM bug report.
