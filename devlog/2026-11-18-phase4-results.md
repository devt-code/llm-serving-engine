# Phase 4 results: prefix caching and API server (2026-11-18)

## What was built
| File | What it does |
|---|---|
| `engine/kv_cache.py` | `BlockManager` gains prefix caching: chained block hashes, hash → block table, reference counts, LRU eviction of cached blocks. `PagedAttentionBatch` replaces the separate paged prefill/decode classes |
| `engine/scheduler.py` | Admission only budgets the *uncached* tokens; `abort()` for cancelled requests |
| `engine/engine.py` | Prefill starts at the end of the cached prefix; `commit()` registers newly full blocks after every step; stops on every generation-config EOS token; `abort_request()` |
| `engine/events.py` | `subscribe()` so the server can stream events live |
| `server/async_engine.py` | Runs the engine on its own thread; asyncio handlers submit requests and receive tokens |
| `server/api.py` | OpenAI-compatible FastAPI server: `/v1/models`, `/v1/completions`, `/v1/chat/completions` (streaming over SSE), `/v1/stats`, `/health`, WebSocket `/v1/events` |
| `tests/test_prefix_cache.py`, `tests/test_server.py` | 13 + 10 tests |
| `bench/prefix.py` | Shared-system-prompt workload, prefix caching on vs off, burst and Poisson arrivals |

### Prefix caching, in short
- A **full** block's K/V depend on its tokens and every token before it, so its key is a chain:
  `hash(block i) = hash(hash(block i-1), tokens of block i)`. Same tokens after a different
  prefix never match.
- On admission, the longest run of leading blocks found in the hash table is reused (ref
  count +1) and only the rest is prefilled. If the *whole* sequence is cached, the last block is
  recomputed anyway: one token must run through the model to produce logits.
- After every forward pass, blocks that just became full (prompt or generated tokens) are
  registered.
- When a block's last user finishes, its contents stay valid: it moves to an LRU list of
  evictable blocks. New allocations take truly free blocks first, then evict the least recently
  used cached block.
- Side effect: a **preempted request gets its own blocks back** as a prefix hit when it resumes,
  so recovering from preemption no longer redoes all its work.

### One attention layout for everything
Prefill over a cached prefix means new tokens must attend to old ones. The Phase 3 paged
prefill packed all prompts into one flat sequence with a block-diagonal mask, costing
(sum of lengths)². With cached context counted, that blows up. The new `PagedAttentionBatch`
gives each request `new` tokens on top of `context` tokens. It scatters queries into a padded
`[B, Tmax]` grid, gathers keys from each request's blocks into `[B, L]`, and runs one masked
SDPA. Cost is B × Tmax × L. Fresh prompts, cached prefixes, and decode (Tmax = 1) are all the same
code path, which also makes chunked prefill a small change later.

### Server design
The engine is synchronous PyTorch, so it runs on one background thread and nothing else touches
it, so no locks. Handlers put commands on a `queue.Queue`. After every step the engine thread pushes
each request's new tokens to that request's `asyncio.Queue` with `call_soon_threadsafe`. When a
streaming client disconnects or a stop string matches, the async generator closes and sends an
`abort`, which frees the request's blocks right away. Text is detokenized incrementally, holding
back partial multi-byte characters (U+FFFD) until the next token completes them.

## Correctness
- `pytest`: **105 passed, 2 skipped** (CUDA tests).
- All earlier exactness tests still pass with prefix caching **on by default**.
- Prefix caching outputs are identical to single-request `generate()` in fp32, with block sizes
  4 and 16, and with concurrent requests + forced preemption. Every reference count is back to
  0 at the end.
- Server: greedy chat output equals the engine's own output token for token; streamed text equals
  non-streamed text; stop strings, `finish_reason`, `usage`, seeded sampling, 400 errors,
  content-part messages and unknown fields (as Open WebUI sends) are all covered. All tests
  use the official `openai` Python client.

## Speed: shared system prompt (M3 Pro, MPS, bf16, AC power)
128 requests. Each = the same ~380-token system prompt + a different question (398-token prompt
after the chat template), 32–96 output tokens, EOS ignored. KV cache 2 GB, max 64 running.

| Arrivals | Prefix caching | Output tok/s | TTFT p50 | TTFT p99 | Latency p50 / p99 | Prefill tokens computed | Hit rate |
|---|---|---:|---:|---:|---:|---:|---:|
| burst (all at t=0) | off | 199.5 | 16,365 ms | 35,325 ms | 27.7 / 39.3 s | 52,800 / 51,160 | 0% |
| burst (all at t=0) | **on** | **446.3** | 5,119 ms | 12,666 ms | 13.0 / 17.5 s | 5,512 / 51,160 | 89% |
| Poisson, 6 req/s | off | 202.0 | 7,184 ms | 14,607 ms | 17.6 / 23.7 s | 52,809 / 51,160 | 0% |
| Poisson, 6 req/s | **on** | **338.8** | **29.3 ms** | **46.0 ms** | 3.9 / 6.7 s | 1,944 / 51,160 | 96% |

(Without caching, "computed" exceeds the prompt total because 4 preempted requests were
recomputed. With caching there were no preemptions: shared blocks take far less memory.)

Raw data: `bench/results/prefix.json`. Reproduce: `python bench/prefix.py`.

### What the numbers say
- **Prefill work drops 10–27×** (52.8k → 5.5k / 1.9k tokens computed).
- **Throughput 2.2× in the burst case** (200 → 446 tok/s): on this GPU, a 400-token prefill per
  request dominates when nothing is shared.
- **Time to first token at 6 req/s: 7.2 s → 29 ms median (≈245×), 14.6 s → 46 ms p99.** Without
  caching the engine can't keep up with 6 req/s (it needs ~400 tok/s of work and does ~200), so a
  queue builds and everyone waits. With caching each new request only prefills its ~20-token
  question and starts streaming almost immediately.
- Burst hit rate is 89%, not ~99%: requests admitted in the *first* prefill step can't hit,
  because nothing is cached until that step finishes. Steady arrivals reach 96%.

## Manual check with the OpenAI client (against `python -m server.api`)
- Chat: "The capital of Japan is Tokyo." Streaming TTFT 34 ms. Stop strings, seeded sampling,
  and 400 errors behave like OpenAI.
- 32 concurrent async chat requests are batched together: 1,391 tokens in 2.9 s (485 tok/s).
- `/v1/events` WebSocket streams `request_*`, `step`, and `blocks_*` events live.
- A streaming client that disconnects after 10 chunks is aborted: running = 0, KV used = 0 a
  second later.

## Not done yet
- **Open WebUI check.** Docker is installed but wasn't running, and the disk is 96% full (the
  image is several GB). To try it:
  ```bash
  python -m server.api --host 0.0.0.0 --port 8000
  docker run -d -p 3000:8080 -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
    -e OPENAI_API_KEY=unused -v open-webui:/app/backend/data --name open-webui ghcr.io/open-webui/open-webui:main
  ```
  then open http://localhost:3000 and pick `qwen3-0.6b`.
- **CUDA check** from Phase 3 (`notebooks/cuda_check.ipynb`) is still open.

## Interview talking points
- Chained block hashing: why the key must include all earlier tokens, and why the last block is
  always recomputed.
- Reference counts + an LRU of "free but still valid" blocks: caching costs no memory while
  space is plentiful and gives it back when it isn't.
- Prefix caching also reduces memory, so it removed all preemptions in this workload.
- Single engine thread + queues instead of locks; aborting on client disconnect so abandoned
  requests don't waste GPU time.
- Burst vs steady arrivals: why TTFT is only meaningful with realistic arrival patterns.
