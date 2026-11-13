# Phase 1 guide: the model runner (Nov 16 – Dec 6)

Goal: run Qwen3-0.6B with your own code, matching Hugging Face exactly.

## What's already there
| File | Status | What it does |
|---|---|---|
| `engine/config.py` | done | Reads `config.json` into a dataclass |
| `engine/loader.py` | done | Builds the model, copies safetensors weights by name, ties `lm_head` to the embeddings |
| `engine/generate.py` | done | Prefill once, then decode one token at a time with the KV cache |
| `engine/model.py` | **yours** | `__init__`s are done (they must match the checkpoint); every `forward` + `KVCache` is a TODO |
| `engine/sampler.py` | **yours** | Greedy / temperature / top-p |
| `tests/test_model.py` | done | Compares each of your components with HF's, using real layer weights |
| `tests/test_sampler.py` | done | Sampler unit tests (no model needed) |
| `bench/engine_single.py` | done | Tokens/s for your engine; same settings as the baseline |

Starting state: `pytest` shows 1 passed (weight loading) and 42 failed. Your job is to make it 43/43.

The tests were checked against a private reference implementation: they all pass when the code is
correct, and six common bugs (interleaved RoPE, wrong RMSNorm cast order, `is_causal` with a cache,
advancing the cache per layer, wrong GQA head mapping, missing q/k norm) each make at least one test fail.

## Suggested order (one or two sessions each)
1. **RMSNorm**: `pytest -x -k rmsnorm`
2. **RoPE** (`RotaryEmbedding.forward`, `apply_rotary`): `pytest -x -k rope`
3. **MLP**: `pytest -x -k mlp`
4. **Attention** (no cache yet; `kv_cache=None` path): `pytest -x -k attention`
5. **DecoderLayer**: `pytest -x -k decoder_layer`
6. **Qwen3Model + Qwen3ForCausalLM forward**: `pytest -x -k logits`
7. **KVCache** + the cache path in attention: `pytest -x -k kv_cache`
8. **Sampler**: `pytest tests/test_sampler.py`
9. **Greedy on 20 prompts**: `pytest -x -k greedy`
10. **bf16 on MPS**: `pytest -x -k mps`
11. **Measure**: `python bench/engine_single.py` and compare to the HF baseline (60 tok/s at batch 1)

## Design choice worth explaining in interviews
There's no batch dimension. Every forward pass takes a flat `[T]` list of tokens plus their positions.
That's how vLLM works, and it's why Phase 2 is manageable: a "batch" of requests at different stages
becomes one long flat sequence, and only attention needs to know where each request starts and ends.

## Done when
- [x] `pytest` → 43 passed
- [x] tokens/s recorded in `bench/results/engine_single.json` and the dev log
- [ ] You can explain: why q/k norm before RoPE, why GQA saves KV memory (8 KV heads vs 16 query heads),
      and why the causal mask needs an offset once a cache exists
