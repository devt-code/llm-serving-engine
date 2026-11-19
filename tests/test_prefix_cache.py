"""Prefix caching: block hashing, reference counts, LRU eviction, and exact outputs.

    pytest tests/test_prefix_cache.py
"""

from types import SimpleNamespace

import pytest

from engine.engine import LLMEngine
from engine.events import EventLog
from engine.generate import generate
from engine.kv_cache import BlockManager
from engine.request import SamplingParams

BS = 4


def _bm(num_blocks=16, watermark=0.0):
    return BlockManager(SimpleNamespace(num_blocks=num_blocks, block_size=BS), watermark=watermark,
                        enable_prefix_caching=True)


def _r(tokens, rid="r"):
    tokens = list(tokens)
    return SimpleNamespace(request_id=rid, block_table=None, block_hashes=[], prompt_ids=tokens,
                           output_ids=[], all_token_ids=tokens, num_preemptions=0, num_computed=0)


def _run(bm, req):
    """Allocate, pretend the prefill ran, register full blocks. Returns cached token count."""
    cached = bm.allocate(req, len(req.all_token_ids))
    req.num_computed = len(req.all_token_ids)
    bm.commit(req)
    return cached


# ---------------------------------------------------------------- block manager (no model)


def test_shared_prefix_reuses_blocks():
    bm = _bm()
    system = list(range(100, 112))  # 12 tokens = 3 full blocks
    a, b = _r(system + [1, 2, 3], "a"), _r(system + [7, 8], "b")
    assert _run(bm, a) == 0
    assert _run(bm, b) == 12
    assert a.block_table[:3] == b.block_table[:3]  # same physical blocks
    assert a.block_table[3] != b.block_table[3]  # private suffix blocks
    assert all(bm.ref_count[blk] == 2 for blk in a.block_table[:3])


def test_different_prefix_no_hit():
    bm = _bm()
    _run(bm, _r(range(12), "a"))
    assert _run(bm, _r([999] + list(range(1, 12)), "b")) == 0  # first block differs -> chain breaks


def test_hash_is_chained_not_per_block():
    """Same tokens in block 2 but a different block 1 must NOT hit: K/V depend on all earlier tokens."""
    bm = _bm()
    _run(bm, _r([1, 1, 1, 1, 5, 5, 5, 5, 0], "a"))
    assert _run(bm, _r([2, 2, 2, 2, 5, 5, 5, 5, 0], "b")) == 0


def test_fully_cached_sequence_recomputes_last_block():
    bm = _bm()
    tokens = list(range(12))  # exactly 3 blocks
    _run(bm, _r(tokens, "a"))
    assert _run(bm, _r(tokens, "b")) == 8  # last block recomputed so there's a token to sample from


def test_freed_blocks_stay_cached_and_hit_later():
    bm = _bm()
    a = _r(list(range(12)) + [50], "a")
    _run(bm, a)
    bm.free(a)
    assert bm.num_free == 16 and len(bm.evictable) == 3  # 3 full blocks kept, 1 partial freed
    assert _run(bm, _r(list(range(12)) + [60], "b")) == 12


def test_shared_block_not_freed_while_in_use():
    bm = _bm()
    a, b = _r(list(range(8)) + [1], "a"), _r(list(range(8)) + [2], "b")
    _run(bm, a)
    _run(bm, b)
    shared = a.block_table[:2]
    bm.free(a)
    assert all(bm.ref_count[blk] == 1 for blk in shared)
    assert not any(blk in bm.evictable or blk in bm.free_blocks for blk in shared)


def test_lru_eviction_when_pool_fills():
    bm = _bm(num_blocks=4)
    a = _r(range(8), "a")  # 2 blocks... but fully cached rule doesn't apply on first run
    _run(bm, a)
    bm.free(a)  # both blocks cached (evictable)
    c = _r(range(100, 116), "c")  # needs all 4 blocks: must evict both cached ones
    assert bm.can_admit(c, 16)
    _run(bm, c)
    assert len(bm.evictable) == 0 and len(bm.hash_to_block) == 4  # a's hashes gone, c's registered
    bm.free(c)
    assert _run(bm, _r(range(8), "a2")) == 0  # a's prefix was evicted


def test_decode_blocks_are_registered():
    """Blocks filled during decode (generated tokens) are cached too."""
    bm = _bm()
    r = _r([1, 2, 3], "r")
    _run(bm, r)
    assert len(r.block_hashes) == 0
    for tok in (4, 5, 6):  # grows to 6 tokens: block 0 becomes full at 4
        r.output_ids.append(tok)
        r.all_token_ids = r.prompt_ids + r.output_ids
        assert bm.ensure_capacity(r, len(r.all_token_ids))
        r.num_computed = len(r.all_token_ids)
        bm.commit(r)
    assert len(r.block_hashes) == 1


def test_can_admit_counts_evictable_hits_correctly():
    bm = _bm(num_blocks=3)
    a = _r(range(8), "a")
    _run(bm, a)
    bm.free(a)  # 2 cached blocks, 1 free
    # Same 8-token prefix + 4 more = 3 blocks: reuses 2 cached, needs 1 new -> fits exactly
    assert bm.can_admit(_r(list(range(8)) + [9, 9, 9, 9], "b"), 12)


# ---------------------------------------------------------------- end to end (needs model)


@pytest.fixture(scope="module")
def shared_prompts(tokenizer):
    system = ("You are a careful assistant for a hardware store. Answer briefly and politely. "
              "Always mention safety when tools are involved, and never recommend illegal uses. ")
    questions = ["How do I hang a heavy mirror?", "What drill bit for tile?", "Is a hammer drill needed for brick?",
                 "How to fix a squeaky door?", "Best glue for wood?", "How do I cut PVC pipe?"]
    return [tokenizer(system + q)["input_ids"] for q in questions]


@pytest.mark.needs_model
@pytest.mark.parametrize("block_size", [4, 16])
def test_prefix_caching_outputs_identical(our_model, shared_prompts, block_size):
    single = [generate(our_model, p, 12) for p in shared_prompts]
    for enable in (True, False):
        log = EventLog(keep_in_memory=True)
        eng = LLMEngine(our_model, max_num_seqs=8, max_model_len=128, max_num_batched_tokens=256,
                        block_size=block_size, num_kv_blocks=4096 // block_size,
                        enable_prefix_caching=enable, event_log=log)
        # One at a time first so later requests can hit what earlier ones cached.
        outs = []
        for p in shared_prompts:
            outs += eng.generate([p], SamplingParams(max_new_tokens=12))
        assert outs == single, f"prefix caching={enable}"
        hits = [e for e in log.events if e["type"] == "prefix_cache_hit"]
        assert (len(hits) >= len(shared_prompts) - 1) == enable


@pytest.mark.needs_model
def test_prefix_caching_with_concurrent_requests_and_preemption(our_model, shared_prompts):
    single = [generate(our_model, p, 20) for p in shared_prompts]
    eng = LLMEngine(our_model, max_num_seqs=8, max_model_len=128, max_num_batched_tokens=256,
                    block_size=4, num_kv_blocks=24, enable_prefix_caching=True)
    eng.generate(shared_prompts[:1], SamplingParams(max_new_tokens=1))  # warm the prefix
    outs = eng.generate(shared_prompts, SamplingParams(max_new_tokens=20))
    assert outs == single
    assert eng.scheduler.num_preemptions > 0
    assert eng.memory.usage()["prefix_hit_rate"] > 0.5
    assert all(c == 0 for c in eng.memory.ref_count), "blocks still referenced after everything finished"


@pytest.mark.needs_model
def test_prefix_hit_skips_prefill_compute(our_model, shared_prompts):
    log = EventLog(keep_in_memory=True)
    eng = LLMEngine(our_model, max_num_seqs=8, max_model_len=128, max_num_batched_tokens=256,
                    block_size=16, num_kv_blocks=256, event_log=log)
    eng.generate(shared_prompts[:1], SamplingParams(max_new_tokens=2))
    eng.generate(shared_prompts[1:2], SamplingParams(max_new_tokens=2))
    prefills = [e for e in log.events if e["type"] == "step" and e["kind"] == "prefill"]
    assert prefills[1]["num_tokens"] <= prefills[0]["num_tokens"] - 16  # at least one 16-token block skipped
