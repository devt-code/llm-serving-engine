"""Phase 3: paged KV cache, block manager, and preemption.

    pytest tests/test_kv_cache.py
"""

from types import SimpleNamespace

import pytest
import torch

from engine.engine import LLMEngine
from engine.events import EventLog
from engine.generate import generate
from engine.kv_cache import BlockManager
from engine.request import SamplingParams
from tests.prompts import PROMPTS

LENS = [5, 20, 12, 3, 17, 9, 20, 1, 14, 8, 11, 6]


# ---------------------------------------------------------------- block manager (no model)


def _bm(num_blocks=10, block_size=4, watermark=0.0):
    cache = SimpleNamespace(num_blocks=num_blocks, block_size=block_size)
    return BlockManager(cache, watermark=watermark)


_tok = iter(range(10**9))


def _r(rid="r", tokens=None):
    """Stand-in request. Distinct token ids by default, so no accidental prefix hits."""
    tokens = list(tokens) if tokens is not None else [next(_tok) for _ in range(256)]
    return SimpleNamespace(
        request_id=rid, block_table=None, block_hashes=[], prompt_ids=tokens, output_ids=[],
        all_token_ids=tokens, num_preemptions=0, num_computed=0,
    )


def test_allocate_rounds_up_to_blocks():
    bm = _bm()
    r = _r()
    bm.allocate(r, 9)  # 9 tokens / 4 per block -> 3 blocks
    assert len(r.block_table) == 3 and bm.num_free == 7
    assert len(set(r.block_table)) == 3


def test_ensure_capacity_adds_block_only_at_boundary():
    bm = _bm()
    r = _r()
    bm.allocate(r, 4)  # exactly 1 full block
    assert bm.ensure_capacity(r, 4) and len(r.block_table) == 1
    assert bm.ensure_capacity(r, 5) and len(r.block_table) == 2  # token 5 needs block 2
    assert bm.ensure_capacity(r, 8) and len(r.block_table) == 2
    assert bm.num_free == 8


def test_ensure_capacity_fails_when_pool_empty():
    bm = _bm(num_blocks=2)
    a, b = _r("a"), _r("b")
    bm.allocate(a, 4)
    bm.allocate(b, 4)
    assert bm.num_free == 0
    assert not bm.ensure_capacity(a, 5)
    assert len(a.block_table) == 1  # unchanged on failure


def test_free_returns_blocks_and_they_are_reused():
    bm = _bm()
    a, b = _r("a"), _r("b")
    bm.allocate(a, 12)
    blocks = list(a.block_table)
    bm.free(a)
    assert bm.num_free == 10 and a.block_table is None
    bm.allocate(b, 12)
    assert sorted(b.block_table) == sorted(blocks)


def test_no_block_shared_between_requests():
    bm = _bm(num_blocks=50)
    reqs = [_r(str(i)) for i in range(10)]
    for i, r in enumerate(reqs):
        bm.allocate(r, i + 1)
        bm.ensure_capacity(r, 3 * (i + 1))
    used = [b for r in reqs for b in r.block_table]
    assert len(used) == len(set(used)) == 50 - bm.num_free


def test_watermark_blocks_admission_not_growth():
    bm = _bm(num_blocks=10, watermark=0.2)  # keep 2 blocks free when admitting
    a = _r()
    assert bm.can_admit(a, 32)  # 8 blocks, leaves 2
    assert not bm.can_admit(a, 33)  # 9 blocks, leaves 1
    bm.allocate(a, 32)
    assert bm.ensure_capacity(a, 40)  # growth may use the watermark blocks


# ---------------------------------------------------------------- end to end (needs model)


@pytest.fixture(scope="module")
def prompts(tokenizer):
    return [tokenizer(p)["input_ids"] for p in PROMPTS[: len(LENS)]]


@pytest.fixture(scope="module")
def single_outputs(our_model, prompts):
    return [generate(our_model, p, n) for p, n in zip(prompts, LENS)]


def paged_engine(model, **kw):
    kw.setdefault("max_num_seqs", 8)
    kw.setdefault("max_model_len", 64)
    kw.setdefault("max_num_batched_tokens", 128)
    return LLMEngine(model, kv_cache="paged", **kw)


@pytest.mark.needs_model
@pytest.mark.parametrize("block_size", [1, 4, 16, 64])
def test_paged_matches_single_for_any_block_size(our_model, prompts, single_outputs, block_size):
    eng = paged_engine(our_model, block_size=block_size, num_kv_blocks=4096 // block_size)
    outs = eng.generate(prompts, [SamplingParams(max_new_tokens=n) for n in LENS])
    assert outs == single_outputs


@pytest.mark.needs_model
@torch.inference_mode()
def test_paged_decode_attention_matches_slot(our_model, tokenizer):
    """Same requests, one decode step: paged and slot caches give the same logits."""
    seqs = [tokenizer(p)["input_ids"] for p in PROMPTS[:5]]
    logits = {}
    for kind in ("slot", "paged"):
        kw = {"num_kv_blocks": 64, "block_size": 4} if kind == "paged" else {}
        eng = LLMEngine(our_model, max_num_seqs=8, max_model_len=64, max_num_batched_tokens=128, kv_cache=kind, **kw)
        reqs = [eng.add_request(s, SamplingParams(max_new_tokens=4)) for s in seqs]
        eng.step()  # prefill all
        for r in reqs:  # what the scheduler does before every decode step
            assert eng.memory.ensure_capacity(r, r.num_computed + 1)
        batch = eng.memory.decode_batch(reqs, [r.num_computed + 1 for r in reqs])
        ids = torch.tensor([r.last_token for r in reqs])
        pos = torch.tensor([r.num_computed for r in reqs])
        logits[kind] = our_model(ids, pos, batch)
    assert torch.allclose(logits["slot"], logits["paged"], atol=1e-4)


@pytest.mark.needs_model
@pytest.mark.parametrize("num_blocks", [12, 20, 40])
def test_preemption_keeps_outputs_exact(our_model, prompts, single_outputs, num_blocks):
    """A pool too small for everyone forces preemption + recomputation. Outputs must not change."""
    log = EventLog(keep_in_memory=True)
    eng = paged_engine(our_model, block_size=4, num_kv_blocks=num_blocks, event_log=log)
    outs = eng.generate(prompts, [SamplingParams(max_new_tokens=n) for n in LENS])
    assert outs == single_outputs
    if num_blocks <= 20:
        assert any(e["type"] == "request_preempted" for e in log.events), "expected preemption with a tiny pool"
    assert eng.memory.num_free == num_blocks, "blocks leaked"


@pytest.mark.needs_model
def test_preempted_requests_resume_and_events_balance(our_model, prompts):
    log = EventLog(keep_in_memory=True)
    eng = paged_engine(our_model, block_size=4, num_kv_blocks=12, event_log=log)
    eng.generate(prompts, [SamplingParams(max_new_tokens=n) for n in LENS])
    ev = log.events
    preempted = {e["id"] for e in ev if e["type"] == "request_preempted"}
    resumed = {e["id"] for e in ev if e["type"] == "request_resumed"}
    assert preempted and preempted == resumed
    # A resumed request can get its own old blocks back as a prefix hit (they stay cached after
    # preemption), so blocks come from either a fresh allocation or a prefix hit.
    allocated = sum(len(e["blocks"]) for e in ev if e["type"] in ("blocks_allocated", "prefix_cache_hit"))
    freed = sum(len(e["blocks"]) for e in ev if e["type"] == "blocks_freed")
    assert allocated == freed
    steps = [e for e in ev if e["type"] == "step"]
    assert all(0 <= s["kv_used"] <= 12 for s in steps)


def test_decode_batch_rejects_short_block_table():
    from engine.kv_cache import PagedDecodeBatch

    cache = SimpleNamespace(block_size=4, device="cpu")
    with pytest.raises(RuntimeError):
        PagedDecodeBatch(cache, [[3]], [5])  # 1 block = 4 tokens, but context is 5


@pytest.mark.needs_model
def test_rejects_request_larger_than_pool(our_model):
    eng = paged_engine(our_model, block_size=4, num_kv_blocks=5)  # 20 tokens total
    with pytest.raises(ValueError):
        eng.add_request(list(range(15)), SamplingParams(max_new_tokens=10))


@pytest.mark.needs_model
def test_paged_fits_more_requests_than_slots_in_same_memory(our_model, prompts):
    """The point of Phase 3: same KV memory, more requests running at once."""
    max_len, slots, bs = 64, 4, 4
    slot_eng = LLMEngine(our_model, max_num_seqs=slots, max_model_len=max_len, max_num_batched_tokens=128, kv_cache="slot")
    paged_eng = LLMEngine(
        our_model, max_num_seqs=64, max_model_len=max_len, max_num_batched_tokens=512, kv_cache="paged",
        block_size=bs, num_kv_blocks=slots * max_len // bs,  # identical bytes
    )
    assert slot_eng.cache.nbytes == paged_eng.cache.nbytes
    sp = [SamplingParams(max_new_tokens=n) for n in LENS]
    slot_eng.generate(prompts, sp)
    paged_eng.generate(prompts, sp)
    assert paged_eng.peak_running > 2 * slot_eng.peak_running


@pytest.mark.needs_model
@torch.inference_mode()
def test_length_grouped_decode_matches_single_group(our_model, tokenizer):
    """Decode attention split into length groups must give the same logits as one padded batch."""
    from engine.kv_cache import PagedAttentionBatch

    eng = LLMEngine(our_model, max_num_seqs=32, max_model_len=256, max_num_batched_tokens=4096,
                    kv_cache="paged", num_kv_blocks=1024, block_size=4, enable_prefix_caching=False)
    seqs = [tokenizer(p * (1 + 6 * (i % 3 == 0)))["input_ids"][:200] for i, p in enumerate(PROMPTS)]
    reqs = [eng.add_request(s, SamplingParams(max_new_tokens=4)) for s in seqs]
    eng.step()
    for r in reqs:
        assert eng.memory.ensure_capacity(r, r.num_computed + 1)
    ctx = [r.num_computed + 1 for r in reqs]
    ids = torch.tensor([r.last_token for r in reqs])
    pos = torch.tensor([r.num_computed for r in reqs])
    grouped = PagedAttentionBatch(eng.cache, [r.block_table for r in reqs], [1] * len(reqs), ctx)
    assert grouped.groups is not None, "mixed lengths should trigger grouping"
    single = PagedAttentionBatch(eng.cache, [r.block_table for r in reqs], [1] * len(reqs), ctx)
    single.groups = None
    a, b = our_model(ids, pos, grouped), our_model(ids, pos, single)
    assert torch.allclose(a, b, atol=1e-4)
