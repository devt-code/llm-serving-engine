"""Phase 2: batched engine (scheduler + slot KV cache) correctness.

The key property: a request's output must not depend on what else is in the batch.
Greedy outputs from the batched engine must equal single-request generate() exactly.

    pytest tests/test_engine.py
"""

import json

import pytest
import torch

from engine.engine import LLMEngine
from engine.events import EventLog
from engine.generate import generate
from engine.request import RequestStatus, SamplingParams
from engine.scheduler import Scheduler
from engine.request import Request
from engine.batch_cache import SlotManager
from types import SimpleNamespace
from tests.prompts import PROMPTS

pytestmark = pytest.mark.needs_model

# Mixed output lengths so requests finish at different times and slots get reused.
LENS = [5, 20, 12, 3, 17, 9, 20, 1, 14, 8, 11, 6]


@pytest.fixture(scope="module")
def prompts(tokenizer):
    return [tokenizer(p)["input_ids"] for p in PROMPTS[: len(LENS)]]


@pytest.fixture(scope="module")
def single_outputs(our_model, prompts):
    return [generate(our_model, p, n) for p, n in zip(prompts, LENS)]


def make_engine(model, **kw):
    kw.setdefault("max_num_seqs", 4)
    kw.setdefault("max_model_len", 128)
    kw.setdefault("max_num_batched_tokens", 128)
    if kw.setdefault("kv_cache", "paged") == "paged":
        kw.setdefault("num_kv_blocks", 256)
    return LLMEngine(model, **kw)


@pytest.mark.parametrize("kv_cache", ["slot", "paged"])
@pytest.mark.parametrize("policy", ["continuous", "static"])
@pytest.mark.parametrize("max_num_seqs", [1, 4, 16])
def test_batched_greedy_matches_single(our_model, prompts, single_outputs, policy, max_num_seqs, kv_cache):
    eng = make_engine(our_model, policy=policy, max_num_seqs=max_num_seqs, kv_cache=kv_cache)
    outs = eng.generate(prompts, [SamplingParams(max_new_tokens=n) for n in LENS])
    for i, (a, b) in enumerate(zip(outs, single_outputs)):
        assert a == b, f"request {i} differs when batched ({kv_cache}, {policy}, max_num_seqs={max_num_seqs})"


def test_prefill_token_budget_splits_admission(our_model, prompts, single_outputs):
    """A small token budget forces several prefill steps before decoding starts."""
    eng = make_engine(our_model, max_num_seqs=8, max_num_batched_tokens=128, max_model_len=64)
    eng.scheduler.max_num_batched_tokens = 20  # smaller than a few prompts combined
    outs = eng.generate(prompts, [SamplingParams(max_new_tokens=n) for n in LENS])
    assert outs == single_outputs


def test_requests_arriving_mid_run(our_model, prompts, single_outputs):
    """Requests added while others are decoding join at the next step and still match."""
    eng = make_engine(our_model, max_num_seqs=4)
    reqs = [eng.add_request(p, SamplingParams(max_new_tokens=n)) for p, n in zip(prompts[:3], LENS[:3])]
    for _ in range(4):
        eng.step()
    reqs += [eng.add_request(p, SamplingParams(max_new_tokens=n)) for p, n in zip(prompts[3:], LENS[3:])]
    while eng.has_unfinished():
        eng.step()
    assert [r.output_ids for r in reqs] == single_outputs


def test_seeded_sampling_independent_of_batch(our_model, prompts):
    """A seeded sampled request gives the same tokens alone or alongside other requests."""
    sp = SamplingParams(max_new_tokens=16, temperature=0.8, top_p=0.95, seed=123)
    alone = make_engine(our_model).generate([prompts[1]], sp)[0]
    eng = make_engine(our_model)
    others = [eng.add_request(p, SamplingParams(max_new_tokens=10)) for p in prompts[2:5]]
    target = eng.add_request(prompts[1], sp)
    while eng.has_unfinished():
        eng.step()
    assert target.output_ids == alone
    assert all(r.is_finished for r in others)


def test_eos_stops_request(our_model, tokenizer):
    eos = our_model.config.eos_token_id
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Say just the word hi."}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    ids = tokenizer(text)["input_ids"]
    eng = make_engine(our_model)
    req = eng.add_request(ids, SamplingParams(max_new_tokens=60))
    while eng.has_unfinished():
        eng.step()
    assert req.finish_reason == "stop" and req.output_ids[-1] == eos and len(req.output_ids) < 60
    # ignore_eos keeps going to max_new_tokens
    eng = make_engine(our_model)
    req = eng.add_request(ids, SamplingParams(max_new_tokens=60, ignore_eos=True))
    while eng.has_unfinished():
        eng.step()
    assert req.finish_reason == "length" and len(req.output_ids) == 60


def test_rejects_too_long(our_model):
    eng = make_engine(our_model, max_model_len=64, max_num_batched_tokens=64)
    with pytest.raises(ValueError):
        eng.add_request(list(range(10)), SamplingParams(max_new_tokens=60))


def test_events_log(our_model, prompts, tmp_path):
    path = tmp_path / "trace.jsonl"
    log = EventLog(path, keep_in_memory=True)
    eng = make_engine(our_model, event_log=log)
    eng.generate(prompts[:6], [SamplingParams(max_new_tokens=n) for n in LENS[:6]])
    log.close()
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert events == log.events
    types = [e["type"] for e in events]
    for t in ("request_arrived", "request_started", "first_token", "request_finished"):
        assert types.count(t) == 6, t
    steps = [e for e in events if e["type"] == "step"]
    assert len(steps) == eng.step_count
    assert all(e["num_seqs"] <= 4 for e in steps)
    assert [e["t"] for e in events] == sorted(e["t"] for e in events)


# ---------------------------------------------------------------- scheduler unit tests (no model)


def _slots(n):
    return SlotManager(SimpleNamespace(num_slots=n, max_len=100))


def _req(n_prompt, n_new=4):
    return Request(prompt_ids=[1] * n_prompt, params=SamplingParams(max_new_tokens=n_new), arrival_time=0.0)


def test_scheduler_continuous_refills_freed_slot():
    s = Scheduler(_slots(2), max_num_seqs=2, max_num_batched_tokens=100, policy="continuous")
    a, b, c = _req(5), _req(5), _req(5)
    for r in (a, b, c):
        s.add(r)
    step = s.schedule()
    assert step.kind == "prefill" and step.requests == [a, b] and {a.slot, b.slot} == {0, 1}
    assert s.schedule().kind == "decode"
    a_slot = a.slot
    a.status = RequestStatus.FINISHED
    assert s.release_finished() == [a]
    step = s.schedule()
    assert step.kind == "prefill" and step.requests == [c] and c.slot == a_slot  # reuses the freed slot


def test_scheduler_static_waits_for_whole_batch():
    s = Scheduler(_slots(2), max_num_seqs=2, max_num_batched_tokens=100, policy="static")
    a, b, c = _req(5), _req(5), _req(5)
    for r in (a, b, c):
        s.add(r)
    assert s.schedule().requests == [a, b]
    a.status = RequestStatus.FINISHED
    assert s.release_finished() == []  # a keeps its slot
    step = s.schedule()
    assert step.kind == "decode" and step.requests == [a, b]  # c still waits
    b.status = RequestStatus.FINISHED
    assert set(map(id, s.release_finished())) == {id(a), id(b)}
    assert s.schedule().requests == [c]


def test_scheduler_token_budget():
    s = Scheduler(_slots(8), max_num_seqs=8, max_num_batched_tokens=10, policy="continuous")
    reqs = [_req(4) for _ in range(5)]
    for r in reqs:
        s.add(r)
    assert len(s.schedule().requests) == 2  # 4 + 4 <= 10, a third would be 12
