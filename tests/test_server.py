"""OpenAI-compatible API server, tested in-process with the official OpenAI client.

    pytest tests/test_server.py
"""

import json

import pytest
from fastapi.testclient import TestClient
from openai import BadRequestError, OpenAI

from engine.engine import LLMEngine
from engine.request import SamplingParams
from server.api import create_app
from server.async_engine import AsyncEngine

pytestmark = pytest.mark.needs_model

MODEL = "qwen3-test"


@pytest.fixture(scope="module")
def http(our_model, tokenizer):
    engine = LLMEngine(our_model, max_num_seqs=8, max_model_len=256, max_num_batched_tokens=512,
                       kv_cache="paged", num_kv_blocks=128)
    app = create_app(AsyncEngine(engine), tokenizer, MODEL)
    with TestClient(app) as client:  # runs the lifespan: starts and stops the engine thread
        yield client


@pytest.fixture(scope="module")
def client(http):
    return OpenAI(base_url="http://testserver/v1", api_key="unused", http_client=http)


def test_health_and_models(http, client):
    assert http.get("/health").json() == {"status": "ok"}
    assert [m.id for m in client.models.list().data] == [MODEL]


def test_chat_matches_engine_greedy(client, our_model, tokenizer):
    msgs = [{"role": "user", "content": "Name a primary color."}]
    r = client.chat.completions.create(model=MODEL, messages=msgs, temperature=0, max_tokens=12)
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer(text)["input_ids"]
    expected = LLMEngine(our_model, max_model_len=256, num_kv_blocks=64).generate([ids], SamplingParams(max_new_tokens=12))[0]
    assert r.choices[0].message.content == tokenizer.decode(expected, skip_special_tokens=True)
    assert r.choices[0].message.role == "assistant"
    assert r.usage.prompt_tokens == len(ids)
    assert r.usage.completion_tokens == len(expected)


def test_chat_stream_equals_non_stream(client):
    msgs = [{"role": "user", "content": "Count from 1 to 5."}]
    full = client.chat.completions.create(model=MODEL, messages=msgs, temperature=0, max_tokens=24)
    chunks = list(client.chat.completions.create(model=MODEL, messages=msgs, temperature=0, max_tokens=24,
                                                 stream=True, stream_options={"include_usage": True}))
    assert chunks[0].choices[0].delta.role == "assistant"
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text == full.choices[0].message.content
    finishes = [c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason]
    assert finishes == [full.choices[0].finish_reason]
    assert chunks[-1].usage.completion_tokens == full.usage.completion_tokens


def test_completion_and_stop_string(client):
    r = client.completions.create(model=MODEL, prompt="1, 2, 3, 4,", max_tokens=20, temperature=0)
    assert r.choices[0].text.strip().startswith("5")
    r = client.completions.create(model=MODEL, prompt="1, 2, 3, 4,", max_tokens=20, temperature=0, stop=[" 7"])
    assert " 7" not in r.choices[0].text and r.choices[0].finish_reason == "stop"
    assert r.choices[0].text.strip().startswith("5")


def test_completion_stream(client):
    full = client.completions.create(model=MODEL, prompt="The sky is", max_tokens=10, temperature=0)
    parts = list(client.completions.create(model=MODEL, prompt="The sky is", max_tokens=10, temperature=0, stream=True))
    assert "".join(p.choices[0].text for p in parts) == full.choices[0].text
    assert full.choices[0].finish_reason in ("stop", "length")


def test_max_tokens_length_finish(client):
    r = client.completions.create(model=MODEL, prompt="Once upon a time", max_tokens=5, temperature=0)
    assert r.choices[0].finish_reason == "length" and r.usage.completion_tokens == 5


def test_seeded_sampling_reproducible(client):
    kw = dict(model=MODEL, messages=[{"role": "user", "content": "Say something."}], temperature=1.0, top_p=0.9,
              seed=11, max_tokens=12)
    a = client.chat.completions.create(**kw).choices[0].message.content
    b = client.chat.completions.create(**kw).choices[0].message.content
    assert a == b


def test_errors(client):
    with pytest.raises(BadRequestError):
        client.completions.create(model=MODEL, prompt="hi", max_tokens=100_000)
    with pytest.raises(BadRequestError):
        client.completions.create(model=MODEL, prompt="hi", max_tokens=4, n=2)


def test_content_parts_and_extra_fields(http):
    """Clients like Open WebUI send content as parts and extra fields; both must be accepted."""
    r = http.post("/v1/chat/completions", json={
        "model": MODEL, "max_tokens": 4, "temperature": 0, "some_unknown_option": True,
        "messages": [{"role": "system", "content": "Be brief."},
                     {"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
    })
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["message"]["content"]


def test_stats_and_events_ws(http):
    with http.websocket_connect("/v1/events") as ws:
        assert ws.receive_json()["type"] == "hello"
        http.post("/v1/completions", json={"prompt": "Hello", "max_tokens": 3, "temperature": 0})
        types = set()
        while "request_finished" not in types:
            types.add(ws.receive_json()["type"])
    assert {"request_arrived", "request_started", "step", "request_finished"} <= types
    s = http.get("/v1/stats").json()
    assert s["running"] == 0 and s["kv_used"] == 0 and s["total_requests"] > 0
