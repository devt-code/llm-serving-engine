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


def test_public_deployment_limits(our_model, tokenizer, tmp_path):
    """The options the Docker image turns on: dashboard at /, output cap, demo cap, busy -> 429."""
    (tmp_path / "index.html").write_text("<h1>dashboard</h1>")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "index-1a2b3c.js").write_text("console.log('dashboard')")
    body = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "temperature": 0}

    def app_with(**options):
        engine = LLMEngine(our_model, max_num_seqs=4, max_model_len=256, max_num_batched_tokens=512,
                           kv_cache="paged", num_kv_blocks=64)
        return create_app(AsyncEngine(engine), tokenizer, MODEL, **options)

    with TestClient(app_with(max_tokens_cap=8, demo_max_count=3, dashboard_dir=str(tmp_path))) as http:
        page = http.get("/")
        assert page.text == "<h1>dashboard</h1>" and page.headers["cache-control"] == "no-store"
        assert "immutable" in http.get("/assets/index-1a2b3c.js").headers["cache-control"]
        assert http.get("/health").json() == {"status": "ok"}  # API routes still win over the mount
        assert http.post("/v1/chat/completions", json={**body, "max_tokens": 9}).status_code == 400
        assert http.post("/v1/chat/completions", json=body).json()["usage"]["completion_tokens"] <= 8
        assert http.post("/v1/demo/load", json={"count": 10, "max_tokens": 8}).json()["queued"] == 3

    with TestClient(app_with(max_waiting=0)) as http:
        r = http.post("/v1/chat/completions", json=body)
        assert r.status_code == 429 and r.json()["error"]["type"] == "rate_limit_error"
        assert http.post("/v1/demo/load", json={"count": 5}).json()["queued"] == 0


def test_idle_event_stream_is_closed(our_model, tokenizer):
    """A dashboard watching an idle engine is disconnected, so a host billed per open connection can scale to zero."""
    from starlette.websockets import WebSocketDisconnect

    engine = LLMEngine(our_model, max_num_seqs=2, max_model_len=256, max_num_batched_tokens=512,
                       kv_cache="paged", num_kv_blocks=32)
    app = create_app(AsyncEngine(engine), tokenizer, MODEL, ws_idle_timeout=0.2, device_label="test device")
    with TestClient(app) as http, http.websocket_connect("/v1/events") as ws:
        assert ws.receive_json()["device"] == "test device"
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()
        assert closed.value.reason == "idle"


def test_response_id_names_the_request_in_engine_events(our_model, tokenizer):
    """The dashboard finds a visitor's own request in the event stream by the response id."""
    engine = LLMEngine(our_model, max_num_seqs=2, max_model_len=256, max_num_batched_tokens=512,
                       kv_cache="paged", num_kv_blocks=32)
    app = create_app(AsyncEngine(engine), tokenizer, MODEL)
    with TestClient(app) as http, http.websocket_connect("/v1/events") as ws:
        ws.receive_json()  # hello
        body = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2}
        rid = http.post("/v1/chat/completions", json=body).json()["id"]
        event = ws.receive_json()
        while event["type"] != "request_arrived":
            event = ws.receive_json()
        assert rid.startswith("chatcmpl-") and event["id"] == rid


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


def test_demo_load_starts_requests_server_side(http):
    import time

    r = http.post("/v1/demo/load", json={"count": 5, "max_tokens": 8, "shared_prefix": True})
    assert r.status_code == 200 and r.json()["queued"] == 5
    deadline = time.time() + 60
    while time.time() < deadline:
        s = http.get("/v1/stats").json()
        if s["running"] == 0 and s["waiting"] == 0 and s["total_requests"] >= 5:
            break
        time.sleep(0.2)
    assert s["running"] == 0 and s["kv_used"] == 0
