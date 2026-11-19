"""OpenAI-compatible HTTP server for the engine.

    python -m server.api                       # serves models/qwen3-0.6b on http://localhost:8000
    python -m server.api --port 8080 --kv-cache-gb 4 --trace bench/traces/server.jsonl

Endpoints
    GET  /health
    GET  /v1/models
    POST /v1/completions          prompt -> text            (stream: server-sent events)
    POST /v1/chat/completions     messages -> assistant msg (stream: server-sent events)
    GET  /v1/stats                running / waiting / KV usage / prefix hit rate
    WS   /v1/events               live engine events as JSON (for the dashboard)

Works with the official OpenAI client:
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
    client.chat.completions.create(model="qwen3-0.6b", messages=[{"role": "user", "content": "hi"}])
"""

import argparse
import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from engine.request import SamplingParams
from server.async_engine import AsyncEngine


# ---------------------------------------------------------------------------- request schemas
# Unknown fields are accepted and ignored, like most OpenAI-compatible servers do, so clients
# that send extra parameters (Open WebUI, LangChain, ...) still work.


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="allow")


class StreamOptions(_Lenient):
    include_usage: bool = False


class CompletionRequest(_Lenient):
    model: str | None = None
    prompt: str | list[int]
    max_tokens: int | None = 16
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    ignore_eos: bool = False  # not OpenAI; useful for benchmarks


class ChatMessage(_Lenient):
    role: str
    content: str | list[dict[str, Any]] | None = ""


class ChatCompletionRequest(_Lenient):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)  # e.g. {"enable_thinking": true}


# ---------------------------------------------------------------------------- helpers


def openai_error(status: int, message: str, type_: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"message": message, "type": type_, "code": status}})


class Detokenizer:
    """Turns a growing list of token ids into text deltas, and applies stop strings.

    Decodes the whole output each time and emits what's new. A token can end in the middle of
    a multi-byte character (emoji, CJK), which decodes to U+FFFD; we hold that back until the
    next token completes it.
    """

    def __init__(self, tokenizer, stop: list[str]):
        self.tok = tokenizer
        self.stop = [s for s in stop if s]
        self.ids: list[int] = []
        self.text = ""  # text emitted so far
        self.stopped = False

    def add(self, new_ids: list[int], final: bool) -> str:
        self.ids.extend(new_ids)
        full = self.tok.decode(self.ids, skip_special_tokens=True)
        if full.endswith("�") and not final:
            return ""
        for s in self.stop:
            idx = full.find(s)
            if idx != -1:
                full = full[:idx]
                self.stopped = True
        delta = full[len(self.text) :]
        self.text = full
        return delta


def _stop_list(stop: str | list[str] | None) -> list[str]:
    if stop is None:
        return []
    return [stop] if isinstance(stop, str) else list(stop)


def _message_text(content) -> str:
    """OpenAI allows content as a string or a list of parts; keep only text parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if p.get("type") == "text")


def _finish_reason(reason: str | None, detok: Detokenizer) -> str:
    if detok.stopped:
        return "stop"
    return {"stop": "stop", "length": "length"}.get(reason or "", "stop")


def _sse(obj: dict | str) -> str:
    return f"data: {obj if isinstance(obj, str) else json.dumps(obj)}\n\n"


# ---------------------------------------------------------------------------- app


def create_app(async_engine: AsyncEngine, tokenizer, model_name: str) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not async_engine._thread.is_alive():
            async_engine.start()
        yield
        async_engine.shutdown()

    app = FastAPI(title="LLM Serving Engine", lifespan=lifespan)
    # The dashboard (Phase 5) runs on another port during development.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    def sampling_params(body, prompt_len: int, max_tokens: int | None, stop_ids=()) -> SamplingParams | JSONResponse:
        if body.n != 1:
            return openai_error(400, "only n=1 is supported")
        room = async_engine.max_request_len - prompt_len
        if room < 1:
            return openai_error(400, f"prompt is {prompt_len} tokens; the limit is {async_engine.max_request_len - 1}")
        if max_tokens is None:
            max_tokens = room
        if max_tokens < 1 or max_tokens > room:
            return openai_error(
                400, f"max_tokens must be between 1 and {room} for this prompt "
                     f"({prompt_len} prompt tokens, {async_engine.max_request_len} total)")
        return SamplingParams(
            max_new_tokens=max_tokens, temperature=max(0.0, body.temperature), top_p=body.top_p,
            seed=body.seed, ignore_eos=getattr(body, "ignore_eos", False), stop_token_ids=tuple(stop_ids),
        )

    # ------------------------------------------------------------ basic

    @app.get("/health")
    async def health():
        if async_engine.error is not None:
            return openai_error(500, f"engine crashed: {async_engine.error!r}", "server_error")
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{
            "id": model_name, "object": "model", "created": int(async_engine.started_at),
            "owned_by": "llm-serving-engine", "max_model_len": async_engine.max_request_len,
        }]}

    @app.get("/v1/stats")
    async def stats():
        return async_engine.stats()

    # ------------------------------------------------------------ completions

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest, request: Request):
        prompt_ids = body.prompt if isinstance(body.prompt, list) else tokenizer(body.prompt)["input_ids"]
        if not prompt_ids:
            return openai_error(400, "prompt is empty")
        params = sampling_params(body, len(prompt_ids), body.max_tokens)
        if isinstance(params, JSONResponse):
            return params
        rid, created = f"cmpl-{uuid.uuid4().hex[:24]}", int(time.time())
        detok = Detokenizer(tokenizer, _stop_list(body.stop))

        def chunk(text: str, finish: str | None) -> dict:
            return {"id": rid, "object": "text_completion", "created": created, "model": model_name,
                    "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish}]}

        if body.stream:
            async def events() -> AsyncIterator[str]:
                usage = None
                async for out in _run(async_engine.generate(prompt_ids, params), detok, request):
                    delta, done = out
                    if delta or done is not None:
                        yield _sse(chunk(delta, done[0] if done else None))
                    if done:
                        usage = done[1]
                if body.stream_options and body.stream_options.include_usage and usage:
                    yield _sse({"id": rid, "object": "text_completion", "created": created, "model": model_name,
                                "choices": [], "usage": usage})
                yield _sse("[DONE]")
            return StreamingResponse(events(), media_type="text/event-stream")

        text, finish, usage = await _collect(async_engine.generate(prompt_ids, params), detok, request)
        if finish is None:
            return openai_error(499, "client disconnected")
        resp = chunk(text, finish)
        resp["usage"] = usage
        return resp

    # ------------------------------------------------------------ chat

    @app.post("/v1/chat/completions")
    async def chat(body: ChatCompletionRequest, request: Request):
        messages = [{"role": m.role, "content": _message_text(m.content)} for m in body.messages]
        kwargs = {"enable_thinking": False, **body.chat_template_kwargs}  # Qwen3: no <think> by default
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
        except Exception as e:
            return openai_error(400, f"could not apply chat template: {e}")
        prompt_ids = tokenizer(text)["input_ids"]
        params = sampling_params(body, len(prompt_ids), body.max_completion_tokens or body.max_tokens)
        if isinstance(params, JSONResponse):
            return params
        rid, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())
        detok = Detokenizer(tokenizer, _stop_list(body.stop))

        def chunk(delta: dict, finish: str | None) -> dict:
            return {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                    "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}]}

        if body.stream:
            async def events() -> AsyncIterator[str]:
                yield _sse(chunk({"role": "assistant", "content": ""}, None))
                usage = None
                async for delta, done in _run(async_engine.generate(prompt_ids, params), detok, request):
                    if delta:
                        yield _sse(chunk({"content": delta}, None))
                    if done:
                        yield _sse(chunk({}, done[0]))
                        usage = done[1]
                if body.stream_options and body.stream_options.include_usage and usage:
                    yield _sse({"id": rid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                                "choices": [], "usage": usage})
                yield _sse("[DONE]")
            return StreamingResponse(events(), media_type="text/event-stream")

        content, finish, usage = await _collect(async_engine.generate(prompt_ids, params), detok, request)
        if finish is None:
            return openai_error(499, "client disconnected")
        return {
            "id": rid, "object": "chat.completion", "created": created, "model": model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "logprobs": None, "finish_reason": finish}],
            "usage": usage,
        }

    # ------------------------------------------------------------ live events for the dashboard

    @app.websocket("/v1/events")
    async def events_ws(ws: WebSocket):
        await ws.accept()
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=10_000)

        def push(event: dict) -> None:  # runs on the engine thread
            def put():
                if not q.full():  # a slow viewer drops events rather than slowing the engine
                    q.put_nowait(event)
            loop.call_soon_threadsafe(put)

        async_engine.engine.events.subscribe(push)
        try:
            await ws.send_json({"type": "hello", "stats": async_engine.stats(), "model": model_name})
            while True:
                await ws.send_json(await q.get())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            async_engine.engine.events.unsubscribe(push)

    return app


async def _run(gen, detok: Detokenizer, request: Request) -> AsyncIterator[tuple[str, tuple | None]]:
    """Drive one generation: yields (text_delta, None) while running and (delta, (finish_reason,
    usage)) at the end. Stops early on a stop string or when the client goes away; closing the
    generator aborts the request inside the engine."""
    try:
        async for out in gen:
            delta = detok.add(out.new_token_ids, final=out.finished)
            usage = {"prompt_tokens": out.num_prompt_tokens, "completion_tokens": out.num_output_tokens,
                     "total_tokens": out.num_prompt_tokens + out.num_output_tokens}
            if out.finished or detok.stopped:
                yield delta, (_finish_reason(out.finish_reason, detok), usage)
                return
            yield delta, None
            if await request.is_disconnected():
                return
    finally:
        await gen.aclose()


async def _collect(gen, detok: Detokenizer, request: Request) -> tuple[str, str | None, dict | None]:
    finish, usage = None, None
    async for _, done in _run(gen, detok, request):
        if done:
            finish, usage = done
    return detok.text, finish, usage


# ---------------------------------------------------------------------------- CLI


def main():
    import torch
    import uvicorn
    from transformers import AutoTokenizer

    from engine.engine import LLMEngine
    from engine.events import EventLog
    from engine.loader import load_model, pick_device

    ap = argparse.ArgumentParser(description="OpenAI-compatible server for the LLM serving engine")
    ap.add_argument("--model", default="models/qwen3-0.6b")
    ap.add_argument("--served-model-name", default=None, help="model id clients use (default: folder name)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--kv-cache-gb", type=float, default=2.0)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-batched-tokens", type=int, default=4096)
    ap.add_argument("--no-prefix-caching", action="store_true")
    ap.add_argument("--trace", default=None, help="also write all engine events to this JSONL file")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"loading {args.model} on {device} ({args.dtype}) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, device=device, dtype=getattr(torch, args.dtype))
    engine = LLMEngine(
        model, max_num_seqs=args.max_num_seqs, max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens, kv_cache="paged", kv_cache_gb=args.kv_cache_gb,
        block_size=args.block_size, enable_prefix_caching=not args.no_prefix_caching,
        event_log=EventLog(args.trace) if args.trace else None,
    )
    name = args.served_model_name or Path(args.model).name
    print(f"KV cache: {engine.cache.num_blocks} blocks x {args.block_size} tokens "
          f"({engine.cache.nbytes / 1e9:.2f} GB), serving as {name!r} on http://{args.host}:{args.port}/v1")
    uvicorn.run(create_app(AsyncEngine(engine), tokenizer, name), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
