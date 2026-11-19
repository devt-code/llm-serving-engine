# LLM Serving Engine

An LLM serving engine built from scratch in PyTorch: continuous batching, paged KV cache,
prefix caching, and an OpenAI-compatible streaming API, serving Qwen3-0.6B. It also has a live
dashboard and is benchmarked against vLLM.

## Setup

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', local_dir='models/qwen3-0.6b')"
```

## Run the server

```bash
python -m server.api            # OpenAI-compatible API on http://localhost:8000/v1
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
for chunk in client.chat.completions.create(
    model="qwen3-0.6b", messages=[{"role": "user", "content": "Hello!"}], stream=True
):
    print(chunk.choices[0].delta.content or "", end="")
```

Live engine events (requests, steps, KV blocks) stream as JSON from `ws://localhost:8000/v1/events`.

## Tests

```bash
pytest          # compares every component and the batched engine against Hugging Face
```

## Results

Numbers are added only after they're measured. See `bench/results/`.
