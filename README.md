# mini-serve

A small LLM serving engine built from scratch in PyTorch: continuous batching, paged KV cache,
prefix caching, and an OpenAI-compatible streaming API, serving Qwen3-0.6B. It also has a live
dashboard and is benchmarked against vLLM.

## Setup

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', local_dir='models/qwen3-0.6b')"
python bench/hf_baseline.py
```

## Results

Numbers are added only after they're measured. See `bench/results/`.
