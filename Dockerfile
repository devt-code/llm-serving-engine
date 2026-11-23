# The engine, its OpenAI-compatible API and the dashboard in one image, running on CPU.
#
#   docker build -t llm-serving-engine .
#   docker run --rm -p 7860:7860 llm-serving-engine        # then open http://localhost:7860
#
# The live demo runs the same server on Modal instead (see deploy/modal/app.py).

FROM node:22-slim AS dashboard
WORKDIR /dashboard
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY dashboard/ ./
# VITE_SAME_ORIGIN: the dashboard talks to the server that serves it and opens in Live mode.
RUN VITE_SAME_ORIGIN=1 npm run build

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HUB_DISABLE_TELEMETRY=1
COPY requirements.txt /tmp/requirements.txt
# CPU-only PyTorch (the default Linux wheel bundles CUDA and is several GB), then only what serving
# needs, at the versions pinned in requirements.txt.
RUN pip install "torch==$(sed -n 's/^torch==//p' /tmp/requirements.txt)" --index-url https://download.pytorch.org/whl/cpu \
 && grep -E '^(fastapi|uvicorn|websockets|safetensors|transformers|huggingface-hub)==' /tmp/requirements.txt > /tmp/serve.txt \
 && pip install -r /tmp/serve.txt

# An unprivileged user (uid 1000, which hosts such as Hugging Face Spaces expect).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
WORKDIR /home/user/app
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', local_dir='models/qwen3-0.6b')"
COPY --chown=user engine/ engine/
COPY --chown=user server/ server/
COPY --chown=user --from=dashboard /dashboard/dist dashboard/dist

EXPOSE 7860
# fp32 on CPU; limits sized for a 2-CPU machine shared by everyone who opens the page.
CMD ["python", "-m", "server.api", "--host", "0.0.0.0", "--port", "7860", "--device", "cpu", "--dtype", "float32", \
     "--dashboard", "dashboard/dist", "--max-num-seqs", "64", "--max-tokens-cap", "512", "--max-waiting", "64", \
     "--demo-max-count", "50"]
