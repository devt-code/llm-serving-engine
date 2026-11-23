"""Deploy the engine, its API and the dashboard to Modal (https://modal.com).

The app runs only while someone is using it. The first request starts a container (up to a
minute), dashboard viewers are disconnected after 2 quiet minutes, and the container shuts down a
minute after its last request. Without a payment method on file, Modal's free Starter plan includes
$1 of compute a month and stops the app once that is used up (until the next month), so it never
charges anything.

On CPU (the default) $1 buys about 7 hours of serving a month. On a GPU the engine is far faster,
but an L4 costs about $0.80 an hour, so set GPU = "L4" only on a workspace with more credits.

    pip install modal && modal token new      # once
    deploy/modal/deploy.sh                    # builds the dashboard, deploys, prints the URL
"""

import sys
from pathlib import Path

import modal

GPU = None  # e.g. "L4"; None runs on CPU
CPU_CORES = 2  # the container sees this many CPUs, and Modal sets OMP_NUM_THREADS to match

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2] if len(HERE.parents) > 2 else HERE.parent
sys.path.insert(0, str(ROOT))  # lets Modal find the engine/ and server/ packages to upload

# The serving dependencies, at the versions pinned in requirements.txt (read only where it exists,
# i.e. on the machine that deploys, not inside the container).
SERVE = {"fastapi", "uvicorn", "websockets", "safetensors", "transformers", "huggingface-hub"}
requirements = ROOT / "requirements.txt"
lines = [line.strip() for line in requirements.read_text().splitlines()] if requirements.exists() else []
TORCH = next((line for line in lines if line.startswith("torch==")), "torch")
PINS = [line for line in lines if line.split("==")[0] in SERVE]

image = (
    modal.Image.debian_slim(python_version="3.11")
    # The CPU build of PyTorch is a fraction of the CUDA build's size, so containers start sooner.
    .pip_install(TORCH, index_url=None if GPU else "https://download.pytorch.org/whl/cpu")
    .pip_install(*PINS)
    .run_commands(
        "python -c \"from huggingface_hub import snapshot_download; "
        "snapshot_download('Qwen/Qwen3-0.6B', local_dir='/models/qwen3-0.6b')\""
    )
    .add_local_dir(ROOT / "dashboard" / "dist-server", "/app/dashboard")
    .add_local_python_source("engine", "server")
)
app = modal.App("llm-serving-engine", image=image)

ARGS = ["--model", "/models/qwen3-0.6b", "--served-model-name", "qwen3-0.6b",
        "--dashboard", "/app/dashboard", "--ws-idle-timeout", "120"]
if GPU:
    ARGS += ["--dtype", "bfloat16", "--kv-cache-gb", "6", "--max-num-seqs", "256",
             "--max-tokens-cap", "1024", "--max-waiting", "256", "--demo-max-count", "100"]
else:  # the same settings as the Docker image
    ARGS += ["--device", "cpu", "--dtype", "float32", "--max-num-seqs", "64",
             "--max-tokens-cap", "512", "--max-waiting", "64", "--demo-max-count", "50"]


# One container at most, so everyone shares one engine (and its dashboard). Hard CPU and memory
# limits fix the cost per hour, since Modal bills the larger of what is reserved and what is used.
@app.function(gpu=GPU, cpu=(CPU_CORES, CPU_CORES), memory=(6144, 8192),
              max_containers=1, scaledown_window=60, timeout=3600)
@modal.concurrent(max_inputs=500)
@modal.asgi_app(label="llm-serving-engine")
def web():
    from server.api import build_app, build_parser

    return build_app(build_parser().parse_args(ARGS))
