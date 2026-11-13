"""Load Hugging Face safetensors weights into our Qwen3 modules.

Our module names mirror the checkpoint's tensor names (e.g.
`model.layers.0.self_attn.q_proj.weight`), so loading is a direct name -> parameter copy.
"""

from pathlib import Path

import torch
from safetensors import safe_open

from engine.config import Qwen3Config
from engine.model import Qwen3ForCausalLM


def pick_device(name: str = "auto") -> torch.device:
    """'auto' picks cuda > mps > cpu. Keeps all engine code device-agnostic."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_weights(model: Qwen3ForCausalLM, model_dir: str | Path) -> None:
    """Copy every checkpoint tensor into the matching parameter. Strict: errors on any mismatch."""
    params = dict(model.named_parameters())
    tied = model.config.tie_word_embeddings
    loaded = set()

    files = sorted(Path(model_dir).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {model_dir}")

    for file in files:
        with safe_open(str(file), framework="pt") as f:
            for name in f.keys():
                if tied and name == "lm_head.weight":
                    continue  # same tensor as embed_tokens; shared below
                if name not in params:
                    raise KeyError(f"checkpoint tensor {name!r} has no matching parameter")
                tensor = f.get_tensor(name)
                param = params[name]
                if tensor.shape != param.shape:
                    raise ValueError(f"{name}: checkpoint {tuple(tensor.shape)} vs model {tuple(param.shape)}")
                with torch.no_grad():
                    param.copy_(tensor.to(param.dtype))
                loaded.add(name)

    if tied:
        model.lm_head.weight = model.model.embed_tokens.weight
        loaded.add("lm_head.weight")

    missing = set(params) - loaded
    if missing:
        raise KeyError(f"parameters not found in checkpoint: {sorted(missing)[:5]} ...")


def load_model(
    model_dir: str | Path = "models/qwen3-0.6b",
    device: str | torch.device = "auto",
    dtype: torch.dtype = torch.bfloat16,
) -> Qwen3ForCausalLM:
    config = Qwen3Config.from_pretrained(model_dir)
    device = pick_device(device) if isinstance(device, str) else device
    with torch.device(device):
        model = Qwen3ForCausalLM(config)
    model = model.to(dtype)
    load_weights(model, model_dir)
    return model.eval()
