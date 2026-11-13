import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / "models" / "qwen3-0.6b"


def pytest_collection_modifyitems(config, items):
    if not (MODEL_DIR / "config.json").exists():
        skip = pytest.mark.skip(reason=f"model not downloaded to {MODEL_DIR}")
        for item in items:
            if "needs_model" in item.keywords:
                item.add_marker(skip)


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_model: requires the Qwen3-0.6B checkpoint")
    config.addinivalue_line("markers", "mps: requires an Apple GPU (MPS)")


@pytest.fixture(scope="session")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_DIR)


@pytest.fixture(scope="session")
def hf_model():
    """Reference model: HF transformers, float32, CPU, eager attention."""
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32, attn_implementation="eager")
    return m.eval()


@pytest.fixture(scope="session")
def our_model():
    """Our model, float32, CPU."""
    from engine.loader import load_model

    return load_model(MODEL_DIR, device="cpu", dtype=torch.float32)
