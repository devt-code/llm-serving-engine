"""CUDA path check (Phase 3): fp16 on an NVIDIA GPU. Skipped when there's no CUDA device.

Run on Kaggle/Colab with notebooks/cuda_check.ipynb, or directly:
    pytest tests/test_cuda.py -v
"""

import pytest
import torch

from tests.prompts import PROMPTS

pytestmark = [
    pytest.mark.needs_model,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device"),
]

DTYPE = torch.float16  # T4 (Kaggle/Colab free tier) has no fast bf16


@pytest.fixture(scope="module")
def cuda_models():
    from transformers import AutoModelForCausalLM

    from engine.loader import load_model
    from tests.conftest import MODEL_DIR

    ours = load_model(MODEL_DIR, device="cuda", dtype=DTYPE)
    ref = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=DTYPE).to("cuda").eval()
    return ours, ref


@torch.inference_mode()
def test_cuda_fp16_logits(cuda_models, tokenizer):
    ours, ref = cuda_models
    agree = total = 0
    for prompt in PROMPTS[:8]:
        ids = torch.tensor(tokenizer(prompt)["input_ids"], device="cuda")
        a = ours(ids, torch.arange(len(ids), device="cuda")).float()
        b = ref(ids[None]).logits[0].float()
        assert torch.isfinite(a).all(), "non-finite logits in fp16 (overflow?)"
        err = (a - b).abs().max().item()
        assert err < 1.0, f"fp16 logits max abs error {err:.3f} on {prompt!r}"
        agree += (a.argmax(-1) == b.argmax(-1)).sum().item()
        total += len(ids)
    assert agree / total >= 0.95, f"top-1 agreement {agree}/{total}"


@torch.inference_mode()
def test_cuda_paged_engine_with_preemption(cuda_models, tokenizer):
    """Paged + continuous batching on CUDA, with a pool small enough to force preemption.
    Compared against our own single-request generate() on the same device and dtype."""
    from engine.engine import LLMEngine
    from engine.events import EventLog
    from engine.generate import generate
    from engine.request import SamplingParams

    ours, _ = cuda_models
    prompts = [tokenizer(p)["input_ids"] for p in PROMPTS]
    single = [generate(ours, p, 32, ignore_eos=True) for p in prompts]
    log = EventLog(keep_in_memory=True)
    eng = LLMEngine(ours, max_num_seqs=20, max_model_len=64, max_num_batched_tokens=512,
                    kv_cache="paged", block_size=16, num_kv_blocks=40, event_log=log)
    batched = eng.generate(prompts, SamplingParams(max_new_tokens=32, ignore_eos=True))
    assert any(e["type"] == "request_preempted" for e in log.events)
    assert eng.memory.num_free == 40
    # fp16 batch invariance: near-ties can flip (see Phase 2 dev log), so require most, not all.
    exact = sum(a == b for a, b in zip(single, batched))
    assert exact >= 0.6 * len(prompts), f"only {exact}/{len(prompts)} identical"
