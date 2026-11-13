"""Unit tests for engine/sampler.py. No model needed: pytest tests/test_sampler.py"""

import torch

from engine.sampler import sample


def test_greedy_is_argmax():
    logits = torch.randn(4, 100, generator=torch.Generator().manual_seed(0))
    out = sample(logits, temperature=0.0)
    assert out.shape == (4,)
    assert out.dtype == torch.long
    assert torch.equal(out, logits.argmax(-1))


def test_greedy_ignores_top_p():
    logits = torch.randn(3, 50, generator=torch.Generator().manual_seed(1))
    assert torch.equal(sample(logits, temperature=0.0, top_p=0.1), logits.argmax(-1))


def test_tiny_top_p_keeps_only_top_token():
    logits = torch.randn(8, 1000, generator=torch.Generator().manual_seed(2))
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        assert torch.equal(sample(logits, temperature=1.0, top_p=1e-6, generator=g), logits.argmax(-1))


def test_top_p_never_samples_outside_nucleus():
    # probs: 0.5, 0.3, 0.15, 0.05 -> with top_p=0.8 the nucleus is {0, 1}
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    logits = probs.log()
    g = torch.Generator().manual_seed(0)
    seen = {int(sample(logits, temperature=1.0, top_p=0.8, generator=g)) for _ in range(500)}
    assert seen == {0, 1}


def test_temperature_sampling_matches_distribution():
    probs = torch.tensor([0.6, 0.3, 0.1])
    logits = probs.log().expand(20000, 3).contiguous()
    g = torch.Generator().manual_seed(0)
    out = sample(logits, temperature=1.0, generator=g)
    freq = torch.bincount(out, minlength=3).float() / out.numel()
    assert torch.allclose(freq, probs, atol=0.02), freq


def test_low_temperature_sharpens():
    probs = torch.tensor([0.4, 0.35, 0.25])
    logits = probs.log().expand(5000, 3).contiguous()
    g = torch.Generator().manual_seed(0)
    out = sample(logits, temperature=0.1, generator=g)
    assert (out == 0).float().mean() > 0.75


def test_seed_is_reproducible():
    logits = torch.randn(16, 500, generator=torch.Generator().manual_seed(3))
    a = sample(logits, temperature=0.8, top_p=0.9, generator=torch.Generator().manual_seed(42))
    b = sample(logits, temperature=0.8, top_p=0.9, generator=torch.Generator().manual_seed(42))
    assert torch.equal(a, b)


def test_bf16_logits():
    logits = torch.randn(2, 300, generator=torch.Generator().manual_seed(4)).to(torch.bfloat16)
    out = sample(logits, temperature=0.7, top_p=0.95, generator=torch.Generator().manual_seed(0))
    assert out.shape == (2,) and out.dtype == torch.long
