"""Compare our Qwen3 against Hugging Face transformers, one component at a time.

Run in this order while implementing (each needs the previous ones):
    pytest tests/test_model.py -x -k rmsnorm
    pytest tests/test_model.py -x -k rope
    pytest tests/test_model.py -x -k mlp
    pytest tests/test_model.py -x -k attention
    pytest tests/test_model.py -x -k decoder_layer
    pytest tests/test_model.py -x -k logits
    pytest tests/test_model.py -x -k kv_cache
    pytest tests/test_model.py -x -k greedy
    pytest tests/test_model.py -x -k mps          # bf16 on the Apple GPU

Reference = HF in float32 on CPU with eager attention. Component tests use layer 0's real
weights. Tolerances are tight in fp32; any real bug shows up as an error orders of magnitude larger.
"""

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from tests.prompts import PROMPTS

pytestmark = pytest.mark.needs_model

T = 11  # odd sequence length on purpose: catches shape assumptions
GEN = torch.Generator().manual_seed(0)


def rand(*shape):
    return torch.randn(*shape, generator=GEN)


def causal_mask(t: int) -> torch.Tensor:
    """Additive mask in HF's layout [1, 1, T, T]."""
    m = torch.full((t, t), float("-inf")).triu(1)
    return m[None, None]


def hf_rope(hf_model, positions: torch.Tensor):
    """HF cos/sin for positions [T] -> each [1, T, head_dim]."""
    dummy = torch.zeros(1, dtype=torch.float32)
    return hf_model.model.rotary_emb(dummy, positions[None])


def assert_close(ours, ref, atol, rtol=0.0, what=""):
    if ours.shape != ref.shape:
        pytest.fail(f"{what}: shape {tuple(ours.shape)} != expected {tuple(ref.shape)}", pytrace=False)
    if not torch.allclose(ours.float(), ref.float(), atol=atol, rtol=rtol):
        err = (ours.float() - ref.float()).abs().max().item()
        pytest.fail(f"{what}: max abs error {err:.3e} (atol={atol}, rtol={rtol})", pytrace=False)


@torch.inference_mode()
def test_weights_loaded(our_model, hf_model):
    ours = dict(our_model.named_parameters())
    for name, p in hf_model.named_parameters():
        assert torch.equal(ours[name], p), name
    assert our_model.lm_head.weight.data_ptr() == our_model.model.embed_tokens.weight.data_ptr(), "lm_head not tied"


# ---------------------------------------------------------------- RMSNorm


@torch.inference_mode()
def test_rmsnorm(our_model, hf_model):
    x = rand(T, 1024) * 3
    assert_close(our_model.model.norm(x), hf_model.model.norm(x), 1e-5, what="final norm")
    # q_norm normalizes the last dim (head_dim=128) of a [T, heads, 128] tensor
    q = rand(T, 16, 128)
    ours = our_model.model.layers[0].self_attn.q_norm(q)
    ref = hf_model.model.layers[0].self_attn.q_norm(q)
    assert_close(ours, ref, 1e-5, what="q_norm")


@torch.inference_mode()
def test_rmsnorm_bf16_order(our_model, hf_model):
    """In bf16, normalize in fp32 -> cast -> multiply by weight (HF's order)."""
    from copy import deepcopy

    ours_norm = deepcopy(our_model.model.norm).to(torch.bfloat16)
    ref_norm = deepcopy(hf_model.model.norm).to(torch.bfloat16)
    x = (rand(T, 1024) * 3).to(torch.bfloat16)
    out = ours_norm(x)
    assert out.dtype == torch.bfloat16, f"dtype {out.dtype}"
    assert torch.equal(out, ref_norm(x)), "bf16 RMSNorm differs; check the order of cast vs weight multiply"


# ---------------------------------------------------------------- RoPE


@torch.inference_mode()
def test_rope_cos_sin(our_model, hf_model):
    positions = torch.tensor([0, 1, 2, 7, 100, 1023, 4095, 32000])
    cos, sin = our_model.model.rotary_emb(positions)
    ref_cos, ref_sin = hf_rope(hf_model, positions)
    assert_close(cos.float(), ref_cos[0], 1e-4, what="cos")
    assert_close(sin.float(), ref_sin[0], 1e-4, what="sin")


@torch.inference_mode()
def test_rope_apply(our_model, hf_model):
    from engine.model import apply_rotary

    positions = torch.arange(T) * 13
    cos, sin = hf_rope(hf_model, positions)  # [1, T, D]
    q = rand(T, 16, 128)  # ours: [T, H, D]
    k = rand(T, 8, 128)
    # HF layout is [B, H, T, D]
    ref_q, ref_k = apply_rotary_pos_emb(q.transpose(0, 1)[None], k.transpose(0, 1)[None], cos, sin)
    assert_close(apply_rotary(q, cos[0], sin[0]), ref_q[0].transpose(0, 1), 1e-5, what="rotated q")
    assert_close(apply_rotary(k, cos[0], sin[0]), ref_k[0].transpose(0, 1), 1e-5, what="rotated k")


# ---------------------------------------------------------------- MLP


@torch.inference_mode()
def test_mlp(our_model, hf_model):
    x = rand(T, 1024)
    ours = our_model.model.layers[0].mlp(x)
    ref = hf_model.model.layers[0].mlp(x[None])[0]
    assert_close(ours, ref, 1e-4, what="mlp")


# ---------------------------------------------------------------- Attention


@torch.inference_mode()
def test_attention(our_model, hf_model):
    x = rand(T, 1024)
    positions = torch.arange(T)
    cos, sin = our_model.model.rotary_emb(positions)
    ours = our_model.model.layers[0].self_attn(x, cos, sin)

    ref_cos, ref_sin = hf_rope(hf_model, positions)
    ref, _ = hf_model.model.layers[0].self_attn(
        hidden_states=x[None], position_embeddings=(ref_cos, ref_sin), attention_mask=causal_mask(T)
    )
    assert_close(ours, ref[0], 1e-4, what="attention (is it causal? is GQA mapping right?)")


@torch.inference_mode()
def test_decoder_layer(our_model, hf_model):
    x = rand(T, 1024)
    positions = torch.arange(T)
    cos, sin = our_model.model.rotary_emb(positions)
    ref_cos, ref_sin = hf_rope(hf_model, positions)
    for i in (0, 13, 27):
        ours = our_model.model.layers[i](x, cos, sin)
        ref = hf_model.model.layers[i](x[None], attention_mask=causal_mask(T), position_embeddings=(ref_cos, ref_sin))
        if isinstance(ref, tuple):
            ref = ref[0]
        assert_close(ours, ref[0], 1e-4, rtol=1e-5, what=f"decoder layer {i}")


# ---------------------------------------------------------------- Full model


@torch.inference_mode()
@pytest.mark.parametrize("prompt", PROMPTS[:4])
def test_logits(our_model, hf_model, tokenizer, prompt):
    ids = torch.tensor(tokenizer(prompt)["input_ids"])
    ours = our_model(ids, torch.arange(len(ids)))
    ref = hf_model(ids[None]).logits[0]
    assert_close(ours, ref, atol=2e-3, rtol=1e-4, what="logits")
    assert torch.equal(ours.argmax(-1), ref.argmax(-1)), "argmax token differs"


# ---------------------------------------------------------------- KV cache


@torch.inference_mode()
def test_kv_cache_decode_matches_full_forward(our_model, tokenizer):
    """Prefill part of a sequence, then decode the rest token by token with the cache.
    Each step's logits must match a single full forward pass without a cache."""
    from engine.model import KVCache

    ids = torch.tensor(tokenizer(PROMPTS[1] + " " + PROMPTS[7])["input_ids"])
    n, split = len(ids), 9
    full = our_model(ids, torch.arange(n))

    cache = KVCache(our_model.config, max_len=n, device=ids.device, dtype=torch.float32)
    step = our_model(ids[:split], torch.arange(split), cache)
    assert cache.seq_len == split, f"seq_len after prefill is {cache.seq_len}, expected {split} (advance once per forward)"
    assert_close(step, full[:split], 1e-3, what="prefill logits with cache")
    for i in range(split, n):
        step = our_model(ids[i : i + 1], torch.tensor([i]), cache)
        assert_close(step[0], full[i], 2e-3, what=f"decode step at position {i}")
    assert cache.seq_len == n


@torch.inference_mode()
def test_kv_cache_chunked_prefill(our_model, tokenizer):
    """Prefill in two chunks (T > 1 with existing cache). The causal mask must be offset."""
    from engine.model import KVCache

    ids = torch.tensor(tokenizer(PROMPTS[11] + " " + PROMPTS[15])["input_ids"])
    n, split = len(ids), 6
    full = our_model(ids, torch.arange(n))
    cache = KVCache(our_model.config, max_len=n, device=ids.device, dtype=torch.float32)
    our_model(ids[:split], torch.arange(split), cache)
    second = our_model(ids[split:], torch.arange(split, n), cache)
    assert_close(second, full[split:], 2e-3, what="second prefill chunk (mask offset when S > T?)")


# ---------------------------------------------------------------- Greedy generation


@torch.inference_mode()
@pytest.mark.parametrize("idx", range(len(PROMPTS)))
def test_greedy_matches_hf(our_model, hf_model, tokenizer, idx):
    from engine.generate import generate

    prompt = PROMPTS[idx]
    ids = tokenizer(prompt)["input_ids"]
    ours = generate(our_model, ids, max_new_tokens=24)
    ref = hf_model.generate(
        torch.tensor([ids]), max_new_tokens=24, do_sample=False, temperature=None, top_p=None, top_k=None
    )[0, len(ids) :].tolist()
    assert ours == ref, f"\nprompt: {prompt!r}\nours: {tokenizer.decode(ours)!r}\nhf:   {tokenizer.decode(ref)!r}"


# ---------------------------------------------------------------- bf16 on MPS


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
@torch.inference_mode()
def test_mps_bf16_logits(tokenizer):
    """bf16 on the Apple GPU vs HF bf16 on the same device. Looser tolerance: bf16 has ~3 significant digits."""
    from transformers import AutoModelForCausalLM

    from engine.loader import load_model
    from tests.conftest import MODEL_DIR

    ours = load_model(MODEL_DIR, device="mps", dtype=torch.bfloat16)
    ref = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16).to("mps").eval()
    agree, total = 0, 0
    for prompt in PROMPTS[:8]:
        ids = torch.tensor(tokenizer(prompt)["input_ids"], device="mps")
        a = ours(ids, torch.arange(len(ids), device="mps")).float()
        b = ref(ids[None]).logits[0].float()
        err = (a - b).abs().max().item()
        assert err < 1.0, f"bf16 logits max abs error {err:.3f} on {prompt!r}"
        agree += (a.argmax(-1) == b.argmax(-1)).sum().item()
        total += len(ids)
    assert agree / total >= 0.95, f"top-1 agreement only {agree}/{total}"
    del ours, ref
    torch.mps.empty_cache()
