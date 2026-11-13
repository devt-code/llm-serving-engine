"""Single-request generation loop: prefill once, then decode one token at a time.

This is the Phase 1 "engine". Phase 2 replaces it with a scheduler that runs many requests.
"""

import torch

from engine.model import KVCache, Qwen3ForCausalLM
from engine.sampler import sample


@torch.inference_mode()
def generate(
    model: Qwen3ForCausalLM,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int | None = None,
    ignore_eos: bool = False,
) -> list[int]:
    """Returns only the newly generated token ids (excludes the prompt)."""
    device = model.lm_head.weight.device
    dtype = model.lm_head.weight.dtype
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    cache = KVCache(model.config, len(prompt_ids) + max_new_tokens, device, dtype)
    eos = model.config.eos_token_id

    # Prefill: the whole prompt in one forward pass.
    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    positions = torch.arange(len(prompt_ids), device=device)
    logits = model(input_ids, positions, cache)[-1:]

    out: list[int] = []
    for _ in range(max_new_tokens):
        next_id = sample(logits, temperature, top_p, generator)  # [1]
        token = int(next_id.item())
        out.append(token)
        if token == eos and not ignore_eos:
            break
        # Decode: one new token, at the next position.
        positions = torch.tensor([cache.seq_len], device=device)
        logits = model(next_id, positions, cache)
    return out
