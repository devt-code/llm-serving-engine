"""Token sampling: greedy, temperature, top-p (nucleus).

Batched from the start so Phase 2 can reuse it unchanged.
"""

import torch


def sample(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """logits: [B, vocab] -> next token ids: [B] (long)

    - temperature == 0  -> greedy (argmax); top_p is ignored.
    - temperature > 0   -> softmax(logits / temperature). Lower temperature sharpens the
                           distribution, higher flattens it.
    - top_p < 1         -> nucleus sampling: keep the smallest set of most likely tokens whose
                           total probability reaches top_p, renormalize, sample from those.
    """
    if temperature == 0.0:
        return logits.argmax(dim=-1)

    probs = torch.softmax(logits.float() / temperature, dim=-1)  # float32 even for bf16 logits

    if top_p < 1.0:
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        # Probability mass of all tokens strictly more likely than this one.
        mass_before = sorted_probs.cumsum(dim=-1) - sorted_probs
        # Drop a token once the tokens before it already reach top_p. The top token always
        # has mass_before == 0, so it is always kept.
        sorted_probs[mass_before >= top_p] = 0.0
        # Put the filtered probabilities back in vocab order and renormalize.
        probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True)

    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
