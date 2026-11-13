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

    - temperature == 0  -> greedy (argmax). Ignore top_p.
    - temperature > 0   -> softmax(logits / temperature), then:
    - top_p < 1         -> keep the smallest set of highest-probability tokens whose total
                           probability is >= top_p (always keep at least the top token),
                           zero out the rest, renormalize, then sample.
    - Use `generator` for sampling (torch.multinomial(..., generator=generator)) so results
      are reproducible with a seed.

    Do the softmax in float32 even if logits are bf16.
    """
    raise NotImplementedError("TODO(you): sample")
