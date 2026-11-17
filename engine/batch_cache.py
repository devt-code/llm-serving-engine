"""KV cache for many requests at once (Phase 2).

Storage: `SlotKVCache` preallocates, per layer, K and V tensors of shape
    [num_slots, max_len, num_kv_heads, head_dim].
Each running request owns one slot (a row) for its whole lifetime, so every request reserves
max_len tokens of KV memory even if it only uses 40. This is how serving systems worked before
vLLM, and it's the waste the paged cache (Phase 3) removes. Memory for Qwen3-0.6B:
    num_slots * max_len * 112 KB  ->  64 slots * 512 tokens = 3.7 GB in bf16.

Each forward pass gets a small "batch" object describing which tokens belong to which slot.
It implements the attention-backend protocol from engine/model.py (`attend` and `advance`):
  - PrefillBatch: several fresh prompts concatenated into one flat sequence.
  - DecodeBatch:  one new token for each of B running requests.

`SlotManager` is the memory manager the scheduler talks to (same interface as the paged
`BlockManager` in engine/kv_cache.py): can_admit / allocate / ensure_capacity / free, plus
building the batch objects.
"""

import heapq

import torch
import torch.nn.functional as F

from engine.config import Qwen3Config


def packed_prefill_layout(seq_lens: list[int], device: torch.device) -> dict[str, torch.Tensor]:
    """Index tensors for packing several sequences into one flat token list.

    Returns:
      seq_idx        [T]    which sequence each token belongs to
      positions      [T]    position of each token inside its own sequence (0,1,2, 0,1, ...)
      mask           [T, T] block-diagonal causal mask (see PrefillBatch docstring)
      last_token_idx [B]    row of each sequence's last token
    """
    lens = torch.tensor(seq_lens, device=device)
    seq_idx = torch.repeat_interleave(torch.arange(len(seq_lens), device=device), lens)
    starts = torch.cumsum(lens, 0) - lens
    positions = torch.arange(int(lens.sum()), device=device) - starts[seq_idx]
    same_seq = seq_idx[:, None] == seq_idx[None, :]
    causal = positions[:, None] >= positions[None, :]
    return {
        "seq_idx": seq_idx,
        "positions": positions,
        "mask": same_seq & causal,  # built once, shared by all 28 layers
        "last_token_idx": torch.cumsum(lens, 0) - 1,
    }


class SlotKVCache:
    def __init__(self, config: Qwen3Config, num_slots: int, max_len: int, device: torch.device, dtype: torch.dtype):
        self.num_slots = num_slots
        self.max_len = max_len
        self.device = device
        shape = (num_slots, max_len, config.num_key_value_heads, config.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.num_hidden_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.num_hidden_layers)]

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v)


class PrefillBatch:
    """Several new prompts packed into one flat sequence of T = sum(prompt_lens) tokens.

    Because there's no batch dimension, packing is just concatenation. Attention must stop
    tokens from one prompt seeing another prompt's tokens, so the mask is block-diagonal
    (one causal triangle per prompt):

        prompt A (3 tokens), prompt B (2 tokens):
                 A0 A1 A2 B0 B1
            A0 [ 1  .  .  .  . ]
            A1 [ 1  1  .  .  . ]
            A2 [ 1  1  1  .  . ]
            B0 [ .  .  .  1  . ]
            B1 [ .  .  .  1  1 ]

    These are fresh prompts with nothing cached yet, so attention only looks at this step's
    own k/v. It also writes them into each request's slot for later decode steps.
    """

    def __init__(self, cache: SlotKVCache, slot_ids: list[int], prompt_lens: list[int]):
        self.cache = cache
        layout = packed_prefill_layout(prompt_lens, cache.device)
        self.positions = layout["positions"]  # [T]
        self.mask = layout["mask"]  # [T, T]
        # Row of each prompt's last token: the only rows we need logits for.
        self.last_token_idx = layout["last_token_idx"]  # [B]
        self.slot_of_token = torch.tensor(slot_ids, device=cache.device)[layout["seq_idx"]]  # [T]

    def attend(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
        # Write every token's k/v into (its slot, its position).
        self.cache.k[layer_idx][self.slot_of_token, self.positions] = k
        self.cache.v[layer_idx][self.slot_of_token, self.positions] = v
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
            attn_mask=self.mask, scale=scale, enable_gqa=True,
        )
        return out.transpose(0, 1)

    def advance(self, num_tokens: int) -> None:
        pass  # per-request lengths are tracked by the engine


class DecodeBatch:
    """One new token for each of B running requests, each with a different context length.

    Request b lives in slot slot_ids[b] and, counting this new token, has context_lens[b]
    tokens. Its new k/v is written at position context_lens[b] - 1.

    To attend without copying anyone's cache, attention runs over the slot array itself:
      keys   = cache[:n, :L]      a VIEW, n = highest used slot + 1, L = longest context
      query  = q scattered into a [n, Hq, 1, D] tensor at each request's slot
      mask   = key position < that slot's context length
    Slots are handed out lowest-first, so n stays close to the number of running requests.
    Unused slots in [0, n) compute garbage that's thrown away; that's the price of no copy.
    """

    def __init__(self, cache: SlotKVCache, slot_ids: list[int], context_lens: list[int]):
        dev = cache.device
        self.cache = cache
        self.slot_ids = torch.tensor(slot_ids, device=dev)  # [B]
        lens = torch.tensor(context_lens, device=dev)
        self.write_pos = lens - 1  # [B]
        self.n = max(slot_ids) + 1
        self.max_len = max(context_lens)
        # Unused slots get length 1 so their (discarded) softmax row isn't empty, which would give NaN.
        lens_per_slot = torch.ones(self.n, dtype=torch.long, device=dev)
        lens_per_slot[self.slot_ids] = lens
        key_pos = torch.arange(self.max_len, device=dev)
        self.mask = (key_pos[None, :] < lens_per_slot[:, None])[:, None, None, :]  # [n, 1, 1, L]

    def attend(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
        """q [B, Hq, D], k/v [B, Hkv, D] -> [B, Hq, D]"""
        kc, vc = self.cache.k[layer_idx], self.cache.v[layer_idx]
        kc[self.slot_ids, self.write_pos] = k
        vc[self.slot_ids, self.write_pos] = v

        keys = kc[: self.n, : self.max_len].transpose(1, 2)  # [n, Hkv, L, D] (view)
        values = vc[: self.n, : self.max_len].transpose(1, 2)
        q_slots = q.new_zeros(self.n, q.shape[1], q.shape[2])
        q_slots[self.slot_ids] = q
        out = F.scaled_dot_product_attention(
            q_slots[:, :, None, :],  # [n, Hq, 1, D]
            keys, values, attn_mask=self.mask, scale=scale, enable_gqa=True,
        )  # [n, Hq, 1, D]
        return out[self.slot_ids, :, 0, :]  # back to request order: [B, Hq, D]

    def advance(self, num_tokens: int) -> None:
        pass


class SlotManager:
    """Memory manager for SlotKVCache: each running request owns one whole slot."""

    kind = "slot"

    def __init__(self, cache: SlotKVCache):
        self.cache = cache
        self.free_slots = list(range(cache.num_slots))  # min-heap: lowest free slot first

    @property
    def max_request_tokens(self) -> int:
        return self.cache.max_len

    def can_admit(self, req, num_tokens: int) -> bool:
        return bool(self.free_slots) and num_tokens <= self.cache.max_len

    def allocate(self, req, num_tokens: int) -> None:
        req.slot = heapq.heappop(self.free_slots)

    def ensure_capacity(self, req, num_tokens: int) -> bool:
        # The whole max_len was reserved at admission, so there's always room.
        return num_tokens <= self.cache.max_len

    def free(self, req) -> None:
        heapq.heappush(self.free_slots, req.slot)
        req.slot = None

    def prefill_batch(self, reqs, seq_lens: list[int]) -> PrefillBatch:
        return PrefillBatch(self.cache, [r.slot for r in reqs], seq_lens)

    def decode_batch(self, reqs, context_lens: list[int]) -> DecodeBatch:
        return DecodeBatch(self.cache, [r.slot for r in reqs], context_lens)

    def usage(self) -> dict:
        total = self.cache.num_slots
        return {"kv_unit": "slot", "kv_used": total - len(self.free_slots), "kv_total": total}
