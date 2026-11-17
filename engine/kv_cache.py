"""Paged KV cache (Phase 3): the core idea of vLLM's PagedAttention (SOSP '23).

The problem with the Phase 2 slot cache: every request reserves max_model_len tokens of KV
memory up front, even though most requests use a fraction of it. Memory, not compute, then
caps how many requests can run at once.

The fix borrows virtual memory from operating systems:
  - KV memory is one shared pool of fixed-size BLOCKS (16 tokens each, all layers).
  - Each request has a BLOCK TABLE: logical block i of the request -> physical block id.
    Token at position p lives in physical block table[p // 16], offset p % 16.
  - Blocks are handed out only when a request actually grows into them, and returned to the
    free list the moment it finishes. Waste is at most 15 tokens per request (its last,
    partly filled block) instead of max_model_len - actual_len.
  - If the pool runs dry mid-decode, a running request is PREEMPTED: its blocks are freed
    and it goes back to the waiting queue, to be recomputed (re-prefilled) later.

Storage per layer: K and V of shape [num_blocks, block_size, num_kv_heads, head_dim].
One block for Qwen3-0.6B = 16 tokens * 112 KB = 1.75 MB (all 28 layers, K and V, bf16).

Attention reads K/V through the block tables with plain PyTorch indexing (a gather). vLLM
fuses that gather into the attention kernel itself (PagedAttention); doing it as a separate
copy costs extra memory traffic, which is measured in the Phase 3 dev log.
"""

import torch
import torch.nn.functional as F

from engine.batch_cache import packed_prefill_layout
from engine.config import Qwen3Config
from engine.events import EventLog


class PagedKVCache:
    def __init__(
        self, config: Qwen3Config, num_blocks: int, block_size: int, device: torch.device, dtype: torch.dtype
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.device = device
        shape = (num_blocks, block_size, config.num_key_value_heads, config.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.num_hidden_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.num_hidden_layers)]

    @staticmethod
    def bytes_per_block(config: Qwen3Config, block_size: int, dtype: torch.dtype) -> int:
        elem = torch.tensor([], dtype=dtype).element_size()
        return 2 * config.num_hidden_layers * block_size * config.num_key_value_heads * config.head_dim * elem

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v)

    def flat(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Views of one layer's pool as [num_blocks * block_size, Hkv, D], indexed by flat slot."""
        return (
            self.k[layer_idx].view(-1, self.num_kv_heads, self.head_dim),
            self.v[layer_idx].view(-1, self.num_kv_heads, self.head_dim),
        )


def _check_tables(block_tables: list[list[int]], seq_lens: list[int], block_size: int) -> None:
    """A table shorter than its sequence would silently write into block 0, which belongs to
    another request. The scheduler always allocates first; this guards against bugs."""
    for table, n in zip(block_tables, seq_lens):
        if len(table) * block_size < n:
            raise RuntimeError(f"block table covers {len(table) * block_size} tokens but sequence has {n}")


def _padded_tables(block_tables: list[list[int]], num_cols: int, device) -> torch.Tensor:
    """[B, num_cols] block-table tensor; unused entries point at block 0 (masked out later)."""
    rows = [t[:num_cols] + [0] * (num_cols - len(t[:num_cols])) for t in block_tables]
    return torch.tensor(rows, dtype=torch.long, device=device)


class PagedPrefillBatch:
    """Several sequences packed into one flat token list (same layout as Phase 2's PrefillBatch).

    Nothing is cached for these sequences yet (new prompts, or preempted requests being
    recomputed from scratch), so attention only looks at this step's own k/v. Each token's
    k/v is also written into its request's blocks:
        flat_slot = block_table[pos // block_size] * block_size + pos % block_size
    """

    def __init__(self, cache: PagedKVCache, block_tables: list[list[int]], seq_lens: list[int]):
        _check_tables(block_tables, seq_lens, cache.block_size)
        self.cache = cache
        layout = packed_prefill_layout(seq_lens, cache.device)
        self.positions = layout["positions"]
        self.mask = layout["mask"]
        self.last_token_idx = layout["last_token_idx"]
        bs = cache.block_size
        tables = _padded_tables(block_tables, max(map(len, block_tables)), cache.device)
        block_of_token = tables[layout["seq_idx"], self.positions // bs]
        self.slot_mapping = block_of_token * bs + self.positions % bs  # [T]

    def attend(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
        k_flat, v_flat = self.cache.flat(layer_idx)
        k_flat[self.slot_mapping] = k
        v_flat[self.slot_mapping] = v
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
            attn_mask=self.mask, scale=scale, enable_gqa=True,
        )
        return out.transpose(0, 1)

    def advance(self, num_tokens: int) -> None:
        pass


class PagedDecodeBatch:
    """One new token for each of B running requests.

    1. Write the new token's k/v at its flat slot.
    2. Gather each request's blocks into a dense [B, L, Hkv, D] tensor, with L = the longest
       context rounded up to whole blocks:  keys = pool[block_tables]   (a copy)
    3. Masked SDPA: key position < that request's context length.
    """

    def __init__(self, cache: PagedKVCache, block_tables: list[list[int]], context_lens: list[int]):
        _check_tables(block_tables, context_lens, cache.block_size)
        dev = cache.device
        bs = cache.block_size
        self.cache = cache
        lens = torch.tensor(context_lens, device=dev)
        num_cols = -(-max(context_lens) // bs)  # ceil
        self.tables = _padded_tables(block_tables, num_cols, dev)  # [B, num_cols]
        self.flat_tables = self.tables.flatten()  # shared by all 28 layers
        self.B = len(context_lens)
        self.L = num_cols * bs
        pos = lens - 1  # where the new token goes
        self.slot_mapping = self.tables[torch.arange(self.B, device=dev), pos // bs] * bs + pos % bs
        key_pos = torch.arange(self.L, device=dev)
        self.mask = (key_pos[None, :] < lens[:, None])[:, None, None, :]  # [B, 1, 1, L]

    def attend(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
        """q [B, Hq, D], k/v [B, Hkv, D] -> [B, Hq, D]"""
        k_flat, v_flat = self.cache.flat(layer_idx)
        k_flat[self.slot_mapping] = k
        v_flat[self.slot_mapping] = v

        Hkv, D = self.cache.num_kv_heads, self.cache.head_dim
        # pool[layer]: [num_blocks, bs, Hkv, D]. Gather each request's blocks in table order:
        # [B * num_cols, bs, Hkv, D] -> [B, L, Hkv, D]. index_select on a flat index is ~4x
        # faster than advanced indexing (pool[tables]) on MPS for this shape (measured:
        # 0.84 vs 3.29 ms per layer at B=32, L=288).
        keys = self.cache.k[layer_idx].index_select(0, self.flat_tables).view(self.B, self.L, Hkv, D).transpose(1, 2)
        values = self.cache.v[layer_idx].index_select(0, self.flat_tables).view(self.B, self.L, Hkv, D).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q[:, :, None, :], keys, values, attn_mask=self.mask, scale=scale, enable_gqa=True,
        )  # [B, Hq, 1, D]
        return out[:, :, 0, :]

    def advance(self, num_tokens: int) -> None:
        pass


class BlockManager:
    """Hands out and reclaims KV blocks. Same interface as SlotManager (engine/batch_cache.py).

    watermark: fraction of blocks kept free when ADMITTING new requests, so running requests
    have room to grow for a while and we don't preempt right after admitting.
    """

    kind = "paged"

    def __init__(self, cache: PagedKVCache, events: EventLog | None = None, watermark: float = 0.01):
        self.cache = cache
        self.block_size = cache.block_size
        self.free_blocks: list[int] = list(range(cache.num_blocks - 1, -1, -1))  # pop() gives block 0 first
        self.watermark_blocks = int(watermark * cache.num_blocks)
        self.events = events

    def blocks_needed(self, num_tokens: int) -> int:
        return -(-num_tokens // self.block_size)

    @property
    def max_request_tokens(self) -> int:
        return self.cache.num_blocks * self.block_size

    @property
    def num_free(self) -> int:
        return len(self.free_blocks)

    def can_admit(self, req, num_tokens: int) -> bool:
        return self.num_free - self.blocks_needed(num_tokens) >= self.watermark_blocks

    def allocate(self, req, num_tokens: int) -> None:
        n = self.blocks_needed(num_tokens)
        req.block_table = [self.free_blocks.pop() for _ in range(n)]
        self._emit("blocks_allocated", req, req.block_table)

    def ensure_capacity(self, req, num_tokens: int) -> bool:
        """Make sure req's table covers num_tokens tokens. False if the pool is out of blocks."""
        missing = self.blocks_needed(num_tokens) - len(req.block_table)
        if missing <= 0:
            return True
        if missing > self.num_free:
            return False
        new = [self.free_blocks.pop() for _ in range(missing)]
        req.block_table.extend(new)
        self._emit("blocks_allocated", req, new)
        return True

    def free(self, req) -> None:
        if req.block_table:
            self._emit("blocks_freed", req, req.block_table)
            self.free_blocks.extend(reversed(req.block_table))
        req.block_table = None

    def prefill_batch(self, reqs, seq_lens: list[int]) -> PagedPrefillBatch:
        return PagedPrefillBatch(self.cache, [r.block_table for r in reqs], seq_lens)

    def decode_batch(self, reqs, context_lens: list[int]) -> PagedDecodeBatch:
        return PagedDecodeBatch(self.cache, [r.block_table for r in reqs], context_lens)

    def usage(self) -> dict:
        total = self.cache.num_blocks
        return {"kv_unit": "block", "kv_used": total - self.num_free, "kv_total": total}

    def _emit(self, type: str, req, blocks: list[int]) -> None:
        if self.events is not None:
            self.events.emit(type, id=req.request_id, blocks=list(blocks))
