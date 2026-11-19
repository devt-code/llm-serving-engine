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

PREFIX CACHING (Phase 4): a FULL block's contents are fully determined by the tokens in it and
every token before it. So each full block gets a hash:
    hash(block i) = hash(hash(block i-1), tokens in block i)
and the block manager keeps a table  hash -> physical block. A new request whose first blocks
hash to blocks already in the pool reuses them (reference counted) instead of recomputing them:
a shared system prompt is computed once, then only each request's own suffix is prefilled.
When a cached block's last user finishes, it isn't wiped: it stays cached (ref count 0) until
the pool needs space, and the least recently used one is evicted first.
"""

import torch
import torch.nn.functional as F

from collections import OrderedDict

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


class PagedAttentionBatch:
    """One forward pass over B requests, each with `new` tokens to compute on top of `context`
    tokens already in its blocks (cached prefix, or earlier decode steps).

    This one layout covers every case:
        fresh prompt          new = prompt_len,          context before = 0
        prompt with prefix    new = uncached suffix,     context before = cached tokens
        decode                new = 1,                   context before = everything so far

    Tokens come in flat ([T] = all requests' new tokens back to back, the model's layout).
    For attention they're scattered into a padded [B, Tmax] grid, keys are gathered from each
    request's blocks into [B, L], and one masked SDPA call runs over [B, Hq, Tmax, L]:
        query at absolute position p may see key position j  iff  j <= p.
    Cost is B x Tmax x L, so requests with very different `new` lengths waste some work on
    padding. (Packing everything into one flat sequence, as the slot cache does, costs
    (sum of lengths)^2 instead, which explodes once cached context is included.)
    """

    def __init__(self, cache: PagedKVCache, block_tables: list[list[int]], new_lens: list[int], context_lens: list[int]):
        """context_lens = total tokens per request after this step (cached/earlier + new)."""
        _check_tables(block_tables, context_lens, cache.block_size)
        dev, bs = cache.device, cache.block_size
        self.cache = cache
        self.B = len(new_lens)
        self.Tmax = max(new_lens)
        new = torch.tensor(new_lens, device=dev)
        ctx = torch.tensor(context_lens, device=dev)
        start = ctx - new  # absolute position of each request's first new token

        # Flat token layout: which request, and absolute position.
        seq_idx = torch.repeat_interleave(torch.arange(self.B, device=dev), new)  # [T]
        local = torch.arange(int(new.sum()), device=dev) - (torch.cumsum(new, 0) - new)[seq_idx]  # 0.. within request
        self.positions = start[seq_idx] + local  # [T], also what RoPE needs
        self.last_token_idx = torch.cumsum(new, 0) - 1  # [B] rows that need logits
        self.q_index = seq_idx * self.Tmax + local  # [T] -> slot in the padded [B * Tmax] grid

        # Where each new token's k/v is written in the pool.
        num_cols = -(-max(context_lens) // bs)
        tables = _padded_tables(block_tables, num_cols, dev)  # [B, num_cols]
        self.slot_mapping = tables[seq_idx, self.positions // bs] * bs + self.positions % bs  # [T]
        self.flat_tables = tables.flatten()  # gather order for keys, shared by all layers
        self.L = num_cols * bs

        # Mask [B, 1, Tmax, L]. Padding query rows get position 0, so they see key 0 only
        # (never an empty row, which would give NaN). Their outputs are discarded.
        t = torch.arange(self.Tmax, device=dev)
        q_pos = torch.where(t[None, :] < new[:, None], start[:, None] + t[None, :], 0)  # [B, Tmax]
        key_pos = torch.arange(self.L, device=dev)
        self.mask = (key_pos[None, None, :] <= q_pos[:, :, None])[:, None]  # causal also bounds by ctx

    def attend(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
        """q [T, Hq, D], k/v [T, Hkv, D] (this step's new tokens) -> [T, Hq, D]"""
        k_flat, v_flat = self.cache.flat(layer_idx)
        k_flat[self.slot_mapping] = k
        v_flat[self.slot_mapping] = v

        Hq, Hkv, D = q.shape[1], self.cache.num_kv_heads, self.cache.head_dim
        # Gather each request's blocks: [B * num_cols, bs, Hkv, D] -> [B, Hkv, L, D].
        # index_select on a flat index is ~4x faster than advanced indexing (pool[tables]) on
        # MPS for this shape (measured: 0.84 vs 3.29 ms per layer at B=32, L=288).
        keys = self.cache.k[layer_idx].index_select(0, self.flat_tables).view(self.B, self.L, Hkv, D).transpose(1, 2)
        values = self.cache.v[layer_idx].index_select(0, self.flat_tables).view(self.B, self.L, Hkv, D).transpose(1, 2)

        if self.Tmax == 1:  # decode: the flat layout already is [B, ...], no scatter needed
            q_pad = q[:, :, None, :]  # [B, Hq, 1, D]
        else:
            q_pad = q.new_zeros(self.B * self.Tmax, Hq, D)
            q_pad[self.q_index] = q
            q_pad = q_pad.view(self.B, self.Tmax, Hq, D).transpose(1, 2)  # [B, Hq, Tmax, D]
        out = F.scaled_dot_product_attention(q_pad, keys, values, attn_mask=self.mask, scale=scale, enable_gqa=True)
        out = out.transpose(1, 2).reshape(self.B * self.Tmax, Hq, D)  # [B * Tmax, Hq, D]
        return out if self.Tmax == 1 else out[self.q_index]

    def advance(self, num_tokens: int) -> None:
        pass


def PagedDecodeBatch(cache: PagedKVCache, block_tables: list[list[int]], context_lens: list[int]) -> PagedAttentionBatch:
    """One new token for each of B running requests."""
    return PagedAttentionBatch(cache, block_tables, [1] * len(context_lens), context_lens)


class BlockManager:
    """Hands out and reclaims KV blocks, with optional prefix caching.
    Same interface as SlotManager (engine/batch_cache.py).

    Block states:
      free       in self.free_blocks, contents meaningless
      in use     ref_count > 0 (one per request whose table holds it)
      cached     ref_count == 0 but contents still valid and indexed by hash, in self.evictable
                 (LRU order). Reusable by a prefix hit, or evicted when free blocks run out.

    watermark: fraction of blocks kept free when ADMITTING new requests, so running requests
    have room to grow for a while and we don't preempt right after admitting.
    """

    kind = "paged"

    def __init__(
        self, cache: PagedKVCache, events: EventLog | None = None, watermark: float = 0.01,
        enable_prefix_caching: bool = False,
    ):
        self.cache = cache
        self.block_size = cache.block_size
        self.free_blocks: list[int] = list(range(cache.num_blocks - 1, -1, -1))  # pop() gives block 0 first
        self.watermark_blocks = int(watermark * cache.num_blocks)
        self.events = events
        self.enable_prefix_caching = enable_prefix_caching
        self.ref_count = [0] * cache.num_blocks
        self.hash_to_block: dict[int, int] = {}
        self.block_hash: dict[int, int] = {}  # physical block -> its hash (only full, registered blocks)
        self.evictable: OrderedDict[int, None] = OrderedDict()  # cached, ref 0; oldest first
        # Stats
        self.prompt_tokens_seen = 0
        self.prefix_hit_tokens = 0

    # ------------------------------------------------------------------ sizing

    def blocks_needed(self, num_tokens: int) -> int:
        return -(-num_tokens // self.block_size)

    @property
    def max_request_tokens(self) -> int:
        return self.cache.num_blocks * self.block_size

    @property
    def num_free(self) -> int:
        """Blocks that can be handed out: truly free plus cached-but-unused (evictable)."""
        return len(self.free_blocks) + len(self.evictable)

    # ------------------------------------------------------------------ prefix lookup

    def _hashes(self, tokens: list[int]) -> list[int]:
        """Chained hash for every FULL block of `tokens`."""
        hashes, h, bs = [], 0, self.block_size
        for i in range(len(tokens) // bs):
            h = hash((h, tuple(tokens[i * bs : (i + 1) * bs])))
            hashes.append(h)
        return hashes

    def _match(self, req) -> list[int]:
        """Physical blocks for the longest cached prefix of req's tokens."""
        if not self.enable_prefix_caching:
            return []
        tokens = req.all_token_ids
        matched = []
        for h in self._hashes(tokens):
            b = self.hash_to_block.get(h)
            if b is None:
                break
            matched.append(b)
        # At least one token must be computed to get logits for the next token. If the whole
        # sequence is cached, recompute its last block (its k/v gets rewritten in a private block).
        if matched and len(matched) * self.block_size >= len(tokens):
            matched.pop()
        return matched

    def num_cached_tokens(self, req) -> int:
        return len(self._match(req)) * self.block_size

    # ------------------------------------------------------------------ allocation

    def can_admit(self, req, num_tokens: int) -> bool:
        matched = self._match(req)
        # Reusing an evictable block takes it out of the free pool too, so it counts as used.
        reused_from_evictable = sum(1 for b in matched if self.ref_count[b] == 0)
        need = self.blocks_needed(num_tokens) - len(matched) + reused_from_evictable
        return self.num_free - need >= self.watermark_blocks

    def allocate(self, req, num_tokens: int) -> int:
        """Build req's block table. Returns how many tokens are already cached (prefix hit)."""
        matched = self._match(req)
        for b in matched:
            if self.ref_count[b] == 0:
                self.evictable.pop(b)
            self.ref_count[b] += 1
        new = [self._take_block() for _ in range(self.blocks_needed(num_tokens) - len(matched))]
        req.block_table = matched + new
        req.block_hashes = self._hashes(req.all_token_ids[: len(matched) * self.block_size])
        cached = len(matched) * self.block_size
        if req.num_preemptions == 0:
            self.prompt_tokens_seen += len(req.prompt_ids)
            self.prefix_hit_tokens += min(cached, len(req.prompt_ids))
        if matched:
            self._emit("prefix_cache_hit", req, matched, cached_tokens=cached)
        if new:
            self._emit("blocks_allocated", req, new)
        return cached

    def ensure_capacity(self, req, num_tokens: int) -> bool:
        """Make sure req's table covers num_tokens tokens. False if the pool is out of blocks."""
        missing = self.blocks_needed(num_tokens) - len(req.block_table)
        if missing <= 0:
            return True
        if missing > self.num_free:
            return False
        new = [self._take_block() for _ in range(missing)]
        req.block_table.extend(new)
        self._emit("blocks_allocated", req, new)
        return True

    def commit(self, req) -> None:
        """Called after each forward pass: register req's newly FULL blocks in the prefix cache."""
        if not self.enable_prefix_caching:
            return
        n_full = req.num_computed // self.block_size
        if n_full <= len(req.block_hashes):
            return
        tokens = req.all_token_ids
        h = req.block_hashes[-1] if req.block_hashes else 0
        bs = self.block_size
        for i in range(len(req.block_hashes), n_full):
            h = hash((h, tuple(tokens[i * bs : (i + 1) * bs])))
            req.block_hashes.append(h)
            b = req.block_table[i]
            if h not in self.hash_to_block:  # another request may have computed the same block first
                self.hash_to_block[h] = b
                self.block_hash[b] = h

    def free(self, req) -> None:
        if req.block_table:
            self._emit("blocks_freed", req, req.block_table)
            # Reverse order: the LAST blocks of a sequence are the least useful prefixes, so they
            # should be evicted first. Insert them into the LRU first (= oldest).
            for b in reversed(req.block_table):
                self.ref_count[b] -= 1
                if self.ref_count[b] > 0:
                    continue  # still shared with another running request
                if b in self.block_hash:
                    self.evictable[b] = None  # keep contents for future prefix hits
                else:
                    self.free_blocks.append(b)
        req.block_table = None
        req.block_hashes = []

    def _take_block(self) -> int:
        if self.free_blocks:
            b = self.free_blocks.pop()
        else:
            b, _ = self.evictable.popitem(last=False)  # least recently used cached block
            del self.hash_to_block[self.block_hash.pop(b)]
            if self.events is not None:
                self.events.emit("block_evicted", block=b)
        self.ref_count[b] = 1
        return b

    # ------------------------------------------------------------------ batches and stats

    def prefill_batch(self, reqs, new_lens: list[int], context_lens: list[int]) -> PagedAttentionBatch:
        return PagedAttentionBatch(self.cache, [r.block_table for r in reqs], new_lens, context_lens)

    def decode_batch(self, reqs, context_lens: list[int]) -> PagedAttentionBatch:
        return PagedDecodeBatch(self.cache, [r.block_table for r in reqs], context_lens)

    def usage(self) -> dict:
        total = self.cache.num_blocks
        return {
            "kv_unit": "block", "kv_used": total - self.num_free, "kv_total": total,
            "kv_cached": len(self.evictable),
            "prefix_hit_rate": round(self.prefix_hit_tokens / self.prompt_tokens_seen, 4) if self.prompt_tokens_seen else 0.0,
        }

    def _emit(self, type: str, req, blocks: list[int], **extra) -> None:
        if self.events is not None:
            self.events.emit(type, id=req.request_id, blocks=list(blocks), **extra)
