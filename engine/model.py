"""Qwen3 implemented from scratch in PyTorch.

Shape conventions (the vLLM style; it makes Phase 2 much easier):
    There is NO batch dimension. A forward pass takes one flat sequence of T tokens.
      input_ids:  [T]            token ids
      positions:  [T]            absolute position of each token (0..T-1 for a fresh prompt,
                                 or e.g. [17] for the 18th token during decode)
      hidden:     [T, hidden_size]
      q:          [T, num_attention_heads, head_dim]
      k, v:       [T, num_key_value_heads, head_dim]
      logits:     [T, vocab_size]

Qwen3-0.6B numbers: hidden 1024, 16 q heads, 8 kv heads, head_dim 128 (so q_proj is
1024 -> 2048, not 1024 -> 1024), MLP 3072, 28 layers, rope_theta 1e6, eps 1e-6.

Tests: pytest tests/test_model.py   (or one component: -k rmsnorm, -k rope, -k attention, ...)
"""

import torch
import torch.nn.functional as F
from torch import nn

from engine.config import Qwen3Config


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., dim] -> [..., dim]

        y = x / sqrt(mean(x^2 over last dim) + eps) * weight

        Unlike LayerNorm there is no mean subtraction and no bias: it only rescales each
        vector to unit root-mean-square, then applies a learned per-dimension gain.
        """
        input_dtype = x.dtype
        # Squaring bf16 values loses precision fast, so do the statistics in float32.
        # F.rms_norm without a weight computes exactly  x * rsqrt(mean(x^2) + eps)  as ONE
        # fused kernel instead of ~5 small ones (pow, mean, add, rsqrt, mul). On a GPU,
        # decode is dominated by per-kernel launch overhead, so this saves ~5 ms per step.
        normed = F.rms_norm(x.float(), (x.shape[-1],), eps=self.eps)
        # Cast back BEFORE multiplying by weight; this is HF's order, and in bf16 the
        # other order gives slightly different numbers (test_rmsnorm_bf16_order checks it).
        return self.weight * normed.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings (RoPE).

    RoPE encodes position by ROTATING q and k vectors instead of adding a position vector.
    Dims are grouped into head_dim/2 pairs; pair i is rotated by angle  p * inv_freq[i],
    where p is the token's position and inv_freq[i] = theta ** (-2i / head_dim).
    Low i -> fast rotation (fine-grained position), high i -> slow (long-range position).
    Because q·k after rotation depends only on the angle DIFFERENCE, attention scores end
    up depending on relative position, which is the whole point.

    Layout: Qwen3 uses "rotate_half" pairing, i.e. dim i is paired with dim i + head_dim/2
    (not adjacent dims). So cos/sin are [angles, angles] concatenated along the last dim.
    """

    def __init__(self, head_dim: int, theta: float, max_position: int):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self.max_position = max_position

        # Precompute a cos/sin table for every position once (in float32 for accuracy),
        # so forward() is just a table lookup. 40960 x 128 x 2 tables = ~40 MB in fp32.
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))  # [D/2]
        pos = torch.arange(max_position, dtype=torch.float32)  # [P]
        angles = torch.outer(pos, inv_freq)  # [P, D/2]
        angles = torch.cat([angles, angles], dim=-1)  # [P, D] (rotate_half layout)
        # persistent=False: not part of the checkpoint, but still moves with .to(device)
        self.register_buffer("cos_cache", angles.cos(), persistent=False)
        self.register_buffer("sin_cache", angles.sin(), persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """positions: [T] (long) -> (cos, sin), each [T, head_dim]"""
        return self.cos_cache[positions], self.sin_cache[positions]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """[a, b] -> [-b, a], where a and b are the two halves of the last dim."""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate q or k by position.

    x:   [T, num_heads, head_dim]
    cos: [T, head_dim]
    sin: [T, head_dim]
    returns [T, num_heads, head_dim], same dtype as x

    For each pair (a, b) this computes the 2D rotation
        a' = a*cos - b*sin
        b' = b*cos + a*sin
    written in vector form as  x*cos + rotate_half(x)*sin.
    """
    cos = cos[:, None, :].to(x.dtype)  # [T, 1, D], broadcasts over heads
    sin = sin[:, None, :].to(x.dtype)
    return x * cos + rotate_half(x) * sin


class KVCache:
    """Simple contiguous KV cache for ONE request (Phase 1).

    Per layer, a K and a V tensor of shape [max_len, num_key_value_heads, head_dim] is
    preallocated up front. That upfront reservation of max_len slots per request is exactly
    the memory waste the paged cache fixes later.

    Attention-backend protocol (shared with engine/batch_cache.py, and later the paged cache):
      - attend(layer_idx, q, k, v, scale) -> [T, Hq, D]
            Store this forward pass's new k/v, then attend q over everything cached so far.
            Each cache type decides how keys are laid out and how the mask is built.
      - advance(T)
            Called by the model ONCE per forward pass, after all layers ran, so lengths
            move forward once per forward pass (not once per layer).
    """

    def __init__(self, config: Qwen3Config, max_len: int, device: torch.device, dtype: torch.dtype):
        self.max_len = max_len
        self.seq_len = 0
        shape = (max_len, config.num_key_value_heads, config.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.num_hidden_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.num_hidden_layers)]

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        start, end = self.seq_len, self.seq_len + k.shape[0]
        if end > self.max_len:
            raise RuntimeError(f"KV cache full: need {end} slots, have {self.max_len}")
        self.k[layer_idx][start:end] = k
        self.v[layer_idx][start:end] = v
        # Slicing returns views (no copy).
        return self.k[layer_idx][:end], self.v[layer_idx][:end]

    def attend(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
        """q [T, Hq, D], k/v [T, Hkv, D] (this step's tokens only) -> [T, Hq, D]"""
        T = q.shape[0]
        # Append this step's k/v and get ALL keys/values so far. Only k and v are cached;
        # q is only needed for the current tokens.
        k, v = self.update(layer_idx, k, v)  # [S, Hkv, D], S = past + T
        S = k.shape[0]

        # Causal mask. Our T queries sit at absolute positions S-T .. S-1, so query i may
        # see keys 0 .. (S - T + i). With no past S == T and this is the usual lower
        # triangle. During decode (T=1) the single query sees every key. SDPA's
        # is_causal=True assumes S == T, which is why we build the mask ourselves.
        q_pos = torch.arange(S - T, S, device=q.device)[:, None]  # [T, 1]
        k_pos = torch.arange(S, device=q.device)[None, :]  # [1, S]
        mask = k_pos <= q_pos  # [T, S], True = may attend

        out = F.scaled_dot_product_attention(
            q.transpose(0, 1),  # [Hq, T, D]   (SDPA wants [heads, seq, dim])
            k.transpose(0, 1),  # [Hkv, S, D]
            v.transpose(0, 1),
            attn_mask=mask,
            scale=scale,
            enable_gqa=True,
        )
        return out.transpose(0, 1)  # [T, Hq, D]

    def advance(self, num_tokens: int) -> None:
        self.seq_len += num_tokens


class Attention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = config.num_kv_groups
        self.scale = self.head_dim**-0.5

        h, d = config.hidden_size, config.head_dim
        self.q_proj = nn.Linear(h, self.num_heads * d, bias=False)
        self.k_proj = nn.Linear(h, self.num_kv_heads * d, bias=False)
        self.v_proj = nn.Linear(h, self.num_kv_heads * d, bias=False)
        self.o_proj = nn.Linear(self.num_heads * d, h, bias=False)
        # Qwen3-specific: RMSNorm over head_dim, applied per head to q and k (before RoPE).
        self.q_norm = RMSNorm(d, config.rms_norm_eps)
        self.k_norm = RMSNorm(d, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        """x: [T, hidden] -> [T, hidden]"""
        T = x.shape[0]

        # 1. Project and split into heads.
        q = self.q_proj(x).view(T, self.num_heads, self.head_dim)  # [T, 16, 128]
        k = self.k_proj(x).view(T, self.num_kv_heads, self.head_dim)  # [T, 8, 128]
        v = self.v_proj(x).view(T, self.num_kv_heads, self.head_dim)  # [T, 8, 128]

        # 2. Qwen3's per-head q/k norm. Keeps q·k magnitudes bounded, which stabilizes
        #    training; at inference we just have to replicate it.
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 3. Encode positions by rotating q and k.
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        # 4. Attention. Grouped-query attention: 16 query heads share 8 KV heads, 2 query
        #    heads per KV head. Query heads 0,1 use KV head 0; 2,3 use KV head 1; and so on.
        #    Only 8 heads are stored in the cache, which halves KV memory compared to 16.
        #    The explicit version would be  k = k.repeat_interleave(2, dim=1)  (same for v),
        #    but that copies the whole cache in every layer at every step. SDPA's
        #    enable_gqa=True does the same head mapping without materializing the copy.
        if kv_cache is None:
            # No cache: plain causal self-attention over these T tokens (S == T, so
            # is_causal=True is correct here).
            out = F.scaled_dot_product_attention(
                q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
                is_causal=True, scale=self.scale, enable_gqa=True,
            ).transpose(0, 1)
        else:
            # The cache stores this step's k/v and attends over everything so far. How it
            # lays out keys and builds the mask depends on the cache type (one request,
            # a batch of requests, or later a paged cache).
            out = kv_cache.attend(self.layer_idx, q, k, v, self.scale)  # [T, 16, 128]

        # 5. Merge heads and project back to the hidden size.
        out = out.reshape(T, self.num_heads * self.head_dim)  # [T, 2048]
        return self.o_proj(out)  # [T, 1024]


class MLP(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU: down(silu(gate(x)) * up(x)).  x: [T, hidden] -> [T, hidden]

        The gate branch decides how much of each up-projected feature passes through.
        """
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        """Pre-norm residual block: normalize the input to each sublayer, then add its output
        back to the un-normalized residual stream."""
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, kv_cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(config, i) for i in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta, config.max_position_embeddings)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        """input_ids [T], positions [T] -> final hidden states [T, hidden] (after the final norm)"""
        # cos/sin depend only on positions, so compute them once and share across all 28 layers.
        cos, sin = self.rotary_emb(positions)
        x = self.embed_tokens(input_ids)  # [T, 1024]
        for layer in self.layers:
            x = layer(x, cos, sin, kv_cache)
        if kv_cache is not None:
            kv_cache.advance(input_ids.shape[0])  # once per forward pass, after every layer wrote
        return self.norm(x)


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        # Tied to model.embed_tokens.weight by the loader (tie_word_embeddings=true).
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: KVCache | None = None,
        logits_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """input_ids [T], positions [T] -> logits [T, vocab_size]
        (or [len(logits_indices), vocab_size] if logits_indices is given)

        logits_indices: which token rows actually need logits. During prefill only the LAST
        token of each prompt is sampled from, so skipping the other rows of this
        1024 x 151936 matmul saves real time on long prompts.
        """
        hidden = self.model(input_ids, positions, kv_cache)
        if logits_indices is not None:
            hidden = hidden[logits_indices]
        return self.lm_head(hidden)
