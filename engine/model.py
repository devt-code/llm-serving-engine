"""Qwen3 implemented from scratch in PyTorch.

Phase 1 skeleton. Every `TODO(you)` is yours to implement. The `__init__` methods are done
because parameter names/shapes must match the checkpoint exactly (see engine/loader.py).

Shape conventions (important — this is the vLLM style and makes Phase 2 much easier):
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

Run tests for one component at a time, e.g.:
    pytest tests/test_model.py -k rmsnorm -x
Order that works well: rmsnorm -> rope -> mlp -> attention -> decoder_layer -> logits
-> kv_cache -> greedy.
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

        Precision gotcha: in bf16, compute the normalization in float32, cast back to the
        input dtype, THEN multiply by weight. The order matters for matching HF exactly.
        """
        raise NotImplementedError("TODO(you): RMSNorm.forward")


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings (RoPE).

    For each pair of dims i in [0, head_dim/2): inv_freq[i] = theta ** (-2i / head_dim).
    The angle for position p is p * inv_freq[i].

    Convention gotcha: Qwen3 (like Llama) uses the "rotate_half" layout, NOT interleaved
    pairs. Dim i is paired with dim i + head_dim/2, so cos/sin are [angles, angles]
    concatenated along the last dim. Pairing (0,1), (2,3)... gives wrong outputs with these
    weights.

    Tip: precomputing a cos/sin table for all positions up to max_position in __init__
    (computed in float32) and indexing it with `positions` in forward is the fast approach.
    Register it with `self.register_buffer(..., persistent=False)` so it moves with
    `.to(device)`. Note the loader calls `model.to(dtype)`, which casts float buffers too.
    """

    def __init__(self, head_dim: int, theta: float, max_position: int):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self.max_position = max_position
        # TODO(you): precompute anything you want here (optional)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """positions: [T] (long) -> (cos, sin), each [T, head_dim]"""
        raise NotImplementedError("TODO(you): RotaryEmbedding.forward")


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate q or k by position.

    x:   [T, num_heads, head_dim]
    cos: [T, head_dim]
    sin: [T, head_dim]
    returns [T, num_heads, head_dim], same dtype as x

    Hint: x * cos + rotate_half(x) * sin, where rotate_half([a, b]) = [-b, a]
    (a, b are the first and second halves of the last dim). Mind the broadcasting over heads.
    """
    raise NotImplementedError("TODO(you): apply_rotary")


class KVCache:
    """Simple contiguous KV cache for ONE request (Phase 1). Replaced by the paged cache in Phase 3.

    Preallocate, per layer, a K and a V tensor of shape [max_len, num_key_value_heads, head_dim].

    Protocol used by the model:
      - During a forward pass over T new tokens, each layer calls
            k_all, v_all = cache.update(layer_idx, k_new, v_new)
        which writes k_new/v_new ([T, kv_heads, head_dim]) into slots [seq_len, seq_len + T)
        and returns views of slots [0, seq_len + T) — i.e. all keys/values so far.
      - After ALL layers have run, the model calls cache.advance(T) once, so seq_len only
        moves forward once per forward pass (not once per layer).
    """

    def __init__(self, config: Qwen3Config, max_len: int, device: torch.device, dtype: torch.dtype):
        self.max_len = max_len
        self.seq_len = 0
        raise NotImplementedError("TODO(you): KVCache.__init__ (allocate per-layer K/V tensors)")

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("TODO(you): KVCache.update")

    def advance(self, num_tokens: int) -> None:
        raise NotImplementedError("TODO(you): KVCache.advance")


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
        """x: [T, hidden] -> [T, hidden]

        Steps:
          1. project to q [T, Hq, D], k [T, Hkv, D], v [T, Hkv, D]
          2. q_norm(q), k_norm(k)   (per head, over D)
          3. RoPE on q and k
          4. if kv_cache: k, v = kv_cache.update(self.layer_idx, k, v)  -> [S, Hkv, D], S >= T
          5. grouped-query attention: each kv head serves `num_kv_groups` q heads
          6. causal mask: query i (absolute position S - T + i) may attend to keys 0..S - T + i
          7. o_proj back to [T, hidden]

        Hint: F.scaled_dot_product_attention expects [heads, seq, dim] (or with a batch dim).
        Careful: its `is_causal=True` assumes query and key lengths are equal and aligned at
        position 0. That is wrong when S > T (decode, or prefill with existing cache).
        """
        raise NotImplementedError("TODO(you): Attention.forward")


class MLP(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU: down(silu(gate(x)) * up(x)).  x: [T, hidden] -> [T, hidden]"""
        raise NotImplementedError("TODO(you): MLP.forward")


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
        """Pre-norm residual block:
            x = x + attn(input_layernorm(x))
            x = x + mlp(post_attention_layernorm(x))
        """
        raise NotImplementedError("TODO(you): DecoderLayer.forward")


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
        """input_ids [T], positions [T] -> final hidden states [T, hidden] (after the final norm)

        Compute cos/sin once and share them across all layers. If kv_cache is given,
        call kv_cache.advance(T) after the last layer.
        """
        raise NotImplementedError("TODO(you): Qwen3Model.forward")


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
    ) -> torch.Tensor:
        """input_ids [T], positions [T] -> logits [T, vocab_size]

        (Later optimization, not now: during prefill you only need logits for the last token.)
        """
        raise NotImplementedError("TODO(you): Qwen3ForCausalLM.forward")
