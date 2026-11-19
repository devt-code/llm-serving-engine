"""Model configuration, read from the checkpoint's config.json."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Qwen3Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    eos_token_id: int
    # All token ids that end generation: config eos plus generation_config.json's eos list
    # (Qwen3: <|im_end|> 151645 and <|endoftext|> 151643).
    stop_token_ids: tuple[int, ...] = ()

    @property
    def num_kv_groups(self) -> int:
        """How many query heads share each KV head (grouped-query attention)."""
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> "Qwen3Config":
        raw = json.loads((Path(model_dir) / "config.json").read_text())
        stop = [raw["eos_token_id"]]
        gen_path = Path(model_dir) / "generation_config.json"
        if gen_path.exists():
            gen_eos = json.loads(gen_path.read_text()).get("eos_token_id", [])
            stop += gen_eos if isinstance(gen_eos, list) else [gen_eos]
        return cls(
            stop_token_ids=tuple(dict.fromkeys(stop)),
            vocab_size=raw["vocab_size"],
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            head_dim=raw.get("head_dim", raw["hidden_size"] // raw["num_attention_heads"]),
            rms_norm_eps=raw["rms_norm_eps"],
            rope_theta=raw["rope_theta"],
            max_position_embeddings=raw["max_position_embeddings"],
            tie_word_embeddings=raw.get("tie_word_embeddings", False),
            eos_token_id=raw["eos_token_id"],
        )
