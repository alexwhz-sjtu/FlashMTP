from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SUPPORTED_ARCHITECTURE = "FlashMTPDraftModel"


def _get(config: Any, name: str, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


@dataclass(frozen=True)
class FlashMTPConfig:
    num_hidden_layers: int
    num_target_layers: int
    block_size: int
    target_layer_ids: tuple[int, ...]
    mask_token_id: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    include_embedding_chs: bool
    markov_head_type: str
    markov_output_mode: str
    markov_rank: int

    @property
    def num_context_tokens(self) -> int:
        return len(self.target_layer_ids) + int(self.include_embedding_chs)

    @property
    def num_captured_tokens(self) -> int:
        """Target-layer features captured by SGLang (excluding raw embedding)."""
        return len(self.target_layer_ids)


def is_flashmtp_config(config: Any) -> bool:
    architectures = _get(config, "architectures", ()) or ()
    return SUPPORTED_ARCHITECTURE in architectures or _get(
        config, "flashmtp_config", None
    ) is not None


def parse_flashmtp_config(config: Any) -> FlashMTPConfig:
    architectures = tuple(_get(config, "architectures", ()) or ())
    if SUPPORTED_ARCHITECTURE not in architectures:
        raise ValueError(
            f"FlashMTP requires architectures=[{SUPPORTED_ARCHITECTURE!r}], got {architectures}."
        )

    raw = _get(config, "flashmtp_config", None)
    if not isinstance(raw, dict):
        try:
            raw = dict(raw)
        except Exception as exc:
            raise ValueError("FlashMTP checkpoint is missing flashmtp_config.") from exc

    if raw.get("pivot_fuse_mode") != "prefix_condition":
        raise ValueError(
            "The SGLang FlashMTP path only supports pivot_fuse_mode='prefix_condition'."
        )
    if raw.get("local_position") is not True:
        raise ValueError(
            "The SGLang FlashMTP path only supports local_position=true."
        )
    if raw.get("left_shift", False):
        raise ValueError("The SGLang FlashMTP path does not support left_shift=true.")

    markov_head_type = str(raw.get("markov_head_type", "none")).lower()
    markov_output_mode = str(raw.get("markov_output_mode", "additive")).lower()
    markov_rank = int(raw.get("markov_rank", 0))
    if markov_head_type not in {"none", "vanilla", "gated", "rnn", "rnn_easy"}:
        raise ValueError(f"Unsupported FlashMTP markov_head_type={markov_head_type!r}.")
    if markov_output_mode not in {"additive", "direct"}:
        raise ValueError(f"Unsupported FlashMTP markov_output_mode={markov_output_mode!r}.")
    if markov_head_type != "none" and markov_rank <= 0:
        raise ValueError("FlashMTP markov_rank must be positive when a serial head is enabled.")
    if markov_head_type == "none" and markov_output_mode == "direct":
        raise ValueError("FlashMTP direct output mode requires a serial head.")
    if markov_head_type != "none" and markov_output_mode != "direct":
        raise ValueError(
            "The SGLang FlashMTP adapter currently supports serial heads only "
            "with markov_output_mode='direct'."
        )

    target_layer_ids = raw.get("target_layer_ids")
    if not isinstance(target_layer_ids, (list, tuple)) or not target_layer_ids:
        raise ValueError("flashmtp_config.target_layer_ids must be a non-empty list.")
    target_layer_ids = tuple(int(x) for x in target_layer_ids)

    def positive(value: Any, name: str) -> int:
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")
        return value

    num_target_layers = positive(
        _get(config, "num_target_layers", None), "num_target_layers"
    )
    if min(target_layer_ids) < 0 or max(target_layer_ids) >= num_target_layers:
        raise ValueError(
            "flashmtp_config.target_layer_ids contains an index outside "
            f"[0, {num_target_layers})."
        )

    mask_token_id = raw.get("mask_token_id")
    if mask_token_id is None or int(mask_token_id) < 0:
        raise ValueError("flashmtp_config.mask_token_id must be a non-negative integer.")

    hidden_size = positive(_get(config, "hidden_size", None), "hidden_size")
    num_attention_heads = positive(
        _get(config, "num_attention_heads", None), "num_attention_heads"
    )
    head_dim = positive(
        _get(config, "head_dim", hidden_size // num_attention_heads), "head_dim"
    )
    return FlashMTPConfig(
        num_hidden_layers=positive(
            _get(config, "num_hidden_layers", None), "num_hidden_layers"
        ),
        num_target_layers=num_target_layers,
        block_size=positive(_get(config, "block_size", None), "block_size"),
        target_layer_ids=target_layer_ids,
        mask_token_id=int(mask_token_id),
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=positive(
            _get(config, "num_key_value_heads", num_attention_heads),
            "num_key_value_heads",
        ),
        head_dim=head_dim,
        vocab_size=positive(_get(config, "vocab_size", None), "vocab_size"),
        include_embedding_chs=bool(raw.get("include_embedding_chs", False)),
        markov_head_type=markov_head_type,
        markov_output_mode=markov_output_mode,
        markov_rank=markov_rank,
    )


def validate_target_compatibility(draft: FlashMTPConfig, target_config: Any) -> None:
    text_config = _get(target_config, "text_config", target_config)
    model_type = _get(text_config, "model_type", None)
    architectures = tuple(_get(target_config, "architectures", ()) or ())
    supported = (
        model_type == "qwen3"
        or model_type in {"qwen3_5_text", "qwen3_5_moe_text"}
    )
    if not supported:
        raise ValueError(
            "FlashMTP SGLang supports Qwen3 and Qwen3.5 text targets; "
            f"got model_type={model_type!r}, "
            f"architectures={architectures}."
        )
    checks = {
        "hidden_size": draft.hidden_size,
        "num_attention_heads": draft.num_attention_heads,
        "num_key_value_heads": draft.num_key_value_heads,
        "head_dim": draft.head_dim,
        "vocab_size": draft.vocab_size,
    }
    for name, expected in checks.items():
        actual = int(_get(text_config, name, -1))
        if actual != expected:
            raise ValueError(
                f"FlashMTP/target {name} mismatch: draft={expected}, target={actual}."
            )
    target_layers = int(_get(text_config, "num_hidden_layers", -1))
    if target_layers != draft.num_target_layers:
        raise ValueError(
            "FlashMTP/target layer-count mismatch: "
            f"draft trained for {draft.num_target_layers}, target has {target_layers}."
        )
