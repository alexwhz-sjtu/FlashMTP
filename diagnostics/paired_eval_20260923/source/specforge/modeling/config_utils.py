"""Model-config helpers shared by FlashMTP training components.

Qwen3.5 checkpoints use a multimodal wrapper config whose language-model
parameters live under ``text_config``.  The FlashMTP draft is intentionally a
small dense Qwen3-style transformer, so Qwen3.5 target metadata is converted
to a compatible :class:`Qwen3Config` instead of copying the target's hybrid
GatedDeltaNet layer layout.
"""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, PretrainedConfig, Qwen3Config

QWEN35_MODEL_TYPES = {"qwen3_5", "qwen3_5_moe"}


def get_pretrained_config_dict(
    pretrained_model_name_or_path: str,
    *,
    cache_dir: str | None = None,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Read a checkpoint config without requiring its model class to exist.

    ``AutoConfig`` in the currently pinned Transformers release does not know
    Qwen3.5. ``PretrainedConfig.get_config_dict`` still provides the raw JSON
    for both local paths and Hub repositories, which is all we need to build a
    dense FlashMTP draft config.
    """

    config_dict, _ = PretrainedConfig.get_config_dict(
        pretrained_model_name_or_path,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    return config_dict


def get_source_model_type(
    pretrained_model_name_or_path: str,
    *,
    cache_dir: str | None = None,
    trust_remote_code: bool = False,
) -> str:
    config_dict = get_pretrained_config_dict(
        pretrained_model_name_or_path,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    return str(config_dict.get("model_type", ""))


def is_qwen35_model_type(model_type: str | None) -> bool:
    return str(model_type or "") in QWEN35_MODEL_TYPES


def _qwen35_text_dict_to_qwen3_config(
    outer_config: dict[str, Any],
) -> Qwen3Config:
    text_config = outer_config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError("Qwen3.5 config is missing the required text_config object.")

    required = (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
    )
    missing = [name for name in required if text_config.get(name) is None]
    if missing:
        raise ValueError(
            "Qwen3.5 text_config is missing required FlashMTP fields: "
            + ", ".join(missing)
        )

    # Qwen3.5 uses partial multimodal RoPE and a hybrid linear/full-attention
    # target. The draft does not need to reproduce the target architecture; it
    # only needs matching hidden/vocabulary dimensions. Use dense Qwen3 layers
    # and a regular 1-D RoPE with the target's theta.
    rope_parameters = text_config.get("rope_parameters") or {}
    num_hidden_layers = int(text_config["num_hidden_layers"])
    kwargs: dict[str, Any] = {
        "vocab_size": int(text_config["vocab_size"]),
        "hidden_size": int(text_config["hidden_size"]),
        "intermediate_size": int(text_config["intermediate_size"]),
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": int(text_config["num_attention_heads"]),
        "num_key_value_heads": int(text_config["num_key_value_heads"]),
        "head_dim": int(text_config["head_dim"]),
        "hidden_act": text_config.get("hidden_act", "silu"),
        "max_position_embeddings": int(
            text_config.get("max_position_embeddings", 32768)
        ),
        "initializer_range": float(text_config.get("initializer_range", 0.02)),
        "rms_norm_eps": float(text_config.get("rms_norm_eps", 1e-6)),
        "use_cache": bool(text_config.get("use_cache", True)),
        "tie_word_embeddings": bool(
            outer_config.get(
                "tie_word_embeddings",
                text_config.get("tie_word_embeddings", False),
            )
        ),
        "rope_theta": float(
            text_config.get("rope_theta", rope_parameters.get("rope_theta", 10000.0))
        ),
        "attention_bias": bool(text_config.get("attention_bias", False)),
        "attention_dropout": float(text_config.get("attention_dropout", 0.0)),
        "use_sliding_window": False,
        "layer_types": ["full_attention"] * num_hidden_layers,
    }
    for token_name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = text_config.get(token_name, outer_config.get(token_name))
        if value is not None:
            kwargs[token_name] = value

    config = Qwen3Config(**kwargs)
    config.flashmtp_source_model_type = str(outer_config.get("model_type", "qwen3_5"))
    config.flashmtp_target_architectures = list(outer_config.get("architectures", []))
    return config


def load_text_model_config(
    pretrained_model_name_or_path: str,
    *,
    cache_dir: str | None = None,
    trust_remote_code: bool = False,
) -> PretrainedConfig:
    """Load the text-backbone config used by FlashMTP.

    Qwen3.5 is normalized to a dense ``Qwen3Config``. Other checkpoints retain
    their existing ``AutoConfig`` behavior, with a nested text config unwrapped
    when one is present.
    """

    raw_config = get_pretrained_config_dict(
        pretrained_model_name_or_path,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    model_type = str(raw_config.get("model_type", ""))
    if is_qwen35_model_type(model_type):
        return _qwen35_text_dict_to_qwen3_config(raw_config)

    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    return getattr(config, "text_config", config)
