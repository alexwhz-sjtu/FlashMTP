"""Load target / draft models for evaluation."""

from __future__ import annotations

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from specforge.modeling.draft.flashmtp import FlashMTPDraftModel


_ATTN_IMPL: str | None = None


def _attn_implementation() -> str:
    global _ATTN_IMPL
    if _ATTN_IMPL is not None:
        return _ATTN_IMPL
    try:
        import flash_attn  # noqa: F401
        _ATTN_IMPL = "flash_attention_2"
    except ImportError:
        logger.warning(
            "flash_attn is not installed. Falling back to torch.sdpa (lower speed)."
        )
        _ATTN_IMPL = "sdpa"
    return _ATTN_IMPL


def load_target_model(model_path: str, device: torch.device) -> AutoModelForCausalLM:
    return (
        AutoModelForCausalLM.from_pretrained(
            model_path, attn_implementation=_attn_implementation(), dtype=torch.bfloat16
        )
        .to(device)
        .eval()
    )


def load_draft_model(draft_path: str, device: torch.device) -> FlashMTPDraftModel:
    return (
        FlashMTPDraftModel.from_pretrained(
            draft_path, attn_implementation=_attn_implementation(), dtype=torch.bfloat16
        )
        .to(device)
        .eval()
    )


def configure_draft_model(
    draft_model: FlashMTPDraftModel,
    *,
    sink_num: int | None = None,
    local_position: str | None = None,
) -> None:
    fcfg = getattr(draft_model.config, "flashmtp_config", None) or {}

    if sink_num is not None and fcfg.get("sink_num") is not None:
        if draft_model.config.flashmtp_config is None:
            draft_model.config.flashmtp_config = {}
        draft_model.config.flashmtp_config["sink_num"] = sink_num
        if hasattr(draft_model, "sink_num"):
            draft_model.sink_num = int(sink_num)
        logger.info(f"Overriding sink_num={sink_num} (legacy)")

    if local_position is not None:
        lp = local_position == "true"
        draft_model.local_position = lp
        if draft_model.config.flashmtp_config is None:
            draft_model.config.flashmtp_config = {}
        draft_model.config.flashmtp_config["local_position"] = lp
        logger.info(f"Overriding local_position={lp}")

    if fcfg.get("sink_num") is not None and hasattr(draft_model, "sink_num"):
        logger.info(
            f"FlashMTP draft (legacy): sink_num={draft_model.sink_num}, "
            f"block_size={draft_model.block_size}"
        )
    logger.info(
        f"FlashMTP draft: num_middle_layers_n={fcfg.get('num_middle_layers_n', 'n/a')}, "
        f"target_layer_ids={getattr(draft_model, 'target_layer_ids', None)}, "
        f"local_position={getattr(draft_model, 'local_position', fcfg.get('local_position', False))}, "
        f"block_size={draft_model.block_size}"
    )


def load_tokenizer(model_path: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_path)
