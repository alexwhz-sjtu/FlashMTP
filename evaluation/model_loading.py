"""Shared FlashMTP target/draft loading for benchmark and profiling."""

from __future__ import annotations

import argparse
from typing import Any

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from specforge.modeling.draft.flashmtp import FlashMTPDraftModel, load_flashmtp_draft_model


def resolve_mask_token_id(
    draft_model: FlashMTPDraftModel,
    tokenizer: AutoTokenizer,
    *,
    cli_mask_token_id: int | None = None,
) -> int:
    """Resolve mask token id using the same priority as training."""
    if cli_mask_token_id is not None:
        return int(cli_mask_token_id)

    mask_token_id = draft_model.mask_token_id
    if mask_token_id is None:
        fcfg = getattr(draft_model.config, "flashmtp_config", None) or {}
        mask_token_id = fcfg.get("mask_token_id")

    if mask_token_id is not None:
        return int(mask_token_id)

    if tokenizer.mask_token_id is not None:
        return int(tokenizer.mask_token_id)

    tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
    if tokenizer.mask_token_id is None:
        raise ValueError(
            "mask_token_id is None. Pass --mask-token-id, use a checkpoint with "
            "flashmtp_config['mask_token_id'], or a tokenizer with mask_token_id."
        )
    return int(tokenizer.mask_token_id)


def flashmtp_config_summary(draft_model: FlashMTPDraftModel) -> dict[str, Any]:
    fcfg = getattr(draft_model.config, "flashmtp_config", None) or {}
    return {
        "pivot_fuse_mode": getattr(draft_model, "pivot_fuse_mode", fcfg.get("pivot_fuse_mode")),
        "num_middle_layers_n": fcfg.get("num_middle_layers_n"),
        "target_layer_ids": getattr(draft_model, "target_layer_ids", None),
        "include_embedding_chs": getattr(
            draft_model, "include_embedding_chs", fcfg.get("include_embedding_chs", False)
        ),
        "local_position": getattr(draft_model, "local_position", fcfg.get("local_position", False)),
        "left_shift": getattr(draft_model, "left_shift", fcfg.get("left_shift", False)),
        "block_size": int(getattr(draft_model, "block_size", fcfg.get("block_size", 0))),
        "markov_head_type": getattr(draft_model, "markov_head_type", fcfg.get("markov_head_type", "none")),
        "markov_output_mode": getattr(
            draft_model, "markov_output_mode", fcfg.get("markov_output_mode", "additive")
        ),
        "markov_rank": getattr(draft_model, "markov_rank", fcfg.get("markov_rank", 0)),
        "mask_token_id": getattr(draft_model, "mask_token_id", fcfg.get("mask_token_id")),
    }


def log_flashmtp_config(draft_model: FlashMTPDraftModel) -> dict[str, Any]:
    summary = flashmtp_config_summary(draft_model)
    logger.info(
        "FlashMTP draft: pivot_fuse_mode={} num_middle_layers_n={} target_layer_ids={} "
        "include_embedding_chs={} local_position={} left_shift={} block_size={} "
        "markov_head_type={} markov_output_mode={} "
        "markov_rank={} mask_token_id={}",
        summary["pivot_fuse_mode"],
        summary["num_middle_layers_n"],
        summary["target_layer_ids"],
        summary["include_embedding_chs"],
        summary["local_position"],
        summary["left_shift"],
        summary["block_size"],
        summary["markov_head_type"],
        summary["markov_output_mode"],
        summary["markov_rank"],
        summary["mask_token_id"],
    )
    return summary


def validate_decode_config(draft_model: FlashMTPDraftModel) -> None:
    """Log serial-head inference settings from the loaded checkpoint."""
    summary = flashmtp_config_summary(draft_model)
    markov_head_type = str(summary["markov_head_type"])
    markov_output_mode = str(summary["markov_output_mode"])
    left_shift = bool(summary["left_shift"])

    if left_shift:
        logger.info(
            "Block alignment: left_shift (config block_size={} is total span; "
            "draft slots={}; proposals={})",
            summary["block_size"],
            draft_model.draft_block_len,
            draft_model.proposal_length,
        )
    else:
        logger.info(
            "Block alignment: legacy (config block_size={} is draft block width; "
            "slot 0 unsupervised; proposals={})",
            summary["block_size"],
            draft_model.proposal_length,
        )

    if markov_head_type == "none":
        return

    logger.info(
        "Serial head enabled for inference: type={} output_mode={} rank={}",
        markov_head_type,
        markov_output_mode,
        summary["markov_rank"],
    )
    if markov_output_mode == "direct":
        logger.info(
            "Direct serial-head mode: draft logits come from the Markov head only "
            "(base LM head is skipped for draft sampling)."
        )
    elif markov_output_mode == "additive":
        logger.info(
            "Additive serial-head mode: draft logits = base LM head(h) + Markov bias."
        )


def has_flash_attention() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except ImportError:
        logger.warning(
            "flash_attn is not installed. Falling back to torch.sdpa. "
            "The speedup will be lower."
        )
        return False


def load_flashmtp_benchmark_models(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[AutoModelForCausalLM, FlashMTPDraftModel, AutoTokenizer, dict[str, Any]]:
    installed_flash_attn = has_flash_attention()
    attn_impl = "flash_attention_2" if installed_flash_attn else "sdpa"

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
        trust_remote_code=getattr(args, "trust_remote_code", False),
    ).to(device).eval()

    draft_model = load_flashmtp_draft_model(
        args.draft_name_or_path,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
        trust_remote_code=getattr(args, "trust_remote_code", False),
    ).to(device).eval()

    if getattr(args, "local_position", None) is not None:
        lp = args.local_position == "true"
        draft_model.local_position = lp
        if draft_model.config.flashmtp_config is None:
            draft_model.config.flashmtp_config = {}
        draft_model.config.flashmtp_config["local_position"] = lp
        logger.info("Overriding local_position={} (from --local-position)", lp)

    if args.block_size is not None:
        draft_model.set_config_block_size(args.block_size)
        logger.info(
            "Overriding config block_size={} (left_shift={})",
            draft_model.block_size,
            draft_model.left_shift,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=getattr(args, "trust_remote_code", False),
    )
    mask_token_id = resolve_mask_token_id(
        draft_model,
        tokenizer,
        cli_mask_token_id=getattr(args, "mask_token_id", None),
    )
    draft_model.mask_token_id = mask_token_id
    if draft_model.config.flashmtp_config is None:
        draft_model.config.flashmtp_config = {}
    draft_model.config.flashmtp_config["mask_token_id"] = mask_token_id
    logger.info("Using mask_token_id={}", mask_token_id)

    summary = log_flashmtp_config(draft_model)
    validate_decode_config(draft_model)
    return target, draft_model, tokenizer, summary
