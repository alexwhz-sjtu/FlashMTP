"""Target/draft loading for DFlash-family benchmarks."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.dflash_family.dflash_family_training import MODEL_CLASSES


def load_dflash_benchmark_models(args, device):
    try:
        import flash_attn  # noqa: F401

        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
    target = (
        AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            attn_implementation=attn_impl,
            dtype=torch.bfloat16,
            trust_remote_code=args.trust_remote_code,
        )
        .to(device)
        .eval()
    )
    draft = (
        MODEL_CLASSES[args.algorithm].from_pretrained(
            args.draft_name_or_path,
            attn_implementation=attn_impl,
            dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=args.trust_remote_code
    )
    method = dict(getattr(draft.config, "dflash_config", None) or {})
    mask_id = args.mask_token_id
    if mask_id is None:
        mask_id = method.get("mask_token_id", tokenizer.mask_token_id)
    if mask_id is None:
        raise ValueError("DFlash-family benchmark requires an in-vocabulary mask token")
    draft.mask_token_id = int(mask_id)
    method["mask_token_id"] = int(mask_id)
    draft.config.dflash_config = method
    return target, draft, tokenizer


__all__ = ["load_dflash_benchmark_models"]
