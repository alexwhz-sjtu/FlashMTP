"""Shared construction and training helpers for DFlash-family entrypoints."""

from __future__ import annotations

import argparse
import copy
from typing import Optional

import torch

from scripts.dlite.dlite_training import (
    add_common_args,
    build_target_and_components,
    build_train_dataloader,
    hidden_states_to_cuda,
    required_hidden_layer_ids,
    validate_common_args,
)
from specforge.core.dflash_family import OnlineDFlashModel, OnlineDSparkModel
from specforge.modeling.config_utils import load_text_model_config
from specforge.modeling.draft.dflash import DFlashDraftModel, build_target_layer_ids
from specforge.modeling.draft.dflash2 import DFlash2DraftModel
from specforge.modeling.draft.dspark import DSparkDraftModel

FAMILY_ALGORITHMS = ("dflash", "dflash2", "dspark")
MODEL_CLASSES = {
    "dflash": DFlashDraftModel,
    "dflash2": DFlash2DraftModel,
    "dspark": DSparkDraftModel,
}


def add_family_args(parser: argparse.ArgumentParser, algorithm: str) -> None:
    if algorithm not in FAMILY_ALGORITHMS:
        raise ValueError(f"Unsupported DFlash-family algorithm: {algorithm}")
    add_common_args(parser)
    defaults = {"dflash": 16, "dflash2": 16, "dspark": 7}
    parser.set_defaults(block_size=defaults[algorithm], num_draft_layers=5)
    model = parser.add_argument_group(f"{algorithm} architecture")
    model.add_argument(
        "--attention-backend",
        choices=("eager", "sdpa", "flex_attention"),
        default="flex_attention",
    )
    model.add_argument("--attention-mode", choices=("gqa", "mha"), default="gqa")
    model.add_argument("--conv-kernel-size", type=int, default=2)
    model.add_argument("--conv-group-size", type=int, default=16)
    model.add_argument("--selector-rank", type=int, default=256)
    model.add_argument("--selector-top-k", type=int, default=16)
    model.add_argument(
        "--markov-head-type", choices=("vanilla", "gated", "rnn"), default="vanilla"
    )
    model.add_argument("--markov-rank", type=int, default=256)
    model.add_argument(
        "--confidence-head", action=argparse.BooleanOptionalAction, default=True
    )
    model.add_argument(
        "--confidence-head-with-markov",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    model.add_argument(
        "--backbone-conv", action=argparse.BooleanOptionalAction, default=False
    )

    train = parser.add_argument_group(f"{algorithm} objective")
    train.add_argument("--init-from")
    train.add_argument("--num-epochs", type=int, default=6)
    train.add_argument("--learning-rate", type=float, default=6e-4)
    train.add_argument("--warmup-ratio", type=float, default=0.04)
    train.add_argument("--loss-decay-gamma", type=float)
    train.add_argument("--objective-chunk-blocks", type=int, default=128)
    train.add_argument("--selector-loss-alpha", type=float, default=1.0)
    train.add_argument("--selector-warmup-ratio", type=float, default=0.0)
    train.add_argument("--selector-ramp-ratio", type=float, default=0.0)
    train.add_argument("--selector-stop-gradient", action="store_true")
    train.add_argument("--ce-loss-alpha", type=float, default=0.1)
    train.add_argument("--l1-loss-alpha", type=float, default=0.9)
    train.add_argument("--confidence-loss-alpha", type=float, default=1.0)


def validate_family_args(parser: argparse.ArgumentParser, args, algorithm: str) -> None:
    validate_common_args(parser, args)
    if args.resume_from and args.init_from:
        parser.error("--resume-from and --init-from are mutually exclusive")
    if args.num_epochs <= 0 or args.learning_rate <= 0:
        parser.error("--num-epochs and --learning-rate must be positive")
    if not 0 <= args.warmup_ratio <= 1:
        parser.error("--warmup-ratio must be in [0, 1]")
    if args.objective_chunk_blocks < 0:
        parser.error("--objective-chunk-blocks must be non-negative")
    if algorithm == "dflash2":
        if args.conv_kernel_size < 1 or args.conv_kernel_size > args.block_size:
            parser.error("--conv-kernel-size must be in [1, block-size]")
        if args.selector_rank < 1 or args.selector_top_k < 1:
            parser.error("selector rank and top-k must be positive")
        if args.selector_loss_alpha < 0:
            parser.error("--selector-loss-alpha must be non-negative")
        if not 0 <= args.selector_warmup_ratio <= 1:
            parser.error("--selector-warmup-ratio must be in [0, 1]")
        if not 0 <= args.selector_ramp_ratio <= 1:
            parser.error("--selector-ramp-ratio must be in [0, 1]")
    if algorithm == "dspark":
        if min(args.ce_loss_alpha, args.l1_loss_alpha, args.confidence_loss_alpha) < 0:
            parser.error("DSpark loss weights must be non-negative")
        if args.ce_loss_alpha + args.l1_loss_alpha + args.confidence_loss_alpha == 0:
            parser.error("At least one DSpark loss weight must be positive")
        if args.markov_rank < 1:
            parser.error("--markov-rank must be positive")
    args.require_target_last_hidden = algorithm == "dspark" and (
        args.l1_loss_alpha > 0
        or (args.confidence_head and args.confidence_loss_alpha > 0)
    )


def _parse_target_layer_ids(args, config) -> list[int]:
    if args.target_layer_ids:
        ids = [int(value.strip()) for value in args.target_layer_ids.split(",") if value.strip()]
        if ids != sorted(set(ids)):
            raise ValueError("--target-layer-ids must be unique and increasing")
        if not ids or ids[0] < 0 or ids[-1] >= int(config.num_target_layers):
            raise ValueError("--target-layer-ids contains an out-of-range layer")
        return ids
    return build_target_layer_ids(
        int(config.num_target_layers), int(config.num_hidden_layers)
    )


def build_family_config(args, algorithm: str, source_config=None):
    config = (
        load_text_model_config(
            args.target_model_path, trust_remote_code=args.trust_remote_code
        )
        if source_config is None
        else copy.deepcopy(source_config)
    )
    if source_config is None:
        target_depth = int(config.num_hidden_layers)
        config.num_hidden_layers = int(args.num_draft_layers)
        config.num_target_layers = target_depth
    config.block_size = int(args.block_size)
    config.architectures = [MODEL_CLASSES[algorithm].__name__]
    config.auto_map = {
        "AutoModel": f"{algorithm}.{MODEL_CLASSES[algorithm].__name__}"
    }
    config._attn_implementation = args.attention_backend
    config.layer_types = ["full_attention"] * int(config.num_hidden_layers)
    config.sliding_window = None
    config.use_sliding_window = False
    if args.attention_mode == "mha":
        config.num_key_value_heads = int(config.num_attention_heads)
    method = dict(getattr(config, "dflash_config", None) or {})
    method.update(
        {
            "block_size": int(args.block_size),
            "target_layer_ids": _parse_target_layer_ids(args, config),
            "mask_token_id": args.mask_token_id,
            "attention_mode": args.attention_mode,
        }
    )
    if algorithm == "dflash2":
        method.update(
            {
                "conv_kernel_size": int(args.conv_kernel_size),
                "conv_group_size": int(args.conv_group_size),
                "selector_rank": int(args.selector_rank),
                "selector_top_k": int(args.selector_top_k),
            }
        )
    elif algorithm == "dspark":
        method.update(
            {
                "projector_type": "dspark",
                "markov_head_type": args.markov_head_type,
                "markov_rank": int(args.markov_rank),
                "enable_confidence_head": bool(args.confidence_head),
                "confidence_head_with_markov": bool(
                    args.confidence_head_with_markov
                ),
                "confidence_head_alpha": float(args.confidence_loss_alpha),
                "dspark_backbone_conv_enabled": bool(args.backbone_conv),
            }
        )
        if args.backbone_conv:
            method.update(
                {
                    "conv_kernel_size": int(args.conv_kernel_size),
                    "conv_group_size": int(args.conv_group_size),
                }
            )
    config.dflash_config = method
    return config


def build_family_draft(args, algorithm: str, source_config=None):
    model = MODEL_CLASSES[algorithm](
        build_family_config(args, algorithm, source_config=source_config)
    )
    return model.cuda().to(torch.bfloat16)


def load_family_draft(path: str, algorithm: str, attention_backend: str):
    return MODEL_CLASSES[algorithm].from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attention_backend,
    ).cuda()


def sync_args_from_draft(args, draft) -> None:
    method = dict(getattr(draft.config, "dflash_config", None) or {})
    args.block_size = int(draft.block_size)
    args.num_draft_layers = int(draft.config.num_hidden_layers)
    args.target_layer_ids = ",".join(str(value) for value in draft.target_layer_ids)
    args.attention_mode = str(method.get("attention_mode", "gqa"))
    draft.config._attn_implementation = args.attention_backend


def concatenate_target_hidden(hidden_states, target_layer_ids: list[int]) -> torch.Tensor:
    if isinstance(hidden_states, dict):
        selected = [hidden_states[int(layer_id)] for layer_id in target_layer_ids]
    else:
        offset = 1 if len(hidden_states) > max(target_layer_ids) + 1 else 0
        selected = [hidden_states[int(layer_id) + offset] for layer_id in target_layer_ids]
    return torch.cat(selected, dim=-1)


def final_target_hidden(hidden_states, num_target_layers: int) -> torch.Tensor:
    layer_id = int(num_target_layers) - 1
    if isinstance(hidden_states, dict):
        return hidden_states[layer_id]
    offset = 1 if len(hidden_states) == int(num_target_layers) + 1 else 0
    return hidden_states[layer_id + offset]


def build_family_components(args, draft):
    return build_target_and_components(args, [draft])


def build_family_dataloader(args, tokenizer, draft, algorithm: str):
    required_layer_ids = required_hidden_layer_ids([draft])
    if args.require_target_last_hidden:
        required_layer_ids.add(int(draft.config.num_target_layers) - 1)
    return build_train_dataloader(
        args,
        tokenizer,
        train_data_path=args.train_data_path,
        cache_namespace=algorithm,
        num_proc=args.build_dataset_num_proc,
        required_layer_ids=required_layer_ids,
    )


def build_online_model(args, algorithm: str, draft, components, process_group=None):
    common = dict(
        draft_model=draft,
        target_lm_head=components.lm_head,
        target_embed_tokens=components.embed_tokens,
        mask_token_id=int(args.mask_token_id),
        block_size=int(draft.block_size),
        attention_backend=args.attention_backend,
        num_anchors=int(args.num_anchors),
        loss_decay_gamma=args.loss_decay_gamma,
        objective_chunk_blocks=int(args.objective_chunk_blocks),
        process_group=process_group,
    )
    if algorithm == "dspark":
        return OnlineDSparkModel(
            **common,
            dspark_ce_loss_alpha=float(args.ce_loss_alpha),
            dspark_l1_loss_alpha=float(args.l1_loss_alpha),
            dspark_confidence_head_alpha=float(args.confidence_loss_alpha),
        )
    return OnlineDFlashModel(
        **common,
        selector_loss_alpha=(
            float(args.selector_loss_alpha) if algorithm == "dflash2" else 0.0
        ),
        selector_warmup_ratio=float(args.selector_warmup_ratio),
        selector_ramp_ratio=float(args.selector_ramp_ratio),
        selector_stop_gradient=bool(args.selector_stop_gradient),
    )


def prepare_family_features(data, target_output, draft, require_last: bool):
    hidden = (
        hidden_states_to_cuda(data["hidden_states"])
        if target_output is None
        else hidden_states_to_cuda(target_output.hidden_states)
    )
    context = concatenate_target_hidden(hidden, list(draft.target_layer_ids))
    last = (
        final_target_hidden(hidden, int(draft.config.num_target_layers))
        if require_last
        else None
    )
    return context, last


def selector_alpha(args, optimizer_step: int, total_steps: int) -> float:
    target = float(args.selector_loss_alpha)
    warmup = int(total_steps * float(args.selector_warmup_ratio))
    ramp = int(total_steps * float(args.selector_ramp_ratio))
    if optimizer_step < warmup:
        return 0.0
    if ramp > 0 and optimizer_step < warmup + ramp:
        return target * (optimizer_step - warmup + 1) / ramp
    return target


def family_training_metadata(
    args, algorithm: str, optimizer_step: int, total_steps: int
) -> dict:
    """Serializable objective and schedule state stored in every checkpoint."""

    objective = {
        "loss_decay_gamma": args.loss_decay_gamma,
        "objective_chunk_blocks": int(args.objective_chunk_blocks),
    }
    if algorithm == "dflash2":
        objective.update(
            {
                "selector_loss_alpha": float(args.selector_loss_alpha),
                "selector_warmup_ratio": float(args.selector_warmup_ratio),
                "selector_ramp_ratio": float(args.selector_ramp_ratio),
                "selector_stop_gradient": bool(args.selector_stop_gradient),
            }
        )
    elif algorithm == "dspark":
        objective.update(
            {
                "ce_loss_alpha": float(args.ce_loss_alpha),
                "l1_loss_alpha": float(args.l1_loss_alpha),
                "confidence_loss_alpha": float(args.confidence_loss_alpha),
            }
        )
    return {
        "loss_config": objective,
        "selector_schedule_step": int(optimizer_step),
        "selector_effective_alpha": (
            selector_alpha(args, optimizer_step, total_steps)
            if algorithm == "dflash2"
            else 0.0
        ),
    }


__all__ = [
    "FAMILY_ALGORITHMS",
    "MODEL_CLASSES",
    "add_family_args",
    "build_family_components",
    "build_family_config",
    "build_family_dataloader",
    "build_family_draft",
    "build_online_model",
    "concatenate_target_hidden",
    "final_target_hidden",
    "family_training_metadata",
    "load_family_draft",
    "prepare_family_features",
    "selector_alpha",
    "sync_args_from_draft",
    "validate_family_args",
]
