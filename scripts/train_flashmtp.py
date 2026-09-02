#!/usr/bin/env python3
# coding=utf-8
"""FlashMTP Training Script."""

import argparse
import logging
import math
import os
import shutil
import time
import warnings
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from datasets import load_dataset
from specforge.args import SGLangBackendArgs, TrackerArgs
from specforge.checkpoint import (
    distributed_training_state_exists,
    load_distributed_training_state,
    save_distributed_training_state,
)
from specforge.core.flashmtp import OnlineFlashMTPModel
from specforge.data import build_eagle3_dataset, prepare_dp_dataloaders
from specforge.distributed import (
    destroy_distributed,
    get_dp_group,
    get_tp_data_shard,
    init_distributed,
)
from specforge.modeling.draft.flashmtp import (
    FlashMTPDraftModel,
    flashmtp_draft_class_from_config,
    load_flashmtp_draft_model,
    is_gemma4_config,
)
from specforge.modeling.target.flashmtp_target_model import (
    FlashMTPTargetModel,
    get_flashmtp_target_model,
)
from specforge.modeling.target.target_utils import (
    TargetEmbeddingsAndHead,
    load_model_text_config,
)
from specforge.optimizer import BF16Optimizer
from specforge.tracker import create_tracker
from specforge.utils import get_last_checkpoint, print_on_rank0, print_with_rank


def parse_args():
    parser = argparse.ArgumentParser(description="Train FlashMTP Draft Model")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--target-model-path", type=str, required=True)
    model_group.add_argument(
        "--target-model-backend",
        type=str,
        default="hf",
        choices=["sglang", "hf"],
        help="Backend for target model: 'sglang' (service) or 'hf' (local)",
    )
    model_group.add_argument("--draft-config-path", type=str, default=None)
    model_group.add_argument("--block-size", type=int, default=16)
    model_group.add_argument(
        "--left-shift",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use left-shift supervision: block_size is total span (anchor + "
        "block_size-1 drafts). Draft input is 1 anchor + block_size-2 MASKs. "
        "Without it, block_size is the draft block width; slot 0 is unsupervised.",
    )
    model_group.add_argument("--num-draft-layers", type=int, default=1)
    model_group.add_argument(
        "--mask-token-id",
        type=int,
        default=None,
        help="MASK token ID. If not provided, auto-detect from tokenizer.",
    )
    model_group.add_argument(
        "--attention-backend",
        type=str,
        default="flex_attention",
        choices=["eager", "sdpa", "flex_attention"],
        help="Attention backend for draft model.",
    )
    model_group.add_argument(
        "--trust-remote-code", action="store_true", help="Trust remote code"
    )
    model_group.add_argument(
        "--num-anchors",
        type=int,
        default=512,
        help="Number of anchor positions per sequence",
    )
    model_group.add_argument(
        "--pivot-fuse-mode",
        type=str,
        default="linear_fuse",
        choices=["linear_fuse", "attention_fuse", "prefix_condition"],
        help="How to fuse multi-layer teacher pivots (v1.1 ablation).",
    )
    model_group.add_argument(
        "--num-middle-layers-n",
        type=int,
        default=0,
        help="Middle teacher layers between first and last (total selected = 2 + N).",
    )
    model_group.add_argument(
        "--loss-decay-gamma",
        type=float,
        default=None,
        help="Gamma for exponential loss decay weighting (paper Eq.4). "
        "Suggested: 7 for block_size=16, 5 for 10, 4 for 8. None disables.",
    )
    model_group.add_argument(
        "--final-ce-weight",
        type=float,
        default=1.0,
        help="Weight for the final serial-head cross-entropy loss.",
    )
    model_group.add_argument(
        "--final-forward-kl-weight",
        type=float,
        default=0.0,
        help="Weight for forward KL(p_target || q_final). 0 disables.",
    )
    model_group.add_argument(
        "--tv-loss-weight",
        type=float,
        default=1.0,
        help="Weight for serial-head L1 total-variation distribution loss. "
        "The loss is skipped when no serial head is enabled.",
    )
    model_group.add_argument(
        "--base-lm-ce-weight",
        type=float,
        default=0.0,
        help="Weight λ for auxiliary CE on target lm_head(backbone hidden). "
        "Total loss = L_final + λ * L_base. 0 disables.",
    )
    model_group.add_argument(
        "--base-lm-forward-kl-weight",
        type=float,
        default=0.0,
        help="Weight for forward KL(p_target || q_base). 0 disables.",
    )
    model_group.add_argument(
        "--base-lm-ce-decay-gamma",
        type=float,
        default=None,
        help="Separate gamma for exponential decay on base LM CE/forward KL. "
        "None disables decay (uniform weights over valid prediction slots).",
    )
    model_group.add_argument(
        "--markov-head-type",
        type=str,
        default="none",
        choices=["none", "vanilla", "gated", "rnn", "rnn_easy"],
        help="Optional serial head applied after the parallel FlashMTP backbone.",
    )
    model_group.add_argument(
        "--markov-output-mode",
        type=str,
        default="additive",
        choices=["additive", "direct"],
        help="'additive' adds the serial-head logits to base LM-head logits; "
        "'direct' uses serial-head logits as the final logits.",
    )
    model_group.add_argument(
        "--markov-rank",
        type=int,
        default=256,
        help="Low-rank token embedding/state dimension for the serial head.",
    )
    model_group.add_argument(
        "--local-position",
        action="store_true",
        help="Draft uses block-local position ids 1..block_size (repeated per parallel "
        "block in training). CHS rotary prefix uses zeros. Target model still uses global ids.",
    )
    model_group.add_argument(
        "--add-noise",
        action="store_true",
        help="Add uniform noise U(-r, r) to each selected-layer target hidden before draft "
        "forward (default r=0.1 from --target-hidden-noise-ratio).",
    )
    model_group.add_argument(
        "--target-hidden-noise-ratio",
        type=float,
        default=0.1,
        help="Half-width r for uniform noise U(-r, r) when --add-noise is set.",
    )
    model_group.add_argument(
        "--temp-rollout",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replace the temperature-generated suffix labels with deterministic "
        "target greedy rollouts from every sampled anchor. All anchors share "
        "only their immutable true prefix and keep private branch KV.",
    )
    model_group.add_argument(
        "--temp-rollout-projection-chunk-size",
        type=int,
        default=0,
        help="Number of anchor hidden states projected through the frozen target "
        "lm_head at once during each greedy rollout step. 0 projects all "
        "active anchors together (default; no chunking).",
    )

    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument("--train-data-path", type=str, required=True)
    dataset_group.add_argument("--eval-data-path", type=str, default=None)
    dataset_group.add_argument("--chat-template", type=str, default="qwen")
    dataset_group.add_argument("--is-preformatted", action="store_true")
    dataset_group.add_argument("--dataloader-num-workers", type=int, default=8)
    dataset_group.add_argument("--chs-concat-mode", type=str, default="feature")
    dataset_group.add_argument(
        "--build-dataset-num-proc",
        type=int,
        default=int(os.environ.get("SPECFORGE_DATA_NUM_PROC", 8)),
    )

    training_group = parser.add_argument_group("training")
    training_group.add_argument("--num-epochs", type=int, default=6)
    training_group.add_argument("--batch-size", type=int, default=1)
    training_group.add_argument(
        "--shard-draft-by-tp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="After target prefill on the full DP batch, each TP rank trains draft "
        "on its own sample (batch dim 0 / tp_size). Default: on when tp_size > 1.",
    )
    training_group.add_argument("--learning-rate", type=float, default=6e-4)
    training_group.add_argument("--max-length", type=int, default=3072)
    training_group.add_argument("--warmup-ratio", type=float, default=0.04)
    training_group.add_argument("--max-grad-norm", type=float, default=1.0)
    training_group.add_argument("--accumulation-steps", type=int, default=1)
    training_group.add_argument(
        "--ce-chunk-size",
        type=int,
        default=2048,
        help="Chunk size for lm_head + CE/TV to reduce peak activation memory.",
    )
    training_group.add_argument("--seed", type=int, default=42)
    training_group.add_argument("--resume", action="store_true")
    training_group.add_argument(
        "--resume-optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restore Adam optimizer moments from checkpoint. Disable to keep "
        "epoch/step/LR from scheduler only when optimizer state is incompatible.",
    )
    training_group.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="Directory of the checkpoint to resume training from",
    )

    output_group = parser.add_argument_group("output")
    output_group.add_argument("--output-dir", type=str, required=True)
    output_group.add_argument("--cache-dir", type=str, default="./cache/train")
    output_group.add_argument("--log-interval", type=int, default=50)
    output_group.add_argument("--eval-interval", type=int, default=1000)
    output_group.add_argument("--save-interval", type=int, default=1000)

    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="The size of the tensor parallel for the target model",
    )

    tracker_group = parser.add_argument_group("tracker")
    TrackerArgs.add_args(tracker_group)

    dist_group = parser.add_argument_group("distributed")
    dist_group.add_argument("--dist-timeout", type=int, default=30)

    # SGLang specific args
    sglang_group = parser.add_argument_group("sglang backend")
    SGLangBackendArgs.add_args(sglang_group)

    return parser.parse_args()


def _sync_config_layer_types_to_draft_depth(draft_config) -> None:
    """Make ``layer_types`` length match ``num_hidden_layers`` for saved config / attention metadata."""
    if (
        not hasattr(draft_config, "num_hidden_layers")
        or draft_config.num_hidden_layers is None
    ):
        return
    n = int(draft_config.num_hidden_layers)
    lt = getattr(draft_config, "layer_types", None)
    if lt is None:
        draft_config.layer_types = ["full_attention"] * n
        return
    lt = list(lt)
    if len(lt) == n:
        return
    if len(lt) > n:
        draft_config.layer_types = lt[:n]
    else:
        fill = lt[-1] if lt else "full_attention"
        draft_config.layer_types = lt + [fill] * (n - len(lt))


def build_models(args) -> Tuple[FlashMTPTargetModel, FlashMTPDraftModel]:
    """Build target model (backend wrapper) and draft model."""
    if args.markov_rank <= 0:
        raise ValueError(f"--markov-rank must be positive, got {args.markov_rank}.")
    if args.final_ce_weight < 0:
        raise ValueError(
            f"--final-ce-weight must be non-negative, got {args.final_ce_weight}."
        )
    if args.final_forward_kl_weight < 0:
        raise ValueError(
            "--final-forward-kl-weight must be non-negative, got "
            f"{args.final_forward_kl_weight}."
        )
    if args.tv_loss_weight < 0:
        raise ValueError(
            f"--tv-loss-weight must be non-negative, got {args.tv_loss_weight}."
        )
    if args.base_lm_forward_kl_weight < 0:
        raise ValueError(
            "--base-lm-forward-kl-weight must be non-negative, got "
            f"{args.base_lm_forward_kl_weight}."
        )
    if args.markov_head_type == "none" and args.markov_output_mode == "direct":
        raise ValueError(
            f"--markov-output-mode {args.markov_output_mode} requires "
            "--markov-head-type vanilla, gated, rnn, or rnn_easy."
        )
    if args.markov_head_type == "gated" and args.markov_output_mode == "direct":
        raise ValueError(
            "--markov-head-type gated only supports --markov-output-mode additive."
        )

    print_on_rank0(
        f"Loading target model from {args.target_model_path} using {args.target_model_backend} backend"
    )

    target_model_kwargs = {}
    if args.target_model_backend == "sglang":
        target_model_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()

    target_model = get_flashmtp_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend=args.target_model_backend,
        torch_dtype=torch.bfloat16,
        device="cuda" if args.target_model_backend == "hf" else None,
        trust_remote_code=args.trust_remote_code,
        **target_model_kwargs,
    )
    if args.temp_rollout:
        request_capacity = int(target_model.model_runner.req_to_token_pool.size)
        required_requests = int(args.num_anchors) + int(args.batch_size)
        if request_capacity < required_requests:
            raise ValueError(
                "SGLang request pool is too small for temp-rollout: "
                f"capacity={request_capacity}, required>={required_requests}. "
                "Increase --max-running-requests."
            )
        kv_capacity = int(
            target_model.model_runner.token_to_kv_pool_allocator.size
        )
        recommended_kv = int(args.max_length) + int(args.num_anchors) * (
            int(args.block_size) - 1
        )
        print_on_rank0(
            "temp-rollout SGLang capacity: "
            f"requests={request_capacity} (required {required_requests}), "
            f"KV={kv_capacity} tokens (worst-case recommendation "
            f"{recommended_kv})."
        )

    if args.draft_config_path:
        draft_config = AutoConfig.from_pretrained(args.draft_config_path)
        draft_config = getattr(draft_config, "text_config", draft_config)
        target_config = load_model_text_config(args.target_model_path)
        print_on_rank0(f"Loaded draft config from {args.draft_config_path}")
    else:
        target_config = load_model_text_config(args.target_model_path)
        draft_config = load_model_text_config(args.target_model_path)
        print_on_rank0("Auto-generated draft config from target model")

    # Command-line architecture settings are authoritative for both generated
    # and explicit draft configs.
    draft_config.num_hidden_layers = args.num_draft_layers
    draft_config.block_size = args.block_size
    draft_config.num_target_layers = target_config.num_hidden_layers

    if (
        not hasattr(draft_config, "flashmtp_config")
        or draft_config.flashmtp_config is None
    ):
        draft_config.flashmtp_config = {}

    draft_config.flashmtp_config["chs_concat_mode"] = "feature"
    draft_config.flashmtp_config["pivot_fuse_mode"] = args.pivot_fuse_mode
    draft_config.flashmtp_config["num_middle_layers_n"] = args.num_middle_layers_n
    # Structural v2 revision: always prepend the anchor-predecessor raw embedding.
    # Old checkpoints omit this key and retain their legacy layout at evaluation.
    draft_config.flashmtp_config["include_embedding_chs"] = True
    draft_config.flashmtp_config["local_position"] = bool(args.local_position)
    draft_config.flashmtp_config["left_shift"] = bool(args.left_shift)
    draft_config.flashmtp_config["markov_head_type"] = args.markov_head_type
    draft_config.flashmtp_config["markov_output_mode"] = args.markov_output_mode
    draft_config.flashmtp_config["markov_rank"] = int(args.markov_rank)

    draft_config._attn_implementation = args.attention_backend
    print_on_rank0(f"Using attention backend: {args.attention_backend}")

    if is_gemma4_config(draft_config):
        draft_config.layer_types = ["full_attention"] * args.num_draft_layers
    else:
        _sync_config_layer_types_to_draft_depth(draft_config)

    draft_cls = flashmtp_draft_class_from_config(draft_config)
    draft_config.architectures = [draft_cls.__name__]
    draft_model = draft_cls(draft_config).cuda().to(torch.bfloat16)

    capture_layer_ids = list(draft_model.target_layer_ids)
    if (
        args.temp_rollout
        or (args.tv_loss_weight != 0.0 and draft_model.markov_head is not None)
        or args.final_forward_kl_weight > 0.0
        or args.base_lm_forward_kl_weight > 0.0
    ):
        final_target_layer_id = draft_model.config.num_target_layers - 1
        if final_target_layer_id not in capture_layer_ids:
            capture_layer_ids.append(final_target_layer_id)
    target_model.set_capture_layers(capture_layer_ids)

    print_on_rank0(
        f"Draft config: block_size={draft_config.block_size}, "
        f"num_hidden_layers={draft_config.num_hidden_layers}, "
        f"num_target_layers={draft_config.num_target_layers}"
    )
    print_on_rank0(
        f"Draft model parameters: {sum(p.numel() for p in draft_model.parameters()):,}"
    )
    print_on_rank0(
        f"local_position={getattr(draft_model, 'local_position', False)}, "
        f"left_shift={getattr(draft_model, 'left_shift', False)}, "
        f"markov_head_type={draft_model.markov_head_type}, "
        f"markov_output_mode={draft_model.markov_output_mode}, "
        f"markov_rank={draft_model.markov_rank}"
    )

    return target_model, draft_model


def _ensure_embed_vocab_for_mask(
    target_components: TargetEmbeddingsAndHead, mask_token_id: int
) -> None:
    """Expand frozen target embed/lm_head when mask_token_id exceeds loaded vocab."""
    needed = int(mask_token_id) + 1
    cur = target_components.embed_tokens.num_embeddings
    if needed <= cur:
        return

    print_on_rank0(
        f"Expanding target embed/lm_head for mask_token_id={mask_token_id}: "
        f"vocab {cur} -> {needed}"
    )
    old_emb = target_components.embed_tokens
    new_emb = torch.nn.Embedding(
        needed,
        old_emb.embedding_dim,
        padding_idx=old_emb.padding_idx,
        device=old_emb.weight.device,
        dtype=old_emb.weight.dtype,
    )
    with torch.no_grad():
        new_emb.weight[:cur].copy_(old_emb.weight)
        init_row = old_emb.weight.mean(dim=0)
        new_emb.weight[cur:].copy_(init_row.unsqueeze(0).expand(needed - cur, -1))
    target_components.embed_tokens = new_emb

    lm = target_components.lm_head
    if lm.weight.data_ptr() != old_emb.weight.data_ptr():
        new_lm = torch.nn.Linear(
            lm.in_features,
            needed,
            bias=lm.bias is not None,
            device=lm.weight.device,
            dtype=lm.weight.dtype,
        )
        with torch.no_grad():
            new_lm.weight[:cur].copy_(lm.weight)
            init_row = lm.weight.mean(dim=0)
            new_lm.weight[cur:].copy_(init_row.unsqueeze(0).expand(needed - cur, -1))
            if lm.bias is not None:
                new_lm.bias[:cur].copy_(lm.bias)
                new_lm.bias[cur:].zero_()
        target_components.lm_head = new_lm
    else:
        target_components.lm_head.weight = target_components.embed_tokens.weight


def build_dataloader(args, tokenizer) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build train and eval dataloaders."""
    import hashlib

    cache_params_string = (
        f"{args.train_data_path}-"
        f"{args.max_length}-"
        f"{args.chat_template}-"
        f"{args.target_model_path}"
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()

    train_dataset = load_dataset("json", data_files=args.train_data_path)["train"]
    # The on-disk cache shard names carry no rank/node identifier, so every rank
    # would write the same files concurrently on a cold cache and corrupt them.
    # Let global rank 0 build it, then the other ranks just read the finished cache.
    train_build_kwargs = dict(
        dataset=train_dataset,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        is_preformatted=args.is_preformatted,
        cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
        cache_key=cache_key,
        num_proc=args.build_dataset_num_proc,
    )
    if dist.get_rank() == 0:
        train_eagle3_dataset = build_eagle3_dataset(**train_build_kwargs)
    dist.barrier()
    if dist.get_rank() != 0:
        train_eagle3_dataset = build_eagle3_dataset(**train_build_kwargs)

    min_loss_tokens = 2 * args.block_size
    original_size = len(train_eagle3_dataset)
    train_eagle3_dataset = train_eagle3_dataset.filter(
        lambda x: x["loss_mask"].sum() >= min_loss_tokens
    )
    print_on_rank0(
        f"Filtered train dataset: {original_size} -> {len(train_eagle3_dataset)} samples"
    )

    train_dataloader = prepare_dp_dataloaders(
        train_eagle3_dataset,
        args.batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        process_group=get_dp_group(),
    )

    eval_dataloader = None
    if args.eval_data_path:
        eval_dataset = load_dataset("json", data_files=args.eval_data_path)["train"]
        # Same constraint as the train build: datasets' auto-named map cache files
        # are identical across ranks, so serialize the build behind rank 0.
        eval_build_kwargs = dict(
            dataset=eval_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            is_preformatted=args.is_preformatted,
        )
        if dist.get_rank() == 0:
            eval_eagle3_dataset = build_eagle3_dataset(**eval_build_kwargs)
        dist.barrier()
        if dist.get_rank() != 0:
            eval_eagle3_dataset = build_eagle3_dataset(**eval_build_kwargs)
        eval_dataloader = prepare_dp_dataloaders(
            eval_eagle3_dataset,
            args.batch_size,
            num_workers=args.dataloader_num_workers,
            shuffle=False,
            process_group=get_dp_group(),
        )

    return train_dataloader, eval_dataloader


def resolve_training_state_dir(checkpoint_dir: str) -> Optional[str]:
    """Prefer an epoch checkpoint containing training state over a flat export."""
    # get_last_checkpoint returns (path, (epoch, step)) on hit but
    # (None, None, None) when no epoch_*_step_* subdirs exist (flat export).
    epoch_ckpt = get_last_checkpoint(checkpoint_dir)[0]
    if epoch_ckpt is not None and distributed_training_state_exists(epoch_ckpt):
        return epoch_ckpt

    if distributed_training_state_exists(checkpoint_dir):
        return checkpoint_dir
    return None


def save_checkpoint(args, epoch, step, flashmtp_model, draft_model, optimizer):
    """Save checkpoint."""
    save_dir = os.path.join(args.output_dir, f"epoch_{epoch}_step_{step}")
    if dist.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)
    dist.barrier()

    with FSDP.state_dict_type(flashmtp_model, StateDictType.FULL_STATE_DICT):
        state_dict = flashmtp_model.state_dict()
        draft_state_dict = {
            k.replace("draft_model.", ""): v
            for k, v in state_dict.items()
            if "draft_model." in k
        }

        optimizer_state = optimizer.state_dict()
        save_distributed_training_state(
            save_dir,
            {
                "epoch": epoch,
                "global_step": step,
                "args": args,
                **optimizer_state,
            },
        )

        if dist.get_rank() == 0:
            draft_model.save_pretrained(save_dir, state_dict=draft_state_dict)

            modeling_src = os.path.join(
                os.path.dirname(__file__),
                "..",
                "specforge",
                "modeling",
                "draft",
                "flashmtp.py",
            )
            modeling_dst = os.path.join(save_dir, "flashmtp.py")
            if os.path.exists(modeling_src):
                shutil.copy(modeling_src, modeling_dst)
            markov_src = os.path.join(
                os.path.dirname(__file__),
                "..",
                "specforge",
                "modeling",
                "draft",
                "flashmtp_markov_head.py",
            )
            markov_dst = os.path.join(save_dir, "flashmtp_markov_head.py")
            if os.path.exists(markov_src):
                shutil.copy(markov_src, markov_dst)

            print_on_rank0(f"Saved checkpoint to {save_dir}")

    dist.barrier()


def record_metrics(
    args,
    loss: float,
    accuracy: float,
    global_step: int,
    tracker,
    optimizer,
    train_dataloader=None,
    mode: str = "train",
    prefix_acc: float | None = None,
    final_ce_loss: float | None = None,
    base_lm_ce_loss: float | None = None,
    tv_loss: float | None = None,
    final_forward_kl_loss: float | None = None,
    base_lm_forward_kl_loss: float | None = None,
    grad_norm: float | None = None,
) -> None:
    logdict = {}

    if mode == "train" and optimizer is not None:
        logdict["train/lr"] = optimizer.get_learning_rate()

    logdict[f"{mode}/loss"] = loss
    logdict[f"{mode}/accuracy"] = accuracy
    if prefix_acc is not None:
        logdict[f"{mode}/prefix_acc"] = prefix_acc
    if final_ce_loss is not None:
        logdict[f"{mode}/final_ce_loss"] = final_ce_loss
    if base_lm_ce_loss is not None:
        logdict[f"{mode}/base_lm_ce_loss"] = base_lm_ce_loss
    if tv_loss is not None:
        logdict[f"{mode}/tv_loss"] = tv_loss
    if final_forward_kl_loss is not None:
        logdict[f"{mode}/final_forward_kl_loss"] = final_forward_kl_loss
    if base_lm_forward_kl_loss is not None:
        logdict[f"{mode}/base_lm_forward_kl_loss"] = base_lm_forward_kl_loss
    if grad_norm is not None:
        logdict[f"{mode}/grad_norm"] = grad_norm

    extra = ""
    if prefix_acc is not None:
        extra = f", PrefixAcc: {prefix_acc:.4f}"
    if final_ce_loss is not None:
        extra += f", FinalCE: {final_ce_loss:.4f}"
    if base_lm_ce_loss is not None:
        extra += f", BaseCE: {base_lm_ce_loss:.4f}"
    if tv_loss is not None:
        extra += f", TV: {tv_loss:.4f}"
    if final_forward_kl_loss is not None:
        extra += f", FinalFKL: {final_forward_kl_loss:.4f}"
    if base_lm_forward_kl_loss is not None:
        extra += f", BaseFKL: {base_lm_forward_kl_loss:.4f}"
    if grad_norm is not None:
        extra += f", GradNorm: {grad_norm:.4f}"
    print_on_rank0(
        f"{mode.capitalize()} - Step {global_step} [{global_step}/{args.num_epochs * len(train_dataloader) // args.accumulation_steps}?], Loss: {loss:.4f}, Acc: {accuracy:.4f}{extra}"
    )

    tracker.log(logdict, step=global_step)


def main():

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging.getLogger().setLevel(logging.INFO)
    warnings.filterwarnings(
        "ignore",
        "The .grad attribute of a Tensor that is not a leaf Tensor is being accessed",
    )

    args = parse_args()
    set_seed(args.seed)

    if args.temp_rollout:
        if args.target_model_backend != "sglang":
            raise ValueError("--temp-rollout requires --target-model-backend sglang.")
        if args.batch_size != 1:
            raise ValueError(
                "--temp-rollout currently requires --batch-size 1; anchors inside "
                "the sample are fully batched."
            )
        if args.temp_rollout_projection_chunk_size < 0:
            raise ValueError(
                "--temp-rollout-projection-chunk-size must be non-negative; "
                "use 0 to disable chunking."
            )

    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    print_with_rank("Initialized distributed")

    if args.shard_draft_by_tp is None:
        args.shard_draft_by_tp = args.tp_size > 1
    if args.shard_draft_by_tp:
        if args.tp_size == 1:
            args.shard_draft_by_tp = False
        elif args.batch_size % args.tp_size != 0:
            raise ValueError(
                f"batch_size ({args.batch_size}) must be divisible by tp_size "
                f"({args.tp_size}) when --shard-draft-by-tp is enabled."
            )
    if args.shard_draft_by_tp:
        print_on_rank0(
            f"shard-draft-by-tp: target batch={args.batch_size} per DP rank "
            f"({args.batch_size // args.tp_size} draft sample per TP rank), "
            f"global unique samples/step="
            f"{args.batch_size * dist.get_world_size(get_dp_group())}"
        )
    else:
        print_on_rank0(
            "shard-draft-by-tp disabled: all TP ranks train draft on the full batch."
        )

    args.target_batch_size = args.batch_size
    target_model, draft_model = build_models(args)

    draft_model_last_checkpoint = None
    if args.ckpt_dir is not None:
        if os.path.isdir(args.ckpt_dir):
            draft_model_last_checkpoint = args.ckpt_dir
            print_on_rank0(f"Using checkpoint: {draft_model_last_checkpoint}")
        else:
            raise ValueError(
                f"Provided ckpt dir {args.ckpt_dir} is not a valid directory."
            )

    # An explicit checkpoint is authoritative. Without --ckpt-dir, --resume
    # discovers the latest epoch_*_step_* directory under output_dir.
    if args.resume and args.ckpt_dir is None and os.path.isdir(args.output_dir):
        draft_model_last_checkpoint, ckpt_info = get_last_checkpoint(args.output_dir)
        print_on_rank0(f"Last checkpoint detected: {draft_model_last_checkpoint}")

    resume_state = None
    draft_weights_from_checkpoint = False
    if draft_model_last_checkpoint:
        loaded_model = load_flashmtp_draft_model(
            draft_model_last_checkpoint, torch_dtype=torch.bfloat16
        )
        requested_markov = (
            draft_model.markov_head_type,
            draft_model.markov_output_mode,
            draft_model.markov_rank,
        )
        checkpoint_markov = (
            loaded_model.markov_head_type,
            loaded_model.markov_output_mode,
            loaded_model.markov_rank,
        )
        if requested_markov != checkpoint_markov:
            raise ValueError(
                "Checkpoint serial-head configuration does not match the "
                "current training arguments: "
                f"requested={requested_markov}, checkpoint={checkpoint_markov}."
            )
        checkpoint_left_shift = bool(loaded_model.left_shift)
        if checkpoint_left_shift != bool(args.left_shift):
            raise ValueError(
                "Checkpoint left_shift configuration does not match the "
                "current training arguments: "
                f"requested={bool(args.left_shift)}, "
                f"checkpoint={checkpoint_left_shift}."
            )
        draft_model.load_state_dict(loaded_model.state_dict())
        draft_model.left_shift = checkpoint_left_shift
        if draft_model.config.flashmtp_config is None:
            draft_model.config.flashmtp_config = {}
        draft_model.config.flashmtp_config["left_shift"] = checkpoint_left_shift
        del loaded_model
        draft_weights_from_checkpoint = True
        print_on_rank0("Loaded draft model weights from checkpoint")

        training_state_dir = resolve_training_state_dir(draft_model_last_checkpoint)
        if training_state_dir is not None:
            resume_state = load_distributed_training_state(
                training_state_dir, map_location="cpu"
            )
            print_on_rank0(f"Loading training state from {training_state_dir}")
            print_on_rank0(
                f"Will resume from epoch {resume_state['epoch']}, "
                f"step {resume_state['global_step']}"
            )

    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)

    if args.mask_token_id is not None:
        mask_token_id = args.mask_token_id
    elif tokenizer.mask_token_id is not None:
        mask_token_id = tokenizer.mask_token_id
    else:
        tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
        mask_token_id = tokenizer.mask_token_id

    print_on_rank0(
        f"****** Important: Make sure using the same mask_token_id with inference.***** \n Using mask_token_id: {mask_token_id} \n"
    )

    draft_model.mask_token_id = mask_token_id

    draft_model.config.flashmtp_config["chs_concat_mode"] = "feature"
    draft_model.config.flashmtp_config["mask_token_id"] = mask_token_id
    draft_model.config.flashmtp_config["target_layer_ids"] = (
        draft_model.target_layer_ids
    )
    draft_model.config.flashmtp_config["pivot_fuse_mode"] = draft_model.pivot_fuse_mode
    draft_model.config.flashmtp_config["num_middle_layers_n"] = (
        draft_model.num_middle_layers_n
    )
    draft_model.config.flashmtp_config["include_embedding_chs"] = bool(
        draft_model.include_embedding_chs
    )
    draft_model.config.flashmtp_config["local_position"] = bool(
        getattr(draft_model, "local_position", False)
    )
    draft_model.config.flashmtp_config["left_shift"] = bool(
        getattr(draft_model, "left_shift", False)
    )
    draft_model.config.flashmtp_config["markov_head_type"] = (
        draft_model.markov_head_type
    )
    draft_model.config.flashmtp_config["markov_output_mode"] = (
        draft_model.markov_output_mode
    )
    draft_model.config.flashmtp_config["markov_rank"] = int(draft_model.markov_rank)
    draft_model.config.flashmtp_config["add_noise"] = bool(args.add_noise)
    draft_model.config.flashmtp_config["target_hidden_noise_ratio"] = float(
        args.target_hidden_noise_ratio
    )
    draft_model.config.flashmtp_config["temp_rollout"] = bool(args.temp_rollout)
    print_on_rank0(f"flashmtp_config: {draft_model.config.flashmtp_config}")

    train_dataloader, eval_dataloader = build_dataloader(args, tokenizer)

    steps_per_epoch = math.ceil(len(train_dataloader) / args.accumulation_steps)
    total_steps = args.num_epochs * steps_per_epoch
    print_on_rank0(f"Total training steps: {total_steps}")
    if total_steps <= 0:
        raise ValueError(
            f"total_steps must be positive, got {total_steps}. "
            f"train_dataloader len={len(train_dataloader)}, "
            f"accumulation_steps={args.accumulation_steps}, num_epochs={args.num_epochs}. "
            "Check TRAIN_DATA_PATH / CHAT_TEMPLATE / loss_mask filter."
        )

    print_on_rank0("Loading target embeddings and head...")
    target_components = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model_path,
        embed_key=None,
        lm_head_key=None,
        device="cuda",
        trust_remote_code=args.trust_remote_code,
    )
    target_vocab_size = int(target_components.lm_head.weight.shape[0])
    _ensure_embed_vocab_for_mask(target_components, mask_token_id)

    flashmtp_model = OnlineFlashMTPModel(
        draft_model=draft_model,
        target_lm_head=target_components.lm_head,
        target_embed_tokens=target_components.embed_tokens,
        block_size=draft_model.block_size,
        mask_token_id=mask_token_id,
        attention_backend=args.attention_backend,
        num_anchors=args.num_anchors,
        loss_decay_gamma=args.loss_decay_gamma,
        final_ce_weight=args.final_ce_weight,
        final_forward_kl_weight=args.final_forward_kl_weight,
        tv_loss_weight=args.tv_loss_weight,
        base_lm_ce_weight=args.base_lm_ce_weight,
        base_lm_forward_kl_weight=args.base_lm_forward_kl_weight,
        base_lm_ce_decay_gamma=args.base_lm_ce_decay_gamma,
        chs_concat_mode="feature",
        add_noise=args.add_noise,
        target_hidden_noise_ratio=args.target_hidden_noise_ratio,
        ce_chunk_size=args.ce_chunk_size,
        left_shift=args.left_shift,
        temp_rollout_enabled=args.temp_rollout,
        temp_rollout_projection_chunk_size=(
            args.temp_rollout_projection_chunk_size
        ),
        target_vocab_size=target_vocab_size,
        eos_token_id=tokenizer.eos_token_id,
    )
    print_on_rank0(
        f"target hidden noise: add_noise={args.add_noise}, "
        f"ratio={args.target_hidden_noise_ratio}, "
        f"ce_chunk_size={args.ce_chunk_size}, "
        f"final_ce_weight={args.final_ce_weight}, "
        f"final_forward_kl_weight={args.final_forward_kl_weight}, "
        f"tv_loss_weight={args.tv_loss_weight}, "
        f"base_lm_ce_weight={args.base_lm_ce_weight}, "
        f"base_lm_forward_kl_weight={args.base_lm_forward_kl_weight}, "
        f"base_lm_ce_decay_gamma={args.base_lm_ce_decay_gamma}"
        f", left_shift={args.left_shift}, temp_rollout={args.temp_rollout}, "
        f"temp_rollout_projection_chunk_size="
        f"{args.temp_rollout_projection_chunk_size}"
    )

    online_flashmtp = flashmtp_model
    flashmtp_model = FSDP(
        flashmtp_model,
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    )
    print_with_rank("Initialized FSDP")

    optimizer = BF16Optimizer(
        draft_model,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=total_steps,
    )
    print_on_rank0(
        "Gradient clipping enabled: global L2 norm, "
        f"max_grad_norm={args.max_grad_norm}."
    )
    skip_steps = 0
    start_epoch = 0
    global_step = 0
    if resume_state is not None:
        loaded_optimizer = optimizer.load_state_dict(
            {
                "optimizer_state_dict": resume_state["optimizer_state_dict"],
                "scheduler_state_dict": resume_state["scheduler_state_dict"],
            },
            load_optimizer=args.resume_optimizer,
        )
        start_epoch = resume_state["epoch"]
        global_step = resume_state["global_step"]
        del resume_state
        if loaded_optimizer:
            print_on_rank0(
                f"Restored optimizer and scheduler, lr={optimizer.get_learning_rate():.6f}"
            )
        else:
            print_on_rank0(
                f"Restored scheduler only, lr={optimizer.get_learning_rate():.6f}"
            )

        skip_steps = global_step - start_epoch * len(train_dataloader)

    print_on_rank0(f"Initializing tracker (report_to={args.report_to})...")
    tracker = create_tracker(args, args.output_dir)
    print_on_rank0("Tracker initialized successfully.")

    last_time = time.time()
    accumulated_micro_steps = 0
    checkpoint_pending = False
    last_grad_norm = None
    print_on_rank0(f"Starting training from epoch {start_epoch}, step {global_step}")

    for epoch in range(start_epoch, args.num_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        draft_model.train()

        if dist.get_rank() == 0:
            progress_bar = tqdm(
                train_dataloader, desc=f"Training Epoch {epoch}", leave=True
            )
        else:
            progress_bar = train_dataloader

        for step_in_epoch, data in enumerate(progress_bar):
            if epoch == start_epoch and step_in_epoch < skip_steps:
                continue
            global_step += 1
            checkpoint_pending = (
                checkpoint_pending or global_step % args.save_interval == 0
            )

            input_ids = data["input_ids"].cuda()
            attention_mask = data["attention_mask"].cuda()
            loss_mask = data["loss_mask"].cuda()

            temp_rollout_handle = None
            if args.temp_rollout:
                prefill_output = target_model.temp_rollout_prefill(
                    input_ids, attention_mask
                )
                temp_rollout_handle = prefill_output.handle
                hidden_states = prefill_output.hidden_states
            else:
                # Baseline: target output is the full true sequence and KV is
                # released immediately by the target backend.
                target_output = target_model.generate_flashmtp_data(
                    input_ids, attention_mask, loss_mask
                )
                hidden_states = target_output.hidden_states
            if isinstance(hidden_states, dict):
                hidden_states = {
                    layer_id: h.cuda() if not h.is_cuda else h
                    for layer_id, h in hidden_states.items()
                }
            else:
                hidden_states = tuple(
                    h.cuda() if not h.is_cuda else h for h in hidden_states
                )

            seq_len = int(input_ids.shape[1])
            if dist.get_rank() == 0 and global_step <= 10:
                print_on_rank0(
                    f"step {global_step}: seq_len={seq_len}, "
                    f"num_captured_layers="
                    f"{len(hidden_states) if isinstance(hidden_states, dict) else len(hidden_states)}"
                )

            if args.temp_rollout:
                (
                    anchor_positions,
                    block_keep_mask,
                    target_hidden,
                    target_anchor_hidden,
                ) = online_flashmtp.prepare_temp_rollout_tensors(
                    input_ids, hidden_states, loss_mask
                )
                target_prediction_hidden = None
                del prefill_output
            else:
                (
                    anchor_positions,
                    block_keep_mask,
                    target_hidden,
                    target_prediction_hidden,
                ) = online_flashmtp.prepare_training_tensors(
                    input_ids, hidden_states, loss_mask
                )
                target_anchor_hidden = None
                del target_output
            del hidden_states

            if args.shard_draft_by_tp:
                input_ids = get_tp_data_shard(input_ids)
                loss_mask = get_tp_data_shard(loss_mask)
                anchor_positions = get_tp_data_shard(anchor_positions)
                block_keep_mask = get_tp_data_shard(block_keep_mask)
                target_hidden = get_tp_data_shard(target_hidden)
                if target_anchor_hidden is not None:
                    target_anchor_hidden = get_tp_data_shard(
                        target_anchor_hidden
                    )
                if target_prediction_hidden is not None:
                    target_prediction_hidden = get_tp_data_shard(
                        target_prediction_hidden
                    )

            try:
                (
                    loss,
                    accuracy,
                    prefix_acc,
                    final_ce_loss,
                    base_ce_loss,
                    tv_loss,
                    final_forward_kl_loss,
                    base_forward_kl_loss,
                ) = flashmtp_model(
                    input_ids=input_ids,
                    loss_mask=loss_mask,
                    anchor_positions=anchor_positions,
                    block_keep_mask=block_keep_mask,
                    target_hidden=target_hidden,
                    target_prediction_hidden=target_prediction_hidden,
                    target_anchor_hidden=target_anchor_hidden,
                    temp_rollout_context=temp_rollout_handle,
                )
            finally:
                if temp_rollout_handle is not None:
                    temp_rollout_handle.close()
            del (
                target_hidden,
                target_prediction_hidden,
                target_anchor_hidden,
                anchor_positions,
                block_keep_mask,
            )

            (loss / args.accumulation_steps).backward()
            accumulated_micro_steps += 1

            if accumulated_micro_steps == args.accumulation_steps:
                last_grad_norm = optimizer.step()
                accumulated_micro_steps = 0

            if global_step % args.log_interval == 0:
                loss_log = loss.clone()
                acc_log = accuracy.clone()
                pfx_log = prefix_acc.clone()
                final_ce_log = final_ce_loss.clone()
                base_ce_log = base_ce_loss.clone()
                tv_loss_log = tv_loss.clone()
                final_forward_kl_log = final_forward_kl_loss.clone()
                base_forward_kl_log = base_forward_kl_loss.clone()
                dist.all_reduce(loss_log)
                dist.all_reduce(acc_log)
                dist.all_reduce(pfx_log)
                dist.all_reduce(final_ce_log)
                dist.all_reduce(base_ce_log)
                dist.all_reduce(tv_loss_log)
                dist.all_reduce(final_forward_kl_log)
                dist.all_reduce(base_forward_kl_log)
                loss_log = loss_log / dist.get_world_size()
                acc_log = acc_log / dist.get_world_size()
                pfx_log = pfx_log / dist.get_world_size()
                final_ce_log = final_ce_log / dist.get_world_size()
                base_ce_log = base_ce_log / dist.get_world_size()
                tv_loss_log = tv_loss_log / dist.get_world_size()
                final_forward_kl_log = (
                    final_forward_kl_log / dist.get_world_size()
                )
                base_forward_kl_log = base_forward_kl_log / dist.get_world_size()

                record_metrics(
                    args,
                    loss_log.item(),
                    acc_log.item(),
                    global_step,
                    tracker,
                    optimizer,
                    train_dataloader,
                    mode="train",
                    prefix_acc=pfx_log.item(),
                    final_ce_loss=final_ce_log.item(),
                    base_lm_ce_loss=base_ce_log.item(),
                    tv_loss=tv_loss_log.item(),
                    final_forward_kl_loss=final_forward_kl_log.item(),
                    base_lm_forward_kl_loss=base_forward_kl_log.item(),
                    grad_norm=last_grad_norm,
                )

            if dist.get_rank() == 0:
                elapsed = time.time() - last_time
                last_time = time.time()
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{accuracy.item():.4f}",
                        "pfx": f"{prefix_acc.item():.4f}",
                        "final_ce": f"{final_ce_loss.item():.4f}",
                        "tv": f"{tv_loss.item():.4f}",
                        "final_fkl": f"{final_forward_kl_loss.item():.4f}",
                        "base_fkl": f"{base_forward_kl_loss.item():.4f}",
                        "iter_time": f"{elapsed:.2f}s",
                    }
                )

            if checkpoint_pending and accumulated_micro_steps == 0:
                save_checkpoint(
                    args, epoch, global_step, flashmtp_model, draft_model, optimizer
                )
                checkpoint_pending = False

        # Do not silently drop a short gradient-accumulation window at epoch end.
        if accumulated_micro_steps > 0:
            optimizer.scale_model_gradients(
                args.accumulation_steps / accumulated_micro_steps
            )
            last_grad_norm = optimizer.step()
            accumulated_micro_steps = 0

        if checkpoint_pending:
            save_checkpoint(
                args, epoch, global_step, flashmtp_model, draft_model, optimizer
            )
            checkpoint_pending = False

    save_checkpoint(
        args, args.num_epochs, global_step, flashmtp_model, draft_model, optimizer
    )

    tracker.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
