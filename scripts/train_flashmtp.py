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
from specforge.core.flashmtp import OnlineFlashMTPModel
from specforge.data import build_eagle3_dataset, prepare_dp_dataloaders
from specforge.distributed import (
    destroy_distributed,
    get_dp_group,
    get_tp_data_shard,
    init_distributed,
)
from specforge.modeling.draft.flashmtp import FlashMTPDraftModel
from specforge.modeling.target.flashmtp_target_model import (
    FlashMTPTargetModel,
    get_flashmtp_target_model,
)
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
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
        "--train-lm-head",
        action="store_true",
        help="Add a trainable draft lm_head (init from target head); share only frozen "
        "embeddings with the target. Default: share frozen target lm_head as today.",
    )
    model_group.add_argument(
        "--markov-head-type",
        type=str,
        default="none",
        choices=["none", "vanilla", "gated", "rnn"],
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
        "--w1-mse",
        type=float,
        default=0.0,
        help="Weight for MSE between draft last-layer hidden and target last-layer "
        "hidden at the first predicted token (block position 1). 0 disables.",
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
        help="Chunk size for lm_head + CE to reduce peak activation memory.",
    )
    training_group.add_argument("--seed", type=int, default=42)
    training_group.add_argument("--resume", action="store_true")
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
    if args.markov_head_type == "none" and args.markov_output_mode == "direct":
        raise ValueError(
            "--markov-output-mode direct requires --markov-head-type "
            "vanilla, gated, or rnn."
        )
    if args.markov_output_mode == "direct" and args.train_lm_head:
        raise ValueError(
            "--train-lm-head cannot be used with --markov-output-mode direct "
            "because the draft LM head is bypassed."
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

    if args.draft_config_path:
        draft_config = AutoConfig.from_pretrained(args.draft_config_path)
        print_on_rank0(f"Loaded draft config from {args.draft_config_path}")
    else:
        target_config = AutoConfig.from_pretrained(args.target_model_path)
        draft_config = AutoConfig.from_pretrained(args.target_model_path)
        draft_config.num_hidden_layers = args.num_draft_layers
        draft_config.block_size = args.block_size
        draft_config.num_target_layers = target_config.num_hidden_layers
        print_on_rank0("Auto-generated draft config from target model")

    if (
        not hasattr(draft_config, "flashmtp_config")
        or draft_config.flashmtp_config is None
    ):
        draft_config.flashmtp_config = {}

    draft_config.flashmtp_config["chs_concat_mode"] = "feature"
    draft_config.flashmtp_config["pivot_fuse_mode"] = args.pivot_fuse_mode
    draft_config.flashmtp_config["num_middle_layers_n"] = args.num_middle_layers_n
    draft_config.flashmtp_config["local_position"] = bool(args.local_position)
    draft_config.flashmtp_config["markov_head_type"] = args.markov_head_type
    draft_config.flashmtp_config["markov_output_mode"] = args.markov_output_mode
    draft_config.flashmtp_config["markov_rank"] = int(args.markov_rank)
    if args.train_lm_head:
        draft_config.flashmtp_config["train_lm_head"] = True
    elif "train_lm_head" not in draft_config.flashmtp_config:
        draft_config.flashmtp_config["train_lm_head"] = False

    draft_config._attn_implementation = args.attention_backend
    print_on_rank0(f"Using attention backend: {args.attention_backend}")

    _sync_config_layer_types_to_draft_depth(draft_config)

    draft_model = FlashMTPDraftModel(draft_config).cuda().to(torch.bfloat16)

    target_model.set_capture_layers(draft_model.target_layer_ids)

    print_on_rank0(
        f"Draft config: block_size={draft_config.block_size}, "
        f"num_hidden_layers={draft_config.num_hidden_layers}, "
        f"num_target_layers={draft_config.num_target_layers}"
    )
    print_on_rank0(
        f"Draft model parameters: {sum(p.numel() for p in draft_model.parameters()):,}"
    )
    print_on_rank0(
        f"train_lm_head={getattr(draft_model, 'train_lm_head', False)} "
        f"(draft_lm_head={'on' if draft_model.draft_lm_head is not None else 'off'}), "
        f"local_position={getattr(draft_model, 'local_position', False)}, "
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
    train_eagle3_dataset = build_eagle3_dataset(
        dataset=train_dataset,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        is_preformatted=args.is_preformatted,
        cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
        cache_key=cache_key,
        num_proc=args.build_dataset_num_proc,
    )

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
        eval_eagle3_dataset = build_eagle3_dataset(
            dataset=eval_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            is_preformatted=args.is_preformatted,
        )
        eval_dataloader = prepare_dp_dataloaders(
            eval_eagle3_dataset,
            args.batch_size,
            num_workers=args.dataloader_num_workers,
            shuffle=False,
            process_group=get_dp_group(),
        )

    return train_dataloader, eval_dataloader


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

        if dist.get_rank() == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": step,
                    "args": args,
                    **optimizer.state_dict(),
                },
                os.path.join(save_dir, "training_state.pt"),
            )

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
    mse_loss: float | None = None,
) -> None:
    logdict = {}

    if mode == "train" and optimizer is not None:
        logdict["train/lr"] = optimizer.get_learning_rate()

    logdict[f"{mode}/loss"] = loss
    logdict[f"{mode}/accuracy"] = accuracy
    if prefix_acc is not None:
        logdict[f"{mode}/prefix_acc"] = prefix_acc
    if mse_loss is not None:
        logdict[f"{mode}/w1_mse_loss"] = mse_loss

    extra = ""
    if prefix_acc is not None:
        extra = f", PrefixAcc: {prefix_acc:.4f}"
    if mse_loss is not None:
        extra += f", W1MSE: {mse_loss:.4f}"
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

    if args.resume and os.path.isdir(args.output_dir):
        draft_model_last_checkpoint, ckpt_info = get_last_checkpoint(
            args.output_dir, prefix=r"epoch_\d+_step"
        )
        print_on_rank0(f"Last checkpoint detected: {draft_model_last_checkpoint}")

    resume_state = None
    draft_weights_from_checkpoint = False
    if draft_model_last_checkpoint:
        loaded_model = FlashMTPDraftModel.from_pretrained(
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
        draft_model.load_state_dict(loaded_model.state_dict())
        del loaded_model
        draft_weights_from_checkpoint = True
        print_on_rank0("Loaded draft model weights from checkpoint")

        training_state_path = os.path.join(
            draft_model_last_checkpoint, "training_state.pt"
        )
        if os.path.exists(training_state_path):
            resume_state = torch.load(
                training_state_path, map_location="cpu", weights_only=False
            )
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
    draft_model.config.flashmtp_config["train_lm_head"] = bool(
        getattr(draft_model, "train_lm_head", False)
    )
    draft_model.config.flashmtp_config["local_position"] = bool(
        getattr(draft_model, "local_position", False)
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
    draft_model.config.flashmtp_config["w1_mse"] = float(args.w1_mse)
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
        embed_key="model.embed_tokens.weight",  # Adjust if Qwen/Llama differs
        lm_head_key="lm_head.weight",
        device="cuda",
        trust_remote_code=args.trust_remote_code,
    )
    _ensure_embed_vocab_for_mask(target_components, mask_token_id)

    if draft_model.draft_lm_head is not None:
        if not draft_weights_from_checkpoint:
            with torch.no_grad():
                draft_model.draft_lm_head.weight.copy_(
                    target_components.lm_head.weight.to(
                        device=draft_model.draft_lm_head.weight.device,
                        dtype=draft_model.draft_lm_head.weight.dtype,
                    )
                )
            print_on_rank0(
                "Initialized draft_lm_head from target lm_head (trainable; embeddings stay shared/frozen)."
            )
        else:
            print_on_rank0(
                "draft_lm_head: using weights from checkpoint (skip copy from target)."
            )

    flashmtp_model = OnlineFlashMTPModel(
        draft_model=draft_model,
        target_lm_head=target_components.lm_head,
        target_embed_tokens=target_components.embed_tokens,
        block_size=draft_model.block_size,
        mask_token_id=mask_token_id,
        attention_backend=args.attention_backend,
        num_anchors=args.num_anchors,
        loss_decay_gamma=args.loss_decay_gamma,
        chs_concat_mode="feature",
        add_noise=args.add_noise,
        target_hidden_noise_ratio=args.target_hidden_noise_ratio,
        w1_mse=args.w1_mse,
        ce_chunk_size=args.ce_chunk_size,
    )
    print_on_rank0(
        f"target hidden noise: add_noise={args.add_noise}, "
        f"ratio={args.target_hidden_noise_ratio}, w1_mse={args.w1_mse}, "
        f"ce_chunk_size={args.ce_chunk_size}"
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
    skip_steps = 0
    start_epoch = 0
    global_step = 0
    if resume_state is not None:
        optimizer.scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        start_epoch = resume_state["epoch"]
        global_step = resume_state["global_step"]
        del resume_state
        print_on_rank0(f"Restored scheduler, lr={optimizer.get_learning_rate():.6f}")

        skip_steps = global_step - start_epoch * len(train_dataloader)

    print_on_rank0(f"Initializing tracker (report_to={args.report_to})...")
    tracker = create_tracker(args, args.output_dir)
    print_on_rank0("Tracker initialized successfully.")

    last_time = time.time()
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

            input_ids = data["input_ids"].cuda()
            attention_mask = data["attention_mask"].cuda()
            loss_mask = data["loss_mask"].cuda()

            # here target output is the full sequence
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

            anchor_positions, block_keep_mask, target_hidden = (
                online_flashmtp.prepare_training_tensors(
                    input_ids, hidden_states, loss_mask
                )
            )
            del target_output, hidden_states

            if args.shard_draft_by_tp:
                input_ids = get_tp_data_shard(input_ids)
                loss_mask = get_tp_data_shard(loss_mask)
                anchor_positions = get_tp_data_shard(anchor_positions)
                block_keep_mask = get_tp_data_shard(block_keep_mask)
                target_hidden = get_tp_data_shard(target_hidden)

            loss, accuracy, prefix_acc, mse_loss = flashmtp_model(
                input_ids=input_ids,
                loss_mask=loss_mask,
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                target_hidden=target_hidden,
            )
            del target_hidden, anchor_positions, block_keep_mask

            (loss / args.accumulation_steps).backward()

            if global_step % args.accumulation_steps == 0:
                optimizer.step()

            if global_step % args.log_interval == 0:
                loss_log = loss.clone()
                acc_log = accuracy.clone()
                pfx_log = prefix_acc.clone()
                mse_log = mse_loss.clone()
                dist.all_reduce(loss_log)
                dist.all_reduce(acc_log)
                dist.all_reduce(pfx_log)
                dist.all_reduce(mse_log)
                loss_log = loss_log / dist.get_world_size()
                acc_log = acc_log / dist.get_world_size()
                pfx_log = pfx_log / dist.get_world_size()
                mse_log = mse_log / dist.get_world_size()

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
                    mse_loss=mse_log.item() if args.w1_mse > 0 else None,
                )

            if dist.get_rank() == 0:
                elapsed = time.time() - last_time
                last_time = time.time()
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{accuracy.item():.4f}",
                        "pfx": f"{prefix_acc.item():.4f}",
                        "iter_time": f"{elapsed:.2f}s",
                    }
                )

            if global_step % args.save_interval == 0:
                save_checkpoint(
                    args, epoch, global_step, flashmtp_model, draft_model, optimizer
                )

    save_checkpoint(
        args, args.num_epochs, global_step, flashmtp_model, draft_model, optimizer
    )

    tracker.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
