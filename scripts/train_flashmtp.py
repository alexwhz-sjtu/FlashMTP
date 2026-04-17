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
from specforge.distributed import destroy_distributed, get_dp_group, init_distributed
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
        "--context-window-size",
        type=int,
        default=1,
        help="Sliding window W: each draft block may attend to at most W teacher token "
        "positions before its anchor (full target hidden is prepended; mask restricts). "
        "W=1 matches the previous single-(anchor-1) conditioning.",
    )
    model_group.add_argument(
        "--loss-decay-gamma",
        type=float,
        default=None,
        help="Temperature for exp decay on completion suffix: w∝exp(-(d-1)/gamma), "
        "where d is 1-based index within the suffix (first predicted token d=1). "
        "None => uniform over supervised positions.",
    )
    model_group.add_argument(
        "--cold-start-loss-weight",
        type=float,
        default=1.0,
        help="Weight for cold-start blocks (only anchor token clean, p=1).",
    )
    model_group.add_argument(
        "--continuation-loss-weight",
        type=float,
        default=1.0,
        help="Max weight for continuation blocks (p in {2,..,B-1}); scaled by warmup. "
        "Must be in (0, 1].",
    )
    model_group.add_argument(
        "--continuation-warmup-epochs",
        type=float,
        default=0.0,
        help="Linear continuation warmup in **epoch** units: "
        "eff_weight = max_weight * min(1, training_epoch / this). "
        "training_epoch = epoch + step_in_epoch/len(dataloader). "
        "0 = full continuation weight from the start.",
    )

    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument("--train-data-path", type=str, required=True)
    dataset_group.add_argument("--eval-data-path", type=str, default=None)
    dataset_group.add_argument("--chat-template", type=str, default="qwen")
    dataset_group.add_argument("--is-preformatted", action="store_true")
    dataset_group.add_argument("--dataloader-num-workers", type=int, default=8)
    dataset_group.add_argument(
        "--build-dataset-num-proc",
        type=int,
        default=int(os.environ.get("SPECFORGE_DATA_NUM_PROC", 8)),
    )

    training_group = parser.add_argument_group("training")
    training_group.add_argument("--num-epochs", type=int, default=6)
    training_group.add_argument("--batch-size", type=int, default=1)
    training_group.add_argument("--learning-rate", type=float, default=6e-4)
    training_group.add_argument("--max-length", type=int, default=3072)
    training_group.add_argument("--warmup-ratio", type=float, default=0.04)
    training_group.add_argument("--max-grad-norm", type=float, default=1.0)
    training_group.add_argument("--accumulation-steps", type=int, default=1)
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


def build_models(args) -> Tuple[FlashMTPTargetModel, FlashMTPDraftModel]:
    """Build target model (backend wrapper) and draft model."""
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

    if not hasattr(draft_config, "flashmtp_config") or draft_config.flashmtp_config is None:
        draft_config.flashmtp_config = {}
        
    draft_config.flashmtp_config["context_window_size"] = args.context_window_size

    draft_config._attn_implementation = args.attention_backend
    print_on_rank0(f"Using attention backend: {args.attention_backend}")

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

    return target_model, draft_model


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

    with FSDP.state_dict_type(draft_model, StateDictType.FULL_STATE_DICT):
        draft_state_dict = draft_model.state_dict()

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

            print_on_rank0(f"Saved checkpoint to {save_dir}")

    dist.barrier()


def record_metrics(
    args,
    total_loss: float,
    ce_acc: float,
    mean_prefix_len: float,
    global_step: int,
    tracker,
    optimizer,
    train_dataloader=None,
    mode: str = "train",
    lambda2_eff: Optional[float] = None,
    mean_prefix_len_cold: Optional[float] = None,
    mean_prefix_len_cont: Optional[float] = None,
) -> None:
    """Log clean-prefix completion metrics."""
    logdict = {}
    if mode == "train" and optimizer is not None:
        logdict["train/lr"] = optimizer.get_learning_rate()
    logdict[f"{mode}/total_loss"] = total_loss
    logdict[f"{mode}/ce_acc"] = ce_acc
    logdict[f"{mode}/mean_prefix_len"] = mean_prefix_len
    if lambda2_eff is not None:
        logdict[f"{mode}/lambda2_eff"] = lambda2_eff
    if mean_prefix_len_cold is not None:
        logdict[f"{mode}/mean_prefix_len_cold"] = mean_prefix_len_cold
    if mean_prefix_len_cont is not None:
        logdict[f"{mode}/mean_prefix_len_cont"] = mean_prefix_len_cont
    total_steps = args.num_epochs * len(train_dataloader) // args.accumulation_steps
    extra = ""
    if lambda2_eff is not None:
        extra += f", λ2_eff: {lambda2_eff:.4f}"
    print_on_rank0(
        f"{mode.capitalize()} - Step {global_step}/{total_steps}, "
        f"Loss: {total_loss:.4f}, ce_acc: {ce_acc:.4f}, mean_prefix_len: {mean_prefix_len:.2f}"
        f"{extra}"
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
    if not (0.0 < args.continuation_loss_weight <= 1.0):
        raise ValueError(
            "--continuation-loss-weight must be in (0, 1]; got "
            f"{args.continuation_loss_weight}"
        )
    if args.continuation_warmup_epochs < 0.0:
        raise ValueError(
            "--continuation-warmup-epochs must be >= 0; got "
            f"{args.continuation_warmup_epochs}"
        )
    set_seed(args.seed)

    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    print_with_rank("Initialized distributed")

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
    if draft_model_last_checkpoint:
        loaded_model = FlashMTPDraftModel.from_pretrained(
            draft_model_last_checkpoint, torch_dtype=torch.bfloat16
        )
        draft_model.load_state_dict(loaded_model.state_dict())
        del loaded_model
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
        
    print_on_rank0(f"****** Important: Make sure using the same mask_token_id with inference.***** \n Using mask_token_id: {mask_token_id} \n")


    draft_model.mask_token_id = mask_token_id
    
    draft_model.config.flashmtp_config["mask_token_id"] = mask_token_id
    draft_model.config.flashmtp_config["target_layer_ids"] = draft_model.target_layer_ids
    draft_model.config.flashmtp_config["context_window_size"] = args.context_window_size
    print_on_rank0(f"flashmtp_config: {draft_model.config.flashmtp_config}")

    # Shard draft only; wrapping the whole OnlineFlashMTPModel breaks lm_head/embed/fc.
    draft_model = FSDP(
        draft_model,
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    )
    print_with_rank("Initialized FSDP on draft_model")

    train_dataloader, eval_dataloader = build_dataloader(args, tokenizer)

    steps_per_epoch = math.ceil(len(train_dataloader) / args.accumulation_steps)
    total_steps = args.num_epochs * steps_per_epoch
    print_on_rank0(f"Total training steps: {total_steps}")

    print_on_rank0("Loading target embeddings and head...")
    target_components = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model_path,
        embed_key="model.embed_tokens.weight",  # Adjust if Qwen/Llama differs
        lm_head_key="lm_head.weight",
        device="cuda",
        trust_remote_code=args.trust_remote_code,
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
        cold_start_loss_weight=args.cold_start_loss_weight,
        continuation_loss_weight=args.continuation_loss_weight,
        continuation_warmup_epochs=args.continuation_warmup_epochs,
        context_window_size=args.context_window_size,
    )

    optimizer = BF16Optimizer(
        draft_model,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=total_steps,
    )
    skip_steps=0
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

    batches_per_epoch = max(len(train_dataloader), 1)

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

            # HF: tuple of (B,L,H) per layer. SGLang: single (B,L,D) fused tensor.
            # Do NOT ``for h in tensor`` — that walks the batch dimension and breaks lm_head.
            _hs = target_output.hidden_states
            if torch.is_tensor(_hs):
                hidden_states = _hs.cuda()
            else:
                hidden_states = tuple(t.cuda() for t in _hs)

            training_epoch = float(epoch) + float(step_in_epoch) / float(
                batches_per_epoch
            )
            loss, loss_dict = flashmtp_model(
                input_ids=input_ids,
                hidden_states=hidden_states,
                loss_mask=loss_mask,
                training_epoch=training_epoch,
            )

            (loss / args.accumulation_steps).backward()

            if global_step % args.accumulation_steps == 0:
                optimizer.step()

            if global_step % args.log_interval == 0:
                total_loss_log = loss_dict["total_loss"].clone()
                dist.all_reduce(total_loss_log)
                total_loss_log = total_loss_log / dist.get_world_size()

                ce_acc_log = loss_dict["ce_acc"].clone()
                mean_p_log = loss_dict["mean_prefix_len"].clone()
                mean_p_cold_log = loss_dict["mean_prefix_len_cold"].clone()
                mean_p_cont_log = loss_dict["mean_prefix_len_cont"].clone()
                lambda2_log = loss_dict["lambda2_eff"].clone()
                dist.all_reduce(ce_acc_log)
                dist.all_reduce(mean_p_log)
                dist.all_reduce(mean_p_cold_log)
                dist.all_reduce(mean_p_cont_log)
                dist.all_reduce(lambda2_log)
                ws = dist.get_world_size()
                ce_acc_log = ce_acc_log / ws
                mean_p_log = mean_p_log / ws
                mean_p_cold_log = mean_p_cold_log / ws
                mean_p_cont_log = mean_p_cont_log / ws
                lambda2_log = lambda2_log / ws
                record_metrics(
                    args,
                    total_loss_log.item(),
                    ce_acc_log.item(),
                    mean_p_log.item(),
                    global_step,
                    tracker,
                    optimizer,
                    train_dataloader,
                    mode="train",
                    lambda2_eff=lambda2_log.item(),
                    mean_prefix_len_cold=mean_p_cold_log.item(),
                    mean_prefix_len_cont=mean_p_cont_log.item(),
                )

            if dist.get_rank() == 0:
                elapsed = time.time() - last_time
                last_time = time.time()
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss_dict['total_loss'].item():.4f}",
                        "ce_acc": f"{loss_dict['ce_acc'].item():.4f}",
                        "mean_p": f"{loss_dict['mean_prefix_len'].item():.2f}",
                        "l2": f"{loss_dict['lambda2_eff'].item():.3f}",
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
