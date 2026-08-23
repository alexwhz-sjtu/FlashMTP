#!/usr/bin/env python3
"""Train the standalone global-position SWA draft teacher."""

import argparse
import logging
import time

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from tqdm import tqdm

from specforge.core.flashmtp import OnlineFlashMTPModel, gather_target_prefill_logits
from specforge.distributed import destroy_distributed, init_distributed
from specforge.modeling.draft.flashmtp import FlashMTPDraftModel
from specforge.optimizer import BF16Optimizer
from specforge.tracker import create_tracker, get_tracker_class
from specforge.utils import print_on_rank0
from scripts.flashmtp_training import (
    add_common_args,
    build_draft_model,
    build_target_and_components,
    build_train_dataloader,
    hidden_states_to_cuda,
    load_training_state,
    log_cuda_peak,
    resume_cursor,
    save_checkpoint,
    select_tp_rank_batch,
    stage_total_steps,
    validate_common_args,
    validate_tp_draft_sharding,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train FlashMTP SWA teacher")
    add_common_args(parser)
    parser.add_argument(
        "--init-from",
        help=(
            "Initialize draft weights/config from a checkpoint but start a fresh "
            "optimizer, data cursor, and global step."
        ),
    )
    parser.add_argument("--num-epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.04)
    parser.add_argument("--final-ce-weight", type=float, default=1.0)
    parser.add_argument("--tv-loss-weight", type=float, default=1.0)
    parser.add_argument("--base-lm-ce-weight", type=float, default=0.0)
    parser.add_argument("--loss-decay-gamma", type=float)
    parser.add_argument("--base-lm-ce-decay-gamma", type=float)
    parser.add_argument("--markov-teacher-forcing-ratio", type=float, default=1.0)
    args = parser.parse_args()
    validate_common_args(parser, args)
    if args.resume_from and args.init_from:
        parser.error("--resume-from and --init-from are mutually exclusive")
    if args.num_epochs <= 0:
        parser.error("--num-epochs must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if not 0.0 <= args.warmup_ratio <= 1.0:
        parser.error("--warmup-ratio must be in [0, 1]")
    loss_weights = (
        args.final_ce_weight,
        args.tv_loss_weight,
        args.base_lm_ce_weight,
    )
    if any(weight < 0 for weight in loss_weights):
        parser.error("Teacher loss weights must be non-negative")
    if sum(loss_weights) == 0:
        parser.error("At least one teacher loss weight must be positive")
    if not 0.0 <= args.markov_teacher_forcing_ratio <= 1.0:
        parser.error("--markov-teacher-forcing-ratio must be in [0, 1]")
    get_tracker_class(args.report_to).validate_args(parser, args)
    return args


def _sync_args_from_checkpoint(args, draft: FlashMTPDraftModel) -> None:
    if not draft.is_teacher:
        raise ValueError("Teacher training can only resume an swa_teacher checkpoint.")
    args.block_size = draft.block_size
    args.num_draft_layers = draft.config.num_hidden_layers
    args.swa_window_size = draft.swa_window_size
    args.anchor_group_size = draft.anchor_group_size
    args.chs_num_layers = draft.chs_num_layers
    args.markov_head_type = draft.markov_head_type
    args.markov_output_mode = draft.markov_output_mode
    args.markov_rank = draft.markov_rank
    # Transformers does not serialize the private attention implementation
    # selector and defaults a reloaded checkpoint to SDPA.  Teacher training
    # supplies a torch BlockMask, which is supported by FlexAttention only.
    draft.config._attn_implementation = "flex_attention"


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    set_seed(args.seed)
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    tp_draft_rank = validate_tp_draft_sharding(args)

    resume_state = load_training_state(args.resume_from)
    if args.init_from:
        draft = FlashMTPDraftModel.from_pretrained(
            args.init_from,
            torch_dtype=torch.bfloat16,
            attn_implementation="flex_attention",
        ).cuda()
        _sync_args_from_checkpoint(args, draft)
        print_on_rank0(
            f"Initialized teacher weights from {args.init_from}; "
            "optimizer and training cursor start fresh."
        )
    elif resume_state is None:
        draft = build_draft_model(args, model_role="swa_teacher")
    else:
        draft = FlashMTPDraftModel.from_pretrained(
            args.resume_from,
            torch_dtype=torch.bfloat16,
            attn_implementation="flex_attention",
        ).cuda()
        _sync_args_from_checkpoint(args, draft)
    target, tokenizer, components, mask_token_id = build_target_and_components(
        args, [draft]
    )
    dataloader = build_train_dataloader(args, tokenizer)
    online = OnlineFlashMTPModel(
        draft_model=draft,
        target_lm_head=components.lm_head,
        target_embed_tokens=components.embed_tokens,
        mask_token_id=mask_token_id,
        block_size=draft.block_size,
        num_anchors=args.num_anchors,
        loss_decay_gamma=args.loss_decay_gamma,
        final_ce_weight=args.final_ce_weight,
        tv_loss_weight=args.tv_loss_weight,
        base_lm_ce_weight=args.base_lm_ce_weight,
        base_lm_ce_decay_gamma=args.base_lm_ce_decay_gamma,
        markov_teacher_forcing_ratio=args.markov_teacher_forcing_ratio,
    )
    fsdp = FSDP(
        online,
        ignored_modules=[components.lm_head, components.embed_tokens],
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    )
    optimizer = BF16Optimizer(
        draft,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=stage_total_steps(
            dataloader, args.num_epochs, args.accumulation_steps
        ),
    )
    if resume_state is not None:
        optimizer.load_state_dict(resume_state)
    tracker = create_tracker(args, args.output_dir)
    start_epoch, start_batch, stage_step, global_step = resume_cursor(
        resume_state, "teacher"
    )
    micro_steps = 0
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(start_epoch, args.num_epochs):
        dataloader.sampler.set_epoch(epoch)
        draft.train()
        iterator = tqdm(dataloader, desc=f"Teacher epoch {epoch}") if dist.get_rank() == 0 else dataloader
        for batch_idx, data in enumerate(iterator):
            if epoch == start_epoch and batch_idx < start_batch:
                continue
            global_step += 1
            stage_step += 1
            debug_t0 = time.perf_counter()
            trace_first_step = global_step == 1
            if trace_first_step:
                print(f"[rank {dist.get_rank()}] step1: batch loaded", flush=True)
            input_ids = data["input_ids"].cuda()
            attention_mask = data["attention_mask"].cuda()
            loss_mask = data["loss_mask"].cuda()
            anchors, block_keep = online.sample_anchor_positions(
                input_ids.size(1), loss_mask
            )
            target_output = target.generate_flashmtp_data(
                input_ids, attention_mask, loss_mask
            )
            if global_step <= 3 and dist.get_rank() == 0:
                torch.cuda.synchronize()
                print(f"[timing] step={global_step} target={time.perf_counter()-debug_t0:.2f}s", flush=True)
            if trace_first_step:
                print(f"[rank {dist.get_rank()}] step1: target prefill done", flush=True)
            hidden_states = hidden_states_to_cuda(target_output.hidden_states)
            if tp_draft_rank is not None:
                input_ids = select_tp_rank_batch(input_ids, tp_draft_rank)
                loss_mask = select_tp_rank_batch(loss_mask, tp_draft_rank)
                anchors = select_tp_rank_batch(anchors, tp_draft_rank)
                block_keep = select_tp_rank_batch(block_keep, tp_draft_rank)
                hidden_states = select_tp_rank_batch(hidden_states, tp_draft_rank)
                target_prefill_logits = select_tp_rank_batch(
                    target_output.logits.cuda(), tp_draft_rank
                )
            else:
                target_prefill_logits = target_output.logits.cuda()
            target_logits = gather_target_prefill_logits(
                target_prefill_logits, anchors, draft.block_size
            )
            if trace_first_step:
                print(f"[rank {dist.get_rank()}] step1: target logits gathered", flush=True)
            del target_output, target_prefill_logits
            # Preparation uses trainable draft modules (for example
            # ``history_fuse``), so it must run inside FSDP.forward().  Calling
            # ``online.prepare_batch`` directly here observes empty local
            # parameter shards on non-owning ranks.
            loss, accuracy, prefix_acc, final_ce, base_ce, tv_loss = fsdp(
                input_ids=input_ids,
                hidden_states=hidden_states,
                loss_mask=loss_mask,
                anchor_positions=anchors,
                block_keep_mask=block_keep,
                target_prefill_logits=target_logits,
                target_logits_are_gathered=True,
            )
            if global_step <= 3 and dist.get_rank() == 0:
                torch.cuda.synchronize()
                print(f"[timing] step={global_step} forward={time.perf_counter()-debug_t0:.2f}s", flush=True)
            if trace_first_step:
                print(f"[rank {dist.get_rank()}] step1: draft forward done", flush=True)
            del hidden_states, target_logits
            (loss / args.accumulation_steps).backward()
            if global_step <= 3 and dist.get_rank() == 0:
                torch.cuda.synchronize()
                print(f"[timing] step={global_step} backward={time.perf_counter()-debug_t0:.2f}s", flush=True)
            if trace_first_step:
                print(f"[rank {dist.get_rank()}] step1: backward done", flush=True)
            micro_steps += 1
            if micro_steps == args.accumulation_steps:
                grad_norm = optimizer.step()
                micro_steps = 0
            else:
                grad_norm = None
            if global_step <= 3 and dist.get_rank() == 0:
                torch.cuda.synchronize()
                print(f"[timing] step={global_step} complete={time.perf_counter()-debug_t0:.2f}s", flush=True)

            if global_step % args.log_interval == 0:
                metrics = torch.stack(
                    [loss.detach(), accuracy, prefix_acc, final_ce.detach(), base_ce.detach(), tv_loss.detach()]
                )
                dist.all_reduce(metrics)
                metrics /= dist.get_world_size()
                payload = {
                        "teacher/loss": metrics[0].item(),
                        "teacher/accuracy": metrics[1].item(),
                        "teacher/prefix_acc": metrics[2].item(),
                        "teacher/final_ce": metrics[3].item(),
                        "teacher/base_ce": metrics[4].item(),
                        "teacher/tv": metrics[5].item(),
                        "teacher/lr": optimizer.get_learning_rate(),
                }
                if grad_norm is not None:
                    payload["teacher/grad_norm"] = grad_norm
                tracker.log(payload, step=global_step)
                print_on_rank0(
                    f"teacher step={global_step} loss={metrics[0]:.4f} acc={metrics[1]:.4f}"
                )
            if global_step % args.save_interval == 0 and micro_steps == 0:
                save_checkpoint(
                    output_dir=args.output_dir,
                    name=f"epoch_{epoch}_step_{global_step}",
                    fsdp_model=fsdp,
                    draft_model=draft,
                    optimizer=optimizer,
                    metadata={
                        "training_stage": "teacher",
                        "stage_epoch": epoch,
                        "next_batch_in_epoch": batch_idx + 1,
                        "stage_step": stage_step,
                        "global_step": global_step,
                        "serial_head_inherited": False,
                    },
                )
        start_batch = 0

    if micro_steps:
        optimizer.scale_model_gradients(args.accumulation_steps / micro_steps)
        optimizer.step()
    save_checkpoint(
        output_dir=args.output_dir,
        name="final",
        fsdp_model=fsdp,
        draft_model=draft,
        optimizer=optimizer,
        metadata={
            "training_stage": "teacher",
            "stage_epoch": args.num_epochs,
            "next_batch_in_epoch": 0,
            "stage_step": stage_step,
            "global_step": global_step,
            "serial_head_inherited": False,
        },
    )
    memory = log_cuda_peak("teacher")
    tracker.log(
        {
            "teacher/cuda_peak_allocated_gib": memory["allocated_gib"],
            "teacher/cuda_peak_reserved_gib": memory["reserved_gib"],
        },
        step=global_step,
    )
    tracker.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
