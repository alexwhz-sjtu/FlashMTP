#!/usr/bin/env python3
"""Train a DLite student directly with supervised target-model objectives."""

import argparse
import logging
import os

# SGLang and FlexAttention compile lazily. Keep worker caches separate so
# concurrent torchrun ranks cannot corrupt one another's artifacts.
_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_global_rank = os.environ.get("RANK", "0")


def _configure_rank_cache(dir_env: str, root_env: str, default_name: str) -> None:
    root = os.environ.get(root_env) or os.environ.get(dir_env)
    if not root:
        root = os.path.join(_project_dir, "cache", default_name)
    os.environ[dir_env] = os.path.join(root, f"rank_{_global_rank}")


_configure_rank_cache("TVM_FFI_CACHE_DIR", "TVM_FFI_CACHE_ROOT", "tvm-ffi")
_configure_rank_cache(
    "TORCHINDUCTOR_CACHE_DIR", "TORCHINDUCTOR_CACHE_ROOT", "torchinductor"
)
_configure_rank_cache("TRITON_CACHE_DIR", "TRITON_CACHE_ROOT", "triton")

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from tqdm import tqdm

from specforge.core.dlite import OnlineDLiteModel, gather_target_prefill_logits
from specforge.distributed import destroy_distributed, init_distributed
from specforge.modeling.draft.dlite import DLiteDraftModel
from specforge.optimizer import BF16Optimizer
from specforge.tracker import create_tracker, get_tracker_class
from specforge.utils import print_on_rank0
from scripts.dlite_training import (
    add_common_args,
    build_draft_model,
    build_target_and_components,
    build_train_dataloader,
    hidden_states_to_cuda,
    load_training_state,
    log_cuda_peak,
    normalize_accumulated_gradients,
    resume_cursor,
    save_checkpoint,
    select_tp_rank_batch,
    stage_total_steps,
    validate_common_args,
    validate_tp_draft_sharding,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Direct supervised DLite student training"
    )
    add_common_args(parser)
    parser.add_argument(
        "--local-position",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the pivot_q_student local-position layout (required).",
    )
    parser.add_argument(
        "--init-from",
        help=(
            "Initialize student weights/config from a checkpoint while starting "
            "a fresh optimizer, scheduler, and data cursor."
        ),
    )
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.04)
    parser.add_argument("--final-ce-weight", type=float, default=0.1)
    parser.add_argument("--tv-loss-weight", type=float, default=1.0)
    parser.add_argument("--base-lm-ce-weight", type=float, default=0.0)
    parser.add_argument("--loss-decay-gamma", type=float)
    parser.add_argument("--base-lm-ce-decay-gamma", type=float)
    args = parser.parse_args()

    validate_common_args(parser, args)
    if not args.train_data_path:
        parser.error("--train-data-path is required")
    if args.resume_from and args.init_from:
        parser.error("--resume-from and --init-from are mutually exclusive")
    if not args.local_position:
        parser.error("Direct DLite student training requires --local-position")
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
        parser.error("Loss weights must be non-negative")
    if sum(loss_weights) == 0:
        parser.error("At least one loss weight must be positive")
    get_tracker_class(args.report_to).validate_args(parser, args)
    return args


def _sync_args_from_checkpoint(args, draft: DLiteDraftModel) -> None:
    if not draft.is_student:
        raise ValueError("SFT requires a pivot_q_student checkpoint.")
    args.block_size = draft.block_size
    args.num_draft_layers = draft.config.num_hidden_layers
    args.chs_num_layers = draft.chs_num_layers
    args.sequential_head = draft.sequential_head_type
    args.sequential_rank = draft.sequential_rank
    draft.config._attn_implementation = "flex_attention"


def _validate_resume_state(args, state: dict, total_steps: int) -> None:
    if state.get("training_stage") != "sft":
        raise ValueError(
            "SFT can only resume an SFT checkpoint; got "
            f"{state.get('training_stage')!r}. Use --init-from to import weights only."
        )
    expected_data = os.path.realpath(args.train_data_path)
    saved_data = state.get("train_data_identity")
    if saved_data is not None and saved_data != expected_data:
        raise ValueError(
            "Training dataset must match the resumed checkpoint: "
            f"saved={saved_data!r}, provided={expected_data!r}."
        )
    for key, requested in (
        ("tp_size", int(args.tp_size)),
        ("scheduler_total_steps", int(total_steps)),
        ("scheduler_learning_rate", float(args.learning_rate)),
        ("scheduler_warmup_ratio", float(args.warmup_ratio)),
    ):
        if state.get(key) is not None and state[key] != requested:
            raise ValueError(
                f"{key} must match the resumed checkpoint: "
                f"saved={state[key]!r}, requested={requested!r}."
            )
    if state.get("shard_draft_by_tp") is not None and bool(
        state["shard_draft_by_tp"]
    ) != bool(args.shard_draft_by_tp):
        raise ValueError("--shard-draft-by-tp must match the resumed checkpoint.")


def _checkpoint_metadata(
    args,
    *,
    epoch: int,
    next_batch: int,
    train_step: int,
    global_step: int,
    optimizer_step: int,
    scheduler_total_steps: int,
) -> dict:
    return {
        "training_stage": "sft",
        "stage_epoch": int(epoch),
        "next_batch_in_epoch": int(next_batch),
        "stage_step": int(train_step),
        "global_step": int(global_step),
        "optimizer_step": int(optimizer_step),
        "scheduler_total_steps": int(scheduler_total_steps),
        "scheduler_learning_rate": float(args.learning_rate),
        "scheduler_warmup_ratio": float(args.warmup_ratio),
        "shard_draft_by_tp": bool(args.shard_draft_by_tp),
        "tp_size": int(args.tp_size),
        "train_data_identity": os.path.realpath(args.train_data_path),
        "local_position": True,
    }


@record
def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    set_seed(args.seed)
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    tp_draft_rank = validate_tp_draft_sharding(args)

    resume_state = load_training_state(args.resume_from)
    if args.init_from:
        student = DLiteDraftModel.from_pretrained(
            args.init_from,
            torch_dtype=torch.bfloat16,
            attn_implementation="flex_attention",
        ).cuda()
        _sync_args_from_checkpoint(args, student)
        print_on_rank0(
            f"Initialized student weights from {args.init_from}; "
            "optimizer and training cursor start fresh."
        )
    elif resume_state is not None:
        student = DLiteDraftModel.from_pretrained(
            args.resume_from,
            torch_dtype=torch.bfloat16,
            attn_implementation="flex_attention",
        ).cuda()
        _sync_args_from_checkpoint(args, student)
    else:
        student = build_draft_model(args, model_role="pivot_q_student")
        print_on_rank0("Initialized pivot_q_student from the target config.")

    student.requires_grad_(True)
    target, tokenizer, components, mask_token_id = build_target_and_components(
        args, [student]
    )
    dataloader = build_train_dataloader(
        args,
        tokenizer,
        train_data_path=args.train_data_path,
        cache_namespace="sft",
        num_proc=args.build_dataset_num_proc,
    )
    scheduler_total_steps = stage_total_steps(
        dataloader, args.num_epochs, args.accumulation_steps
    )
    if resume_state is not None:
        _validate_resume_state(args, resume_state, scheduler_total_steps)

    online = OnlineDLiteModel(
        draft_model=student,
        target_lm_head=components.lm_head,
        target_embed_tokens=components.embed_tokens,
        mask_token_id=mask_token_id,
        block_size=student.block_size,
        num_anchors=args.num_anchors,
        loss_decay_gamma=args.loss_decay_gamma,
        final_ce_weight=args.final_ce_weight,
        tv_loss_weight=args.tv_loss_weight,
        base_lm_ce_weight=args.base_lm_ce_weight,
        base_lm_ce_decay_gamma=args.base_lm_ce_decay_gamma,
        use_target_greedy_ce_labels=True,
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
        student,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=scheduler_total_steps,
    )
    optimizer_step = 0
    if resume_state is not None:
        optimizer.load_state_dict(resume_state)
        optimizer_step = int(resume_state.get("optimizer_step", 0))

    tracker = create_tracker(args, args.output_dir)
    start_epoch, start_batch, train_step, global_step = resume_cursor(
        resume_state, "sft"
    )
    micro_steps = 0
    loss_denominator_sum = None
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(start_epoch, args.num_epochs):
        dataloader.sampler.set_epoch(epoch)
        student.train()
        iterator = (
            tqdm(dataloader, desc=f"SFT epoch {epoch}")
            if dist.get_rank() == 0
            else dataloader
        )
        for batch_idx, data in enumerate(iterator):
            if epoch == start_epoch and batch_idx < start_batch:
                continue
            global_step += 1
            train_step += 1
            input_ids = data["input_ids"].cuda()
            attention_mask = data["attention_mask"].cuda()
            loss_mask = data["loss_mask"].cuda()
            anchors, block_keep = online.sample_anchor_positions(
                input_ids.size(1), loss_mask
            )
            target_output = target.generate_dlite_data(
                input_ids, attention_mask, loss_mask
            )
            hidden_states = hidden_states_to_cuda(target_output.hidden_states)
            target_prefill_logits = target_output.logits.cuda()
            if tp_draft_rank is not None:
                input_ids = select_tp_rank_batch(input_ids, tp_draft_rank)
                loss_mask = select_tp_rank_batch(loss_mask, tp_draft_rank)
                anchors = select_tp_rank_batch(anchors, tp_draft_rank)
                block_keep = select_tp_rank_batch(block_keep, tp_draft_rank)
                hidden_states = select_tp_rank_batch(hidden_states, tp_draft_rank)
                target_prefill_logits = select_tp_rank_batch(
                    target_prefill_logits, tp_draft_rank
                )
            target_logits = gather_target_prefill_logits(
                target_prefill_logits, anchors, student.block_size
            )
            del target_output, target_prefill_logits
            (
                loss,
                accuracy,
                prefix_acc,
                final_ce,
                base_ce,
                tv_loss,
                loss_numerator,
                loss_denominator,
            ) = fsdp(
                input_ids=input_ids,
                hidden_states=hidden_states,
                loss_mask=loss_mask,
                anchor_positions=anchors,
                block_keep_mask=block_keep,
                target_prefill_logits=target_logits,
                target_logits_are_gathered=True,
            )
            del hidden_states, target_logits
            (loss_numerator / args.accumulation_steps).backward()
            loss_denominator_sum = (
                loss_denominator.detach().clone()
                if loss_denominator_sum is None
                else loss_denominator_sum + loss_denominator.detach()
            )
            micro_steps += 1
            grad_norm = None
            if micro_steps == args.accumulation_steps:
                normalize_accumulated_gradients(
                    optimizer,
                    loss_denominator_sum,
                    args.accumulation_steps,
                    group=fsdp.process_group,
                )
                grad_norm = optimizer.step()
                optimizer_step += 1
                micro_steps = 0
                loss_denominator_sum = None

            if global_step % args.log_interval == 0:
                metrics = torch.stack(
                    [
                        loss.detach(),
                        accuracy,
                        prefix_acc,
                        final_ce.detach(),
                        base_ce.detach(),
                        tv_loss.detach(),
                    ]
                )
                dist.all_reduce(metrics)
                metrics /= dist.get_world_size()
                payload = {
                    "train/loss": metrics[0].item(),
                    "train/accuracy": metrics[1].item(),
                    "train/prefix_acc": metrics[2].item(),
                    "train/final_ce": metrics[3].item(),
                    "train/base_ce": metrics[4].item(),
                    "train/tv": metrics[5].item(),
                    "train/lr": optimizer.get_learning_rate(),
                }
                if grad_norm is not None:
                    payload["train/grad_norm"] = grad_norm
                tracker.log(payload, step=global_step)
                print_on_rank0(
                    f"sft step={global_step} loss={metrics[0]:.4f} "
                    f"acc={metrics[1]:.4f}"
                )
            if global_step % args.save_interval == 0 and micro_steps == 0:
                save_checkpoint(
                    output_dir=args.output_dir,
                    name=f"epoch_{epoch}_step_{train_step}",
                    fsdp_model=fsdp,
                    draft_model=student,
                    optimizer=optimizer,
                    metadata=_checkpoint_metadata(
                        args,
                        epoch=epoch,
                        next_batch=batch_idx + 1,
                        train_step=train_step,
                        global_step=global_step,
                        optimizer_step=optimizer_step,
                        scheduler_total_steps=scheduler_total_steps,
                    ),
                )
        start_batch = 0

    if micro_steps:
        normalize_accumulated_gradients(
            optimizer,
            loss_denominator_sum,
            args.accumulation_steps,
            group=fsdp.process_group,
        )
        optimizer.step()
        optimizer_step += 1

    save_checkpoint(
        output_dir=args.output_dir,
        name="final",
        fsdp_model=fsdp,
        draft_model=student,
        optimizer=optimizer,
        metadata=_checkpoint_metadata(
            args,
            epoch=args.num_epochs,
            next_batch=0,
            train_step=train_step,
            global_step=global_step,
            optimizer_step=optimizer_step,
            scheduler_total_steps=scheduler_total_steps,
        ),
    )
    memory = log_cuda_peak("sft")
    tracker.log(
        {
            "train/cuda_peak_allocated_gib": memory["allocated_gib"],
            "train/cuda_peak_reserved_gib": memory["reserved_gib"],
        },
        step=global_step,
    )
    tracker.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
