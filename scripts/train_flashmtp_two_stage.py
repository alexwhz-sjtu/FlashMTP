#!/usr/bin/env python3
"""Distill an SWA teacher into PivotQ, then fine-tune with label losses."""

import argparse
import copy
import gc
import logging
import os

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from tqdm import tqdm

from specforge.core.flashmtp import (
    OnlineFlashMTPModel,
    compute_stage1_distillation_loss,
    gather_target_prefill_logits,
)
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
    build_two_stage_dataloaders,
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


STUDENT_INIT_MODES = ("scratch", "shared_init")
SHARED_BACKBONE_MODULES = (
    "layers",
    "norm",
    "layer_depth_embedding",
    "context_norm",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Two-stage FlashMTP distillation")
    add_common_args(parser)
    parser.add_argument("--teacher-draft-path")
    parser.add_argument(
        "--stage1-train-data-path",
        help="Stage 1 distillation JSONL (falls back to --train-data-path).",
    )
    parser.add_argument(
        "--stage2-train-data-path",
        help="Stage 2 supervised JSONL (falls back to --train-data-path).",
    )
    parser.add_argument("--stage1-build-dataset-num-proc", type=int)
    parser.add_argument("--stage2-build-dataset-num-proc", type=int)
    parser.add_argument(
        "--student-init-mode",
        choices=STUDENT_INIT_MODES,
        help=(
            "Fresh Stage 1 initialization. 'scratch' randomly initializes the "
            "student; 'shared_init' copies the teacher's shared parallel backbone. "
            "On resume, the checkpoint mode is used when this option is omitted."
        ),
    )
    parser.add_argument("--stage1-epochs", type=int, required=True)
    parser.add_argument("--stage1-learning-rate", type=float, required=True)
    parser.add_argument("--stage1-warmup-ratio", type=float, required=True)
    parser.add_argument("--stage1-tv-weight", type=float, default=1.0)
    parser.add_argument("--stage1-hidden-weight", type=float, default=1.0)
    parser.add_argument("--stage1-smooth-l1-beta", type=float, default=1.0)
    parser.add_argument("--stage1-loss-decay-gamma", type=float)
    parser.add_argument("--stage2-epochs", type=int, required=True)
    parser.add_argument("--stage2-learning-rate", type=float, required=True)
    parser.add_argument("--stage2-warmup-ratio", type=float, required=True)
    parser.add_argument("--stage2-final-ce-weight", type=float, default=1.0)
    parser.add_argument("--stage2-tv-weight", type=float, default=1.0)
    parser.add_argument("--stage2-base-ce-weight", type=float, default=0.0)
    parser.add_argument("--stage2-loss-decay-gamma", type=float)
    parser.add_argument("--stage2-base-ce-decay-gamma", type=float)
    args = parser.parse_args()
    validate_common_args(parser, args)
    args.stage1_train_data_path = (
        args.stage1_train_data_path or args.train_data_path
    )
    args.stage2_train_data_path = (
        args.stage2_train_data_path or args.train_data_path
    )
    if not args.stage1_train_data_path:
        parser.error(
            "--stage1-train-data-path is required (or provide --train-data-path)"
        )
    if not args.stage2_train_data_path:
        parser.error(
            "--stage2-train-data-path is required (or provide --train-data-path)"
        )
    for name in (
        "stage1_build_dataset_num_proc",
        "stage2_build_dataset_num_proc",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("stage1_epochs", "stage2_epochs", "accumulation_steps"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("stage1_learning_rate", "stage2_learning_rate"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("stage1_warmup_ratio", "stage2_warmup_ratio"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.stage1_smooth_l1_beta < 0:
        parser.error("--stage1-smooth-l1-beta must be non-negative")
    if args.stage1_tv_weight < 0 or args.stage1_hidden_weight < 0:
        parser.error("Stage 1 loss weights must be non-negative")
    if args.stage1_tv_weight + args.stage1_hidden_weight == 0:
        parser.error("At least one Stage 1 loss weight must be positive")
    stage2_weights = (
        args.stage2_final_ce_weight,
        args.stage2_tv_weight,
        args.stage2_base_ce_weight,
    )
    if any(weight < 0 for weight in stage2_weights):
        parser.error("Stage 2 loss weights must be non-negative")
    if sum(stage2_weights) == 0:
        parser.error("At least one Stage 2 loss weight must be positive")
    if args.resume_from is None and args.teacher_draft_path is None:
        parser.error("--teacher-draft-path is required for fresh training")
    get_tracker_class(args.report_to).validate_args(parser, args)
    return args


def _sync_args_from_model(args, draft: FlashMTPDraftModel) -> None:
    args.block_size = draft.block_size
    args.num_draft_layers = draft.config.num_hidden_layers
    args.swa_window_size = draft.swa_window_size
    args.anchor_group_size = draft.anchor_group_size
    args.chs_num_layers = draft.chs_num_layers
    args.markov_head_type = draft.markov_head_type
    args.markov_output_mode = draft.markov_output_mode
    args.markov_rank = draft.markov_rank
    draft.config._attn_implementation = "flex_attention"


def _structure_signature(draft: FlashMTPDraftModel) -> tuple:
    return (
        draft.swa_window_size,
        draft.anchor_group_size,
        draft.chs_num_layers,
        draft.block_size,
        draft.config.num_hidden_layers,
        draft.markov_head_type,
        draft.markov_output_mode,
        draft.markov_rank,
        draft.config.vocab_size,
    )


def _resolve_student_init_mode(requested_mode, resume_state) -> str:
    if resume_state is None:
        return requested_mode or "scratch"
    saved_mode = resume_state.get("student_init_mode", "scratch")
    if saved_mode not in STUDENT_INIT_MODES:
        raise ValueError(
            f"Checkpoint has unsupported student_init_mode={saved_mode!r}."
        )
    saved_inherited = resume_state.get("shared_backbone_inherited")
    expected_inherited = saved_mode == "shared_init"
    if saved_inherited is not None and bool(saved_inherited) != expected_inherited:
        raise ValueError(
            "Checkpoint shared-backbone metadata is inconsistent with "
            f"student_init_mode={saved_mode!r}."
        )
    if requested_mode is not None and requested_mode != saved_mode:
        raise ValueError(
            "Student init mode must match the resumed checkpoint: "
            f"saved={saved_mode!r}, requested={requested_mode!r}."
        )
    return saved_mode


def _copy_shared_backbone(
    teacher: FlashMTPDraftModel, student: FlashMTPDraftModel
) -> None:
    """Initialize student modules shared with the teacher's parallel backbone."""
    if not teacher.is_teacher or not student.is_student:
        raise ValueError("Shared init requires an swa_teacher and a pivot_q_student")
    if _structure_signature(teacher) != _structure_signature(student):
        raise ValueError("Teacher/student structures must match for shared init")
    for module_name in SHARED_BACKBONE_MODULES:
        student_module = getattr(student, module_name)
        teacher_module = getattr(teacher, module_name)
        student_module.load_state_dict(teacher_module.state_dict(), strict=True)


def _set_student_stage1_trainable(student: FlashMTPDraftModel) -> None:
    student.requires_grad_(True)
    student.history_fuse.requires_grad_(False)
    student.history_norm.requires_grad_(False)
    if student.markov_head is not None:
        student.markov_head.requires_grad_(False)


def _set_student_stage2_trainable(student: FlashMTPDraftModel) -> None:
    student.requires_grad_(True)
    student.history_fuse.requires_grad_(False)
    student.history_norm.requires_grad_(False)


def _copy_serial_head(teacher: FlashMTPDraftModel, student: FlashMTPDraftModel) -> None:
    teacher_signature = (
        teacher.markov_head_type,
        teacher.markov_output_mode,
        teacher.markov_rank,
        teacher.block_size,
        teacher.config.vocab_size,
    )
    student_signature = (
        student.markov_head_type,
        student.markov_output_mode,
        student.markov_rank,
        student.block_size,
        student.config.vocab_size,
    )
    if teacher_signature != student_signature:
        raise ValueError(
            f"Teacher/student serial configuration mismatch: {teacher_signature} != {student_signature}"
        )
    if teacher.markov_head is None or student.markov_head is None:
        if teacher.markov_head is not student.markov_head:
            raise ValueError("Teacher/student serial heads do not match")
        return
    student.markov_head.load_state_dict(teacher.markov_head.state_dict(), strict=True)


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    set_seed(args.seed)
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    tp_draft_rank = validate_tp_draft_sharding(args)

    resume_state = load_training_state(args.resume_from)
    resume_stage = None if resume_state is None else resume_state.get("training_stage")
    if resume_stage not in (None, "stage1", "transition", "stage2"):
        raise ValueError(f"Unsupported two-stage checkpoint stage: {resume_stage!r}")
    if (
        resume_state is not None
        and "shard_draft_by_tp" in resume_state
        and bool(resume_state["shard_draft_by_tp"])
        != bool(args.shard_draft_by_tp)
    ):
        raise ValueError(
            "--shard-draft-by-tp must match the resumed checkpoint: "
            f"saved={bool(resume_state['shard_draft_by_tp'])}, "
            f"requested={bool(args.shard_draft_by_tp)}."
        )
    if (
        resume_state is not None
        and "tp_size" in resume_state
        and int(resume_state["tp_size"]) != int(args.tp_size)
    ):
        raise ValueError(
            "--tp-size must match the resumed checkpoint: "
            f"saved={int(resume_state['tp_size'])}, requested={int(args.tp_size)}."
        )
    stage1_data_identity = os.path.realpath(args.stage1_train_data_path)
    stage2_data_identity = os.path.realpath(args.stage2_train_data_path)
    for stage, current_identity in (
        ("stage1", stage1_data_identity),
        ("stage2", stage2_data_identity),
    ):
        key = f"{stage}_train_data_identity"
        if (
            resume_state is not None
            and resume_state.get(key) is not None
            and resume_state[key] != current_identity
        ):
            raise ValueError(
                f"{stage} dataset must match the resumed checkpoint: "
                f"saved={resume_state[key]!r}, provided={current_identity!r}."
            )
    student_init_mode = _resolve_student_init_mode(
        args.student_init_mode, resume_state
    )
    args.student_init_mode = student_init_mode
    print_on_rank0(f"Student init mode: {student_init_mode}")
    provided_teacher_identity = (
        os.path.realpath(args.teacher_draft_path) if args.teacher_draft_path else None
    )
    saved_teacher_identity = (
        None
        if resume_state is None
        else resume_state.get("teacher_checkpoint_identity")
    )
    if (
        resume_stage == "stage1"
        and saved_teacher_identity is not None
        and saved_teacher_identity != provided_teacher_identity
    ):
        raise ValueError(
            "Stage 1 must resume with the same teacher checkpoint: "
            f"saved={saved_teacher_identity!r}, provided={provided_teacher_identity!r}."
        )
    teacher_identity = saved_teacher_identity or provided_teacher_identity

    teacher = None
    if resume_stage in (None, "stage1"):
        if not args.teacher_draft_path:
            raise ValueError("--teacher-draft-path is required for fresh or Stage 1 training")
        teacher = FlashMTPDraftModel.from_pretrained(
            args.teacher_draft_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flex_attention",
        ).cuda().eval()
        if not teacher.is_teacher:
            raise ValueError("--teacher-draft-path must contain an swa_teacher checkpoint")
        teacher.requires_grad_(False)
        _sync_args_from_model(args, teacher)
        if resume_stage == "stage1":
            student = FlashMTPDraftModel.from_pretrained(
                args.resume_from,
                torch_dtype=torch.bfloat16,
                attn_implementation="flex_attention",
            ).cuda()
            if not student.is_student:
                raise ValueError("Stage 1 checkpoint must contain a pivot_q_student")
            if _structure_signature(student) != _structure_signature(teacher):
                raise ValueError("Stage 1 student structure no longer matches the teacher")
        else:
            student = build_draft_model(
                args,
                model_role="pivot_q_student",
                source_config=copy.deepcopy(teacher.config),
            )
            if student_init_mode == "shared_init":
                _copy_shared_backbone(teacher, student)
                print_on_rank0(
                    "Initialized student shared parallel backbone from teacher"
                )
    else:
        student = FlashMTPDraftModel.from_pretrained(
            args.resume_from,
            torch_dtype=torch.bfloat16,
            attn_implementation="flex_attention",
        ).cuda()
        if not student.is_student:
            raise ValueError("Transition/Stage 2 checkpoint must contain a pivot_q_student")
        if not bool(resume_state.get("serial_head_inherited")):
            raise ValueError("Transition/Stage 2 checkpoint has no inherited serial head")
        _sync_args_from_model(args, student)

    if resume_stage in (None, "stage1"):
        _set_student_stage1_trainable(student)
    else:
        _set_student_stage2_trainable(student)
    drafts_for_target = [student] if teacher is None else [teacher, student]
    target, tokenizer, components, mask_token_id = build_target_and_components(
        args, drafts_for_target
    )
    if resume_stage in (None, "stage1"):
        stage1_dataloader, stage2_dataloader = build_two_stage_dataloaders(
            args, tokenizer
        )
    else:
        stage1_dataloader = None
        stage2_dataloader = build_train_dataloader(
            args,
            tokenizer,
            train_data_path=args.stage2_train_data_path,
            cache_namespace="stage2",
            num_proc=args.stage2_build_dataset_num_proc,
        )

    teacher_online = None
    if teacher is not None:
        teacher_online = OnlineFlashMTPModel(
            draft_model=teacher,
            target_lm_head=components.lm_head,
            target_embed_tokens=components.embed_tokens,
            mask_token_id=mask_token_id,
            block_size=teacher.block_size,
            num_anchors=args.num_anchors,
        )
    student_online = OnlineFlashMTPModel(
        draft_model=student,
        target_lm_head=components.lm_head,
        target_embed_tokens=components.embed_tokens,
        mask_token_id=mask_token_id,
        block_size=student.block_size,
        num_anchors=args.num_anchors,
        loss_decay_gamma=args.stage2_loss_decay_gamma,
        final_ce_weight=args.stage2_final_ce_weight,
        tv_loss_weight=args.stage2_tv_weight,
        base_lm_ce_weight=args.stage2_base_ce_weight,
        base_lm_ce_decay_gamma=args.stage2_base_ce_decay_gamma,
    )
    fsdp = FSDP(
        student_online,
        ignored_modules=[components.lm_head, components.embed_tokens],
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    )
    stage1_optimizer = None
    if resume_stage in (None, "stage1"):
        stage1_optimizer = BF16Optimizer(
            student,
            lr=args.stage1_learning_rate,
            max_grad_norm=args.max_grad_norm,
            warmup_ratio=args.stage1_warmup_ratio,
            total_steps=stage_total_steps(
                stage1_dataloader,
                args.stage1_epochs,
                args.accumulation_steps,
            ),
        )
        if resume_stage == "stage1":
            stage1_optimizer.load_state_dict(resume_state)
    tracker = create_tracker(args, args.output_dir)
    if resume_stage == "stage1":
        stage1_start_epoch, stage1_start_batch, stage1_step, global_step = resume_cursor(
            resume_state, "stage1"
        )
    elif resume_state is not None:
        stage1_start_epoch = stage1_start_batch = stage1_step = 0
        global_step = int(resume_state.get("global_step", 0))
    else:
        stage1_start_epoch = stage1_start_batch = stage1_step = global_step = 0
    micro_steps = 0
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(stage1_start_epoch, args.stage1_epochs) if stage1_optimizer is not None else ():
        stage1_dataloader.sampler.set_epoch(epoch)
        student.train()
        teacher.eval()
        iterator = tqdm(stage1_dataloader, desc=f"Stage1 epoch {epoch}") if dist.get_rank() == 0 else stage1_dataloader
        for batch_idx, data in enumerate(iterator):
            if epoch == stage1_start_epoch and batch_idx < stage1_start_batch:
                continue
            global_step += 1
            stage1_step += 1
            input_ids = data["input_ids"].cuda()
            attention_mask = data["attention_mask"].cuda()
            loss_mask = data["loss_mask"].cuda()
            anchors, block_keep = student_online.sample_anchor_positions(
                input_ids.size(1), loss_mask
            )
            target_output = target.generate_flashmtp_data(
                input_ids, attention_mask, loss_mask, return_logits=False
            )
            hidden_states = hidden_states_to_cuda(target_output.hidden_states)
            del target_output
            if tp_draft_rank is not None:
                # The TP target sees the shared full batch.  From this point on,
                # teacher and student on rank r both consume only sample r.
                input_ids = select_tp_rank_batch(input_ids, tp_draft_rank)
                loss_mask = select_tp_rank_batch(loss_mask, tp_draft_rank)
                anchors = select_tp_rank_batch(anchors, tp_draft_rank)
                block_keep = select_tp_rank_batch(block_keep, tp_draft_rank)
                hidden_states = select_tp_rank_batch(
                    hidden_states, tp_draft_rank
                )
            student_batch = student_online.prepare_batch(
                input_ids,
                hidden_states,
                loss_mask,
                anchor_positions=anchors,
                block_keep_mask=block_keep,
            )
            with torch.no_grad():
                teacher_batch = teacher_online.prepare_batch(
                    input_ids,
                    hidden_states,
                    loss_mask,
                    anchor_positions=anchors,
                    block_keep_mask=block_keep,
                    shared_query_embeddings=student_batch.query_embeddings,
                )
                teacher_hidden = teacher_online.forward_backbone(
                    teacher_batch, seq_len=input_ids.size(1)
                )
            student_hidden = fsdp(
                prepared_batch=student_batch,
                seq_len=input_ids.size(1),
                return_backbone=True,
            )
            loss, tv_loss, hidden_loss, prefix_acc = compute_stage1_distillation_loss(
                student_hidden=student_hidden,
                teacher_hidden=teacher_hidden,
                lm_head=components.lm_head,
                raw_weight_mask=student_batch.raw_weight_mask,
                labels=student_batch.labels,
                tv_weight=args.stage1_tv_weight,
                hidden_weight=args.stage1_hidden_weight,
                smooth_l1_beta=args.stage1_smooth_l1_beta,
                loss_decay_gamma=args.stage1_loss_decay_gamma,
            )
            del hidden_states, teacher_hidden, student_hidden, teacher_batch, student_batch
            (loss / args.accumulation_steps).backward()
            micro_steps += 1
            grad_norm = None
            if micro_steps == args.accumulation_steps:
                grad_norm = stage1_optimizer.step()
                micro_steps = 0
            if global_step % args.log_interval == 0:
                metrics = torch.stack(
                    [
                        loss.detach(),
                        tv_loss.detach(),
                        hidden_loss.detach(),
                        prefix_acc.detach(),
                    ]
                )
                dist.all_reduce(metrics)
                metrics /= dist.get_world_size()
                payload = {
                    "stage1/loss": metrics[0].item(),
                    "stage1/tv": metrics[1].item(),
                    "stage1/hidden": metrics[2].item(),
                    "stage1/prefix_acc": metrics[3].item(),
                    "stage1/lr": stage1_optimizer.get_learning_rate(),
                }
                if grad_norm is not None:
                    payload["stage1/grad_norm"] = grad_norm
                tracker.log(payload, step=global_step)
                print_on_rank0(
                    f"stage1 step={global_step} loss={metrics[0]:.4f} "
                    f"prefix_acc={metrics[3]:.4f}"
                )
            if global_step % args.save_interval == 0 and micro_steps == 0:
                save_checkpoint(
                    output_dir=args.output_dir,
                    name=f"stage1/epoch_{epoch}_step_{stage1_step}",
                    fsdp_model=fsdp,
                    draft_model=student,
                    optimizer=stage1_optimizer,
                    metadata={
                        "training_stage": "stage1",
                        "stage_epoch": epoch,
                        "next_batch_in_epoch": batch_idx + 1,
                        "stage_step": stage1_step,
                        "global_step": global_step,
                        "serial_head_inherited": False,
                        "student_init_mode": student_init_mode,
                        "shared_backbone_inherited": student_init_mode == "shared_init",
                        "teacher_checkpoint_identity": teacher_identity,
                        "shard_draft_by_tp": bool(args.shard_draft_by_tp),
                        "tp_size": int(args.tp_size),
                        "stage1_train_data_identity": stage1_data_identity,
                        "stage2_train_data_identity": stage2_data_identity,
                    },
                )
        stage1_start_batch = 0

    if stage1_optimizer is not None:
        if micro_steps:
            stage1_optimizer.scale_model_gradients(args.accumulation_steps / micro_steps)
            stage1_optimizer.step()
            micro_steps = 0
        save_checkpoint(
            output_dir=args.output_dir,
            name="stage1/final",
            fsdp_model=fsdp,
            draft_model=student,
            optimizer=stage1_optimizer,
            metadata={
                "training_stage": "stage1",
                "stage_epoch": args.stage1_epochs,
                "next_batch_in_epoch": 0,
                "stage_step": stage1_step,
                "global_step": global_step,
                "serial_head_inherited": False,
                "student_init_mode": student_init_mode,
                "shared_backbone_inherited": student_init_mode == "shared_init",
                "teacher_checkpoint_identity": teacher_identity,
                "shard_draft_by_tp": bool(args.shard_draft_by_tp),
                "tp_size": int(args.tp_size),
                "stage1_train_data_identity": stage1_data_identity,
                "stage2_train_data_identity": stage2_data_identity,
            },
        )
        memory = log_cuda_peak("stage1")
        tracker.log(
            {
                "stage1/cuda_peak_allocated_gib": memory["allocated_gib"],
                "stage1/cuda_peak_reserved_gib": memory["reserved_gib"],
            },
            step=global_step,
        )

        with FSDP.summon_full_params(fsdp, writeback=True):
            _copy_serial_head(teacher, student)
        _set_student_stage2_trainable(student)
        save_checkpoint(
            output_dir=args.output_dir,
            name="transition",
            fsdp_model=fsdp,
            draft_model=student,
            optimizer=stage1_optimizer,
            metadata={
                "training_stage": "transition",
                "stage_epoch": 0,
                "next_batch_in_epoch": 0,
                "stage_step": 0,
                "global_step": global_step,
                "serial_head_inherited": True,
                "student_init_mode": student_init_mode,
                "shared_backbone_inherited": student_init_mode == "shared_init",
                "teacher_checkpoint_identity": teacher_identity,
                "shard_draft_by_tp": bool(args.shard_draft_by_tp),
                "tp_size": int(args.tp_size),
                "stage1_train_data_identity": stage1_data_identity,
                "stage2_train_data_identity": stage2_data_identity,
            },
        )
    # drafts_for_target also owns the teacher.  Keeping that list alive would
    # silently retain the full teacher on every rank throughout Stage 2.
    del teacher_online, teacher, stage1_optimizer, drafts_for_target
    target.set_capture_layers(student.target_layer_ids)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    post_release_allocated = torch.cuda.memory_allocated() / 1024**3
    post_release_reserved = torch.cuda.memory_reserved() / 1024**3
    print_on_rank0(
        "After teacher release: "
        f"allocated={post_release_allocated:.2f} GiB, "
        f"reserved={post_release_reserved:.2f} GiB"
    )
    tracker.log(
        {
            "stage2/post_teacher_release_allocated_gib": post_release_allocated,
            "stage2/post_teacher_release_reserved_gib": post_release_reserved,
        },
        step=global_step,
    )

    stage2_optimizer = BF16Optimizer(
        student,
        lr=args.stage2_learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.stage2_warmup_ratio,
        total_steps=stage_total_steps(
            stage2_dataloader,
            args.stage2_epochs,
            args.accumulation_steps,
        ),
    )
    if resume_stage == "stage2":
        stage2_optimizer.load_state_dict(resume_state)
        stage2_start_epoch, stage2_start_batch, stage2_step, restored_global = resume_cursor(
            resume_state, "stage2"
        )
        global_step = restored_global
    else:
        stage2_start_epoch = stage2_start_batch = stage2_step = 0

    for epoch in range(stage2_start_epoch, args.stage2_epochs):
        stage2_dataloader.sampler.set_epoch(epoch)
        student.train()
        iterator = tqdm(stage2_dataloader, desc=f"Stage2 epoch {epoch}") if dist.get_rank() == 0 else stage2_dataloader
        for batch_idx, data in enumerate(iterator):
            if epoch == stage2_start_epoch and batch_idx < stage2_start_batch:
                continue
            global_step += 1
            stage2_step += 1
            input_ids = data["input_ids"].cuda()
            attention_mask = data["attention_mask"].cuda()
            loss_mask = data["loss_mask"].cuda()
            anchors, block_keep = student_online.sample_anchor_positions(
                input_ids.size(1), loss_mask
            )
            target_output = target.generate_flashmtp_data(
                input_ids, attention_mask, loss_mask
            )
            hidden_states = hidden_states_to_cuda(target_output.hidden_states)
            target_prefill_logits = target_output.logits.cuda()
            if tp_draft_rank is not None:
                input_ids = select_tp_rank_batch(input_ids, tp_draft_rank)
                loss_mask = select_tp_rank_batch(loss_mask, tp_draft_rank)
                anchors = select_tp_rank_batch(anchors, tp_draft_rank)
                block_keep = select_tp_rank_batch(block_keep, tp_draft_rank)
                hidden_states = select_tp_rank_batch(
                    hidden_states, tp_draft_rank
                )
                target_prefill_logits = select_tp_rank_batch(
                    target_prefill_logits, tp_draft_rank
                )
            target_logits = gather_target_prefill_logits(
                target_prefill_logits, anchors, student.block_size
            )
            del target_output, target_prefill_logits
            prepared = student_online.prepare_batch(
                input_ids,
                hidden_states,
                loss_mask,
                anchor_positions=anchors,
                block_keep_mask=block_keep,
            )
            loss, accuracy, prefix_acc, final_ce, base_ce, tv_loss = fsdp(
                prepared_batch=prepared,
                seq_len=input_ids.size(1),
                target_prefill_logits=target_logits,
                target_logits_are_gathered=True,
            )
            del hidden_states, target_logits, prepared
            (loss / args.accumulation_steps).backward()
            micro_steps += 1
            grad_norm = None
            if micro_steps == args.accumulation_steps:
                grad_norm = stage2_optimizer.step()
                micro_steps = 0
            if global_step % args.log_interval == 0:
                metrics = torch.stack(
                    [loss.detach(), accuracy, prefix_acc, final_ce.detach(), base_ce.detach(), tv_loss.detach()]
                )
                dist.all_reduce(metrics)
                metrics /= dist.get_world_size()
                payload = {
                    "stage2/loss": metrics[0].item(),
                    "stage2/accuracy": metrics[1].item(),
                    "stage2/prefix_acc": metrics[2].item(),
                    "stage2/final_ce": metrics[3].item(),
                    "stage2/base_ce": metrics[4].item(),
                    "stage2/tv": metrics[5].item(),
                    "stage2/lr": stage2_optimizer.get_learning_rate(),
                }
                if grad_norm is not None:
                    payload["stage2/grad_norm"] = grad_norm
                tracker.log(payload, step=global_step)
                print_on_rank0(f"stage2 step={global_step} loss={metrics[0]:.4f}")
            if global_step % args.save_interval == 0 and micro_steps == 0:
                save_checkpoint(
                    output_dir=args.output_dir,
                    name=f"stage2/epoch_{epoch}_step_{stage2_step}",
                    fsdp_model=fsdp,
                    draft_model=student,
                    optimizer=stage2_optimizer,
                    metadata={
                        "training_stage": "stage2",
                        "stage_epoch": epoch,
                        "next_batch_in_epoch": batch_idx + 1,
                        "stage_step": stage2_step,
                        "global_step": global_step,
                        "serial_head_inherited": True,
                        "student_init_mode": student_init_mode,
                        "shared_backbone_inherited": student_init_mode == "shared_init",
                        "teacher_checkpoint_identity": teacher_identity,
                        "shard_draft_by_tp": bool(args.shard_draft_by_tp),
                        "tp_size": int(args.tp_size),
                        "stage1_train_data_identity": stage1_data_identity,
                        "stage2_train_data_identity": stage2_data_identity,
                    },
                )
        stage2_start_batch = 0

    if micro_steps:
        stage2_optimizer.scale_model_gradients(args.accumulation_steps / micro_steps)
        stage2_optimizer.step()
    save_checkpoint(
        output_dir=args.output_dir,
        name="final",
        fsdp_model=fsdp,
        draft_model=student,
        optimizer=stage2_optimizer,
        metadata={
            "training_stage": "stage2",
            "stage_epoch": args.stage2_epochs,
            "next_batch_in_epoch": 0,
            "stage_step": stage2_step,
            "global_step": global_step,
            "serial_head_inherited": True,
            "student_init_mode": student_init_mode,
            "shared_backbone_inherited": student_init_mode == "shared_init",
            "teacher_checkpoint_identity": teacher_identity,
            "shard_draft_by_tp": bool(args.shard_draft_by_tp),
            "tp_size": int(args.tp_size),
            "stage1_train_data_identity": stage1_data_identity,
            "stage2_train_data_identity": stage2_data_identity,
        },
    )
    memory = log_cuda_peak("stage2")
    tracker.log(
        {
            "stage2/cuda_peak_allocated_gib": memory["allocated_gib"],
            "stage2/cuda_peak_reserved_gib": memory["reserved_gib"],
        },
        step=global_step,
    )
    tracker.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
