#!/usr/bin/env python3
"""Train a DLite student with optional teacher distillation."""

import argparse
import copy
import gc
import logging
import math
import os

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp import (
    MixedPrecision,
    ShardingStrategy,
)
from tqdm import tqdm

from scripts.dlite_training import (
    add_common_args,
    build_draft_model,
    build_target_and_components,
    build_train_dataloader,
    hidden_states_to_cuda,
    load_cached_target_data,
    load_training_state,
    log_cuda_peak,
    normalize_accumulated_gradients,
    required_hidden_layer_ids,
    resume_cursor,
    save_checkpoint,
    select_tp_rank_batch,
    stage_total_steps,
    training_data_identity,
    validate_common_args,
    validate_tp_draft_sharding,
)
from specforge.core.dlite import (
    OnlineDLiteModel,
    compute_stage1_distillation_loss,
    gather_target_prefill_logits,
)
from specforge.distributed import destroy_distributed, init_distributed
from specforge.modeling.draft.dlite import DLiteDraftModel
from specforge.optimizer import BF16Optimizer
from specforge.tracker import create_tracker, get_tracker_class
from specforge.utils import print_on_rank0

TRANSITION_EPOCHS = 1


def parse_args():
    parser = argparse.ArgumentParser(description="Two-stage DLite distillation")
    add_common_args(parser)
    parser.add_argument("--teacher-draft-path")
    parser.add_argument("--stage1-epochs", type=int, required=True)
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Base LR for the single optimizer/scheduler shared by both stages.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        help="Warmup ratio over the combined Stage1+Stage2 optimizer steps.",
    )
    parser.add_argument("--stage1-kl-weight", type=float, default=1.0)
    parser.add_argument("--stage1-loss-decay-gamma", type=float)
    parser.add_argument("--stage2-epochs", type=int, required=True)
    parser.add_argument("--stage2-final-ce-weight", type=float, default=1.0)
    parser.add_argument("--stage2-tv-weight", type=float, default=1.0)
    parser.add_argument("--stage2-base-ce-weight", type=float, default=0.0)
    parser.add_argument("--stage2-loss-decay-gamma", type=float)
    parser.add_argument("--stage2-base-ce-decay-gamma", type=float)
    args = parser.parse_args()
    validate_common_args(parser, args)
    for name in ("stage2_epochs", "accumulation_steps"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.stage1_epochs <= 0:
        parser.error("--stage1-epochs must be positive")
    if args.learning_rate is None or args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.warmup_ratio is None or not 0.0 <= args.warmup_ratio <= 1.0:
        parser.error("--warmup-ratio must be in [0, 1]")
    if args.stage1_kl_weight <= 0:
        parser.error("--stage1-kl-weight must be positive")
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


def _sync_args_from_model(args, draft: DLiteDraftModel) -> None:
    args.dlite_version = draft.architecture_version
    args.block_size = draft.block_size
    args.num_draft_layers = draft.config.num_hidden_layers
    args.swa_window_size = draft.swa_window_size
    args.chs_num_layers = draft.chs_num_layers
    args.target_layer_ids = ",".join(str(value) for value in draft.target_layer_ids)
    args.sequential_head = draft.sequential_head_type
    args.sequential_rank = draft.sequential_rank
    draft.config._attn_implementation = "flex_attention"


def _structure_signature(draft: DLiteDraftModel) -> tuple:
    return (
        draft.architecture_version,
        draft.swa_window_size,
        draft.chs_num_layers,
        draft.block_size,
        draft.config.num_hidden_layers,
        draft.sequential_head_type,
        draft.sequential_rank,
        draft.config.vocab_size,
    )


def _non_depth_structure_signature(draft: DLiteDraftModel) -> tuple:
    signature = _structure_signature(draft)
    return signature[:4] + signature[5:]


def _set_student_stage1_trainable(student: DLiteDraftModel) -> None:
    student.requires_grad_(True)
    if student.sequential_head is not None:
        student.sequential_head.requires_grad_(False)


def _set_student_stage2_trainable(student: DLiteDraftModel) -> None:
    student.requires_grad_(True)


def _cosine_transition_scales(batch_idx: int, num_batches: int) -> tuple[float, float]:
    """Fade Stage 1 out and Stage 2 in over one transition epoch."""
    if num_batches <= 0:
        raise ValueError(f"Transition dataloader must be non-empty, got {num_batches}")
    if not 0 <= int(batch_idx) < int(num_batches):
        raise ValueError(
            f"Transition batch index must be in [0, {num_batches}), got {batch_idx}"
        )
    if num_batches == 1:
        return 0.5, 0.5
    progress = float(batch_idx) / float(num_batches - 1)
    stage2_scale = 0.5 * (1.0 - math.cos(math.pi * progress))
    return 1.0 - stage2_scale, stage2_scale


def _copy_serial_head(teacher: DLiteDraftModel, student: DLiteDraftModel) -> None:
    teacher_signature = (
        teacher.architecture_version,
        teacher.sequential_head_type,
        teacher.sequential_rank,
        teacher.block_size,
        teacher.config.vocab_size,
    )
    student_signature = (
        student.architecture_version,
        student.sequential_head_type,
        student.sequential_rank,
        student.block_size,
        student.config.vocab_size,
    )
    if teacher_signature != student_signature:
        raise ValueError(
            f"Teacher/student serial configuration mismatch: {teacher_signature} != {student_signature}"
        )
    if teacher.sequential_head is None or student.sequential_head is None:
        if teacher.sequential_head is not student.sequential_head:
            raise ValueError("Teacher/student serial heads do not match")
        return
    student.sequential_head.load_state_dict(
        teacher.sequential_head.state_dict(), strict=True
    )


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
    resume_transition_complete = (
        resume_stage == "transition"
        and int(resume_state.get("stage_epoch", 0)) >= TRANSITION_EPOCHS
    )
    if (
        resume_state is not None
        and "shard_draft_by_tp" in resume_state
        and bool(resume_state["shard_draft_by_tp"]) != bool(args.shard_draft_by_tp)
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
    train_data_identity = training_data_identity(args)
    train_data_mode = "regen_full" if args.train_hidden_states_path else "online"
    if (
        resume_state is not None
        and resume_state.get("train_data_identity") is not None
        and resume_state["train_data_identity"] != train_data_identity
    ):
        raise ValueError(
            "Training dataset must match the resumed checkpoint: "
            f"saved={resume_state['train_data_identity']!r}, "
            f"provided={train_data_identity!r}."
        )
    if (
        resume_state is not None
        and resume_state.get("train_data_mode", train_data_mode) != train_data_mode
    ):
        raise ValueError("Training data mode must match the resumed checkpoint.")
    print_on_rank0("Student init mode: scratch")
    print_on_rank0(
        "Continuous two-stage LR schedule: "
        f"lr={args.learning_rate:g}, warmup_ratio={args.warmup_ratio:g}"
    )
    provided_teacher_identity = (
        os.path.realpath(args.teacher_draft_path) if args.teacher_draft_path else None
    )
    saved_teacher_identity = (
        None
        if resume_state is None
        else resume_state.get("teacher_checkpoint_identity")
    )
    if (
        (
            resume_stage == "stage1"
            or (resume_stage == "transition" and not resume_transition_complete)
        )
        and saved_teacher_identity is not None
        and saved_teacher_identity != provided_teacher_identity
    ):
        raise ValueError(
            "Stage 1/transition must resume with the same teacher checkpoint: "
            f"saved={saved_teacher_identity!r}, provided={provided_teacher_identity!r}."
        )
    teacher_identity = saved_teacher_identity or provided_teacher_identity

    teacher = None
    if resume_stage in (None, "stage1"):
        if not args.teacher_draft_path:
            raise ValueError(
                "--teacher-draft-path is required for fresh or Stage 1 training"
            )
        teacher = (
            DLiteDraftModel.from_pretrained(
                args.teacher_draft_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flex_attention",
            )
            .cuda()
            .eval()
        )
        if not teacher.is_teacher:
            raise ValueError(
                "--teacher-draft-path must contain an swa_teacher checkpoint"
            )
        teacher.requires_grad_(False)
        _sync_args_from_model(args, teacher)
        if resume_stage == "stage1":
            student = DLiteDraftModel.from_pretrained(
                args.resume_from,
                torch_dtype=torch.bfloat16,
                attn_implementation="flex_attention",
            ).cuda()
            if not student.is_student:
                raise ValueError("Stage 1 checkpoint must contain a pivot_q_student")
            if _structure_signature(student) != _structure_signature(teacher):
                raise ValueError(
                    "Stage 1 student structure no longer matches the teacher"
                )
            args.num_draft_layers = student.config.num_hidden_layers
        else:
            student_config = copy.deepcopy(teacher.config)
            student = build_draft_model(
                args,
                model_role="pivot_q_student",
                source_config=student_config,
            )
            args.num_draft_layers = student.config.num_hidden_layers

        # Stage 1 always distills through the teacher's trained serial head.
        # The student owns a copy so it is checkpointed and can be unfrozen in
        # Stage 2 without retaining the teacher model.
        _copy_serial_head(teacher, student)
        print_on_rank0(
            "Initialized student serial head from teacher; frozen during Stage 1"
        )
    else:
        student = DLiteDraftModel.from_pretrained(
            args.resume_from,
            torch_dtype=torch.bfloat16,
            attn_implementation="flex_attention",
        ).cuda()
        if not student.is_student:
            raise ValueError(
                "Transition/Stage 2 checkpoint must contain a pivot_q_student"
            )
        if not bool(resume_state.get("serial_head_inherited")):
            raise ValueError(
                "Transition/Stage 2 checkpoint has no initialized serial head"
            )
        _sync_args_from_model(args, student)

        if resume_stage == "transition" and not resume_transition_complete:
            if not args.teacher_draft_path:
                raise ValueError(
                    "--teacher-draft-path is required to resume the transition epoch"
                )
            teacher = (
                DLiteDraftModel.from_pretrained(
                    args.teacher_draft_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flex_attention",
                )
                .cuda()
                .eval()
            )
            if not teacher.is_teacher:
                raise ValueError(
                    "--teacher-draft-path must contain an swa_teacher checkpoint"
                )
            teacher.requires_grad_(False)
            if _non_depth_structure_signature(
                student
            ) != _non_depth_structure_signature(teacher):
                raise ValueError(
                    "Transition student structure except draft depth must match "
                    "the teacher"
                )

    # Wrap FSDP and construct the optimizer with every parameter that may be
    # trained in either stage. Stage 1 freezes the serial head only after this
    # complete parameter set has been registered with both.
    _set_student_stage2_trainable(student)
    drafts_for_target = [student] if teacher is None else [teacher, student]
    target, tokenizer, components, mask_token_id = build_target_and_components(
        args, drafts_for_target
    )
    needs_stage1_dataloader = resume_stage in (None, "stage1") or (
        resume_stage == "transition" and not resume_transition_complete
    )
    train_dataloader = build_train_dataloader(
        args,
        tokenizer,
        train_data_path=args.train_data_path,
        cache_namespace="train",
        num_proc=args.build_dataset_num_proc,
        required_layer_ids=required_hidden_layer_ids(drafts_for_target),
    )
    stage1_dataloader = train_dataloader if needs_stage1_dataloader else None
    stage2_dataloader = train_dataloader

    teacher_online = None
    if teacher is not None:
        teacher_online = OnlineDLiteModel(
            draft_model=teacher,
            target_lm_head=components.lm_head,
            target_embed_tokens=components.embed_tokens,
            mask_token_id=mask_token_id,
            block_size=teacher.block_size,
            num_anchors=args.num_anchors,
        )
    student_online = OnlineDLiteModel(
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
        use_target_greedy_ce_labels=True,
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
    tracker = create_tracker(args, args.output_dir)
    if resume_stage == "stage1":
        stage1_start_epoch, stage1_start_batch, stage1_step, global_step = (
            resume_cursor(resume_state, "stage1")
        )
    elif resume_state is not None:
        stage1_start_epoch = stage1_start_batch = stage1_step = 0
        global_step = int(resume_state.get("global_step", 0))
    else:
        stage1_start_epoch = stage1_start_batch = stage1_step = global_step = 0
    stage2_schedule_steps = stage_total_steps(
        stage2_dataloader,
        args.stage2_epochs,
        args.accumulation_steps,
    )
    transition_schedule_steps = (
        stage_total_steps(
            stage1_dataloader,
            TRANSITION_EPOCHS,
            args.accumulation_steps,
        )
        if stage1_dataloader is not None
        else 0
    )
    if stage1_dataloader is not None:
        stage1_schedule_steps = stage_total_steps(
            stage1_dataloader,
            args.stage1_epochs,
            args.accumulation_steps,
        )
        computed_schedule_steps = (
            stage1_schedule_steps + transition_schedule_steps + stage2_schedule_steps
        )
    else:
        stage1_schedule_steps = 0
        completed_legacy_steps = (
            int(resume_state.get("optimizer_step", 0))
            if resume_state is not None
            and resume_state.get("optimizer_step") is not None
            else (global_step + args.accumulation_steps - 1) // args.accumulation_steps
        )
        computed_schedule_steps = completed_legacy_steps + stage2_schedule_steps
    continuous_checkpoint = bool(
        resume_state is not None and resume_state.get("continuous_two_stage_scheduler")
    )
    if (
        continuous_checkpoint
        and resume_stage in ("stage1", "transition")
        and int(resume_state.get("transition_epochs", 0)) != TRANSITION_EPOCHS
    ):
        raise ValueError(
            "The resumed checkpoint predates the cosine transition epoch. "
            "Resume from Stage 2 or start a fresh run with this training version."
        )
    if continuous_checkpoint:
        for key, requested in (
            ("scheduler_learning_rate", args.learning_rate),
            ("scheduler_warmup_ratio", args.warmup_ratio),
        ):
            if resume_state.get(key) is not None and float(resume_state[key]) != float(
                requested
            ):
                raise ValueError(
                    "Continuous scheduler configuration must match the resumed "
                    f"checkpoint: {key} saved={resume_state[key]!r}, "
                    f"requested={requested!r}."
                )
    scheduler_total_steps = (
        int(resume_state["scheduler_total_steps"])
        if continuous_checkpoint
        else computed_schedule_steps
    )
    if scheduler_total_steps <= 0:
        raise ValueError("The combined Stage1+Stage2 schedule has no optimizer steps.")
    if (
        continuous_checkpoint
        and stage1_dataloader is not None
        and scheduler_total_steps != computed_schedule_steps
    ):
        raise ValueError(
            "Combined scheduler length must match the resumed checkpoint: "
            f"saved={scheduler_total_steps}, requested={computed_schedule_steps}."
        )

    # Register the complete Stage-2 trainable set once, then freeze the serial
    # head for Stage 1 without rebuilding either optimizer or scheduler.
    _set_student_stage1_trainable(student)
    stage1_optimizer_parameters = [
        parameter for parameter in student.parameters() if parameter.requires_grad
    ]
    stage1_parameter_ids = {id(parameter) for parameter in stage1_optimizer_parameters}
    _set_student_stage2_trainable(student)
    stage2_optimizer_parameters = [
        parameter for parameter in student.parameters() if parameter.requires_grad
    ]
    # Preserve the exact legacy Stage-1 parameter prefix so its Adam moments
    # remain positionally compatible, then append parameters first unfrozen in
    # Stage 2 (the serial head).
    continuous_parameter_order = stage1_optimizer_parameters + [
        parameter
        for parameter in stage2_optimizer_parameters
        if id(parameter) not in stage1_parameter_ids
    ]
    optimizer_parameter_order = (
        str(resume_state.get("optimizer_parameter_order", "stage1_then_stage2"))
        if continuous_checkpoint
        else ("model_order" if resume_stage == "stage2" else "stage1_then_stage2")
    )
    if optimizer_parameter_order not in ("stage1_then_stage2", "model_order"):
        raise ValueError(
            "Unsupported optimizer_parameter_order in checkpoint: "
            f"{optimizer_parameter_order!r}."
        )
    optimizer_parameters = (
        stage2_optimizer_parameters
        if optimizer_parameter_order == "model_order"
        else continuous_parameter_order
    )
    optimizer = BF16Optimizer(
        student,
        parameters=optimizer_parameters,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=scheduler_total_steps,
    )
    optimizer_step = (
        int(resume_state.get("optimizer_step", 0))
        if resume_state is not None and resume_state.get("optimizer_step") is not None
        else (global_step + args.accumulation_steps - 1) // args.accumulation_steps
    )
    if resume_state is not None:
        optimizer.load_state_dict(
            resume_state,
            load_scheduler=continuous_checkpoint,
        )
        if not continuous_checkpoint:
            print_on_rank0(
                "Migrating legacy per-stage scheduler state to the continuous "
                "two-stage schedule."
            )
            optimizer.advance_scheduler(optimizer_step)
    if resume_stage in (None, "stage1"):
        _set_student_stage1_trainable(student)
    micro_steps = 0
    loss_denominator_sum = None
    torch.cuda.reset_peak_memory_stats()

    for epoch in (
        range(stage1_start_epoch, args.stage1_epochs)
        if resume_stage in (None, "stage1")
        else ()
    ):
        stage1_dataloader.sampler.set_epoch(epoch)
        student.train()
        teacher.eval()
        iterator = (
            tqdm(stage1_dataloader, desc=f"Stage1 epoch {epoch}")
            if dist.get_rank() == 0
            else stage1_dataloader
        )
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
            if args.train_hidden_states_path:
                hidden_states, _ = load_cached_target_data(
                    data,
                    anchors=anchors,
                    block_size=student.block_size,
                    lm_head=components.lm_head,
                    need_logits=False,
                )
            else:
                target_output = target.generate_dlite_data(
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
                hidden_states = select_tp_rank_batch(hidden_states, tp_draft_rank)
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
                teacher_serial_logits = teacher_online.compute_serial_logits(
                    teacher_hidden, teacher_batch
                )
            student_hidden, student_serial_logits = fsdp(
                prepared_batch=student_batch,
                seq_len=input_ids.size(1),
                return_backbone_and_serial_logits=True,
            )
            loss, kl_loss, loss_numerator, loss_denominator = (
                compute_stage1_distillation_loss(
                    student_serial_logits=student_serial_logits,
                    teacher_serial_logits=teacher_serial_logits,
                    raw_weight_mask=student_batch.raw_weight_mask,
                    kl_weight=args.stage1_kl_weight,
                    loss_decay_gamma=args.stage1_loss_decay_gamma,
                )
            )
            del (
                hidden_states,
                teacher_hidden,
                teacher_serial_logits,
                student_hidden,
                student_serial_logits,
                teacher_batch,
                student_batch,
            )
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
                metrics = torch.stack([loss.detach(), kl_loss.detach()])
                dist.all_reduce(metrics)
                metrics /= dist.get_world_size()
                payload = {
                    "train/loss": metrics[0].item(),
                    "train/kl": metrics[1].item(),
                    "train/lr": optimizer.get_learning_rate(),
                }
                if grad_norm is not None:
                    payload["train/grad_norm"] = grad_norm
                tracker.log(payload, step=global_step)
                print_on_rank0(f"stage1 step={global_step} loss={metrics[0]:.4f}")
            if global_step % args.save_interval == 0 and micro_steps == 0:
                save_checkpoint(
                    output_dir=args.output_dir,
                    name=f"stage1/epoch_{epoch}_step_{stage1_step}",
                    fsdp_model=fsdp,
                    draft_model=student,
                    optimizer=optimizer,
                    metadata={
                        "training_stage": "stage1",
                        "stage_epoch": epoch,
                        "next_batch_in_epoch": batch_idx + 1,
                        "stage_step": stage1_step,
                        "global_step": global_step,
                        "serial_head_inherited": True,
                        "continuous_two_stage_scheduler": True,
                        "transition_epochs": TRANSITION_EPOCHS,
                        "scheduler_total_steps": scheduler_total_steps,
                        "scheduler_learning_rate": args.learning_rate,
                        "scheduler_warmup_ratio": args.warmup_ratio,
                        "optimizer_step": optimizer_step,
                        "optimizer_parameter_order": optimizer_parameter_order,
                        "teacher_checkpoint_identity": teacher_identity,
                        "shard_draft_by_tp": bool(args.shard_draft_by_tp),
                        "tp_size": int(args.tp_size),
                        "train_data_identity": train_data_identity,
                        "train_data_mode": train_data_mode,
                    },
                )
        stage1_start_batch = 0

    if resume_stage in (None, "stage1"):
        if micro_steps:
            normalize_accumulated_gradients(
                optimizer,
                loss_denominator_sum,
                args.accumulation_steps,
                group=fsdp.process_group,
            )
            optimizer.step()
            optimizer_step += 1
            micro_steps = 0
            loss_denominator_sum = None
        save_checkpoint(
            output_dir=args.output_dir,
            name="stage1/final",
            fsdp_model=fsdp,
            draft_model=student,
            optimizer=optimizer,
            metadata={
                "training_stage": "stage1",
                "stage_epoch": args.stage1_epochs,
                "next_batch_in_epoch": 0,
                "stage_step": stage1_step,
                "global_step": global_step,
                "serial_head_inherited": True,
                "continuous_two_stage_scheduler": True,
                "transition_epochs": TRANSITION_EPOCHS,
                "scheduler_total_steps": scheduler_total_steps,
                "scheduler_learning_rate": args.learning_rate,
                "scheduler_warmup_ratio": args.warmup_ratio,
                "optimizer_step": optimizer_step,
                "optimizer_parameter_order": optimizer_parameter_order,
                "teacher_checkpoint_identity": teacher_identity,
                "shard_draft_by_tp": bool(args.shard_draft_by_tp),
                "tp_size": int(args.tp_size),
                "train_data_identity": train_data_identity,
                "train_data_mode": train_data_mode,
            },
        )
        memory = log_cuda_peak("stage1")
        tracker.log(
            {
                "train/cuda_peak_allocated_gib": memory["allocated_gib"],
                "train/cuda_peak_reserved_gib": memory["reserved_gib"],
            },
            step=global_step,
        )

        _set_student_stage2_trainable(student)
        save_checkpoint(
            output_dir=args.output_dir,
            name="transition/start",
            fsdp_model=fsdp,
            draft_model=student,
            optimizer=optimizer,
            metadata={
                "training_stage": "transition",
                "stage_epoch": 0,
                "next_batch_in_epoch": 0,
                "stage_step": 0,
                "global_step": global_step,
                "serial_head_inherited": True,
                "continuous_two_stage_scheduler": True,
                "transition_epochs": TRANSITION_EPOCHS,
                "scheduler_total_steps": scheduler_total_steps,
                "scheduler_learning_rate": args.learning_rate,
                "scheduler_warmup_ratio": args.warmup_ratio,
                "optimizer_step": optimizer_step,
                "optimizer_parameter_order": optimizer_parameter_order,
                "teacher_checkpoint_identity": teacher_identity,
                "shard_draft_by_tp": bool(args.shard_draft_by_tp),
                "tp_size": int(args.tp_size),
                "train_data_identity": train_data_identity,
                "train_data_mode": train_data_mode,
            },
        )

    if resume_stage == "transition":
        (
            transition_start_epoch,
            transition_start_batch,
            transition_step,
            restored_global,
        ) = resume_cursor(resume_state, "transition")
        global_step = restored_global
    else:
        transition_start_epoch = transition_start_batch = transition_step = 0

    if resume_stage in (None, "stage1", "transition"):
        _set_student_stage2_trainable(student)
    for epoch in (
        range(transition_start_epoch, TRANSITION_EPOCHS)
        if resume_stage in (None, "stage1", "transition")
        else ()
    ):
        if teacher is None or teacher_online is None:
            raise ValueError("The transition epoch requires the teacher model")
        stage1_dataloader.sampler.set_epoch(args.stage1_epochs + epoch)
        student.train()
        teacher.eval()
        iterator = (
            tqdm(stage1_dataloader, desc=f"Transition epoch {epoch}")
            if dist.get_rank() == 0
            else stage1_dataloader
        )
        num_transition_batches = len(stage1_dataloader)
        for batch_idx, data in enumerate(iterator):
            if epoch == transition_start_epoch and batch_idx < transition_start_batch:
                continue
            stage1_scale, stage2_scale = _cosine_transition_scales(
                batch_idx, num_transition_batches
            )
            global_step += 1
            transition_step += 1
            input_ids = data["input_ids"].cuda()
            attention_mask = data["attention_mask"].cuda()
            loss_mask = data["loss_mask"].cuda()
            anchors, block_keep = student_online.sample_anchor_positions(
                input_ids.size(1), loss_mask
            )
            if args.train_hidden_states_path:
                hidden_states, target_logits = load_cached_target_data(
                    data,
                    anchors=anchors,
                    block_size=student.block_size,
                    lm_head=components.lm_head,
                    need_logits=True,
                )
            else:
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
            if not args.train_hidden_states_path:
                target_logits = gather_target_prefill_logits(
                    target_prefill_logits, anchors, student.block_size
                )
                del target_output, target_prefill_logits
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
                teacher_serial_logits = teacher_online.compute_serial_logits(
                    teacher_hidden, teacher_batch
                )
            (
                student_hidden,
                student_serial_logits,
                stage2_outputs,
            ) = fsdp(
                prepared_batch=student_batch,
                seq_len=input_ids.size(1),
                target_prefill_logits=target_logits,
                target_logits_are_gathered=True,
                return_transition_outputs=True,
            )
            (
                stage2_loss,
                accuracy,
                stage2_prefix_acc,
                final_ce,
                base_ce,
                tv_loss,
                stage2_loss_numerator,
                stage2_loss_denominator,
            ) = stage2_outputs
            (
                stage1_loss,
                kl_loss,
                stage1_loss_numerator,
                stage1_loss_denominator,
            ) = compute_stage1_distillation_loss(
                student_serial_logits=student_serial_logits,
                teacher_serial_logits=teacher_serial_logits,
                raw_weight_mask=student_batch.raw_weight_mask,
                kl_weight=args.stage1_kl_weight,
                loss_decay_gamma=args.stage1_loss_decay_gamma,
            )
            loss = stage1_scale * stage1_loss + stage2_scale * stage2_loss
            loss_numerator = (
                stage1_scale * stage1_loss_numerator
                + stage2_scale * stage2_loss_numerator
            )
            loss_denominator = (
                stage1_scale * stage1_loss_denominator
                + stage2_scale * stage2_loss_denominator
            )
            del (
                hidden_states,
                target_logits,
                teacher_hidden,
                teacher_serial_logits,
                student_hidden,
                student_serial_logits,
                teacher_batch,
                student_batch,
                stage2_outputs,
            )
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
                        stage1_loss.detach(),
                        stage2_loss.detach(),
                        kl_loss.detach(),
                        tv_loss.detach(),
                        final_ce.detach(),
                        base_ce.detach(),
                        accuracy.detach(),
                        stage2_prefix_acc.detach(),
                        loss.new_tensor(stage1_scale),
                        loss.new_tensor(stage2_scale),
                    ]
                )
                dist.all_reduce(metrics)
                metrics /= dist.get_world_size()
                payload = {
                    "train/loss": metrics[0].item(),
                    "train/stage1_loss": metrics[1].item(),
                    "train/stage2_loss": metrics[2].item(),
                    "train/kl": metrics[3].item(),
                    "train/tv": metrics[4].item(),
                    "train/final_ce": metrics[5].item(),
                    "train/base_ce": metrics[6].item(),
                    "train/accuracy": metrics[7].item(),
                    "train/stage2_prefix_acc": metrics[8].item(),
                    # Keep the main prefix-accuracy curve continuous: Stage 1
                    # logs its prefix accuracy under this key, while the
                    # transition and Stage 2 use the Stage 2 definition.
                    "train/prefix_acc": metrics[8].item(),
                    "train/stage1_weight_scale": metrics[9].item(),
                    "train/stage2_weight_scale": metrics[10].item(),
                    "train/lr": optimizer.get_learning_rate(),
                }
                if grad_norm is not None:
                    payload["train/grad_norm"] = grad_norm
                tracker.log(payload, step=global_step)
                print_on_rank0(
                    f"transition step={global_step} loss={metrics[0]:.4f} "
                    f"stage1_scale={metrics[9]:.4f} "
                    f"stage2_scale={metrics[10]:.4f}"
                )
            if global_step % args.save_interval == 0 and micro_steps == 0:
                save_checkpoint(
                    output_dir=args.output_dir,
                    name=f"transition/epoch_{epoch}_step_{transition_step}",
                    fsdp_model=fsdp,
                    draft_model=student,
                    optimizer=optimizer,
                    metadata={
                        "training_stage": "transition",
                        "stage_epoch": epoch,
                        "next_batch_in_epoch": batch_idx + 1,
                        "stage_step": transition_step,
                        "global_step": global_step,
                        "serial_head_inherited": True,
                        "continuous_two_stage_scheduler": True,
                        "transition_epochs": TRANSITION_EPOCHS,
                        "scheduler_total_steps": scheduler_total_steps,
                        "scheduler_learning_rate": args.learning_rate,
                        "scheduler_warmup_ratio": args.warmup_ratio,
                        "optimizer_step": optimizer_step,
                        "optimizer_parameter_order": optimizer_parameter_order,
                        "teacher_checkpoint_identity": teacher_identity,
                        "shard_draft_by_tp": bool(args.shard_draft_by_tp),
                        "tp_size": int(args.tp_size),
                        "train_data_identity": train_data_identity,
                        "train_data_mode": train_data_mode,
                    },
                )
        transition_start_batch = 0

    if resume_stage in (None, "stage1", "transition"):
        if micro_steps:
            normalize_accumulated_gradients(
                optimizer,
                loss_denominator_sum,
                args.accumulation_steps,
                group=fsdp.process_group,
            )
            optimizer.step()
            optimizer_step += 1
            micro_steps = 0
            loss_denominator_sum = None
        save_checkpoint(
            output_dir=args.output_dir,
            name="transition",
            fsdp_model=fsdp,
            draft_model=student,
            optimizer=optimizer,
            metadata={
                "training_stage": "transition",
                "stage_epoch": TRANSITION_EPOCHS,
                "next_batch_in_epoch": 0,
                "stage_step": transition_step,
                "global_step": global_step,
                "serial_head_inherited": True,
                "continuous_two_stage_scheduler": True,
                "transition_epochs": TRANSITION_EPOCHS,
                "scheduler_total_steps": scheduler_total_steps,
                "scheduler_learning_rate": args.learning_rate,
                "scheduler_warmup_ratio": args.warmup_ratio,
                "optimizer_step": optimizer_step,
                "optimizer_parameter_order": optimizer_parameter_order,
                "teacher_checkpoint_identity": teacher_identity,
                "shard_draft_by_tp": bool(args.shard_draft_by_tp),
                "tp_size": int(args.tp_size),
                "train_data_identity": train_data_identity,
                "train_data_mode": train_data_mode,
            },
        )
        memory = log_cuda_peak("transition")
        tracker.log(
            {
                "train/cuda_peak_allocated_gib": memory["allocated_gib"],
                "train/cuda_peak_reserved_gib": memory["reserved_gib"],
            },
            step=global_step,
        )

    # drafts_for_target also owns the teacher.  Keeping that list alive would
    # silently retain the full teacher on every rank throughout Stage 2.
    del teacher_online, teacher, drafts_for_target
    if target is not None:
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
            "train/post_teacher_release_allocated_gib": post_release_allocated,
            "train/post_teacher_release_reserved_gib": post_release_reserved,
        },
        step=global_step,
    )

    if resume_stage == "stage2":
        stage2_start_epoch, stage2_start_batch, stage2_step, restored_global = (
            resume_cursor(resume_state, "stage2")
        )
        global_step = restored_global
    else:
        stage2_start_epoch = stage2_start_batch = stage2_step = 0

    for epoch in range(stage2_start_epoch, args.stage2_epochs):
        stage2_dataloader.sampler.set_epoch(epoch)
        student.train()
        iterator = (
            tqdm(stage2_dataloader, desc=f"Stage2 epoch {epoch}")
            if dist.get_rank() == 0
            else stage2_dataloader
        )
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
            if args.train_hidden_states_path:
                hidden_states, target_logits = load_cached_target_data(
                    data,
                    anchors=anchors,
                    block_size=student.block_size,
                    lm_head=components.lm_head,
                    need_logits=True,
                )
            else:
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
            if not args.train_hidden_states_path:
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
                prepared_batch=prepared,
                seq_len=input_ids.size(1),
                target_prefill_logits=target_logits,
                target_logits_are_gathered=True,
            )
            del hidden_states, target_logits, prepared
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
                print_on_rank0(f"stage2 step={global_step} loss={metrics[0]:.4f}")
            if global_step % args.save_interval == 0 and micro_steps == 0:
                save_checkpoint(
                    output_dir=args.output_dir,
                    name=f"stage2/epoch_{epoch}_step_{stage2_step}",
                    fsdp_model=fsdp,
                    draft_model=student,
                    optimizer=optimizer,
                    metadata={
                        "training_stage": "stage2",
                        "stage_epoch": epoch,
                        "next_batch_in_epoch": batch_idx + 1,
                        "stage_step": stage2_step,
                        "global_step": global_step,
                        "serial_head_inherited": True,
                        "continuous_two_stage_scheduler": True,
                        "transition_epochs": TRANSITION_EPOCHS,
                        "scheduler_total_steps": scheduler_total_steps,
                        "scheduler_learning_rate": args.learning_rate,
                        "scheduler_warmup_ratio": args.warmup_ratio,
                        "optimizer_step": optimizer_step,
                        "optimizer_parameter_order": optimizer_parameter_order,
                        "teacher_checkpoint_identity": teacher_identity,
                        "shard_draft_by_tp": bool(args.shard_draft_by_tp),
                        "tp_size": int(args.tp_size),
                        "train_data_identity": train_data_identity,
                        "train_data_mode": train_data_mode,
                    },
                )
        stage2_start_batch = 0

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
        metadata={
            "training_stage": "stage2",
            "stage_epoch": args.stage2_epochs,
            "next_batch_in_epoch": 0,
            "stage_step": stage2_step,
            "global_step": global_step,
            "serial_head_inherited": True,
            "continuous_two_stage_scheduler": True,
            "transition_epochs": TRANSITION_EPOCHS,
            "scheduler_total_steps": scheduler_total_steps,
            "scheduler_learning_rate": args.learning_rate,
            "scheduler_warmup_ratio": args.warmup_ratio,
            "optimizer_step": optimizer_step,
            "optimizer_parameter_order": optimizer_parameter_order,
            "teacher_checkpoint_identity": teacher_identity,
            "shard_draft_by_tp": bool(args.shard_draft_by_tp),
            "tp_size": int(args.tp_size),
            "train_data_identity": train_data_identity,
            "train_data_mode": train_data_mode,
        },
    )
    memory = log_cuda_peak("stage2")
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
