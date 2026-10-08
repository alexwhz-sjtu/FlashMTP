"""Executable FSDP loop shared by DFlash, DFlash2, and DSpark."""

from __future__ import annotations

import argparse
import logging
import os

_project_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_global_rank = os.environ.get("RANK", "0")
for _name, _default in (
    ("TVM_FFI_CACHE_DIR", "tvm-ffi"),
    ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
    ("TRITON_CACHE_DIR", "triton"),
):
    _root = os.environ.get(f"{_name.removesuffix('_DIR')}_ROOT") or os.environ.get(_name)
    if not _root:
        _root = os.path.join(_project_dir, "cache", _default)
    os.environ[_name] = os.path.join(_root, f"rank_{_global_rank}")

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from tqdm import tqdm

from scripts.dflash_family.dflash_family_training import (
    add_family_args,
    build_family_components,
    build_family_dataloader,
    build_family_draft,
    build_online_model,
    family_training_metadata,
    load_family_draft,
    prepare_family_features,
    selector_alpha,
    sync_args_from_draft,
    validate_family_args,
)
from scripts.dlite.dlite_training import (
    load_training_state,
    log_cuda_peak,
    normalize_accumulated_gradients,
    resume_cursor,
    save_checkpoint,
    select_tp_rank_batch,
    stage_total_steps,
    training_data_identity,
    validate_tp_draft_sharding,
)
from specforge.distributed import destroy_distributed, init_distributed
from specforge.optimizer import BF16Optimizer
from specforge.tracker import create_tracker, get_tracker_class
from specforge.utils import print_on_rank0


def parse_args(algorithm: str):
    parser = argparse.ArgumentParser(description=f"Train {algorithm} draft model")
    add_family_args(parser, algorithm)
    args = parser.parse_args()
    validate_family_args(parser, args, algorithm)
    get_tracker_class(args.report_to).validate_args(parser, args)
    return args


def _metadata(args, algorithm, epoch, next_batch, step, optimizer_step, total_steps):
    return {
        "training_stage": algorithm,
        "algorithm": algorithm,
        "stage_epoch": int(epoch),
        "next_batch_in_epoch": int(next_batch),
        "stage_step": int(step),
        "global_step": int(step),
        "optimizer_step": int(optimizer_step),
        "scheduler_total_steps": int(total_steps),
        "scheduler_learning_rate": float(args.learning_rate),
        "scheduler_warmup_ratio": float(args.warmup_ratio),
        "train_data_identity": training_data_identity(args),
        "train_data_mode": "regen_full" if args.train_hidden_states_path else "online",
        "tp_size": int(args.tp_size),
        **family_training_metadata(args, algorithm, optimizer_step, total_steps),
    }


def _validate_resume(args, algorithm, state, total_steps):
    if state is None:
        return
    expected = {
        "training_stage": algorithm,
        "train_data_identity": training_data_identity(args),
        "scheduler_total_steps": int(total_steps),
        "tp_size": int(args.tp_size),
        "loss_config": family_training_metadata(
            args,
            algorithm,
            int(state.get("optimizer_step", 0)) if state else 0,
            total_steps,
        )["loss_config"],
    }
    for key, value in expected.items():
        if state.get(key, value) != value:
            raise ValueError(
                f"Resume mismatch for {key}: checkpoint={state.get(key)!r}, current={value!r}"
            )


@record
def run(algorithm: str) -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(algorithm)
    set_seed(args.seed)
    if args.disaggregate:
        from scripts.dflash_family.dflash_disaggregate import run_disaggregated

        run_disaggregated(args, algorithm)
        return

    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    tp_draft_rank = validate_tp_draft_sharding(args)
    resume_state = load_training_state(args.resume_from)
    source = args.resume_from or args.init_from
    if source:
        draft = load_family_draft(source, algorithm, args.attention_backend)
        sync_args_from_draft(args, draft)
        print_on_rank0(f"Loaded {algorithm} draft from {source}.")
    else:
        draft = build_family_draft(args, algorithm)
        print_on_rank0(f"Initialized {algorithm} draft from target config.")
    draft.requires_grad_(True)

    target, tokenizer, components, _ = build_family_components(args, draft)
    dataloader = build_family_dataloader(args, tokenizer, draft, algorithm)
    total_steps = stage_total_steps(
        dataloader, args.num_epochs, args.accumulation_steps
    )
    _validate_resume(args, algorithm, resume_state, total_steps)
    online = build_online_model(args, algorithm, draft, components)
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
        total_steps=total_steps,
    )
    optimizer_step = 0
    if resume_state is not None:
        optimizer.load_state_dict(resume_state)
        optimizer_step = int(resume_state.get("optimizer_step", 0))

    tracker = create_tracker(args, args.output_dir)
    start_epoch, start_batch, train_step, _ = resume_cursor(resume_state, algorithm)
    micro_steps = 0
    denominator_sum = None
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, args.num_epochs):
        dataloader.sampler.set_epoch(epoch)
        draft.train()
        iterator = tqdm(dataloader, desc=f"{algorithm} epoch {epoch}") if dist.get_rank() == 0 else dataloader
        for batch_idx, data in enumerate(iterator):
            if epoch == start_epoch and batch_idx < start_batch:
                continue
            train_step += 1
            input_ids = data["input_ids"].cuda(non_blocking=True)
            attention_mask = data["attention_mask"].cuda(non_blocking=True)
            loss_mask = data["loss_mask"].cuda(non_blocking=True)
            target_output = None
            if target is not None:
                target_output = target.generate_dlite_data(
                    input_ids, attention_mask, loss_mask, return_logits=False
                )
            context_hidden, target_last_hidden = prepare_family_features(
                data, target_output, draft, args.require_target_last_hidden
            )
            if tp_draft_rank is not None:
                input_ids = select_tp_rank_batch(input_ids, tp_draft_rank)
                loss_mask = select_tp_rank_batch(loss_mask, tp_draft_rank)
                context_hidden = select_tp_rank_batch(context_hidden, tp_draft_rank)
                if target_last_hidden is not None:
                    target_last_hidden = select_tp_rank_batch(
                        target_last_hidden, tp_draft_rank
                    )
            kwargs = dict(
                input_ids=input_ids,
                hidden_states=context_hidden,
                loss_mask=loss_mask,
                target_last_hidden_states=target_last_hidden,
                collect_detailed_metrics=(train_step % args.log_interval == 0),
            )
            if algorithm == "dflash2":
                kwargs["selector_loss_alpha"] = selector_alpha(
                    args, optimizer_step, total_steps
                )
            loss, accuracy, metrics = fsdp(**kwargs)
            loss_numerator, loss_denominator = metrics["loss_terms"]
            (loss_numerator / args.accumulation_steps).backward()
            denominator_sum = (
                loss_denominator.detach().clone()
                if denominator_sum is None
                else denominator_sum + loss_denominator.detach()
            )
            micro_steps += 1
            grad_norm = None
            if micro_steps == args.accumulation_steps:
                normalize_accumulated_gradients(
                    optimizer,
                    denominator_sum,
                    args.accumulation_steps,
                    group=fsdp.process_group,
                )
                grad_norm = optimizer.step()
                optimizer_step += 1
                micro_steps = 0
                denominator_sum = None
            if train_step % args.log_interval == 0:
                values = torch.stack([loss.detach(), accuracy.detach()])
                dist.all_reduce(values)
                values /= dist.get_world_size()
                payload = {
                    "train/loss": values[0].item(),
                    "train/accuracy": values[1].item(),
                    "train/lr": optimizer.get_learning_rate(),
                }
                if algorithm == "dflash2":
                    payload["train/selector_alpha"] = float(
                        metrics.get("selector_loss_alpha", 0.0)
                    )
                if grad_norm is not None:
                    payload["train/grad_norm"] = grad_norm
                tracker.log(payload, step=train_step)
                print_on_rank0(
                    f"{algorithm} step={train_step} loss={values[0]:.4f} acc={values[1]:.4f}"
                )
            if train_step % args.save_interval == 0 and micro_steps == 0:
                save_checkpoint(
                    output_dir=args.output_dir,
                    name=f"epoch_{epoch}_step_{train_step}",
                    fsdp_model=fsdp,
                    draft_model=draft,
                    optimizer=optimizer,
                    metadata=_metadata(
                        args, algorithm, epoch, batch_idx + 1, train_step, optimizer_step, total_steps
                    ),
                )
        start_batch = 0
    if micro_steps:
        normalize_accumulated_gradients(
            optimizer, denominator_sum, args.accumulation_steps, group=fsdp.process_group
        )
        optimizer.step()
        optimizer_step += 1
    save_checkpoint(
        output_dir=args.output_dir,
        name="final",
        fsdp_model=fsdp,
        draft_model=draft,
        optimizer=optimizer,
        metadata=_metadata(
            args, algorithm, args.num_epochs, 0, train_step, optimizer_step, total_steps
        ),
    )
    memory = log_cuda_peak(algorithm)
    tracker.log(
        {
            "train/cuda_peak_allocated_gib": memory["allocated_gib"],
            "train/cuda_peak_reserved_gib": memory["reserved_gib"],
        },
        step=train_step,
    )
    tracker.close()
    destroy_distributed()


__all__ = ["parse_args", "run"]
