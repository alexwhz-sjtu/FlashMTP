#!/usr/bin/env python3
# coding=utf-8
"""FlashMTP v3.3 Phase 2: streak distillation (load phase-1 draft by default)."""

import argparse
import logging
import math
import os
import time
import warnings

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from tqdm import tqdm

from specforge.core.flashmtp_streak import FlashMTPStreakModel
from specforge.distributed import destroy_distributed, init_distributed
from specforge.flashmtp_train_utils import (
    add_flashmtp_common_args,
    build_dataloader,
    build_models,
    init_tokenizer_and_mask,
    load_draft_checkpoint,
    maybe_resume_ckpt,
    save_checkpoint,
)
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.optimizer import BF16Optimizer
from specforge.tracker import create_tracker
from specforge.utils import print_on_rank0, print_with_rank


def parse_args():
    p = argparse.ArgumentParser(description="FlashMTP Streak (v3.3 phase 2)")
    add_flashmtp_common_args(p)
    g = p.add_argument_group("streak")
    g.add_argument(
        "--init-ckpt",
        type=str,
        default=None,
        help="Phase-1 MDLM checkpoint dir (epoch_*_step_*). If unset, train from random init.",
    )
    g.add_argument("--log-prob-min", type=float, default=-40.0)
    g.add_argument(
        "--streak-weight",
        type=float,
        default=1.0,
        help="LS-RSL streak loss 系数；通常作为主 loss。",
    )
    g.add_argument(
        "--streak-ce-weight",
        type=float,
        default=0.0,
        help="块内除 anchor 外的逐位置平均 CE 系数；0 关闭。",
    )
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings(
        "ignore",
        "The .grad attribute of a Tensor that is not a leaf Tensor is being accessed",
    )
    args = parse_args()
    set_seed(args.seed)
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    print_with_rank("distributed ok")

    target_model, draft_model = build_models(args)
    tokenizer, mask_id = init_tokenizer_and_mask(args)
    draft_model.mask_token_id = mask_id
    draft_model.config.flashmtp_config["mask_token_id"] = mask_id

    ckpt_dir, resume_state = maybe_resume_ckpt(args, draft_model)
    if ckpt_dir:
        print_on_rank0(f"resume ckpt: {ckpt_dir}")
    elif args.init_ckpt and os.path.isdir(args.init_ckpt):
        load_draft_checkpoint(draft_model, args.init_ckpt)
        print_on_rank0(f"init from MDLM ckpt: {args.init_ckpt}")

    train_loader, _ = build_dataloader(args, tokenizer)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / args.accumulation_steps))
    total_steps = args.num_epochs * steps_per_epoch

    target_components = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model_path,
        embed_key="model.embed_tokens.weight",
        lm_head_key="lm_head.weight",
        device="cuda",
        trust_remote_code=args.trust_remote_code,
    )

    # 前向核心：块首 clean，其余 [MASK]；LS-RSL streak 是主项，CE_aux 是不调权逐位置辅助项。
    wrapper = FlashMTPStreakModel(
        draft_model=draft_model,
        target_lm_head=target_components.lm_head,
        target_embed_tokens=target_components.embed_tokens,
        mask_token_id=mask_id,
        block_size=draft_model.block_size,
        attention_backend=args.attention_backend,
        num_anchors=args.num_anchors,
        log_prob_min=args.log_prob_min,
        streak_weight=args.streak_weight,
        ce_aux_weight=args.streak_ce_weight,
    )
    wrapper = FSDP(
        wrapper,
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    )

    optimizer = BF16Optimizer(
        draft_model,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=total_steps,
    )
    start_epoch = 0
    global_step = 0
    skip_steps = 0
    if resume_state:
        optimizer.scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        if "optimizer_state_dict" in resume_state:
            optimizer.optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        start_epoch = resume_state["epoch"]
        global_step = resume_state["global_step"]
        skip_steps = global_step - start_epoch * len(train_loader)
        print_on_rank0(f"resume epoch={start_epoch} step={global_step}")

    tracker = create_tracker(args, args.output_dir)
    last_t = time.time()

    for epoch in range(start_epoch, args.num_epochs):
        train_loader.sampler.set_epoch(epoch)
        draft_model.train()
        bar = tqdm(train_loader, desc=f"Streak ep{epoch}", disable=dist.get_rank() != 0)
        for i, data in enumerate(bar):
            if epoch == start_epoch and i < skip_steps:
                continue
            global_step += 1
            input_ids = data["input_ids"].cuda()
            attention_mask = data["attention_mask"].cuda()
            loss_mask = data["loss_mask"].cuda()

            # 目标侧：提供各层 hidden；若后端返回 teacher_logits，则用于 LS-RSL 的目标锚点。
            to = target_model.generate_flashmtp_data(input_ids, attention_mask, loss_mask)
            hidden_states = tuple(h.cuda() for h in to.hidden_states)
            tlog = to.teacher_logits.cuda() if to.teacher_logits is not None else None

            # 总损失 = streak_weight * LS-RSL + streak_ce_weight * CE_aux。
            loss, acc, loss_streak, loss_ce = wrapper(
                input_ids=input_ids,
                attention_mask=attention_mask,
                hidden_states=hidden_states,
                loss_mask=loss_mask,
                teacher_logits=tlog,
            )
            (loss / args.accumulation_steps).backward()
            if global_step % args.accumulation_steps == 0:
                optimizer.step()

            if global_step % args.log_interval == 0:
                lf = loss.detach().clone()
                af = acc.detach().clone()
                sf = loss_streak.detach().clone()
                cf = loss_ce.detach().clone()
                dist.all_reduce(lf)
                dist.all_reduce(af)
                dist.all_reduce(sf)
                dist.all_reduce(cf)
                ws = float(dist.get_world_size())
                lf, af, sf, cf = lf / ws, af / ws, sf / ws, cf / ws
                tracker.log(
                    {
                        "train/loss": lf.item(),
                        "train/streak_loss": sf.item(),
                        "train/ce_loss": cf.item(),
                        "train/acc": af.item(),
                        "train/lr": optimizer.get_learning_rate(),
                    },
                    step=global_step,
                )
                print_on_rank0(
                    f"step {global_step} loss={lf.item():.4f} streak={sf.item():.4f} "
                    f"ce={cf.item():.4f} acc={af.item():.4f}"
                )

            if dist.get_rank() == 0:
                dt = time.time() - last_t
                last_t = time.time()
                bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc.item():.4f}", t=f"{dt:.2f}s")

            if global_step % args.save_interval == 0:
                save_checkpoint(args, epoch, global_step, wrapper, draft_model, optimizer)

    save_checkpoint(args, args.num_epochs, global_step, wrapper, draft_model, optimizer)
    tracker.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
