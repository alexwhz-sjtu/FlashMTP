#!/usr/bin/env python3
"""Step-by-step GPU memory probe for FlashMTP training (rank 0 logs only)."""

import argparse
import gc
import math
import os
import sys

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from transformers import AutoConfig, AutoTokenizer

# Reuse training helpers
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset

from scripts.train_flashmtp import (  # noqa: E402
    _ensure_embed_vocab_for_mask,
    build_dataloader,
    build_models,
    parse_args as train_parse_args,
)
from specforge.args import SGLangBackendArgs  # noqa: E402
from specforge.core.flashmtp import OnlineFlashMTPModel  # noqa: E402
from specforge.distributed import get_dp_group, init_distributed  # noqa: E402
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead  # noqa: E402
from specforge.utils import print_on_rank0  # noqa: E402


def _gb(x: int) -> float:
    return x / (1024**3)


def mem_line(tag: str) -> str:
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    peak = torch.cuda.max_memory_allocated()
    return (
        f"[MEM] {tag}: allocated={_gb(alloc):.2f} GiB, "
        f"reserved={_gb(reserved):.2f} GiB, peak={_gb(peak):.2f} GiB"
    )


def log_mem(tag: str) -> None:
    if dist.get_rank() == 0:
        print(mem_line(tag), flush=True)


def tensor_nbytes(t) -> int:
    if isinstance(t, dict):
        return sum(tensor_nbytes(v) for v in t.values())
    if isinstance(t, (tuple, list)):
        return sum(tensor_nbytes(v) for v in t)
    return t.numel() * t.element_size()


def main():
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--steps", type=int, default=3)
    probe.add_argument("--mem-fraction", type=float, default=None)
    probe_args, remaining = probe.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    args = train_parse_args()
    if probe_args.mem_fraction is not None:
        args.sglang_mem_fraction_static = probe_args.mem_fraction

    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    log_mem("00_after_distributed_init")

    target_model, draft_model = build_models(args)
    log_mem("01_after_target_and_draft_built")

    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    mask_token_id = args.mask_token_id
    if mask_token_id is None:
        mask_token_id = tokenizer.convert_tokens_to_ids("<|MASK|>")
    if mask_token_id is None or mask_token_id < 0:
        mask_token_id = 151669

    train_dataloader, _ = build_dataloader(args, tokenizer)
    log_mem("02_after_dataloader_built")

    target_components = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model_path,
        embed_key="model.embed_tokens.weight",
        lm_head_key="lm_head.weight",
        device="cuda",
        trust_remote_code=args.trust_remote_code,
    )
    _ensure_embed_vocab_for_mask(target_components, mask_token_id)
    log_mem("03_after_embed_and_lm_head")

    online_flashmtp = OnlineFlashMTPModel(
        draft_model=draft_model,
        target_lm_head=target_components.lm_head,
        target_embed_tokens=target_components.embed_tokens,
        block_size=draft_model.block_size,
        mask_token_id=mask_token_id,
        attention_backend=args.attention_backend,
        num_anchors=args.num_anchors,
        loss_decay_gamma=args.loss_decay_gamma,
        final_ce_weight=args.final_ce_weight,
        tv_loss_weight=args.tv_loss_weight,
        chs_concat_mode="feature",
        add_noise=args.add_noise,
        target_hidden_noise_ratio=args.target_hidden_noise_ratio,
        ce_chunk_size=args.ce_chunk_size,
    )
    flashmtp_model = FSDP(
        online_flashmtp,
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    )
    log_mem("04_after_fsdp_wrap")

    if dist.get_rank() == 0:
        print(
            f"[CFG] mem_fraction_static={args.sglang_mem_fraction_static}, "
            f"num_anchors={args.num_anchors}, ce_chunk_size={args.ce_chunk_size}, "
            f"capture_layers={getattr(target_model, 'capture_layer_ids', None)}",
            flush=True,
        )

    train_dataloader.sampler.set_epoch(0)
    data_iter = iter(train_dataloader)

    for step in range(1, probe_args.steps + 1):
        torch.cuda.reset_peak_memory_stats()
        data = next(data_iter)
        input_ids = data["input_ids"].cuda()
        attention_mask = data["attention_mask"].cuda()
        loss_mask = data["loss_mask"].cuda()
        seq_len = int(input_ids.shape[1])
        log_mem(f"05_step{step}_after_batch(seq_len={seq_len})")

        target_output = target_model.generate_flashmtp_data(
            input_ids, attention_mask, loss_mask
        )
        hidden_states = target_output.hidden_states
        if isinstance(hidden_states, dict):
            hidden_states = {
                k: (v.cuda() if not v.is_cuda else v) for k, v in hidden_states.items()
            }
        else:
            hidden_states = tuple(h.cuda() if not h.is_cuda else h for h in hidden_states)

        hs_bytes = tensor_nbytes(hidden_states)
        if dist.get_rank() == 0:
            n_layers = (
                len(hidden_states)
                if isinstance(hidden_states, dict)
                else len(hidden_states)
            )
            print(
                f"[HS] step{step}: hidden_states ~{_gb(hs_bytes):.2f} GiB, "
                f"layers={n_layers}, seq_len={seq_len}",
                flush=True,
            )
        log_mem(f"06_step{step}_after_target_forward")

        (
            anchor_positions,
            block_keep_mask,
            target_hidden,
            target_prediction_hidden,
        ) = online_flashmtp.prepare_training_tensors(
            input_ids, hidden_states, loss_mask
        )
        th_bytes = tensor_nbytes(target_hidden)
        tph_bytes = (
            tensor_nbytes(target_prediction_hidden)
            if target_prediction_hidden is not None
            else 0
        )
        del target_output, hidden_states
        gc.collect()
        torch.cuda.empty_cache()
        if dist.get_rank() == 0:
            print(
                f"[HS] step{step}: target_hidden ~{_gb(th_bytes):.2f} GiB "
                f"(shape={tuple(target_hidden.shape)}), "
                f"target_prediction_hidden ~{_gb(tph_bytes):.2f} GiB",
                flush=True,
            )
        log_mem(f"07_step{step}_after_del_full_hidden_states")

        loss, accuracy, prefix_acc, base_ce_loss, tv_loss = flashmtp_model(
            input_ids=input_ids,
            loss_mask=loss_mask,
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            target_hidden=target_hidden,
            target_prediction_hidden=target_prediction_hidden,
        )
        log_mem(f"08_step{step}_after_draft_forward(loss={loss.item():.4f})")

        loss.backward()
        log_mem(f"09_step{step}_after_backward")

        flashmtp_model.zero_grad(set_to_none=True)
        del loss, accuracy, prefix_acc, base_ce_loss, tv_loss
        del (
            target_hidden,
            target_prediction_hidden,
            anchor_positions,
            block_keep_mask,
            input_ids,
        )
        del attention_mask, loss_mask
        gc.collect()
        torch.cuda.empty_cache()
        log_mem(f"10_step{step}_after_cleanup")

    if dist.get_rank() == 0:
        print("[DONE] memory profile finished.", flush=True)
    destroy = __import__(
        "specforge.distributed", fromlist=["destroy_distributed"]
    ).destroy_distributed
    destroy()


if __name__ == "__main__":
    main()
