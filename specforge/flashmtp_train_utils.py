# coding=utf-8
"""Shared helpers for FlashMTP training scripts (v3.3 MDLM / Streak)."""

import argparse
import hashlib
import os
import shutil
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

from datasets import load_dataset
from specforge.args import SGLangBackendArgs, TrackerArgs
from specforge.data import build_eagle3_dataset, prepare_dp_dataloaders
from specforge.distributed import get_dp_group
from specforge.modeling.draft.flashmtp import FlashMTPDraftModel
from specforge.modeling.target.flashmtp_target_model import (
    FlashMTPTargetModel,
    get_flashmtp_target_model,
)
from specforge.optimizer import BF16Optimizer
from specforge.tracker import create_tracker
from specforge.utils import get_last_checkpoint, print_on_rank0


def _align_draft_config_layer_depth(draft_config: Any, num_layers: int) -> None:
    """Qwen3 等配置里 layer_types 长度必须与 num_hidden_layers 一致。"""
    draft_config.num_hidden_layers = num_layers
    lt = getattr(draft_config, "layer_types", None)
    if lt is not None and len(lt) != num_layers:
        if len(lt) > num_layers:
            draft_config.layer_types = list(lt[:num_layers])
        else:
            pad = lt[-1] if lt else "full_attention"
            draft_config.layer_types = list(lt) + [pad] * (num_layers - len(lt))


def add_flashmtp_common_args(parser: argparse.ArgumentParser) -> None:
    model_group = parser.add_argument_group("model")
    model_group.add_argument("--target-model-path", type=str, required=True)
    model_group.add_argument(
        "--target-model-backend",
        type=str,
        default="hf",
        choices=["sglang", "hf"],
    )
    model_group.add_argument("--draft-config-path", type=str, default=None)
    model_group.add_argument("--block-size", type=int, default=16)
    model_group.add_argument("--num-draft-layers", type=int, default=1)
    model_group.add_argument("--mask-token-id", type=int, default=None)
    model_group.add_argument(
        "--attention-backend",
        type=str,
        default="flex_attention",
        choices=["eager", "sdpa", "flex_attention"],
    )
    model_group.add_argument("--trust-remote-code", action="store_true")
    model_group.add_argument("--num-anchors", type=int, default=512)
    model_group.add_argument("--chs-fusion-layer-idx", type=int, default=0)
    model_group.add_argument(
        "--chs-concat-mode",
        type=str,
        default="feature",
        choices=["feature", "seq"],
    )

    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument("--train-data-path", type=str, required=True)
    dataset_group.add_argument("--eval-data-path", type=str, default=None)
    dataset_group.add_argument("--chat-template", type=str, default="qwen")
    dataset_group.add_argument("--is-preformatted", action="store_true")
    dataset_group.add_argument("--dataloader-num-workers", type=int, default=0)
    dataset_group.add_argument(
        "--build-dataset-num-proc",
        type=int,
        default=int(os.environ.get("SPECFORGE_DATA_NUM_PROC", 4)),
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
    training_group.add_argument("--ckpt-dir", type=str, default=None)

    output_group = parser.add_argument_group("output")
    output_group.add_argument("--output-dir", type=str, required=True)
    output_group.add_argument("--cache-dir", type=str, default="./cache/train")
    output_group.add_argument("--log-interval", type=int, default=50)
    output_group.add_argument("--eval-interval", type=int, default=1000)
    output_group.add_argument("--save-interval", type=int, default=1000)

    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument("--tp-size", type=int, default=1)

    tracker_group = parser.add_argument_group("tracker")
    TrackerArgs.add_args(tracker_group)

    dist_group = parser.add_argument_group("distributed")
    dist_group.add_argument("--dist-timeout", type=int, default=30)

    sglang_group = parser.add_argument_group("sglang backend")
    SGLangBackendArgs.add_args(sglang_group)


def build_models(args: Any) -> Tuple[FlashMTPTargetModel, FlashMTPDraftModel]:
    print_on_rank0(
        f"Loading target from {args.target_model_path} ({args.target_model_backend})"
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
        _align_draft_config_layer_depth(draft_config, int(draft_config.num_hidden_layers))
    else:
        target_config = AutoConfig.from_pretrained(args.target_model_path)
        draft_config = AutoConfig.from_pretrained(args.target_model_path)
        draft_config.block_size = args.block_size
        draft_config.num_target_layers = target_config.num_hidden_layers
        _align_draft_config_layer_depth(draft_config, args.num_draft_layers)

    if not hasattr(draft_config, "flashmtp_config") or draft_config.flashmtp_config is None:
        draft_config.flashmtp_config = {}

    if args.chs_concat_mode == "seq":
        raise NotImplementedError("Use --chs-concat-mode feature for FlashMTP v3.3.")

    target_cfg = AutoConfig.from_pretrained(args.target_model_path)
    n_chs_default = target_cfg.num_hidden_layers + 1
    n_chs = int(draft_config.flashmtp_config.get("num_chs_source_tokens", n_chs_default))
    draft_config.flashmtp_config["num_chs_source_tokens"] = n_chs
    draft_config.flashmtp_config["chs_fusion_layer_idx"] = args.chs_fusion_layer_idx
    draft_config.flashmtp_config["chs_concat_mode"] = args.chs_concat_mode
    draft_config._attn_implementation = args.attention_backend

    draft_model = FlashMTPDraftModel(draft_config).cuda().to(torch.bfloat16)
    target_model.set_capture_layers(list(range(target_cfg.num_hidden_layers + 1)))
    print_on_rank0(
        f"Draft: layers={draft_config.num_hidden_layers}, block={draft_config.block_size}, "
        f"chs_tokens={n_chs}"
    )
    return target_model, draft_model


def build_dataloader(
    args: Any, tokenizer
) -> Tuple[DataLoader, Optional[DataLoader]]:
    cache_params_string = (
        f"{args.train_data_path}-{args.max_length}-{args.chat_template}-"
        f"{args.target_model_path}"
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()

    train_dataset = load_dataset("json", data_files=args.train_data_path)["train"]
    train_ds = build_eagle3_dataset(
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
    n0 = len(train_ds)
    train_ds = train_ds.filter(lambda x: x["loss_mask"].sum() >= min_loss_tokens)
    print_on_rank0(f"Train filter: {n0} -> {len(train_ds)}")

    train_loader = prepare_dp_dataloaders(
        train_ds,
        args.batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        process_group=get_dp_group(),
    )

    eval_loader = None
    if args.eval_data_path:
        eval_dataset = load_dataset("json", data_files=args.eval_data_path)["train"]
        eval_ds = build_eagle3_dataset(
            dataset=eval_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            is_preformatted=args.is_preformatted,
        )
        eval_loader = prepare_dp_dataloaders(
            eval_ds,
            args.batch_size,
            num_workers=args.dataloader_num_workers,
            shuffle=False,
            process_group=get_dp_group(),
        )

    return train_loader, eval_loader


def resolve_mask_token_id(tokenizer, args: Any) -> int:
    if args.mask_token_id is not None:
        return args.mask_token_id
    if tokenizer.mask_token_id is not None:
        return tokenizer.mask_token_id
    tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
    return tokenizer.mask_token_id


def save_checkpoint(
    args: Any,
    epoch: int,
    step: int,
    fsdp_module: FSDP,
    draft_model: FlashMTPDraftModel,
    optimizer: BF16Optimizer,
) -> None:
    save_dir = os.path.join(args.output_dir, f"epoch_{epoch}_step_{step}")
    if dist.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)
    dist.barrier()

    with FSDP.state_dict_type(fsdp_module, StateDictType.FULL_STATE_DICT):
        state_dict = fsdp_module.state_dict()
        draft_state = {
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
            draft_model.save_pretrained(save_dir, state_dict=draft_state)
            src = os.path.join(
                os.path.dirname(__file__), "modeling", "draft", "flashmtp.py"
            )
            dst = os.path.join(save_dir, "flashmtp.py")
            if os.path.exists(src):
                shutil.copy(src, dst)
            print_on_rank0(f"Saved checkpoint to {save_dir}")
    dist.barrier()


def load_draft_checkpoint(
    draft_model: FlashMTPDraftModel, ckpt_dir: str
) -> Optional[dict]:
    """只加载权重到当前 draft_model，避免 checkpoint 内 config 与运行时层数不一致。"""
    st_path = os.path.join(ckpt_dir, "model.safetensors")
    bin_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    if os.path.isfile(st_path):
        try:
            from safetensors.torch import load_file

            state = load_file(st_path)
        except ImportError:
            state = torch.load(st_path, map_location="cpu", weights_only=True)
    elif os.path.isfile(bin_path):
        state = torch.load(bin_path, map_location="cpu", weights_only=False)
    else:
        raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin in {ckpt_dir}")
    draft_model.load_state_dict(state, strict=True)
    print_on_rank0(f"Loaded draft weights from {ckpt_dir}")
    path = os.path.join(ckpt_dir, "training_state.pt")
    if os.path.isfile(path):
        return torch.load(path, map_location="cpu", weights_only=False)
    return None


def maybe_resume_ckpt(
    args: Any, draft_model: FlashMTPDraftModel
) -> Tuple[Optional[str], Optional[dict]]:
    if args.ckpt_dir and os.path.isdir(args.ckpt_dir):
        return args.ckpt_dir, load_draft_checkpoint(draft_model, args.ckpt_dir)
    if args.resume and os.path.isdir(args.output_dir):
        last_dir, _ = get_last_checkpoint(args.output_dir)
        if last_dir:
            return last_dir, load_draft_checkpoint(draft_model, last_dir)
    return None, None


def init_tokenizer_and_mask(args: Any):
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    mid = resolve_mask_token_id(tokenizer, args)
    print_on_rank0(f"mask_token_id={mid}")
    return tokenizer, mid
