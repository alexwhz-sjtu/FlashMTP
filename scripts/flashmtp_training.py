"""Shared construction/checkpoint helpers for current FlashMTP training entrypoints."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import math
import os
import shutil
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
from transformers import AutoConfig, AutoTokenizer

from datasets import load_dataset
from specforge.args import SGLangBackendArgs, TrackerArgs
from specforge.checkpoint import (
    load_distributed_training_state,
    save_distributed_training_state,
)
from specforge.data import build_eagle3_dataset, prepare_dp_dataloaders
from specforge.distributed import get_dp_group, get_tp_group
from specforge.modeling.draft.flashmtp import (
    FLASHMTP_ARCHITECTURE_VERSION,
    FlashMTPDraftModel,
)
from specforge.modeling.target.flashmtp_target_model import get_flashmtp_target_model
from specforge.modeling.target.target_utils import (
    SGLangTPEmbeddingAdapter,
    SGLangTPLMHeadAdapter,
    SharedTargetEmbeddingsAndHead,
    TargetEmbeddingsAndHead,
)
from specforge.utils import print_on_rank0


def add_common_args(parser: argparse.ArgumentParser) -> None:
    model = parser.add_argument_group("model")
    model.add_argument("--target-model-path", required=True)
    model.add_argument("--target-model-backend", default="hf", choices=["hf", "sglang"])
    model.add_argument("--block-size", type=int, default=8)
    model.add_argument("--num-draft-layers", type=int, default=5)
    model.add_argument("--swa-window-size", type=int, default=32)
    model.add_argument("--anchor-group-size", type=int, default=8)
    model.add_argument("--chs-num-layers", type=int, default=7)
    model.add_argument(
        "--mask-token-id",
        type=int,
        default=151669,
        help="In-vocabulary v2 MASK row (default: Qwen3 token 151669).",
    )
    model.add_argument("--num-anchors", type=int, default=512)
    model.add_argument("--markov-head-type", default="vanilla", choices=["none", "vanilla", "gated", "rnn", "rnn_easy"])
    model.add_argument("--markov-output-mode", default="additive", choices=["additive", "direct"])
    model.add_argument("--markov-rank", type=int, default=256)
    model.add_argument("--trust-remote-code", action="store_true")

    data = parser.add_argument_group("dataset")
    data.add_argument(
        "--train-data-path",
        help="Single-dataset path (teacher training or two-stage compatibility fallback).",
    )
    data.add_argument("--chat-template", default="qwen")
    data.add_argument("--is-preformatted", action="store_true")
    data.add_argument("--max-length", type=int, default=4096)
    data.add_argument("--batch-size", type=int, default=1)
    data.add_argument("--dataloader-num-workers", type=int, default=8)
    data.add_argument("--build-dataset-num-proc", type=int, default=8)
    data.add_argument("--cache-dir", default="./cache/train")

    train = parser.add_argument_group("training")
    train.add_argument("--accumulation-steps", type=int, default=1)
    train.add_argument("--max-grad-norm", type=float, default=1.0)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--tp-size", type=int, default=1)
    train.add_argument(
        "--shard-draft-by-tp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run target prefill on the full TP-group batch, then train each "
            "TP rank on its corresponding batch slice."
        ),
    )
    train.add_argument("--dist-timeout", type=int, default=1200)
    train.add_argument("--resume-from")
    sglang = parser.add_argument_group("sglang target backend")
    SGLangBackendArgs.add_args(sglang)

    output = parser.add_argument_group("output")
    output.add_argument("--output-dir", required=True)
    output.add_argument("--log-interval", type=int, default=50)
    output.add_argument("--save-interval", type=int, default=20000)
    TrackerArgs.add_args(parser.add_argument_group("tracker"))


def validate_common_args(parser: argparse.ArgumentParser, args) -> None:
    """Reject invalid launch values before distributed/model initialization."""
    positive_integer_args = (
        "block_size",
        "num_draft_layers",
        "swa_window_size",
        "anchor_group_size",
        "chs_num_layers",
        "num_anchors",
        "max_length",
        "batch_size",
        "build_dataset_num_proc",
        "accumulation_steps",
        "tp_size",
        "dist_timeout",
        "log_interval",
        "save_interval",
    )
    for name in positive_integer_args:
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if int(args.block_size) < 2:
        parser.error("--block-size must be at least 2")
    if int(args.max_length) < 2 * int(args.block_size):
        parser.error("--max-length must be at least twice --block-size")
    for name in ("dataloader_num_workers",):
        if int(getattr(args, name)) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if float(args.max_grad_norm) <= 0:
        parser.error("--max-grad-norm must be positive")
    if args.mask_token_id is not None and int(args.mask_token_id) < 0:
        parser.error("--mask-token-id must be non-negative")


def build_draft_config(args, *, model_role: str, source_config=None):
    config = (
        AutoConfig.from_pretrained(args.target_model_path)
        if source_config is None
        else source_config
    )
    if source_config is None:
        target_depth = int(config.num_hidden_layers)
        config.num_hidden_layers = int(args.num_draft_layers)
        config.num_target_layers = target_depth
        config.block_size = int(args.block_size)
    flash = dict(getattr(config, "flashmtp_config", None) or {})
    flash.clear()
    flash.update(
        architecture_version=FLASHMTP_ARCHITECTURE_VERSION,
        model_role=model_role,
        swa_window_size=int(args.swa_window_size),
        anchor_group_size=int(args.anchor_group_size),
        chs_num_layers=int(args.chs_num_layers),
        markov_head_type=args.markov_head_type,
        markov_output_mode=args.markov_output_mode,
        markov_rank=int(args.markov_rank),
    )
    config.flashmtp_config = flash
    config._attn_implementation = "flex_attention"
    layer_types = list(getattr(config, "layer_types", []) or [])
    config.layer_types = (
        layer_types[: config.num_hidden_layers]
        if len(layer_types) >= config.num_hidden_layers
        else ["full_attention"] * config.num_hidden_layers
    )
    return config


def build_draft_model(args, *, model_role: str, source_config=None):
    config = build_draft_config(args, model_role=model_role, source_config=source_config)
    return FlashMTPDraftModel(config).cuda().to(torch.bfloat16)


def build_target_model(args, draft_models: list[FlashMTPDraftModel]):
    backend_kwargs = {}
    if args.target_model_backend == "sglang":
        backend_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()
        if backend_kwargs["max_running_requests"] is None:
            backend_kwargs["max_running_requests"] = int(args.batch_size)
        if backend_kwargs["max_total_tokens"] is None:
            backend_kwargs["max_total_tokens"] = int(args.batch_size) * int(
                args.max_length
            )
    target = get_flashmtp_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend=args.target_model_backend,
        torch_dtype=torch.bfloat16,
        device="cuda" if args.target_model_backend == "hf" else None,
        trust_remote_code=args.trust_remote_code,
        **backend_kwargs,
    )
    capture = set()
    for draft in draft_models:
        capture.update(draft.target_layer_ids)
        if draft.is_teacher:
            capture.update(draft.history_layer_ids)
    target.set_capture_layers(sorted(capture))
    return target


def build_target_and_components(args, draft_models: list[FlashMTPDraftModel]):
    """Build one target and bind its tokenizer/embedding/head consistently."""
    target = build_target_model(args, draft_models)
    tokenizer, components, mask_token_id = resolve_tokenizer_and_components(
        args, draft_models, target=target
    )
    return target, tokenizer, components, mask_token_id


def resolve_tokenizer_and_components(args, draft_models, target=None):
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    if args.mask_token_id is not None:
        mask_token_id = int(args.mask_token_id)
    elif tokenizer.mask_token_id is not None:
        mask_token_id = int(tokenizer.mask_token_id)
    else:
        tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
        mask_token_id = int(tokenizer.mask_token_id)
    if args.target_model_backend == "sglang":
        if target is None or not hasattr(target, "model_runner"):
            raise ValueError("SGLang target is required to reuse target components.")
        target_model = target.model_runner.model
        target_embedding = target_model.get_input_embeddings()
        target_lm_head = target_model.lm_head
        components = SharedTargetEmbeddingsAndHead(
            SGLangTPEmbeddingAdapter(
                target_embedding, get_tp_group(), mask_token_id
            ),
            SGLangTPLMHeadAdapter(target_lm_head, get_tp_group()),
        )
        components.requires_grad_(False)
        print_on_rank0(
            "Reusing SGLang target-resident TP embedding and LM head; "
            "no independent full-vocabulary copies were loaded."
        )
    else:
        components = TargetEmbeddingsAndHead.from_pretrained(
            args.target_model_path,
            embed_key="model.embed_tokens.weight",
            lm_head_key="lm_head.weight",
            device="cuda",
            trust_remote_code=args.trust_remote_code,
        )
        if not 0 <= mask_token_id < components.embed_tokens.num_embeddings:
            raise ValueError(
                "FlashMTP vocab_row MASK mode requires an existing target "
                f"embedding row, but mask_token_id={mask_token_id} and target "
                f"vocab size={components.embed_tokens.num_embeddings}. Pass "
                "--mask-token-id with an unused in-vocabulary row."
            )
    for draft in draft_models:
        draft.mask_token_id = mask_token_id
        draft.mask_embedding_mode = "vocab_row"
        draft.config.flashmtp_config["mask_token_id"] = mask_token_id
        draft.config.flashmtp_config["mask_embedding_mode"] = "vocab_row"
    return tokenizer, components, mask_token_id


def _build_processed_dataset(
    args,
    tokenizer,
    *,
    train_data_path: str,
    cache_namespace: str,
    num_proc: Optional[int] = None,
):
    cache_key = hashlib.md5(
        (
            f"{train_data_path}-{args.max_length}-{args.chat_template}-"
            f"{args.target_model_path}-preformatted={args.is_preformatted}"
        ).encode()
    ).hexdigest()
    raw = load_dataset("json", data_files=train_data_path)["train"]
    return build_eagle3_dataset(
        dataset=raw,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        is_preformatted=args.is_preformatted,
        cache_dir=os.path.join(
            args.cache_dir, cache_namespace, "processed_dataset"
        ),
        cache_key=cache_key,
        num_proc=(
            args.build_dataset_num_proc if num_proc is None else int(num_proc)
        ),
    )


def _has_valid_anchor_supervision(value, *, block_size: int) -> bool:
    """Match the anchor sampler's per-example supervision requirements."""
    loss_mask = torch.as_tensor(value["loss_mask"]).reshape(-1)
    minimum = 2 * int(block_size)
    if int(loss_mask.sum().item()) < minimum:
        return False
    max_anchor = int(loss_mask.numel()) - int(block_size)
    if max_anchor < 1:
        return False
    current = loss_mask[1 : max_anchor + 1] > 0.5
    following = loss_mask[2 : max_anchor + 2] > 0.5
    return bool((current & following).any().item())


def _prepare_dataloader(args, dataset, *, train_data_path: str):
    minimum = 2 * int(args.block_size)
    dataset = dataset.filter(
        _has_valid_anchor_supervision,
        fn_kwargs={"block_size": int(args.block_size)},
        desc=(
            "Filtering examples with at least "
            f"{minimum} labels and one trainable anchor"
        ),
    )
    dataloader = prepare_dp_dataloaders(
        dataset,
        args.batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        process_group=get_dp_group(),
        pad_to_length=args.max_length,
    )
    if len(dataloader) == 0:
        raise ValueError(
            f"Training dataset {train_data_path!r} has no full batches after filtering."
        )
    return dataloader


def build_train_dataloader(
    args,
    tokenizer,
    *,
    train_data_path: Optional[str] = None,
    cache_namespace: str = "single",
    num_proc: Optional[int] = None,
):
    train_data_path = train_data_path or args.train_data_path
    if not train_data_path:
        raise ValueError("A training data path is required.")
    dataset = None
    if dist.get_rank() == 0:
        dataset = _build_processed_dataset(
            args,
            tokenizer,
            train_data_path=train_data_path,
            cache_namespace=cache_namespace,
            num_proc=num_proc,
        )
    dist.barrier()
    if dataset is None:
        dataset = _build_processed_dataset(
            args,
            tokenizer,
            train_data_path=train_data_path,
            cache_namespace=cache_namespace,
            num_proc=num_proc,
        )
    return _prepare_dataloader(
        args, dataset, train_data_path=train_data_path
    )


def build_two_stage_dataloaders(args, tokenizer):
    """Preprocess Stage 1/2 concurrently, then create independent dataloaders."""
    stage1_path = args.stage1_train_data_path
    stage2_path = args.stage2_train_data_path
    same_dataset = os.path.realpath(stage1_path) == os.path.realpath(stage2_path)
    stage1_namespace = "shared" if same_dataset else "stage1"
    stage2_namespace = "shared" if same_dataset else "stage2"
    stage1_num_proc = (
        args.stage1_build_dataset_num_proc or args.build_dataset_num_proc
    )
    stage2_num_proc = (
        args.stage2_build_dataset_num_proc or args.build_dataset_num_proc
    )

    stage1_dataset = stage2_dataset = None
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if same_dataset:
        if rank == 0:
            stage1_dataset = _build_processed_dataset(
                args,
                tokenizer,
                train_data_path=stage1_path,
                cache_namespace=stage1_namespace,
                num_proc=stage1_num_proc,
            )
            stage2_dataset = stage1_dataset
    elif world_size >= 2:
        # Shared storage lets two global ranks build disjoint caches at once.
        if rank == 0:
            stage1_dataset = _build_processed_dataset(
                args,
                tokenizer,
                train_data_path=stage1_path,
                cache_namespace=stage1_namespace,
                num_proc=stage1_num_proc,
            )
        elif rank == 1:
            stage2_dataset = _build_processed_dataset(
                args,
                tokenizer,
                train_data_path=stage2_path,
                cache_namespace=stage2_namespace,
                num_proc=stage2_num_proc,
            )
    else:
        # Keep the requested startup concurrency for single-process debugging.
        with ThreadPoolExecutor(max_workers=2) as executor:
            stage1_future = executor.submit(
                _build_processed_dataset,
                args,
                tokenizer,
                train_data_path=stage1_path,
                cache_namespace=stage1_namespace,
                num_proc=stage1_num_proc,
            )
            stage2_future = executor.submit(
                _build_processed_dataset,
                args,
                tokenizer,
                train_data_path=stage2_path,
                cache_namespace=stage2_namespace,
                num_proc=stage2_num_proc,
            )
            stage1_dataset = stage1_future.result()
            stage2_dataset = stage2_future.result()

    dist.barrier()
    if stage1_dataset is None:
        stage1_dataset = _build_processed_dataset(
            args,
            tokenizer,
            train_data_path=stage1_path,
            cache_namespace=stage1_namespace,
            num_proc=stage1_num_proc,
        )
    if stage2_dataset is None:
        stage2_dataset = (
            stage1_dataset
            if same_dataset
            else _build_processed_dataset(
                args,
                tokenizer,
                train_data_path=stage2_path,
                cache_namespace=stage2_namespace,
                num_proc=stage2_num_proc,
            )
        )
    print_on_rank0(
        "Stage 1 and Stage 2 dataset preprocessing is complete: "
        f"stage1={stage1_path!r}, stage2={stage2_path!r}."
    )
    return (
        _prepare_dataloader(args, stage1_dataset, train_data_path=stage1_path),
        _prepare_dataloader(args, stage2_dataset, train_data_path=stage2_path),
    )


def hidden_states_to_cuda(hidden_states):
    if isinstance(hidden_states, dict):
        return {key: value.cuda() for key, value in hidden_states.items()}
    return tuple(value.cuda() for value in hidden_states)


def validate_tp_draft_sharding(args) -> Optional[int]:
    """Validate one-rank/one-sample draft sharding and return the TP rank."""
    if not args.shard_draft_by_tp:
        return None
    if args.target_model_backend != "sglang":
        raise ValueError(
            "--shard-draft-by-tp requires --target-model-backend sglang so "
            "the target is actually tensor parallel."
        )
    tp_group = get_tp_group()
    tp_size = dist.get_world_size(tp_group)
    if tp_size <= 1:
        raise ValueError("--shard-draft-by-tp requires --tp-size > 1.")
    if int(args.batch_size) != tp_size:
        raise ValueError(
            "One-sample-per-TP-rank draft sharding requires target batch size "
            f"to equal tp_size; got batch_size={args.batch_size}, tp_size={tp_size}."
        )
    print_on_rank0(
        f"shard-draft-by-tp enabled: target batch={args.batch_size}; "
        "each TP rank trains one distinct draft sample."
    )
    return dist.get_rank(tp_group)


def select_tp_rank_batch(value, tp_rank: int):
    """Copy rank ``tp_rank``'s sample so full target tensors can be released."""
    if isinstance(value, dict):
        return {
            key: select_tp_rank_batch(item, tp_rank)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(select_tp_rank_batch(item, tp_rank) for item in value)
    if isinstance(value, list):
        return [select_tp_rank_batch(item, tp_rank) for item in value]
    if not isinstance(value, torch.Tensor) or value.ndim == 0:
        raise TypeError("TP draft batch values must be batch-first tensors or containers.")
    if not 0 <= int(tp_rank) < value.size(0):
        raise ValueError(
            f"TP rank {tp_rank} cannot select from batch size {value.size(0)}."
        )
    return value.narrow(0, int(tp_rank), 1).clone(
        memory_format=torch.contiguous_format
    )


def save_checkpoint(
    *,
    output_dir: str,
    name: str,
    fsdp_model: FSDP,
    draft_model: FlashMTPDraftModel,
    optimizer,
    metadata: dict,
) -> str:
    save_dir = os.path.join(output_dir, name)
    if dist.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)
    dist.barrier()
    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT):
        full_state = fsdp_model.state_dict()
        draft_state = {
            key.split("draft_model.", 1)[1]: value
            for key, value in full_state.items()
            if "draft_model." in key
        }
        model_metadata = {
            "architecture_version": draft_model.architecture_version,
            "model_role": draft_model.model_role,
            "swa_window_size": draft_model.swa_window_size,
            "anchor_group_size": draft_model.anchor_group_size,
            "chs_num_layers": draft_model.chs_num_layers,
            "block_size": draft_model.block_size,
            "num_draft_layers": draft_model.config.num_hidden_layers,
            "markov_head_type": draft_model.markov_head_type,
            "markov_output_mode": draft_model.markov_output_mode,
            "markov_rank": draft_model.markov_rank,
        }
        save_distributed_training_state(
            save_dir, {**model_metadata, **metadata, **optimizer.state_dict()}
        )
        if dist.get_rank() == 0:
            draft_model.save_pretrained(save_dir, state_dict=draft_state)
            for filename in ("flashmtp.py", "flashmtp_markov_head.py"):
                source = os.path.join(
                    os.path.dirname(__file__), "..", "specforge", "modeling", "draft", filename
                )
                if os.path.exists(source):
                    shutil.copy(source, os.path.join(save_dir, filename))
    dist.barrier()
    print_on_rank0(f"Saved checkpoint to {save_dir}")
    return save_dir


def load_training_state(checkpoint_dir: Optional[str]) -> Optional[dict]:
    if checkpoint_dir is None:
        return None
    state = load_distributed_training_state(checkpoint_dir, map_location="cpu")
    if state is None:
        raise FileNotFoundError(
            f"No training_state.pt was found in checkpoint {checkpoint_dir!r}."
        )
    return state


def resume_cursor(state: Optional[dict], stage: str) -> tuple[int, int, int, int]:
    """Return epoch, next batch, stage step, and monotonic global step."""
    if state is None:
        return 0, 0, 0, 0
    if state.get("training_stage") != stage:
        raise ValueError(
            f"Expected a {stage!r} checkpoint, got "
            f"{state.get('training_stage')!r}."
        )
    return (
        int(state.get("stage_epoch", 0)),
        int(state.get("next_batch_in_epoch", 0)),
        int(state.get("stage_step", 0)),
        int(state.get("global_step", 0)),
    )


def stage_total_steps(dataloader, epochs: int, accumulation_steps: int) -> int:
    # The loops intentionally carry a partial accumulation across epoch
    # boundaries and flush only once at the end of the stage.  Therefore the
    # scheduler must count all micro-batches together rather than rounding each
    # epoch independently.
    return math.ceil(
        int(epochs) * len(dataloader) / int(accumulation_steps)
    )


def log_cuda_peak(stage: str) -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated_gib": 0.0, "reserved_gib": 0.0}
    allocated = torch.cuda.max_memory_allocated() / 1024**3
    reserved = torch.cuda.max_memory_reserved() / 1024**3
    print_on_rank0(
        f"{stage} CUDA peak: allocated={allocated:.2f} GiB, reserved={reserved:.2f} GiB"
    )
    return {"allocated_gib": allocated, "reserved_gib": reserved}


__all__ = [
    "add_common_args",
    "build_draft_model",
    "build_target_and_components",
    "build_target_model",
    "build_train_dataloader",
    "build_two_stage_dataloaders",
    "hidden_states_to_cuda",
    "load_training_state",
    "log_cuda_peak",
    "resolve_tokenizer_and_components",
    "resume_cursor",
    "save_checkpoint",
    "select_tp_rank_batch",
    "stage_total_steps",
    "validate_common_args",
    "validate_tp_draft_sharding",
]
