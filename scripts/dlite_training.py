"""Shared construction/checkpoint helpers for DLite training entrypoints."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import shutil
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.utils.data import DataLoader, DistributedSampler, Subset
from transformers import AutoTokenizer

from datasets import load_dataset
from specforge.args import SGLangBackendArgs, TrackerArgs
from specforge.checkpoint import (
    load_distributed_training_state,
    save_distributed_training_state,
)
from specforge.data import (
    RegenFullCollator,
    RegenFullDataset,
    build_training_dataset,
    load_hidden_cache_manifest,
    prepare_dp_dataloaders,
)
from specforge.distributed import get_dp_group, get_tp_group
from specforge.modeling.config_utils import (
    is_qwen35_model_type,
    load_text_model_config,
)
from specforge.modeling.draft.dlite import (
    DLITE_ARCHITECTURE_VERSION,
    DLITE_ARCHITECTURE_VERSIONS,
    DLiteDraftModel,
    build_target_layer_ids,
)
from specforge.modeling.target.dlite_target_model import get_dlite_target_model
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
    model.add_argument(
        "--embedding-key",
        default="model.embed_tokens.weight",
        help="Target checkpoint embedding key used by standalone/offline components.",
    )
    model.add_argument(
        "--lm-head-key",
        default="lm_head.weight",
        help="Target checkpoint LM-head key used by standalone/offline components.",
    )
    model.add_argument(
        "--dlite-version",
        default=DLITE_ARCHITECTURE_VERSION,
        choices=DLITE_ARCHITECTURE_VERSIONS,
    )
    model.add_argument("--block-size", type=int, default=8)
    model.add_argument("--num-draft-layers", type=int, default=5)
    model.add_argument("--swa-window-size", type=int, default=32)
    model.add_argument("--chs-num-layers", type=int, default=7)
    model.add_argument(
        "--target-layer-ids",
        help=(
            "Comma-separated zero-based target layer IDs. When provided, this "
            "takes precedence over --chs-num-layers."
        ),
    )
    model.add_argument(
        "--mask-token-id",
        type=int,
        default=None,
        help=(
            "Optional in-vocabulary MASK embedding row. By default, preserve the "
            "checkpoint value, use tokenizer.mask_token_id, or select the first "
            "model vocabulary row unused by the tokenizer."
        ),
    )
    model.add_argument("--num-anchors", type=int, default=512)
    model.add_argument("--sequential-head", default="rnn", choices=["rnn"])
    model.add_argument("--sequential-rank", type=int, default=256)
    model.add_argument("--trust-remote-code", action="store_true")

    data = parser.add_argument_group("dataset")
    data.add_argument(
        "--train-data-path",
        help="Online token-only JSONL. Mutually exclusive with --train-hidden-states-path.",
    )
    data.add_argument(
        "--train-hidden-states-path",
        help="Offline regen_full cache directory. Mutually exclusive with --train-data-path.",
    )
    data.add_argument("--chat-template", default="qwen")
    data.add_argument("--is-preformatted", action="store_true")
    data.add_argument(
        "--pad-to-max-length",
        action="store_true",
        help="Pad every microbatch to --max-length (benchmark/control mode).",
    )
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
    if bool(args.train_data_path) == bool(args.train_hidden_states_path):
        parser.error(
            "exactly one of --train-data-path and --train-hidden-states-path is required"
        )
    if args.train_hidden_states_path:
        if int(args.tp_size) != 1:
            parser.error("offline regen_full training requires --tp-size 1")
        if args.shard_draft_by_tp:
            parser.error(
                "offline regen_full training does not support --shard-draft-by-tp"
            )


def build_draft_config(args, *, model_role: str, source_config=None):
    config = (
        load_text_model_config(
            args.target_model_path, trust_remote_code=args.trust_remote_code
        )
        if source_config is None
        else source_config
    )
    if source_config is None:
        target_depth = int(config.num_hidden_layers)
        config.num_hidden_layers = int(args.num_draft_layers)
        config.num_target_layers = target_depth
        config.block_size = int(args.block_size)
    if args.target_layer_ids:
        target_layer_ids = [
            int(value.strip())
            for value in str(args.target_layer_ids).split(",")
            if value.strip()
        ]
        if not target_layer_ids:
            raise ValueError("--target-layer-ids must contain at least one layer ID.")
        if target_layer_ids != sorted(set(target_layer_ids)):
            raise ValueError(
                "--target-layer-ids must be unique and in strictly increasing order."
            )
        invalid = [
            layer_id
            for layer_id in target_layer_ids
            if not 0 <= layer_id < int(config.num_target_layers)
        ]
        if invalid:
            raise ValueError(
                f"Target layer IDs {invalid} are outside [0, "
                f"{int(config.num_target_layers) - 1}]."
            )
        chs_num_layers = len(target_layer_ids)
        args.chs_num_layers = chs_num_layers
    else:
        chs_num_layers = int(args.chs_num_layers)
        target_layer_ids = build_target_layer_ids(
            int(config.num_target_layers), chs_num_layers
        )
    dlite = dict(
        architecture_version=args.dlite_version,
        model_role=model_role,
        chs_num_layers=chs_num_layers,
        target_layer_ids=target_layer_ids,
        sequential_head=args.sequential_head,
        sequential_rank=int(args.sequential_rank),
        mask_token_id=(None if args.mask_token_id is None else int(args.mask_token_id)),
    )
    if model_role == "swa_teacher":
        dlite["swa_window_size"] = int(args.swa_window_size)
        dlite["history_layer_ids"] = [
            0,
            int(config.num_target_layers) // 2,
            int(config.num_target_layers) - 1,
        ]
    config.dlite_config = dlite
    config._attn_implementation = "flex_attention"
    layer_types = list(getattr(config, "layer_types", []) or [])
    config.layer_types = (
        layer_types[: config.num_hidden_layers]
        if len(layer_types) >= config.num_hidden_layers
        else ["full_attention"] * config.num_hidden_layers
    )
    return config


def build_draft_model(args, *, model_role: str, source_config=None):
    config = build_draft_config(
        args, model_role=model_role, source_config=source_config
    )
    return DLiteDraftModel(config).cuda().to(torch.bfloat16)


def build_target_model(args, draft_models: list[DLiteDraftModel]):
    target_config = load_text_model_config(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    source_model_type = getattr(
        target_config, "dlite_source_model_type", target_config.model_type
    )
    if (
        is_qwen35_model_type(source_model_type)
        and args.target_model_backend != "sglang"
    ):
        raise ValueError(
            "Qwen3.5 targets require --target-model-backend sglang with the "
            "pinned Transformers version; the DLite draft remains a dense "
            "Qwen3-style model."
        )
    backend_kwargs = {}
    if args.target_model_backend == "sglang":
        if is_qwen35_model_type(source_model_type):
            # SGLang 0.5.9's Qwen3.5 hybrid GDN path is validated with FA3;
            # FlashInfer attempts to JIT an incompatible fallback prefill path.
            args.sglang_attention_backend = "fa3"
        backend_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()
        if backend_kwargs["max_running_requests"] is None:
            backend_kwargs["max_running_requests"] = int(args.batch_size)
        if backend_kwargs["max_total_tokens"] is None:
            backend_kwargs["max_total_tokens"] = int(args.batch_size) * int(
                args.max_length
            )
    target = get_dlite_target_model(
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


def build_target_and_components(args, draft_models: list[DLiteDraftModel]):
    """Build one target and bind its tokenizer/embedding/head consistently."""
    offline = bool(args.train_hidden_states_path)
    target = None if offline else build_target_model(args, draft_models)
    tokenizer, components, mask_token_id = resolve_tokenizer_and_components(
        args,
        draft_models,
        target=target,
        standalone_components=offline,
    )
    return target, tokenizer, components, mask_token_id


def _select_mask_token_id(
    *,
    explicit_id: Optional[int],
    configured_ids: list[Optional[int]],
    tokenizer_mask_id: Optional[int],
    used_token_ids: set[int],
    vocab_size: int,
) -> tuple[int, str]:
    """Select a real embedding row without assuming a model-specific token ID."""
    vocab_size = int(vocab_size)
    if vocab_size <= 0:
        raise ValueError(f"Target vocabulary size must be positive, got {vocab_size}.")

    if explicit_id is not None:
        candidate = int(explicit_id)
        source = "explicit --mask-token-id"
    else:
        configured = {int(value) for value in configured_ids if value is not None}
        if len(configured) > 1:
            raise ValueError(
                "Draft checkpoints disagree on mask_token_id: "
                f"{sorted(configured)}. Pass --mask-token-id explicitly."
            )
        if configured:
            candidate = configured.pop()
            source = "draft checkpoint"
        elif tokenizer_mask_id is not None:
            candidate = int(tokenizer_mask_id)
            source = "tokenizer.mask_token_id"
        else:
            candidate = next(
                (
                    token_id
                    for token_id in range(vocab_size)
                    if token_id not in used_token_ids
                ),
                -1,
            )
            source = "first tokenizer-unused model vocabulary row"

    if not 0 <= candidate < vocab_size:
        if candidate < 0 and explicit_id is None:
            raise ValueError(
                "No tokenizer-unused row exists inside the target embedding vocabulary. "
                "Configure a model-provided mask token or pass --mask-token-id."
            )
        raise ValueError(
            f"MASK token id {candidate} selected from {source} is outside target "
            f"vocabulary [0, {vocab_size})."
        )
    return candidate, source


def resolve_tokenizer_and_components(
    args, draft_models, target=None, *, standalone_components: bool = False
):
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    target_config = load_text_model_config(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    configured_mask_ids = [
        draft.config.dlite_config.get("mask_token_id")
        for draft in draft_models
        if getattr(draft.config, "dlite_config", None)
    ]
    mask_token_id, mask_source = _select_mask_token_id(
        explicit_id=args.mask_token_id,
        configured_ids=configured_mask_ids,
        tokenizer_mask_id=tokenizer.mask_token_id,
        used_token_ids={int(value) for value in tokenizer.get_vocab().values()},
        vocab_size=int(target_config.vocab_size),
    )
    args.mask_token_id = mask_token_id
    print_on_rank0(
        f"Resolved MASK token id {mask_token_id} from {mask_source}; "
        f"target vocab_size={int(target_config.vocab_size)}."
    )
    if args.target_model_backend == "sglang" and not standalone_components:
        if target is None or not hasattr(target, "model_runner"):
            raise ValueError("SGLang target is required to reuse target components.")
        target_model = target.model_runner.model
        target_embedding = target_model.get_input_embeddings()
        target_lm_head = target_model.lm_head
        components = SharedTargetEmbeddingsAndHead(
            SGLangTPEmbeddingAdapter(target_embedding, get_tp_group(), mask_token_id),
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
            embed_key=args.embedding_key,
            lm_head_key=args.lm_head_key,
            device="cuda",
            trust_remote_code=args.trust_remote_code,
        )
        if not 0 <= mask_token_id < components.embed_tokens.num_embeddings:
            raise ValueError(
                "DLite vocab_row MASK mode requires an existing target "
                f"embedding row, but mask_token_id={mask_token_id} and target "
                f"vocab size={components.embed_tokens.num_embeddings}. Pass "
                "--mask-token-id with an unused in-vocabulary row."
            )
    for draft in draft_models:
        draft.mask_token_id = mask_token_id
        draft.mask_embedding_mode = "vocab_row"
        draft.config.dlite_config["mask_token_id"] = mask_token_id
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
    return build_training_dataset(
        dataset=raw,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        is_preformatted=args.is_preformatted,
        cache_dir=os.path.join(args.cache_dir, cache_namespace, "processed_dataset"),
        cache_key=cache_key,
        num_proc=(args.build_dataset_num_proc if num_proc is None else int(num_proc)),
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
        pad_to_length=args.max_length if args.pad_to_max_length else None,
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
    required_layer_ids: Optional[set[int]] = None,
):
    if args.train_hidden_states_path:
        if train_data_path is not None:
            raise ValueError("online train_data_path cannot be used in offline mode")
        return build_hidden_cache_dataloader(
            args,
            tokenizer,
            required_layer_ids=set(required_layer_ids or ()),
        )
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
    return _prepare_dataloader(args, dataset, train_data_path=train_data_path)


def build_hidden_cache_dataloader(args, tokenizer, *, required_layer_ids: set[int]):
    target_config = load_text_model_config(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    verify = not dist.is_initialized() or dist.get_rank() == 0
    manifest = load_hidden_cache_manifest(
        args.train_hidden_states_path,
        target_model=args.target_model_path,
        num_hidden_layers=int(target_config.num_hidden_layers),
        hidden_size=int(target_config.hidden_size),
        required_layer_ids=required_layer_ids,
        max_length=int(args.max_length),
        verify_checksums=verify,
    )
    if dist.is_initialized():
        dist.barrier()
        if not verify:
            manifest = load_hidden_cache_manifest(
                args.train_hidden_states_path,
                target_model=args.target_model_path,
                num_hidden_layers=int(target_config.num_hidden_layers),
                hidden_size=int(target_config.hidden_size),
                required_layer_ids=required_layer_ids,
                max_length=int(args.max_length),
                verify_checksums=False,
            )
    dataset = RegenFullDataset(
        args.train_hidden_states_path,
        manifest,
        required_layer_ids=required_layer_ids,
        max_length=args.max_length,
    )
    scan_here = not dist.is_initialized() or dist.get_rank() == 0
    valid_indices = (
        [
            index
            for index in range(len(dataset))
            if _has_valid_anchor_supervision(dataset[index], block_size=args.block_size)
        ]
        if scan_here
        else None
    )
    dataset.close()
    if dist.is_initialized():
        shared_indices = [valid_indices]
        dist.broadcast_object_list(shared_indices, src=0)
        valid_indices = shared_indices[0]
    assert valid_indices is not None
    dataset = Subset(dataset, valid_indices)
    process_group = get_dp_group()
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(process_group),
        rank=dist.get_rank(process_group),
        shuffle=True,
    )
    if args.dataloader_num_workers == 0:
        prefetch_factor = None
    else:
        prefetch_factor = 2
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.dataloader_num_workers,
        prefetch_factor=prefetch_factor,
        collate_fn=RegenFullCollator(
            tokenizer.pad_token_id or 0,
            args.max_length if args.pad_to_max_length else None,
        ),
        drop_last=True,
    )
    if len(dataloader) == 0:
        raise ValueError(
            f"Training cache {args.train_hidden_states_path!r} has no full batches "
            "after filtering."
        )
    return dataloader


def training_data_identity(args) -> str:
    path = args.train_hidden_states_path or args.train_data_path
    identity = os.path.realpath(path)
    if args.train_hidden_states_path:
        manifest_path = os.path.join(identity, "manifest.json")
        digest = hashlib.sha256()
        with open(manifest_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        identity = f"{identity}#manifest_sha256={digest.hexdigest()}"
    return identity


def required_hidden_layer_ids(draft_models: list[DLiteDraftModel]) -> set[int]:
    required: set[int] = set()
    for draft in draft_models:
        required.update(int(value) for value in draft.target_layer_ids)
        if draft.is_teacher:
            required.update(int(value) for value in draft.history_layer_ids)
    return required


def hidden_states_to_cuda(hidden_states):
    if isinstance(hidden_states, dict):
        return {key: value.cuda() for key, value in hidden_states.items()}
    return tuple(value.cuda() for value in hidden_states)


def project_cached_target_logits(
    final_hidden_state: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_size: int,
    lm_head: torch.nn.Module,
) -> torch.Tensor:
    """Project only cached positions needed by the DLite target loss."""
    offsets = torch.arange(int(block_size) - 1, device=anchor_positions.device).view(
        1, 1, -1
    )
    positions = anchor_positions.unsqueeze(-1) + offsets
    if bool((positions >= final_hidden_state.size(1)).any()):
        raise ValueError("Cached final hidden states do not cover target positions.")
    expanded = final_hidden_state.unsqueeze(1).expand(
        -1, anchor_positions.size(1), -1, -1
    )
    selected = torch.gather(
        expanded,
        2,
        positions.unsqueeze(-1).expand(-1, -1, -1, final_hidden_state.size(-1)),
    )
    with torch.no_grad():
        return lm_head(selected)


def load_cached_target_data(
    data,
    *,
    anchors: torch.Tensor,
    block_size: int,
    lm_head: torch.nn.Module,
    need_logits: bool,
):
    hidden_states = hidden_states_to_cuda(data["hidden_states"])
    target_logits = None
    if need_logits:
        final_layer_id = int(data["final_norm_layer_id"])
        final_hidden = hidden_states[final_layer_id]
        target_logits = project_cached_target_logits(
            final_hidden, anchors, block_size, lm_head
        )
    return hidden_states, target_logits


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
        return {key: select_tp_rank_batch(item, tp_rank) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(select_tp_rank_batch(item, tp_rank) for item in value)
    if isinstance(value, list):
        return [select_tp_rank_batch(item, tp_rank) for item in value]
    if not isinstance(value, torch.Tensor) or value.ndim == 0:
        raise TypeError(
            "TP draft batch values must be batch-first tensors or containers."
        )
    if not 0 <= int(tp_rank) < value.size(0):
        raise ValueError(
            f"TP rank {tp_rank} cannot select from batch size {value.size(0)}."
        )
    return value.narrow(0, int(tp_rank), 1).clone(memory_format=torch.contiguous_format)


def save_checkpoint(
    *,
    output_dir: str,
    name: str,
    fsdp_model: FSDP,
    draft_model: DLiteDraftModel,
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
            "chs_num_layers": draft_model.chs_num_layers,
            "block_size": draft_model.block_size,
            "num_draft_layers": draft_model.config.num_hidden_layers,
            "sequential_head": draft_model.sequential_head_type,
            "sequential_rank": draft_model.sequential_rank,
        }
        save_distributed_training_state(
            save_dir, {**model_metadata, **metadata, **optimizer.state_dict()}
        )
        if dist.get_rank() == 0:
            draft_model.save_pretrained(save_dir, state_dict=draft_state)
            for filename in ("dlite.py", "sequential_head.py"):
                source = os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "specforge",
                    "modeling",
                    "draft",
                    filename,
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
            f"Expected a {stage!r} checkpoint, got {state.get('training_stage')!r}."
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
    return math.ceil(int(epochs) * len(dataloader) / int(accumulation_steps))


def normalize_accumulated_gradients(
    optimizer,
    local_denominator: torch.Tensor,
    accumulation_steps: int,
    group=None,
) -> float:
    """Normalize rank-averaged gradients by the global optimizer-window weight.

    Every microbatch backpropagates its additive loss numerator divided by the
    configured accumulation length. FSDP averages those gradients across ranks.
    Multiplying by ``world_size * accumulation_steps / global_denominator``
    therefore produces the gradient of the summed numerator divided by the sum
    of effective weights across every microbatch and DP rank in the window.
    """
    if int(accumulation_steps) <= 0:
        raise ValueError("accumulation_steps must be positive")
    denominator = local_denominator.detach().float().clone()
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size(group=group)
        if world_size > 1:
            dist.all_reduce(denominator, op=dist.ReduceOp.SUM, group=group)
    denominator_value = float(denominator.item())
    if not math.isfinite(denominator_value) or denominator_value <= 0:
        raise ValueError("global loss denominator must be finite and positive")
    scale = world_size * int(accumulation_steps) / denominator_value
    optimizer.scale_model_gradients(scale)
    return denominator_value


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
    "hidden_states_to_cuda",
    "load_training_state",
    "log_cuda_peak",
    "normalize_accumulated_gradients",
    "resolve_tokenizer_and_components",
    "resume_cursor",
    "save_checkpoint",
    "select_tp_rank_batch",
    "stage_total_steps",
    "validate_common_args",
    "validate_tp_draft_sharding",
]
