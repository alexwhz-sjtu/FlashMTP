"""Disaggregated target production and draft-only FSDP training for DLite."""

from __future__ import annotations

import copy
import faulthandler
import math
import os
import signal
import traceback
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from scripts.dlite.dlite_training import (
    _build_processed_dataset,
    _has_valid_anchor_supervision,
    build_draft_model,
    build_target_model,
    normalize_accumulated_gradients,
    project_cached_target_logits,
    resolve_tokenizer_and_components,
    save_checkpoint,
    training_data_identity,
)
from specforge.core.dlite import (
    OnlineDLiteModel,
    compute_stage1_distillation_loss,
    prepare_history_hidden_states,
    prepare_target_hidden,
)
from specforge.disaggregate import (
    DraftBatchPacket,
    DraftPacketSpec,
    NodePacketTransport,
    build_node_routes,
)
from specforge.distributed import destroy_distributed, init_disaggregated
from specforge.modeling.draft.dlite import DLiteDraftModel
from specforge.optimizer import BF16Optimizer
from specforge.tracker import create_tracker


@dataclass(frozen=True)
class Phase:
    name: str
    phase_id: int
    epochs: int
    need_history: bool
    need_prediction_hidden: bool


def _debug(topology, message: str) -> None:
    if os.environ.get("SPECFORGE_DISAGG_DEBUG") == "1":
        print(f"[disagg rank {topology.rank} {topology.role}] {message}", flush=True)


class FixedCollator:
    def __init__(self, max_length: int, pad_token_id: int):
        self.max_length = int(max_length)
        self.pad_token_id = int(pad_token_id)

    def _pad(self, value, *, fill=0, dtype=None):
        value = torch.as_tensor(value).reshape(-1)[: self.max_length]
        output = torch.full((self.max_length,), fill, dtype=dtype or value.dtype)
        output[: value.numel()].copy_(value.to(output.dtype))
        return output

    def __call__(self, features):
        return {
            "input_ids": torch.stack(
                [
                    self._pad(row["input_ids"], fill=self.pad_token_id)
                    for row in features
                ]
            ),
            "attention_mask": torch.stack(
                [self._pad(row["attention_mask"], fill=0) for row in features]
            ),
            "loss_mask": torch.stack(
                [
                    self._pad(row["loss_mask"], fill=0, dtype=torch.float32)
                    for row in features
                ]
            ),
        }


def topology_identity(args, topology) -> dict:
    return {
        "execution_mode": "disaggregate",
        "target_ranks_per_node": int(args.target_ranks_per_node),
        "draft_ranks_per_node": int(args.draft_ranks_per_node),
        "target_tp_size": int(args.target_tp_size),
        "target_ep_size": int(args.sglang_ep_size),
        "node_batch_size": int(args.node_batch_size),
        "global_batch_size": int(args.node_batch_size) * int(topology.nnodes),
        "draft_world_size": len(topology.draft_global_ranks),
    }


def validate_resume_topology(state: Optional[dict], identity: dict) -> None:
    if state is None:
        return
    for key, current in identity.items():
        saved = state.get(key)
        if saved is not None and saved != current:
            raise ValueError(
                f"Disaggregated resume topology mismatch for {key}: "
                f"checkpoint={saved!r}, current={current!r}"
            )


def validate_resume_model(state: Optional[dict], draft: DLiteDraftModel) -> None:
    if state is None:
        return
    expected = {
        "architecture_version": draft.architecture_version,
        "model_role": draft.model_role,
        "target_layer_ids": [int(value) for value in draft.target_layer_ids],
        "num_target_layers": int(draft.config.num_target_layers),
        "target_hidden_size": int(draft.config.hidden_size),
    }
    for key, current in expected.items():
        saved = state.get(key)
        if saved is not None and saved != current:
            raise ValueError(
                f"Disaggregated resume model mismatch for {key}: "
                f"checkpoint={saved!r}, current={current!r}"
            )


def _load_common_state(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    common = os.path.join(path, "training_state.pt")
    if not os.path.isfile(common):
        raise FileNotFoundError(f"No training_state.pt in {path!r}")
    return torch.load(common, map_location="cpu", weights_only=False)


def _sync_args_from_draft(args, draft: DLiteDraftModel) -> None:
    args.block_size = int(draft.block_size)
    args.dlite_version = draft.architecture_version
    args.num_draft_layers = int(draft.config.num_hidden_layers)
    args.chs_num_layers = int(draft.chs_num_layers)
    args.target_layer_ids = ",".join(str(value) for value in draft.target_layer_ids)
    args.sequential_head = draft.sequential_head_type
    args.sequential_rank = int(draft.sequential_rank)
    if draft.is_teacher:
        args.swa_window_size = int(draft.swa_window_size)


def _copy_serial_head(source: DLiteDraftModel, destination: DLiteDraftModel) -> None:
    if source.sequential_head is None or destination.sequential_head is None:
        if source.sequential_head is not destination.sequential_head:
            raise ValueError("teacher/student sequential heads do not match")
        return
    destination.sequential_head.load_state_dict(source.sequential_head.state_dict())


def _initialize_drafts(args, mode: str, common_state: Optional[dict]):
    resume_stage = common_state.get("training_stage") if common_state else None
    teacher = None
    if mode == "teacher":
        source = args.resume_from or args.init_from
        draft = (
            DLiteDraftModel.from_pretrained(
                source,
                torch_dtype=torch.bfloat16,
                attn_implementation="flex_attention",
            ).cuda()
            if source
            else build_draft_model(args, model_role="swa_teacher")
        )
        _sync_args_from_draft(args, draft)
        return draft, None, resume_stage
    if mode == "sft":
        source = args.resume_from or args.init_from
        draft = (
            DLiteDraftModel.from_pretrained(
                source,
                torch_dtype=torch.bfloat16,
                attn_implementation="flex_attention",
            ).cuda()
            if source
            else build_draft_model(args, model_role="pivot_q_student")
        )
        if not draft.is_student:
            raise ValueError("SFT requires a pivot_q_student draft")
        _sync_args_from_draft(args, draft)
        return draft, None, resume_stage

    if common_state:
        student = DLiteDraftModel.from_pretrained(
            args.resume_from,
            torch_dtype=torch.bfloat16,
            attn_implementation="flex_attention",
        ).cuda()
        if not student.is_student:
            raise ValueError("two-stage resume requires a pivot_q_student checkpoint")
    else:
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
            raise ValueError("--teacher-draft-path must contain an swa_teacher")
        teacher.requires_grad_(False)
        _sync_args_from_draft(args, teacher)
        student = build_draft_model(
            args,
            model_role="pivot_q_student",
            source_config=copy.deepcopy(teacher.config),
        )
        _copy_serial_head(teacher, student)
    _sync_args_from_draft(args, student)
    if resume_stage in ("stage1", "transition") and teacher is None:
        if not args.teacher_draft_path:
            raise ValueError("Stage 1/transition resume requires --teacher-draft-path")
        teacher = (
            DLiteDraftModel.from_pretrained(
                args.teacher_draft_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flex_attention",
            )
            .cuda()
            .eval()
        )
        teacher.requires_grad_(False)
    return student, teacher, resume_stage


def _draft_metadata(draft, teacher=None) -> dict:
    layer_ids = set(int(value) for value in draft.target_layer_ids)
    history_ids = set()
    if teacher is not None:
        layer_ids.update(int(value) for value in teacher.target_layer_ids)
        history_ids.update(int(value) for value in teacher.history_layer_ids)
    if draft.is_teacher:
        history_ids.update(int(value) for value in draft.history_layer_ids)
    return {
        "target_layer_ids": sorted(layer_ids),
        "history_layer_ids": sorted(history_ids),
        "num_target_layers": int(draft.config.num_target_layers),
        "hidden_size": int(draft.config.hidden_size),
        "block_size": int(draft.block_size),
        "draft_query_length": int(draft.draft_query_length),
        "chs_num_layers": int(draft.chs_num_layers),
    }


def _broadcast_metadata(topology, local_metadata):
    source = topology.draft_global_ranks[0]
    payload = [local_metadata if topology.rank == source else None]
    dist.broadcast_object_list(payload, src=source)
    return payload[0]


def _build_target_loader(args, tokenizer, topology):
    dataset = None
    if topology.rank == 0:
        dataset = _build_processed_dataset(
            args,
            tokenizer,
            train_data_path=args.train_data_path,
            cache_namespace="disaggregate",
            num_proc=args.build_dataset_num_proc,
        )
        dataset = dataset.filter(
            _has_valid_anchor_supervision,
            fn_kwargs={"block_size": int(args.block_size)},
        )
    dist.barrier()
    if topology.is_target_leader and topology.rank != 0:
        dataset = _build_processed_dataset(
            args,
            tokenizer,
            train_data_path=args.train_data_path,
            cache_namespace="disaggregate",
            num_proc=args.build_dataset_num_proc,
        )
        dataset = dataset.filter(
            _has_valid_anchor_supervision,
            fn_kwargs={"block_size": int(args.block_size)},
        )
    size = torch.tensor(
        [len(dataset) if topology.rank == 0 else 0], device="cuda", dtype=torch.long
    )
    dist.broadcast(size, src=0)
    producers_total = topology.nnodes * topology.target_replicas_per_node
    local_batch = int(args.node_batch_size) // topology.target_replicas_per_node
    steps = int(size.item()) // (producers_total * local_batch)
    if steps <= 0:
        raise ValueError("dataset is too small for one disaggregated global batch")
    loader = None
    if topology.is_target_leader:
        producer_rank = topology.node_rank * topology.target_replicas_per_node + int(
            topology.target_replica_local_rank
        )
        sampler = DistributedSampler(
            dataset,
            num_replicas=producers_total,
            rank=producer_rank,
            shuffle=True,
            drop_last=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=local_batch,
            sampler=sampler,
            num_workers=args.dataloader_num_workers,
            collate_fn=FixedCollator(
                args.max_length, tokenizer.pad_token_id or tokenizer.eos_token_id or 0
            ),
            drop_last=True,
            prefetch_factor=2 if args.dataloader_num_workers else None,
        )
    return loader, steps


def _broadcast_input(args, topology, data):
    shape = (int(args.target_batch_size), int(args.max_length))
    if topology.is_target_leader:
        values = (
            data["input_ids"].cuda(non_blocking=True),
            data["attention_mask"].cuda(non_blocking=True),
            data["loss_mask"].cuda(non_blocking=True),
        )
    else:
        values = (
            torch.empty(shape, dtype=torch.long, device="cuda"),
            torch.empty(shape, dtype=torch.long, device="cuda"),
            torch.empty(shape, dtype=torch.float32, device="cuda"),
        )
    for value in values:
        dist.broadcast(
            value,
            src=topology.target_tp_leader_global_rank,
            group=topology.target_tp_group,
        )
    return values


def aligned_anchor_count(num_anchors: int, query_length: int, chs_slots: int) -> int:
    alignment = 1
    for value in (query_length, query_length + chs_slots):
        alignment = math.lcm(alignment, 128 // math.gcd(128, int(value)))
    return math.ceil(int(num_anchors) / alignment) * alignment


def _sample_anchors(args, topology, loss_mask, batch_id, metadata):
    bsz, seq_len = loss_mask.shape
    count = aligned_anchor_count(
        args.num_anchors,
        metadata["draft_query_length"],
        metadata["chs_num_layers"],
    )
    if topology.is_target_leader:
        max_anchor = max(seq_len - int(args.block_size), 0)
        valid = loss_mask[:, : max_anchor + 1] > 0.5
        valid = valid & (loss_mask[:, 1 : max_anchor + 2] > 0.5)
        if valid.size(1):
            valid[:, 0] = False
        generator = torch.Generator(device=loss_mask.device)
        generator.manual_seed(
            int(args.seed)
            + 1_000_003 * int(batch_id)
            + 10_007 * int(topology.node_rank)
            + 101 * int(topology.target_replica_local_rank)
        )
        random = torch.rand(valid.shape, generator=generator, device=loss_mask.device)
        random.masked_fill_(~valid, 2.0)
        width = min(count, valid.size(1))
        indices = random.argsort(dim=1)[:, :width]
        selected = torch.gather(valid, 1, indices)
        sentinel = max_anchor + 1
        indices.masked_fill_(~selected, sentinel)
        if width < count:
            indices = torch.cat(
                [indices, indices.new_full((bsz, count - width), sentinel)], dim=1
            )
        indices = indices.sort(dim=1).values
        keep = indices.ne(sentinel)
        anchors = torch.where(keep, indices, torch.zeros_like(indices))
        if not bool(keep.any()):
            raise ValueError("batch has no valid DLite anchors")
    else:
        anchors = torch.empty((bsz, count), dtype=torch.long, device="cuda")
        keep = torch.empty((bsz, count), dtype=torch.bool, device="cuda")
    for value in (anchors, keep):
        dist.broadcast(
            value,
            src=topology.target_tp_leader_global_rank,
            group=topology.target_tp_group,
        )
    return anchors, keep


def _prediction_hidden(final_hidden, anchors, block_size):
    offsets = torch.arange(block_size - 1, device=anchors.device).view(1, 1, -1)
    positions = anchors.unsqueeze(-1) + offsets
    expanded = final_hidden.unsqueeze(1).expand(-1, anchors.size(1), -1, -1)
    return torch.gather(
        expanded,
        2,
        positions.unsqueeze(-1).expand(-1, -1, -1, final_hidden.size(-1)),
    )


def _produce_packet(args, topology, target, phase, metadata, values, batch_id):
    input_ids, attention_mask, loss_mask = values
    output = target.generate_dlite_data(
        input_ids, attention_mask, loss_mask, return_logits=False
    )
    anchors, keep = _sample_anchors(args, topology, loss_mask, batch_id, metadata)
    if not topology.is_target_leader:
        return None
    hidden = output.hidden_states
    target_hidden = prepare_target_hidden(
        hidden,
        anchors,
        metadata["target_layer_ids"],
        metadata["num_target_layers"],
    ).to(torch.bfloat16)
    history = None
    if phase.need_history:
        history = prepare_history_hidden_states(
            hidden,
            metadata["history_layer_ids"],
            metadata["num_target_layers"],
        ).to(torch.bfloat16)
    prediction = None
    if phase.need_prediction_hidden:
        final_layer = metadata["num_target_layers"] - 1
        final_hidden = hidden[final_layer] if isinstance(hidden, dict) else hidden[-1]
        prediction = _prediction_hidden(
            final_hidden, anchors, metadata["block_size"]
        ).to(torch.bfloat16)
    return DraftBatchPacket(
        input_ids=input_ids,
        loss_mask=loss_mask,
        anchor_positions=anchors,
        block_keep_mask=keep,
        target_hidden=target_hidden,
        target_history_hidden=history,
        target_prediction_hidden=prediction,
    )


def _packet_spec(args, topology, phase, metadata):
    local_batch = (
        int(args.node_batch_size) // topology.target_replicas_per_node
        if topology.is_target
        else int(args.node_batch_size) // topology.draft_ranks_per_node
    )
    return DraftPacketSpec(
        batch_size=local_batch,
        max_length=int(args.max_length),
        num_anchors=aligned_anchor_count(
            args.num_anchors,
            metadata["draft_query_length"],
            metadata["chs_num_layers"],
        ),
        num_target_layers=len(metadata["target_layer_ids"]),
        hidden_size=metadata["hidden_size"],
        prediction_length=metadata["block_size"] - 1,
        num_history_layers=(
            len(metadata["history_layer_ids"]) if phase.need_history else 0
        ),
        include_target_prediction_hidden=phase.need_prediction_hidden,
    )


def _run_target(args, topology, metadata, phases, loader, steps, routes, start_id):
    _debug(topology, "building target model")
    capture = set(metadata["target_layer_ids"])
    capture.update(metadata["history_layer_ids"])
    if any(phase.need_prediction_hidden for phase in phases):
        capture.add(metadata["num_target_layers"] - 1)
    proxies = [
        SimpleNamespace(
            target_layer_ids=sorted(capture),
            is_teacher=bool(metadata["history_layer_ids"]),
            history_layer_ids=metadata["history_layer_ids"],
        )
    ]
    target = build_target_model(args, proxies)
    _debug(topology, "target model ready")
    transport = (
        NodePacketTransport(topology=topology, routes=routes, profile=args.profile)
        if topology.is_target_leader
        else None
    )
    batch_id = 0
    for phase in phases:
        bridge_ms = 0.0
        spec = _packet_spec(args, topology, phase, metadata)
        slots = [None] * int(args.pipeline_depth)
        phase_batches = phase.epochs * steps
        if phase.name == "teacher" and args.max_steps is not None:
            phase_batches = min(phase_batches, int(args.max_steps))
        phase_progress = 0
        for epoch in range(phase.epochs):
            if topology.is_target_leader:
                loader.sampler.set_epoch(phase.phase_id * 1_000_000 + epoch)
                iterator = iter(loader)
            for _ in range(steps):
                if phase_progress >= phase_batches:
                    break
                if batch_id < start_id:
                    if topology.is_target_leader:
                        next(iterator)
                    batch_id += 1
                    phase_progress += 1
                    continue
                slot = batch_id % int(args.pipeline_depth)
                if topology.is_target_leader and slots[slot] is not None:
                    assert transport is not None
                    bridge_ms += transport.wait_send(slots[slot])
                _debug(topology, f"phase={phase.name} batch={batch_id} loading input")
                data = next(iterator) if topology.is_target_leader else None
                _debug(
                    topology, f"phase={phase.name} batch={batch_id} broadcasting input"
                )
                values = _broadcast_input(args, topology, data)
                _debug(topology, f"phase={phase.name} batch={batch_id} input ready")
                _debug(topology, f"phase={phase.name} batch={batch_id} prefill start")
                packet = _produce_packet(
                    args, topology, target, phase, metadata, values, batch_id
                )
                _debug(topology, f"phase={phase.name} batch={batch_id} prefill done")
                if topology.is_target_leader:
                    assert transport is not None
                    slots[slot] = transport.send(
                        packet,
                        batch_id=batch_id,
                        phase_id=phase.phase_id,
                        schema_id=spec.schema_id,
                    )
                batch_id += 1
                phase_progress += 1
            if phase_progress >= phase_batches:
                break
        if topology.is_target_leader:
            assert transport is not None
            for handle in slots:
                if handle is not None:
                    bridge_ms += transport.wait_send(handle)
            if args.profile:
                print(
                    f"disagg target bridge phase={phase.name} "
                    f"producer={topology.target_replica_local_rank} ms={bridge_ms:.3f}",
                    flush=True,
                )
        # Drains this packet schema before any role moves to the next phase.
        dist.barrier()


def _select_packet_layers(packet, union_ids, requested_ids):
    indices = [union_ids.index(int(layer_id)) for layer_id in requested_ids]
    return packet.target_hidden[:, :, indices, :]


def _target_logits(packet, phase, metadata, lm_head):
    with torch.no_grad():
        if packet.target_prediction_hidden is not None:
            return lm_head(packet.target_prediction_hidden)
        if packet.target_history_hidden is None:
            raise ValueError(
                f"phase {phase.name} has no final hidden for target logits"
            )
        final_layer = metadata["num_target_layers"] - 1
        history_index = metadata["history_layer_ids"].index(final_layer)
        return project_cached_target_logits(
            packet.target_history_hidden[:, :, history_index, :],
            packet.anchor_positions,
            metadata["block_size"],
            lm_head,
        )


def _metadata_for_checkpoint(args, topology, phase, next_id, **extra):
    return {
        "training_stage": phase.name,
        "disagg_batch_id_next": int(next_id),
        "global_step": int(next_id),
        "train_data_identity": training_data_identity(args),
        "train_data_mode": "online",
        "target_model_path": os.path.realpath(args.target_model_path),
        **topology_identity(args, topology),
        **extra,
    }


def _save(
    args, topology, fsdp, draft, optimizer, phase, next_id, *, name=None, **extra
):
    return save_checkpoint(
        output_dir=args.output_dir,
        name=name or f"{phase.name}/step_{next_id}",
        fsdp_model=fsdp,
        draft_model=draft,
        optimizer=optimizer,
        metadata=_metadata_for_checkpoint(args, topology, phase, next_id, **extra),
        process_group=topology.draft_group,
        coordinator_global_rank=topology.draft_global_ranks[0],
    )


def _make_online(args, draft, components, process_group, *, teacher_defaults=False):
    kwargs = dict(
        draft_model=draft,
        target_lm_head=components.lm_head,
        target_embed_tokens=components.embed_tokens,
        mask_token_id=args.mask_token_id,
        block_size=draft.block_size,
        num_anchors=args.num_anchors,
        process_group=process_group,
    )
    if not teacher_defaults:
        if hasattr(args, "stage2_final_ce_weight"):
            kwargs.update(
                loss_decay_gamma=args.stage2_loss_decay_gamma,
                final_ce_weight=args.stage2_final_ce_weight,
                tv_loss_weight=args.stage2_tv_weight,
                base_lm_ce_weight=args.stage2_base_ce_weight,
                base_lm_ce_decay_gamma=args.stage2_base_ce_decay_gamma,
                use_target_greedy_ce_labels=True,
            )
        else:
            kwargs.update(
                loss_decay_gamma=args.loss_decay_gamma,
                final_ce_weight=args.final_ce_weight,
                tv_loss_weight=args.tv_loss_weight,
                base_lm_ce_weight=args.base_lm_ce_weight,
                base_lm_ce_decay_gamma=args.base_lm_ce_decay_gamma,
                use_target_greedy_ce_labels=True,
            )
    return OnlineDLiteModel(**kwargs)


def _run_draft(
    args,
    topology,
    mode,
    metadata,
    phases,
    steps,
    routes,
    start_id,
    common_state,
    initialized_drafts,
):
    draft, teacher, resume_stage = initialized_drafts
    _debug(topology, "loading standalone target embedding/head")
    draft.requires_grad_(True)
    tokenizer, components, _ = resolve_tokenizer_and_components(
        args,
        [draft] + ([teacher] if teacher is not None else []),
        standalone_components=True,
    )
    _debug(topology, "standalone target embedding/head ready")
    online = _make_online(args, draft, components, topology.draft_group)
    teacher_online = (
        _make_online(
            args, teacher, components, topology.draft_group, teacher_defaults=True
        )
        if teacher is not None
        else None
    )
    fsdp = FSDP(
        online,
        process_group=topology.draft_group,
        ignored_modules=[components.lm_head, components.embed_tokens],
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    )
    _debug(topology, "draft FSDP ready")
    total_batches = sum(phase.epochs * steps for phase in phases)
    total_optimizer_steps = math.ceil(total_batches / int(args.accumulation_steps))
    _debug(topology, "building optimizer")
    optimizer = BF16Optimizer(
        draft,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=total_optimizer_steps,
        process_group=topology.draft_group,
    )
    _debug(topology, "optimizer ready")
    if common_state:
        from specforge.checkpoint import load_distributed_training_state

        local_state = load_distributed_training_state(
            args.resume_from,
            map_location="cpu",
            process_group=topology.draft_group,
        )
        optimizer.load_state_dict(local_state)
    args.tracker_global_rank = topology.draft_global_ranks[0]
    tracker = create_tracker(args, args.output_dir)
    _debug(topology, "tracker ready")
    transport = NodePacketTransport(
        topology=topology, routes=routes, profile=args.profile
    )
    local_batch = int(args.node_batch_size) // int(args.draft_ranks_per_node)
    micro_batch = int(args.draft_micro_batch_size)
    micro_count = local_batch // micro_batch
    batch_id = 0
    accumulated = 0
    denominator_sum = None
    for phase in phases:
        bridge_ms = 0.0
        spec = _packet_spec(args, topology, phase, metadata)
        phase_total = phase.epochs * steps
        if phase.name == "teacher" and args.max_steps is not None:
            phase_total = min(phase_total, int(args.max_steps))
        phase_start = batch_id
        phase_end = phase_start + phase_total
        if phase_end <= start_id:
            batch_id = phase_end
            dist.barrier()
            continue
        slots = [
            DraftBatchPacket.empty(spec, device="cuda")
            for _ in range(args.pipeline_depth)
        ]
        _debug(topology, f"phase={phase.name} packet slots ready")
        if mode == "two_stage" and draft.sequential_head is not None:
            draft.sequential_head.requires_grad_(phase.name != "stage1")
        handles = [None] * int(args.pipeline_depth)
        first = max(batch_id, start_id)
        first_slot = first % int(args.pipeline_depth)
        _debug(topology, f"phase={phase.name} posting first receive batch={first}")
        handles[first_slot] = transport.receive(
            slots[first_slot],
            batch_id=first,
            phase_id=phase.phase_id,
            schema_id=spec.schema_id,
        )
        _debug(topology, f"phase={phase.name} posted first receive batch={first}")
        for batch_id in range(first, phase_end):
            slot = batch_id % int(args.pipeline_depth)
            bridge_ms += transport.wait_receive(handles[slot])
            _debug(topology, f"phase={phase.name} received batch={batch_id}")
            next_id = batch_id + 1
            if next_id < phase_end:
                next_slot = next_id % int(args.pipeline_depth)
                if handles[next_slot] is None:
                    handles[next_slot] = transport.receive(
                        slots[next_slot],
                        batch_id=next_id,
                        phase_id=phase.phase_id,
                        schema_id=spec.schema_id,
                    )
            packet = slots[slot]
            metric_sum = None
            for micro in range(micro_count):
                begin, end = micro * micro_batch, (micro + 1) * micro_batch
                input_ids = packet.input_ids[begin:end]
                loss_mask = packet.loss_mask[begin:end]
                anchors = packet.anchor_positions[begin:end]
                keep = packet.block_keep_mask[begin:end]
                target_hidden = _select_packet_layers(
                    packet,
                    metadata["target_layer_ids"],
                    draft.target_layer_ids,
                )[begin:end]
                history = (
                    packet.target_history_hidden[begin:end]
                    if packet.target_history_hidden is not None
                    else None
                )
                if mode == "two_stage" and phase.name in ("stage1", "transition"):
                    if teacher_online is None:
                        raise ValueError(
                            "Stage 1/transition requires the teacher draft"
                        )
                    student_batch = online.prepare_batch(
                        input_ids,
                        None,
                        loss_mask,
                        anchor_positions=anchors,
                        block_keep_mask=keep,
                        target_hidden=target_hidden,
                    )
                    teacher_hidden = _select_packet_layers(
                        packet,
                        metadata["target_layer_ids"],
                        teacher.target_layer_ids,
                    )[begin:end]
                    with torch.no_grad():
                        teacher_batch = teacher_online.prepare_batch(
                            input_ids,
                            None,
                            loss_mask,
                            anchor_positions=anchors,
                            block_keep_mask=keep,
                            shared_query_embeddings=student_batch.query_embeddings,
                            target_hidden=teacher_hidden,
                            raw_history_hidden=history,
                        )
                        teacher_prediction = teacher_online.forward_backbone(
                            teacher_batch, seq_len=input_ids.size(1)
                        )
                        teacher_logits = teacher_online.compute_serial_logits(
                            teacher_prediction, teacher_batch
                        )
                    if phase.name == "stage1":
                        _student_hidden, student_logits = fsdp(
                            prepared_batch=student_batch,
                            seq_len=input_ids.size(1),
                            return_backbone_and_serial_logits=True,
                        )
                        loss, kl, numerator, denominator = (
                            compute_stage1_distillation_loss(
                                student_serial_logits=student_logits,
                                teacher_serial_logits=teacher_logits,
                                raw_weight_mask=student_batch.raw_weight_mask,
                                kl_weight=args.stage1_kl_weight,
                                loss_decay_gamma=args.stage1_loss_decay_gamma,
                                process_group=topology.draft_group,
                            )
                        )
                        values = (loss, kl)
                    else:
                        target_logits = _target_logits(
                            DraftBatchPacket(
                                input_ids,
                                loss_mask,
                                anchors,
                                keep,
                                target_hidden,
                                history,
                                None,
                            ),
                            phase,
                            metadata,
                            components.lm_head,
                        )
                        _, student_logits, supervised = fsdp(
                            prepared_batch=student_batch,
                            seq_len=input_ids.size(1),
                            target_prefill_logits=target_logits,
                            target_logits_are_gathered=True,
                            return_transition_outputs=True,
                        )
                        stage2_loss, *stage2_metrics, stage2_num, stage2_den = (
                            supervised
                        )
                        stage1_loss, kl, stage1_num, stage1_den = (
                            compute_stage1_distillation_loss(
                                student_serial_logits=student_logits,
                                teacher_serial_logits=teacher_logits,
                                raw_weight_mask=student_batch.raw_weight_mask,
                                kl_weight=args.stage1_kl_weight,
                                loss_decay_gamma=args.stage1_loss_decay_gamma,
                                process_group=topology.draft_group,
                            )
                        )
                        progress = (batch_id - phase_start) / max(phase_total - 1, 1)
                        stage1_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
                        stage2_scale = 1.0 - stage1_scale
                        loss = stage1_scale * stage1_loss + stage2_scale * stage2_loss
                        numerator = (
                            stage1_scale * stage1_num + stage2_scale * stage2_num
                        )
                        denominator = (
                            stage1_scale * stage1_den + stage2_scale * stage2_den
                        )
                        values = (loss, kl, stage2_loss, *stage2_metrics[:5])
                else:
                    target_logits = _target_logits(
                        DraftBatchPacket(
                            input_ids,
                            loss_mask,
                            anchors,
                            keep,
                            target_hidden,
                            history,
                            (
                                packet.target_prediction_hidden[begin:end]
                                if packet.target_prediction_hidden is not None
                                else None
                            ),
                        ),
                        phase,
                        metadata,
                        components.lm_head,
                    )
                    values = fsdp(
                        input_ids=input_ids,
                        loss_mask=loss_mask,
                        target_hidden=target_hidden,
                        raw_history_hidden=history,
                        anchor_positions=anchors,
                        block_keep_mask=keep,
                        target_prefill_logits=target_logits,
                        target_logits_are_gathered=True,
                    )
                    loss, *_, numerator, denominator = values
                # Denominators are summed over every local microbatch below.
                # Divide only by gradient accumulation here; the shared
                # normalization then forms sum(numerator) / sum(denominator)
                # across all draft ranks and microbatches.
                (numerator / int(args.accumulation_steps)).backward()
                denominator_sum = (
                    denominator.detach().clone()
                    if denominator_sum is None
                    else denominator_sum + denominator.detach()
                )
                detached = torch.stack([value.detach().float() for value in values])
                metric_sum = detached if metric_sum is None else metric_sum + detached
            accumulated += 1
            if accumulated == int(args.accumulation_steps):
                normalize_accumulated_gradients(
                    optimizer,
                    denominator_sum,
                    int(args.accumulation_steps),
                    group=topology.draft_group,
                )
                optimizer.step()
                accumulated = 0
                denominator_sum = None
            handles[slot] = None
            future = batch_id + int(args.pipeline_depth)
            if future < phase_end:
                handles[slot] = transport.receive(
                    slots[slot],
                    batch_id=future,
                    phase_id=phase.phase_id,
                    schema_id=spec.schema_id,
                )
            global_step = batch_id + 1
            if global_step % int(args.log_interval) == 0:
                metrics = metric_sum / micro_count
                dist.all_reduce(metrics, group=topology.draft_group)
                metrics /= dist.get_world_size(topology.draft_group)
                if dist.get_rank(topology.draft_group) == 0:
                    print(
                        f"disagg {phase.name} step={global_step} "
                        f"loss={float(metrics[0]):.4f}",
                        flush=True,
                    )
                    tracker.log(
                        {f"{phase.name}/loss": float(metrics[0])}, step=global_step
                    )
            if global_step % int(args.save_interval) == 0 and not accumulated:
                _save(args, topology, fsdp, draft, optimizer, phase, global_step)
        if accumulated:
            normalize_accumulated_gradients(
                optimizer,
                denominator_sum,
                int(args.accumulation_steps),
                group=topology.draft_group,
            )
            optimizer.step()
            accumulated = 0
            denominator_sum = None
        batch_id = phase_end
        if not (mode == "teacher" and args.no_final_save):
            _save(
                args,
                topology,
                fsdp,
                draft,
                optimizer,
                phase,
                batch_id,
                name=("final" if phase is phases[-1] else f"{phase.name}/final"),
                serial_head_inherited=(mode == "two_stage"),
            )
        if args.profile and dist.get_rank(topology.draft_group) == 0:
            print(
                f"disagg draft bridge phase={phase.name} ms={bridge_ms:.3f}",
                flush=True,
            )
        dist.barrier()
    tracker.close()


def _phases(args, mode: str):
    if mode == "teacher":
        return [Phase("teacher", 1, int(args.num_epochs), True, False)]
    if mode == "sft":
        return [Phase("sft", 1, int(args.num_epochs), False, True)]
    return [
        Phase("stage1", 1, int(args.stage1_epochs), True, False),
        Phase("transition", 2, 1, True, False),
        Phase("stage2", 3, int(args.stage2_epochs), False, True),
    ]


def run_disaggregated(args, *, mode: str) -> None:
    if os.environ.get("SPECFORGE_DISAGG_DEBUG") == "1":
        # A distributed hang can otherwise hide the rank and collective that
        # stopped making progress.  SIGUSR1 is diagnostic-only and leaves the
        # process running after dumping every Python thread.
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    topology = init_disaggregated(
        timeout=args.dist_timeout,
        target_ranks_per_node=args.target_ranks_per_node,
        draft_ranks_per_node=args.draft_ranks_per_node,
        target_tp_size=args.target_tp_size,
        target_ep_size=args.sglang_ep_size,
    )
    _debug(topology, "topology initialized")
    if topology.rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.cache_dir, exist_ok=True)
    dist.barrier()
    common_state = _load_common_state(args.resume_from)
    identity = topology_identity(args, topology)
    validate_resume_topology(common_state, identity)
    start_id = int(common_state.get("disagg_batch_id_next", 0)) if common_state else 0
    if common_state:
        expected_target = os.path.realpath(args.target_model_path)
        saved_target = common_state.get("target_model_path")
        if (
            saved_target is not None
            and os.path.realpath(saved_target) != expected_target
        ):
            raise ValueError(
                "Disaggregated resume target model mismatch: "
                f"checkpoint={saved_target!r}, current={expected_target!r}"
            )
        expected_data = training_data_identity(args)
        if common_state.get("train_data_identity", expected_data) != expected_data:
            raise ValueError("training dataset must match the resumed checkpoint")

    local_metadata = None
    initialized_drafts = None
    if topology.is_draft:
        # Build once here to publish the exact checkpoint/fresh-model layer contract.
        draft, teacher, resume_stage = _initialize_drafts(args, mode, common_state)
        validate_resume_model(common_state, draft)
        initialized_drafts = (draft, teacher, resume_stage)
        local_metadata = _draft_metadata(draft, teacher)
    metadata = _broadcast_metadata(topology, local_metadata)
    _debug(topology, "draft metadata broadcast")
    args.block_size = metadata["block_size"]
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    loader, steps = _build_target_loader(args, tokenizer, topology)
    _debug(topology, f"dataset ready steps={steps}")
    phases = _phases(args, mode)
    routes = build_node_routes(
        producers=topology.target_replicas_per_node,
        drafts=topology.draft_ranks_per_node,
        node_batch_size=args.node_batch_size,
    )
    if topology.rank == 0:
        print(
            "disaggregate topology: "
            f"target/node={topology.target_ranks_per_node}, "
            f"tp={topology.target_tp_size}, ep={topology.target_ep_size}, "
            f"producers/node={topology.target_replicas_per_node}, "
            f"draft/node={topology.draft_ranks_per_node}, steps/epoch={steps}",
            flush=True,
        )
    completed = False
    try:
        if topology.is_target:
            _run_target(
                args, topology, metadata, phases, loader, steps, routes, start_id
            )
        else:
            _run_draft(
                args,
                topology,
                mode,
                metadata,
                phases,
                steps,
                routes,
                start_id,
                common_state,
                initialized_drafts,
            )
        dist.barrier()
        completed = True
    except BaseException:
        # Print before process-group teardown: destroy_process_group can itself
        # wait for peers that are still blocked in the opposite role.
        traceback.print_exc()
        raise
    finally:
        # A failed rank must exit promptly so torch elastic can terminate peers
        # blocked in the opposite role.  Collective process-group teardown can
        # otherwise hide the original exception until the distributed timeout.
        if completed:
            destroy_distributed()


__all__ = [
    "FixedCollator",
    "Phase",
    "aligned_anchor_count",
    "run_disaggregated",
    "topology_identity",
    "validate_resume_topology",
]
