"""Node-local target/draft disaggregation for DFlash-family training."""

from __future__ import annotations

import math
import os
import traceback
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from transformers import AutoTokenizer

from scripts.dflash_family.dflash_family_training import (
    build_family_draft,
    build_online_model,
    concatenate_target_hidden,
    final_target_hidden,
    family_training_metadata,
    load_family_draft,
    selector_alpha,
    sync_args_from_draft,
)
from scripts.dlite.dlite_disaggregate import (
    _broadcast_input,
    _build_target_loader,
    _load_common_state,
    topology_identity,
    validate_resume_topology,
)
from scripts.dlite.dlite_training import (
    build_target_model,
    normalize_accumulated_gradients,
    resolve_tokenizer_and_components,
    save_checkpoint,
    training_data_identity,
)
from specforge.disaggregate import (
    FamilyBatchPacket,
    FamilyPacketSpec,
    NodePacketTransport,
    build_node_routes,
)
from specforge.distributed import destroy_distributed, init_disaggregated
from specforge.optimizer import BF16Optimizer
from specforge.tracker import create_tracker


def _broadcast_metadata(topology, metadata):
    source = topology.draft_global_ranks[0]
    payload = [metadata if topology.rank == source else None]
    dist.broadcast_object_list(payload, src=source)
    return payload[0]


def _metadata(draft, algorithm, require_last):
    return {
        "algorithm": algorithm,
        "target_layer_ids": [int(value) for value in draft.target_layer_ids],
        "num_target_layers": int(draft.config.num_target_layers),
        "hidden_size": int(draft.config.hidden_size),
        "context_width": len(draft.target_layer_ids) * int(draft.config.hidden_size),
        "block_size": int(draft.block_size),
        "include_target_last_hidden": bool(require_last),
    }


def _packet_spec(args, topology, metadata):
    local_batch = (
        int(args.node_batch_size) // topology.target_replicas_per_node
        if topology.is_target
        else int(args.node_batch_size) // topology.draft_ranks_per_node
    )
    return FamilyPacketSpec(
        batch_size=local_batch,
        max_length=int(args.max_length),
        context_width=int(metadata["context_width"]),
        hidden_size=int(metadata["hidden_size"]),
        include_target_last_hidden=bool(metadata["include_target_last_hidden"]),
    )


def _target_packet(args, topology, target, metadata, values):
    input_ids, attention_mask, loss_mask = values
    output = target.generate_dlite_data(
        input_ids, attention_mask, loss_mask, return_logits=False
    )
    if not topology.is_target_leader:
        return None
    hidden = output.hidden_states
    context = concatenate_target_hidden(hidden, metadata["target_layer_ids"])
    last = (
        final_target_hidden(hidden, metadata["num_target_layers"])
        if metadata["include_target_last_hidden"]
        else None
    )
    return FamilyBatchPacket(
        input_ids=input_ids,
        loss_mask=loss_mask,
        target_context_hidden=context.to(torch.bfloat16),
        target_last_hidden=(last.to(torch.bfloat16) if last is not None else None),
    )


def _run_target(args, topology, metadata, loader, steps, routes, start_id):
    proxy = SimpleNamespace(
        target_layer_ids=metadata["target_layer_ids"], is_teacher=False
    )
    args.require_target_last_hidden = metadata["include_target_last_hidden"]
    target = build_target_model(args, [proxy])
    transport = (
        NodePacketTransport(topology=topology, routes=routes, profile=args.profile)
        if topology.is_target_leader
        else None
    )
    handles = [None] * int(args.pipeline_depth)
    batch_id = 0
    for epoch in range(int(args.num_epochs)):
        if topology.is_target_leader:
            loader.sampler.set_epoch(epoch)
            iterator = iter(loader)
        for _ in range(steps):
            data = next(iterator) if topology.is_target_leader else None
            values = _broadcast_input(args, topology, data)
            packet = _target_packet(args, topology, target, metadata, values)
            if batch_id >= start_id and topology.is_target_leader:
                slot = batch_id % int(args.pipeline_depth)
                if handles[slot] is not None:
                    transport.wait_send(handles[slot])
                handles[slot] = transport.send(
                    packet, batch_id=batch_id, phase_id=1, schema_id=_packet_spec(args, topology, metadata).schema_id
                )
            batch_id += 1
    if topology.is_target_leader:
        for handle in handles:
            if handle is not None:
                transport.wait_send(handle)


def _save(
    args, topology, algorithm, fsdp, draft, optimizer, next_id,
    optimizer_step, total_steps, steps_per_epoch, name
):
    save_checkpoint(
        output_dir=args.output_dir,
        name=name,
        fsdp_model=fsdp,
        draft_model=draft,
        optimizer=optimizer,
        process_group=topology.draft_group,
        coordinator_global_rank=topology.draft_global_ranks[0],
        metadata={
            "training_stage": algorithm,
            "algorithm": algorithm,
            "execution_mode": "disaggregate",
            "disagg_batch_id_next": int(next_id),
            "stage_epoch": int(next_id // max(steps_per_epoch, 1)),
            "next_batch_in_epoch": int(next_id % max(steps_per_epoch, 1)),
            "stage_step": int(next_id),
            "global_step": int(next_id),
            "optimizer_step": int(optimizer_step),
            "scheduler_total_steps": int(total_steps),
            "train_data_identity": training_data_identity(args),
            "target_model_path": os.path.realpath(args.target_model_path),
            **family_training_metadata(
                args, algorithm, optimizer_step, total_steps
            ),
            **topology_identity(args, topology),
        },
    )


def _run_draft(
    args, topology, algorithm, metadata, steps, routes, start_id, state, draft
):
    draft.requires_grad_(True)
    _, components, _ = resolve_tokenizer_and_components(
        args, [draft], standalone_components=True
    )
    online = build_online_model(
        args, algorithm, draft, components, process_group=topology.draft_group
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
    total_batches = int(args.num_epochs) * int(steps)
    total_optimizer_steps = math.ceil(total_batches / int(args.accumulation_steps))
    optimizer = BF16Optimizer(
        draft,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=total_optimizer_steps,
        process_group=topology.draft_group,
    )
    if state:
        from specforge.checkpoint import load_distributed_training_state

        optimizer.load_state_dict(
            load_distributed_training_state(
                args.resume_from, map_location="cpu", process_group=topology.draft_group
            )
        )
    args.tracker_global_rank = topology.draft_global_ranks[0]
    tracker = create_tracker(args, args.output_dir)
    transport = NodePacketTransport(
        topology=topology, routes=routes, profile=args.profile
    )
    spec = _packet_spec(args, topology, metadata)
    slots = [
        FamilyBatchPacket.empty(spec, device="cuda")
        for _ in range(int(args.pipeline_depth))
    ]
    handles = [None] * int(args.pipeline_depth)
    if start_id < total_batches:
        slot = start_id % int(args.pipeline_depth)
        handles[slot] = transport.receive(
            slots[slot], batch_id=start_id, phase_id=1, schema_id=spec.schema_id
        )
    local_batch = int(args.node_batch_size) // int(args.draft_ranks_per_node)
    micro_batch = int(args.draft_micro_batch_size)
    micro_count = local_batch // micro_batch
    accumulated = 0
    denominator_sum = None
    optimizer_step = int(state.get("optimizer_step", 0)) if state else 0
    for batch_id in range(start_id, total_batches):
        slot = batch_id % int(args.pipeline_depth)
        transport.wait_receive(handles[slot])
        next_id = batch_id + 1
        if next_id < total_batches:
            next_slot = next_id % int(args.pipeline_depth)
            if handles[next_slot] is None:
                handles[next_slot] = transport.receive(
                    slots[next_slot],
                    batch_id=next_id,
                    phase_id=1,
                    schema_id=spec.schema_id,
                )
        packet = slots[slot]
        metric_sum = None
        for micro in range(micro_count):
            begin, end = micro * micro_batch, (micro + 1) * micro_batch
            kwargs = dict(
                input_ids=packet.input_ids[begin:end],
                hidden_states=packet.target_context_hidden[begin:end],
                loss_mask=packet.loss_mask[begin:end],
                target_last_hidden_states=(
                    packet.target_last_hidden[begin:end]
                    if packet.target_last_hidden is not None
                    else None
                ),
                collect_detailed_metrics=(next_id % int(args.log_interval) == 0),
            )
            if algorithm == "dflash2":
                kwargs["selector_loss_alpha"] = selector_alpha(
                    args, optimizer_step, total_optimizer_steps
                )
            loss, accuracy, metrics = fsdp(**kwargs)
            numerator, denominator = metrics["loss_terms"]
            (numerator / int(args.accumulation_steps)).backward()
            denominator_sum = (
                denominator.detach().clone()
                if denominator_sum is None
                else denominator_sum + denominator.detach()
            )
            values = torch.stack([loss.detach(), accuracy.detach()])
            metric_sum = values if metric_sum is None else metric_sum + values
        accumulated += 1
        if accumulated == int(args.accumulation_steps):
            normalize_accumulated_gradients(
                optimizer,
                denominator_sum,
                int(args.accumulation_steps),
                group=topology.draft_group,
            )
            optimizer.step()
            optimizer_step += 1
            accumulated = 0
            denominator_sum = None
        handles[slot] = None
        future = batch_id + int(args.pipeline_depth)
        if future < total_batches:
            handles[slot] = transport.receive(
                slots[slot], batch_id=future, phase_id=1, schema_id=spec.schema_id
            )
        if next_id % int(args.log_interval) == 0:
            values = metric_sum / micro_count
            dist.all_reduce(values, group=topology.draft_group)
            values /= dist.get_world_size(topology.draft_group)
            if dist.get_rank(topology.draft_group) == 0:
                print(
                    f"disagg {algorithm} step={next_id} loss={float(values[0]):.4f} acc={float(values[1]):.4f}",
                    flush=True,
                )
                tracker.log(
                    {"train/loss": float(values[0]), "train/accuracy": float(values[1])},
                    step=next_id,
                )
        if next_id % int(args.save_interval) == 0 and not accumulated:
            _save(
                args, topology, algorithm, fsdp, draft, optimizer, next_id,
                optimizer_step, total_optimizer_steps, steps, f"step_{next_id}"
            )
    if accumulated:
        normalize_accumulated_gradients(
            optimizer,
            denominator_sum,
            int(args.accumulation_steps),
            group=topology.draft_group,
        )
        optimizer.step()
        optimizer_step += 1
    _save(
        args, topology, algorithm, fsdp, draft, optimizer, total_batches,
        optimizer_step, total_optimizer_steps, steps, "final"
    )
    tracker.close()


def run_disaggregated(args, algorithm: str) -> None:
    topology = init_disaggregated(
        timeout=args.dist_timeout,
        target_ranks_per_node=args.target_ranks_per_node,
        draft_ranks_per_node=args.draft_ranks_per_node,
        target_tp_size=args.target_tp_size,
        target_ep_size=args.sglang_ep_size,
    )
    if topology.rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.cache_dir, exist_ok=True)
    dist.barrier()
    state = _load_common_state(args.resume_from)
    validate_resume_topology(state, topology_identity(args, topology))
    start_id = int(state.get("disagg_batch_id_next", 0)) if state else 0
    draft = None
    local_metadata = None
    if topology.is_draft:
        source = args.resume_from or args.init_from
        draft = (
            load_family_draft(source, algorithm, args.attention_backend)
            if source
            else build_family_draft(args, algorithm)
        )
        sync_args_from_draft(args, draft)
        local_metadata = _metadata(draft, algorithm, args.require_target_last_hidden)
    metadata = _broadcast_metadata(topology, local_metadata)
    args.block_size = int(metadata["block_size"])
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    loader, steps = _build_target_loader(args, tokenizer, topology)
    total_optimizer_steps = math.ceil(
        int(args.num_epochs) * int(steps) / int(args.accumulation_steps)
    )
    if state:
        expected = {
            "training_stage": algorithm,
            "algorithm": algorithm,
            "execution_mode": "disaggregate",
            "scheduler_total_steps": total_optimizer_steps,
            "train_data_identity": training_data_identity(args),
            "loss_config": family_training_metadata(
                args,
                algorithm,
                int(state.get("optimizer_step", 0)),
                total_optimizer_steps,
            )["loss_config"],
        }
        for key, value in expected.items():
            if state.get(key, value) != value:
                raise ValueError(
                    f"Resume mismatch for {key}: "
                    f"checkpoint={state.get(key)!r}, current={value!r}"
                )
    routes = build_node_routes(
        producers=topology.target_replicas_per_node,
        drafts=topology.draft_ranks_per_node,
        node_batch_size=args.node_batch_size,
    )
    completed = False
    try:
        if topology.is_target:
            _run_target(args, topology, metadata, loader, steps, routes, start_id)
        else:
            _run_draft(
                args, topology, algorithm, metadata, steps, routes, start_id, state, draft
            )
        dist.barrier()
        completed = True
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if completed:
            destroy_distributed()


__all__ = ["run_disaggregated"]
