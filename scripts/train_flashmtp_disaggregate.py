#!/usr/bin/env python3
"""Node-local target production overlapped with global draft training."""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import time
from typing import Optional

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoConfig, AutoTokenizer

from specforge.args import SGLangBackendArgs
from specforge.checkpoint import (
    load_distributed_training_state,
    save_distributed_training_state,
)
from specforge.core.flashmtp import (
    OnlineFlashMTPModel,
    prepare_target_anchor_hidden,
    prepare_target_hidden,
    prepare_target_prediction_hidden,
)
from specforge.data import build_eagle3_dataset
from specforge.disaggregate import (
    DraftBatchPacket,
    DraftPacketSpec,
    NodePacketTransport,
    build_node_routes,
)
from specforge.distributed import destroy_distributed, init_disaggregated
from specforge.modeling.draft.flashmtp import (
    build_ablation_target_layer_ids,
    flashmtp_draft_class_from_config,
    is_gemma4_config,
    load_flashmtp_draft_model,
)
from specforge.modeling.target.flashmtp_target_model import get_flashmtp_target_model
from specforge.modeling.target.target_utils import (
    TargetEmbeddingsAndHead,
    load_model_text_config,
)
from specforge.optimizer import BF16Optimizer
from specforge.tracker import create_tracker
from specforge.utils import get_last_checkpoint

try:
    from scripts.train_flashmtp import (
        _ensure_embed_vocab_for_mask,
        _sync_config_layer_types_to_draft_depth,
    )
except ModuleNotFoundError:
    from train_flashmtp import (
        _ensure_embed_vocab_for_mask,
        _sync_config_layer_types_to_draft_depth,
    )


def _log(topology, message: str, *, draft_only: bool = False) -> None:
    if draft_only:
        emit = topology.is_draft and dist.get_rank(topology.draft_group) == 0
    else:
        emit = topology.rank == 0
    if emit:
        print(message, flush=True)


def _validate_args(args) -> None:
    required = {
        "--target-ranks-per-node": args.target_ranks_per_node,
        "--draft-ranks-per-node": args.draft_ranks_per_node,
        "--node-batch-size": args.node_batch_size,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"disaggregate mode requires {', '.join(missing)}")
    args.target_tp_size = args.target_tp_size or args.tp_size
    if args.pipeline_depth < 2:
        raise ValueError("--pipeline-depth must be at least 2 for overlap.")
    producers = args.target_ranks_per_node // args.target_tp_size
    if args.target_ranks_per_node % args.target_tp_size:
        raise ValueError("target ranks per node must be divisible by target TP size.")
    if args.node_batch_size % producers:
        raise ValueError("node batch size must be divisible by target producers.")
    if args.node_batch_size % args.draft_ranks_per_node:
        raise ValueError("node batch size must be divisible by draft ranks.")
    if args.target_tp_size > 1 and args.target_model_backend != "sglang":
        raise ValueError("disaggregated target TP > 1 requires the SGLang backend.")
    if args.temp_rollout and args.target_model_backend != "sglang":
        raise ValueError("temp-rollout requires the SGLang backend.")
    if args.draft_micro_batch_size is not None:
        local = args.node_batch_size // args.draft_ranks_per_node
        if local % args.draft_micro_batch_size:
            raise ValueError(
                f"draft local batch {local} must be divisible by draft micro batch "
                f"{args.draft_micro_batch_size}."
            )
    args.target_batch_size = args.node_batch_size // producers


def _make_draft_config(args):
    target_config = load_model_text_config(args.target_model_path)
    if args.draft_config_path:
        config = AutoConfig.from_pretrained(args.draft_config_path)
        config = getattr(config, "text_config", config)
    else:
        config = load_model_text_config(args.target_model_path)
    config.num_hidden_layers = args.num_draft_layers
    config.block_size = args.block_size
    config.num_target_layers = target_config.num_hidden_layers
    if not hasattr(config, "flashmtp_config") or config.flashmtp_config is None:
        config.flashmtp_config = {}
    flash = config.flashmtp_config
    flash["chs_concat_mode"] = "feature"
    flash["pivot_fuse_mode"] = args.pivot_fuse_mode
    flash["num_middle_layers_n"] = args.num_middle_layers_n
    flash["include_embedding_chs"] = True
    flash["local_position"] = bool(args.local_position)
    flash["left_shift"] = bool(args.left_shift)
    flash["markov_head_type"] = args.markov_head_type
    flash["markov_output_mode"] = args.markov_output_mode
    flash["markov_rank"] = int(args.markov_rank)
    if flash.get("target_layer_ids") is None:
        flash["target_layer_ids"] = build_ablation_target_layer_ids(
            config.num_target_layers, args.num_middle_layers_n
        )
    config._attn_implementation = args.attention_backend
    if is_gemma4_config(config):
        config.layer_types = ["full_attention"] * args.num_draft_layers
    else:
        _sync_config_layer_types_to_draft_depth(config)
    draft_cls = flashmtp_draft_class_from_config(config)
    config.architectures = [draft_cls.__name__]
    return config, draft_cls


class _FixedCollator:
    def __init__(self, max_length: int):
        self.max_length = max_length

    def _pad(self, tensor: torch.Tensor, *, dtype=None) -> torch.Tensor:
        tensor = tensor.view(-1)[: self.max_length]
        output = torch.zeros(self.max_length, dtype=dtype or tensor.dtype)
        output[: tensor.numel()].copy_(tensor.to(output.dtype))
        return output

    def __call__(self, features):
        return {
            "input_ids": torch.stack([self._pad(x["input_ids"]) for x in features]),
            "attention_mask": torch.stack(
                [self._pad(x["attention_mask"]) for x in features]
            ),
            "loss_mask": torch.stack(
                [self._pad(x["loss_mask"], dtype=torch.float32) for x in features]
            ),
        }


def _build_target_dataset(args, tokenizer, topology):
    """Build/cache once globally; only target TP leaders retain a DataLoader."""
    cache_params = (
        f"{args.train_data_path}-{args.max_length}-{args.chat_template}-"
        f"{args.target_model_path}"
    )
    cache_key = hashlib.md5(cache_params.encode()).hexdigest()
    build_kwargs = dict(
        dataset=load_dataset("json", data_files=args.train_data_path)["train"]
        if topology.rank == 0
        else None,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        is_preformatted=args.is_preformatted,
        cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
        cache_key=cache_key,
        num_proc=args.build_dataset_num_proc,
    )
    dataset = None
    if topology.rank == 0:
        dataset = build_eagle3_dataset(**build_kwargs)
        dataset = dataset.filter(
            lambda x: x["loss_mask"].sum() >= 2 * args.block_size
        )
    dist.barrier()
    if topology.is_target_leader and topology.rank != 0:
        raw = load_dataset("json", data_files=args.train_data_path)["train"]
        build_kwargs["dataset"] = raw
        dataset = build_eagle3_dataset(**build_kwargs)
        dataset = dataset.filter(
            lambda x: x["loss_mask"].sum() >= 2 * args.block_size
        )

    size = torch.tensor(
        [len(dataset) if topology.rank == 0 else 0], dtype=torch.long, device="cuda"
    )
    dist.broadcast(size, src=0)
    dataset_size = int(size.item())
    producers_total = topology.nnodes * topology.target_replicas_per_node
    target_local_batch = args.node_batch_size // topology.target_replicas_per_node
    steps_per_epoch = dataset_size // (producers_total * target_local_batch)
    if steps_per_epoch <= 0:
        raise ValueError(
            f"dataset size {dataset_size} is smaller than global producer batch "
            f"{producers_total * target_local_batch}."
        )
    loader = None
    if topology.is_target_leader:
        producer_global_rank = (
            topology.node_rank * topology.target_replicas_per_node
            + topology.target_replica_local_rank
        )
        sampler = DistributedSampler(
            dataset,
            num_replicas=producers_total,
            rank=producer_global_rank,
            shuffle=True,
            drop_last=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=target_local_batch,
            sampler=sampler,
            num_workers=args.dataloader_num_workers,
            collate_fn=_FixedCollator(args.max_length),
            drop_last=True,
            prefetch_factor=2 if args.dataloader_num_workers else None,
        )
    return loader, steps_per_epoch


def _broadcast_target_input(args, topology, data):
    batch = args.node_batch_size // topology.target_replicas_per_node
    shape = (batch, args.max_length)
    if topology.is_target_leader:
        input_ids = data["input_ids"].cuda(non_blocking=True)
        attention_mask = data["attention_mask"].cuda(non_blocking=True)
        loss_mask = data["loss_mask"].cuda(non_blocking=True)
    else:
        input_ids = torch.empty(shape, dtype=torch.long, device="cuda")
        attention_mask = torch.empty(shape, dtype=torch.long, device="cuda")
        loss_mask = torch.empty(shape, dtype=torch.float32, device="cuda")
    source = topology.target_tp_leader_global_rank
    for tensor in (input_ids, attention_mask, loss_mask):
        dist.broadcast(tensor, src=source, group=topology.target_tp_group)
    return input_ids, attention_mask, loss_mask


def _sample_fixed_anchors(args, topology, loss_mask, batch_id: int):
    bsz, seq_len = loss_mask.shape
    n = args.num_anchors
    if topology.is_target_leader:
        max_anchor = max(seq_len - (args.block_size - 1) - 1, 0)
        valid = loss_mask[:, : max_anchor + 1] > 0.5
        if args.left_shift:
            valid = valid & (loss_mask[:, 1 : max_anchor + 2] > 0.5)
        counts = valid.sum(dim=1)
        if int(counts.max().item()) <= 1:
            raise ValueError("batch has no valid FlashMTP anchors.")
        generator = torch.Generator(device=loss_mask.device)
        producer = int(topology.target_replica_local_rank)
        generator.manual_seed(
            int(args.seed)
            + 1_000_003 * int(batch_id)
            + 10_007 * int(topology.node_rank)
            + 101 * producer
        )
        random_values = torch.rand(
            valid.shape, device=loss_mask.device, generator=generator
        )
        random_values.masked_fill_(~valid, 2.0)
        sorted_indices = random_values.argsort(dim=1)
        take = min(n, sorted_indices.size(1))
        anchors = torch.zeros((bsz, n), dtype=torch.long, device=loss_mask.device)
        anchors[:, :take] = sorted_indices[:, :take].sort(dim=1).values
        keep = torch.arange(n, device=loss_mask.device).unsqueeze(0) < counts.clamp(max=n).unsqueeze(1)
        anchors.masked_fill_(~keep, 0)
    else:
        anchors = torch.empty((bsz, n), dtype=torch.long, device=loss_mask.device)
        keep = torch.empty((bsz, n), dtype=torch.bool, device=loss_mask.device)
    source = topology.target_tp_leader_global_rank
    dist.broadcast(anchors, src=source, group=topology.target_tp_group)
    dist.broadcast(keep, src=source, group=topology.target_tp_group)
    return anchors, keep


@torch.no_grad()
def _produce_packet(
    args,
    topology,
    target_model,
    target_lm_head,
    target_vocab_size,
    eos_token_id,
    target_layer_ids,
    num_target_layers,
    input_ids,
    attention_mask,
    loss_mask,
    batch_id,
):
    handle = None
    if args.temp_rollout:
        output = target_model.temp_rollout_prefill(input_ids, attention_mask)
        hidden_states = output.hidden_states
        handle = output.handle
    else:
        output = target_model.generate_flashmtp_data(
            input_ids, attention_mask, loss_mask
        )
        hidden_states = output.hidden_states
    anchors, keep = _sample_fixed_anchors(args, topology, loss_mask, batch_id)
    if not topology.is_target_leader:
        if handle is not None:
            # Non-leaders still participate in every branch extend below.
            pass
        target_hidden = None
    else:
        target_hidden = prepare_target_hidden(
            hidden_states, anchors, target_layer_ids, num_target_layers
        ).to(torch.bfloat16)

    need_teacher_hidden = args.tv_loss_weight != 0 and args.markov_head_type != "none"
    target_prediction_hidden = None
    rollout_ids = None
    rollout_validity = None
    if args.temp_rollout:
        if topology.is_target_leader:
            current_hidden = prepare_target_anchor_hidden(
                hidden_states, anchors, num_target_layers
            )
            rollout_ids = torch.zeros(
                (*anchors.shape, args.block_size - 1),
                dtype=torch.long,
                device="cuda",
            )
            rollout_validity = torch.zeros_like(rollout_ids, dtype=torch.bool)
            alive = keep.clone()
            predecessors = [] if need_teacher_hidden else None
        else:
            current_hidden = None
            rollout_ids = torch.zeros(
                (*anchors.shape, args.block_size - 1),
                dtype=torch.long,
                device="cuda",
            )
            alive = keep.clone()
            predecessors = None
        try:
            for position in range(args.block_size - 1):
                if topology.is_target_leader:
                    if predecessors is not None:
                        predecessors.append(current_hidden.unsqueeze(2))
                    rollout_validity[..., position] = alive
                    flat_hidden = current_hidden.reshape(-1, current_hidden.size(-1))
                    flat_alive = alive.reshape(-1)
                    flat_tokens = torch.zeros(
                        flat_hidden.size(0), dtype=torch.long, device="cuda"
                    )
                    active = flat_alive.nonzero(as_tuple=False).flatten()
                    projection_chunk_size = (
                        max(active.numel(), 1)
                        if args.temp_rollout_projection_chunk_size == 0
                        else args.temp_rollout_projection_chunk_size
                    )
                    for start in range(0, active.numel(), projection_chunk_size):
                        selected = active[
                            start : start + projection_chunk_size
                        ]
                        logits = target_lm_head(flat_hidden.index_select(0, selected))
                        flat_tokens[selected] = logits[..., :target_vocab_size].argmax(-1)
                    step_tokens = flat_tokens.view_as(alive)
                else:
                    step_tokens = torch.empty_like(alive, dtype=torch.long)
                dist.broadcast(
                    step_tokens,
                    src=topology.target_tp_leader_global_rank,
                    group=topology.target_tp_group,
                )
                rollout_ids[..., position] = step_tokens
                next_alive = alive
                if eos_token_id is not None:
                    next_alive = alive & step_tokens.ne(eos_token_id)
                if position + 1 < args.block_size - 1:
                    next_hidden = handle.extend_step(
                        anchors, rollout_ids[..., : position + 1], next_alive
                    )
                    if topology.is_target_leader:
                        current_hidden = next_hidden
                alive = next_alive
            if topology.is_target_leader and predecessors is not None:
                target_prediction_hidden = torch.cat(predecessors, dim=2).to(
                    torch.bfloat16
                )
        finally:
            handle.close()
    elif topology.is_target_leader and need_teacher_hidden:
        target_prediction_hidden = prepare_target_prediction_hidden(
            hidden_states,
            anchors,
            args.block_size,
            num_target_layers,
            left_shift=args.left_shift,
        ).to(torch.bfloat16)

    if not topology.is_target_leader:
        return None
    return DraftBatchPacket(
        input_ids=input_ids,
        loss_mask=loss_mask,
        anchor_positions=anchors,
        block_keep_mask=keep,
        target_hidden=target_hidden,
        target_prediction_hidden=target_prediction_hidden,
        rollout_ids=rollout_ids if args.temp_rollout else None,
        rollout_validity=rollout_validity if args.temp_rollout else None,
    )


def _save_draft_checkpoint(args, topology, epoch, step, model, draft, optimizer):
    save_dir = os.path.join(args.output_dir, f"epoch_{epoch}_step_{step}")
    draft_rank = dist.get_rank(topology.draft_group)
    if draft_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
    dist.barrier(group=topology.draft_group)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        state = model.state_dict()
        draft_state = {
            key.replace("draft_model.", ""): value
            for key, value in state.items()
            if "draft_model." in key
        }
        save_distributed_training_state(
            save_dir,
            {
                "epoch": epoch,
                "global_step": step,
                "args": args,
                **optimizer.state_dict(),
            },
            process_group=topology.draft_group,
        )
        if draft_rank == 0:
            draft.save_pretrained(save_dir, state_dict=draft_state)
            for filename in ("flashmtp.py", "flashmtp_markov_head.py"):
                source = os.path.join(
                    os.path.dirname(__file__), "..", "specforge", "modeling", "draft", filename
                )
                if os.path.exists(source):
                    shutil.copy(source, os.path.join(save_dir, filename))
    dist.barrier(group=topology.draft_group)


def _run_target(
    args,
    topology,
    config,
    target_layer_ids,
    tokenizer,
    loader,
    steps_per_epoch,
    routes,
    start_batch_id,
):
    kwargs = (
        SGLangBackendArgs.from_args(args).to_kwargs()
        if args.target_model_backend == "sglang"
        else {}
    )
    target = get_flashmtp_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend=args.target_model_backend,
        torch_dtype=torch.bfloat16,
        device="cuda" if args.target_model_backend == "hf" else None,
        trust_remote_code=args.trust_remote_code,
        **kwargs,
    )
    capture = list(target_layer_ids)
    need_final = args.temp_rollout or (
        args.tv_loss_weight != 0 and args.markov_head_type != "none"
    )
    final_layer = config.num_target_layers - 1
    if need_final and final_layer not in capture:
        capture.append(final_layer)
    target.set_capture_layers(capture)

    if args.temp_rollout:
        local_batch = args.node_batch_size // topology.target_replicas_per_node
        required_requests = local_batch * (args.num_anchors + 1)
        request_capacity = int(target.model_runner.req_to_token_pool.size)
        if request_capacity < required_requests:
            raise ValueError(
                f"SGLang request capacity {request_capacity} is below disaggregated "
                f"temp-rollout requirement {required_requests}."
            )
        recommended_kv = local_batch * args.max_length + local_batch * args.num_anchors * (
            args.block_size - 1
        )
        kv_capacity = int(target.model_runner.token_to_kv_pool_allocator.size)
        if topology.rank == 0:
            print(
                f"temp-rollout capacity: requests={request_capacity}/"
                f"{required_requests}, KV={kv_capacity}/{recommended_kv}",
                flush=True,
            )

    target_components = None
    if args.temp_rollout and topology.is_target_leader:
        target_components = TargetEmbeddingsAndHead.from_pretrained(
            args.target_model_path,
            device="cuda",
            trust_remote_code=args.trust_remote_code,
        )
    target_vocab_size = int(load_model_text_config(args.target_model_path).vocab_size)
    transport = (
        NodePacketTransport(
            topology=topology,
            routes=routes,
            profile=args.profile,
        )
        if topology.is_target_leader
        else None
    )
    send_slots = [None] * args.pipeline_depth
    timing_produce_events = []
    timing_produce_wall_s = 0.0
    timing_send_wait_s = 0.0
    timing_send_comm_ms = 0.0
    timing_send_count = 0
    timing_interval_start = time.perf_counter() if args.profile else None
    batch_id = start_batch_id
    start_epoch, first_epoch_skip = divmod(start_batch_id, steps_per_epoch)
    for epoch in range(start_epoch, args.num_epochs):
        if topology.is_target_leader:
            loader.sampler.set_epoch(epoch)
            iterator = iter(loader)
            if epoch == start_epoch:
                for _ in range(first_epoch_skip):
                    next(iterator)
        step_begin = first_epoch_skip if epoch == start_epoch else 0
        for _step in range(step_begin, steps_per_epoch):
            slot = batch_id % args.pipeline_depth
            if topology.is_target_leader and send_slots[slot] is not None:
                if args.profile:
                    wait_start = time.perf_counter()
                    timing_send_comm_ms += transport.wait_send(send_slots[slot])
                    timing_send_count += 1
                    timing_send_wait_s += time.perf_counter() - wait_start
                else:
                    transport.wait_send(send_slots[slot])
            data = next(iterator) if topology.is_target_leader else None
            input_ids, attention_mask, loss_mask = _broadcast_target_input(
                args, topology, data
            )
            if args.profile:
                produce_start = torch.cuda.Event(enable_timing=True)
                produce_end = torch.cuda.Event(enable_timing=True)
                produce_wall_start = time.perf_counter()
                produce_start.record()
            packet = _produce_packet(
                args,
                topology,
                target,
                target_components.lm_head if target_components is not None else None,
                target_vocab_size,
                tokenizer.eos_token_id,
                target_layer_ids,
                config.num_target_layers,
                input_ids,
                attention_mask,
                loss_mask,
                batch_id,
            )
            if args.profile:
                produce_end.record()
                timing_produce_events.append((produce_start, produce_end))
                timing_produce_wall_s += time.perf_counter() - produce_wall_start
            if topology.is_target_leader:
                send_slots[slot] = transport.send(packet, batch_id)
            global_step = batch_id + 1
            if args.profile and global_step % args.log_interval == 0:
                # Synchronize only at the existing logging cadence.  This keeps
                # per-step overlap intact while making CUDA event times exact.
                timing_produce_events[-1][1].synchronize()
                window = len(timing_produce_events)
                produce_gpu_ms = sum(
                    start.elapsed_time(end)
                    for start, end in timing_produce_events
                ) / window
                produce_wall_ms = timing_produce_wall_s * 1000.0 / window
                send_wait_ms = timing_send_wait_s * 1000.0 / window
                send_comm_ms = timing_send_comm_ms / max(timing_send_count, 1)
                interval_ms = (
                    (time.perf_counter() - timing_interval_start) * 1000.0 / window
                )
                if topology.is_target_leader:
                    print(
                        f"target timing rank={topology.rank} step={global_step}: "
                        f"produce_gpu={produce_gpu_ms:.2f}ms, "
                        f"produce_wall={produce_wall_ms:.2f}ms, "
                        f"send_comm={send_comm_ms:.2f}ms, "
                        f"send_wait={send_wait_ms:.2f}ms, "
                        f"interval={interval_ms:.2f}ms",
                        flush=True,
                    )
                timing_produce_events.clear()
                timing_produce_wall_s = 0.0
                timing_send_wait_s = 0.0
                timing_send_comm_ms = 0.0
                timing_send_count = 0
                timing_interval_start = time.perf_counter()
            batch_id += 1
    if topology.is_target_leader:
        for handle in send_slots:
            if handle is not None:
                transport.wait_send(handle)


def _run_draft(
    args,
    topology,
    config,
    draft_cls,
    target_layer_ids,
    tokenizer,
    steps_per_epoch,
    routes,
    start_batch_id,
    resume_dir,
    resume_training_state,
):
    draft = draft_cls(config).cuda().to(torch.bfloat16)
    if resume_dir is not None:
        loaded = load_flashmtp_draft_model(
            resume_dir, torch_dtype=torch.bfloat16
        )
        draft.load_state_dict(loaded.state_dict())
        del loaded
    mask_token_id = args.mask_token_id
    if mask_token_id is None:
        mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
        mask_token_id = tokenizer.mask_token_id
    draft.mask_token_id = mask_token_id
    flash = draft.config.flashmtp_config
    flash["mask_token_id"] = mask_token_id
    flash["target_layer_ids"] = draft.target_layer_ids
    flash["temp_rollout"] = bool(args.temp_rollout)

    components = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model_path,
        device="cuda",
        trust_remote_code=args.trust_remote_code,
    )
    target_vocab_size = int(components.lm_head.weight.shape[0])
    _ensure_embed_vocab_for_mask(components, mask_token_id)
    online = OnlineFlashMTPModel(
        draft_model=draft,
        target_lm_head=components.lm_head,
        target_embed_tokens=components.embed_tokens,
        block_size=args.block_size,
        mask_token_id=mask_token_id,
        attention_backend=args.attention_backend,
        num_anchors=args.num_anchors,
        loss_decay_gamma=args.loss_decay_gamma,
        final_ce_weight=args.final_ce_weight,
        tv_loss_weight=args.tv_loss_weight,
        base_lm_ce_weight=args.base_lm_ce_weight,
        base_lm_ce_decay_gamma=args.base_lm_ce_decay_gamma,
        ce_chunk_size=args.ce_chunk_size,
        left_shift=args.left_shift,
        temp_rollout_enabled=args.temp_rollout,
        temp_rollout_projection_chunk_size=args.temp_rollout_projection_chunk_size,
        target_vocab_size=target_vocab_size,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = FSDP(
        online,
        process_group=topology.draft_group,
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    )
    total_optimizer_steps = args.num_epochs * math.ceil(
        steps_per_epoch / args.accumulation_steps
    )
    optimizer = BF16Optimizer(
        draft,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=total_optimizer_steps,
        process_group=topology.draft_group,
    )
    if resume_training_state:
        resume_state = load_distributed_training_state(
            resume_dir,
            map_location="cpu",
            process_group=topology.draft_group,
        )
        if resume_state is None:
            raise RuntimeError(f"missing training state in {resume_dir}")
        optimizer.load_state_dict(
            {
                "optimizer_state_dict": resume_state["optimizer_state_dict"],
                "scheduler_state_dict": resume_state["scheduler_state_dict"],
            },
            load_optimizer=args.resume_optimizer,
        )
    args.tracker_global_rank = topology.draft_global_ranks[0]
    tracker = create_tracker(args, args.output_dir) if dist.get_rank(topology.draft_group) == 0 else None

    local_batch = args.node_batch_size // args.draft_ranks_per_node
    micro_batch = args.draft_micro_batch_size or local_batch
    micro_count = local_batch // micro_batch
    need_teacher_hidden = args.tv_loss_weight != 0 and args.markov_head_type != "none"
    spec = DraftPacketSpec(
        batch_size=local_batch,
        max_length=args.max_length,
        num_anchors=args.num_anchors,
        num_target_layers=len(target_layer_ids),
        hidden_size=config.hidden_size,
        prediction_length=args.block_size - 1,
        include_target_prediction_hidden=need_teacher_hidden,
    )
    slots = [
        DraftBatchPacket.empty(spec, device="cuda", temp_rollout=args.temp_rollout)
        for _ in range(args.pipeline_depth)
    ]
    transport = NodePacketTransport(
        topology=topology,
        routes=routes,
        profile=args.profile,
    )
    total_batches = args.num_epochs * steps_per_epoch
    receive_handles = [None] * args.pipeline_depth
    first_slot = start_batch_id % args.pipeline_depth
    receive_handles[first_slot] = transport.receive(
        slots[first_slot], start_batch_id
    )
    timing_compute_events = []
    timing_compute_wall_s = 0.0
    timing_receive_wait_s = 0.0
    timing_receive_comm_ms = 0.0
    timing_interval_start = time.perf_counter() if args.profile else None
    accumulated = 0
    last_grad_norm = None
    checkpoint_pending = False
    for batch_id in range(start_batch_id, total_batches):
        slot = batch_id % args.pipeline_depth
        if args.profile:
            receive_wait_start = time.perf_counter()
            timing_receive_comm_ms += transport.wait_receive(receive_handles[slot])
            timing_receive_wait_s += time.perf_counter() - receive_wait_start
        else:
            transport.wait_receive(receive_handles[slot])
        # The just-consumed slot can be posted again only after this packet's
        # backward; the next distinct slot is safe to prepost now.
        next_id = batch_id + 1
        if next_id < total_batches:
            next_slot = next_id % args.pipeline_depth
            if receive_handles[next_slot] is None:
                receive_handles[next_slot] = transport.receive(slots[next_slot], next_id)
        packet = slots[slot]
        metric_sums = None
        if args.profile:
            compute_start = torch.cuda.Event(enable_timing=True)
            compute_end = torch.cuda.Event(enable_timing=True)
            compute_wall_start = time.perf_counter()
            compute_start.record()
        for micro in range(micro_count):
            start, end = micro * micro_batch, (micro + 1) * micro_batch
            values = model(
                input_ids=packet.input_ids[start:end],
                loss_mask=packet.loss_mask[start:end],
                anchor_positions=packet.anchor_positions[start:end],
                block_keep_mask=packet.block_keep_mask[start:end],
                target_hidden=packet.target_hidden[start:end],
                target_prediction_hidden=(
                    packet.target_prediction_hidden[start:end]
                    if packet.target_prediction_hidden is not None
                    else None
                ),
                rollout_ids=(
                    packet.rollout_ids[start:end]
                    if packet.rollout_ids is not None
                    else None
                ),
                rollout_validity=(
                    packet.rollout_validity[start:end]
                    if packet.rollout_validity is not None
                    else None
                ),
            )
            loss, *metrics = values
            (loss / (args.accumulation_steps * micro_count)).backward()
            detached = torch.stack([value.detach().float() for value in values])
            metric_sums = detached if metric_sums is None else metric_sums + detached
        accumulated += 1
        if accumulated == args.accumulation_steps:
            last_grad_norm = optimizer.step()
            accumulated = 0
        if args.profile:
            compute_end.record()
            timing_compute_events.append((compute_start, compute_end))
            timing_compute_wall_s += time.perf_counter() - compute_wall_start
        receive_handles[slot] = None
        future_id = batch_id + args.pipeline_depth
        if future_id < total_batches:
            receive_handles[slot] = transport.receive(slots[slot], future_id)

        global_step = batch_id + 1
        checkpoint_pending = (
            checkpoint_pending or global_step % args.save_interval == 0
        )
        if global_step % args.log_interval == 0:
            metrics = metric_sums / micro_count
            dist.all_reduce(metrics, group=topology.draft_group)
            metrics /= dist.get_world_size(topology.draft_group)
            timing = timing_max = None
            if args.profile:
                timing_compute_events[-1][1].synchronize()
                timing_window = len(timing_compute_events)
                timing = torch.tensor(
                    [
                        sum(
                            start.elapsed_time(end)
                            for start, end in timing_compute_events
                        ) / timing_window,
                        timing_compute_wall_s * 1000.0 / timing_window,
                        timing_receive_wait_s * 1000.0 / timing_window,
                        timing_receive_comm_ms / timing_window,
                        (time.perf_counter() - timing_interval_start)
                        * 1000.0
                        / timing_window,
                    ],
                    dtype=torch.float64,
                    device="cuda",
                )
                timing_max = timing.clone()
                dist.all_reduce(
                    timing, op=dist.ReduceOp.SUM, group=topology.draft_group
                )
                timing /= dist.get_world_size(topology.draft_group)
                dist.all_reduce(
                    timing_max, op=dist.ReduceOp.MAX, group=topology.draft_group
                )
            if dist.get_rank(topology.draft_group) == 0:
                names = ("loss", "accuracy", "prefix_acc", "final_ce", "base_ce", "tv")
                message = ", ".join(
                    f"{name}={float(value):.4f}" for name, value in zip(names, metrics)
                )
                learning_rate = optimizer.get_learning_rate()
                progress_pct = 100.0 * global_step / total_batches
                print(
                    f"train step {global_step}/{total_batches} "
                    f"({progress_pct:.2f}%): {message}, lr={learning_rate:.6g}",
                    flush=True,
                )
                if args.profile:
                    print(
                        f"draft timing step={global_step}: "
                        f"compute_gpu={float(timing[0]):.2f}ms "
                        f"(max={float(timing_max[0]):.2f}), "
                        f"compute_wall={float(timing[1]):.2f}ms, "
                        f"receive_wait={float(timing[2]):.2f}ms "
                        f"(max={float(timing_max[2]):.2f}), "
                        f"recv_post_to_done={float(timing[3]):.2f}ms "
                        f"(max={float(timing_max[3]):.2f}), "
                        f"interval={float(timing[4]):.2f}ms",
                        flush=True,
                    )
                if tracker is not None:
                    log_values = {
                        f"train/{name}": float(value)
                        for name, value in zip(names, metrics)
                    }
                    log_values["train/lr"] = float(learning_rate)
                    if args.profile:
                        log_values.update(
                            {
                                "timing/draft_compute_gpu_ms": float(timing[0]),
                                "timing/draft_compute_wall_ms": float(timing[1]),
                                "timing/draft_receive_wait_ms": float(timing[2]),
                                "timing/draft_recv_post_to_done_ms": float(timing[3]),
                                "timing/draft_interval_ms": float(timing[4]),
                            }
                        )
                    tracker.log(log_values, step=global_step)
            if args.profile:
                timing_compute_events.clear()
                timing_compute_wall_s = 0.0
                timing_receive_wait_s = 0.0
                timing_receive_comm_ms = 0.0
                timing_interval_start = time.perf_counter()
        # Match the colocated path: do not carry a short accumulation window
        # across epoch boundaries.
        epoch_boundary = global_step % steps_per_epoch == 0
        if epoch_boundary and accumulated:
            optimizer.scale_model_gradients(
                args.accumulation_steps / accumulated
            )
            last_grad_norm = optimizer.step()
            accumulated = 0
        if checkpoint_pending and accumulated == 0:
            _save_draft_checkpoint(
                args,
                topology,
                batch_id // steps_per_epoch,
                global_step,
                model,
                draft,
                optimizer,
            )
            checkpoint_pending = False
    if accumulated:
        optimizer.scale_model_gradients(args.accumulation_steps / accumulated)
        optimizer.step()
    _save_draft_checkpoint(
        args, topology, args.num_epochs, total_batches, model, draft, optimizer
    )
    if tracker is not None:
        tracker.close()


def run_disaggregated(args) -> None:
    _validate_args(args)
    topology = init_disaggregated(
        timeout=args.dist_timeout,
        target_ranks_per_node=args.target_ranks_per_node,
        draft_ranks_per_node=args.draft_ranks_per_node,
        target_tp_size=args.target_tp_size,
    )
    if topology.rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.cache_dir, exist_ok=True)
    dist.barrier()
    config, draft_cls = _make_draft_config(args)
    target_layer_ids = list(config.flashmtp_config["target_layer_ids"])
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    loader, steps_per_epoch = _build_target_dataset(args, tokenizer, topology)
    routes = build_node_routes(
        producers=topology.target_replicas_per_node,
        drafts=topology.draft_ranks_per_node,
        node_batch_size=args.node_batch_size,
    )
    resume_dir = None
    resume_training_state = False
    start_batch_id = 0
    if args.ckpt_dir:
        resume_dir = args.ckpt_dir
    elif args.resume and os.path.isdir(args.output_dir):
        resume_dir = get_last_checkpoint(args.output_dir)[0]
    if resume_dir is not None:
        common_path = os.path.join(resume_dir, "training_state.pt")
        if os.path.isfile(common_path):
            common_state = torch.load(
                common_path,
                map_location="cpu",
                weights_only=False,
            )
            start_batch_id = int(common_state["global_step"])
            resume_training_state = True
            if start_batch_id >= args.num_epochs * steps_per_epoch:
                raise ValueError(
                    f"checkpoint step {start_batch_id} already reaches configured training end."
                )
    _log(
        topology,
        "disaggregate topology: "
        f"nodes={topology.nnodes}, target/node={topology.target_ranks_per_node}, "
        f"target_tp={topology.target_tp_size}, producers/node="
        f"{topology.target_replicas_per_node}, draft/node="
        f"{topology.draft_ranks_per_node}, node_batch={args.node_batch_size}, "
        f"global_batch={args.node_batch_size * topology.nnodes}, "
        f"steps/epoch={steps_per_epoch}",
    )
    try:
        if topology.is_target:
            _run_target(
                args,
                topology,
                config,
                target_layer_ids,
                tokenizer,
                loader,
                steps_per_epoch,
                routes,
                start_batch_id,
            )
        else:
            _run_draft(
                args,
                topology,
                config,
                draft_cls,
                target_layer_ids,
                tokenizer,
                steps_per_epoch,
                routes,
                start_batch_id,
                resume_dir,
                resume_training_state,
            )
        dist.barrier()
    finally:
        destroy_distributed()
