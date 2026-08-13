"""FlashMTP speculative decode profiling with ``torch.cuda.Event`` (GPU time).

Mirrors the decode loop in ``FlashMTPDraftModel.spec_generate`` / ``spec_generate_with_profile``:
- Target prefill: first ``target(...)`` on the prompt.
- Draft forward: each draft ``self(...)`` in the speculative loop (transformer + lm_head).
- Target verify: each subsequent ``target(...)`` that verifies a draft block.

Supports batched decode (same prompt replicated on batch dim, greedy / temperature≈0).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoModelForCausalLM, DynamicCache

from specforge.modeling.draft.flashmtp import (
    FlashMTPDraftModel,
    gather_pivot_multilayer_inference,
    sample,
)


@dataclass
class FlashMTPProfileStats:
    input_len: int
    batch_size: int
    block_size: int
    num_decode_steps: int
    num_draft_forwards: int
    num_target_verifies: int
    # milliseconds (GPU), single timed run after warmup
    target_prefill_ms: float
    target_verify_total_ms: float
    draft_forward_total_ms: float
    target_verify_avg_ms: float
    draft_forward_avg_ms: float
    decode_target_draft_ratio: float


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _elapsed_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    return float(start.elapsed_time(end))


@torch.inference_mode()
def profile_flashmtp_generate(
    model: FlashMTPDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    stop_token_ids: list[int] | None,
    temperature: float = 0.0,
    batch_size: int = 1,
) -> tuple[SimpleNamespace, FlashMTPProfileStats]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if input_ids.dim() != 2:
        raise ValueError("input_ids must be [batch, seq_len]")
    if input_ids.shape[0] == 1 and batch_size > 1:
        input_ids = input_ids.expand(batch_size, -1).contiguous()
    elif input_ids.shape[0] != batch_size:
        raise ValueError(
            f"input_ids batch {input_ids.shape[0]} does not match batch_size={batch_size}"
        )
    if batch_size > 1 and temperature >= 1e-5:
        raise ValueError(
            "batch_size>1 requires temperature≈0 for synchronized streams."
        )

    device = target.device
    bsz = int(input_ids.shape[0])
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    block_size = model.block_size
    proposal_length = model.proposal_length

    output_ids = torch.full(
        (bsz, max_length + proposal_length + 1),
        mask_token_id,
        dtype=torch.long,
        device=device,
    )
    position_ids = (
        torch.arange(output_ids.shape[1], device=device).unsqueeze(0).expand(bsz, -1)
    )
    past_key_values_target = DynamicCache()

    ev_tp0 = torch.cuda.Event(enable_timing=True)
    ev_tp1 = torch.cuda.Event(enable_timing=True)
    ev_tv0 = torch.cuda.Event(enable_timing=True)
    ev_tv1 = torch.cuda.Event(enable_timing=True)
    ev_df0 = torch.cuda.Event(enable_timing=True)
    ev_df1 = torch.cuda.Event(enable_timing=True)

    _sync(device)
    ev_tp0.record()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )
    ev_tp1.record()

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
        output.logits, temperature
    )
    target_hidden = gather_pivot_multilayer_inference(
        output.hidden_states,
        model.target_layer_ids,
        -1,
        model.config.num_target_layers,
        include_embedding_chs=model.include_embedding_chs,
    )

    start = num_input_tokens
    decode_steps = 0
    draft_forward_times_ms: list[float] = []
    target_verify_times_ms: list[float] = []

    while start < max_length:
        draft_input_ids = output_ids[:, start : start + block_size].clone()
        draft_target_pos = position_ids[:, start : start + block_size]
        if model.local_position:
            draft_block_pos = (
                torch.arange(1, block_size + 1, device=device, dtype=torch.long)
                .unsqueeze(0)
                .expand(bsz, -1)
            )
        else:
            draft_block_pos = draft_target_pos

        noise_embedding = target.model.embed_tokens(draft_input_ids)
        chs = model.chs_len_per_block
        if model.local_position:
            ctx_pos_part = torch.zeros(bsz, chs, dtype=torch.long, device=device)
        else:
            ctx_pos_part = torch.full(
                (bsz, chs), start - 1, dtype=torch.long, device=device
            )
        full_rotary = torch.cat([ctx_pos_part, draft_block_pos], dim=-1)

        _sync(device)
        ev_df0.record()
        block_hidden = model(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=draft_block_pos,
            rotary_position_ids=full_rotary,
            past_key_values=None,
            use_cache=False,
            is_causal=False,
        )
        draft_hidden = model._prediction_hidden(block_hidden)
        lm_head = target.lm_head
        sampled_draft_tokens, _ = model.sample_draft_tokens(
            draft_hidden=draft_hidden,
            lm_head=lm_head,
            first_prev_token_ids=draft_input_ids[:, 0],
            temperature=temperature,
        )
        ev_df1.record()
        ev_df1.synchronize()
        draft_forward_times_ms.append(_elapsed_ms(ev_df0, ev_df1))

        verify_output_ids = torch.cat(
            [draft_input_ids[:, :1], sampled_draft_tokens], dim=1
        )
        verify_position_ids = position_ids[:, start : start + proposal_length + 1]

        _sync(device)
        ev_tv0.record()
        output = target(
            verify_output_ids,
            position_ids=verify_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )
        ev_tv1.record()
        ev_tv1.synchronize()
        target_verify_times_ms.append(_elapsed_ms(ev_tv0, ev_tv1))

        posterior = sample(output.logits, temperature)
        acceptance_lengths_per_row = (
            (verify_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)
        )
        if not bool(
            torch.all(acceptance_lengths_per_row == acceptance_lengths_per_row[0])
        ):
            raise RuntimeError(
                "Per-row acceptance lengths differ under batched decode."
            )
        acceptance_length = int(acceptance_lengths_per_row[0].item())
        output_ids[:, start : start + acceptance_length + 1] = verify_output_ids[
            :, : acceptance_length + 1
        ]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        start += acceptance_length + 1
        past_key_values_target.crop(start)
        pivot_index = min(acceptance_length, output.hidden_states[0].shape[1] - 1)
        target_hidden = gather_pivot_multilayer_inference(
            output.hidden_states,
            model.target_layer_ids,
            pivot_index,
            model.config.num_target_layers,
            include_embedding_chs=model.include_embedding_chs,
        )
        decode_steps += 1

        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:]
            for stop_token_id in stop_token_ids
        ):
            break

    ev_tp1.synchronize()
    target_prefill_ms = _elapsed_ms(ev_tp0, ev_tp1)

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids is not None:
        stop_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = torch.isin(
            output_ids[0][num_input_tokens:], stop_tensor
        ).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    tv_sum = float(sum(target_verify_times_ms))
    df_sum = float(sum(draft_forward_times_ms))
    n_tv = len(target_verify_times_ms)
    n_df = len(draft_forward_times_ms)
    ratio = (tv_sum / df_sum) if df_sum > 1e-9 else float("inf")

    stats = FlashMTPProfileStats(
        input_len=int(num_input_tokens),
        batch_size=bsz,
        block_size=int(block_size),
        num_decode_steps=decode_steps,
        num_draft_forwards=n_df,
        num_target_verifies=n_tv,
        target_prefill_ms=float(target_prefill_ms),
        target_verify_total_ms=tv_sum,
        draft_forward_total_ms=df_sum,
        target_verify_avg_ms=(tv_sum / n_tv) if n_tv else 0.0,
        draft_forward_avg_ms=(df_sum / n_df) if n_df else 0.0,
        decode_target_draft_ratio=ratio,
    )

    run = SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        batch_size=bsz,
    )
    return run, stats


def stats_to_jsonable(stats: FlashMTPProfileStats) -> dict[str, Any]:
    d = asdict(stats)
    if d["decode_target_draft_ratio"] == float("inf"):
        d["decode_target_draft_ratio"] = None
    return d
