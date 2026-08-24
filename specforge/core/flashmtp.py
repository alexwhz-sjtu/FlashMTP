# coding=utf-8
"""Training primitives for the current SWA-teacher/PivotQ-student architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from specforge.modeling.draft.flashmtp import FlashMTPDraftModel
from specforge.modeling.draft.flashmtp_markov_head import markov_output_uses_base_lm_head

try:
    from torch.nn.attention.flex_attention import create_block_mask
    from specforge.modeling.draft.flex_attention import compile_friendly_create_block_mask
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    create_block_mask = None
    compile_friendly_create_block_mask = None
    FLEX_ATTENTION_AVAILABLE = False


HiddenStatesInput = Union[
    tuple[torch.Tensor, ...], list[torch.Tensor], Dict[int, torch.Tensor]
]


def infer_hidden_states_embedding_offset(
    hidden_states: tuple | list, num_transformer_layers: int
) -> int:
    if len(hidden_states) == num_transformer_layers:
        return 0
    if len(hidden_states) == num_transformer_layers + 1:
        return 1
    return 1 if len(hidden_states) > num_transformer_layers else 0


def _hidden_at_layer(
    hidden_states: HiddenStatesInput,
    layer_id: int,
    num_transformer_layers: int,
) -> torch.Tensor:
    if isinstance(hidden_states, dict):
        return hidden_states[layer_id]
    offset = infer_hidden_states_embedding_offset(hidden_states, num_transformer_layers)
    return hidden_states[layer_id + offset]


def prepare_target_hidden(
    hidden_states: HiddenStatesInput,
    anchor_positions: torch.Tensor,
    target_layer_ids: list[int],
    num_transformer_layers: int,
) -> torch.Tensor:
    """Gather CHS at ``anchor-1`` as ``(batch, anchors, layers, hidden)``."""
    positions = (anchor_positions - 1).clamp(min=0)
    pieces = []
    for layer_id in target_layer_ids:
        layer_hidden = _hidden_at_layer(hidden_states, layer_id, num_transformer_layers)
        pieces.append(
            torch.gather(
                layer_hidden,
                1,
                positions.unsqueeze(-1).expand(-1, -1, layer_hidden.size(-1)),
            )
        )
    return torch.stack(pieces, dim=2)


def prepare_history_hidden_states(
    hidden_states: HiddenStatesInput,
    history_layer_ids: list[int],
    num_transformer_layers: int,
) -> torch.Tensor:
    """Stack first/middle/last target layers as ``(batch, seq, 3, hidden)``."""
    return torch.stack(
        [
            _hidden_at_layer(hidden_states, layer_id, num_transformer_layers)
            for layer_id in history_layer_ids
        ],
        dim=2,
    )


def gather_token_group(
    input_ids: torch.Tensor,
    anchor_positions: torch.Tensor,
    anchor_group_size: int,
    fill_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather ``a-G+1..a`` with masked left padding."""
    seq_len = input_ids.size(1)
    offsets = torch.arange(
        -int(anchor_group_size) + 1, 1, device=input_ids.device
    ).view(1, 1, -1)
    positions = anchor_positions.unsqueeze(-1) + offsets
    keep = positions >= 0
    safe = positions.clamp(min=0, max=seq_len - 1)
    gathered = torch.gather(
        input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1), 2, safe
    )
    gathered = torch.where(
        keep, gathered, torch.full_like(gathered, int(fill_token_id))
    )
    return gathered, keep, positions.clamp(min=0)


def gather_target_prefill_logits(
    target_logits: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Gather causal logits at ``a..a+B-2`` for labels ``a+1..a+B-1``."""
    offsets = torch.arange(int(block_size) - 1, device=anchor_positions.device).view(
        1, 1, -1
    )
    positions = anchor_positions.unsqueeze(-1) + offsets
    if bool((positions >= target_logits.size(1)).any()):
        raise ValueError("Target prefill logits do not cover all prediction positions.")
    expanded = target_logits.unsqueeze(1).expand(
        -1, anchor_positions.size(1), -1, -1
    )
    return torch.gather(
        expanded,
        2,
        positions.unsqueeze(-1).expand(-1, -1, -1, target_logits.size(-1)),
    )


def _make_flex_mask(
    *,
    model_role: str,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    token_keep_mask: torch.Tensor,
    seq_len: int,
    swa_window_size: int,
    chs_slots: int,
    query_len: int,
    device: torch.device,
):
    if not FLEX_ATTENTION_AVAILABLE:
        raise RuntimeError("flex_attention is not available")
    bsz, num_blocks = anchor_positions.shape
    total_chs = num_blocks * chs_slots
    total_q = num_blocks * query_len
    is_teacher = model_role == "swa_teacher"
    context_prefix = int(seq_len) if is_teacher else 0
    kv_len = context_prefix + total_chs + total_q
    chs_start = context_prefix
    qkv_start = context_prefix + total_chs
    token_count = token_keep_mask.size(-1)
    max_block = max(num_blocks - 1, 0)

    def mask_mod(b, h, q_idx, kv_idx):
        q_block = q_idx // query_len
        safe_q_block = q_block.clamp(min=0, max=max_block)
        block_valid = block_keep_mask[b, safe_q_block] & (q_block <= max_block)
        is_history = kv_idx < context_prefix
        anchor = anchor_positions[b, safe_q_block]
        history_valid = (
            is_history
            & (kv_idx >= anchor - int(swa_window_size))
            & (kv_idx <= anchor - 2)
        )
        is_chs = (kv_idx >= chs_start) & (kv_idx < qkv_start)
        chs_block = (kv_idx - chs_start) // chs_slots
        chs_valid = is_chs & (chs_block == q_block)
        is_query_kv = kv_idx >= qkv_start
        query_kv_block = (kv_idx - qkv_start) // query_len
        query_slot = (kv_idx - qkv_start) % query_len
        query_kv_valid = is_query_kv & (query_kv_block == q_block)
        safe_query_block = query_kv_block.clamp(min=0, max=max_block)
        safe_query_slot = query_slot.clamp(min=0, max=query_len - 1)
        padded_token = (safe_query_slot < token_count) & ~token_keep_mask[
            b,
            safe_query_block,
            safe_query_slot.clamp(max=token_count - 1),
        ]
        valid_attention = block_valid & (
            history_valid | chs_valid | (query_kv_valid & ~padded_token)
        )
        # Padded blocks have zero loss, but flex attention still evaluates their
        # query rows. Keep their own finite CHS slots visible so those rows are
        # never entirely masked before the loss mask is applied.
        invalid_fallback = chs_valid & ~block_valid & (q_block <= max_block)
        return valid_attention | invalid_fallback

    mask_mod.__name__ = f"flashmtp_{model_role}_n{num_blocks}_q{query_len}_s{chs_slots}"
    create_fn = compile_friendly_create_block_mask or create_block_mask
    return create_fn(
        mask_mod,
        B=bsz,
        H=None,
        Q_LEN=total_q,
        KV_LEN=kv_len,
        device=device,
    )


@dataclass
class PreparedFlashMTPBatch:
    anchor_positions: torch.Tensor
    block_keep_mask: torch.Tensor
    target_hidden: torch.Tensor
    shared_fused_history: Optional[torch.Tensor]
    query_embeddings: torch.Tensor
    token_keep_mask: torch.Tensor
    token_position_ids: torch.Tensor
    labels: torch.Tensor
    prev_token_ids: torch.Tensor
    raw_weight_mask: torch.Tensor
    binary_eval_mask: torch.Tensor
    initial_prev_token_ids: Optional[torch.Tensor]


@dataclass
class FlashMTPLossOutput:
    loss: torch.Tensor
    accuracy: torch.Tensor
    prefix_acc: torch.Tensor
    final_ce_loss: torch.Tensor
    base_ce_loss: torch.Tensor
    tv_loss: torch.Tensor

    def as_tuple(self):
        return (
            self.loss,
            self.accuracy,
            self.prefix_acc,
            self.final_ce_loss,
            self.base_ce_loss,
            self.tv_loss,
        )


class OnlineFlashMTPModel(nn.Module):
    """One-pass backbone/logits/loss wrapper for the current architecture."""

    def __init__(
        self,
        draft_model: FlashMTPDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = None,
        final_ce_weight: float = 1.0,
        tv_loss_weight: float = 1.0,
        base_lm_ce_weight: float = 0.0,
        base_lm_ce_decay_gamma: Optional[float] = None,
        markov_teacher_forcing_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        if attention_backend != "flex_attention":
            raise ValueError("The current architecture supports flex_attention only.")
        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.mask_token_id = int(mask_token_id)
        self.block_size = int(block_size)
        if self.block_size <= 1:
            raise ValueError(
                f"block_size must be at least 2 for next-token loss, got {block_size}."
            )
        self.attention_backend = attention_backend
        self.num_anchors = int(num_anchors)
        self.loss_decay_gamma = loss_decay_gamma
        self.final_ce_weight = float(final_ce_weight)
        self.tv_loss_weight = float(tv_loss_weight)
        self.base_lm_ce_weight = float(base_lm_ce_weight)
        for name, value in (
            ("final_ce_weight", self.final_ce_weight),
            ("tv_loss_weight", self.tv_loss_weight),
            ("base_lm_ce_weight", self.base_lm_ce_weight),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}.")
        if self.final_ce_weight + self.tv_loss_weight + self.base_lm_ce_weight == 0:
            raise ValueError("At least one FlashMTP loss weight must be positive.")
        self.base_lm_ce_decay_gamma = base_lm_ce_decay_gamma
        self.markov_teacher_forcing_ratio = float(markov_teacher_forcing_ratio)
        if not 0.0 <= self.markov_teacher_forcing_ratio <= 1.0:
            raise ValueError("markov_teacher_forcing_ratio must be in [0, 1].")

    def sample_anchor_positions(
        self, seq_len: int, loss_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_anchor = max(seq_len - self.block_size, 0)
        valid = loss_mask[:, : max_anchor + 1] > 0.5
        valid = valid & (loss_mask[:, 1 : max_anchor + 2] > 0.5)
        if valid.size(1):
            valid[:, 0] = False
        counts = valid.sum(dim=1)
        num = min(self.num_anchors, int(counts.max().item()))
        if num <= 0:
            raise ValueError("No valid anchor positions in this batch.")
        random_values = torch.rand_like(valid, dtype=torch.float32)
        random_values.masked_fill_(~valid, 2.0)
        indices = random_values.argsort(dim=1)[:, :num]
        # Rows with fewer than ``num`` valid anchors are padded by argsort with
        # invalid positions.  Move those padding entries behind the real
        # anchors *before* sorting by sequence position.  Sorting the raw
        # indices first would interleave padding with real anchors while the
        # old prefix-shaped keep mask still marked the first ``count`` entries
        # as valid, potentially discarding every supervised anchor in a row.
        selected_valid = torch.gather(valid, 1, indices)
        padding_position = max_anchor + 1
        indices = indices.masked_fill(~selected_valid, padding_position)
        indices = indices.sort(dim=1).values
        keep = indices != padding_position
        return torch.where(keep, indices, torch.zeros_like(indices)), keep

    def _build_labels_and_weights(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = input_ids.size(1)
        offsets = torch.arange(self.block_size, device=input_ids.device).view(
            1, 1, -1
        )
        positions = anchor_positions.unsqueeze(-1) + offsets
        valid = positions < seq_len
        safe = positions.clamp(max=seq_len - 1)
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe,
        )
        gathered_loss_mask = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe,
        )
        weights = (
            block_keep_mask.unsqueeze(-1).float()
            * valid.float()
            * gathered_loss_mask
        )[:, :, 1:]
        return target_ids[:, :, 1:], target_ids[:, :, :-1], weights, weights > 0

    def prepare_batch(
        self,
        input_ids: torch.Tensor,
        hidden_states: HiddenStatesInput,
        loss_mask: torch.Tensor,
        *,
        anchor_positions: Optional[torch.Tensor] = None,
        block_keep_mask: Optional[torch.Tensor] = None,
        shared_query_embeddings: Optional[torch.Tensor] = None,
    ) -> PreparedFlashMTPBatch:
        if anchor_positions is None or block_keep_mask is None:
            anchor_positions, block_keep_mask = self.sample_anchor_positions(
                input_ids.size(1), loss_mask
            )
        target_hidden = prepare_target_hidden(
            hidden_states,
            anchor_positions,
            self.draft_model.target_layer_ids,
            self.draft_model.config.num_target_layers,
        )
        token_ids, token_keep, token_positions = gather_token_group(
            input_ids,
            anchor_positions,
            self.draft_model.anchor_group_size,
            self.mask_token_id,
        )
        if shared_query_embeddings is None:
            token_embeddings = self.embed_tokens(token_ids)
            token_embeddings = token_embeddings * token_keep.unsqueeze(-1).to(
                token_embeddings.dtype
            )
            mask_ids = torch.full(
                (*token_ids.shape[:2], self.block_size - 1),
                self.mask_token_id,
                device=input_ids.device,
                dtype=torch.long,
            )
            query_embeddings = torch.cat(
                [token_embeddings, self.embed_tokens(mask_ids)], dim=2
            )
        else:
            query_embeddings = shared_query_embeddings
        shared_fused_history = None
        if self.draft_model.is_teacher:
            raw_history = prepare_history_hidden_states(
                hidden_states,
                self.draft_model.history_layer_ids,
                self.draft_model.config.num_target_layers,
            )
            shared_fused_history = self.draft_model.fuse_history_hidden(raw_history)
        labels, prev_ids, weights, binary = self._build_labels_and_weights(
            input_ids, loss_mask, anchor_positions, block_keep_mask
        )
        initial_prev = None
        if self.draft_model.seed_rnn_from_predecessor:
            initial_prev = torch.gather(
                input_ids, 1, (anchor_positions - 1).clamp(min=0)
            )
        return PreparedFlashMTPBatch(
            anchor_positions,
            block_keep_mask,
            target_hidden,
            shared_fused_history,
            query_embeddings,
            token_keep,
            token_positions,
            labels,
            prev_ids,
            weights,
            binary,
            initial_prev,
        )

    def forward_backbone(
        self, batch: PreparedFlashMTPBatch, *, seq_len: int
    ) -> torch.Tensor:
        bsz, num_blocks = batch.anchor_positions.shape
        query_len = self.draft_model.draft_query_length
        query_embeddings = batch.query_embeddings.reshape(
            bsz, num_blocks * query_len, -1
        )
        context_pos, draft_pos = self.draft_model.build_block_position_ids(
            batch.anchor_positions, batch.token_position_ids, batch.token_keep_mask
        )
        attention_mask = _make_flex_mask(
            model_role=self.draft_model.model_role,
            anchor_positions=batch.anchor_positions,
            block_keep_mask=batch.block_keep_mask,
            token_keep_mask=batch.token_keep_mask,
            seq_len=seq_len,
            swa_window_size=self.draft_model.swa_window_size,
            chs_slots=self.draft_model.chs_num_layers,
            query_len=query_len,
            device=query_embeddings.device,
        )
        if self.draft_model.is_teacher:
            if batch.shared_fused_history is None:
                raise ValueError("Teacher requires shared fused history.")
            shared_pos = torch.arange(seq_len, device=query_embeddings.device).view(
                1, -1
            ).expand(bsz, -1)
            # The logical per-block context is [W-1 fuse, S CHS], but the fuse
            # sequence is physically stored once and selected per anchor by the
            # flex mask. Its RoPE ids are therefore ``shared_pos`` exactly once.
            logical_context = context_pos.view(
                bsz, num_blocks, self.draft_model.chs_len_per_block
            )
            chs_pos = logical_context[:, :, -self.draft_model.chs_num_layers :]
            rotary = torch.cat(
                [shared_pos, chs_pos.reshape(bsz, -1), draft_pos], dim=-1
            )
            output = self.draft_model(
                position_ids=draft_pos,
                noise_embedding=query_embeddings,
                target_hidden=batch.target_hidden,
                shared_history=batch.shared_fused_history,
                attention_mask=attention_mask,
                rotary_position_ids=rotary,
            )
        else:
            empty_history = query_embeddings.new_empty(
                bsz, num_blocks, 0, query_embeddings.size(-1)
            )
            rotary = torch.cat([context_pos, draft_pos], dim=-1)
            output = self.draft_model(
                position_ids=draft_pos,
                noise_embedding=query_embeddings,
                target_hidden=batch.target_hidden,
                history_hidden=empty_history,
                attention_mask=attention_mask,
                rotary_position_ids=rotary,
            )
        output = output.view(bsz, num_blocks, query_len, output.size(-1))
        return output[:, :, -self.draft_model.proposal_length :, :]

    @staticmethod
    def _position_weights(raw: torch.Tensor, gamma: Optional[float]) -> torch.Tensor:
        if gamma is None or gamma <= 0:
            return raw
        offsets = torch.arange(raw.size(-1), device=raw.device).view(1, 1, -1)
        return raw * torch.exp(-offsets.float() / float(gamma))

    @staticmethod
    def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return (values * weights).sum() / weights.sum().clamp_min(1e-6)

    def compute_supervised_loss(
        self,
        prediction_hidden: torch.Tensor,
        batch: PreparedFlashMTPBatch,
        target_prefill_logits: torch.Tensor,
    ) -> FlashMTPLossOutput:
        """Compute each active full-vocabulary tensor exactly once in FP32."""
        active_positions = batch.binary_eval_mask.bool()
        if not active_positions.any():
            raise ValueError("FlashMTP loss has no supervised label positions.")

        # Backend attention can return arbitrary values for padded query rows.
        # Clear them before the serial head; masking the eventual scalar loss is
        # insufficient because NaN * 0 remains NaN.
        prediction_hidden = torch.where(
            active_positions.unsqueeze(-1),
            prediction_hidden,
            torch.zeros_like(prediction_hidden),
        )
        prev_token_ids = torch.where(
            active_positions,
            batch.prev_token_ids,
            torch.zeros_like(batch.prev_token_ids),
        )
        initial_prev_token_ids = batch.initial_prev_token_ids
        if initial_prev_token_ids is not None:
            initial_prev_token_ids = torch.where(
                batch.block_keep_mask,
                initial_prev_token_ids,
                torch.zeros_like(initial_prev_token_ids),
            )

        final_weights = self._position_weights(
            batch.raw_weight_mask, self.loss_decay_gamma
        )
        base_weights = self._position_weights(
            batch.raw_weight_mask, self.base_lm_ce_decay_gamma
        )
        active_hidden = prediction_hidden[active_positions]
        active_labels = batch.labels[active_positions]
        active_final_weights = final_weights[active_positions].float()
        active_base_weights = base_weights[active_positions].float()
        markov_head = self.draft_model.markov_head
        output_mode = self.draft_model.markov_output_mode
        use_base_logits = (
            markov_head is None
            or markov_output_uses_base_lm_head(output_mode)
            or self.base_lm_ce_weight > 0
        )
        base_logits = self.lm_head(active_hidden).float() if use_base_logits else None
        if markov_head is None:
            assert base_logits is not None
            final_logits = base_logits
        else:
            if self.markov_teacher_forcing_ratio < 1.0:
                latent = markov_head.forward_scheduled_sampling(
                    hidden_states=prediction_hidden,
                    prev_token_ids=prev_token_ids,
                    output_mode=output_mode,
                    initial_prev_token_ids=initial_prev_token_ids,
                    teacher_forcing_ratio=self.markov_teacher_forcing_ratio,
                )
            else:
                latent = markov_head.forward_teacher_forcing(
                    hidden_states=prediction_hidden,
                    prev_token_ids=prev_token_ids,
                    output_mode=output_mode,
                    initial_prev_token_ids=initial_prev_token_ids,
                )
            head_logits = markov_head.project_logits(latent[active_positions]).float()
            final_logits = (
                base_logits + head_logits
                if markov_output_uses_base_lm_head(output_mode)
                else head_logits
            )
        if (active_labels < 0).any() or (
            active_labels >= final_logits.size(-1)
        ).any():
            raise ValueError(
                "Supervised labels must be within the output vocabulary: "
                f"min={int(active_labels.min().item())}, "
                f"max={int(active_labels.max().item())}, "
                f"vocab_size={final_logits.size(-1)}."
            )
        final_ce_values = F.cross_entropy(
            final_logits.float(), active_labels, reduction="none"
        )
        final_ce = self._weighted_mean(final_ce_values, active_final_weights)
        if self.base_lm_ce_weight > 0:
            assert base_logits is not None
            base_ce_values = F.cross_entropy(
                base_logits.float(), active_labels, reduction="none"
            )
            base_ce = self._weighted_mean(base_ce_values, active_base_weights)
        else:
            base_ce = prediction_hidden.new_zeros((), dtype=torch.float32)
        if self.tv_loss_weight > 0:
            active_target_logits = target_prefill_logits[active_positions]
            if active_target_logits.size(-1) < final_logits.size(-1):
                active_target_logits = F.pad(
                    active_target_logits,
                    (0, final_logits.size(-1) - active_target_logits.size(-1)),
                    value=torch.finfo(active_target_logits.dtype).min,
                )
            elif active_target_logits.size(-1) != final_logits.size(-1):
                raise ValueError("Target and draft vocab sizes do not match.")
            target_probs = F.softmax(active_target_logits.float(), dim=-1)
            final_probs = F.softmax(final_logits.float(), dim=-1)
            tv_values = (final_probs - target_probs).abs().sum(dim=-1)
            tv_loss = self._weighted_mean(tv_values, active_final_weights)
        else:
            tv_loss = prediction_hidden.new_zeros((), dtype=torch.float32)
        total = (
            self.final_ce_weight * final_ce
            + self.tv_loss_weight * tv_loss
            + self.base_lm_ce_weight * base_ce
        )
        with torch.no_grad():
            predictions = torch.zeros_like(batch.labels)
            predictions[active_positions] = final_logits.argmax(dim=-1)
            valid = batch.binary_eval_mask
            accuracy = ((predictions == batch.labels) & valid).sum().float() / valid.sum().clamp_min(1)
            correct = (predictions == batch.labels) & valid
            prefix = correct.cumprod(dim=-1).sum(dim=-1).float() + 1.0
            valid_blocks = batch.block_keep_mask & valid.any(dim=-1)
            prefix_acc = prefix[valid_blocks].mean() if bool(valid_blocks.any()) else prediction_hidden.new_zeros(())
        return FlashMTPLossOutput(
            total, accuracy, prefix_acc, final_ce, base_ce, tv_loss
        )

    def forward(
        self,
        *,
        input_ids: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        hidden_states: Optional[HiddenStatesInput] = None,
        target_prefill_logits: Optional[torch.Tensor] = None,
        anchor_positions: Optional[torch.Tensor] = None,
        block_keep_mask: Optional[torch.Tensor] = None,
        prepared_batch: Optional[PreparedFlashMTPBatch] = None,
        seq_len: Optional[int] = None,
        return_backbone: bool = False,
        target_logits_are_gathered: bool = False,
    ):
        if prepared_batch is None:
            if input_ids is None or loss_mask is None or hidden_states is None:
                raise ValueError("input_ids, loss_mask and hidden_states are required")
            prepared_batch = self.prepare_batch(
                input_ids,
                hidden_states,
                loss_mask,
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
            )
            seq_len = input_ids.size(1)
        if seq_len is None:
            raise ValueError("seq_len is required with prepared_batch")
        prediction_hidden = self.forward_backbone(prepared_batch, seq_len=seq_len)
        if return_backbone:
            return prediction_hidden
        if target_prefill_logits is None:
            raise ValueError("target_prefill_logits are required for supervised loss")
        gathered_target_logits = (
            target_prefill_logits
            if target_logits_are_gathered
            else gather_target_prefill_logits(
                target_prefill_logits,
                prepared_batch.anchor_positions,
                self.block_size,
            )
        )
        return self.compute_supervised_loss(
            prediction_hidden, prepared_batch, gathered_target_logits
        ).as_tuple()


def compute_stage1_distillation_loss(
    *,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    lm_head: nn.Module,
    raw_weight_mask: torch.Tensor,
    tv_weight: float,
    hidden_weight: float,
    smooth_l1_beta: float,
    loss_decay_gamma: Optional[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Active-position FP32 TV + SmoothL1 distillation with a detached teacher."""
    offsets = torch.arange(raw_weight_mask.size(-1), device=raw_weight_mask.device)
    decay = (
        torch.ones_like(offsets, dtype=torch.float32)
        if loss_decay_gamma is None or loss_decay_gamma <= 0
        else torch.exp(-offsets.float() / float(loss_decay_gamma))
    )
    weights = raw_weight_mask * decay.view(1, 1, -1)
    active_positions = weights > 0
    if not active_positions.any():
        raise ValueError("Stage-1 distillation has no supervised positions.")
    active_weights = weights[active_positions].float()
    student_active = student_hidden[active_positions]
    teacher_active = teacher_hidden[active_positions]
    denominator = active_weights.sum().clamp_min(1e-6)
    with torch.no_grad():
        teacher_probs = F.softmax(lm_head(teacher_active).float(), dim=-1)
    student_probs = F.softmax(lm_head(student_active).float(), dim=-1)
    tv_values = (student_probs - teacher_probs).abs().sum(dim=-1)
    tv_loss = (tv_values * active_weights).sum() / denominator
    hidden_values = F.smooth_l1_loss(
        student_active.float(),
        teacher_active.detach().float(),
        reduction="none",
        beta=float(smooth_l1_beta),
    ).mean(dim=-1)
    hidden_loss = (hidden_values * active_weights).sum() / denominator
    return (
        float(tv_weight) * tv_loss + float(hidden_weight) * hidden_loss,
        tv_loss,
        hidden_loss,
    )


__all__ = [
    "FlashMTPLossOutput",
    "HiddenStatesInput",
    "OnlineFlashMTPModel",
    "PreparedFlashMTPBatch",
    "compute_stage1_distillation_loss",
    "gather_target_prefill_logits",
    "gather_token_group",
    "prepare_history_hidden_states",
    "prepare_target_hidden",
]
