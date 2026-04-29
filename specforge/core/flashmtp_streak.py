# coding=utf-8
"""FlashMTP v3.3 Phase-2: streak surrogate (maximize sum_m prod_{j<=m} q_j(x_j|p))."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from specforge.modeling.draft.flashmtp import (
    FlashMTPDraftModel,
    stack_hidden_states_for_positions,
)

from specforge.core.flashmtp import CHS_LEN_PER_BLOCK, create_flashmtp_block_mask


class FlashMTPStreakModel(nn.Module):
    """All-[MASK] draft block; loss = -sum_m exp(sum_{j<=m} log q_j(x_j))."""

    def __init__(
        self,
        draft_model: FlashMTPDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        log_prob_min: float = -40.0,
    ):
        super().__init__()
        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.mask_token_id = mask_token_id
        self.block_size = block_size
        self.attention_backend = attention_backend
        self.num_anchors = num_anchors
        self.log_prob_min = log_prob_min

    def _sample_anchor_positions(
        self,
        seq_len: int,
        loss_mask: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = self.block_size
        bsz = loss_mask.shape[0]
        max_anchor = max(seq_len - bs, 0)
        valid = loss_mask[:, : max_anchor + 1] > 0.5
        valid_counts = valid.sum(dim=1)
        max_n = min(self.num_anchors, int(valid_counts.max().item()) - 1)
        if max_n <= 0:
            raise ValueError("Streak: need longer supervised spans; check data filter.")
        indices = torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        masked_indices = torch.where(
            valid, indices, torch.tensor(seq_len + 1, device=device)
        )
        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, torch.tensor(2.0, device=device))
        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        anchors = gathered[:, :max_n].sort(dim=1).values
        keep_mask = (
            torch.arange(max_n, device=device).unsqueeze(0)
            < valid_counts.unsqueeze(1).clamp(max=max_n)
        )
        anchors = torch.where(
            keep_mask, anchors, torch.tensor(0, dtype=torch.long, device=device)
        )
        return anchors, keep_mask

    def _create_position_ids(self, anchor_positions: torch.Tensor) -> torch.Tensor:
        bsz, n_blocks = anchor_positions.shape
        device = anchor_positions.device
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        return (anchor_positions.unsqueeze(-1) + offsets).view(bsz, -1)

    def _all_mask_embed(self, anchor_positions: torch.Tensor):
        bsz, n = anchor_positions.shape
        bs = self.block_size
        device = anchor_positions.device
        noise_ids = torch.full(
            (bsz, n * bs), self.mask_token_id, dtype=torch.long, device=device
        )
        return self.embed_tokens(noise_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states,
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )
        noise_embedding = self._all_mask_embed(anchor_positions)
        draft_position_ids = self._create_position_ids(anchor_positions)
        flashmtp_attn_mask = create_flashmtp_block_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            chs_len_per_block=CHS_LEN_PER_BLOCK,
            block_size=self.block_size,
            device=device,
        )
        context_positions = (anchor_positions - 1).clamp(min=0)
        target_hidden = stack_hidden_states_for_positions(
            hidden_states, context_positions
        )
        output_hidden = self.draft_model(
            position_ids=draft_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            attention_mask=flashmtp_attn_mask,
        )
        logits = self.lm_head(output_hidden)
        v = logits.size(-1)
        log_probs = F.log_softmax(logits.float().view(bsz, -1, v), dim=-1)

        label_offsets = torch.arange(0, self.block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        lm_g = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        n = anchor_positions.size(1)
        bs = self.block_size
        flat_tgt = target_ids.reshape(bsz, n * bs).unsqueeze(-1)
        lp = (
            log_probs.gather(-1, flat_tgt)
            .squeeze(-1)
            .clamp(min=self.log_prob_min)
            .view(bsz, n, bs)
        )
        w = (
            block_keep_mask.unsqueeze(-1)
            .expand(-1, -1, bs)
            .float()
            * valid_label_mask.float()
            * (lm_g > 0.5).float()
        )
        weighted_lp = lp * w
        cum = weighted_lp.cumsum(dim=-1)
        streak_sum = cum.exp().sum(dim=-1)
        loss = -streak_sum.mean()

        with torch.no_grad():
            pred = torch.argmax(log_probs, dim=-1).view(bsz, n, bs)
            correct = (pred == target_ids) & (w > 0.5)
            acc = correct.sum().float() / (w > 0.5).sum().float().clamp(min=1.0)

        return loss, acc
