# coding=utf-8
"""FlashMTP v3.3 Phase-2: positive confidence-aware streak surrogate."""


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
    """块首为真实 token 嵌入、块内其余为 [MASK]。

    Streak 项用 count - score 写成正数形式，等价于最大化 streak score；
    confidence 权重只缩放 streak 梯度。CE_aux 是逐位置平均 CE，不做位置/置信度调权。
    """

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
        streak_weight: float = 1.0,
        ce_aux_weight: float = 0.0,
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
        self.streak_weight = streak_weight
        self.ce_aux_weight = ce_aux_weight

    def _sample_anchor_positions(
        self,
        seq_len: int,
        loss_mask: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 与 MDLM 相同：在可监督区间内随机抽 N 个块起点，每个块长度 bs。
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

    def _noise_embed_for_streak(
        self, input_ids: torch.Tensor, anchor_positions: torch.Tensor
    ) -> torch.Tensor:
        # 与训练/推理约定一致：块内第一位（anchor token）用真 token 嵌入，其余槽位为 [MASK]。
        bsz, n = anchor_positions.shape
        bs = self.block_size
        device = input_ids.device
        row = torch.arange(bsz, device=device, dtype=torch.long).unsqueeze(1).expand(
            -1, n
        )
        anchor_tok = input_ids[row, anchor_positions]
        noise_ids = torch.full(
            (bsz, n, bs), self.mask_token_id, dtype=torch.long, device=device
        )
        noise_ids[:, :, 0] = anchor_tok
        return self.embed_tokens(noise_ids.view(bsz, n * bs))

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states,
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )
        noise_embedding = self._noise_embed_for_streak(input_ids, anchor_positions)
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
        # 草案：块首 clean 嵌入 + 后续 [MASK]；标签与 streak 仍仅 pos_in_block>0。
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
        # 各位置对真 token 的 log q；先在 (B, N*bs) 维 gather 再 reshape，与 target_ids 对齐。
        flat_tgt = target_ids.reshape(bsz, n * bs).unsqueeze(-1)
        lp = (
            log_probs.gather(-1, flat_tgt)
            .squeeze(-1)
            .clamp(min=self.log_prob_min)
            .view(bsz, n, bs)
        )
        # 块内仅 pos_in_block>0 参与 streak/CE/acc；m=0 对应块首，不参与外层 sum。
        pos_in_block_ok = (label_offsets > 0).float()
        valid_pos = (
            block_keep_mask.unsqueeze(-1)
            .expand(-1, -1, bs)
            .float()
            * valid_label_mask.float()
            * (lm_g > 0.5).float()
            * pos_in_block_ok
        )

        # confidence-aware straight-through log-prob:
        # forward(lp_tilde) == lp, backward(d lp_tilde / d lp) == conf_weight.
        conf = lp.detach().exp()
        conf_weight = torch.sigmoid(-10.0 * (conf - 0.6))
        lp_tilde = lp.detach() + conf_weight * (lp - lp.detach())

        lp_tail = lp_tilde[..., 1:]
        valid_tail = valid_pos[..., 1:]
        prefix_valid = valid_tail.cumprod(dim=-1)
        prefix_log = (lp_tail * valid_tail).cumsum(dim=-1)
        streak_score = prefix_log.exp() * prefix_valid
        denom_streak = prefix_valid.sum() + 1e-6
        # Positive form averaged over supervised prefix positions (normally B - 1).
        loss_streak = (prefix_valid - streak_score).sum() / denom_streak

        if self.ce_aux_weight > 0:
            logits_blk = logits.view(bsz, n, bs, v)
            ce = F.cross_entropy(
                logits_blk.float().reshape(-1, v),
                target_ids.reshape(-1),
                reduction="none",
            ).reshape(bsz, n, bs)
            # CE is also averaged over supervised positions (normally B - 1 per block).
            denom_ce = valid_pos.sum() + 1e-6
            loss_ce = (ce * valid_pos).sum() / denom_ce
        else:
            loss_ce = torch.zeros((), device=device, dtype=loss_streak.dtype)

        loss_total = self.streak_weight * loss_streak + self.ce_aux_weight * loss_ce

        with torch.no_grad():
            pred = torch.argmax(log_probs, dim=-1).view(bsz, n, bs)
            correct = (pred == target_ids) & (valid_pos > 0.5)
            acc = correct.sum().float() / (valid_pos > 0.5).sum().float().clamp(min=1.0)

        return (
            loss_total,
            acc,
            loss_streak.detach(),
            loss_ce.detach(),
        )
