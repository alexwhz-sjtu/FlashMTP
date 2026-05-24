# coding=utf-8
"""FlashMTP v3.3 Phase-2: log-smoothed relative streak surrogate."""


from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from specforge.core.flashmtp import (
    _draft_position_ids_for_flashmtp_training,
    _full_rotary_position_ids_for_flashmtp,
    create_flashmtp_block_mask,
)
from specforge.modeling.draft.flashmtp import FlashMTPDraftModel, prepare_target_hidden


class FlashMTPStreakModel(nn.Module):
    """块首为真实 token 嵌入、块内其余为 [MASK]。

    默认 streak 使用 Log-Smoothed Relative Streak Loss（LS-RSL）：教师 target 概率作锚点
    ``T=max(0.5,p_teacher)``，相对 log 上做分段映射 ``log_phi``；高置信时对 streak 不回传梯度。

    ``streak_raw_probs=True`` 时简化为直接使用草案在真标签上的 log 概率 ``log q``（经
    ``log_prob_min`` 截断），不做教师锚点与 ``log_phi`` 映射。
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
        streak_raw_probs: bool = False,
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
        self.streak_raw_probs = streak_raw_probs

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

    @torch.no_grad()
    def _teacher_target_probs(
        self,
        hidden_states,
        teacher_logits: Optional[torch.Tensor],
        teacher_context_indices: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """返回教师在 target token 上的概率 p_j，形状为 (B, N, bs)。"""
        bsz, n, bs = target_ids.shape
        if teacher_logits is not None:
            v = teacher_logits.size(-1)
            te_blk = torch.gather(
                teacher_logits.float(),
                1,
                teacher_context_indices.reshape(bsz, -1)
                .unsqueeze(-1)
                .expand(-1, -1, v),
            )
            te_log_probs = F.log_softmax(te_blk, dim=-1)
        else:
            if isinstance(hidden_states, torch.Tensor) and hidden_states.dim() == 4:
                final_hidden = hidden_states[:, :, -1, :]
            elif isinstance(hidden_states, torch.Tensor):
                final_hidden = hidden_states
            else:
                final_hidden = hidden_states[-1]
            h = torch.gather(
                final_hidden,
                1,
                teacher_context_indices.reshape(bsz, -1)
                .unsqueeze(-1)
                .expand(-1, -1, final_hidden.size(-1)),
            )
            te_log_probs = F.log_softmax(self.lm_head(h).float(), dim=-1)
        return (
            te_log_probs.gather(-1, target_ids.reshape(bsz, -1).unsqueeze(-1))
            .squeeze(-1)
            .exp()
            .view(bsz, n, bs)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states,
        loss_mask: torch.Tensor,
        teacher_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )
        noise_embedding = self._noise_embed_for_streak(input_ids, anchor_positions)
        draft_position_ids = _draft_position_ids_for_flashmtp_training(
            anchor_positions,
            self.block_size,
            self.draft_model.local_position,
        )
        full_rotary_position_ids = _full_rotary_position_ids_for_flashmtp(
            anchor_positions,
            draft_position_ids,
            self.draft_model.chs_len_per_block,
            self.draft_model.local_position,
        )
        chs_len = self.draft_model.chs_len_per_block
        flashmtp_attn_mask = create_flashmtp_block_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            chs_len_per_block=chs_len,
            block_size=self.block_size,
            device=device,
        )
        target_hidden = prepare_target_hidden(
            hidden_states,
            anchor_positions,
            self.draft_model.target_layer_ids,
            self.draft_model.config.num_target_layers,
        )
        output_hidden = self.draft_model(
            position_ids=draft_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            attention_mask=flashmtp_attn_mask,
            rotary_position_ids=full_rotary_position_ids,
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

        if self.streak_raw_probs:
            # 简化 streak：前缀积用草案对真标签的 log 概率之和（等价于 Σ log q），无教师锚点、无 log_phi。
            lp_tail = lp[..., 1:]
        else:
            teacher_p = self._teacher_target_probs(
                hidden_states=hidden_states,
                teacher_logits=teacher_logits,
                teacher_context_indices=(safe_label_indices - 1).clamp(min=0),
                target_ids=target_ids,
            )
            target_anchor = torch.maximum(
                torch.full_like(teacher_p, 0.5), teacher_p
            ).clamp_min(1e-12)

            # LS-RSL: log_rho = log(q/T), T = max(0.5, p_teacher(y*)).
            log_rho = lp - target_anchor.log()
            high_conf_alpha = 0.5
            pos_value = torch.log1p(high_conf_alpha * log_rho.clamp_min(0.0))
            log_phi = torch.where(log_rho <= 0, log_rho, pos_value.detach())
            lp_tail = log_phi[..., 1:]
        valid_tail = valid_pos[..., 1:]
        prefix_valid = valid_tail.cumprod(dim=-1)
        prefix_log = (lp_tail * valid_tail).cumsum(dim=-1)
        relative_streak = prefix_log.exp() * prefix_valid
        block_has_streak = (prefix_valid.sum(dim=-1) > 0).float()
        valid_block_count = block_has_streak.sum() + 1e-6

        # Zero means the relative streak sum reaches the valid prefix count.
        streak_sum = relative_streak.sum(dim=-1).clamp_min(1e-12)
        target_streak = prefix_valid.sum(dim=-1).clamp_min(1.0)

        # log sum
        loss_streak = (
            (target_streak.log() - streak_sum.log()) * block_has_streak
        ).sum() / valid_block_count

        # sum
        # loss_streak = (
        #     (target_streak - streak_sum) * block_has_streak
        # ).sum() / valid_block_count

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
