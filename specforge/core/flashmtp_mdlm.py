# coding=utf-8
"""FlashMTP v3.3 Phase-1: MDLM-style random mask training (CE + optional KL to target)."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from specforge.modeling.draft.flashmtp import (
    FlashMTPDraftModel,
)

from specforge.core.flashmtp import CHS_LEN_PER_BLOCK, create_flashmtp_block_mask

def kl_to_teacher(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    kl_topk: int,
) -> torch.Tensor:
    """KL(teacher‖student)，对 batch 维做 batchmean；kl_topk∈(0,V) 时只在教师 top-k 子空间上算，省显存。"""
    s = student_logits.float()
    t = teacher_logits.float()
    v = s.size(-1)
    if kl_topk <= 0 or kl_topk >= v:
        return F.kl_div(
            F.log_softmax(s, -1),
            F.softmax(t, -1),
            reduction="batchmean",
            log_target=False,
        )
    _, topi = t.topk(kl_topk, dim=-1)
    s_sel = s.gather(-1, topi)
    t_sel = t.gather(-1, topi)
    return F.kl_div(
        F.log_softmax(s_sel, -1),
        F.softmax(t_sel, -1),
        reduction="batchmean",
        log_target=False,
    )


class FlashMTPMDLMModel(nn.Module):
    """MDLM: random mask ratio per block, CE on masked tokens; optional KL to target logits."""

    def __init__(
        self,
        draft_model: FlashMTPDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        mask_ratio_min: float = 0.1,
        mask_ratio_max: float = 1.0,
        kl_weight: float = 0.0,
        kl_topk: int = 0,
        ce_weight: float = 1.0,
    ):
        super().__init__()
        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.mask_token_id = mask_token_id
        self.block_size = block_size
        self.attention_backend = attention_backend
        self.num_anchors = num_anchors
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max
        self.kl_weight = kl_weight
        self.kl_topk = kl_topk
        self.ce_weight = ce_weight

    def _sample_anchor_positions(
        self,
        seq_len: int,
        loss_mask: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 在可监督前缀内随机抽锚点起点；每个起点标定一块 [p, p+bs)；输出 anchors 与块级有效 mask。
        bs = self.block_size
        bsz = loss_mask.shape[0]
        max_anchor = max(seq_len - bs, 0)
        valid = loss_mask[:, : max_anchor + 1] > 0.5
        valid_counts = valid.sum(dim=1)
        max_n = min(self.num_anchors, int(valid_counts.max().item()) - 1)
        if max_n <= 0:
            raise ValueError("MDLM: need longer supervised spans; check data filter.")
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

    def _mdlm_noise_and_mask(
        self,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (noise_embeddings (B, N*bs, H), ce_mask (B, N, bs) bool)."""
        # 每块独立采样掩码率；仅 loss_mask 内且非块首（anchor token，offset>0）可置 MASK；
        # 若某行在可掩位置上仍全未掩，则随机强制掩 1 处（保证该行 CE 有监督）。
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size
        device = input_ids.device
        offsets = torch.arange(bs, device=device).view(1, 1, -1)
        block_idx = anchor_positions.unsqueeze(-1) + offsets
        block_idx = block_idx.clamp(max=seq_len - 1)
        block_tokens = torch.gather(
            input_ids.unsqueeze(1).expand(-1, n, -1), 2, block_idx
        )
        lm = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, n, -1), 2, block_idx
        )
        valid_block = block_keep_mask.view(bsz, n, 1).expand(-1, -1, bs)
        pos_maskable = offsets > 0
        can_mask = (lm > 0.5) & valid_block & pos_maskable
        ratio = torch.empty(bsz, n, 1, device=device).uniform_(
            self.mask_ratio_min, self.mask_ratio_max
        )
        u = torch.rand(bsz, n, bs, device=device)
        masked = (u < ratio) & can_mask
        row_need = (~masked.any(dim=-1)) & can_mask.any(dim=-1)
        if row_need.any():
            pick = torch.argmax(
                torch.where(
                    can_mask,
                    torch.rand(bsz, n, bs, device=device),
                    torch.full((), -1.0, device=device),
                ),
                dim=-1,
            )
            b_idx, n_idx = torch.where(row_need)
            masked[b_idx, n_idx, pick[b_idx, n_idx]] = True
        masked_ids = torch.where(
            masked,
            torch.full((), self.mask_token_id, device=device, dtype=torch.long),
            block_tokens,
        )
        noise_emb = self.embed_tokens(masked_ids.view(bsz, n * bs))
        return noise_emb, masked

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        hidden_states,
        loss_mask: torch.Tensor,
        teacher_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        # 1) 锚点 + 块内随机 MASK → 噪声嵌入；
        # 2) v5 直接传完整历史 hidden，draft 内部用 pivot attend pivot 之前的融合历史。
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )
        noise_embedding, ce_mask = self._mdlm_noise_and_mask(
            input_ids, anchor_positions, block_keep_mask, loss_mask
        )
        draft_position_ids = self._create_position_ids(anchor_positions)
        flashmtp_attn_mask = create_flashmtp_block_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            chs_len_per_block=CHS_LEN_PER_BLOCK,
            block_size=self.block_size,
            device=device,
        )
        output_hidden = self.draft_model(
            position_ids=draft_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            target_attention_mask=attention_mask,
            attention_mask=flashmtp_attn_mask,
        )
        logits = self.lm_head(output_hidden)
        # CE：仅 ce_mask ∧ 序列边界 ∧ loss_mask 处，对真 token_id 做交叉熵；可选在相同位置上对 teacher_logits 加 KL。
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
        # 块内仅 pos_in_block>0 参与 CE/KL/acc（排除块首，与 pivot 后第一候选位对齐）。
        pos_in_block_ok = (label_offsets > 0).float()
        w = (
            ce_mask.float()
            * valid_label_mask.float()
            * (lm_g > 0.5).float()
            * pos_in_block_ok
        )
        w_flat = w.view(-1)
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = target_ids.view(-1)
        ce = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        denom = w_flat.sum() + 1e-6
        loss_ce = (ce * w_flat).sum() / denom

        loss_kl = torch.zeros((), device=device, dtype=loss_ce.dtype)
        # 教师 logits 按「全序列」位置 gather 到块内，与 logits 展平后对齐，仅在 w 为真处平均 KL。
        if self.kl_weight > 0 and teacher_logits is not None and w_flat.sum() > 0:
            li = safe_label_indices.reshape(bsz, -1)
            v = teacher_logits.size(-1)
            te_blk = torch.gather(
                teacher_logits,
                1,
                li.unsqueeze(-1).expand(-1, -1, v),
            )
            st_blk = logits.reshape(bsz, -1, v)
            w_blk = w.reshape(bsz, -1) > 0.5
            if w_blk.any():
                loss_kl = kl_to_teacher(
                    st_blk[w_blk], te_blk[w_blk], self.kl_topk
                )

        loss = self.ce_weight * loss_ce + self.kl_weight * loss_kl

        with torch.no_grad():
            pred = torch.argmax(flat_logits, dim=-1)
            correct = (pred == flat_targets) & (w_flat > 0.5)
            acc = correct.sum().float() / (w_flat > 0.5).sum().float().clamp(min=1.0)

        return loss, acc
