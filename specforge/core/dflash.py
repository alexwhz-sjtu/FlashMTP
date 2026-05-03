# coding=utf-8
"""DFlash Training Wrapper."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from specforge.modeling.draft.dflash import DFlashDraftModel

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    BlockMask = None
    create_block_mask = None


def create_dflash_block_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    S: int,
    block_size: int,
    device: torch.device,
):
    """Construct Flex Attention BlockMask for DFlash training.

    KV: [Context (S tokens) | Block_0 | Block_1 | ... | Block_{n-1}]
    Q:  [Block_0 | Block_1 | ... | Block_{n-1}]

    Rules:
      1. Each block sees context strictly before its anchor (kv_idx < anchor_pos).
      2. Intra-block attention is bidirectional.
      3. Different blocks are invisible to each other.
      4. Invalid blocks (block_keep_mask=False) see nothing.
    """

    def dflash_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        anchor_pos = anchor_positions[b, q_block_id]

        is_context = kv_idx < S
        # Strictly less than: matches inference where target_hidden[anchor_pos]
        # is not available as context.
        mask_context = is_context & (kv_idx < anchor_pos)

        is_draft = kv_idx >= S
        kv_block_id = (kv_idx - S) // block_size
        mask_draft = is_draft & (q_block_id == kv_block_id)

        is_valid_block = block_keep_mask[b, q_block_id]
        return (mask_context | mask_draft) & is_valid_block

    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size

    return create_block_mask(
        dflash_mask_mod, B=B, H=None, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device
    )


class OnlineDFlashModel(nn.Module):
    """DFlash online training wrapper with CE + LS-RSL streak loss."""

    def __init__(
        self,
        draft_model: DFlashDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        streak_weight: float = 1.0,
        ce_weight: float = 1.0,
        streak_decay_gamma: float = 7.0,
        log_prob_min: float = -40.0,
    ):
        super().__init__()
        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.attention_backend = attention_backend
        self.num_anchors = num_anchors
        self.streak_weight = streak_weight
        self.ce_weight = ce_weight
        self.streak_decay_gamma = streak_decay_gamma
        self.log_prob_min = log_prob_min

        self._cached_block_mask: Optional[BlockMask] = None
        self._cached_seq_len: Optional[int] = None
        self._cached_bsz: Optional[int] = None

    def _sample_anchor_positions(
        self, seq_len: int, loss_mask: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample anchor positions per sample; returns (anchors, keep_mask)."""
        bs = self.block_size
        bsz = loss_mask.shape[0]
        max_anchor = max(seq_len - bs, 0)

        valid = loss_mask[:, : max_anchor + 1] > 0.5
        valid_counts = valid.sum(dim=1)
        max_n = min(self.num_anchors, int(valid_counts.max().item()) - 1)

        if max_n <= 0:
            raise ValueError("should preprocess the data.")

        indices = (
            torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        )
        masked_indices = torch.where(
            valid, indices, torch.tensor(seq_len + 1, device=device)
        )

        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, torch.tensor(2.0, device=device))

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        anchors = gathered[:, :max_n].sort(dim=1).values

        keep_mask = torch.arange(max_n, device=device).unsqueeze(
            0
        ) < valid_counts.unsqueeze(1).clamp(max=max_n)
        anchors = torch.where(
            keep_mask, anchors, torch.tensor(0, dtype=torch.long, device=device)
        )

        return anchors, keep_mask

    def prepare_noise_input(
        self, input_ids: torch.Tensor, block_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Prepare noise input: first token of each block is real, rest are MASK."""
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        if block_ids is not None:
            is_block_start = torch.ones(bsz, seq_len, dtype=torch.bool, device=device)
            is_block_start[:, 1:] = block_ids[:, 1:] != block_ids[:, :-1]
        else:
            positions = torch.arange(seq_len, device=device)
            is_block_start = (positions % self.block_size) == 0
            is_block_start = is_block_start.unsqueeze(0).expand(bsz, -1)

        noise_input_ids = torch.full_like(input_ids, self.mask_token_id)
        noise_input_ids[is_block_start] = input_ids[is_block_start]
        return noise_input_ids

    def _create_position_ids(self, anchor_positions: torch.Tensor) -> torch.Tensor:
        """Create absolute position IDs for parallel draft blocks."""
        bsz, n_blocks = anchor_positions.shape
        device = anchor_positions.device
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        pos_ids = anchor_positions.unsqueeze(-1) + offsets
        return pos_ids.view(bsz, -1)

    def _create_noise_embed(self, input_ids, anchor_positions, block_keep_mask):
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size
        device = input_ids.device

        noise_ids = torch.full(
            (bsz, n * bs), self.mask_token_id, dtype=torch.long, device=device
        )

        block_starts = torch.arange(n, device=device) * bs
        block_starts = block_starts.unsqueeze(0).expand(bsz, -1)

        valid_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, valid_anchor_positions)

        flat_batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n)
        noise_ids[flat_batch_idx, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(self.mask_token_id, dtype=torch.long, device=device),
        )

        return self.embed_tokens(noise_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        teacher_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Parallel block-wise training forward pass."""
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )

        noise_embedding = self._create_noise_embed(
            input_ids, anchor_positions, block_keep_mask
        )

        context_position_ids = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        )
        draft_position_ids = self._create_position_ids(anchor_positions)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)

        dflash_attn_mask = create_dflash_block_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            S=seq_len,
            block_size=self.block_size,
            device=device,
        )

        output_hidden = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            attention_mask=dflash_attn_mask,
        )

        logits = self.lm_head(output_hidden)
        vocab_size = logits.size(-1)

        # --- Labels: same-position prediction (position k predicts token anchor+k) ---
        label_offsets = torch.arange(0, self.block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )

        # --- Valid mask: block validity * bounds * exclude anchor (pos 0) * loss_mask ---
        valid_pos = (
            block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        )
        valid_pos = valid_pos * valid_label_mask.float()

        pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
        valid_pos = valid_pos * (pos_in_block > 0).float()

        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        valid_pos = valid_pos * original_loss_mask_gathered

        binary_eval_mask = valid_pos.view(-1)

        # --- CE: average over valid supervised positions, without position weights. ---
        flat_logits = logits.view(-1, vocab_size)
        flat_targets = target_ids.view(-1)
        flat_valid = valid_pos.view(-1)

        loss_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        valid_token_count = flat_valid.sum() + 1e-6
        loss_ce = (loss_per_token * flat_valid).sum() / valid_token_count

        # --- LS-RSL streak loss from FlashMTP v3.3 method 2. ---
        log_probs = F.log_softmax(logits.float().view(bsz, -1, vocab_size), dim=-1)
        log_q = (
            log_probs.gather(-1, target_ids.view(bsz, -1).unsqueeze(-1))
            .squeeze(-1)
            .clamp(min=self.log_prob_min)
            .view(bsz, anchor_positions.size(1), self.block_size)
        )

        if teacher_logits is not None:
            teacher_context_indices = (safe_label_indices - 1).clamp(min=0)
            teacher_vocab_size = teacher_logits.size(-1)
            teacher_block_logits = torch.gather(
                teacher_logits.float(),
                1,
                teacher_context_indices.view(bsz, -1)
                .unsqueeze(-1)
                .expand(-1, -1, teacher_vocab_size),
            )
            teacher_log_probs = F.log_softmax(teacher_block_logits, dim=-1)
            teacher_p = (
                teacher_log_probs.gather(-1, target_ids.view(bsz, -1).unsqueeze(-1))
                .squeeze(-1)
                .exp()
                .view(bsz, anchor_positions.size(1), self.block_size)
            )
        else:
            teacher_p = torch.full_like(log_q, 0.5).exp().new_full(log_q.shape, 0.5)

        target_anchor = torch.maximum(
            torch.full_like(teacher_p, 0.5), teacher_p
        ).clamp_min(1e-12)
        log_rho = log_q - target_anchor.log()

        gamma = max(self.block_size - 1, 1)
        streak_positions = torch.arange(self.block_size, device=device).view(1, 1, -1)
        streak_weights = torch.exp(
            -((gamma - streak_positions).clamp(min=0)).float()
            / self.streak_decay_gamma
        )
        log_phi = torch.where(
            log_rho < 0,
            log_rho,
            torch.log1p(streak_weights * log_rho.clamp_min(0.0)),
        )

        log_phi_tail = log_phi[..., 1:]
        valid_tail = valid_pos[..., 1:]
        prefix_valid = valid_tail.cumprod(dim=-1)
        prefix_log = (log_phi_tail * valid_tail).cumsum(dim=-1)
        relative_streak = prefix_log.exp() * prefix_valid
        streak_sum = relative_streak.sum(dim=-1).clamp_min(1e-12)
        block_has_streak = (prefix_valid.sum(dim=-1) > 0).float()
        valid_block_count = block_has_streak.sum() + 1e-6
        loss_streak = (-(streak_sum.log()) * block_has_streak).sum() / valid_block_count

        loss = self.ce_weight * loss_ce + self.streak_weight * loss_streak

        # --- Accuracy ---
        with torch.no_grad():
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & (binary_eval_mask > 0.5)
            actual_token_count = binary_eval_mask.sum() + 1e-6
            accuracy = correct.sum().float() / actual_token_count

        return loss, accuracy, loss_streak.detach(), loss_ce.detach()
