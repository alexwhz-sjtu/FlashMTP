# coding=utf-8
"""FlashMTP Training Wrapper."""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from specforge.modeling.draft.flashmtp import FlashMTPDraftModel

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    BlockMask = None
    create_block_mask = None


def prepare_full_sequence_target_hidden(
    hidden_states: Tuple[torch.Tensor, ...],
    target_layer_ids: List[int],
) -> torch.Tensor:
    """Stack teacher layers on the feature axis (feature mode), full sequence length.

    Returns:
        (B, seq_len, num_layers * H) for use with draft ``fc`` -> (B, seq_len, H).
    """
    parts = [hidden_states[lid] for lid in target_layer_ids]
    return torch.cat(parts, dim=-1)


def create_flashmtp_block_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    seq_len: int,
    window_size: int,
    block_size: int,
    device: torch.device,
):
    """BlockMask: Q = noise only; KV = [full target seq | noise blocks].

    Query index ``q_idx`` in ``[0, N * block_size)`` maps to block ``q_idx // block_size``.

    Key index ``kv_idx``:
      - ``kv_idx < seq_len``: positions in the teacher sequence (absolute indices).
      - ``kv_idx >= seq_len``: noise KV for block ``(kv_idx - seq_len) // block_size``.

    Rules:
      - Block ``b`` queries may attend to target positions in
        ``[max(anchor_b - W, 0), anchor_b - 1]`` (inclusive).
      - Same-block noise KV is visible (bidirectional within block).
      - No cross-block noise attention.
      - Invalid blocks: ``block_keep_mask`` False -> no attention.
    """
    B, N = anchor_positions.shape
    W = max(int(window_size), 1)

    def flashmtp_mask_mod(batch, h, q_idx, kv_idx):
        q_blk = q_idx // block_size
        valid_q = block_keep_mask[batch, q_blk]

        in_target = kv_idx < seq_len
        noise_rel = kv_idx - seq_len
        kv_blk = noise_rel // block_size
        same_block_noise = (kv_idx >= seq_len) & (kv_blk == q_blk)

        anchor = anchor_positions[batch, q_blk]
        w_start = torch.clamp(anchor - W, min=0)
        w_end = anchor - 1
        in_window = in_target & (kv_idx >= w_start) & (kv_idx <= w_end)

        return valid_q & (same_block_noise | in_window)

    Q_LEN = N * block_size
    KV_LEN = seq_len + N * block_size

    return create_block_mask(
        flashmtp_mask_mod,
        B=B,
        H=None,
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        device=device,
    )


class OnlineFlashMTPModel(nn.Module):
    """FlashMTP online training wrapper with block-wise CE or KL-to-teacher loss."""

    def __init__(
        self,
        draft_model: FlashMTPDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = None,
        loss_type: str = "ce",
        distill_temperature: float = 2.0,
        kl_topk: int = 10,
        ce_loss_weight: float = 1.0,
        kl_loss_weight: float = 0.0,
        chs_window_size: int = 1,
    ):
        super().__init__()
        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.attention_backend = attention_backend
        self.num_anchors = num_anchors
        self.loss_decay_gamma = loss_decay_gamma
        if loss_type not in ("ce", "kl", "ce_kl"):
            raise ValueError(
                f"loss_type must be 'ce', 'kl', or 'ce_kl', got {loss_type!r}"
            )
        self.loss_type = loss_type
        self.distill_temperature = distill_temperature
        self.kl_topk = kl_topk
        self.ce_loss_weight = float(ce_loss_weight)
        self.kl_loss_weight = float(kl_loss_weight)
        if loss_type == "ce_kl" and self.ce_loss_weight <= 0 and self.kl_loss_weight <= 0:
            raise ValueError(
                "ce_kl requires at least one of ce_loss_weight or kl_loss_weight > 0"
            )
        self.chs_window_size = max(int(chs_window_size), 1)

        self._cached_block_mask: Optional[BlockMask] = None
        self._cached_seq_len: Optional[int] = None
        self._cached_bsz: Optional[int] = None

    def _sample_anchor_positions(
        self,
        seq_len: int,
        loss_mask: torch.Tensor,
        device: torch.device,
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

        indices = torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        masked_indices = torch.where(
            valid, indices, torch.tensor(seq_len + 1, device=device)
        )

        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, torch.tensor(2.0, device=device))

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        anchors = gathered[:, :max_n].sort(dim=1).values

        keep_mask = torch.arange(max_n, device=device).unsqueeze(0) < valid_counts.unsqueeze(
            1
        ).clamp(max=max_n)
        anchors = torch.where(
            keep_mask, anchors, torch.tensor(0, dtype=torch.long, device=device)
        )

        return anchors, keep_mask

    def prepare_noise_input(
        self,
        input_ids: torch.Tensor,
        block_ids: Optional[torch.Tensor] = None,
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
        """Absolute position IDs for draft noise tokens (anchor + offset)."""
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

    @staticmethod
    def _last_target_hidden(
        hidden_states: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    ) -> torch.Tensor:
        """Last tensor in the stack (HF: last transformer layer before lm_head)."""
        if isinstance(hidden_states, torch.Tensor):
            return hidden_states
        if not hidden_states:
            raise ValueError("hidden_states must be non-empty for KL / teacher logits.")
        return hidden_states[-1]

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parallel block-wise training forward pass."""
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )

        noise_embedding = self._create_noise_embed(
            input_ids, anchor_positions, block_keep_mask
        )

        w = self.chs_window_size
        n_anchors = anchor_positions.size(1)

        draft_position_ids = self._create_position_ids(anchor_positions)

        context_position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(
            bsz, -1
        )
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=-1)

        flashmtp_attn_mask = create_flashmtp_block_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            seq_len=seq_len,
            window_size=w,
            block_size=self.block_size,
            device=device,
        )

        target_hidden = prepare_full_sequence_target_hidden(
            hidden_states,
            self.draft_model.target_layer_ids,
        )

        output_hidden = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            attention_mask=flashmtp_attn_mask,
        )

        logits = self.lm_head(output_hidden)

        label_offsets = torch.arange(0, self.block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )

        weight_mask = (
            block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        )
        weight_mask = weight_mask * valid_label_mask.float()

        pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
        weight_mask = weight_mask * (pos_in_block > 0).float()

        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        weight_mask = weight_mask * original_loss_mask_gathered

        binary_eval_mask = weight_mask.view(-1)

        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            k = torch.arange(self.block_size, device=device).view(1, 1, -1)
            decay_weights = torch.exp(
                -(k - 1).clamp(min=0).float() / self.loss_decay_gamma
            )
            weight_mask = weight_mask * decay_weights

        logits_3d = logits.view(bsz, n_anchors, self.block_size, -1)
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = target_ids.view(-1)
        flat_weights = weight_mask.view(-1)

        need_ce = self.loss_type == "ce" or (
            self.loss_type == "ce_kl" and self.ce_loss_weight > 0
        )
        need_kl = self.loss_type == "kl" or (
            self.loss_type == "ce_kl" and self.kl_loss_weight > 0
        )

        loss = flat_logits.new_tensor(0.0)
        if need_ce:
            loss_per_token = F.cross_entropy(
                flat_logits, flat_targets, reduction="none"
            )
            valid_token_count = flat_weights.sum() + 1e-6
            ce_loss = (loss_per_token * flat_weights).sum() / valid_token_count
            if self.loss_type == "ce":
                loss = ce_loss
            else:
                loss = loss + self.ce_loss_weight * ce_loss

        if need_kl:
            context_idx = (safe_label_indices - 1).clamp(min=0)
            h_last = self._last_target_hidden(hidden_states)
            flat_ctx = context_idx.reshape(bsz, -1)
            gather_idx = flat_ctx.unsqueeze(-1).expand(-1, -1, h_last.size(-1))
            teacher_h = torch.gather(h_last, 1, gather_idx)
            teacher_h = teacher_h.view(
                bsz, n_anchors, self.block_size, h_last.size(-1)
            )
            with torch.no_grad():
                teacher_logits = self.lm_head(teacher_h)
            t = self.distill_temperature
            vocab = teacher_logits.size(-1)
            k = self.kl_topk
            if k is not None and k > 0:
                k = min(int(k), vocab)
            use_topk = k is not None and k > 0 and k < vocab

            if use_topk:
                tl = teacher_logits.reshape(-1, vocab)
                sl = logits_3d.reshape(-1, vocab)
                _, top_idx = torch.topk(tl, k=k, dim=-1)
                t_top = tl.gather(1, top_idx)
                s_top = sl.gather(1, top_idx)
                p_t = F.softmax(t_top / t, dim=-1).detach()
                log_p_s = F.log_softmax(s_top / t, dim=-1)
                kl_flat = F.kl_div(
                    log_p_s,
                    p_t,
                    reduction="none",
                    log_target=False,
                ).sum(dim=-1)
                kl_per_tok = kl_flat.view(bsz, n_anchors, self.block_size)
            else:
                log_p_s = F.log_softmax(logits_3d / t, dim=-1)
                p_t = F.softmax(teacher_logits / t, dim=-1).detach()
                kl_per_tok = F.kl_div(
                    log_p_s,
                    p_t,
                    reduction="none",
                    log_target=False,
                ).sum(dim=-1)
            kl_loss = (kl_per_tok * (t * t) * weight_mask).sum() / (
                weight_mask.sum() + 1e-6
            )
            if self.loss_type == "kl":
                loss = kl_loss
            else:
                loss = loss + self.kl_loss_weight * kl_loss

        with torch.no_grad():
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & (binary_eval_mask > 0.5)
            actual_token_count = binary_eval_mask.sum() + 1e-6
            accuracy = correct.sum().float() / actual_token_count

        return loss, accuracy
