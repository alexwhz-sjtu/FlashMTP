# coding=utf-8
"""FlashMTP Training Wrapper."""

import math
from typing import Optional, Tuple

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


def create_dflash_teacher_block_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    seq_len: int,
    block_size: int,
    device: torch.device,
):
    """DFlash teacher mask aligned to the sampled FlashMTP training blocks."""

    def dflash_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        anchor_pos = anchor_positions[b, q_block_id]

        is_context = kv_idx < seq_len
        mask_context = is_context & (kv_idx < anchor_pos)

        is_draft = kv_idx >= seq_len
        kv_block_id = (kv_idx - seq_len) // block_size
        mask_draft = is_draft & (q_block_id == kv_block_id)

        is_valid_block = block_keep_mask[b, q_block_id]
        return (mask_context | mask_draft) & is_valid_block

    bsz, n_blocks = anchor_positions.shape
    return create_block_mask(
        dflash_mask_mod,
        B=bsz,
        H=None,
        Q_LEN=n_blocks * block_size,
        KV_LEN=seq_len + n_blocks * block_size,
        device=device,
    )


def prepare_dflash_teacher_hidden(
    hidden_states: tuple[torch.Tensor] | list[torch.Tensor],
    target_layer_ids: list[int],
    num_transformer_layers: int,
) -> torch.Tensor:
    """Build DFlash full-history hidden condition from FlashMTP target hidden states."""
    off = infer_hidden_states_embedding_offset(hidden_states, num_transformer_layers)
    selected = [hidden_states[layer_id + off] for layer_id in target_layer_ids]
    return torch.cat(selected, dim=-1)


def infer_hidden_states_embedding_offset(
    hidden_states: tuple | list, num_transformer_layers: int
) -> int:
    """Return index offset so that transformer layer k is at hidden_states[k + offset].

    Training HF path uses ``outputs.hidden_states[1:]`` (offset 0). Inference often passes
    the full tuple including embeddings at index 0 (offset 1).
    """
    lt = len(hidden_states)
    if lt == num_transformer_layers:
        return 0
    if lt == num_transformer_layers + 1:
        return 1
    # Fallback: assume embedding prefix if tuple is longer than layer count
    return 1 if lt > num_transformer_layers else 0


def prepare_target_hidden(
    hidden_states: tuple[torch.Tensor],  # tuple of (B, seq_len, H) per layer (+ optional embed)
    anchor_positions: torch.Tensor,  # (B, N)
    target_layer_ids: list[int],
    num_transformer_layers: int,
) -> torch.Tensor:
    """Gather pivot hidden states for all selected transformer layers.

    ``target_layer_ids`` are **0-based transformer layer indices** (shallow=0, deep=L-1).

    Returns:
        (B, N, S, H) with ``S = len(target_layer_ids)``, positions ``anchor-1`` per block.
    """
    context_positions = (anchor_positions - 1).clamp(min=0)  # (B, N)
    off = infer_hidden_states_embedding_offset(hidden_states, num_transformer_layers)
    pieces: list[torch.Tensor] = []
    for layer_id in target_layer_ids:
        layer_hidden = hidden_states[layer_id + off]
        layer_selected = torch.gather(
            layer_hidden,
            dim=1,
            index=context_positions.unsqueeze(-1).expand(
                -1, -1, layer_hidden.size(-1)
            ),
        )
        pieces.append(layer_selected)
    return torch.stack(pieces, dim=2)  # (B, N, S, H)


def create_flashmtp_block_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    chs_len_per_block: int,
    block_size: int,
    device: torch.device,
):
    """Construct Flex Attention BlockMask for FlashMTP training with per-block CHS.

    Args:
        anchor_positions: (B, N) tensor of anchor positions for each block
        block_keep_mask: (B, N) boolean mask indicating valid blocks
        chs_len_per_block: Number of tokens per CHS segment. FlashMTP uses
            feature-concat CHS, so this is 1.
        block_size: Number of tokens per draft block
        device: torch device

    Layout:
        KV: [CHS_0 | CHS_1 | ... | CHS_{N-1} | Block_0 | Block_1 | ... | Block_{N-1}]
            - Each CHS_i has length chs_len_per_block
            - Each Block_i has length block_size
        Q:  [Block_0 | Block_1 | ... | Block_{N-1}]

    Rules:
      1. Block_i only sees CHS_i (its own feature-concat context token).
      2. Intra-block attention is bidirectional.
      3. Different blocks are invisible to each other.
      4. Invalid blocks (block_keep_mask=False) see nothing.
    """

    def flashmtp_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size

        # Total length of all CHS segments
        total_chs_len = N * chs_len_per_block

        # Check if kv_idx falls within the CHS region
        is_context = kv_idx < total_chs_len
        # Which CHS segment this kv belongs to
        chs_block_id = kv_idx // chs_len_per_block
        # Block i only attends to CHS i (all CHS tokens are needed)
        mask_context = is_context & (chs_block_id == q_block_id)

        # Check if kv_idx falls within the draft block region
        is_draft = kv_idx >= total_chs_len
        # Which block this draft kv belongs to
        kv_block_id = (kv_idx - total_chs_len) // block_size
        # Block i only attends to Block i (bidirectional)
        mask_draft = is_draft & (kv_block_id == q_block_id)

        is_valid_block = block_keep_mask[b, q_block_id]
        return (mask_context | mask_draft) & is_valid_block

    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = N * chs_len_per_block + N * block_size

    return create_block_mask(
        flashmtp_mask_mod, B=B, H=None, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device
    )


class OnlineFlashMTPModel(nn.Module):
    """FlashMTP online training wrapper with optional DFlash distillation."""

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
            chs_concat_mode: str = "feature",
            dflash_teacher_model: Optional[nn.Module] = None,
            dflash_distill_weight: float = 1.0,
            dflash_distill_temperature: float = 2.0,
            dflash_distill_top_k: int = 128,
            dflash_distill_pos_mode: str = "prefix",
            dflash_ce_pos_mode: str = "student_wrong",
            dflash_distill_decay_gamma: Optional[float] = None,
            dflash_align_mode: str = "final",
            dflash_mid_align: str = "half",
            dflash_mid_weight: float = 0.0,
            dflash_ce_weight: float = 1.0,
            dflash_milestone_epoch: float = 0.0,
            dflash_distill_min_scale: float = 0.0,
            dflash_ce_min_scale: float = 0.0,
            dflash_total_epochs: int = 1,
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
        self.dflash_teacher_model = dflash_teacher_model
        self.use_dflash_distill = self.dflash_teacher_model is not None
        self.dflash_distill_weight = float(dflash_distill_weight)
        self.dflash_distill_temperature = float(dflash_distill_temperature)
        self.dflash_distill_top_k = int(dflash_distill_top_k)
        self.dflash_distill_pos_mode = dflash_distill_pos_mode
        self.dflash_ce_pos_mode = dflash_ce_pos_mode
        self.dflash_distill_decay_gamma = (
            None
            if dflash_distill_decay_gamma is None
            else float(dflash_distill_decay_gamma)
        )
        self.dflash_align_mode = dflash_align_mode
        self.dflash_mid_align = dflash_mid_align
        self.dflash_mid_weight = float(dflash_mid_weight)
        self.dflash_ce_weight = float(dflash_ce_weight)
        self.dflash_milestone_epoch = float(dflash_milestone_epoch)
        self.dflash_distill_min_scale = min(max(float(dflash_distill_min_scale), 0.0), 1.0)
        self.dflash_ce_min_scale = min(max(float(dflash_ce_min_scale), 0.0), 1.0)
        self.dflash_total_epochs = max(float(dflash_total_epochs), 1.0)
        self.chs_concat_mode = "feature"
        self.draft_model.chs_concat_mode = "feature"

        if self.dflash_align_mode not in ("final", "final+mid"):
            raise ValueError("dflash_align_mode must be one of final, final+mid")
        if self.dflash_mid_align not in ("full", "half"):
            raise ValueError("dflash_mid_align must be one of full, half")
        if self.dflash_distill_pos_mode not in ("prefix", "all"):
            raise ValueError("dflash_distill_pos_mode must be one of prefix, all")
        if self.dflash_ce_pos_mode not in ("prefix", "student_wrong"):
            raise ValueError(
                "dflash_ce_pos_mode must be one of prefix, student_wrong"
            )
        if self.dflash_teacher_model is not None:
            self.dflash_teacher_model.eval()
            for param in self.dflash_teacher_model.parameters():
                param.requires_grad_(False)

        self._cached_block_mask: Optional[BlockMask] = None
        self._cached_seq_len: Optional[int] = None
        self._cached_bsz: Optional[int] = None

    def _sample_anchor_positions(
            self, seq_len: int, loss_mask: torch.Tensor,
            device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample anchor positions per sample; returns (anchors, keep_mask)."""
        bs = self.block_size
        bsz = loss_mask.shape[0]
        max_anchor = max(seq_len - bs, 0)

        valid = loss_mask[:, :max_anchor + 1] > 0.5
        valid_counts = valid.sum(dim=1)
        max_n = min(self.num_anchors, int(valid_counts.max().item()) - 1)

        if max_n <= 0:
            raise ValueError("should preprocess the data.")

        indices = (torch.arange(max_anchor + 1,
                                device=device).unsqueeze(0).expand(bsz, -1))
        masked_indices = torch.where(valid, indices,
                                     torch.tensor(seq_len + 1, device=device))

        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals,
                                  torch.tensor(2.0, device=device))

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        anchors = gathered[:, :max_n].sort(dim=1).values

        keep_mask = torch.arange(
            max_n,
            device=device).unsqueeze(0) < valid_counts.unsqueeze(1).clamp(
                max=max_n)
        anchors = torch.where(keep_mask, anchors,
                              torch.tensor(0, dtype=torch.long, device=device))

        return anchors, keep_mask

    def prepare_noise_input(
            self,
            input_ids: torch.Tensor,
            block_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Prepare noise input: first token of each block is real, rest are MASK."""
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        if block_ids is not None:
            is_block_start = torch.ones(bsz,
                                        seq_len,
                                        dtype=torch.bool,
                                        device=device)
            is_block_start[:, 1:] = block_ids[:, 1:] != block_ids[:, :-1]
        else:
            positions = torch.arange(seq_len, device=device)
            is_block_start = (positions % self.block_size) == 0
            is_block_start = is_block_start.unsqueeze(0).expand(bsz, -1)

        noise_input_ids = torch.full_like(input_ids, self.mask_token_id)
        noise_input_ids[is_block_start] = input_ids[is_block_start]
        return noise_input_ids

    def _create_draft_position_ids(self,
                                   anchor_positions: torch.Tensor) -> torch.Tensor:
        """Draft token position ids: global (anchor + offset) or block-local 1..block_size."""
        bsz, n_blocks = anchor_positions.shape
        device = anchor_positions.device
        bs = self.block_size
        if getattr(self.draft_model, "local_position", False):
            local = torch.arange(1, bs + 1, device=device).view(1, 1, -1).expand(
                bsz, n_blocks, -1
            )
            return local.reshape(bsz, -1)
        offsets = torch.arange(bs, device=device).view(1, 1, -1)
        pos_ids = anchor_positions.unsqueeze(-1) + offsets
        return pos_ids.view(bsz, -1)

    def _create_noise_embed(self, input_ids, anchor_positions,
                            block_keep_mask):
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size
        device = input_ids.device

        noise_ids = torch.full((bsz, n * bs),
                               self.mask_token_id,
                               dtype=torch.long,
                               device=device)

        block_starts = torch.arange(n, device=device) * bs
        block_starts = block_starts.unsqueeze(0).expand(bsz, -1)

        valid_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, valid_anchor_positions)

        flat_batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(
            bsz, n)

        # substitute the anchor position with label token (bonus token in inference)
        noise_ids[flat_batch_idx, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(self.mask_token_id, dtype=torch.long, device=device),
        )

        return self.embed_tokens(noise_ids)

    def _compute_dflash_teacher_outputs(
        self,
        hidden_states: tuple[torch.Tensor] | list[torch.Tensor],
        noise_embedding: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        seq_len: int,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, ...]]]:
        assert self.dflash_teacher_model is not None
        device = noise_embedding.device
        bsz = noise_embedding.shape[0]
        n_blk = anchor_positions.shape[1]

        context_position_ids = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        )
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        draft_position_ids = (anchor_positions.unsqueeze(-1) + offsets).view(
            bsz, -1
        )
        full_position_ids = torch.cat(
            [context_position_ids, draft_position_ids], dim=1
        )
        dflash_attn_mask = create_dflash_teacher_block_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            seq_len=seq_len,
            block_size=self.block_size,
            device=device,
        )
        dflash_target_hidden = prepare_dflash_teacher_hidden(
            hidden_states,
            self.dflash_teacher_model.target_layer_ids,
            self.dflash_teacher_model.config.num_target_layers,
        )
        with torch.no_grad():
            teacher_output = self.dflash_teacher_model(
                position_ids=full_position_ids,
                noise_embedding=noise_embedding,
                target_hidden=dflash_target_hidden,
                attention_mask=dflash_attn_mask,
                output_hidden_states=output_hidden_states,
            )
            if output_hidden_states:
                teacher_hidden, teacher_layer_hidden = teacher_output
            else:
                teacher_hidden = teacher_output
                teacher_layer_hidden = None
            teacher_logits = self.lm_head(teacher_hidden)
        return (
            teacher_logits.view(bsz, n_blk, self.block_size, -1),
            teacher_layer_hidden,
        )

    def _compute_dflash_teacher_logits(
        self,
        hidden_states: tuple[torch.Tensor] | list[torch.Tensor],
        noise_embedding: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        teacher_logits, _ = self._compute_dflash_teacher_outputs(
            hidden_states=hidden_states,
            noise_embedding=noise_embedding,
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            seq_len=seq_len,
            output_hidden_states=False,
        )
        return teacher_logits

    def _compute_dflash_distill_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        target_ids: torch.Tensor,
        distill_weight_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-k KL from DFlash to FlashMTP on teacher-correct slots."""
        device = student_logits.device
        zero = torch.zeros((), device=device, dtype=student_logits.dtype)
        valid = distill_weight_mask > 0
        if not valid.any():
            return zero, zero

        vocab_size = teacher_logits.size(-1)
        top_k = min(max(self.dflash_distill_top_k, 1), vocab_size)
        tau = max(self.dflash_distill_temperature, 1e-6)

        flat_teacher = teacher_logits.reshape(-1, vocab_size)
        flat_student = student_logits.reshape(-1, vocab_size)
        flat_targets = target_ids.reshape(-1)
        flat_weights = distill_weight_mask.reshape(-1).float()
        active_mask = self._compute_dflash_distill_active_mask(
            teacher_logits, target_ids, distill_weight_mask
        )
        active_indices = active_mask.reshape(-1).nonzero(
            as_tuple=False
        ).squeeze(-1)

        if active_indices.numel() == 0:
            return zero, zero

        active_targets = flat_targets[active_indices]
        top_n = min(top_k + 1, vocab_size)
        teacher_topk_all = torch.topk(flat_teacher, k=top_n, dim=-1).indices
        teacher_topk = teacher_topk_all[active_indices]
        in_topk = (teacher_topk == active_targets.unsqueeze(-1)).any(dim=-1)
        candidate_ids = teacher_topk.clone()
        candidate_ids[~in_topk, -1] = active_targets[~in_topk]

        row_ids = active_indices.unsqueeze(-1).expand_as(candidate_ids)
        teacher_selected = flat_teacher[row_ids, candidate_ids].float()
        student_selected = flat_student[row_ids, candidate_ids].float()

        teacher_log_probs = F.log_softmax(teacher_selected / tau, dim=-1)
        teacher_probs = teacher_log_probs.exp()
        student_log_probs = F.log_softmax(student_selected / tau, dim=-1)
        kl_per_slot = (
            teacher_probs * (teacher_log_probs - student_log_probs)
        ).sum(dim=-1) * (tau ** 2)

        active_weights = flat_weights[active_indices]
        kl_loss = (kl_per_slot * active_weights).sum() / (
            active_weights.sum() + 1e-6
        )
        active_ratio = active_indices.numel() / (
            valid.reshape(-1).float().sum() + 1e-6
        )
        return kl_loss.to(student_logits.dtype), active_ratio.to(student_logits.dtype)

    def _compute_dflash_distill_active_mask(
        self,
        teacher_logits: torch.Tensor,
        target_ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.dflash_distill_pos_mode == "all":
            return valid_mask > 0
        return self._compute_dflash_prefix_correct_mask(
            teacher_logits=teacher_logits,
            target_ids=target_ids,
            valid_mask=valid_mask,
        )

    def _compute_dflash_correct_mask(
        self,
        teacher_logits: torch.Tensor,
        target_ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Slots where DFlash teacher top1 matches the true label."""
        teacher_top1 = teacher_logits.argmax(dim=-1)
        return (teacher_top1 == target_ids) & (valid_mask > 0)

    def _compute_dflash_prefix_correct_mask(
        self,
        teacher_logits: torch.Tensor,
        target_ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Continuous teacher-correct prefix inside each block, excluding anchor."""
        correct = self._compute_dflash_correct_mask(
            teacher_logits=teacher_logits,
            target_ids=target_ids,
            valid_mask=valid_mask,
        )
        prefix_correct = torch.zeros_like(correct, dtype=torch.bool)
        if correct.size(-1) > 1:
            prefix_correct[..., 1:] = correct[..., 1:].float().cumprod(
                dim=-1
            ).bool()
        return prefix_correct

    def _compute_ce_prefix_mask(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        target_ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher prefix plus one student frontier error once student catches up."""
        teacher_prefix = self._compute_dflash_prefix_correct_mask(
            teacher_logits=teacher_logits,
            target_ids=target_ids,
            valid_mask=valid_mask,
        )
        student_prefix = self._compute_dflash_prefix_correct_mask(
            teacher_logits=student_logits,
            target_ids=target_ids,
            valid_mask=valid_mask,
        )

        if target_ids.size(-1) <= 1:
            return teacher_prefix

        teacher_len = teacher_prefix[..., 1:].sum(dim=-1)
        student_len = student_prefix[..., 1:].sum(dim=-1)
        frontier_pos = student_len + 1
        eligible = (student_len >= teacher_len) & (frontier_pos < target_ids.size(-1))

        extra_mask = torch.zeros_like(teacher_prefix, dtype=torch.bool)
        if eligible.any():
            b_idx, n_idx = eligible.nonzero(as_tuple=True)
            pos_idx = frontier_pos[b_idx, n_idx]
            extra_mask[b_idx, n_idx, pos_idx] = True
            extra_mask = extra_mask & (valid_mask > 0)

        return teacher_prefix | extra_mask

    def _compute_ce_student_wrong_mask(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        target_ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Slots where teacher top1 is correct but student top1 is wrong."""
        teacher_correct = self._compute_dflash_correct_mask(
            teacher_logits=teacher_logits,
            target_ids=target_ids,
            valid_mask=valid_mask,
        )
        student_wrong = student_logits.argmax(dim=-1) != target_ids
        return teacher_correct & student_wrong

    def _compute_dflash_mid_loss(
        self,
        student_layer_hidden: tuple[torch.Tensor, ...],
        teacher_layer_hidden: tuple[torch.Tensor, ...],
        weight_mask: torch.Tensor,
        bsz: int,
        n_blk: int,
    ) -> torch.Tensor:
        """Norm-hidden MSE on distill-selected slots."""
        device = weight_mask.device
        zero = torch.zeros((), device=device, dtype=student_layer_hidden[-1].dtype)
        n_layers = min(len(student_layer_hidden), len(teacher_layer_hidden))
        if n_layers <= 1:
            return zero

        if self.dflash_mid_align == "full":
            layer_ids = list(range(n_layers - 1))
        else:
            layer_ids = [min(n_layers // 2, n_layers - 2)]

        flat_weights = weight_mask.reshape(-1).float()
        weight_sum = flat_weights.sum()
        if weight_sum <= 0:
            return zero

        losses = []
        for layer_id in layer_ids:
            student_h = student_layer_hidden[layer_id].view(
                bsz, n_blk, self.block_size, -1
            )
            teacher_h = teacher_layer_hidden[layer_id].view(
                bsz, n_blk, self.block_size, -1
            )
            student_norm = F.normalize(student_h.float(), p=2, dim=-1, eps=1e-6)
            teacher_norm = F.normalize(teacher_h.float(), p=2, dim=-1, eps=1e-6)
            mse_per_slot = (student_norm - teacher_norm).pow(2).mean(dim=-1)
            losses.append(
                (mse_per_slot.reshape(-1) * flat_weights).sum()
                / (weight_sum + 1e-6)
            )

        return torch.stack(losses).mean().to(student_layer_hidden[-1].dtype)

    def _dflash_cosine_coefficients(
        self,
        current_epoch: Optional[float],
    ) -> tuple[float, float, float]:
        if not self.use_dflash_distill:
            return 0.0, 0.0, 1.0

        epoch_value = 0.0 if current_epoch is None else float(current_epoch)
        if epoch_value < self.dflash_milestone_epoch:
            mid_weight = self.dflash_mid_weight if self.dflash_align_mode == "final+mid" else 0.0
            return (
                self.dflash_distill_weight,
                mid_weight,
                self.dflash_ce_weight * self.dflash_ce_min_scale,
            )

        denom = max(self.dflash_total_epochs - self.dflash_milestone_epoch, 1e-6)
        t = min(max((epoch_value - self.dflash_milestone_epoch) / denom, 0.0), 1.0)
        cosine_scale = 0.5 * (1.0 + math.cos(math.pi * t))
        distill_scale = self.dflash_distill_min_scale + (
            1.0 - self.dflash_distill_min_scale
        ) * cosine_scale
        ce_scale = self.dflash_ce_min_scale + (
            1.0 - self.dflash_ce_min_scale
        ) * (1.0 - cosine_scale)
        mid_weight = self.dflash_mid_weight if self.dflash_align_mode == "final+mid" else 0.0
        return (
            self.dflash_distill_weight * distill_scale,
            mid_weight * distill_scale,
            self.dflash_ce_weight * ce_scale,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: tuple,
        loss_mask: torch.Tensor,
        current_epoch: Optional[float] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Parallel block-wise training forward pass."""
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # TODO: keep_mask meaning: Valid anchor position
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device)

        noise_embedding = self._create_noise_embed(input_ids, anchor_positions,
                                                   block_keep_mask)

        # Non-local: CHS rotary ids follow anchor-1 (gather index for pivot hs).
        # local_position: CHS rotary ids are 0; draft ids are block-local 1..block_size.
        draft_position_ids = self._create_draft_position_ids(anchor_positions)

        chs = self.draft_model.chs_len_per_block
        bsz, n_blk = anchor_positions.shape
        if getattr(self.draft_model, "local_position", False):
            ctx_pos_flat = torch.zeros(
                bsz, n_blk * chs, device=device, dtype=torch.long
            )
        else:
            ctx_base = (anchor_positions - 1).clamp(min=0)
            ctx_pos_flat = ctx_base.unsqueeze(-1).expand(bsz, n_blk, chs).reshape(
                bsz, n_blk * chs
            )
        full_rotary_position_ids = torch.cat(
            [ctx_pos_flat, draft_position_ids], dim=-1
        )

        flashmtp_attn_mask = create_flashmtp_block_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            chs_len_per_block=chs,
            block_size=self.block_size,
            device=device,
        )

        target_hidden = prepare_target_hidden(
            hidden_states,
            anchor_positions,
            self.draft_model.target_layer_ids,
            self.draft_model.config.num_target_layers,
        )

        need_mid_hidden = (
            self.dflash_align_mode == "final+mid"
            and self.dflash_mid_weight > 0
            and self.use_dflash_distill
        )
        draft_output = self.draft_model(
            position_ids=draft_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            attention_mask=flashmtp_attn_mask,
            rotary_position_ids=full_rotary_position_ids,
            output_hidden_states=need_mid_hidden,
        )
        if need_mid_hidden:
            output_hidden, student_layer_hidden = draft_output
        else:
            output_hidden = draft_output
            student_layer_hidden = None

        logits = self.lm_head(output_hidden)

        # --- Labels: same-position prediction (position k predicts token anchor+k) ---
        label_offsets = torch.arange(0, self.block_size,
                                     device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )

        # --- Weight mask: block validity * bounds * exclude anchor (pos 0) * loss_mask ---
        weight_mask = (block_keep_mask.unsqueeze(-1).expand(
            -1, -1, self.block_size).float())
        weight_mask = weight_mask * valid_label_mask.float()

        pos_in_block = torch.arange(self.block_size,
                                    device=device).view(1, 1, -1)
        weight_mask = weight_mask * (pos_in_block > 0).float()

        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        weight_mask = weight_mask * original_loss_mask_gathered

        binary_eval_mask = weight_mask.view(-1)

        distill_weight_mask = weight_mask.clone()

        # --- CE loss decay: exp(-(k-1)/γ) so k=1 (1st prediction) gets weight 1.0 ---
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            k = torch.arange(self.block_size, device=device).view(1, 1, -1)
            decay_weights = torch.exp(-(k - 1).clamp(min=0).float() /
                                      self.loss_decay_gamma)
            weight_mask = weight_mask * decay_weights

        # --- KL/mid loss decay: disabled by default; independent from CE decay ---
        if (
            self.dflash_distill_decay_gamma is not None
            and self.dflash_distill_decay_gamma > 0
        ):
            k = torch.arange(self.block_size, device=device).view(1, 1, -1)
            decay_weights = torch.exp(
                -(k - 1).clamp(min=0).float()
                / self.dflash_distill_decay_gamma
            )
            distill_weight_mask = distill_weight_mask * decay_weights

        teacher_logits = None
        teacher_layer_hidden = None
        needs_dflash_teacher = (
            self.use_dflash_distill
            and (
                self.dflash_distill_weight > 0
                or need_mid_hidden
                or self.dflash_ce_pos_mode in ("prefix", "student_wrong")
            )
        )
        if needs_dflash_teacher:
            teacher_logits, teacher_layer_hidden = self._compute_dflash_teacher_outputs(
                hidden_states=hidden_states,
                noise_embedding=noise_embedding,
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                seq_len=seq_len,
                output_hidden_states=need_mid_hidden,
            )

        if self.use_dflash_distill and self.dflash_ce_pos_mode == "prefix":
            ce_prefix_mask = self._compute_ce_prefix_mask(
                student_logits=logits.view(
                    bsz, anchor_positions.size(1), self.block_size, -1
                ),
                teacher_logits=teacher_logits,
                target_ids=target_ids,
                valid_mask=weight_mask,
            )
            weight_mask = weight_mask * ce_prefix_mask.float()
        elif self.use_dflash_distill and self.dflash_ce_pos_mode == "student_wrong":
            ce_student_wrong_mask = self._compute_ce_student_wrong_mask(
                student_logits=logits.view(
                    bsz, anchor_positions.size(1), self.block_size, -1
                ),
                teacher_logits=teacher_logits,
                target_ids=target_ids,
                valid_mask=weight_mask,
            )
            weight_mask = weight_mask * ce_student_wrong_mask.float()

        # --- Cross entropy ---
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = target_ids.view(-1)
        flat_weights = weight_mask.view(-1)

        loss_per_token = F.cross_entropy(flat_logits,
                                         flat_targets,
                                         reduction="none")
        valid_token_count = flat_weights.sum() + 1e-6
        ce_loss = (loss_per_token * flat_weights).sum() / valid_token_count
        kl_weight, mid_weight, ce_weight = self._dflash_cosine_coefficients(
            current_epoch
        )
        if self.use_dflash_distill:
            loss = ce_loss * ce_weight
        else:
            loss = ce_loss

        dflash_kl_loss = torch.zeros((), device=device, dtype=loss.dtype)
        dflash_kl_active_ratio = torch.zeros((), device=device, dtype=loss.dtype)
        dflash_mid_loss = torch.zeros((), device=device, dtype=loss.dtype)
        if self.use_dflash_distill and self.dflash_distill_weight > 0:
            if teacher_logits is None:
                teacher_logits = self._compute_dflash_teacher_logits(
                    hidden_states=hidden_states,
                    noise_embedding=noise_embedding,
                    anchor_positions=anchor_positions,
                    block_keep_mask=block_keep_mask,
                    seq_len=seq_len,
                )
            dflash_kl_loss, dflash_kl_active_ratio = self._compute_dflash_distill_loss(
                student_logits=logits.view(
                    bsz, anchor_positions.size(1), self.block_size, -1
                ),
                teacher_logits=teacher_logits,
                target_ids=target_ids,
                distill_weight_mask=distill_weight_mask,
            )
            loss = loss + kl_weight * dflash_kl_loss

        if need_mid_hidden and mid_weight > 0:
            if teacher_logits is None or teacher_layer_hidden is None:
                teacher_logits, teacher_layer_hidden = self._compute_dflash_teacher_outputs(
                    hidden_states=hidden_states,
                    noise_embedding=noise_embedding,
                    anchor_positions=anchor_positions,
                    block_keep_mask=block_keep_mask,
                    seq_len=seq_len,
                    output_hidden_states=True,
                )
            dflash_active_mask = self._compute_dflash_distill_active_mask(
                teacher_logits, target_ids, distill_weight_mask
            )
            dflash_mid_weight_mask = (
                distill_weight_mask * dflash_active_mask.float()
            )
            dflash_mid_loss = self._compute_dflash_mid_loss(
                student_layer_hidden=student_layer_hidden,
                teacher_layer_hidden=teacher_layer_hidden,
                weight_mask=dflash_mid_weight_mask,
                bsz=bsz,
                n_blk=anchor_positions.size(1),
            )
            loss = loss + mid_weight * dflash_mid_loss

        # --- Accuracy ---
        with torch.no_grad():
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & (binary_eval_mask > 0.5)
            actual_token_count = binary_eval_mask.sum() + 1e-6
            accuracy = correct.sum().float() / actual_token_count

            # --- prefix metric (aligned with FlashMTP_exp): mean per-block acceptance length ---
            # cumprod only on in-block indices 1: (exclude anchor); +1.0; average over blocks
            # that are kept and have at least one valid speculative position.
            pred_ids_by_block = logits.argmax(dim=-1).view(
                bsz, anchor_positions.size(1), self.block_size
            )
            correct_by_block = pred_ids_by_block == target_ids
            valid_by_block = binary_eval_mask.view(
                bsz, anchor_positions.size(1), self.block_size
            ) > 0.5
            prefix_correct = (
                correct_by_block[:, :, 1:] & valid_by_block[:, :, 1:]
            ).cumprod(dim=-1)
            prefix_lengths = prefix_correct.sum(dim=-1).float() + 1.0
            valid_blocks = block_keep_mask & valid_by_block[:, :, 1:].any(dim=-1)
            prefix_count = valid_blocks.sum().float()
            prefix_sum = (
                prefix_lengths[valid_blocks].sum()
                if valid_blocks.any()
                else torch.zeros((), device=device, dtype=torch.float32)
            )
            prefix_acc = prefix_sum / prefix_count.clamp(min=1.0)

        return (
            loss,
            accuracy,
            prefix_acc,
            ce_loss,
            dflash_kl_loss,
            dflash_kl_active_ratio,
            dflash_mid_loss,
            torch.tensor(kl_weight, device=device, dtype=loss.dtype),
            torch.tensor(mid_weight, device=device, dtype=loss.dtype),
            torch.tensor(ce_weight, device=device, dtype=loss.dtype),
        )
