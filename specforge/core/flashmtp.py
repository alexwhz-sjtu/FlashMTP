# coding=utf-8
"""FlashMTP Training Wrapper."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from specforge.modeling.draft.flashmtp import FlashMTPDraftModel, flashmtp_slot_group

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    BlockMask = None
    create_block_mask = None

def prepare_target_hidden(
    hidden_states: tuple[torch.Tensor],  # (num_layers,)[(B, seq_len, H)]
    anchor_positions: torch.Tensor,  # (B, N)
    target_layer_ids: list[int],
) -> torch.Tensor:
    """Convert full hidden states to feature-concat CHS format for FlashMTP.

    Args:
        hidden_states: All layers' hidden states from target model
        anchor_positions: Anchor positions for each block
        target_layer_ids: List of layer IDs to extract

    Returns:
        (B, N, H*L) - L layers concatenated along feature dim
    """
    # 获取位置 p-1 的 hidden states (用来预测位置 p)
    context_positions = (anchor_positions - 1).clamp(min=0)  # (B, N)

    # 提取 anchor positions 对应的 hidden states
    # hidden_states[layer] shape: (B, seq_len, H)
    selected_states = []
    for layer_id in target_layer_ids:
        layer_hidden = hidden_states[layer_id]  # (B, seq_len, H)
        # Gather: (B, N, H)
        layer_selected = torch.gather(
            layer_hidden,
            dim=1,
            index=context_positions.unsqueeze(-1).expand(-1, -1, layer_hidden.size(-1))
        )
        selected_states.append(layer_selected)

    return torch.cat(selected_states, dim=-1)  # (B, N, H*L)

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
        KV: [CHS_0 | ... | CHS_{N-1} | Clean_0, Mask_0 | Clean_1, Mask_1 | ...]
            - Each CHS_i has length chs_len_per_block
            - Each Clean_i and Mask_i has length block_size
        Q:  [Clean_0, Mask_0 | Clean_1, Mask_1 | ...]

    Rules:
      1. Block_i only sees CHS_i (its own feature-concat context token).
      2. Clean queries only see previous/current clean semantic groups.
      3. Mask queries see previous clean groups and the current mask group.
      4. Tokens inside the same predicted mask group are bidirectional.
      5. Different sampled training blocks are invisible to each other.
      6. Invalid blocks (block_keep_mask=False) see nothing.
    """

    def flashmtp_mask_mod(b, h, q_idx, kv_idx):
        stream_block_size = 2 * block_size
        q_block_id = q_idx // stream_block_size
        q_stream_offset = q_idx % stream_block_size
        q_is_mask = q_stream_offset >= block_size
        q_slot = q_stream_offset % block_size
        q_group = flashmtp_slot_group(q_slot)

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
        draft_kv_idx = kv_idx - total_chs_len
        kv_block_id = draft_kv_idx // stream_block_size
        kv_stream_offset = draft_kv_idx % stream_block_size
        kv_is_mask = kv_stream_offset >= block_size
        kv_slot = kv_stream_offset % block_size
        kv_group = flashmtp_slot_group(kv_slot)

        same_block = kv_block_id == q_block_id
        clean_query_visible = (~q_is_mask) & (~kv_is_mask) & (kv_group <= q_group)
        mask_query_visible = q_is_mask & (
            ((~kv_is_mask) & (kv_group < q_group))
            | (kv_is_mask & (kv_group == q_group))
        )
        mask_draft = is_draft & same_block & (clean_query_visible | mask_query_visible)

        is_valid_block = block_keep_mask[b, q_block_id]
        return (mask_context | mask_draft) & is_valid_block

    B, N = anchor_positions.shape
    Q_LEN = N * 2 * block_size
    KV_LEN = N * chs_len_per_block + N * 2 * block_size

    return create_block_mask(
        flashmtp_mask_mod, B=B, H=None, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device
    )


class OnlineFlashMTPModel(nn.Module):
    """FlashMTP online training wrapper with block-wise CE loss."""

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
        self.chs_concat_mode = "feature"
        self.draft_model.chs_concat_mode = "feature"

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

    def compute_valid_token_count(
            self,
            seq_len: int,
            loss_mask: torch.Tensor,
            anchor_positions: torch.Tensor,
            block_keep_mask: torch.Tensor) -> torch.Tensor:
        """Return the CE denominator for a set of anchors without building logits."""
        device = loss_mask.device
        label_offsets = torch.arange(0, self.block_size,
                                     device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

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

        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            k = torch.arange(self.block_size, device=device).view(1, 1, -1)
            decay_weights = torch.exp(-(k - 1).clamp(min=0).float() /
                                      self.loss_decay_gamma)
            weight_mask = weight_mask * decay_weights

        return weight_mask.view(-1).sum() + 1e-6

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

    def _create_position_ids(
            self,
            anchor_positions: torch.Tensor,
            repeat_streams: int = 1) -> torch.Tensor:
        """Create absolute position IDs for parallel draft blocks."""
        bsz, n_blocks = anchor_positions.shape
        device = anchor_positions.device
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        pos_ids = anchor_positions.unsqueeze(-1) + offsets
        if repeat_streams > 1:
            pos_ids = pos_ids.unsqueeze(2).expand(
                -1, -1, repeat_streams, -1
            ).reshape(bsz, n_blocks, repeat_streams * self.block_size)
        return pos_ids.view(bsz, -1)

    def _create_noise_embed(self, input_ids, anchor_positions,
                            block_keep_mask):
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size
        device = input_ids.device

        offsets = torch.arange(bs, device=device).view(1, 1, -1)
        clean_indices = anchor_positions.unsqueeze(-1) + offsets
        valid_clean_mask = clean_indices < seq_len
        safe_clean_indices = clean_indices.clamp(max=seq_len - 1)

        clean_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, n, -1),
            2,
            safe_clean_indices,
        )
        clean_ids = torch.where(
            valid_clean_mask & block_keep_mask.unsqueeze(-1),
            clean_ids,
            torch.full_like(clean_ids, self.mask_token_id),
        )

        mask_ids = torch.full_like(clean_ids, self.mask_token_id)
        draft_ids = torch.cat([clean_ids, mask_ids], dim=-1).reshape(
            bsz, n * 2 * bs
        )

        return self.embed_tokens(draft_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        anchor_positions: Optional[torch.Tensor] = None,
        block_keep_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Parallel block-wise training forward pass."""
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        if anchor_positions is None or block_keep_mask is None:
            anchor_positions, block_keep_mask = self._sample_anchor_positions(
                seq_len, loss_mask, device)

        noise_embedding = self._create_noise_embed(input_ids, anchor_positions,
                                                   block_keep_mask)

        # CHS uses the target hidden at anchor-1, so its RoPE position is anchor-1.
        context_position_ids = (anchor_positions - 1).clamp(min=0)  # (bsz, n_blocks)

        draft_position_ids = self._create_position_ids(
            anchor_positions,
            repeat_streams=2,
        )  # (bsz, n_blocks * 2 * block_size)

        full_position_ids = torch.cat(
            [context_position_ids, draft_position_ids],
            dim=-1,
        )  # (bsz, n_blocks + n_blocks * 2 * block_size)

        flashmtp_attn_mask = create_flashmtp_block_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            chs_len_per_block=1,
            block_size=self.block_size,
            device=device,
        )

        # only use the hidden states from the target model at anchor positions (CHS) as input to the draft model
        target_hidden = prepare_target_hidden(
            hidden_states, anchor_positions, self.draft_model.target_layer_ids)

        # print(f"target_hidden shape after prepare: {target_hidden.shape}")
        # print(f"full_position_ids shape: {full_position_ids.shape}")
        # print(f"noise_embedding shape: {noise_embedding.shape}")

        output_hidden = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            attention_mask=flashmtp_attn_mask,
        )

        stream_hidden = output_hidden.reshape(
            bsz, anchor_positions.size(1), 2, self.block_size, -1
        )
        mask_hidden = stream_hidden[:, :, 1, :, :].reshape(
            bsz, anchor_positions.size(1) * self.block_size, -1
        )
        logits = self.lm_head(mask_hidden)

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

        # --- Loss decay: exp(-(k-1)/γ) so k=1 (1st prediction) gets weight 1.0 ---
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            k = torch.arange(self.block_size, device=device).view(1, 1, -1)
            decay_weights = torch.exp(-(k - 1).clamp(min=0).float() /
                                      self.loss_decay_gamma)
            weight_mask = weight_mask * decay_weights

        # --- Cross entropy ---
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = target_ids.view(-1)
        flat_weights = weight_mask.view(-1)

        loss_per_token = F.cross_entropy(flat_logits,
                                         flat_targets,
                                         reduction="none")
        valid_token_count = flat_weights.sum() + 1e-6
        loss = (loss_per_token * flat_weights).sum() / valid_token_count

        # --- Accuracy ---
        with torch.no_grad():
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & (binary_eval_mask > 0.5)
            actual_token_count = binary_eval_mask.sum() + 1e-6
            correct_count = correct.sum().float()
            accuracy = correct.sum().float() / actual_token_count

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
            prefix_denominators = valid_by_block[:, :, 1:].sum(dim=-1).float() + 1.0
            prefix_ratios = prefix_lengths / prefix_denominators.clamp(min=1.0)
            prefix_count = valid_blocks.sum().float()
            prefix_sum = (
                prefix_ratios[valid_blocks].sum()
                if valid_blocks.any()
                else torch.zeros((), device=device)
            )
            prefix_accuracy = prefix_sum / prefix_count.clamp(min=1.0)

        metrics = {
            "prefix_accuracy": prefix_accuracy.detach(),
            "loss_numerator": (loss_per_token * flat_weights).sum().detach(),
            "valid_token_count": valid_token_count.detach(),
            "correct_count": correct_count.detach(),
            "actual_token_count": actual_token_count.detach(),
            "prefix_sum": prefix_sum.detach(),
            "prefix_count": prefix_count.detach(),
        }

        return loss, accuracy, metrics
