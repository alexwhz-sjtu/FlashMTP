# coding=utf-8
"""FlashMTP Training Wrapper."""

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

def prepare_target_hidden(
    hidden_states: tuple[torch.Tensor],  # (num_layers,)[(B, seq_len, H)]
    anchor_positions: torch.Tensor,  # (B, N)
    target_layer_ids: list[int],
    chs_concat_mode: str = "feature",
) -> torch.Tensor:
    """Convert full hidden states to CHS format for FlashMTP.

    Args:
        hidden_states: All layers' hidden states from target model
        anchor_positions: Anchor positions for each block
        target_layer_ids: List of layer IDs to extract
        chs_concat_mode: "feature"

    Returns:
        - feature mode: (B, N, H*L) - L layers concatenated along feature dim
    """
    if chs_concat_mode != "feature":
        raise ValueError("FlashMTP v6 only supports chs_concat_mode='feature'.")
    # 获取位置 p-1 的 hidden states (用来预测位置 p)
    context_positions = (anchor_positions - 1).clamp(min=0)  # (B, N)

    # 提取 anchor positions 对应的 hidden states
    # hidden_states[layer] shape: (B, seq_len, H)
    selected_states = []
    use_captured_order = len(hidden_states) == len(target_layer_ids)
    for capture_idx, layer_id in enumerate(target_layer_ids):
        layer_hidden = hidden_states[capture_idx if use_captured_order else layer_id]
        # Gather: (B, N, H)
        layer_selected = torch.gather(
            layer_hidden,
            dim=1,
            index=context_positions.unsqueeze(-1).expand(-1, -1, layer_hidden.size(-1))
        )
        selected_states.append(layer_selected)

    # 按特征维度拼接: (B, N, H*L)
    return torch.cat(selected_states, dim=-1)  # (B, N, H*L)

def _stage_id_from_position(pos, stage_ranges):
    stage_id = torch.zeros_like(pos)
    for idx, (start, _end) in enumerate(stage_ranges):
        stage_id = torch.where(pos >= start, torch.full_like(pos, idx), stage_id)
    return stage_id


def create_flashmtp_block_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    chs_len_per_block: int,
    block_size: int,
    stage_ranges: list[tuple[int, int]],
    device: torch.device,
):
    """Construct Flex Attention BlockMask for FlashMTP v6 progressive training.

    Args:
        anchor_positions: (B, N) tensor of anchor positions for each block
        block_keep_mask: (B, N) boolean mask indicating valid blocks
        chs_len_per_block: Number of tokens per CHS segment
            - For seq concat mode: num_target_layers (L)
            - For feature concat mode: 1
        block_size: Number of tokens per draft block
        device: torch device

    Layout:
        KV: [CHS_0 | CHS_1 | ... | CHS_{N-1} | Block_0 | Block_1 | ... | Block_{N-1}]
            - Each CHS_i has length chs_len_per_block
            - Each Block_i has length block_size
        Q:  [Block_0 | Block_1 | ... | Block_{N-1}]

    Rules:
      1. Block_i only sees CHS_i (its own pivot context).
      2. Within a draft block, a query sees previous stages and its own stage.
      3. Tokens inside the same stage are bidirectionally visible.
      4. Invalid blocks (block_keep_mask=False) see nothing.
    """

    def flashmtp_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        q_pos = q_idx % block_size
        q_stage = _stage_id_from_position(q_pos, stage_ranges)

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
        kv_pos = (kv_idx - total_chs_len) % block_size
        kv_stage = _stage_id_from_position(kv_pos, stage_ranges)
        mask_draft = is_draft & (kv_block_id == q_block_id) & (kv_stage <= q_stage)

        is_valid_block = block_keep_mask[b, q_block_id]
        return (mask_context | mask_draft) & is_valid_block

    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = N * chs_len_per_block + N * block_size

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
        del target_lm_head
        self.embed_tokens = target_embed_tokens
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.attention_backend = attention_backend
        self.num_anchors = num_anchors
        self.loss_decay_gamma = loss_decay_gamma
        if chs_concat_mode != "feature":
            raise ValueError("FlashMTP v6 only supports chs_concat_mode='feature'.")
        self.chs_concat_mode = chs_concat_mode
        self.draft_model.chs_concat_mode = chs_concat_mode
        self.stage_ranges = draft_model.stage_ranges

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

    def _create_position_ids(self,
                             anchor_positions: torch.Tensor) -> torch.Tensor:
        """Create absolute position IDs for parallel draft blocks."""
        bsz, n_blocks = anchor_positions.shape
        device = anchor_positions.device
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
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

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parallel block-wise training forward pass."""
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # TODO: keep_mask meaning: Valid anchor position
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device)

        noise_embedding = self._create_noise_embed(input_ids, anchor_positions,
                                                   block_keep_mask)

        draft_position_ids = self._create_position_ids(
            anchor_positions)  # (bsz, n_blocks * block_size)

        full_position_ids = draft_position_ids

        chs_len_per_block = 1

        flashmtp_attn_mask = create_flashmtp_block_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            chs_len_per_block=chs_len_per_block,
            block_size=self.block_size,
            stage_ranges=self.stage_ranges,
            device=device,
        )

        # only use the hidden states from the target model at anchor positions (CHS) as input to the draft model
        target_hidden = prepare_target_hidden(
            hidden_states, anchor_positions, self.draft_model.target_layer_ids,
            self.chs_concat_mode)

        # print(f"target_hidden shape after prepare: {target_hidden.shape}")
        # print(f"full_position_ids shape: {full_position_ids.shape}")
        # print(f"noise_embedding shape: {noise_embedding.shape}")

        draft_output = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            anchor_position=anchor_positions,
            attention_mask=flashmtp_attn_mask,
        )
        logits = draft_output["logits"]

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

        # --- Multi-stage CE: each draft layer supervises only its assigned block slice. ---
        logits_by_block = logits.view(
            bsz, anchor_positions.size(1), self.block_size, logits.size(-1)
        )
        loss = logits.sum() * 0.0
        for stage_idx, (start, end) in enumerate(self.stage_ranges):
            stage_logits = logits_by_block[:, :, start:end, :].reshape(
                -1, logits.size(-1)
            )
            stage_targets = target_ids[:, :, start:end].reshape(-1)
            stage_weights = weight_mask[:, :, start:end].reshape(-1)
            stage_token_count = stage_weights.sum()
            if stage_token_count.item() <= 0:
                continue

            stage_ce = F.cross_entropy(
                stage_logits, stage_targets, reduction="none"
            )
            stage_loss = (stage_ce * stage_weights).sum() / (stage_token_count + 1e-6)
            if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
                stage_weight = torch.exp(
                    torch.tensor(
                        -stage_idx / self.loss_decay_gamma,
                        device=device,
                        dtype=stage_loss.dtype,
                    )
                )
            else:
                stage_weight = torch.ones((), device=device, dtype=stage_loss.dtype)
            loss = loss + stage_weight * stage_loss

        # --- Accuracy ---
        with torch.no_grad():
            pred_ids = torch.argmax(logits_by_block, dim=-1)
            correct = (pred_ids == target_ids) & (weight_mask > 0.5)
            actual_token_count = weight_mask.sum() + 1e-6
            accuracy = correct.sum().float() / actual_token_count

        return loss, accuracy
