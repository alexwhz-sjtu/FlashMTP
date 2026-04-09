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
) -> torch.Tensor:
    """Extract anchor position hidden states for FlashMTP v2 feature injection.

    In v2, the hidden states at anchor_position-1 (from all target layers)
    are concatenated along feature dimension and will be injected into each
    noise embedding position after FC projection.

    Args:
        hidden_states: All layers' hidden states from target model
        anchor_positions: Anchor positions for each block (sorted, increasing)
        target_layer_ids: List of layer IDs to extract

    Returns:
        (B, N, H*L) - L layers concatenated along feature dimension
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

    # v2 only uses feature concat mode
    # 按特征维度拼接: (B, N, H*L)
    return torch.cat(selected_states, dim=-1)  # (B, N, H*L)


def create_flashmtp_block_mask(
    seq_len: int,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    block_size: int,
    device: torch.device,
    kvcache_window_size: int = 0,  # W: window size for KVCache visibility
):
    """Construct Flex Attention BlockMask for FlashMTP v2 training.

    Layout: [Full Sequence (seq_len tokens) | Block_0 | Block_1 | ... | Block_{N-1}]

    Attention rules:
    1. Full Sequence part: causal attention (standard autoregressive)
    2. Block_i can see Full Sequence in [anchor_i - W + 1, anchor_i] window
    3. Block_i uses bidirectional attention internally
    4. Block_i cannot see other blocks
    5. Invalid blocks (block_keep_mask=False) don't participate in attention

    Args:
        seq_len: Length of the full input sequence
        anchor_positions: (B, N) tensor of anchor positions for each block
        block_keep_mask: (B, N) boolean mask indicating valid blocks
        block_size: Number of tokens per draft block
        device: torch device
        kvcache_window_size: Window size W. If 0, blocks see all KVCache up to anchor.
    """
    B, N = anchor_positions.shape

    def flashmtp_mask_mod(b, h, q_idx, kv_idx):
        # Full sequence region: [0, seq_len-1]
        # Block region: [seq_len, seq_len + N*block_size - 1]

        q_in_full_seq = q_idx < seq_len
        kv_in_full_seq = kv_idx < seq_len

        # Calculate block_id for query in block region
        # block_id = 0 to N-1
        q_block_id = (q_idx - seq_len) // block_size
        kv_block_id = (kv_idx - seq_len) // block_size

        # Get anchor position for current query block (if in block region)
        # anchor_positions[b] gives all anchor positions for batch item b
        # Shape: (N,), values are positions in [0, seq_len-1]
        anchor_pos = anchor_positions[b, q_block_id.clamp(min=0, max=N-1)]

        # Case 1: Both in Full Sequence -> causal attention
        both_in_full = q_in_full_seq & kv_in_full_seq
        causal_mask = kv_idx <= q_idx

        # Case 2: Query in Block, KV in Full Sequence
        # Block_i can see Full Sequence in [anchor_i - W + 1, anchor_i] window
        block_q_full_kv = (~q_in_full_seq) & kv_in_full_seq

        # Calculate window boundaries
        # anchor_pos is the position, block needs to see [anchor_pos - W + 1, anchor_pos - 1]
        # Note: anchor itself is NOT included because it's the bonus token at block start

        window_start = (anchor_pos - kvcache_window_size + 1).clamp(min=0)
        window_end = anchor_pos - 1  # exclusive of anchor (bonus token is in block)

        # KV position must be in [window_start, window_end]
        kv_in_window = (kv_idx >= window_start) & (kv_idx <= window_end)
        block_can_see_full = block_q_full_kv & kv_in_window

        # Case 3: Both in Block -> bidirectional within same block only
        both_in_block = (~q_in_full_seq) & (~kv_in_full_seq)
        same_block = q_block_id == kv_block_id
        block_bidirectional = both_in_block & same_block

        # Combine all valid attention patterns
        can_attend = (both_in_full & causal_mask) | block_can_see_full | block_bidirectional

        # Valid query if: (in Full Sequence) OR (in valid Block)
        is_valid = q_in_full_seq | block_keep_mask[b, q_block_id.clamp(min=0, max=N-1)]

        return can_attend & is_valid

    Q_LEN = seq_len + N * block_size
    KV_LEN = seq_len + N * block_size

    return create_block_mask(flashmtp_mask_mod,
                             B=B,
                             H=None,
                             Q_LEN=Q_LEN,
                             KV_LEN=KV_LEN,
                             device=device)


class OnlineFlashMTPModel(nn.Module):
    """FlashMTP v2 online training wrapper with block-wise CE loss.

    Key v2 features:
    - Full sequence uses causal attention (standard autoregressive)
    - Noise blocks use bidirectional attention
    - Each block can see recent W tokens from full sequence (sliding window)
    - Hidden states from target model are injected into noise embeddings
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
            loss_decay_gamma: Optional[float] = None,
            kvcache_window_size: int = 0,  # W: window size for KVCache visibility
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
        self.kvcache_window_size = kvcache_window_size

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

    def _create_position_ids(
        self,
        seq_len: int,
        anchor_positions: torch.Tensor
    ) -> torch.Tensor:
        """Create absolute position IDs for full sequence + draft blocks.

        Layout: [Full Sequence (0 to seq_len-1) | Block_0 | Block_1 | ... | Block_{N-1}]

        Args:
            seq_len: Length of the full input sequence
            anchor_positions: (B, N) anchor positions for each block

        Returns:
            (B, seq_len + N*block_size) position IDs
        """
        bsz, n_blocks = anchor_positions.shape
        device = anchor_positions.device

        # Full sequence positions: 0 to seq_len-1
        full_seq_positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)

        # Block positions: anchor_position + offset for each position in block
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        block_positions = anchor_positions.unsqueeze(-1) + offsets  # (B, N, block_size)
        block_positions = block_positions.view(bsz, -1)  # (B, N*block_size)

        # Concatenate: full sequence + block positions
        return torch.cat([full_seq_positions, block_positions], dim=-1)  # (B, seq_len + N*block_size)

    def _create_block_embeddings(
        self,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor
    ) -> torch.Tensor:
        """Create block embeddings with MASK tokens and bonus token at block start.

        Args:
            input_ids: (B, seq_len) input token IDs
            anchor_positions: (B, N) anchor positions for each block
            block_keep_mask: (B, N) boolean mask for valid blocks

        Returns:
            (B, N*block_size) block embeddings
        """
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size
        device = input_ids.device

        # Create token IDs for blocks: [MASK, MASK, ...] with bonus token at start
        block_ids = torch.full((bsz, n * bs),
                               self.mask_token_id,
                               dtype=torch.long,
                               device=device)

        # Block starts at positions 0, block_size, 2*block_size, ...
        block_starts = torch.arange(n, device=device) * bs
        block_starts = block_starts.unsqueeze(0).expand(bsz, -1)

        # Get anchor tokens (bonus tokens)
        valid_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, valid_anchor_positions)

        flat_batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n)

        # Place bonus token at the start of each block
        block_ids[flat_batch_idx, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(self.mask_token_id, dtype=torch.long, device=device),
        )

        return self.embed_tokens(block_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parallel block-wise training forward pass for FlashMTP v2.

        Layout: [Full Sequence (seq_len tokens) | Block_0 | ... | Block_{N-1}]
        - Full Sequence uses causal attention (standard autoregressive)
        - Blocks use bidirectional attention
        - Block_i can see Full Sequence in [anchor_i - W + 1, anchor_i] window
        - Blocks cannot see each other
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # Sample anchor positions (sorted, increasing)
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device)

        # Create full sequence embeddings (will be the KVCache part)
        full_seq_embeds = self.embed_tokens(input_ids)  # (B, seq_len, H)

        # Create block embeddings (with MASK and bonus token at start)
        block_embeds = self._create_block_embeddings(
            input_ids, anchor_positions, block_keep_mask)  # (B, N*block_size, H)

        # Concatenate: [Full Sequence | Block_0 | ... | Block_{N-1}]
        noise_embedding = torch.cat([full_seq_embeds, block_embeds], dim=1)

        # Create position IDs for the full layout
        full_position_ids = self._create_position_ids(seq_len, anchor_positions)

        # Create v2 mask: Full Seq causal + Block bidirectional + sliding window
        flashmtp_attn_mask = create_flashmtp_block_mask(
            seq_len=seq_len,
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            block_size=self.block_size,
            device=device,
            kvcache_window_size=self.kvcache_window_size,
        )

        # Prepare target hidden states for injection into blocks
        # Shape: (B, N, H*L) where L is number of target layers
        target_hidden = prepare_target_hidden(
            hidden_states, anchor_positions, self.draft_model.target_layer_ids)

        # Forward through draft model
        # Draft model will internally inject target_hidden into block positions
        output_hidden = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            attention_mask=flashmtp_attn_mask,
        )  # (B, N*block_size, H) - only block outputs

        # Compute logits for block outputs
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
            accuracy = correct.sum().float() / actual_token_count

        return loss, accuracy
