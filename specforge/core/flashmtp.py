# coding=utf-8
"""FlashMTP Training Wrapper."""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from specforge.modeling.draft.flashmtp import FlashMTPDraftModel

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask

    from specforge.modeling.draft.flex_attention import (
        compile_friendly_create_block_mask,
    )

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    BlockMask = None
    create_block_mask = None
    compile_friendly_create_block_mask = None


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


HiddenStatesInput = Union[
    tuple[torch.Tensor, ...],
    list[torch.Tensor],
    Dict[int, torch.Tensor],
]


def prepare_target_hidden(
    hidden_states: HiddenStatesInput,
    anchor_positions: torch.Tensor,  # (B, N)
    target_layer_ids: list[int],
    num_transformer_layers: int,
    context_window_size: int = 1,
) -> torch.Tensor:
    """Gather a causal window of hidden states from all selected target layers.

    ``target_layer_ids`` are **0-based transformer layer indices** (shallow=0, deep=L-1).

    ``hidden_states`` may be:
    - a tuple/list indexed by transformer layer id (+ optional embedding offset), or
    - a dict mapping layer id -> (B, seq_len, H) (SGLang partial capture).

    The context slots are position-major and layer-minor::

        [pos_0/layer_0, ..., pos_0/layer_S, pos_1/layer_0, ...]

    Returns:
        ``(B, N, W*S, H)``, where the window ends at ``anchor-1``. Positions
        before the start of the sequence are zero-filled.
    """
    window_size = int(context_window_size)
    if window_size <= 0:
        raise ValueError(
            f"context_window_size must be positive, got {context_window_size}"
        )
    offsets = torch.arange(
        1 - window_size, 1, device=anchor_positions.device, dtype=torch.long
    ).view(1, 1, window_size)
    context_positions = anchor_positions.unsqueeze(-1) - 1 + offsets  # (B, N, W)
    valid_context = context_positions >= 0
    safe_context_positions = context_positions.clamp(min=0)
    bsz, n_blk, _ = safe_context_positions.shape
    pieces: list[torch.Tensor] = []
    for layer_id in target_layer_ids:
        if isinstance(hidden_states, dict):
            layer_hidden = hidden_states[layer_id]
        else:
            off = infer_hidden_states_embedding_offset(
                hidden_states, num_transformer_layers
            )
            layer_hidden = hidden_states[layer_id + off]
        layer_selected = torch.gather(
            layer_hidden,
            dim=1,
            index=safe_context_positions.reshape(bsz, n_blk * window_size)
            .unsqueeze(-1)
            .expand(-1, -1, layer_hidden.size(-1)),
        ).view(bsz, n_blk, window_size, layer_hidden.size(-1))
        layer_selected = layer_selected.masked_fill(~valid_context.unsqueeze(-1), 0)
        pieces.append(layer_selected)
    stacked = torch.stack(pieces, dim=3)  # (B, N, W, S, H)
    return stacked.reshape(bsz, n_blk, window_size * len(target_layer_ids), -1)


def prepare_shared_target_hidden(
    hidden_states: HiddenStatesInput,
    target_layer_ids: list[int],
    num_transformer_layers: int,
) -> torch.Tensor:
    """Keep each sequence position once and expand only across target layers.

    The returned context bank is position-major and layer-minor with shape
    ``(B, L, S, H)``.  Sliding-window visibility is applied later by the
    attention mask instead of materializing one ``W*S`` window per block.
    """
    pieces: list[torch.Tensor] = []
    offset = None
    if not isinstance(hidden_states, dict):
        offset = infer_hidden_states_embedding_offset(
            hidden_states, num_transformer_layers
        )
    for layer_id in target_layer_ids:
        if isinstance(hidden_states, dict):
            layer_hidden = hidden_states[layer_id]
        else:
            assert offset is not None
            layer_hidden = hidden_states[layer_id + offset]
        pieces.append(layer_hidden)
    return torch.stack(pieces, dim=2)  # (B, L, S, H)


def add_noise_to_target_hidden(
    target_hidden: torch.Tensor,
    noise_ratio: float = 0.1,
) -> torch.Tensor:
    """Add uniform noise to each selected-layer pivot hidden (training augmentation).

    Samples i.i.d. from U(-noise_ratio, noise_ratio) per element (default U(-0.1, 0.1)).
    """
    if noise_ratio <= 0:
        return target_hidden
    noise = torch.empty_like(target_hidden).uniform_(-noise_ratio, noise_ratio)
    return target_hidden + noise


def create_flashmtp_block_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    chs_len_per_block: int,
    block_size: int,
    device: torch.device,
    context_valid_mask: Optional[torch.Tensor] = None,
):
    """Construct Flex Attention BlockMask for FlashMTP training with per-block CHS.

    Args:
        anchor_positions: (B, N) tensor of anchor positions for each block
        block_keep_mask: (B, N) boolean mask indicating valid blocks
        chs_len_per_block: Number of context slots per block. For
            prefix_condition this is context_window_size * selected layer count.
        block_size: Number of tokens per draft block
        device: torch device

    Layout:
        KV: [CHS_0 | CHS_1 | ... | CHS_{N-1} | Block_0 | Block_1 | ... | Block_{N-1}]
            - Each CHS_i has length chs_len_per_block
            - Each Block_i has length block_size
        Q:  [Block_0 | Block_1 | ... | Block_{N-1}]

    Rules:
      1. Block_i only sees CHS_i (its own multi-position, multi-layer context).
      2. Intra-block draft attention is bidirectional.
      3. Different blocks are invisible to each other.
      4. Invalid blocks (block_keep_mask=False) see nothing.
    """
    block_size = int(block_size)
    chs_len_per_block = int(chs_len_per_block)
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if chs_len_per_block <= 0:
        raise ValueError(f"chs_len_per_block must be positive, got {chs_len_per_block}")

    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = N * chs_len_per_block + N * block_size
    total_chs_len = N * chs_len_per_block
    max_block_id = max(N - 1, 0)

    def flashmtp_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        q_block_ok = q_block_id <= max_block_id

        is_context = kv_idx < total_chs_len
        chs_block_id = kv_idx // chs_len_per_block
        mask_context = is_context & (chs_block_id == q_block_id)
        if context_valid_mask is not None:
            safe_chs_block_id = chs_block_id.clamp(min=0, max=max_block_id)
            chs_slot = (kv_idx % chs_len_per_block).clamp(
                min=0, max=chs_len_per_block - 1
            )
            mask_context = (
                mask_context & context_valid_mask[b, safe_chs_block_id, chs_slot]
            )

        is_draft = kv_idx >= total_chs_len
        kv_block_id = (kv_idx - total_chs_len) // block_size
        mask_draft = is_draft & (kv_block_id == q_block_id)

        # flex_attention vmap may probe out-of-range q_idx; clamp before indexing.
        safe_q_block_id = q_block_id.clamp(min=0, max=max_block_id)
        is_valid_block = block_keep_mask[b, safe_q_block_id] & q_block_ok
        return (mask_context | mask_draft) & is_valid_block

    flashmtp_mask_mod.__name__ = (
        f"flashmtp_mask_N{N}_bs{block_size}_chs{chs_len_per_block}"
    )

    create_fn = (
        compile_friendly_create_block_mask
        if compile_friendly_create_block_mask is not None
        else create_block_mask
    )
    return create_fn(
        flashmtp_mask_mod, B=B, H=None, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device
    )


def create_flashmtp_shared_context_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    sequence_length: int,
    num_context_layers: int,
    context_window_size: int,
    block_size: int,
    device: torch.device,
):
    """Construct the SWA mask over one shared, sequence-level context bank.

    Layout::

        KV: [position_0/layers ... position_L-1/layers | draft blocks]
        Q:  [draft block 0 | ... | draft block N-1]

    A query in block ``i`` sees all selected-layer slots for source positions
    ``[anchor_i - W, anchor_i)`` and all draft tokens in block ``i``.  No
    per-block copy of the context window is present in KV.
    """
    sequence_length = int(sequence_length)
    num_context_layers = int(num_context_layers)
    context_window_size = int(context_window_size)
    block_size = int(block_size)
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")
    if num_context_layers <= 0:
        raise ValueError(
            f"num_context_layers must be positive, got {num_context_layers}"
        )
    if context_window_size <= 0:
        raise ValueError(
            f"context_window_size must be positive, got {context_window_size}"
        )
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    B, N = anchor_positions.shape
    q_len = N * block_size
    total_context_len = sequence_length * num_context_layers
    kv_len = total_context_len + q_len
    max_block_id = max(N - 1, 0)

    def flashmtp_shared_context_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        q_block_ok = q_block_id <= max_block_id
        safe_q_block_id = q_block_id.clamp(min=0, max=max_block_id)
        anchor = anchor_positions[b, safe_q_block_id]

        is_context = kv_idx < total_context_len
        context_position = kv_idx // num_context_layers
        mask_context = (
            is_context
            & (context_position >= anchor - context_window_size)
            & (context_position < anchor)
        )

        is_draft = kv_idx >= total_context_len
        kv_block_id = (kv_idx - total_context_len) // block_size
        mask_draft = is_draft & (kv_block_id == q_block_id)

        is_valid_block = block_keep_mask[b, safe_q_block_id] & q_block_ok
        return (mask_context | mask_draft) & is_valid_block

    flashmtp_shared_context_mask_mod.__name__ = (
        f"flashmtp_shared_N{N}_bs{block_size}_L{sequence_length}_"
        f"S{num_context_layers}_W{context_window_size}"
    )
    create_fn = (
        compile_friendly_create_block_mask
        if compile_friendly_create_block_mask is not None
        else create_block_mask
    )
    return create_fn(
        flashmtp_shared_context_mask_mod,
        B=B,
        H=None,
        Q_LEN=q_len,
        KV_LEN=kv_len,
        device=device,
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
        add_noise: bool = False,
        target_hidden_noise_ratio: float = 0.1,
        w1_mse: float = 0.0,
        ce_chunk_size: int = 2048,
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
        self.add_noise = add_noise
        self.target_hidden_noise_ratio = target_hidden_noise_ratio
        self.w1_mse = w1_mse
        self.ce_chunk_size = max(int(ce_chunk_size), 1)
        self.chs_concat_mode = "feature"
        self.draft_model.chs_concat_mode = "feature"

        self._cached_block_mask: Optional[BlockMask] = None
        self._cached_seq_len: Optional[int] = None
        self._cached_bsz: Optional[int] = None

    @property
    def uses_shared_training_context(self) -> bool:
        """Whether training uses one sequence-level CHS bank for SWA."""
        return (
            self.draft_model.pivot_fuse_mode == "prefix_condition"
            and self.draft_model.context_window_size > 1
            and not self.add_noise
        )

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

    def _create_draft_position_ids(
        self, anchor_positions: torch.Tensor
    ) -> torch.Tensor:
        """Draft token position ids: global (anchor + offset) or block-local 1..block_size."""
        bsz, n_blocks = anchor_positions.shape
        device = anchor_positions.device
        bs = self.block_size
        if getattr(self.draft_model, "local_position", False):
            local = (
                torch.arange(1, bs + 1, device=device)
                .view(1, 1, -1)
                .expand(bsz, n_blocks, -1)
            )
            return local.reshape(bsz, -1)
        offsets = torch.arange(bs, device=device).view(1, 1, -1)
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

        # substitute the anchor position with label token (bonus token in inference)
        noise_ids[flat_batch_idx, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(self.mask_token_id, dtype=torch.long, device=device),
        )

        return self.embed_tokens(noise_ids)

    @torch.no_grad()
    def prepare_training_tensors(
        self,
        input_ids: torch.Tensor,
        hidden_states: HiddenStatesInput,
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample anchors and prepare either shared SWA context or legacy pivots."""
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )
        if self.uses_shared_training_context:
            target_hidden = prepare_shared_target_hidden(
                hidden_states,
                self.draft_model.target_layer_ids,
                self.draft_model.config.num_target_layers,
            )
        else:
            target_hidden = prepare_target_hidden(
                hidden_states,
                anchor_positions,
                self.draft_model.target_layer_ids,
                self.draft_model.config.num_target_layers,
                self.draft_model.context_window_size,
            )
        if self.add_noise:
            target_hidden = add_noise_to_target_hidden(
                target_hidden, noise_ratio=self.target_hidden_noise_ratio
            )
        return anchor_positions, block_keep_mask, target_hidden

    def _lm_head_module(self, output_hidden: torch.Tensor) -> nn.Module:
        if self.draft_model.draft_lm_head is not None:
            return self.draft_model.draft_lm_head
        return self.lm_head

    def _chunked_weighted_ce_and_metrics(
        self,
        output_hidden: torch.Tensor,
        target_ids: torch.Tensor,
        weight_mask: torch.Tensor,
        binary_eval_mask: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        bsz: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Chunk lm_head + CE to avoid materializing [T, vocab] logits at once."""
        device = output_hidden.device
        flat_hidden = output_hidden.reshape(-1, output_hidden.size(-1))
        num_tokens = flat_hidden.size(0)
        flat_targets = target_ids.reshape(-1)
        flat_weights = weight_mask.reshape(-1)
        valid_token_count = flat_weights.sum() + 1e-6
        lm_head = self._lm_head_module(output_hidden)

        def _chunk_ce(
            oh: torch.Tensor, targets_chunk: torch.Tensor, weights_chunk: torch.Tensor
        ):
            logits_chunk = lm_head(oh)
            loss_chunk = F.cross_entropy(logits_chunk, targets_chunk, reduction="none")
            return (loss_chunk * weights_chunk).sum()

        loss_num = output_hidden.new_zeros(())
        correct_sum = output_hidden.new_zeros((), dtype=torch.float32)
        pred_chunks: list[torch.Tensor] = []
        chunk_size = self.ce_chunk_size

        for start in range(0, num_tokens, chunk_size):
            end = min(start + chunk_size, num_tokens)
            oh = flat_hidden[start:end]
            targets_chunk = flat_targets[start:end]
            weights_chunk = flat_weights[start:end]
            eval_mask_chunk = binary_eval_mask[start:end]

            chunk_sum = checkpoint(
                _chunk_ce,
                oh,
                targets_chunk,
                weights_chunk,
                use_reentrant=False,
            )
            loss_num = loss_num + chunk_sum

            with torch.no_grad():
                logits_chunk = lm_head(oh)
                pred_chunk = logits_chunk.argmax(dim=-1)
                pred_chunks.append(pred_chunk)
                correct_sum = (
                    correct_sum
                    + ((pred_chunk == targets_chunk) & (eval_mask_chunk > 0.5))
                    .sum()
                    .float()
                )

        loss = loss_num / valid_token_count
        pred_ids = torch.cat(pred_chunks, dim=0)
        actual_token_count = binary_eval_mask.sum() + 1e-6
        accuracy = correct_sum / actual_token_count

        with torch.no_grad():
            pred_ids_by_block = pred_ids.view(
                bsz, anchor_positions.size(1), self.block_size
            )
            correct_by_block = pred_ids_by_block == target_ids
            valid_by_block = (
                binary_eval_mask.view(bsz, anchor_positions.size(1), self.block_size)
                > 0.5
            )
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

        return loss, accuracy, prefix_acc

    def forward(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        hidden_states: Optional[HiddenStatesInput] = None,
        anchor_positions: Optional[torch.Tensor] = None,
        block_keep_mask: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Parallel block-wise training forward pass."""
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        if target_hidden is None:
            if hidden_states is None:
                raise ValueError(
                    "Either hidden_states or target_hidden must be provided."
                )
            anchor_positions, block_keep_mask = self._sample_anchor_positions(
                seq_len, loss_mask, device
            )
            if self.uses_shared_training_context:
                target_hidden = prepare_shared_target_hidden(
                    hidden_states,
                    self.draft_model.target_layer_ids,
                    self.draft_model.config.num_target_layers,
                )
            else:
                target_hidden = prepare_target_hidden(
                    hidden_states,
                    anchor_positions,
                    self.draft_model.target_layer_ids,
                    self.draft_model.config.num_target_layers,
                    self.draft_model.context_window_size,
                )
            if self.add_noise:
                target_hidden = add_noise_to_target_hidden(
                    target_hidden, noise_ratio=self.target_hidden_noise_ratio
                )
        elif anchor_positions is None or block_keep_mask is None:
            raise ValueError(
                "anchor_positions and block_keep_mask are required when target_hidden is precomputed."
            )

        noise_embedding = self._create_noise_embed(
            input_ids, anchor_positions, block_keep_mask
        )

        # CHS always uses global target positions. Every layer slot belonging to the
        # same source-token position receives the same RoPE position id.
        draft_position_ids = self._create_draft_position_ids(anchor_positions)

        chs = self.draft_model.chs_len_per_block
        bsz, n_blk = anchor_positions.shape
        if self.uses_shared_training_context:
            num_layers = len(self.draft_model.target_layer_ids)
            expected_shape = (bsz, seq_len, num_layers)
            if target_hidden.shape[:3] != expected_shape:
                raise ValueError(
                    "Shared SWA target_hidden must have leading shape "
                    f"{expected_shape}, got {tuple(target_hidden.shape[:3])}."
                )
            ctx_pos_flat = (
                torch.arange(seq_len, device=device, dtype=torch.long)
                .view(1, seq_len, 1)
                .expand(bsz, -1, num_layers)
                .reshape(bsz, seq_len * num_layers)
            )
            flashmtp_attn_mask = create_flashmtp_shared_context_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                sequence_length=seq_len,
                num_context_layers=num_layers,
                context_window_size=self.draft_model.context_window_size,
                block_size=self.block_size,
                device=device,
            )
        else:
            ctx_pos_flat = self.draft_model.context_position_ids(anchor_positions - 1)
            if self.draft_model.pivot_fuse_mode == "prefix_condition":
                window = self.draft_model.context_window_size
                num_layers = len(self.draft_model.target_layer_ids)
                window_offsets = torch.arange(
                    1 - window, 1, device=device, dtype=torch.long
                ).view(1, 1, window)
                context_valid_mask = (
                    (anchor_positions.unsqueeze(-1) - 1 + window_offsets >= 0)
                    .unsqueeze(-1)
                    .expand(-1, -1, -1, num_layers)
                )
                context_valid_mask = context_valid_mask.reshape(bsz, n_blk, chs)
            else:
                context_valid_mask = (anchor_positions - 1 >= 0).unsqueeze(-1)

            flashmtp_attn_mask = create_flashmtp_block_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                chs_len_per_block=chs,
                block_size=self.block_size,
                device=device,
                context_valid_mask=context_valid_mask,
            )

        full_rotary_position_ids = torch.cat([ctx_pos_flat, draft_position_ids], dim=-1)

        output_hidden = self.draft_model(
            position_ids=draft_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            attention_mask=flashmtp_attn_mask,
            rotary_position_ids=full_rotary_position_ids,
        )

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

        # --- Weight mask: block validity * bounds * exclude anchor (pos 0) * loss_mask ---
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

        # --- Loss decay: exp(-(k-1)/γ) so k=1 (1st prediction) gets weight 1.0 ---
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            k = torch.arange(self.block_size, device=device).view(1, 1, -1)
            decay_weights = torch.exp(
                -(k - 1).clamp(min=0).float() / self.loss_decay_gamma
            )
            weight_mask = weight_mask * decay_weights

        loss, accuracy, prefix_acc = self._chunked_weighted_ce_and_metrics(
            output_hidden=output_hidden,
            target_ids=target_ids,
            weight_mask=weight_mask,
            binary_eval_mask=binary_eval_mask,
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            bsz=bsz,
        )

        # --- First predicted token hidden MSE (block pos 1 -> teacher last layer at anchor) ---
        mse_loss = torch.zeros((), device=device, dtype=loss.dtype)
        if self.w1_mse > 0:
            if hidden_states is None:
                raise ValueError("hidden_states required when w1_mse > 0.")
            n_blk = anchor_positions.size(1)
            h_dim = output_hidden.size(-1)
            draft_first_h = output_hidden.view(bsz, n_blk, self.block_size, h_dim)[
                :, :, 1, :
            ]
            first_label_indices = (anchor_positions + 1).clamp(max=seq_len - 1)
            teacher_pos = (first_label_indices - 1).clamp(min=0)
            if isinstance(hidden_states, dict):
                last_layer_id = self.draft_model.config.num_target_layers - 1
                teacher_seq_hidden = hidden_states[last_layer_id]
            else:
                off = infer_hidden_states_embedding_offset(
                    hidden_states, self.draft_model.config.num_target_layers
                )
                teacher_seq_hidden = hidden_states[-1 + off]
            teacher_first_h = torch.gather(
                teacher_seq_hidden,
                1,
                teacher_pos.unsqueeze(-1).expand(-1, -1, h_dim),
            )
            mse_mask = block_keep_mask.float()
            mse_mask = mse_mask * (first_label_indices < seq_len).float()
            mse_mask = mse_mask * torch.gather(
                loss_mask,
                1,
                first_label_indices,
            )
            mse_per_block = F.mse_loss(
                draft_first_h.float(),
                teacher_first_h.float(),
                reduction="none",
            ).mean(dim=-1)
            mse_loss = (mse_per_block * mse_mask).sum() / (mse_mask.sum() + 1e-6)
            loss = loss + self.w1_mse * mse_loss

        return loss, accuracy, prefix_acc, mse_loss
