# coding=utf-8
"""FlashMTP Training Wrapper."""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from specforge.modeling.draft.flashmtp import (
    FlashMTPDraftModel,
)
from specforge.modeling.draft.flashmtp_markov_head import markov_output_uses_base_lm_head

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
) -> torch.Tensor:
    """Gather pivot hidden states for all selected transformer layers.

    ``target_layer_ids`` are **0-based transformer layer indices** (shallow=0, deep=L-1).

    ``hidden_states`` may be:
    - a tuple/list indexed by transformer layer id (+ optional embedding offset), or
    - a dict mapping layer id -> (B, seq_len, H) (SGLang partial capture).

    Returns:
        (B, N, S, H) with ``S = len(target_layer_ids)``, positions ``anchor-1`` per block.
    """
    context_positions = (anchor_positions - 1).clamp(min=0)  # (B, N)
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
            index=context_positions.unsqueeze(-1).expand(-1, -1, layer_hidden.size(-1)),
        )
        pieces.append(layer_selected)
    return torch.stack(pieces, dim=2)  # (B, N, S, H)


def prepare_target_prediction_logits(
    target_logits: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Gather prefill logits that supervise prediction slots 1..block_size-1."""
    if block_size <= 1:
        raise ValueError(f"block_size must be greater than 1, got {block_size}")
    if target_logits.ndim != 3:
        raise ValueError(
            "target_logits must have shape (B,T,V), got "
            f"{tuple(target_logits.shape)}."
        )

    prediction_length = block_size - 1
    offsets = torch.arange(prediction_length, device=anchor_positions.device).view(
        1, 1, -1
    )
    target_positions = anchor_positions.unsqueeze(-1) + offsets
    safe_positions = target_positions.clamp(max=target_logits.size(1) - 1)
    expanded_logits = target_logits.unsqueeze(1).expand(
        -1, anchor_positions.size(1), -1, -1
    )
    return torch.gather(
        expanded_logits,
        dim=2,
        index=safe_positions.unsqueeze(-1).expand(
            -1, -1, -1, target_logits.size(-1)
        ),
    )


def pack_history_token_embeddings(
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    embed_tokens: nn.Module,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack target token embeddings needed by ``token`` and ``pivot_q`` history."""
    full = embed_tokens(input_ids)
    has_supervision = (loss_mask > 0.5).any(dim=1)
    if not bool(has_supervision.all()):
        raise ValueError("Every sample must contain at least one supervised token.")
    first_supervised = (loss_mask > 0.5).to(torch.int64).argmax(dim=1)
    start_positions = (first_supervised - int(window_size)).clamp(min=0)
    source_lengths = full.shape[1] - start_positions
    max_source_len = int(source_lengths.max().item())
    relative = torch.arange(max_source_len, device=full.device).unsqueeze(0)
    absolute = start_positions.unsqueeze(1) + relative
    valid = relative < source_lengths.unsqueeze(1)
    safe_absolute = absolute.clamp(max=full.shape[1] - 1)
    packed = torch.gather(
        full,
        dim=1,
        index=safe_absolute.unsqueeze(-1).expand(-1, -1, full.shape[-1]),
    )
    packed = packed * valid.unsqueeze(-1).to(packed.dtype)
    return packed, start_positions, source_lengths


def gather_sliding_history(
    fused_history: torch.Tensor,
    anchor_positions: torch.Tensor,
    window_size: int,
    source_start_positions: Optional[torch.Tensor] = None,
    source_lengths: Optional[torch.Tensor] = None,
    include_pivot: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather the ``W-1`` left-padded history slots for every anchor.

    Fused hidden history uses ``a-W .. a-2``. Token and pivot-Q history set
    ``include_pivot=True`` and uses ``a-W+1 .. a-1`` so its last token and the
    separately supplied CHS pivot intentionally have the same RoPE position.

    Returns ``(history, valid_mask, position_ids)`` with shapes
    ``(B,N,W-1,H)``, ``(B,N,W-1)``, and ``(B,N,W-1)``.
    """
    bsz, seq_len, hidden_size = fused_history.shape
    if source_start_positions is None:
        source_start_positions = torch.zeros(
            bsz, dtype=torch.long, device=fused_history.device
        )
    if source_lengths is None:
        source_lengths = torch.full(
            (bsz,), seq_len, dtype=torch.long, device=fused_history.device
        )
    history_len = int(window_size) - 1
    n_blocks = anchor_positions.shape[1]
    if history_len == 0:
        empty_hidden = fused_history.new_empty(bsz, n_blocks, 0, hidden_size)
        empty_mask = torch.empty(
            bsz, n_blocks, 0, dtype=torch.bool, device=fused_history.device
        )
        empty_pos = torch.empty(
            bsz, n_blocks, 0, dtype=torch.long, device=fused_history.device
        )
        return empty_hidden, empty_mask, empty_pos

    end_offset = 0 if include_pivot else -1
    start_offset = end_offset - history_len
    offsets = torch.arange(
        start_offset, end_offset, device=anchor_positions.device
    ).view(1, 1, history_len)
    positions = anchor_positions.unsqueeze(-1) + offsets
    relative_positions = positions - source_start_positions.view(bsz, 1, 1)
    valid = (relative_positions >= 0) & (
        relative_positions < source_lengths.view(bsz, 1, 1)
    )
    safe_relative_positions = relative_positions.clamp(min=0, max=seq_len - 1)
    expanded = fused_history.unsqueeze(1).expand(-1, n_blocks, -1, -1)
    gathered = torch.gather(
        expanded,
        dim=2,
        index=safe_relative_positions.unsqueeze(-1).expand(
            -1, -1, -1, hidden_size
        ),
    )
    gathered = gathered * valid.unsqueeze(-1).to(gathered.dtype)
    safe_absolute_positions = positions.clamp(min=0)
    return gathered, valid, safe_absolute_positions


def create_flashmtp_block_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    context_keep_mask: torch.Tensor,
    chs_len_per_block: int,
    block_size: int,
    device: torch.device,
    draft_keep_mask: Optional[torch.Tensor] = None,
):
    """Construct Flex Attention BlockMask for FlashMTP training with per-block CHS.

    Args:
        anchor_positions: (B, N) tensor of anchor positions for each block
        block_keep_mask: (B, N) boolean mask indicating valid blocks
        context_keep_mask: (B, N, C) validity mask for dynamic history slots
            and current multi-layer CHS slots.
        chs_len_per_block: Context slots per block (``S+W-1`` in V5).
        block_size: Draft query length per block (window Q + anchor + MASK
            tokens in ``pivot_q``; otherwise unsupervised anchor + MASK).
        device: torch device
        draft_keep_mask: Optional ``(B, N, block_size)`` validity mask for
            query-side KV slots. Used by ``pivot_q`` to hide left-padded
            window queries. When omitted, every draft slot in a valid block
            is visible.

    Layout:
        KV: [CHS_0 | CHS_1 | ... | CHS_{N-1} | Block_0 | Block_1 | ... | Block_{N-1}]
            - Each CHS_i has length chs_len_per_block
            - Each Block_i has length block_size (unsupervised anchor + MASK queries)
        Q:  [Block_0 | Block_1 | ... | Block_{N-1}]

    Rules:
      1. Block_i only sees valid slots in its own sliding-CHS context.
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
        context_slot = kv_idx % chs_len_per_block

        is_draft = kv_idx >= total_chs_len
        kv_block_id = (kv_idx - total_chs_len) // block_size
        mask_draft = is_draft & (kv_block_id == q_block_id)

        # flex_attention vmap may probe out-of-range q_idx; clamp before indexing.
        safe_q_block_id = q_block_id.clamp(min=0, max=max_block_id)
        safe_context_slot = context_slot.clamp(
            min=0, max=chs_len_per_block - 1
        )
        context_is_valid = context_keep_mask[
            b, safe_q_block_id, safe_context_slot
        ]
        mask_context = (
            is_context
            & (chs_block_id == q_block_id)
            & context_is_valid
        )
        if draft_keep_mask is not None:
            draft_slot = (kv_idx - total_chs_len) % block_size
            safe_kv_block_id = kv_block_id.clamp(min=0, max=max_block_id)
            safe_draft_slot = draft_slot.clamp(min=0, max=block_size - 1)
            mask_draft = mask_draft & draft_keep_mask[
                b, safe_kv_block_id, safe_draft_slot
            ]
        is_valid_block = block_keep_mask[b, safe_q_block_id] & q_block_ok
        return (mask_context | mask_draft) & is_valid_block

    flashmtp_mask_mod.__name__ = (
        f"flashmtp_mask_N{N}_bs{block_size}_chs{chs_len_per_block}"
        f"_dk{int(draft_keep_mask is not None)}"
    )

    create_fn = (
        compile_friendly_create_block_mask
        if compile_friendly_create_block_mask is not None
        else create_block_mask
    )
    return create_fn(
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
        final_ce_weight: float = 1.0,
        tv_loss_weight: float = 1.0,
        base_lm_ce_weight: float = 0.0,
        base_lm_ce_decay_gamma: Optional[float] = None,
        ce_chunk_size: int = 2048,
        anchor_chunk_size: int = 0,
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
        self.final_ce_weight = float(final_ce_weight)
        self.tv_loss_weight = float(tv_loss_weight)
        self.base_lm_ce_weight = float(base_lm_ce_weight)
        self.base_lm_ce_decay_gamma = base_lm_ce_decay_gamma
        self.ce_chunk_size = max(int(ce_chunk_size), 1)
        self.anchor_chunk_size = max(int(anchor_chunk_size), 0)
        self._cached_block_mask: Optional[BlockMask] = None
        self._cached_seq_len: Optional[int] = None
        self._cached_bsz: Optional[int] = None

    def _sample_anchor_positions(
        self, seq_len: int, loss_mask: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample anchor positions per sample; returns (anchors, keep_mask)."""
        bsz = loss_mask.shape[0]
        max_label_offset = self.block_size - 1
        max_anchor = max(seq_len - max_label_offset - 1, 0)

        valid = loss_mask[:, : max_anchor + 1] > 0.5
        if valid.shape[1] > 0:
            valid[:, 0] = False
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

    def _create_noise_embed(self, input_ids, anchor_positions, block_keep_mask):
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        draft_len = self.draft_model.core_draft_query_length
        device = input_ids.device

        noise_ids = torch.full(
            (bsz, n * draft_len), self.mask_token_id, dtype=torch.long, device=device
        )

        block_starts = torch.arange(n, device=device) * draft_len
        block_starts = block_starts.unsqueeze(0).expand(bsz, -1)
        flat_batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n)
        mask_id = torch.tensor(self.mask_token_id, dtype=torch.long, device=device)

        valid_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, valid_anchor_positions)
        noise_ids[flat_batch_idx, block_starts] = torch.where(
            block_keep_mask, anchor_tokens, mask_id
        )
        return self.embed_tokens(noise_ids)

    def _prepend_token_embedding_chs(
        self,
        target_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Validate CHS shape and preserve old checkpoint layouts when needed."""
        selected_chs = self.draft_model.chs_num_layers
        if not self.draft_model.include_token_embedding_chs:
            if target_hidden.size(2) != selected_chs:
                raise ValueError(
                    f"target_hidden must contain {selected_chs} transformer CHS slots; "
                    f"got {target_hidden.size(2)}."
                )
            return target_hidden
        if target_hidden.size(2) == selected_chs + 1:
            return target_hidden
        if target_hidden.size(2) != selected_chs:
            raise ValueError(
                f"target_hidden must contain {selected_chs} transformer CHS slots "
                f"before embedding prepend; got {target_hidden.size(2)}."
            )
        predecessor_positions = (anchor_positions - 1).clamp(
            min=0, max=input_ids.size(1) - 1
        )
        predecessor_ids = torch.gather(input_ids, 1, predecessor_positions)
        predecessor_embeddings = self.embed_tokens(predecessor_ids).unsqueeze(2)
        return torch.cat([predecessor_embeddings, target_hidden], dim=2)

    def _prepare_history_sources(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare packed token embeddings for pivot-Q history windows."""
        return pack_history_token_embeddings(
            input_ids,
            loss_mask,
            self.embed_tokens,
            self.draft_model.history_source_lookback,
        )

    def prepare_training_tensors(
        self,
        input_ids: torch.Tensor,
        hidden_states: HiddenStatesInput,
        loss_mask: torch.Tensor,
        target_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        """Sample anchors and gather teacher pivots/distribution states."""
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )
        target_hidden = prepare_target_hidden(
            hidden_states,
            anchor_positions,
            self.draft_model.target_layer_ids,
            self.draft_model.config.num_target_layers,
        )
        (
            history_hidden_states,
            history_start_positions,
            history_source_lengths,
        ) = self._prepare_history_sources(
            input_ids,
            loss_mask,
        )
        target_prediction_logits = None
        if self.tv_loss_weight != 0.0 and self.draft_model.markov_head is not None:
            if target_logits is None:
                raise ValueError("target_logits is required when TV loss is enabled.")
            target_prediction_logits = prepare_target_prediction_logits(
                target_logits,
                anchor_positions,
                self.block_size,
            )
        return (
            anchor_positions,
            block_keep_mask,
            target_hidden,
            history_hidden_states,
            history_start_positions,
            history_source_lengths,
            target_prediction_logits,
        )

    def _forward_packed_context(
        self,
        *,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        target_hidden: torch.Tensor,
        history_hidden_states: torch.Tensor,
        history_start_positions: torch.Tensor,
        history_source_lengths: torch.Tensor,
        noise_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Per-anchor CHS KV plus token-window draft queries."""
        device = input_ids.device
        bsz, n_blk = anchor_positions.shape
        history_hidden, history_keep_mask, history_position_ids = (
            gather_sliding_history(
                history_hidden_states,
                anchor_positions,
                self.draft_model.sliding_window_size,
                history_start_positions,
                history_source_lengths,
                include_pivot=True,
            )
        )
        history_keep_mask = history_keep_mask & block_keep_mask.unsqueeze(-1)
        current_keep_mask = block_keep_mask.unsqueeze(-1).expand(
            -1, -1, self.draft_model.condition_slot_count
        )
        hidden_size = noise_embedding.size(-1)
        core_len = self.draft_model.core_draft_query_length
        draft_queries = noise_embedding.view(bsz, n_blk, core_len, hidden_size)
        noise_embedding = torch.cat([history_hidden, draft_queries], dim=2).reshape(
            bsz,
            n_blk * (history_hidden.size(2) + core_len),
            hidden_size,
        )
        context_keep_mask = current_keep_mask
        core_keep = block_keep_mask.unsqueeze(-1).expand(-1, -1, core_len)
        draft_keep_mask = torch.cat([history_keep_mask, core_keep], dim=-1)
        ctx_pos_flat, draft_position_ids = self.draft_model.build_block_position_ids(
            anchor_positions=anchor_positions,
            history_position_ids=history_position_ids,
            history_keep_mask=history_keep_mask,
        )
        full_rotary_position_ids = torch.cat([ctx_pos_flat, draft_position_ids], dim=-1)
        flashmtp_attn_mask = create_flashmtp_block_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            context_keep_mask=context_keep_mask,
            chs_len_per_block=self.draft_model.chs_len_per_block,
            block_size=self.draft_model.draft_query_length,
            device=device,
            draft_keep_mask=draft_keep_mask,
        )
        return self.draft_model(
            position_ids=draft_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            attention_mask=flashmtp_attn_mask,
            rotary_position_ids=full_rotary_position_ids,
        )

    def _lm_head_module(self, output_hidden: torch.Tensor) -> nn.Module:
        return self.lm_head

    def _chunked_weighted_ce_and_metrics(
        self,
        prediction_hidden: torch.Tensor,
        prev_token_ids: torch.Tensor,
        labels: torch.Tensor,
        weight_mask: torch.Tensor,
        binary_eval_mask: torch.Tensor,
        block_keep_mask: torch.Tensor,
        base_weight_mask: Optional[torch.Tensor] = None,
        target_prediction_logits: Optional[torch.Tensor] = None,
        initial_prev_token_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Teacher-forced serial head + chunked CE/TV computation.

        Target distributions reuse logits from target prefill. Draft logits are
        materialized one loss chunk at a time.
        """
        device = prediction_hidden.device
        flat_hidden = prediction_hidden.reshape(-1, prediction_hidden.size(-1))
        num_tokens = flat_hidden.size(0)
        flat_targets = labels.reshape(-1)
        flat_weights = weight_mask.reshape(-1)
        flat_eval_mask = binary_eval_mask.reshape(-1)
        valid_token_count = flat_weights.sum() + 1e-6
        markov_head = self.draft_model.markov_head
        output_mode = self.draft_model.markov_output_mode
        use_base_lm_ce = (
            self.base_lm_ce_weight > 0.0 and base_weight_mask is not None
        )
        use_tv_loss = self.tv_loss_weight != 0.0 and markov_head is not None
        if use_tv_loss and target_prediction_logits is None:
            raise ValueError(
                "target_prediction_logits is required when serial-head TV loss "
                "is enabled."
            )
        base_lm_head = (
            self._lm_head_module(prediction_hidden) if use_base_lm_ce else None
        )
        lm_head = (
            self._lm_head_module(prediction_hidden)
            if markov_head is None or output_mode == "additive"
            else None
        )
        flat_markov_latent = None
        if markov_head is not None:
            markov_latent = markov_head.forward_teacher_forcing(
                hidden_states=prediction_hidden,
                prev_token_ids=prev_token_ids,
                output_mode=output_mode,
                initial_prev_token_ids=initial_prev_token_ids,
            )
            flat_markov_latent = markov_latent.reshape(-1, markov_latent.size(-1))

        def _final_logits(
            oh: torch.Tensor,
            latent: Optional[torch.Tensor],
        ) -> torch.Tensor:
            if markov_head is None:
                assert lm_head is not None
                return lm_head(oh)
            assert latent is not None
            head_logits = markov_head.project_logits(latent)
            if not markov_output_uses_base_lm_head(output_mode):
                return head_logits
            assert lm_head is not None
            return lm_head(oh) + head_logits

        def _chunk_ce_and_tv(
            oh: torch.Tensor,
            latent: torch.Tensor,
            target_logits_chunk: torch.Tensor,
            targets_chunk: torch.Tensor,
            weights_chunk: torch.Tensor,
        ):
            logits_chunk = _final_logits(oh, None if markov_head is None else latent)
            loss_chunk = F.cross_entropy(logits_chunk, targets_chunk, reduction="none")
            ce_sum = (loss_chunk * weights_chunk).sum()
            if not use_tv_loss:
                return ce_sum, ce_sum.new_zeros(())

            target_vocab_size = target_logits_chunk.size(-1)
            if target_vocab_size > logits_chunk.size(-1):
                raise ValueError(
                    "Target prefill vocab exceeds draft vocab: "
                    f"{target_vocab_size} > {logits_chunk.size(-1)}."
                )
            # The standalone draft head may have one extra synthetic MASK row.
            # Target prefill has no distribution for that row, so exclude it
            # from TV normalization while retaining it for draft CE above.
            draft_probs = F.softmax(
                logits_chunk[..., :target_vocab_size], dim=-1
            )
            target_probs = F.softmax(target_logits_chunk, dim=-1)
            tv_per_position = (draft_probs - target_probs).abs().sum(dim=-1)
            tv_sum = (tv_per_position * weights_chunk).sum()
            return ce_sum, tv_sum

        def _base_chunk_ce(
            oh: torch.Tensor,
            targets_chunk: torch.Tensor,
            weights_chunk: torch.Tensor,
        ):
            assert base_lm_head is not None
            logits_chunk = base_lm_head(oh)
            loss_chunk = F.cross_entropy(logits_chunk, targets_chunk, reduction="none")
            return (loss_chunk * weights_chunk).sum()

        loss_num = prediction_hidden.new_zeros(())
        tv_loss_num = prediction_hidden.new_zeros(())
        base_loss_num = prediction_hidden.new_zeros(())
        flat_target_prediction_logits = (
            target_prediction_logits.reshape(
                -1, target_prediction_logits.size(-1)
            )
            if use_tv_loss and target_prediction_logits is not None
            else None
        )
        flat_base_weights = (
            base_weight_mask.reshape(-1) if use_base_lm_ce else None
        )
        base_valid_token_count = (
            flat_base_weights.sum() + 1e-6 if use_base_lm_ce else None
        )
        correct_sum = prediction_hidden.new_zeros((), dtype=torch.float32)
        pred_chunks: list[torch.Tensor] = []
        chunk_size = self.ce_chunk_size

        for start in range(0, num_tokens, chunk_size):
            end = min(start + chunk_size, num_tokens)
            oh = flat_hidden[start:end]
            targets_chunk = flat_targets[start:end]
            weights_chunk = flat_weights[start:end]
            eval_mask_chunk = flat_eval_mask[start:end]
            latent_chunk = (
                flat_markov_latent[start:end]
                if flat_markov_latent is not None
                else oh.new_empty(oh.size(0), 0)
            )
            target_logits_chunk = (
                flat_target_prediction_logits[start:end]
                if flat_target_prediction_logits is not None
                else oh.new_empty(oh.size(0), 0)
            )

            chunk_ce_sum, chunk_tv_sum = checkpoint(
                _chunk_ce_and_tv,
                oh,
                latent_chunk,
                target_logits_chunk,
                targets_chunk,
                weights_chunk,
                use_reentrant=False,
            )
            loss_num = loss_num + chunk_ce_sum
            tv_loss_num = tv_loss_num + chunk_tv_sum

            if use_base_lm_ce:
                assert flat_base_weights is not None
                base_weights_chunk = flat_base_weights[start:end]
                base_chunk_sum = checkpoint(
                    _base_chunk_ce,
                    oh,
                    targets_chunk,
                    base_weights_chunk,
                    use_reentrant=False,
                )
                base_loss_num = base_loss_num + base_chunk_sum

            with torch.no_grad():
                logits_chunk = _final_logits(
                    oh,
                    None if flat_markov_latent is None else latent_chunk,
                )
                pred_chunk = logits_chunk.argmax(dim=-1)
                pred_chunks.append(pred_chunk)
                correct_sum = (
                    correct_sum
                    + ((pred_chunk == targets_chunk) & eval_mask_chunk).sum().float()
                )

        final_ce_loss = loss_num / valid_token_count
        actual_token_count = binary_eval_mask.sum() + 1e-6
        tv_loss = (
            # The numerator already includes the per-position decay weights,
            # so divide by their sum to compute a true weighted mean.
            tv_loss_num / valid_token_count
            if use_tv_loss
            else prediction_hidden.new_zeros(())
        )
        base_ce_loss = (
            base_loss_num / base_valid_token_count
            if use_base_lm_ce and base_valid_token_count is not None
            else prediction_hidden.new_zeros(())
        )
        loss = (
            self.final_ce_weight * final_ce_loss
            + self.tv_loss_weight * tv_loss
            + self.base_lm_ce_weight * base_ce_loss
        )
        pred_ids = torch.cat(pred_chunks, dim=0)
        accuracy = correct_sum / actual_token_count

        with torch.no_grad():
            pred_ids_by_block = pred_ids.view_as(labels)
            correct_by_block = pred_ids_by_block == labels
            valid_by_block = binary_eval_mask.bool()
            prefix_correct = (correct_by_block & valid_by_block).cumprod(dim=-1)
            prefix_lengths = prefix_correct.sum(dim=-1).float() + 1.0
            valid_blocks = block_keep_mask & valid_by_block.any(dim=-1)
            prefix_count = valid_blocks.sum().float()
            prefix_sum = (
                prefix_lengths[valid_blocks].sum()
                if valid_blocks.any()
                else torch.zeros((), device=device, dtype=torch.float32)
            )
            prefix_acc = prefix_sum / prefix_count.clamp(min=1.0)

        return loss, accuracy, prefix_acc, final_ce_loss, base_ce_loss, tv_loss

    def forward(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        hidden_states: Optional[HiddenStatesInput] = None,
        anchor_positions: Optional[torch.Tensor] = None,
        block_keep_mask: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        history_hidden_states: Optional[torch.Tensor] = None,
        history_start_positions: Optional[torch.Tensor] = None,
        history_source_lengths: Optional[torch.Tensor] = None,
        target_prediction_logits: Optional[torch.Tensor] = None,
        target_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[
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

        if target_hidden is None:
            if hidden_states is None:
                raise ValueError(
                    "Either hidden_states or target_hidden must be provided."
                )
            anchor_positions, block_keep_mask = self._sample_anchor_positions(
                seq_len, loss_mask, device
            )
            target_hidden = prepare_target_hidden(
                hidden_states,
                anchor_positions,
                self.draft_model.target_layer_ids,
                self.draft_model.config.num_target_layers,
            )
            (
                history_hidden_states,
                history_start_positions,
                history_source_lengths,
            ) = self._prepare_history_sources(
                input_ids,
                loss_mask,
            )
            if (
                target_prediction_logits is None
                and self.tv_loss_weight != 0.0
                and self.draft_model.markov_head is not None
            ):
                if target_logits is None:
                    raise ValueError("target_logits is required when TV loss is enabled.")
                target_prediction_logits = prepare_target_prediction_logits(
                    target_logits,
                    anchor_positions,
                    self.block_size,
                )
        elif anchor_positions is None or block_keep_mask is None:
            raise ValueError(
                "anchor_positions and block_keep_mask are required when target_hidden is precomputed."
            )
        if history_hidden_states is None:
            raise ValueError("history_hidden_states is required for sliding CHS.")
        if history_start_positions is None or history_source_lengths is None:
            raise ValueError(
                "history_start_positions and history_source_lengths are required."
            )
        target_hidden = self._prepend_token_embedding_chs(
            target_hidden, input_ids, anchor_positions
        )

        # FlexAttention's backward workspace grows sharply with the packed
        # query/KV lengths.  Keep NUM_ANCHORS as the sampling contract, but
        # execute large anchor sets in independent pieces.  The weighted
        # reduction below is algebraically equivalent to one packed forward;
        # it only changes peak workspace, not the sampled anchors or objective.
        n_anchors = anchor_positions.shape[1]
        if self.anchor_chunk_size > 0 and n_anchors > self.anchor_chunk_size:
            chunk_results = []
            chunk_weights = []
            label_offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
            pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
            loss_decay = None
            if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
                k = torch.arange(1, self.block_size, device=device).view(1, 1, -1)
                loss_decay = torch.exp(-(k - 1).float() / self.loss_decay_gamma)
            base_decay = None
            if self.base_lm_ce_weight > 0.0:
                if self.base_lm_ce_decay_gamma is not None and self.base_lm_ce_decay_gamma > 0:
                    k = torch.arange(1, self.block_size, device=device).view(1, 1, -1)
                    base_decay = torch.exp(
                        -(k - 1).float() / self.base_lm_ce_decay_gamma
                    )

            for start in range(0, n_anchors, self.anchor_chunk_size):
                end = min(start + self.anchor_chunk_size, n_anchors)
                chunk_anchor = anchor_positions[:, start:end]
                chunk_keep = block_keep_mask[:, start:end]
                chunk_target = target_hidden[:, start:end]
                chunk_target_prediction = (
                    target_prediction_logits[:, start:end]
                    if target_prediction_logits is not None
                    else None
                )
                result = self.forward(
                    input_ids=input_ids,
                    loss_mask=loss_mask,
                    anchor_positions=chunk_anchor,
                    block_keep_mask=chunk_keep,
                    target_hidden=chunk_target,
                    history_hidden_states=history_hidden_states,
                    history_start_positions=history_start_positions,
                    history_source_lengths=history_source_lengths,
                    target_prediction_logits=chunk_target_prediction,
                )

                label_indices = chunk_anchor.unsqueeze(-1) + label_offsets
                valid = label_indices < seq_len
                safe_indices = label_indices.clamp(max=seq_len - 1)
                raw_weight = chunk_keep.unsqueeze(-1).float() * valid.float()
                raw_weight = raw_weight * (pos_in_block > 0).float()
                raw_weight = raw_weight * torch.gather(
                    loss_mask.unsqueeze(1).expand(-1, end - start, -1),
                    2,
                    safe_indices,
                )
                pred_weight = raw_weight[:, :, 1:]
                binary = pred_weight > 0
                final_weight = pred_weight if loss_decay is None else pred_weight * loss_decay
                base_weight = pred_weight if base_decay is None else pred_weight * base_decay
                chunk_results.append(result)
                chunk_weights.append(
                    (
                        final_weight.sum(),
                        base_weight.sum(),
                        binary.sum().float(),
                        (chunk_keep & binary.any(dim=-1)).sum().float(),
                    )
                )

            final_den = sum(w[0] for w in chunk_weights) + 1e-6
            base_den = sum(w[1] for w in chunk_weights) + 1e-6
            acc_den = sum(w[2] for w in chunk_weights) + 1e-6
            prefix_den = sum(w[3] for w in chunk_weights).clamp(min=1.0)
            final_ce_loss = sum(
                r[3] * (w[0] + 1e-6) for r, w in zip(chunk_results, chunk_weights)
            ) / final_den
            tv_loss = sum(
                r[5] * (w[0] + 1e-6) for r, w in zip(chunk_results, chunk_weights)
            ) / final_den
            base_ce_loss = sum(
                r[4] * (w[1] + 1e-6) for r, w in zip(chunk_results, chunk_weights)
            ) / base_den
            accuracy = sum(
                r[1] * (w[2] + 1e-6) for r, w in zip(chunk_results, chunk_weights)
            ) / acc_den
            prefix_acc = sum(
                r[2] * w[3].clamp(min=1.0)
                for r, w in zip(chunk_results, chunk_weights)
            ) / prefix_den
            loss = (
                self.final_ce_weight * final_ce_loss
                + self.tv_loss_weight * tv_loss
                + self.base_lm_ce_weight * base_ce_loss
            )
            return loss, accuracy, prefix_acc, final_ce_loss, base_ce_loss, tv_loss

        noise_embedding = self._create_noise_embed(
            input_ids, anchor_positions, block_keep_mask
        )

        output_hidden = self._forward_packed_context(
            input_ids=input_ids,
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            target_hidden=target_hidden,
            history_hidden_states=history_hidden_states,
            history_start_positions=history_start_positions,
            history_source_lengths=history_source_lengths,
            noise_embedding=noise_embedding,
        )

        bsz, n_blk = anchor_positions.shape
        device = input_ids.device

        label_offsets = torch.arange(
            self.block_size, device=device
        ).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )

        # --- Weight mask: block validity * bounds * loss_mask ---
        draft_len = self.block_size
        weight_mask = (
            block_keep_mask.unsqueeze(-1).expand(-1, -1, draft_len).float()
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

        output_hidden_4d = output_hidden.view(
            bsz,
            anchor_positions.size(1),
            self.draft_model.draft_query_length,
            output_hidden.size(-1),
        )
        prediction_hidden = output_hidden_4d[
            :, :, self.draft_model.unsupervised_query_count :, :
        ]
        prev_token_ids = target_ids[:, :, :-1]
        labels = target_ids[:, :, 1:]
        prediction_weight_mask = weight_mask[:, :, 1:]
        binary_eval_mask = prediction_weight_mask > 0
        initial_prev_token_ids = None
        if self.draft_model.seed_rnn_from_predecessor:
            predecessor_positions = (anchor_positions - 1).clamp(
                min=0, max=seq_len - 1
            )
            initial_prev_token_ids = torch.gather(
                input_ids, 1, predecessor_positions
            )

        base_prediction_weight_mask = None
        if self.base_lm_ce_weight > 0.0:
            base_prediction_weight_mask = prediction_weight_mask.clone()
            if (
                self.base_lm_ce_decay_gamma is not None
                and self.base_lm_ce_decay_gamma > 0
            ):
                prediction_length = self.block_size - 1
                k_pred = torch.arange(1, prediction_length + 1, device=device).view(1, 1, -1)
                base_decay_weights = torch.exp(
                    -(k_pred - 1).clamp(min=0).float() / self.base_lm_ce_decay_gamma
                )
                base_prediction_weight_mask = (
                    base_prediction_weight_mask * base_decay_weights
                )

        # --- Loss decay: exp(-(k-1)/γ) so k=1 (1st prediction) gets weight 1.0 ---
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            prediction_length = self.block_size - 1
            k = torch.arange(1, prediction_length + 1, device=device).view(1, 1, -1)
            decay_weights = torch.exp(-(k - 1).float() / self.loss_decay_gamma)
            prediction_weight_mask = prediction_weight_mask * decay_weights

        loss, accuracy, prefix_acc, final_ce_loss, base_ce_loss, tv_loss = (
            self._chunked_weighted_ce_and_metrics(
                prediction_hidden=prediction_hidden,
                prev_token_ids=prev_token_ids,
                labels=labels,
                weight_mask=prediction_weight_mask,
                binary_eval_mask=binary_eval_mask,
                block_keep_mask=block_keep_mask,
                base_weight_mask=base_prediction_weight_mask,
                target_prediction_logits=target_prediction_logits,
                initial_prev_token_ids=initial_prev_token_ids,
            )
        )

        return (
            loss,
            accuracy,
            prefix_acc,
            final_ce_loss,
            base_ce_loss,
            tv_loss,
        )
