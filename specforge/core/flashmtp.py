# coding=utf-8
"""FlashMTP Training Wrapper."""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from specforge.modeling.draft.flashmtp import FlashMTPDraftModel
from specforge.modeling.draft.flashmtp_markov_head import (
    markov_output_uses_base_lm_head,
)

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
    input_embeddings: Optional[torch.Tensor] = None,
    include_embedding_chs: bool = False,
) -> torch.Tensor:
    """Gather the fixed embedding prefix and selected transformer-layer pivots.

    ``target_layer_ids`` are **0-based transformer layer indices** (shallow=0, deep=L-1).

    ``hidden_states`` may be:
    - a tuple/list indexed by transformer layer id (+ optional embedding offset), or
    - a dict mapping layer id -> (B, seq_len, H) (SGLang partial capture).

    When ``include_embedding_chs`` is enabled, the raw input embedding at
    ``anchor-1`` is slot 0. It is an extra, fixed conditioning slot and is not
    represented in ``target_layer_ids``. The disabled mode preserves old
    checkpoints' input layout.

    Returns:
        (B, N, S, H), or (B, N, 1+S, H) when ``include_embedding_chs`` is
        enabled, with positions ``anchor-1`` per block.
    """
    context_positions = (anchor_positions - 1).clamp(min=0)  # (B, N)
    pieces: list[torch.Tensor] = []
    if include_embedding_chs:
        if input_embeddings is None:
            raise ValueError(
                "input_embeddings is required when include_embedding_chs=True."
            )
        embedding_selected = torch.gather(
            input_embeddings,
            dim=1,
            index=context_positions.unsqueeze(-1).expand(
                -1, -1, input_embeddings.size(-1)
            ),
        )
        pieces.append(embedding_selected)
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
    return torch.stack(pieces, dim=2)


def prepare_target_prediction_hidden(
    hidden_states: HiddenStatesInput,
    anchor_positions: torch.Tensor,
    block_size: int,
    num_transformer_layers: int,
    left_shift: bool = False,
) -> torch.Tensor:
    """Gather causal target states aligned with the supervised draft logits.

    Legacy mode supervises slots 1..B-1; left-shift mode supervises slots
    0..B-2 for ``block_size`` total span B (anchor plus ``B-1`` drafts). In both
    cases the causal predecessor states start at ``anchor``.
    """
    if block_size <= 1:
        raise ValueError(f"block_size must be greater than 1, got {block_size}")

    last_layer_id = num_transformer_layers - 1
    if isinstance(hidden_states, dict):
        if last_layer_id not in hidden_states:
            raise ValueError(
                "Target model did not return its final hidden layer "
                f"(layer id {last_layer_id}), which is required for TV loss."
            )
        last_hidden = hidden_states[last_layer_id]
    else:
        off = infer_hidden_states_embedding_offset(
            hidden_states, num_transformer_layers
        )
        last_hidden = hidden_states[last_layer_id + off]

    prediction_length = block_size - 1
    offsets = torch.arange(prediction_length, device=anchor_positions.device).view(
        1, 1, -1
    )
    target_positions = anchor_positions.unsqueeze(-1) + offsets
    safe_positions = target_positions.clamp(max=last_hidden.size(1) - 1)
    expanded_hidden = last_hidden.unsqueeze(1).expand(
        -1, anchor_positions.size(1), -1, -1
    )
    return torch.gather(
        expanded_hidden,
        dim=2,
        index=safe_positions.unsqueeze(-1).expand(-1, -1, -1, last_hidden.size(-1)),
    )


def add_noise_to_target_hidden(
    target_hidden: torch.Tensor,
    noise_ratio: float = 0.1,
    preserve_first_slot: bool = False,
) -> torch.Tensor:
    """Add uniform noise to conditioning slots (training augmentation).

    Samples i.i.d. from U(-noise_ratio, noise_ratio) per element (default U(-0.1, 0.1)).
    ``preserve_first_slot`` keeps the fixed raw-embedding prefix unchanged.
    """
    if noise_ratio <= 0:
        return target_hidden
    noise = torch.empty_like(target_hidden).uniform_(-noise_ratio, noise_ratio)
    if preserve_first_slot:
        noise[..., 0, :] = 0
    return target_hidden + noise


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
        chs_len_per_block: Physical context-prefix tokens per block. This includes
            the fixed raw-embedding slot when context is kept as a sequence.
        block_size: Number of tokens per draft block
        device: torch device

    Layout:
        KV: [CHS_0 | CHS_1 | ... | CHS_{N-1} | Block_0 | Block_1 | ... | Block_{N-1}]
            - Each CHS_i has length chs_len_per_block
            - Each Block_i has length block_size
        Q:  [Block_0 | Block_1 | ... | Block_{N-1}]

    Rules:
      1. Block_i only sees CHS_i (its own feature-concat context token).
      2. Intra-block draft attention is bidirectional.
      3. Different blocks are invisible to each other.
      4. Invalid/padded blocks see their own context token only. Their loss is
         zero, but keeping one finite attention key avoids all-masked softmax
         rows (which otherwise produce NaN before the zero loss mask is applied).
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

        is_draft = kv_idx >= total_chs_len
        kv_block_id = (kv_idx - total_chs_len) // block_size
        mask_draft = is_draft & (kv_block_id == q_block_id)

        # flex_attention vmap may probe out-of-range q_idx; clamp before indexing.
        safe_q_block_id = q_block_id.clamp(min=0, max=max_block_id)
        is_valid_block = block_keep_mask[b, safe_q_block_id] & q_block_ok
        valid_attention = (mask_context | mask_draft) & is_valid_block
        invalid_fallback = mask_context & ~is_valid_block & q_block_ok
        return valid_attention | invalid_fallback

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
        chs_concat_mode: str = "feature",
        add_noise: bool = False,
        target_hidden_noise_ratio: float = 0.1,
        ce_chunk_size: int = 2048,
        left_shift: Optional[bool] = None,
    ):
        super().__init__()
        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.block_size = block_size
        if self.block_size <= 1:
            raise ValueError(
                f"block_size must be at least 2 for next-token loss, got {block_size}."
            )
        self.mask_token_id = mask_token_id
        self.attention_backend = attention_backend
        self.num_anchors = num_anchors
        self.loss_decay_gamma = loss_decay_gamma
        self.final_ce_weight = float(final_ce_weight)
        self.tv_loss_weight = float(tv_loss_weight)
        self.base_lm_ce_weight = float(base_lm_ce_weight)
        for name, value in (
            ("final_ce_weight", self.final_ce_weight),
            ("tv_loss_weight", self.tv_loss_weight),
            ("base_lm_ce_weight", self.base_lm_ce_weight),
        ):
            if value < 0:
                raise ValueError(
                    f"{name} must be non-negative, got {value}."
                )
        if self.final_ce_weight + self.tv_loss_weight + self.base_lm_ce_weight == 0:
            raise ValueError("At least one FlashMTP loss weight must be positive.")
        self.base_lm_ce_decay_gamma = base_lm_ce_decay_gamma
        self.add_noise = add_noise
        self.target_hidden_noise_ratio = float(target_hidden_noise_ratio)
        if self.target_hidden_noise_ratio < 0:
            raise ValueError(
                "target_hidden_noise_ratio must be non-negative, got "
                f"{target_hidden_noise_ratio}."
            )
        self.ce_chunk_size = max(int(ce_chunk_size), 1)
        configured_left_shift = bool(getattr(draft_model, "left_shift", False))
        self.left_shift = (
            configured_left_shift if left_shift is None else bool(left_shift)
        )
        if self.left_shift != configured_left_shift:
            raise ValueError(
                "OnlineFlashMTPModel left_shift must match the draft checkpoint/config: "
                f"wrapper={self.left_shift}, draft={configured_left_shift}."
            )
        self.chs_concat_mode = "feature"
        self.draft_model.chs_concat_mode = "feature"

        self._cached_block_mask: Optional[BlockMask] = None
        self._cached_seq_len: Optional[int] = None
        self._cached_bsz: Optional[int] = None

    def _draft_block_len(self) -> int:
        """Parallel draft slots per anchor; left_shift uses block_size-1 slots."""
        if self.left_shift:
            return self.block_size - 1
        return self.block_size

    def _sample_anchor_positions(
        self, seq_len: int, loss_mask: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample anchor positions per sample; returns (anchors, keep_mask)."""
        bsz = loss_mask.shape[0]
        max_label_offset = self.block_size - 1
        max_anchor = max(seq_len - max_label_offset - 1, 0)

        valid = loss_mask[:, : max_anchor + 1] > 0.5
        if self.left_shift:
            valid = valid & (loss_mask[:, 1 : max_anchor + 2] > 0.5)
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
        """Draft token position ids: global (anchor + offset) or block-local 1..draft_len."""
        bsz, n_blocks = anchor_positions.shape
        device = anchor_positions.device
        draft_len = self._draft_block_len()
        if getattr(self.draft_model, "local_position", False):
            local = (
                torch.arange(1, draft_len + 1, device=device)
                .view(1, 1, -1)
                .expand(bsz, n_blocks, -1)
            )
            return local.reshape(bsz, -1)
        offsets = torch.arange(draft_len, device=device).view(1, 1, -1)
        pos_ids = anchor_positions.unsqueeze(-1) + offsets
        return pos_ids.view(bsz, -1)

    def _create_noise_embed(self, input_ids, anchor_positions, block_keep_mask):
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        draft_len = self._draft_block_len()
        device = input_ids.device

        noise_ids = torch.full(
            (bsz, n * draft_len), self.mask_token_id, dtype=torch.long, device=device
        )

        block_starts = torch.arange(n, device=device) * draft_len
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

    def prepare_training_tensors(
        self,
        input_ids: torch.Tensor,
        hidden_states: HiddenStatesInput,
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
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
            self.embed_tokens(input_ids),
            include_embedding_chs=self.draft_model.include_embedding_chs,
        )
        if self.add_noise:
            target_hidden = add_noise_to_target_hidden(
                target_hidden,
                noise_ratio=self.target_hidden_noise_ratio,
                preserve_first_slot=self.draft_model.include_embedding_chs,
            )
        target_prediction_hidden = None
        if self.tv_loss_weight != 0.0 and self.draft_model.markov_head is not None:
            target_prediction_hidden = prepare_target_prediction_hidden(
                hidden_states,
                anchor_positions,
                self.block_size,
                self.draft_model.config.num_target_layers,
                left_shift=self.left_shift,
            )
        return (
            anchor_positions,
            block_keep_mask,
            target_hidden,
            target_prediction_hidden,
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
        target_prediction_hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Teacher-forced serial head + chunked CE/TV projection.

        Low-rank Markov/RNN states are materialized for all prediction
        positions, while full-vocabulary logits exist only for one loss chunk.
        """
        device = prediction_hidden.device
        active_positions = binary_eval_mask.bool()
        if not active_positions.any():
            raise ValueError("FlashMTP loss has no supervised label positions.")

        # Padded blocks may contain arbitrary values from backend attention.
        # Sanitize them before any trainable projection; merely multiplying their
        # eventual loss by zero is unsafe because NaN * 0 is still NaN.
        prediction_hidden = torch.where(
            active_positions.unsqueeze(-1),
            prediction_hidden,
            torch.zeros_like(prediction_hidden),
        )
        prev_token_ids = torch.where(
            active_positions, prev_token_ids, torch.zeros_like(prev_token_ids)
        )
        if target_prediction_hidden is not None:
            target_prediction_hidden = torch.where(
                active_positions.unsqueeze(-1),
                target_prediction_hidden,
                torch.zeros_like(target_prediction_hidden),
            )

        flat_hidden = prediction_hidden.reshape(-1, prediction_hidden.size(-1))
        num_tokens = flat_hidden.size(0)
        flat_targets = labels.reshape(-1)
        flat_weights = weight_mask.reshape(-1)
        flat_eval_mask = binary_eval_mask.reshape(-1)
        valid_token_count = flat_weights.sum() + 1e-6
        markov_head = self.draft_model.markov_head
        output_mode = self.draft_model.markov_output_mode
        use_base_lm_ce = self.base_lm_ce_weight > 0.0 and base_weight_mask is not None
        use_tv_loss = self.tv_loss_weight != 0.0 and markov_head is not None
        if use_tv_loss and target_prediction_hidden is None:
            raise ValueError(
                "target_prediction_hidden is required when serial-head TV loss "
                "is enabled."
            )
        base_lm_head = (
            self._lm_head_module(prediction_hidden) if use_base_lm_ce else None
        )
        target_lm_head = (
            self._lm_head_module(prediction_hidden) if use_tv_loss else None
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
            )
            flat_markov_latent = markov_latent.reshape(-1, markov_latent.size(-1))

        def _final_logits(
            oh: torch.Tensor,
            latent: Optional[torch.Tensor],
        ) -> torch.Tensor:
            if markov_head is None:
                assert lm_head is not None
                return lm_head(oh).float()
            assert latent is not None
            head_logits = markov_head.project_logits(latent).float()
            if not markov_output_uses_base_lm_head(output_mode):
                return head_logits
            assert lm_head is not None
            return lm_head(oh).float() + head_logits

        def _chunk_ce_and_tv(
            oh: torch.Tensor,
            latent: torch.Tensor,
            target_oh: torch.Tensor,
            targets_chunk: torch.Tensor,
            weights_chunk: torch.Tensor,
        ):
            active_chunk = weights_chunk > 0
            if not active_chunk.any():
                zero = weights_chunk.new_zeros(())
                return zero, zero

            oh = oh[active_chunk]
            latent = latent[active_chunk]
            target_oh = target_oh[active_chunk]
            targets_chunk = targets_chunk[active_chunk]
            weights_chunk = weights_chunk[active_chunk].float()
            logits_chunk = _final_logits(oh, None if markov_head is None else latent)
            if (targets_chunk < 0).any() or (
                targets_chunk >= logits_chunk.size(-1)
            ).any():
                raise ValueError(
                    "Supervised labels must be within the output vocabulary: "
                    f"min={int(targets_chunk.min().item())}, "
                    f"max={int(targets_chunk.max().item())}, "
                    f"vocab_size={logits_chunk.size(-1)}."
                )
            loss_chunk = F.cross_entropy(
                logits_chunk.float(), targets_chunk, reduction="none"
            )
            ce_sum = (loss_chunk * weights_chunk).sum()
            if not use_tv_loss:
                return ce_sum, ce_sum.new_zeros(())

            assert target_lm_head is not None
            target_logits_chunk = target_lm_head(target_oh).float()
            draft_probs = F.softmax(logits_chunk.float(), dim=-1)
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
            active_chunk = weights_chunk > 0
            if not active_chunk.any():
                return weights_chunk.new_zeros(())
            logits_chunk = base_lm_head(oh[active_chunk]).float()
            targets_chunk = targets_chunk[active_chunk]
            weights_chunk = weights_chunk[active_chunk].float()
            if (targets_chunk < 0).any() or (
                targets_chunk >= logits_chunk.size(-1)
            ).any():
                raise ValueError(
                    "Supervised base-LM labels must be within the output vocabulary."
                )
            loss_chunk = F.cross_entropy(logits_chunk, targets_chunk, reduction="none")
            return (loss_chunk * weights_chunk).sum()

        loss_num = prediction_hidden.new_zeros(())
        tv_loss_num = prediction_hidden.new_zeros(())
        base_loss_num = prediction_hidden.new_zeros(())
        flat_target_prediction_hidden = (
            target_prediction_hidden.reshape(-1, target_prediction_hidden.size(-1))
            if use_tv_loss and target_prediction_hidden is not None
            else None
        )
        flat_base_weights = base_weight_mask.reshape(-1) if use_base_lm_ce else None
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
            target_oh_chunk = (
                flat_target_prediction_hidden[start:end]
                if flat_target_prediction_hidden is not None
                else oh.new_empty(oh.size(0), 0)
            )

            chunk_ce_sum, chunk_tv_sum = checkpoint(
                _chunk_ce_and_tv,
                oh,
                latent_chunk,
                target_oh_chunk,
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
            # Keep TV on the same position-weighted scale as its numerator.
            # Using the binary token count here would shrink TV under decay.
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
        target_prediction_hidden: Optional[torch.Tensor] = None,
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
                self.embed_tokens(input_ids),
                include_embedding_chs=self.draft_model.include_embedding_chs,
            )
            if self.add_noise:
                target_hidden = add_noise_to_target_hidden(
                    target_hidden,
                    noise_ratio=self.target_hidden_noise_ratio,
                    preserve_first_slot=self.draft_model.include_embedding_chs,
                )
            if (
                target_prediction_hidden is None
                and self.tv_loss_weight != 0.0
                and self.draft_model.markov_head is not None
            ):
                target_prediction_hidden = prepare_target_prediction_hidden(
                    hidden_states,
                    anchor_positions,
                    self.block_size,
                    self.draft_model.config.num_target_layers,
                    left_shift=self.left_shift,
                )
        elif anchor_positions is None or block_keep_mask is None:
            raise ValueError(
                "anchor_positions and block_keep_mask are required when target_hidden is precomputed."
            )

        noise_embedding = self._create_noise_embed(
            input_ids, anchor_positions, block_keep_mask
        )

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
            ctx_pos_flat = (
                ctx_base.unsqueeze(-1).expand(bsz, n_blk, chs).reshape(bsz, n_blk * chs)
            )
        full_rotary_position_ids = torch.cat([ctx_pos_flat, draft_position_ids], dim=-1)

        flashmtp_attn_mask = create_flashmtp_block_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            chs_len_per_block=chs,
            block_size=self._draft_block_len(),
            device=device,
        )

        output_hidden = self.draft_model(
            position_ids=draft_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            attention_mask=flashmtp_attn_mask,
            rotary_position_ids=full_rotary_position_ids,
        )

        # DeepSpec-style left shift makes slot k predict anchor+k+1.  Legacy mode
        # keeps the known anchor in slot 0 and predicts same-position tokens.
        # left_shift: block_size is total span; labels are anchor+1..anchor+(B-1).
        label_count = self.block_size - 1
        label_start = 1 if self.left_shift else 0
        label_offsets = torch.arange(
            label_start,
            label_start + (label_count if self.left_shift else self.block_size),
            device=device,
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
        draft_len = self._draft_block_len()
        weight_mask = block_keep_mask.unsqueeze(-1).expand(-1, -1, draft_len).float()
        weight_mask = weight_mask * valid_label_mask.float()

        if not self.left_shift:
            pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
            weight_mask = weight_mask * (pos_in_block > 0).float()

        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        weight_mask = weight_mask * original_loss_mask_gathered

        # In left-shift mode every hidden slot is supervised and the predecessor
        # sequence is [anchor, token(anchor+1), ..., token(anchor+B-1)].
        # Legacy mode drops hidden slot 0 and retains its old B-1 predictions.
        output_hidden_4d = output_hidden.view(
            bsz,
            anchor_positions.size(1),
            draft_len,
            output_hidden.size(-1),
        )
        if self.left_shift:
            prediction_hidden = output_hidden_4d
            anchor_token_ids = torch.gather(input_ids, 1, anchor_positions)
            prev_token_ids = torch.cat(
                [anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]], dim=-1
            )
            labels = target_ids
            prediction_weight_mask = weight_mask
        else:
            prediction_hidden = output_hidden_4d[:, :, 1:, :]
            prev_token_ids = target_ids[:, :, :-1]
            labels = target_ids[:, :, 1:]
            prediction_weight_mask = weight_mask[:, :, 1:]
        binary_eval_mask = prediction_weight_mask > 0

        base_prediction_weight_mask = None
        if self.base_lm_ce_weight > 0.0:
            base_prediction_weight_mask = prediction_weight_mask.clone()
            if (
                self.base_lm_ce_decay_gamma is not None
                and self.base_lm_ce_decay_gamma > 0
            ):
                prediction_length = self.block_size - 1
                k_pred = torch.arange(1, prediction_length + 1, device=device).view(
                    1, 1, -1
                )
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
                target_prediction_hidden=target_prediction_hidden,
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
