# coding=utf-8
"""FlashMTP Training Wrapper - v3 Diffusion-based Training."""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from specforge.modeling.draft.flashmtp import FlashMTPDraftModel
from specforge.core.loss import kl_divergence_loss

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    BlockMask = None
    create_block_mask = None


def normalize_teacher_hidden_to_layer_tuple(
    hidden_states: Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]],
    *,
    target_layer_ids: List[int],
    hidden_size: int,
) -> Tuple[torch.Tensor, ...]:
    """Unify HF (one tensor per layer) and SGLang (fused last dim = num_layers * H).

    ``train_flashmtp.py`` must not iterate a 3D tensor with ``for h in tensor`` (that
    walks the batch dimension). Pass the tensor through this helper instead.

    Args:
        hidden_states: Either a tuple/list of ``(B, seq, H)`` per layer (HF), or a
            single ``(B, seq, D)`` tensor where ``D == len(layers) * H`` (SGLang cat).
        target_layer_ids: Layer indices used by ``prepare_target_hidden`` (e.g. 0..L-1).
        hidden_size: Model hidden size ``H``.

    Returns:
        Tuple of length ``max(target_layer_ids) + 1`` where index ``i`` is layer ``i``.
    """
    if not target_layer_ids:
        raise ValueError("target_layer_ids must be non-empty")
    max_layer = max(target_layer_ids)
    n_layers_needed = max_layer + 1

    if isinstance(hidden_states, torch.Tensor):
        h = hidden_states
        d = h.size(-1)
        if d == hidden_size:
            if n_layers_needed != 1:
                raise ValueError(
                    "Teacher hidden last dim equals hidden_size (single layer) but "
                    f"draft expects layers 0..{max_layer}. Use HF backend with "
                    "output_hidden_states=True, or SGLang with multi-layer capture "
                    f"so that last dim is {n_layers_needed * hidden_size}."
                )
            return (h,)
        if d == n_layers_needed * hidden_size:
            return tuple(
                h[..., i * hidden_size : (i + 1) * hidden_size]
                for i in range(n_layers_needed)
            )
        raise ValueError(
            f"Unexpected teacher hidden last dim {d}: expected "
            f"{hidden_size} (single layer) or {n_layers_needed * hidden_size} "
            f"({n_layers_needed} fused layers × hidden_size)."
        )

    seq = tuple(hidden_states)
    if not seq:
        raise ValueError("hidden_states tuple/list is empty")

    first = seq[0]
    if first.dim() != 3:
        raise ValueError(
            f"Expected per-layer tensors of shape (B, seq, H), got dim {first.dim()}"
        )

    if len(seq) == n_layers_needed:
        # Per-layer width can be k*H (e.g. HF path returning fused pairs); keep last H.
        fixed: List[torch.Tensor] = []
        for t in seq:
            d = t.shape[-1]
            if d == hidden_size:
                fixed.append(t)
            elif d > hidden_size and d % hidden_size == 0:
                fixed.append(t[..., -hidden_size:].contiguous())
            else:
                raise ValueError(
                    f"Teacher layer hidden dim {d} is not hidden_size={hidden_size} "
                    f"or a multiple thereof (layer tuple len={len(seq)})."
                )
        return tuple(fixed)

    if len(seq) == 1 and isinstance(seq[0], torch.Tensor):
        return normalize_teacher_hidden_to_layer_tuple(
            seq[0],
            target_layer_ids=target_layer_ids,
            hidden_size=hidden_size,
        )

    raise ValueError(
        f"Teacher hidden tuple length {len(seq)} does not match layer indices "
        f"0..{max_layer} (need {n_layers_needed} layers)."
    )


def prepare_target_hidden(
    hidden_states: Tuple[torch.Tensor, ...],  # (num_layers,)[(B, seq_len, H)]
    anchor_positions: torch.Tensor,  # (B, N)
    target_layer_ids: list[int],
    chs_concat_mode: str = "seq",
) -> torch.Tensor:
    """Convert full hidden states to CHS format for FlashMTP.

    Args:
        hidden_states: All layers' hidden states from target model
        anchor_positions: Anchor positions for each block
        target_layer_ids: List of layer IDs to extract
        chs_concat_mode: "seq" or "feature"

    Returns:
        - seq mode: (B, N*L, H) - L layers concatenated along sequence dim
        - feature mode: (B, N, H*L) - L layers concatenated along feature dim
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

    if chs_concat_mode == "seq":
        # 按序列维度拼接: (B, N*L, H)
        return torch.cat(selected_states, dim=1)  # (B, N*L, H)
    else:  # feature mode
        # 按特征维度拼接: (B, N, H*L)
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
      1. Block_i only sees CHS_i (its own context).
         For seq mode: within CHS_i, only tokens < anchor_pos are visible.
         For feature mode: CHS_i is a single token (always visible if valid).
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
    """FlashMTP v3 online training wrapper with diffusion-based consistency distillation."""

    def __init__(
        self,
        draft_model: FlashMTPDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        chs_concat_mode: str = "seq",  # "seq" or "feature"
        # v3 diffusion training params
        w_distill: float = 1.0,
        w_cons: float = 0.6,
        inner_block_size: int = 1,
        enable_cons_after_epoch: int = 1,
    ):
        super().__init__()
        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.attention_backend = attention_backend
        self.num_anchors = num_anchors
        self.chs_concat_mode = chs_concat_mode
        self.draft_model.chs_concat_mode = chs_concat_mode

        # v3 params
        self.w_distill = w_distill
        self.w_cons = w_cons
        self.inner_block_size = inner_block_size
        self.enable_cons_after_epoch = enable_cons_after_epoch

        self._cached_block_mask: Optional[BlockMask] = None
        self._cached_seq_len: Optional[int] = None
        self._cached_bsz: Optional[int] = None

        # Width for lm_head inputs: must match draft/target hidden_size (and tied lm_head).
        # Do not use `target_lm_head.in_features` here — it can read incorrectly before/under FSDP.
        self._target_lm_in_features = int(draft_model.config.hidden_size)

    def _apply_lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply lm_head with FSDP handling.

        Coerces the last dimension to ``_target_lm_in_features`` when hidden is ``k * H``.
        Do not read ``lm_head.weight.shape`` before forward under FSDP (shard can be 1D).
        """
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        need = int(self._target_lm_in_features)
        hsz = int(self.draft_model.config.hidden_size)
        while x.size(-1) != need:
            d = x.size(-1)
            if d < need:
                raise RuntimeError(
                    f"lm_head: last dim {d} < required in_features {need}"
                )
            if d % need == 0:
                x = x[..., -need:].contiguous()
            elif d > hsz and d % hsz == 0:
                x = x[..., -hsz:].contiguous()
            else:
                raise RuntimeError(
                    f"lm_head: cannot map last dim {d} to in_features={need} "
                    f"(hidden_size={hsz})"
                )
        return self.lm_head(x)

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

    def _create_position_ids(self,
                             anchor_positions: torch.Tensor) -> torch.Tensor:
        """Create absolute position IDs for parallel draft blocks."""
        bsz, n_blocks = anchor_positions.shape
        device = anchor_positions.device
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        pos_ids = anchor_positions.unsqueeze(-1) + offsets
        return pos_ids.view(bsz, -1)

    def _create_noise_embed_for_v3(
        self,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        unmask_positions: torch.Tensor,  # (B, N) - how many tokens are unmasked in each block
    ) -> torch.Tensor:
        """
        Create noise embedding for v3 diffusion training.

        For each block:
        - First `unmask_positions[b, n]` tokens are real (from input_ids)
        - Remaining tokens are MASK

        Args:
            input_ids: (B, seq_len) full sequence
            anchor_positions: (B, N) anchor position for each block
            block_keep_mask: (B, N) valid block mask
            unmask_positions: (B, N) number of unmasked tokens in each block (includes anchor)

        Returns:
            noise_embedding: (B, N*block_size, hidden_size)
        """
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size
        device = input_ids.device

        # Initialize all as MASK
        noise_ids = torch.full((bsz, n * bs),
                               self.mask_token_id,
                               dtype=torch.long,
                               device=device)

        # For each block, fill in the unmasked tokens
        for b in range(bsz):
            for i in range(n):
                if not block_keep_mask[b, i]:
                    continue
                anchor_pos = anchor_positions[b, i].item()
                num_unmask = unmask_positions[b, i].item()

                # Block start in noise_ids
                block_start = i * bs

                # Fill in unmasked tokens
                for j in range(num_unmask):
                    pos = anchor_pos + j
                    if pos < seq_len:
                        noise_ids[b, block_start + j] = input_ids[b, pos]

        return self.embed_tokens(noise_ids)

    def sample_intermediate_states(
        self,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample intermediate states y and y* for v3 diffusion training.

        For each block of size B=block_size:
        - Sample p in [1, B - inner_block_size] (tokens already revealed in y)
        - y: first p tokens are real, rest are MASK
        - y*: first p + inner_block_size tokens are real, rest are MASK

        Args:
            input_ids: (B, seq_len) full sequence
            anchor_positions: (B, N) anchor positions
            block_keep_mask: (B, N) valid block mask

        Returns:
            - y_unmask_count: (B, N) number of unmasked tokens in y for each block
            - y_star_unmask_count: (B, N) number of unmasked tokens in y* for each block
            - U_y_mask: (B, N, block_size) mask for U_y positions (newly unmasked in y*)
            - S_y_mask: (B, N, block_size) mask for S_y positions (still masked in y*)
        """
        bsz, n_blocks = anchor_positions.shape
        device = input_ids.device

        min_p = 1
        max_p = self.block_size - self.inner_block_size

        y_unmask_count = torch.zeros((bsz, n_blocks), dtype=torch.long, device=device)

        for b in range(bsz):
            for i in range(n_blocks):
                if block_keep_mask[b, i]:
                    y_unmask_count[b, i] = torch.randint(
                        min_p, max_p + 1, (1,), device=device
                    ).item()

        y_star_unmask_count = y_unmask_count + self.inner_block_size
        y_star_unmask_count = torch.clamp(y_star_unmask_count, max=self.block_size)

        # Create U_y mask: positions newly unmasked in y* (i.e., [y_unmask_count, y_star_unmask_count))
        pos_indices = torch.arange(self.block_size, device=device).view(1, 1, -1)
        U_y_mask = (pos_indices >= y_unmask_count.unsqueeze(-1)) & \
                   (pos_indices < y_star_unmask_count.unsqueeze(-1))
        U_y_mask = U_y_mask & block_keep_mask.unsqueeze(-1)

        # Create S_y mask: positions still masked in y* (i.e., >= y_star_unmask_count)
        S_y_mask = pos_indices >= y_star_unmask_count.unsqueeze(-1)
        S_y_mask = S_y_mask & block_keep_mask.unsqueeze(-1)

        return y_unmask_count, y_star_unmask_count, U_y_mask, S_y_mask

    def get_teacher_logits_at_positions(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get teacher logits at all block positions using teacher's lm_head.

        For each position in each block, use the corresponding hidden state
        from the last layer to compute teacher's prediction.

        Args:
            hidden_states: tuple of (num_layers,) tensors of (B, seq_len, H)
            anchor_positions: (B, N) anchor positions
            block_keep_mask: (B, N) valid block mask

        Returns:
            teacher_logits: (B, N, block_size, vocab_size)
        """
        bsz, n_blocks = anchor_positions.shape
        device = anchor_positions.device

        # Last transformer layer (same indexing as prepare_target_hidden)
        last_layer_hidden = hidden_states[-1]  # (B, seq_len, H)
        inf = self._target_lm_in_features
        if last_layer_hidden.shape[-1] != inf:
            ld = last_layer_hidden.shape[-1]
            if ld > inf and ld % inf == 0:
                last_layer_hidden = last_layer_hidden[..., -inf:].contiguous()
            else:
                raise ValueError(
                    f"Teacher last-layer hidden dim {ld} cannot align to lm_head in={inf}"
                )

        # Gather hidden states for all positions in all blocks
        # Block position i corresponds to anchor_position + i
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)  # (1, 1, block_size)
        all_positions = anchor_positions.unsqueeze(-1) + offsets  # (B, N, block_size)

        # Clamp to valid range
        seq_len = last_layer_hidden.shape[1]
        all_positions_clamped = all_positions.clamp(0, seq_len - 1)

        # Gather hidden states: (B, N, block_size, H)
        gathered_hidden = torch.gather(
            last_layer_hidden.unsqueeze(1).expand(-1, n_blocks, -1, -1),
            dim=2,
            index=all_positions_clamped.unsqueeze(-1).expand(-1, -1, -1, last_layer_hidden.shape[-1])
        )

        # Apply lm_head to get logits: (B, N, block_size, vocab_size)
        original_shape = gathered_hidden.shape
        flat_hidden = gathered_hidden.reshape(-1, original_shape[-1])
        logits = self._apply_lm_head(flat_hidden)
        logits = logits.reshape(*original_shape[:-1], -1)

        return logits

    def forward_v3(
        self,
        input_ids: torch.Tensor,
        hidden_states: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        loss_mask: torch.Tensor,
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        v3 diffusion-based training forward pass.

        The second draft forward (state y*) runs only when consistency loss is used:
        ``epoch >= enable_cons_after_epoch`` and ``w_cons > 0``.

        Args:
            input_ids: (B, seq_len) input token IDs
            hidden_states: HF-style per-layer tuple or SGLang fused ``(B, L, n_layers*H)``
            loss_mask: (B, seq_len) loss mask
            epoch: current training epoch (0-based)

        Returns:
            - total_loss: scalar
            - loss_dict: dict containing individual loss components and metrics
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        hidden_states = normalize_teacher_hidden_to_layer_tuple(
            hidden_states,
            target_layer_ids=self.draft_model.target_layer_ids,
            hidden_size=self.draft_model.config.hidden_size,
        )

        # 1. Sample anchor positions
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device)
        n_blocks = anchor_positions.shape[1]

        # 2. Sample intermediate states y and y*
        y_unmask_count, y_star_unmask_count, U_y_mask, S_y_mask = \
            self.sample_intermediate_states(input_ids, anchor_positions, block_keep_mask)

        # 3. Get teacher logits for all positions (used for distillation)
        with torch.no_grad():
            teacher_logits = self.get_teacher_logits_at_positions(
                hidden_states, anchor_positions, block_keep_mask
            )  # (B, N, block_size, vocab_size)

        # 4. Prepare CHS (context hidden states)
        num_target_layers = getattr(self.draft_model.config, "num_target_layers", 1)
        chs_len_per_block = num_target_layers if self.chs_concat_mode == "seq" else 1

        target_hidden = prepare_target_hidden(
            hidden_states, anchor_positions, self.draft_model.target_layer_ids,
            self.chs_concat_mode)

        # 5. Create attention mask
        flashmtp_attn_mask = create_flashmtp_block_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            chs_len_per_block=chs_len_per_block,
            block_size=self.block_size,
            device=device,
        )

        # 6. Create position IDs
        draft_position_ids = self._create_position_ids(anchor_positions)
        if self.chs_concat_mode == "seq":
            full_position_ids = draft_position_ids
        else:
            full_position_ids = torch.cat(
                [anchor_positions, draft_position_ids], dim=-1)

        # 7. Forward for state y (with y_unmask_count tokens revealed)
        noise_embed_y = self._create_noise_embed_for_v3(
            input_ids, anchor_positions, block_keep_mask, y_unmask_count
        )
        output_hidden_y = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embed_y,
            target_hidden=target_hidden,
            attention_mask=flashmtp_attn_mask,
        )
        student_logits_y = self._apply_lm_head(output_hidden_y)  # (B, N*block_size, vocab_size)
        student_logits_y = student_logits_y.view(bsz, n_blocks, self.block_size, -1)

        need_cons = (
            epoch >= self.enable_cons_after_epoch and self.w_cons > 0.0
        )
        student_logits_y_star = None
        if need_cons:
            noise_embed_y_star = self._create_noise_embed_for_v3(
                input_ids, anchor_positions, block_keep_mask, y_star_unmask_count
            )
            output_hidden_y_star = self.draft_model(
                position_ids=full_position_ids,
                noise_embedding=noise_embed_y_star,
                target_hidden=target_hidden,
                attention_mask=flashmtp_attn_mask,
            )
            student_logits_y_star = self._apply_lm_head(output_hidden_y_star)
            student_logits_y_star = student_logits_y_star.view(
                bsz, n_blocks, self.block_size, -1
            )

        # 9. Compute Distillation Loss on U_y (positions newly unmasked in y*)
        # L_distill = E[ (1/|U_y|) * sum_{i in U_y} D_KL(p_i^T || q_phi(.|y,x)_i) ]
        distill_loss = kl_divergence_loss(
            student_logits_y,  # student predicts from state y
            teacher_logits,    # teacher's distribution
            U_y_mask,
            reduction="mean"
        )

        # 10. Compute Consistency Loss on S_y (positions still masked in y*)
        # L_cons = E[ (1/|S_y|) * sum_{i in S_y} D_KL(q_phi^-(.|y*,x)_i || q_phi(.|y,x)_i) ]
        # Note: q_phi^- is stop-gradient
        if need_cons and student_logits_y_star is not None:
            cons_loss = kl_divergence_loss(
                student_logits_y,  # student from state y
                student_logits_y_star.detach(),  # student from state y* (stop-gradient)
                S_y_mask,
                reduction="mean",
            )
        else:
            cons_loss = torch.tensor(0.0, device=device)

        # 11. Total loss
        w_cons = self.w_cons if need_cons else 0.0
        total_loss = self.w_distill * distill_loss + w_cons * cons_loss

        # 12. Compute accuracies for monitoring
        with torch.no_grad():
            # Distillation accuracy: student matches teacher's argmax on U_y
            student_pred_y = torch.argmax(student_logits_y, dim=-1)
            teacher_pred = torch.argmax(teacher_logits, dim=-1)
            distill_correct = (student_pred_y == teacher_pred) & U_y_mask
            distill_acc = distill_correct.sum().float() / (U_y_mask.sum() + 1e-8)

            # Consistency accuracy: student_y matches student_y_star on S_y
            if need_cons and student_logits_y_star is not None:
                student_pred_y_star = torch.argmax(student_logits_y_star, dim=-1)
                cons_correct = (student_pred_y == student_pred_y_star) & S_y_mask
                cons_acc = cons_correct.sum().float() / (S_y_mask.sum() + 1e-8)
            else:
                cons_acc = torch.tensor(0.0, device=device)

        loss_dict = {
            "total_loss": total_loss,
            "distill_loss": distill_loss,
            "cons_loss": cons_loss,
            "distill_acc": distill_acc,
            "cons_acc": cons_acc,
        }

        return total_loss, loss_dict

    # ==================== Legacy forward (kept for reference) ====================

    def _create_noise_embed(self, input_ids, anchor_positions,
                            block_keep_mask):
        """Create noise embedding for base structure (first token real, rest MASK)."""
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
        hidden_states: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Legacy forward for base structure - kept for backward compatibility."""
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        hidden_states = normalize_teacher_hidden_to_layer_tuple(
            hidden_states,
            target_layer_ids=self.draft_model.target_layer_ids,
            hidden_size=self.draft_model.config.hidden_size,
        )

        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device)

        noise_embedding = self._create_noise_embed(input_ids, anchor_positions,
                                                   block_keep_mask)

        context_position_ids = anchor_positions
        draft_position_ids = self._create_position_ids(anchor_positions)

        if self.chs_concat_mode == "seq":
            full_position_ids = draft_position_ids
        else:
            full_position_ids = torch.cat(
                [context_position_ids, draft_position_ids],
                dim=-1)

        num_target_layers = getattr(self.draft_model.config,
                                    "num_target_layers", 1)
        chs_len_per_block = num_target_layers if self.chs_concat_mode == "seq" else 1

        flashmtp_attn_mask = create_flashmtp_block_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            chs_len_per_block=chs_len_per_block,
            block_size=self.block_size,
            device=device,
        )

        target_hidden = prepare_target_hidden(
            hidden_states, anchor_positions, self.draft_model.target_layer_ids,
            self.chs_concat_mode)

        output_hidden = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            attention_mask=flashmtp_attn_mask,
        )

        logits = self._apply_lm_head(output_hidden)

        # Labels: same-position prediction
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

        # Weight mask
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

        # Cross entropy
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = target_ids.view(-1)
        flat_weights = weight_mask.view(-1)

        loss_per_token = F.cross_entropy(flat_logits,
                                         flat_targets,
                                         reduction="none")
        valid_token_count = flat_weights.sum() + 1e-6
        loss = (loss_per_token * flat_weights).sum() / valid_token_count

        # Accuracy
        with torch.no_grad():
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & (binary_eval_mask > 0.5)
            actual_token_count = binary_eval_mask.sum() + 1e-6
            accuracy = correct.sum().float() / actual_token_count

        return loss, accuracy
