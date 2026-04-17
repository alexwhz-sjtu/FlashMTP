# coding=utf-8
"""FlashMTP training wrapper: clean-prefix completion (iterative block generation)."""

from typing import Dict, List, Optional, Tuple, Union

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
    """Online FlashMTP training: CHS-conditioned draft with clean-prefix completion CE."""

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
        loss_decay_gamma: Optional[float] = None,
        prefix_len_sample_bias: float = 0.6,
        cold_start_loss_weight: float = 1.0,
        continuation_loss_weight: float = 1.0,
        continuation_warmup_epochs: float = 0.0,
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

        self.loss_decay_gamma = loss_decay_gamma
        if not (0.0 < prefix_len_sample_bias <= 1.0):
            raise ValueError(
                "prefix_len_sample_bias must be in (0, 1]; use 1.0 for uniform sampling"
            )
        self.prefix_len_sample_bias = prefix_len_sample_bias

        self.cold_start_loss_weight = cold_start_loss_weight
        if not (0.0 < continuation_loss_weight <= 1.0):
            raise ValueError(
                "continuation_loss_weight must be in (0, 1]; got "
                f"{continuation_loss_weight}"
            )
        self.continuation_loss_weight = continuation_loss_weight
        if continuation_warmup_epochs < 0.0:
            raise ValueError(
                f"continuation_warmup_epochs must be >= 0; got {continuation_warmup_epochs}"
            )
        self.continuation_warmup_epochs = float(continuation_warmup_epochs)

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

    def _create_masked_prefix_noise_embedding(
        self,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        unmask_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build draft block inputs: first ``unmask_positions`` tokens per block from
        ``input_ids``, remainder MASK embeddings.

        Args:
            unmask_positions: (B, N) prefix length (number of real tokens from anchor).

        Returns:
            (B, N*block_size, hidden_size)
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

    def sample_clean_prefix_lengths(
        self,
        block_keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Sample clean prefix length p per block with front-heavy mass on small p.

        Inference often starts with no in-block prefix (p=0); training matches that by
        sampling p from a truncated geometric-style law: P(p=k) ∝ r^k for k in
        {0,...,B-1}, with ``r = prefix_len_sample_bias`` (<1 favors small p). Set
        ``prefix_len_sample_bias=1.0`` to recover uniform sampling.

        Args:
            block_keep_mask: (B, N) valid block mask

        Returns:
            prefix_len: (B, N) long, in [0, block_size - 1] for valid blocks
        """
        bsz, n_blocks = block_keep_mask.shape
        device = block_keep_mask.device
        bs = self.block_size
        r = self.prefix_len_sample_bias
        if r >= 1.0:
            p = torch.randint(0, bs, (bsz, n_blocks), device=device, dtype=torch.long)
        else:
            idx = torch.arange(bs, device=device, dtype=torch.float32)
            probs = torch.pow(
                torch.tensor(r, device=device, dtype=torch.float32), idx
            )
            probs = probs / probs.sum()
            flat = torch.multinomial(
                probs, num_samples=bsz * n_blocks, replacement=True
            )
            p = flat.view(bsz, n_blocks)
        return torch.where(block_keep_mask, p, torch.zeros_like(p))

    def sample_continuation_prefix_lengths(
        self,
        block_keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Sample clean prefix length for continuation loss: p in {2, ..., B-1} (biased).

        Matches ``sample_clean_prefix_lengths`` mass shape (P(k) ∝ r^k) but restricted to
        k >= 2 so that anchor-only cold-start (p=1) is handled by the other loss term.
        For ``block_size < 3`` the valid range is empty; returns ones (caller should not
        use continuation mode in that case).
        """
        bsz, n_blocks = block_keep_mask.shape
        device = block_keep_mask.device
        bs = self.block_size
        r = self.prefix_len_sample_bias
        if bs < 3:
            return torch.ones((bsz, n_blocks), device=device, dtype=torch.long)

        ks_idx = torch.arange(2, bs, device=device, dtype=torch.long)
        ks_f = ks_idx.float()
        n_choices = ks_idx.numel()
        if r >= 1.0:
            flat = torch.randint(
                0,
                int(n_choices),
                (bsz * n_blocks,),
                device=device,
                dtype=torch.long,
            )
        else:
            probs = torch.pow(torch.tensor(r, device=device, dtype=torch.float32), ks_f)
            probs = probs / probs.sum()
            flat = torch.multinomial(
                probs, num_samples=bsz * n_blocks, replacement=True
            )
        p = ks_idx[flat].view(bsz, n_blocks)
        return torch.where(block_keep_mask, p, torch.zeros_like(p))

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        loss_mask: torch.Tensor,
        training_epoch: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Clean-prefix completion: each anchor block is duplicated — cold-start (p=1) and
        continuation (p ~ biased on {{2..B-1}}) — in one forward (2N parallel blocks).

        Args:
            training_epoch: Fractional training time in epochs, e.g.
                ``epoch + step_in_epoch / len(dataloader)``, used for continuation warmup.
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        hidden_states = normalize_teacher_hidden_to_layer_tuple(
            hidden_states,
            target_layer_ids=self.draft_model.target_layer_ids,
            hidden_size=self.draft_model.config.hidden_size,
        )

        anchor_orig, keep_orig = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )
        n_orig = anchor_orig.shape[1]

        p_cont = self.sample_continuation_prefix_lengths(keep_orig)
        ones = torch.ones_like(p_cont)
        prefix_len = torch.empty(
            bsz, 2 * n_orig, dtype=torch.long, device=device
        )
        prefix_len[:, 0::2] = torch.where(keep_orig, ones, torch.zeros_like(ones))
        prefix_len[:, 1::2] = torch.where(keep_orig, p_cont, torch.zeros_like(p_cont))

        anchor_positions = anchor_orig.repeat_interleave(2, dim=1)
        block_keep_mask = keep_orig.repeat_interleave(2, dim=1)
        n_blocks = anchor_positions.shape[1]

        is_cold_start = torch.zeros(
            bsz, n_blocks, dtype=torch.bool, device=device
        )
        is_cold_start[:, 0::2] = keep_orig
        is_continuation = torch.zeros(
            bsz, n_blocks, dtype=torch.bool, device=device
        )
        is_continuation[:, 1::2] = keep_orig

        we = self.continuation_warmup_epochs
        if we <= 0.0:
            lambda2_eff = float(self.continuation_loss_weight)
        else:
            lambda2_eff = self.continuation_loss_weight * min(
                1.0, float(training_epoch) / float(we)
            )

        num_target_layers = getattr(self.draft_model.config, "num_target_layers", 1)
        chs_len_per_block = num_target_layers if self.chs_concat_mode == "seq" else 1

        target_hidden = prepare_target_hidden(
            hidden_states,
            anchor_positions,
            self.draft_model.target_layer_ids,
            self.chs_concat_mode,
        )

        flashmtp_attn_mask = create_flashmtp_block_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            chs_len_per_block=chs_len_per_block,
            block_size=self.block_size,
            device=device,
        )

        draft_position_ids = self._create_position_ids(anchor_positions)
        if self.chs_concat_mode == "seq":
            full_position_ids = draft_position_ids
        else:
            full_position_ids = torch.cat(
                [anchor_positions, draft_position_ids], dim=-1
            )

        noise_embed = self._create_masked_prefix_noise_embedding(
            input_ids, anchor_positions, block_keep_mask, prefix_len
        )

        output_hidden = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embed,
            target_hidden=target_hidden,
            attention_mask=flashmtp_attn_mask,
        )
        student_logits = self._apply_lm_head(output_hidden)
        student_logits = student_logits.view(bsz, n_blocks, self.block_size, -1)

        pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + pos_in_block
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, n_blocks, -1),
            2,
            safe_label_indices,
        )

        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, n_blocks, -1),
            2,
            safe_label_indices,
        )

        completion_mask = pos_in_block >= prefix_len.unsqueeze(-1)
        completion_mask = (
            completion_mask
            & block_keep_mask.unsqueeze(-1)
            & valid_label_mask
            & (original_loss_mask_gathered > 0.5)
        )

        ce_per_pos = F.cross_entropy(
            student_logits.reshape(-1, student_logits.size(-1)),
            target_ids.reshape(-1),
            reduction="none",
        ).view(bsz, n_blocks, self.block_size)

        # d: 1-based index within the completion suffix (first supervised token has d=1).
        d = pos_in_block - prefix_len.unsqueeze(-1) + 1
        gamma = self.loss_decay_gamma
        if gamma is None or gamma <= 0:
            raw_w = completion_mask.float()
        else:
            w_d = torch.exp(-(d - 1).float().clamp(min=0) / gamma)
            raw_w = w_d * completion_mask.float()

        denom = raw_w.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        tilde_w = raw_w / denom

        weighted = ce_per_pos * tilde_w
        per_block_loss = weighted.sum(dim=-1)

        cold_mask = is_cold_start.float()
        cont_mask = is_continuation.float()
        n_cold = cold_mask.sum().clamp(min=0.0)
        n_cont = cont_mask.sum().clamp(min=0.0)

        cold_sum = (per_block_loss * cold_mask).sum()
        cont_sum = (per_block_loss * cont_mask).sum()

        w1 = float(self.cold_start_loss_weight)
        total_loss = torch.zeros((), device=device, dtype=per_block_loss.dtype)
        if n_cold > 0:
            total_loss = total_loss + w1 * cold_sum / n_cold
        if n_cont > 0:
            total_loss = total_loss + lambda2_eff * cont_sum / n_cont

        n_valid = block_keep_mask.sum().float().clamp(min=1.0)

        with torch.no_grad():
            preds = torch.argmax(student_logits, dim=-1)
            correct = (preds == target_ids) & completion_mask
            ce_acc = correct.sum().float() / completion_mask.sum().float().clamp(
                min=1.0
            )
            mean_p = (prefix_len.float() * block_keep_mask.float()).sum() / n_valid
            mean_p_cold = torch.where(
                n_cold > 0,
                (prefix_len.float() * cold_mask).sum() / n_cold,
                torch.zeros((), device=device),
            )
            mean_p_cont = torch.where(
                n_cont > 0,
                (prefix_len.float() * cont_mask).sum() / n_cont,
                torch.zeros((), device=device),
            )

        loss_dict = {
            "total_loss": total_loss,
            "ce_acc": ce_acc,
            "mean_prefix_len": mean_p,
            "mean_prefix_len_cold": mean_p_cold,
            "mean_prefix_len_cont": mean_p_cont,
            "lambda2_eff": torch.tensor(lambda2_eff, device=device),
        }

        return total_loss, loss_dict
