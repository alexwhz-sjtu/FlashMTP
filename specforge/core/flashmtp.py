# coding=utf-8
"""FlashMTP Training Wrapper."""

from typing import Optional, Tuple, Union

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
    chs_concat_mode: str = "seq",
    window_size: int = 1,
) -> torch.Tensor:
    """Convert full hidden states to CHS format for FlashMTP.

    Uses a sliding window of the latest ``window_size`` target positions ending at
    ``anchor - 1`` (i.e. ``anchor - W + k`` for ``k = 0 .. W-1``, clamped to 0).

    Args:
        hidden_states: All layers' hidden states from target model
        anchor_positions: Anchor positions for each block
        target_layer_ids: List of layer IDs to extract
        chs_concat_mode: "seq" or "feature"
        window_size: Number of consecutive target token positions (W) in the window.

    Returns:
        - seq mode: (B, N*W*L, H) — layer-major: each layer block is
          ``n0w0..n0w_{W-1}, n1w0, ...``
        - feature mode: (B, N*W, H*L) — L layers concatenated along feature dim per (n,w)
    """
    device = anchor_positions.device
    w = max(int(window_size), 1)
    bsz, n_blocks = anchor_positions.shape
    # Latest W positions before anchor: [anchor-W, ..., anchor-1]
    rel = torch.arange(w, device=device).view(1, 1, w)
    win_pos = (anchor_positions.unsqueeze(-1) - w + rel).clamp(min=0)  # (B, N, w)
    flat_pos = win_pos.reshape(bsz, n_blocks * w)

    selected_states = []
    for layer_id in target_layer_ids:
        layer_hidden = hidden_states[layer_id]  # (B, seq_len, H)
        layer_selected = torch.gather(
            layer_hidden,
            dim=1,
            index=flat_pos.unsqueeze(-1).expand(-1, -1, layer_hidden.size(-1)),
        )
        selected_states.append(layer_selected)

    if chs_concat_mode == "seq":
        # Layer-major along sequence dim: (B, N*w*L, H)
        return torch.cat(selected_states, dim=1)
    else:
        return torch.cat(selected_states, dim=-1)  # (B, N*w, H*L)


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
        QKV: [CHS_0 | CHS_1 | ... | CHS_{N-1} | Block_0 | Block_1 | ... | Block_{N-1}]
            - Each CHS_i has length chs_len_per_block
            - Each Block_i has length block_size
            - [CHS_i:Block_i] serves as Q

    Rules:
      1. Block_i only sees CHS_i (its own context).
         For seq mode: within CHS_i, only tokens < anchor_pos are visible.
         For feature mode: CHS_i is a single token (always visible if valid).
      2. Intra-block attention is bidirectional.
      3. Different blocks are invisible to each other.
      4. Invalid blocks (block_keep_mask=False) see nothing.
    """

    def flashmtp_mask_mod(b, h, q_idx, kv_idx):
        # Total length of all CHS segments
        total_chs_len = N * chs_len_per_block

        # Determine which block group q_idx belongs to
        # Use torch.where instead of if-else for vmap compatibility
        q_in_chs = q_idx < total_chs_len

        # For CHS region: block_id = q_idx // chs_len_per_block
        q_block_id_chs = q_idx // chs_len_per_block
        # For Block region: block_id = (q_idx - total_chs_len) // block_size
        q_block_id_blk = (q_idx - total_chs_len) // block_size
        q_block_id = torch.where(q_in_chs, q_block_id_chs, q_block_id_blk)

        # Determine which block group kv_idx belongs to
        kv_in_chs = kv_idx < total_chs_len
        kv_block_id_chs = kv_idx // chs_len_per_block
        kv_block_id_blk = (kv_idx - total_chs_len) // block_size
        kv_block_id = torch.where(kv_in_chs, kv_block_id_chs, kv_block_id_blk)

        # Same block group can see each other (bidirectional within group)
        same_group = q_block_id == kv_block_id

        # Valid if: same group AND (q in CHS OR block is valid)
        # CHS queries are always valid, Block queries only valid if block_keep_mask is True
        is_valid = q_in_chs | block_keep_mask[b, q_block_id]

        return same_group & is_valid

    B, N = anchor_positions.shape
    Q_LEN = N * chs_len_per_block + N * block_size
    KV_LEN = N * chs_len_per_block + N * block_size

    return create_block_mask(flashmtp_mask_mod,
                             B=B,
                             H=None,
                             Q_LEN=Q_LEN,
                             KV_LEN=KV_LEN,
                             device=device)


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
            chs_concat_mode: str = "seq",  # "seq" or "feature"
            loss_type: str = "ce",  # "ce", "kl", or "ce_kl"
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
        self.chs_concat_mode = chs_concat_mode
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
        self.draft_model.chs_concat_mode = chs_concat_mode

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

        # TODO: keep_mask meaning: Valid anchor position
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device)

        noise_embedding = self._create_noise_embed(input_ids, anchor_positions,
                                                   block_keep_mask)

        w = self.chs_window_size
        rel = torch.arange(w, device=device).view(1, 1, w)
        win_pos = (anchor_positions.unsqueeze(-1) - w + rel).clamp(min=0)  # (bsz, n_anchors, w)
        n_anchors = anchor_positions.size(1)
        flat_ctx_pos = win_pos.reshape(bsz, n_anchors * w)

        draft_position_ids = self._create_position_ids(
            anchor_positions)  # (bsz, n_blocks * block_size)

        num_target_layers = len(self.draft_model.target_layer_ids)
        # RoPE: CHS token order is layer-major (see prepare_target_hidden); repeat each
        # (n,w) position id once per target layer.
        if self.chs_concat_mode == "seq":
            context_position_ids_expanded = flat_ctx_pos.repeat(1, num_target_layers)
            full_position_ids = torch.cat(
                [context_position_ids_expanded, draft_position_ids],
                dim=-1,
            )
        else:
            full_position_ids = torch.cat(
                [flat_ctx_pos, draft_position_ids],
                dim=-1,
            )

        # CHS length per block: W spatial steps × (L layers in seq mode, else 1)
        chs_len_per_block = (
            w * num_target_layers if self.chs_concat_mode == "seq" else w
        )

        flashmtp_attn_mask = create_flashmtp_block_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            chs_len_per_block=chs_len_per_block,
            block_size=self.block_size,
            device=device,
        )

        # Target-side context (CHS): sliding window of W positions before each anchor.
        target_hidden = prepare_target_hidden(
            hidden_states,
            anchor_positions,
            self.draft_model.target_layer_ids,
            self.chs_concat_mode,
            window_size=w,
        )

        # print(f"target_hidden shape after prepare: {target_hidden.shape}")
        # print(f"full_position_ids shape: {full_position_ids.shape}")
        # print(f"noise_embedding shape: {noise_embedding.shape}")

        output_hidden = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            attention_mask=flashmtp_attn_mask,
        )

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
            # Teacher: causal LM logits at position (label - 1), same lm_head as target.
            # HF hidden_states[-1] is last layer output at each token position.
            context_idx = (safe_label_indices - 1).clamp(min=0)
            h_last = self._last_target_hidden(hidden_states)
            # gather requires index.ndim == h_last.ndim; flatten block dims then reshape
            flat_ctx = context_idx.reshape(bsz, -1)
            gather_idx = flat_ctx.unsqueeze(-1).expand(-1, -1, h_last.size(-1))
            teacher_h = torch.gather(h_last, 1, gather_idx)
            teacher_h = teacher_h.view(
                bsz, n_anchors, self.block_size, h_last.size(-1))
            with torch.no_grad():
                teacher_logits = self.lm_head(teacher_h)
            t = self.distill_temperature
            vocab = teacher_logits.size(-1)
            k = self.kl_topk
            if k is not None and k > 0:
                k = min(int(k), vocab)
            use_topk = k is not None and k > 0 and k < vocab

            if use_topk:
                # 仅对齐教师 logits 的 top-k token：在子集上做 softmax / log_softmax 后 KL
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
                weight_mask.sum() + 1e-6)
            if self.loss_type == "kl":
                loss = kl_loss
            else:
                loss = loss + self.kl_loss_weight * kl_loss

        # --- Accuracy ---
        with torch.no_grad():
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & (binary_eval_mask > 0.5)
            actual_token_count = binary_eval_mask.sum() + 1e-6
            accuracy = correct.sum().float() / actual_token_count

        return loss, accuracy
