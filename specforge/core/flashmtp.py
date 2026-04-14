# coding=utf-8
"""FlashMTP Training Wrapper."""

from typing import Dict, Optional, Tuple, Union

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
    """Convert full hidden states to CHS format for FlashMTP (seq mode).

    ``hidden_states`` follows HuggingFace order: index ``0`` is embedding output;
    index ``k >= 1`` is the output after decoder layer ``k - 1``.
    ``target_layer_ids`` entries are indices into that tuple.

    Returns:
        (B, N*L, H) interleaved per-anchor:
        [layer0_anchor0, layer1_anchor0, ..., layer0_anchor1, layer1_anchor1, ...]
    """
    context_positions = (anchor_positions - 1).clamp(min=0)  # (B, N)

    selected_states = []
    for layer_id in target_layer_ids:
        layer_hidden = hidden_states[layer_id]  # (B, seq_len, H)
        layer_selected = torch.gather(
            layer_hidden,
            dim=1,
            index=context_positions.unsqueeze(-1).expand(-1, -1, layer_hidden.size(-1))
        )
        selected_states.append(layer_selected)

    # Per-anchor interleave: (B, N, L, H) -> (B, N*L, H)
    stacked = torch.stack(selected_states, dim=2)  # (B, N, L, H)
    B, N, L, H = stacked.shape
    return stacked.reshape(B, N * L, H)


def create_flashmtp_block_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    chs_len_per_block: int,
    block_size: int,
    device: torch.device,
):
    """Construct Flex Attention BlockMask for FlashMTP training with asymmetric visibility.

    Args:
        anchor_positions: (B, N) tensor of anchor positions for each block
        block_keep_mask: (B, N) boolean mask indicating valid blocks
        chs_len_per_block: Number of layer tokens per CHS segment (= num_target_layers)
        block_size: Number of mask tokens per draft block
        device: torch device

    Layout:
        QKV: [CHS_0 | CHS_1 | ... | CHS_{N-1} | Block_0 | Block_1 | ... | Block_{N-1}]

    Asymmetric visibility rules:
      - Layer(CHS) -> Layer(CHS): allowed (within same block group)
      - Layer(CHS) -> Mask(Block): FORBIDDEN
      - Mask(Block) -> Layer(CHS): allowed (mask reads from layer conditions)
      - Mask(Block) -> Mask(Block): allowed (bidirectional within same block)
      - Different block groups are invisible to each other.
      - Invalid blocks (block_keep_mask=False) see nothing.
    """

    def flashmtp_mask_mod(b, h, q_idx, kv_idx):
        total_chs_len = N * chs_len_per_block

        q_in_chs = q_idx < total_chs_len
        kv_in_chs = kv_idx < total_chs_len

        q_block_id_chs = q_idx // chs_len_per_block
        q_block_id_blk = (q_idx - total_chs_len) // block_size
        q_block_id = torch.where(q_in_chs, q_block_id_chs, q_block_id_blk)

        kv_block_id_chs = kv_idx // chs_len_per_block
        kv_block_id_blk = (kv_idx - total_chs_len) // block_size
        kv_block_id = torch.where(kv_in_chs, kv_block_id_chs, kv_block_id_blk)

        same_group = q_block_id == kv_block_id

        is_valid = q_in_chs | block_keep_mask[b, q_block_id]

        # Asymmetric: layer tokens CANNOT attend to mask tokens
        layer_to_mask = q_in_chs & ~kv_in_chs

        return same_group & is_valid & ~layer_to_mask

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
    """FlashMTP online training wrapper with block-wise CE and/or KL-to-teacher loss."""

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
            ce_loss_weight: float = 1.0,
            kl_loss_weight: float = 0.0,
            distill_temperature: float = 2.0,
            kl_topk: int = 10,
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
        self.ce_loss_weight = float(ce_loss_weight)
        self.kl_loss_weight = float(kl_loss_weight)
        if self.ce_loss_weight <= 0.0 and self.kl_loss_weight <= 0.0:
            raise ValueError(
                "At least one of ce_loss_weight or kl_loss_weight must be positive."
            )
        self.distill_temperature = distill_temperature
        self.kl_topk = kl_topk

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

    def _create_local_position_ids(self,
                                   n_blocks: int,
                                   bsz: int,
                                   device: torch.device) -> torch.Tensor:
        """Create local position IDs (0..block_size-1) for mask tokens within each block."""
        local_pos = torch.arange(self.block_size, device=device)
        # (1, n_blocks * block_size): [0,1,..,B-1, 0,1,..,B-1, ...]
        return local_pos.repeat(n_blocks).unsqueeze(0).expand(bsz, -1)

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
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        """Parallel block-wise training forward pass.

        Returns:
            total_loss, accuracy, loss_parts — ``loss_parts`` maps ``ce`` / ``kl`` to
            unweighted mean component losses (or None if that branch was skipped).
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device)
        n_blocks = anchor_positions.shape[1]

        noise_embedding = self._create_noise_embed(input_ids, anchor_positions,
                                                   block_keep_mask)

        # Local position IDs for mask tokens only (0..block_size-1 per block)
        # Layer tokens get no positional encoding — only Layer-ID embedding
        mask_position_ids = self._create_local_position_ids(
            n_blocks, bsz, device)  # (bsz, n_blocks * block_size)

        num_target_layers = getattr(self.draft_model.config,
                                    "num_target_layers", 1)
        chs_len_per_block = num_target_layers

        flashmtp_attn_mask = create_flashmtp_block_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            chs_len_per_block=chs_len_per_block,
            block_size=self.block_size,
            device=device,
        )

        target_hidden = prepare_target_hidden(
            hidden_states, anchor_positions, self.draft_model.target_layer_ids)

        output_hidden = self.draft_model(
            position_ids=mask_position_ids,
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

        n_anchors = anchor_positions.size(1)
        logits_3d = logits.view(bsz, n_anchors, self.block_size, -1)
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = target_ids.view(-1)
        flat_weights = weight_mask.view(-1)

        loss_parts: Dict[str, Optional[torch.Tensor]] = {"ce": None, "kl": None}
        loss = logits.new_zeros(())

        if self.ce_loss_weight > 0.0:
            loss_per_token = F.cross_entropy(
                flat_logits, flat_targets, reduction="none"
            )
            valid_token_count = flat_weights.sum() + 1e-6
            ce_loss = (loss_per_token * flat_weights).sum() / valid_token_count
            loss_parts["ce"] = ce_loss.detach()
            loss = loss + self.ce_loss_weight * ce_loss

        if self.kl_loss_weight > 0.0:
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
                weight_mask.sum() + 1e-6
            )
            loss_parts["kl"] = kl_loss.detach()
            loss = loss + self.kl_loss_weight * kl_loss

        # --- Accuracy ---
        with torch.no_grad():
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & (binary_eval_mask > 0.5)
            actual_token_count = binary_eval_mask.sum() + 1e-6
            accuracy = correct.sum().float() / actual_token_count

        return loss, accuracy, loss_parts
