# coding=utf-8
"""FlashMTP training wrapper: discrete diffusion draft training."""

import math
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
        target_layer_ids: Layer indices used by ``build_full_context_target_hidden`` (e.g. 0..L-1).
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


def build_full_context_target_hidden(
    hidden_states: Tuple[torch.Tensor, ...],
    target_layer_ids: list[int],
) -> torch.Tensor:
    """将整段 teacher hidden 在特征维拼接，形状 ``(B, T, H*L)``，放在噪声块之前。

    KV 下标 ``kv_idx`` 对应 token 位置 ``t = kv_idx``（与 ``seq_len`` 一致）。

    训练时与 ``create_flashmtp_block_mask(..., context_window=W)`` 联用：
    第 i 个噪声块只能看到 token 位置 ``[anchor_i - W, anchor_i - 1]`` 上的 context（及块内）。
    """
    return torch.cat(
        [hidden_states[layer_id] for layer_id in target_layer_ids],
        dim=-1,
    )


def create_flashmtp_block_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    block_size: int,
    device: torch.device,
    *,
    seq_len: int,
    context_window: int,
):
    """Construct Flex Attention BlockMask for FlashMTP training（滑动窗口 context）。

    Args:
        anchor_positions: (B, N) 每个块的 anchor（块内第一个预测位置）
        block_keep_mask: (B, N) 块是否有效
        block_size: 每块噪声 token 数
        seq_len: 序列长度（与 teacher hidden 一致）
        context_window: W，每个块仅可见 anchor 之前长度为 W 的 target 区间
            ``t in [max(0, anchor - W), anchor - 1]``（含端点）

    Layout:
        KV: [FullTarget_0 ... FullTarget_{T-1} | Block_0 | ... | Block_{N-1}]
        Q:  [Block_0 | Block_1 | ... | Block_{N-1}]

    Rules:
      1. Block_i 仅可见满足 ``max(0, anchor_i - W) <= t < anchor_i`` 的 context KV。
      2. 块内注意力双向。
      3. 块与块之间互不可见。
      4. ``block_keep_mask=False`` 的块无可见内容。
    """
    B, N = anchor_positions.shape
    ctx_kv_len = seq_len

    def flashmtp_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        anchor = anchor_positions[b, q_block_id]
        t_low = (anchor - context_window).clamp(min=0)
        # 可见 token 位置 t ∈ [t_low, anchor) 即 anchor 前至多 W 个位置
        is_context = kv_idx < ctx_kv_len
        t = kv_idx
        mask_context = is_context & (t >= t_low) & (t < anchor)

        is_draft = kv_idx >= ctx_kv_len
        kv_block_id = (kv_idx - ctx_kv_len) // block_size
        mask_draft = is_draft & (kv_block_id == q_block_id)

        is_valid_block = block_keep_mask[b, q_block_id]
        return (mask_context | mask_draft) & is_valid_block

    Q_LEN = N * block_size
    KV_LEN = ctx_kv_len + N * block_size

    return create_block_mask(
        flashmtp_mask_mod, B=B, H=None, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device
    )


class OnlineFlashMTPModel(nn.Module):
    """Online FlashMTP training: CHS-conditioned draft with discrete diffusion loss."""

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
        prefix_len_sample_bias: float = 0.6,
        context_window_size: int = 1,
        diffusion_mask_schedule: str = "uniform",
        diffusion_mask_ratio_min: float = 0.1,
        diffusion_mask_ratio_max: float = 1.0,
        loss_weight_ce: float = 1.0,
        loss_weight_kl: float = 0.0,
        loss_weight_mse: float = 0.0,
        loss_kl_topk: int = 0,
    ):
        super().__init__()
        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.attention_backend = attention_backend
        self.num_anchors = num_anchors
        if context_window_size < 1:
            raise ValueError(
                f"context_window_size must be >= 1; got {context_window_size}"
            )
        self.context_window_size = context_window_size

        self.loss_decay_gamma = loss_decay_gamma
        if not (0.0 < prefix_len_sample_bias <= 1.0):
            raise ValueError(
                "prefix_len_sample_bias must be in (0, 1]; use 1.0 for uniform sampling"
            )
        self.prefix_len_sample_bias = prefix_len_sample_bias

        if diffusion_mask_schedule not in ("uniform", "cosine"):
            raise ValueError(
                f"diffusion_mask_schedule must be 'uniform' or 'cosine'; "
                f"got {diffusion_mask_schedule!r}"
            )
        self.diffusion_mask_schedule = diffusion_mask_schedule
        if not (0.0 <= diffusion_mask_ratio_min <= diffusion_mask_ratio_max <= 1.0):
            raise ValueError(
                "Need 0 <= diffusion_mask_ratio_min <= diffusion_mask_ratio_max <= 1.0"
            )
        self.diffusion_mask_ratio_min = float(diffusion_mask_ratio_min)
        self.diffusion_mask_ratio_max = float(diffusion_mask_ratio_max)

        self.loss_weight_ce = float(loss_weight_ce)
        self.loss_weight_kl = float(loss_weight_kl)
        self.loss_weight_mse = float(loss_weight_mse)
        if self.loss_weight_ce < 0.0 or self.loss_weight_kl < 0.0 or self.loss_weight_mse < 0.0:
            raise ValueError("loss_weight_ce, loss_weight_kl, loss_weight_mse must be >= 0")
        self.loss_kl_topk = int(loss_kl_topk)
        if self.loss_kl_topk < 0:
            raise ValueError(f"loss_kl_topk must be >= 0; got {self.loss_kl_topk}")

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

    def _sample_diffusion_mask_ratios(
        self,
        block_keep_mask: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Sample per-block mask ratio in [min, max] using ``diffusion_mask_schedule``."""
        bsz, n_blocks = block_keep_mask.shape
        u = torch.rand(bsz, n_blocks, device=device, dtype=dtype)
        lo = self.diffusion_mask_ratio_min
        hi = self.diffusion_mask_ratio_max
        if self.diffusion_mask_schedule == "uniform":
            r = lo + (hi - lo) * u
        else:
            # t in [0,1] -> sin^2(pi t / 2) in [0,1], smoother toward high mask late in "time"
            t = u
            r = lo + (hi - lo) * (torch.sin(t * (math.pi / 2.0)) ** 2)
        return torch.where(block_keep_mask, r, torch.zeros_like(r))

    def _sample_diffusion_mask_pattern(
        self,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        mask_ratio: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Randomly choose masked positions per block (True = predict / supervise).

        The in-block anchor token (j=0) is always clean (never masked), and never
        contributes to loss. We therefore sample masks only from j>=1 valid positions.
        """
        bsz, n_blocks = anchor_positions.shape
        bs = self.block_size
        device = anchor_positions.device
        mask_pattern = torch.zeros(
            bsz, n_blocks, bs, dtype=torch.bool, device=device
        )

        pos_in_seq = anchor_positions.unsqueeze(-1) + torch.arange(
            bs, device=device, dtype=torch.long
        ).view(1, 1, -1)
        in_bounds = pos_in_seq < seq_len

        for bi in range(bsz):
            for ni in range(n_blocks):
                if not block_keep_mask[bi, ni].item():
                    continue
                r = float(mask_ratio[bi, ni].item())
                valid_j = in_bounds[bi, ni].nonzero(as_tuple=False).view(-1)
                if valid_j.numel() == 0:
                    continue
                # Anchor token (j=0) must stay clean.
                candidate_j = valid_j[valid_j > 0]
                n_candidates = int(candidate_j.numel())
                if n_candidates <= 0:
                    continue
                k_float = max(1.0, min(float(n_candidates), r * float(n_candidates)))
                k_mask = int(round(k_float))
                k_mask = max(1, min(n_candidates, k_mask))
                perm = candidate_j[
                    torch.randperm(n_candidates, device=device)[:k_mask]
                ]
                mask_pattern[bi, ni, perm] = True

        return mask_pattern

    def _create_random_mask_noise_embedding(
        self,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        mask_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Draft block inputs: unmasked positions use data tokens; masked use [MASK]."""
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size
        device = input_ids.device

        pos_in_seq = anchor_positions.unsqueeze(-1) + torch.arange(
            bs, device=device, dtype=torch.long
        ).view(1, 1, -1)
        safe_pos = pos_in_seq.clamp(max=seq_len - 1)
        gathered = torch.gather(
            input_ids.unsqueeze(1).expand(-1, n, -1),
            2,
            safe_pos,
        )
        use_clean = (
            (~mask_positions)
            & (pos_in_seq < seq_len)
            & block_keep_mask.unsqueeze(-1)
        )
        noise_ids = torch.where(
            use_clean,
            gathered,
            torch.full((), self.mask_token_id, device=device, dtype=torch.long),
        )
        noise_ids = noise_ids.view(bsz, n * bs)
        return self.embed_tokens(noise_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        loss_mask: torch.Tensor,
        training_epoch: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """FlashMTP training with discrete diffusion objective."""
        hidden_states_t = normalize_teacher_hidden_to_layer_tuple(
            hidden_states,
            target_layer_ids=self.draft_model.target_layer_ids,
            hidden_size=self.draft_model.config.hidden_size,
        )

        return self._forward_discrete_diffusion(
            input_ids, loss_mask, hidden_states_t, training_epoch
        )

    def _forward_discrete_diffusion(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        hidden_states: Tuple[torch.Tensor, ...],
        training_epoch: float,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Noise-schedule random masking: predict masked tokens from visible context (CE + KL + MSE)."""
        del training_epoch  # reserved for future schedule / warmup
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        last_layer_id = self.draft_model.target_layer_ids[-1]
        teacher_last_hidden = hidden_states[last_layer_id]

        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )
        n_blocks = anchor_positions.shape[1]

        mask_ratio = self._sample_diffusion_mask_ratios(block_keep_mask, device)
        mask_pattern = self._sample_diffusion_mask_pattern(
            anchor_positions, block_keep_mask, mask_ratio, seq_len
        )

        target_hidden = build_full_context_target_hidden(
            hidden_states,
            self.draft_model.target_layer_ids,
        )

        flashmtp_attn_mask = create_flashmtp_block_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            block_size=self.block_size,
            device=device,
            seq_len=seq_len,
            context_window=self.context_window_size,
        )

        noise_embed = self._create_random_mask_noise_embedding(
            input_ids, anchor_positions, block_keep_mask, mask_pattern
        )

        full_position_ids = self._create_position_ids(anchor_positions)
        target_position_ids = torch.arange(
            seq_len, device=device, dtype=torch.long
        ).unsqueeze(0).expand(bsz, -1)

        output_hidden = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embed,
            target_hidden=target_hidden,
            target_position_ids=target_position_ids,
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

        # Never supervise the in-block anchor token (j=0), even if user loss_mask=1.
        non_anchor_positions = pos_in_block > 0
        supervision_mask = (
            mask_pattern
            & block_keep_mask.unsqueeze(-1)
            & valid_label_mask
            & non_anchor_positions
            & (original_loss_mask_gathered > 0.5)
        )

        j = pos_in_block.float()
        gamma = self.loss_decay_gamma
        if gamma is None or gamma <= 0:
            raw_w = supervision_mask.float()
        else:
            raw_w = torch.exp(-j / float(gamma)) * supervision_mask.float()

        denom = raw_w.sum(dim=-1, keepdim=True)
        tilde_w = torch.where(
            denom > 0,
            raw_w / denom.clamp(min=1e-8),
            torch.zeros_like(raw_w),
        )

        ce_per_pos = F.cross_entropy(
            student_logits.reshape(-1, student_logits.size(-1)),
            target_ids.reshape(-1),
            reduction="none",
        ).view(bsz, n_blocks, self.block_size)

        b_idx = torch.arange(bsz, device=device).view(-1, 1, 1).expand(
            bsz, n_blocks, self.block_size
        )
        teacher_h_at = teacher_last_hidden[b_idx, safe_label_indices, :]
        student_h_block = output_hidden.view(bsz, n_blocks, self.block_size, -1)

        w_ce = self.loss_weight_ce
        w_kl = self.loss_weight_kl
        w_mse = self.loss_weight_mse

        ce_block = (ce_per_pos * tilde_w).sum(dim=-1)

        if w_kl > 0.0:
            with torch.no_grad():
                t_flat = self.lm_head(
                    teacher_last_hidden.reshape(-1, teacher_last_hidden.size(-1))
                )
                teacher_logits_full = t_flat.view(
                    bsz, seq_len, t_flat.size(-1)
                )
            teacher_logits_at = teacher_logits_full[b_idx, safe_label_indices, :]
            if self.loss_kl_topk > 0 and self.loss_kl_topk < teacher_logits_at.size(-1):
                k = self.loss_kl_topk
                # KL over teacher top-k support only; both sides normalized on same support.
                teacher_topk_logits, teacher_topk_idx = torch.topk(
                    teacher_logits_at.float(), k=k, dim=-1
                )
                student_topk_logits = torch.gather(
                    student_logits.float(), dim=-1, index=teacher_topk_idx
                )
                kl_per_pos = F.kl_div(
                    F.log_softmax(student_topk_logits, dim=-1),
                    F.softmax(teacher_topk_logits, dim=-1),
                    reduction="none",
                ).sum(dim=-1)
            else:
                kl_per_pos = F.kl_div(
                    F.log_softmax(student_logits.float(), dim=-1),
                    F.softmax(teacher_logits_at.float(), dim=-1),
                    reduction="none",
                ).sum(dim=-1)
            kl_block = (kl_per_pos * tilde_w).sum(dim=-1)
        else:
            kl_block = torch.zeros(
                (bsz, n_blocks), device=device, dtype=ce_block.dtype
            )

        if w_mse > 0.0:
            mse_per_pos = F.mse_loss(
                student_h_block.float(),
                teacher_h_at.detach().float(),
                reduction="none",
            ).mean(dim=-1)
            mse_block = (mse_per_pos * tilde_w).sum(dim=-1)
        else:
            mse_block = torch.zeros(
                (bsz, n_blocks), device=device, dtype=ce_block.dtype
            )

        per_block_loss = w_ce * ce_block + w_kl * kl_block + w_mse * mse_block
        n_blk = block_keep_mask.sum().float().clamp(min=1.0)
        total_loss = (per_block_loss * block_keep_mask.float()).sum() / n_blk

        with torch.no_grad():
            preds = torch.argmax(student_logits, dim=-1)
            correct = (preds == target_ids) & supervision_mask
            ce_acc = correct.sum().float() / supervision_mask.sum().float().clamp(
                min=1.0
            )
            br = mask_pattern.float().sum(dim=-1) / float(self.block_size)
            mean_mask_ratio = (br * block_keep_mask.float()).sum() / block_keep_mask.float().sum().clamp(
                min=1.0
            )

        loss_dict = {
            "total_loss": total_loss,
            "ce_acc": ce_acc,
            "mean_prefix_len": torch.tensor(0.0, device=device),
            "loss_ce_mean": (ce_block * block_keep_mask.float()).sum() / n_blk,
            "loss_kl_mean": (kl_block * block_keep_mask.float()).sum() / n_blk,
            "loss_mse_mean": (mse_block * block_keep_mask.float()).sum() / n_blk,
            "mean_mask_ratio": mean_mask_ratio,
        }

        return total_loss, loss_dict
