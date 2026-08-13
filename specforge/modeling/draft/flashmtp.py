import time
from copy import deepcopy
from typing import Callable, Optional

import torch
from torch import nn
from transformers import DynamicCache
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from ...utils import print_on_rank0
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3Attention,
    Qwen3Config,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)
from typing_extensions import Tuple, Unpack


def _cuda_sync_time(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


STOCHASTIC_VERIFICATION_MODES = ("match", "rejection")


def _validate_stochastic_verification_mode(mode: str) -> str:
    mode = str(mode).lower()
    if mode not in STOCHASTIC_VERIFICATION_MODES:
        raise ValueError(
            f"Unknown stochastic_verification_mode={mode!r}; expected one of "
            f"{STOCHASTIC_VERIFICATION_MODES}."
        )
    return mode


def _logits_to_probs(
    logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.nn.functional.one_hot(
            torch.argmax(logits, dim=-1),
            num_classes=logits.shape[-1],
        ).float()
    return torch.softmax(logits.float() / float(temperature), dim=-1)


def _sample_from_probs(probs: torch.Tensor) -> torch.Tensor:
    vocab_size = probs.shape[-1]
    return torch.multinomial(
        probs.reshape(-1, vocab_size), num_samples=1
    ).reshape(*probs.shape[:-1])


def _sample_residual(
    target_probs: torch.Tensor, draft_probs: torch.Tensor
) -> torch.Tensor:
    residual = torch.clamp(target_probs - draft_probs, min=0.0)
    residual_mass = residual.sum(dim=-1, keepdim=True)
    residual = torch.where(
        residual_mass > 1e-8,
        residual / residual_mass.clamp_min(1e-8),
        target_probs,
    )
    return _sample_from_probs(residual)


def rejection_sample_verify(
    *,
    proposed_tokens: torch.Tensor,
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    temperature: float,
) -> tuple[int, torch.Tensor]:
    """Verify one stochastic draft block using speculative rejection sampling.

    Returns the number of accepted draft tokens and the correction/bonus token.
    This first implementation intentionally supports batch size one, matching
    the sequential acceptance semantics used by DSpARK.
    """
    if temperature < 1e-5:
        raise ValueError("rejection sampling requires temperature > 0.")
    if proposed_tokens.ndim != 2 or proposed_tokens.shape[0] != 1:
        raise ValueError(
            "rejection sampling currently requires proposed_tokens shape [1, K]."
        )
    proposal_count = proposed_tokens.shape[1]
    if draft_logits.shape[:2] != (1, proposal_count):
        raise ValueError(
            "draft_logits must have shape [1, K, vocab] matching proposed_tokens."
        )
    if target_logits.shape[:2] != (1, proposal_count + 1):
        raise ValueError(
            "target_logits must have shape [1, K + 1, vocab]."
        )
    if draft_logits.shape[-1] != target_logits.shape[-1]:
        raise ValueError("draft and target logits must use the same vocabulary.")

    target_probs = _logits_to_probs(target_logits, temperature)
    if proposal_count == 0:
        return 0, _sample_from_probs(target_probs[:, 0, :])

    draft_probs = _logits_to_probs(draft_logits, temperature)
    token_index = proposed_tokens.unsqueeze(-1)
    selected_target = target_probs[:, :proposal_count, :].gather(
        dim=-1, index=token_index
    ).squeeze(-1)
    selected_draft = draft_probs.gather(
        dim=-1, index=token_index
    ).squeeze(-1)
    accept_probs = torch.minimum(
        torch.ones_like(selected_target),
        selected_target / selected_draft.clamp_min(1e-20),
    )
    accepted_prefix = (
        (torch.rand_like(accept_probs) < accept_probs)
        .to(torch.int64)
        .cumprod(dim=1)
    )
    accepted_count = int(accepted_prefix.sum().item())

    if accepted_count < proposal_count:
        next_token = _sample_residual(
            target_probs[:, accepted_count, :],
            draft_probs[:, accepted_count, :],
        )
    else:
        next_token = _sample_from_probs(
            target_probs[:, proposal_count, :]
        )
    return accepted_count, next_token


from .flashmtp_markov_head import (
    FlashMTPMarkovHead,
    MARKOV_HEAD_TYPES,
    MARKOV_OUTPUT_MODES,
    markov_output_uses_base_lm_head,
)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    k_len = k.size(-2)
    q_cos = cos[..., -q_len:, :]
    q_sin = sin[..., -q_len:, :]
    k_cos = cos[..., -k_len:, :]
    k_sin = sin[..., -k_len:, :]
    q_embed = (q * q_cos) + (rotate_half(q) * q_sin)
    k_embed = (k * k_cos) + (rotate_half(k) * k_sin)
    return q_embed, k_embed


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


def _infer_hs_embedding_offset(
    hidden_states: tuple | list, num_transformer_layers: int
) -> int:
    lt = len(hidden_states)
    if lt == num_transformer_layers:
        return 0
    if lt == num_transformer_layers + 1:
        return 1
    return 1 if lt > num_transformer_layers else 0


def build_ablation_target_layer_ids(
    num_transformer_layers: int, n_middle: int
) -> list[int]:
    """First + last transformer layer (0-based), plus ``n_middle`` evenly spaced interior layers."""
    L = num_transformer_layers
    if L <= 0:
        return [0]
    picked: set[int] = {0, L - 1}
    n_middle = max(0, int(n_middle))
    lo, hi = 1, L - 2
    if n_middle > 0 and lo <= hi:
        interior_span = hi - lo
        cap = hi - lo + 1
        n_take = min(n_middle, cap)
        if n_take == 1:
            picked.add((lo + hi) // 2)
        else:
            for i in range(n_take):
                if interior_span == 0:
                    idx = lo
                else:
                    idx = lo + int(round(i * interior_span / (n_take - 1)))
                picked.add(int(max(lo, min(hi, idx))))
    if n_middle > 0 and L > 2 and len(picked) - 2 < n_middle:
        print_on_rank0(
            f"[FlashMTP] num_middle_layers_n={n_middle} capped by depth; "
            f"using {len(picked)} target layers on L={L}."
        )
    return sorted(picked)


def gather_pivot_multilayer_inference(
    hidden_states: tuple | list,
    target_layer_ids: list[int],
    token_index: int,
    num_transformer_layers: int,
    include_embedding_chs: bool = False,
) -> torch.Tensor:
    """Return the fixed embedding prefix plus selected pivots for inference.

    With ``include_embedding_chs=True``, slot 0 is the target model's raw
    embedding output at ``token_index`` and the shape is ``(B, 1, 1+S, H)``.
    Otherwise the legacy ``(B, 1, S, H)`` layout is returned.
    """
    off = _infer_hs_embedding_offset(hidden_states, num_transformer_layers)
    pieces: list[torch.Tensor] = []
    if include_embedding_chs:
        if off != 1:
            raise ValueError(
                "Inference hidden_states must include the embedding output at index 0 "
                "when include_embedding_chs=True."
            )
        pieces.append(hidden_states[0][:, token_index, :].unsqueeze(1))
    for layer_id in target_layer_ids:
        layer_h = hidden_states[layer_id + off]
        pieces.append(layer_h[:, token_index, :].unsqueeze(1))
    return torch.stack(pieces, dim=2)


class PivotAttentionFuse(nn.Module):
    """Last layer attends previous layers via Qwen3Attention + RoPE positions 0..S-1; residual on last hs."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        attn_cfg = deepcopy(config)
        attn_cfg._attn_implementation = "eager"
        self.attention = Qwen3Attention(attn_cfg, layer_idx=0)
        self.rotary = Qwen3RotaryEmbedding(attn_cfg)
        self.out_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, S, H)
        bsz, n_blk, s_len, h = x.shape
        flat = x.view(bsz * n_blk, s_len, h)
        pos = (
            torch.arange(s_len, device=x.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(bsz * n_blk, -1)
        )
        pos_emb = self.rotary(flat, pos)
        attn_out, _ = self.attention(
            flat, position_embeddings=pos_emb, attention_mask=None, past_key_values=None
        )
        last_out = attn_out[:, -1, :]
        last_h = x[:, :, -1, :].reshape(bsz * n_blk, h)
        fused = self.out_norm(last_out + last_h).view(bsz, n_blk, h)
        return fused


class Qwen3FlashMTPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        chs_concat_mode: str,
        pivot_fuse_mode: str,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.pivot_fuse_mode = pivot_fuse_mode
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

        self.chs_concat_mode = chs_concat_mode

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]

        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)

        cos, sin = position_embeddings

        if self.pivot_fuse_mode == "prefix_condition" and ctx_len > 0:
            k_ctx = k_ctx.view(bsz, ctx_len, -1, self.head_dim).transpose(1, 2)
            k_noise = k_noise.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
            q, k_noise = apply_rotary_pos_emb(q, k_noise, cos, sin)
            k = torch.cat([k_ctx, k_noise], dim=2)
            k = self.k_norm(k)
            v_ctx = v_ctx.view(bsz, ctx_len, -1, self.head_dim).transpose(1, 2)
            v_noise = v_noise.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
            v = torch.cat([v_ctx, v_noise], dim=2)
        else:
            k = torch.cat([k_ctx, k_noise], dim=1).view(
                bsz, ctx_len + q_len, -1, self.head_dim
            )
            v = torch.cat([v_ctx, v_noise], dim=1).view(
                bsz, ctx_len + q_len, -1, self.head_dim
            )
            k = self.k_norm(k).transpose(1, 2)
            v = v.transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3FlashMTPDecoderLayer(GradientCheckpointingLayer):
    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        chs_concat_mode: str,
        pivot_fuse_mode: str,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3FlashMTPAttention(
            config=config,
            layer_idx=layer_idx,
            chs_concat_mode=chs_concat_mode,
            pivot_fuse_mode=pivot_fuse_mode,
        )
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class FlashMTPDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3FlashMTPDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        flashmtp_config = getattr(config, "flashmtp_config", {}) or {}
        self.chs_concat_mode = "feature"
        flashmtp_config["chs_concat_mode"] = "feature"
        self.pivot_fuse_mode = flashmtp_config.get("pivot_fuse_mode", "linear_fuse")
        if self.pivot_fuse_mode not in (
            "linear_fuse",
            "attention_fuse",
            "prefix_condition",
        ):
            raise ValueError(
                f"Unknown pivot_fuse_mode={self.pivot_fuse_mode!r}; "
                "expected linear_fuse | attention_fuse | prefix_condition"
            )
        self.num_middle_layers_n = int(flashmtp_config.get("num_middle_layers_n", 0))
        # Missing on old checkpoints: preserve their original CHS layout and
        # projection shapes. New training writes this field explicitly.
        self.include_embedding_chs = bool(
            flashmtp_config.get("include_embedding_chs", False)
        )
        flashmtp_config["include_embedding_chs"] = self.include_embedding_chs

        if flashmtp_config.get("target_layer_ids") is not None:
            self.target_layer_ids = list(flashmtp_config["target_layer_ids"])
        elif flashmtp_config.get("use_legacy_layer_sampling", False):
            self.target_layer_ids = build_target_layer_ids(
                config.num_target_layers, config.num_hidden_layers
            )
        else:
            self.target_layer_ids = build_ablation_target_layer_ids(
                config.num_target_layers, self.num_middle_layers_n
            )

        flashmtp_config.setdefault("pivot_fuse_mode", self.pivot_fuse_mode)
        flashmtp_config.setdefault("num_middle_layers_n", self.num_middle_layers_n)
        flashmtp_config["target_layer_ids"] = self.target_layer_ids
        self.local_position = bool(flashmtp_config.get("local_position", False))
        flashmtp_config["local_position"] = self.local_position
        self.left_shift = bool(flashmtp_config.get("left_shift", False))
        flashmtp_config["left_shift"] = self.left_shift
        if self.left_shift and int(config.block_size) <= 1:
            raise ValueError(
                "left_shift requires block_size >= 2 (anchor plus at least one "
                "draft prediction)."
            )
        self.markov_head_type = str(
            flashmtp_config.get("markov_head_type", "none")
        ).lower()
        self.markov_output_mode = str(
            flashmtp_config.get("markov_output_mode", "additive")
        ).lower()
        configured_markov_rank = int(flashmtp_config.get("markov_rank", 0))
        self.markov_rank = (
            configured_markov_rank if self.markov_head_type != "none" else 0
        )
        if self.markov_head_type not in ("none", *MARKOV_HEAD_TYPES):
            raise ValueError(
                f"Unknown markov_head_type={self.markov_head_type!r}; expected "
                f"none or one of {MARKOV_HEAD_TYPES}."
            )
        if self.markov_output_mode not in MARKOV_OUTPUT_MODES:
            raise ValueError(
                f"Unknown markov_output_mode={self.markov_output_mode!r}; "
                f"expected one of {MARKOV_OUTPUT_MODES}."
            )
        if self.markov_head_type != "none" and self.markov_rank <= 0:
            raise ValueError(
                f"markov_rank must be positive when a Markov head is enabled, "
                f"got {self.markov_rank}."
            )
        if (
            self.markov_head_type == "none"
            and self.markov_output_mode == "direct"
        ):
            raise ValueError(
                f"markov_output_mode={self.markov_output_mode!r} requires a Markov head."
            )
        flashmtp_config["markov_head_type"] = self.markov_head_type
        flashmtp_config["markov_output_mode"] = self.markov_output_mode
        flashmtp_config["markov_rank"] = self.markov_rank
        self.markov_head = (
            None
            if self.markov_head_type == "none"
            else FlashMTPMarkovHead(
                head_type=self.markov_head_type,
                vocab_size=config.vocab_size,
                markov_rank=self.markov_rank,
                hidden_size=config.hidden_size,
                markov_output_mode=self.markov_output_mode,
            )
        )
        self._compiled_serial_sampler_cache: dict[tuple[str, float], Callable] = {}
        config.flashmtp_config = flashmtp_config

        self.layers = nn.ModuleList(
            [
                Qwen3FlashMTPDecoderLayer(
                    config,
                    layer_idx,
                    self.chs_concat_mode,
                    self.pivot_fuse_mode,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.block_size = config.block_size
        self.mask_token_id = flashmtp_config.get("mask_token_id", None)
        self._last_decode_stats = {}

        # The raw embedding at anchor-1 is an extra fixed conditioning slot. It
        # is deliberately not included in target_layer_ids / num_middle_layers_n.
        s_layers = len(self.target_layer_ids)
        conditioning_slots = s_layers + int(self.include_embedding_chs)
        h = config.hidden_size
        if self.pivot_fuse_mode == "linear_fuse":
            self.fc = nn.Linear(conditioning_slots * h, h, bias=False)
            self.pivot_attn_fuse = None
            self.layer_depth_embedding = None
        elif self.pivot_fuse_mode == "attention_fuse":
            self.fc = None
            self.pivot_attn_fuse = PivotAttentionFuse(config)
            self.layer_depth_embedding = None
        else:
            self.fc = None
            self.pivot_attn_fuse = None
            self.layer_depth_embedding = nn.Embedding(config.num_target_layers, h)

        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        print_on_rank0(
            f"FlashMTP: pivot_fuse_mode={self.pivot_fuse_mode}, "
            f"num_middle_layers_n={self.num_middle_layers_n}, "
            f"target_layer_ids={self.target_layer_ids}, "
            f"include_embedding_chs={self.include_embedding_chs}, "
            f"local_position={self.local_position}, "
            f"left_shift={self.left_shift}, "
            f"markov_head_type={self.markov_head_type}, "
            f"markov_output_mode={self.markov_output_mode}, "
            f"markov_rank={self.markov_rank}"
        )

        self.post_init()

    @property
    def chs_len_per_block(self) -> int:
        """Physical KV-prefix length; selected CHS layer counts remain unchanged."""
        return (
            len(self.target_layer_ids) + int(self.include_embedding_chs)
            if self.pivot_fuse_mode == "prefix_condition"
            else 1
        )

    def get_last_decode_stats(self) -> dict:
        return dict(self._last_decode_stats)

    @property
    def draft_block_len(self) -> int:
        """Parallel draft slots per anchor (1 anchor + remaining MASK tokens)."""
        if self.left_shift:
            return self.block_size - 1
        return self.block_size

    @property
    def proposal_length(self) -> int:
        """Draft tokens proposed after the anchor; total span is block_size."""
        return self.block_size - 1

    @property
    def max_verify_block_size(self) -> int:
        """Anchor-inclusive target verification window (equals config block_size)."""
        return self.proposal_length + 1

    def set_config_block_size(self, block_size: int) -> None:
        """Update config block_size while keeping left_shift semantics unchanged."""
        self.block_size = int(block_size)
        self.config.block_size = int(block_size)

    def _prediction_hidden(self, block_hidden: torch.Tensor) -> torch.Tensor:
        """Select hidden slots carrying logits under the checkpoint alignment."""
        draft_len = self.draft_block_len
        hidden = block_hidden[:, -draft_len:, :]
        if not self.left_shift:
            # Legacy training supervises slots 1..block_size-1; slot 0 is anchor context only.
            hidden = hidden[:, 1:, :]
        return hidden

    def _fuse_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """Fuse ``[raw embedding, selected CHS layers]`` conditioning slots."""
        bsz, n_blk, s_len, h = target_hidden.shape
        expected_slots = len(self.target_layer_ids) + int(self.include_embedding_chs)
        if s_len != expected_slots:
            layout = (
                "one fixed embedding slot followed by "
                if self.include_embedding_chs
                else ""
            )
            raise ValueError(
                f"target_hidden must contain {layout}{len(self.target_layer_ids)} "
                f"selected CHS layers; got {s_len} slots."
            )
        if self.pivot_fuse_mode == "linear_fuse":
            flat = target_hidden.reshape(bsz, n_blk, s_len * h)
            return self.hidden_norm(self.fc(flat))
        if self.pivot_fuse_mode == "attention_fuse":
            assert self.pivot_attn_fuse is not None
            return self.pivot_attn_fuse(target_hidden)
        assert self.layer_depth_embedding is not None
        depth_ids = torch.tensor(
            self.target_layer_ids, device=target_hidden.device, dtype=torch.long
        )
        depth_emb = self.layer_depth_embedding(depth_ids).view(
            1, 1, len(self.target_layer_ids), h
        )
        if self.include_embedding_chs:
            raw_embedding = target_hidden[:, :, :1, :]
            layer_ctx = target_hidden[:, :, 1:, :] + depth_emb
            ctx = torch.cat([raw_embedding, layer_ctx], dim=2)
        else:
            ctx = target_hidden + depth_emb
        return self.hidden_norm(ctx.reshape(bsz, n_blk * s_len, h))

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        rotary_position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states = noise_embedding
        assert target_hidden is not None and target_hidden.ndim == 4
        noise_len = hidden_states.shape[1]
        if position_ids.shape[1] != noise_len:
            draft_pos = position_ids[:, -noise_len:]
        else:
            draft_pos = position_ids

        target_hidden = self._fuse_target_hidden(target_hidden)
        rotary_pos = (
            rotary_position_ids if rotary_position_ids is not None else draft_pos
        )
        total_len = rotary_pos.shape[1]
        dummy = hidden_states.new_zeros(
            hidden_states.shape[0], total_len, hidden_states.shape[-1]
        )
        position_embeddings = self.rotary_emb(dummy, rotary_pos)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=draft_pos,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    def sample_draft_tokens(
        self,
        *,
        draft_hidden: torch.Tensor,
        lm_head: nn.Module,
        first_prev_token_ids: torch.Tensor,
        temperature: float = 0.0,
        compile_serial_head: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample standard FlashMTP draft positions using configured head semantics."""
        base_logits = None
        if self.markov_head is None or markov_output_uses_base_lm_head(
            self.markov_output_mode
        ):
            base_logits = lm_head(draft_hidden)
        if self.markov_head is None:
            assert base_logits is not None
            return sample(base_logits, temperature), base_logits
        if compile_serial_head:
            cache_key = (self.markov_output_mode, float(temperature))
            compiled_sampler = self._compiled_serial_sampler_cache.get(cache_key)
            if compiled_sampler is None:
                markov_head = self.markov_head
                output_mode = self.markov_output_mode
                fixed_temperature = float(temperature)
                if markov_output_uses_base_lm_head(output_mode):
                    def serial_sampler(
                        hidden_states: torch.Tensor,
                        previous_ids: torch.Tensor,
                        additive_logits: torch.Tensor,
                    ) -> tuple[torch.Tensor, torch.Tensor]:
                        return markov_head.sample_block_tokens(
                            hidden_states=hidden_states,
                            first_prev_token_ids=previous_ids,
                            output_mode=output_mode,
                            base_logits=additive_logits,
                            temperature=fixed_temperature,
                        )
                else:
                    def serial_sampler(
                        hidden_states: torch.Tensor,
                        previous_ids: torch.Tensor,
                    ) -> tuple[torch.Tensor, torch.Tensor]:
                        return markov_head.sample_block_tokens(
                            hidden_states=hidden_states,
                            first_prev_token_ids=previous_ids,
                            output_mode=output_mode,
                            base_logits=None,
                            temperature=fixed_temperature,
                        )
                compiled_sampler = torch.compile(
                    serial_sampler,
                    mode="reduce-overhead",
                    fullgraph=True,
                )
                self._compiled_serial_sampler_cache[cache_key] = compiled_sampler
            if base_logits is None:
                return compiled_sampler(draft_hidden, first_prev_token_ids)
            return compiled_sampler(
                draft_hidden, first_prev_token_ids, base_logits
            )
        return self.markov_head.sample_block_tokens(
            hidden_states=draft_hidden,
            first_prev_token_ids=first_prev_token_ids,
            output_mode=self.markov_output_mode,
            base_logits=base_logits,
            temperature=temperature,
        )

    @staticmethod
    def _format_token_topk(
        logits: torch.Tensor,
        tokenizer,
        top_k: int,
    ) -> list[dict]:
        probs = torch.softmax(logits.float(), dim=-1)
        top_k = max(int(top_k), 1)
        top_probs, top_ids = torch.topk(probs, k=min(top_k, probs.shape[-1]), dim=-1)
        entries = []
        for token_id, prob in zip(top_ids.tolist(), top_probs.tolist()):
            entries.append(
                {
                    "id": int(token_id),
                    "token": tokenizer.decode([int(token_id)]),
                    "confidence": float(prob),
                }
            )
        return entries

    @staticmethod
    def _format_token_list(token_ids: torch.Tensor, tokenizer) -> list[dict]:
        return [
            {"id": int(tid), "token": tokenizer.decode([int(tid)])}
            for tid in token_ids.view(-1).tolist()
        ]

    @staticmethod
    def _serialize_topk_entries(entries: list[dict]) -> list[dict]:
        return [
            {
                "token_id": int(e["id"]),
                "token": e["token"],
                "confidence": float(e["confidence"]),
            }
            for e in entries
        ]

    @torch.inference_mode()
    def spec_generate_with_profile(
        self,
        target: nn.Module,
        tokenizer,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
        top_k: int = 4,
        print_fn: Callable[..., None] = print,
        profile_records: Optional[list] = None,
    ) -> torch.LongTensor:
        """Same decode path as ``spec_generate``, plus optional JSONL-friendly profiling.

        Legacy checkpoints use hidden slots ``1 .. block_size-1``.  With
        ``left_shift=true``, ``block_size`` is the total span (anchor plus
        ``block_size-1`` drafts); the draft block has ``block_size-1`` slots.
        """
        self.eval()
        self._last_decode_stats = {
            "accept_lengths": [],
            "target_total_time": 0.0,
            "draft_total_time": 0.0,
            "steps": 0,
        }
        if self.mask_token_id is None:
            raise ValueError(
                "FlashMTPDraftModel.mask_token_id is None. Set flashmtp_config['mask_token_id'] "
                "or draft_model.mask_token_id before spec_generate_with_profile()."
            )
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens
        draft_block_len = self.draft_block_len
        proposal_length = self.proposal_length

        output_ids = torch.full(
            (1, max_length + proposal_length + 1),
            self.mask_token_id,
            dtype=torch.long,
            device=target.device,
        )
        position_ids = torch.arange(
            output_ids.shape[1], device=target.device
        ).unsqueeze(0)
        past_key_values_target = DynamicCache()

        if target.device.type == "cuda":
            torch.cuda.synchronize(target.device)
        target_start = time.perf_counter()
        output = target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )
        if target.device.type == "cuda":
            torch.cuda.synchronize(target.device)
        self._last_decode_stats["target_total_time"] += (
            time.perf_counter() - target_start
        )

        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
            output.logits, temperature
        )
        anchor_token = output_ids[:, num_input_tokens : num_input_tokens + 1]
        print_fn(
            f"\n[Prefill anchor] pos={num_input_tokens} "
            f"token={self._format_token_list(anchor_token[0], tokenizer)}"
        )

        prefill_slot0_confidence: Optional[float] = None
        if profile_records is not None:
            logits_prefill = output.logits[0, -1].float()
            probs_prefill = torch.softmax(logits_prefill, dim=-1)
            aid = int(output_ids[0, num_input_tokens].item())
            prefill_slot0_confidence = float(probs_prefill[aid])

        target_hidden = gather_pivot_multilayer_inference(
            output.hidden_states,
            self.target_layer_ids,
            -1,
            self.config.num_target_layers,
            include_embedding_chs=self.include_embedding_chs,
        )

        start = input_ids.shape[1]
        while start < max_length:
            spec_step_idx = int(self._last_decode_stats["steps"])
            print_fn(f"\n[Spec step {spec_step_idx}] start={start}")
            draft_input_ids = output_ids[:, start : start + draft_block_len].clone()
            draft_target_pos = position_ids[:, start : start + draft_block_len]
            if self.local_position:
                draft_block_pos = torch.arange(
                    1, draft_block_len + 1, device=target.device, dtype=torch.long
                ).unsqueeze(0)
            else:
                draft_block_pos = draft_target_pos
            block_start_abs = int(start)
            slot0_tid = int(output_ids[0, start].item())

            noise_embedding = target.model.embed_tokens(draft_input_ids)
            if target.device.type == "cuda":
                torch.cuda.synchronize(target.device)
            draft_start = time.perf_counter()
            chs = self.chs_len_per_block
            if self.local_position:
                ctx_pos_part = torch.zeros(
                    1, chs, dtype=torch.long, device=target.device
                )
            else:
                ctx_pos_part = torch.full(
                    (1, chs),
                    start - 1,
                    dtype=torch.long,
                    device=target.device,
                )
            full_rotary = torch.cat([ctx_pos_part, draft_block_pos], dim=-1)
            block_hidden = self(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=draft_block_pos,
                rotary_position_ids=full_rotary,
                past_key_values=None,
                use_cache=False,
                is_causal=False,
            )
            draft_hidden = self._prediction_hidden(block_hidden)
            lm_head = target.lm_head
            sampled_draft_tokens, draft_logits = self.sample_draft_tokens(
                draft_hidden=draft_hidden,
                lm_head=lm_head,
                first_prev_token_ids=draft_input_ids[:, 0],
                temperature=temperature,
            )
            if target.device.type == "cuda":
                torch.cuda.synchronize(target.device)
            self._last_decode_stats["draft_total_time"] += (
                time.perf_counter() - draft_start
            )

            draft_topk_by_slot: dict[int, list[dict]] = {}
            for slot in range(1, proposal_length + 1):
                topk_entries = self._format_token_topk(
                    draft_logits[0, slot - 1], tokenizer, top_k
                )
                if profile_records is not None:
                    draft_topk_by_slot[slot] = self._serialize_topk_entries(
                        topk_entries
                    )
                print_fn(
                    f"  [Draft slot {slot} | abs_pos={start + slot}] "
                    f"top{top_k}={topk_entries}"
                )

            verify_output_ids = torch.cat(
                [draft_input_ids[:, :1], sampled_draft_tokens], dim=1
            )

            draft_probs_on_logits: Optional[torch.Tensor] = None
            if profile_records is not None:
                draft_probs_on_logits = torch.softmax(draft_logits.float(), dim=-1)

            pending_profile_chunk: list[dict] = []
            if profile_records is not None:
                slot0_row = {
                    "abs_pos": block_start_abs,
                    "slot": 0,
                    "token_id": slot0_tid,
                    "token": tokenizer.decode([slot0_tid]),
                }
                if (
                    prefill_slot0_confidence is not None
                    and block_start_abs == num_input_tokens
                ):
                    slot0_row["confidence"] = round(prefill_slot0_confidence, 4)
                pending_profile_chunk.append(slot0_row)
                for s in sorted(draft_topk_by_slot):
                    cand_list = [
                        {
                            "id": int(c["token_id"]),
                            "t": c["token"],
                            "p": round(float(c["confidence"]), 3),
                        }
                        for c in draft_topk_by_slot[s]
                    ]
                    sampled_tid = int(verify_output_ids[0, s].item())
                    row = {
                        "abs_pos": block_start_abs + int(s),
                        "slot": int(s),
                        "top_k": int(top_k),
                        "candidates": cand_list,
                        "sampled_token_id": sampled_tid,
                        "sampled_token": tokenizer.decode([sampled_tid]),
                        "draft_p_sampled": round(
                            float(draft_probs_on_logits[0, s - 1, sampled_tid].item()),
                            4,
                        ),
                    }
                    pending_profile_chunk.append(row)

            if target.device.type == "cuda":
                torch.cuda.synchronize(target.device)
            target_start = time.perf_counter()
            verify_position_ids = position_ids[
                :, start : start + proposal_length + 1
            ]
            output = target(
                verify_output_ids,
                position_ids=verify_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
            )
            if target.device.type == "cuda":
                torch.cuda.synchronize(target.device)
            self._last_decode_stats["target_total_time"] += (
                time.perf_counter() - target_start
            )

            posterior = sample(output.logits, temperature)
            acceptance_length = (
                (verify_output_ids[:, 1:] == posterior[:, :-1])
                .cumprod(dim=1)
                .sum(dim=1)[0]
                .item()
            )
            accepted_tokens = verify_output_ids[:, : acceptance_length + 1]
            print_fn(
                f"  [Accept] length={acceptance_length + 1} "
                f"tokens={self._format_token_list(accepted_tokens[0], tokenizer)}"
            )

            target_verify: list[dict] = []
            for target_pos in range(min(acceptance_length + 1, output.logits.shape[1])):
                role = "accepted" if target_pos < acceptance_length else "correction"
                target_token = posterior[:, target_pos : target_pos + 1]
                target_top3 = self._format_token_topk(
                    output.logits[0, target_pos], tokenizer, 3
                )
                if profile_records is not None:
                    tid = int(posterior[0, target_pos].item())
                    probs_tp = torch.softmax(
                        output.logits[0, target_pos].float(), dim=-1
                    )
                    target_verify.append(
                        {
                            "verify_step": int(target_pos),
                            "role": role,
                            "abs_pos": int(block_start_abs + target_pos + 1),
                            "chosen_token_id": tid,
                            "chosen_token": tokenizer.decode([tid]),
                            "target_p_chosen": round(float(probs_tp[tid].item()), 4),
                        }
                    )
                print_fn(
                    f"  [Target {role} pos={start + target_pos + 1}] "
                    f"token={self._format_token_list(target_token[0], tokenizer)} "
                    f"top3={target_top3}"
                )

            output_ids[:, start : start + acceptance_length + 1] = accepted_tokens
            output_ids[:, start + acceptance_length + 1] = posterior[
                :, acceptance_length
            ]
            start += acceptance_length + 1
            past_key_values_target.crop(start)
            pivot_index = min(acceptance_length, output.hidden_states[0].shape[1] - 1)
            target_hidden = gather_pivot_multilayer_inference(
                output.hidden_states,
                self.target_layer_ids,
                pivot_index,
                self.config.num_target_layers,
                include_embedding_chs=self.include_embedding_chs,
            )
            accept_len_out = int(acceptance_length + 1)
            self._last_decode_stats["accept_lengths"].append(accept_len_out)
            if profile_records is not None:
                profile_records.append(
                    {
                        "block_start": block_start_abs,
                        "accept_length": accept_len_out,
                        "speculative_match_count": int(acceptance_length),
                        "target_verify": target_verify,
                    }
                )
                profile_records.extend(pending_profile_chunk)
            self._last_decode_stats["steps"] += 1
            if stop_token_ids is not None and any(
                stop_token_id in output_ids[:, num_input_tokens:]
                for stop_token_id in stop_token_ids
            ):
                break

        output_ids = output_ids[:, :max_length]
        output_ids = output_ids[:, output_ids[0] != self.mask_token_id]
        if stop_token_ids is not None:
            stop_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
            stop_token_indices = torch.isin(
                output_ids[0][num_input_tokens:], stop_tensor
            ).nonzero(as_tuple=True)[0]
            if stop_token_indices.numel() > 0:
                output_ids = output_ids[
                    :, : num_input_tokens + stop_token_indices[0] + 1
                ]

        return output_ids

    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
        decode_timing_after_first_token: bool = False,
        verify_block_size: Optional[int] = None,
        stochastic_verification_mode: str = "match",
        compile_serial_head: bool = False,
    ):
        self.eval()
        stochastic_verification_mode = _validate_stochastic_verification_mode(
            stochastic_verification_mode
        )
        self._last_decode_stats = {
            "accept_lengths": [],
            "decode_wall_time": 0.0,
            "target_total_time": 0.0,
            "draft_total_time": 0.0,
            "steps": 0,
            "verification_mode": stochastic_verification_mode,
            "compile_serial_head": bool(compile_serial_head),
        }
        bsz = input_ids.shape[0]
        use_rejection_sampling = (
            temperature >= 1e-5
            and stochastic_verification_mode == "rejection"
        )
        if use_rejection_sampling and bsz != 1:
            raise ValueError(
                "stochastic rejection sampling currently requires batch size 1."
            )
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens

        draft_block_len = self.draft_block_len
        proposal_length = self.proposal_length
        verify_block_size = (
            proposal_length + 1
            if verify_block_size is None
            else int(verify_block_size)
        )
        if not 1 <= verify_block_size <= proposal_length + 1:
            raise ValueError(
                f"verify_block_size must be in [1, {proposal_length + 1}], got "
                f"{verify_block_size}"
            )
        output_ids = torch.full(
            (bsz, max_length + proposal_length + 1),
            self.mask_token_id,
            dtype=torch.long,
            device=target.device,
        )
        position_ids = (
            torch.arange(output_ids.shape[1], device=target.device)
            .unsqueeze(0)
            .expand(bsz, -1)
        )

        past_key_values_target = DynamicCache()

        # Prefill stage (not included in decode wall time)
        output = target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )

        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
            output.logits, temperature
        )
        target_hidden = gather_pivot_multilayer_inference(
            output.hidden_states,
            self.target_layer_ids,
            -1,
            self.config.num_target_layers,
            include_embedding_chs=self.include_embedding_chs,
        )

        # Decode stage: single cuda-synced wall clock (draft + target + bookkeeping)
        decode_start: float | None = (
            None if decode_timing_after_first_token else _cuda_sync_time(target.device)
        )
        acceptance_lengths = []
        start = input_ids.shape[1]
        while start < max_length:
            draft_input_ids = output_ids[:, start : start + draft_block_len].clone()
            draft_target_pos = position_ids[:, start : start + draft_block_len]
            if self.local_position:
                draft_block_pos = (
                    torch.arange(
                        1, draft_block_len + 1, device=target.device, dtype=torch.long
                    )
                    .unsqueeze(0)
                    .expand(bsz, -1)
                )
            else:
                draft_block_pos = draft_target_pos
            noise_embedding = target.model.embed_tokens(draft_input_ids)
            chs = self.chs_len_per_block
            if self.local_position:
                ctx_pos_part = torch.zeros(
                    bsz, chs, dtype=torch.long, device=target.device
                )
            else:
                ctx_pos_part = torch.full(
                    (bsz, chs),
                    start - 1,
                    dtype=torch.long,
                    device=target.device,
                )
            full_rotary = torch.cat([ctx_pos_part, draft_block_pos], dim=-1)
            block_hidden = self(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=draft_block_pos,
                rotary_position_ids=full_rotary,
                past_key_values=None,
                use_cache=False,
                is_causal=False,
            )
            draft_hidden = self._prediction_hidden(block_hidden)
            lm_head = target.lm_head
            draft_temperature = temperature if use_rejection_sampling else 0.0
            sampled_draft_tokens, draft_logits = self.sample_draft_tokens(
                draft_hidden=draft_hidden,
                lm_head=lm_head,
                first_prev_token_ids=draft_input_ids[:, 0],
                temperature=draft_temperature,
                compile_serial_head=compile_serial_head,
            )
            all_verify_output_ids = torch.cat(
                [draft_input_ids[:, :1], sampled_draft_tokens], dim=1
            )

            # The draft always consumes and predicts the full configured block.  Only
            # the requested prefix is sent to the target; the remaining draft tokens
            # are deliberately discarded before verification.
            verify_output_ids = all_verify_output_ids[:, :verify_block_size]
            verify_position_ids = position_ids[:, start : start + verify_block_size]
            output = target(
                verify_output_ids,
                position_ids=verify_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
            )

            if use_rejection_sampling:
                proposal_count = verify_block_size - 1
                acceptance_length, next_token = rejection_sample_verify(
                    proposed_tokens=verify_output_ids[:, 1:],
                    draft_logits=draft_logits[:, :proposal_count, :],
                    target_logits=output.logits,
                    temperature=temperature,
                )
            else:
                posterior = sample(output.logits, temperature)
                acceptance_lengths_per_row = (
                    (verify_output_ids[:, 1:] == posterior[:, :-1])
                    .cumprod(dim=1)
                    .sum(dim=1)
                )
                acceptance_length = int(
                    acceptance_lengths_per_row.min().item()
                )
                next_token = posterior[:, acceptance_length]
            output_ids[:, start : start + acceptance_length + 1] = verify_output_ids[
                :, : acceptance_length + 1
            ]
            output_ids[:, start + acceptance_length + 1] = next_token
            start += acceptance_length + 1
            past_key_values_target.crop(start)
            pivot_index = min(acceptance_length, output.hidden_states[0].shape[1] - 1)
            target_hidden = gather_pivot_multilayer_inference(
                output.hidden_states,
                self.target_layer_ids,
                pivot_index,
                self.config.num_target_layers,
                include_embedding_chs=self.include_embedding_chs,
            )
            acceptance_lengths.append(acceptance_length + 1)
            self._last_decode_stats["accept_lengths"].append(acceptance_length + 1)
            self._last_decode_stats["steps"] += 1

            if decode_timing_after_first_token and decode_start is None:
                decode_start = _cuda_sync_time(target.device)

            if stop_token_ids is not None and any(
                stop_token_id in output_ids[:, num_input_tokens:]
                for stop_token_id in stop_token_ids
            ):
                break
        if decode_start is None:
            decode_start = _cuda_sync_time(target.device)
        decode_wall_time = _cuda_sync_time(target.device) - decode_start
        self._last_decode_stats["decode_wall_time"] = decode_wall_time
        # Legacy fields: whole decode wall (no draft/target split)
        self._last_decode_stats["target_total_time"] = decode_wall_time
        self._last_decode_stats["draft_total_time"] = 0.0

        output_ids = output_ids[:, :max_length]
        output_ids = output_ids[:, output_ids[0] != self.mask_token_id]
        if stop_token_ids is not None:
            stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
            stop_token_indices = torch.isin(
                output_ids[0][num_input_tokens:], stop_token_ids
            ).nonzero(as_tuple=True)[0]
            if stop_token_indices.numel() > 0:
                output_ids = output_ids[
                    :, : num_input_tokens + stop_token_indices[0] + 1
                ]

        return output_ids
