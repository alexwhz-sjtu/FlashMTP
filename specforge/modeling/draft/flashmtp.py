import math
from typing import Callable, List, Optional, Sequence, Union

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


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def stack_hidden_states_for_positions(
    hidden_states: Union[Sequence[torch.Tensor], torch.Tensor],
    context_positions: torch.Tensor,
) -> torch.Tensor:
    """Gather embed + every decoder layer at context_positions -> (B, N, S, H).

    hidden_states: tuple of (B, T, H) for each of S sources, or (B, T, S, H) stacked.
    context_positions: (B, N) indices along T.
    """
    if isinstance(hidden_states, torch.Tensor) and hidden_states.dim() == 4:
        b, t, s, d = hidden_states.shape
        n = context_positions.size(1)
        b_idx = (
            torch.arange(b, device=hidden_states.device)[:, None].expand(b, n)
        )
        return hidden_states[b_idx, context_positions, :, :]
    if isinstance(hidden_states, torch.Tensor):
        raise ValueError("Expected tuple of (B, T, H) or stacked (B, T, S, H).")
    out = []
    for h in hidden_states:
        g = torch.gather(
            h,
            1,
            context_positions.unsqueeze(-1).expand(
                -1, -1, h.size(-1)
            ),
        )
        out.append(g)
    return torch.stack(out, dim=2)


def build_target_layer_ids(num_target_layers: int, num_layers: int = 5) -> list[int]:
    """DFlash-style evenly spaced target layers, excluding very early/final layers."""
    if num_layers <= 1:
        return [num_target_layers // 2]
    start = 1
    end = max(start, num_target_layers - 3)
    span = end - start
    return [
        int(round(start + (i * span) / (num_layers - 1)))
        for i in range(num_layers)
    ]


def _sinusoidal_position_embedding(
    position_ids: torch.Tensor,
    dim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create deterministic absolute position features for arbitrary position ids."""
    half_dim = dim // 2
    inv_freq = torch.exp(
        torch.arange(0, half_dim, device=position_ids.device, dtype=torch.float32)
        * (-math.log(10000.0) / max(half_dim, 1))
    )
    angles = position_ids.float().unsqueeze(-1) * inv_freq
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb.to(dtype=dtype)


def _select_and_concat_target_layers(
    hidden_states: Union[Sequence[torch.Tensor], torch.Tensor],
    layer_ids: Sequence[int],
) -> torch.Tensor:
    """Select target decoder layers and concatenate on feature dim."""
    if isinstance(hidden_states, torch.Tensor):
        if hidden_states.dim() == 4:
            selected_states = [hidden_states[:, :, i, :] for i in layer_ids]
        elif hidden_states.dim() == 3:
            return hidden_states
        else:
            raise ValueError("Expected hidden_states tensor with dim 3 or 4.")
    else:
        offset = 1  # HF hidden_states[0] is embedding output; layers start at 1.
        selected_states = [hidden_states[layer_id + offset] for layer_id in layer_ids]
    return torch.cat(selected_states, dim=-1)


class CHSHistoryCrossAttentionFusion(nn.Module):
    """Attempt 2 CHS: DFlash-style layer concat + one-head history cross-attn.

    For every sampled anchor p, query is the fused HS at p-1 and keys/values are the
    fused HS of all positions <= p-1. Absolute sinusoidal position features encode the
    natural history order before the one-head cross-attention.
    """

    def __init__(
        self,
        config: Qwen3Config,
        target_layer_ids: Sequence[int],
    ):
        super().__init__()
        self.target_layer_ids = list(target_layer_ids)
        d = config.hidden_size
        self.fc = nn.Linear(len(self.target_layer_ids) * d, d, bias=False)
        self.hidden_norm = Qwen3RMSNorm(d, eps=config.rms_norm_eps)
        self.q_proj = nn.Linear(d, d, bias=config.attention_bias)
        self.k_proj = nn.Linear(d, d, bias=config.attention_bias)
        self.v_proj = nn.Linear(d, d, bias=config.attention_bias)
        self.o_proj = nn.Linear(d, d, bias=config.attention_bias)
        self.q_norm = Qwen3RMSNorm(d, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(d, eps=config.rms_norm_eps)
        self.out_norm = Qwen3RMSNorm(d, eps=config.rms_norm_eps)
        self.attention_dropout = config.attention_dropout
        self.scaling = d**-0.5

    def forward(
        self,
        hidden_states: Union[Sequence[torch.Tensor], torch.Tensor],
        context_positions: torch.Tensor,
    ) -> torch.Tensor:
        fused_history = self.hidden_norm(
            self.fc(_select_and_concat_target_layers(hidden_states, self.target_layer_ids))
        )
        bsz, seq_len, dim = fused_history.shape
        n = context_positions.size(1)
        safe_context_positions = context_positions.clamp(0, seq_len - 1)

        batch_idx = torch.arange(bsz, device=fused_history.device)[:, None].expand(bsz, n)
        query_states = fused_history[batch_idx, safe_context_positions, :]

        history_pos = torch.arange(seq_len, device=fused_history.device).view(1, seq_len)
        history_pos_emb = _sinusoidal_position_embedding(
            history_pos.expand(bsz, -1), dim, fused_history.dtype
        )
        query_pos_emb = _sinusoidal_position_embedding(
            safe_context_positions, dim, fused_history.dtype
        )

        positioned_history = fused_history + history_pos_emb
        positioned_query = query_states + query_pos_emb

        q = self.q_norm(self.q_proj(positioned_query))
        k = self.k_norm(self.k_proj(positioned_history))
        v = self.v_proj(fused_history)

        attn_scores = torch.einsum("bnd,btd->bnt", q, k) * self.scaling
        history_mask = history_pos <= safe_context_positions.unsqueeze(-1)
        attn_scores = attn_scores.masked_fill(~history_mask, torch.finfo(attn_scores.dtype).min)
        attn_weights = torch.softmax(attn_scores.float(), dim=-1).to(dtype=attn_scores.dtype)
        attn_weights = torch.dropout(
            attn_weights, p=self.attention_dropout, train=self.training
        )
        attn_output = torch.einsum("bnt,btd->bnd", attn_weights, v)
        return self.out_norm(query_states + self.o_proj(attn_output))


class CHSQueryFusion(nn.Module):
    """Depth-axis non-causal self-attn: last slot (top-layer HS) queries all S sources.

    Sequence order is [embed, layer_0, …, layer_{L-1}] with S=L+1. RoPE uses positions
    0..S-1. The output is the last position's representation (last layer's Q attends
    to every K/V) and is fed as the single CHS condition token per block.
    """

    def __init__(
        self,
        config: Qwen3Config,
        num_chs_source_tokens: int,
        chs_fusion_layer_idx: int = 0,
    ):
        super().__init__()
        d = config.hidden_size
        self.S = num_chs_source_tokens
        self.input_layernorm = Qwen3RMSNorm(d, eps=config.rms_norm_eps)
        # Same Qwen3 self-attn as the draft stack; not causal (no attention mask)
        self.attn = Qwen3Attention(config, layer_idx=chs_fusion_layer_idx)
        self.attn.is_causal = False
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.out_norm = Qwen3RMSNorm(d, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, S, d) — embed + each target layer at anchor-1
        bsz, n, s, d = x.shape
        if s != self.S:
            raise ValueError(
                f"CHSQueryFusion: expected S={self.S} stacked inputs, got {s}."
            )
        flat = x.view(bsz * n, s, d)
        slen = s
        position_ids = torch.arange(
            0, slen, device=flat.device, dtype=torch.long
        ).view(1, -1)
        position_ids = position_ids.expand(flat.size(0), -1)
        position_embeddings = self.rotary_emb(flat, position_ids)
        h = self.input_layernorm(flat)
        attn_out, _ = self.attn(
            h,
            position_embeddings,
            attention_mask=None,
            past_key_value=None,
        )
        # Last index = final decoder layer; its output is the fusion readout
        out = attn_out[:, -1, :].view(bsz, n, d)
        return self.out_norm(out)


class Qwen3FlashMTPAttention(nn.Module):
    """Non-causal self-attn: K/V = context + draft, full RoPE on k."""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
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

        k = torch.cat([k_ctx, k_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        v = torch.cat([v_ctx, v_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )

        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = position_embeddings
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
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3FlashMTPAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
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
        ] = None,
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


def extract_stacked_chs(
    hidden_states: Union[Sequence[torch.Tensor], torch.Tensor],
    time_indices: torch.Tensor,
) -> torch.Tensor:
    """Stacked CHS at per-token indices (B, T) from target forward, shape (B, T, S, H)."""
    if isinstance(hidden_states, torch.Tensor) and hidden_states.dim() == 4:
        b, t, s, d = hidden_states.shape
        npos = time_indices.size(1)
        b_idx = torch.arange(b, device=hidden_states.device)[:, None].expand(b, npos)
        return hidden_states[b_idx, time_indices, :, :]
    if isinstance(hidden_states, torch.Tensor):
        raise ValueError("Expected tuple of (B, T, H) or stacked (B, T, S, H).")
    out = []
    for h in hidden_states:
        g = torch.gather(
            h, 1, time_indices.unsqueeze(-1).expand(-1, -1, h.size(-1))
        )
        out.append(g)
    return torch.stack(out, dim=2)


def _merged_flashmtp_config(config: Qwen3Config) -> dict:
    """Train 写入 flashmtp_config；部分 checkpoint 仅存 dflashconfig，需合并。"""
    fc = getattr(config, "flashmtp_config", None) or {}
    dc = getattr(config, "dflashconfig", None) or {}
    if not fc and not dc:
        raw = config.to_dict() if hasattr(config, "to_dict") else {}
        fc = raw.get("flashmtp_config") or {}
        dc = raw.get("dflashconfig") or {}
    merged = {**(dict(dc) if dc else {}), **(dict(fc) if fc else {})}
    return merged


class FlashMTPDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3FlashMTPDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        flashmtp_config = _merged_flashmtp_config(config)
        self.chs_fusion_type = flashmtp_config.get(
            "chs_fusion_type", "attempt2_history_cross_attention"
        )
        nsrc = int(
            flashmtp_config.get(
                "num_chs_source_tokens",
                getattr(config, "num_target_layers", 0) + 1,
            )
        )
        self.num_chs_source_tokens = nsrc
        chs_fusion_layer_idx = int(
            flashmtp_config.get("chs_fusion_layer_idx", 0)
        )
        self.layers = nn.ModuleList(
            [
                Qwen3FlashMTPDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.target_layer_ids = flashmtp_config.get(
            "target_layer_ids",
            build_target_layer_ids(getattr(config, "num_target_layers", 0), 5),
        )
        if self.chs_fusion_type == "attempt2_history_cross_attention":
            self.chs_fusion = CHSHistoryCrossAttentionFusion(
                config,
                target_layer_ids=self.target_layer_ids,
            )
        else:
            self.chs_fusion = CHSQueryFusion(
                config,
                nsrc,
                chs_fusion_layer_idx=chs_fusion_layer_idx,
            )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.block_size = config.block_size
        self.mask_token_id = flashmtp_config.get("mask_token_id", None)
        if self.chs_fusion_type == "attempt2_history_cross_attention":
            print_on_rank0(
                "FlashMTP Attempt2: target_layer_ids="
                f"{self.target_layer_ids}, CHS fusion=feature concat+FC "
                "+ single-head history cross-attn"
            )
        else:
            print_on_rank0(
                "FlashMTP: num_chs_source_tokens="
                f"{self.num_chs_source_tokens} (embed+layers), "
                "CHS fusion=depth self-attn, readout=last layer slot"
            )
        self.post_init()

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        context_positions: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states = noise_embedding
        if target_hidden is None:
            raise ValueError("target_hidden is required.")
        bs = self.block_size
        dlen = hidden_states.size(1)
        if dlen % bs != 0:
            raise ValueError(
                f"Draft sequence length {dlen} must be divisible by block_size {bs}."
            )
        n_blocks = dlen // bs
        first_of_block = position_ids[:, ::bs]
        inferred_context_positions = (first_of_block - 1).clamp(min=0)
        if context_positions is None:
            context_positions = inferred_context_positions
        if self.chs_fusion_type == "attempt2_history_cross_attention":
            target_hidden = self.chs_fusion(target_hidden, context_positions)
        else:
            # (B, N, S, H) before fusion
            target_hidden = self.chs_fusion(target_hidden)
        # RoPE must cover K/V = [context | draft]. Previously only `hidden_states` (draft)
        # was used, so cos/sin length did not match concat(k_ctx, k_noise) in attention.
        if target_hidden.size(1) != n_blocks:
            raise ValueError(
                f"Fused target_hidden len {target_hidden.size(1)} != num_blocks {n_blocks}."
            )
        # Block i draft starts at position_ids[:, i * bs] (= anchor i); CHS uses anchor-1.
        ctx_position_ids = context_positions
        pos_full = torch.cat([ctx_position_ids, position_ids], dim=1)
        h_full = torch.cat([target_hidden, hidden_states], dim=1)
        position_embeddings = self.rotary_emb(h_full, pos_full)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
        accept_lengths_out: Optional[List[int]] = None,
    ):
        self.eval()
        if self.mask_token_id is None:
            raise ValueError(
                "mask_token_id is None: set config.flashmtp_config['mask_token_id'] "
                "or config.dflashconfig['mask_token_id'] (training checkpoint)."
            )
        dev = input_ids.device
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens

        block_size = self.block_size
        output_ids = torch.full(
            (1, max_length + block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=dev,
        )
        position_ids = torch.arange(output_ids.shape[1], device=dev).unsqueeze(0)

        past_key_values_target = DynamicCache()
        # 与训练一致：每个投机块单独一次 draft 前向，不用 KV cache（训练为整段 N*bs 单次前向）
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
        pre_idx = (num_input_tokens - 1) * torch.ones(1, 1, device=dev, dtype=torch.long)
        pre_idx = pre_idx.clamp(min=0)
        target_hidden = extract_stacked_chs(output.hidden_states, pre_idx)

        start = input_ids.shape[1]
        while start < max_length:
            block_output_ids = output_ids[:, start : start + block_size].clone()
            block_position_ids = position_ids[:, start : start + block_size]
            noise_embedding = target.model.embed_tokens(block_output_ids)
            # 与 OnlineFlashMTPModel._create_position_ids 一致：块内为 [anchor, anchor+1, ...]
            block_position_ids_for_draft = position_ids[
                :, start : start + block_size
            ]
            draft_logits = target.lm_head(
                self(
                    target_hidden=target_hidden,
                    noise_embedding=noise_embedding,
                    position_ids=block_position_ids_for_draft,
                    past_key_values=None,
                    use_cache=False,
                    is_causal=False,
                )[:, -block_size + 1 :, :]
            )
            block_output_ids[:, 1:] = sample(draft_logits)

            output = target(
                block_output_ids,
                position_ids=block_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
            )

            posterior = sample(output.logits, temperature)
            acceptance_length = (
                (block_output_ids[:, 1:] == posterior[:, :-1])
                .cumprod(dim=1)
                .sum(dim=1)[0]
                .item()
            )
            # 与 DFlash/eval 一致：连续接受的 draft 位置数 + 块首 seed，记为一步的「接收长度」
            accept_len_report = int(acceptance_length) + 1
            if accept_lengths_out is not None:
                accept_lengths_out.append(accept_len_report)
            output_ids[:, start : start + acceptance_length + 1] = block_output_ids[
                :, : acceptance_length + 1
            ]
            output_ids[:, start + acceptance_length + 1] = posterior[
                :, acceptance_length
            ]
            start += acceptance_length + 1
            past_key_values_target.crop(start)
            # 下一块的 CHS 对应新 anchor 的前一位置：即本步 target 块内最后采纳位置（与训练 context=anchor-1 对齐）
            hs_len = output.hidden_states[0].shape[1]
            last_chunk_idx = min(int(acceptance_length), hs_len - 1)
            last_chunk_idx = max(last_chunk_idx, 0)
            target_hidden = extract_stacked_chs(
                output.hidden_states,
                torch.tensor([[last_chunk_idx]], device=dev, dtype=torch.long),
            )
            if stop_token_ids is not None and any(
                stop_token_id in output_ids[:, num_input_tokens:]
                for stop_token_id in stop_token_ids
            ):
                break
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
