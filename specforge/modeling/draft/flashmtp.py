import time
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
    k_len = k.size(-2)
    q_cos = cos[..., -q_len:, :]
    q_sin = sin[..., -q_len:, :]
    k_cos = cos[..., -k_len:, :]
    k_sin = sin[..., -k_len:, :]
    q_embed = (q * q_cos) + (rotate_half(q) * q_sin)
    k_embed = (k * k_cos) + (rotate_half(k) * k_sin)
    return q_embed, k_embed


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
    """Select seven target layers evenly, always including first and last."""
    del num_draft_layers
    num_fusion_layers = 7
    if num_target_layers < num_fusion_layers:
        raise ValueError(
            f"FlashMTP requires at least {num_fusion_layers} target layers, "
            f"got {num_target_layers}."
        )

    last_layer_id = num_target_layers - 1
    middle_layer_ids = [
        int(round((i * last_layer_id) / (num_fusion_layers - 1)))
        for i in range(1, num_fusion_layers - 1)
    ]
    return [0, *middle_layer_ids, last_layer_id]


class Qwen3FlashMTPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int, chs_concat_mode: str):
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
        k_ctx = self.k_proj(target_hidden) # (B, N*L, H)  

        k_noise = self.k_proj(hidden_states)  # (B, N*S, H) 
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)

        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)

        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = position_embeddings
        raw_cos, raw_sin = cos, sin

        # Query uses draft positions; keys use their own context/draft positions.
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        ctx_cos, draft_cos = cos[..., :ctx_len, :], cos[..., ctx_len:, :]
        ctx_sin, draft_sin = sin[..., :ctx_len, :], sin[..., ctx_len:, :]
        k_ctx_part = k[:, :, :ctx_len, :]
        k_noise_part = k[:, :, ctx_len:, :]
        q = (q * draft_cos) + (rotate_half(q) * draft_sin)
        k_ctx_part = (k_ctx_part * ctx_cos) + (rotate_half(k_ctx_part) * ctx_sin)
        k_noise_part = (k_noise_part * draft_cos) + (
            rotate_half(k_noise_part) * draft_sin
        )
        k = torch.cat([k_ctx_part, k_noise_part], dim=2)

        if past_key_values is not None:
            cache_kwargs = {
                "sin": raw_sin,
                "cos": raw_cos,
                "cache_position": cache_position,
            }
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
    def __init__(self, config: Qwen3Config, layer_idx: int, chs_concat_mode: str):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3FlashMTPAttention(config=config, layer_idx=layer_idx, chs_concat_mode=chs_concat_mode)
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


def extract_context_features_at_positions(
    hidden_states: list[torch.Tensor],
    layer_ids: Optional[list[int]],
    positions: torch.LongTensor,
) -> torch.Tensor:
    """Extract v5.1 feature-concat target hidden at explicit token positions."""
    offset = 1
    if positions.dim() == 1:
        positions = positions.unsqueeze(0)

    selected_states = []
    for layer_id in layer_ids:
        layer_hidden = hidden_states[layer_id + offset]
        safe_positions = positions.to(layer_hidden.device).clamp(
            min=0, max=layer_hidden.shape[1] - 1
        )
        layer_selected = torch.gather(
            layer_hidden,
            dim=1,
            index=safe_positions.unsqueeze(-1).expand(
                -1, -1, layer_hidden.shape[-1]
            ),
        )
        selected_states.append(layer_selected)
    return torch.cat(selected_states, dim=-1)


def build_v51_context_feature(
    fused_hidden_history: torch.Tensor,
    pivot_position: int,
    context_size: int = 6,
    pivot_window_size: int = 4,
) -> torch.Tensor:
    """Build [0, 1, pivot-3, pivot-2, pivot-1, pivot] context features."""
    if context_size != pivot_window_size + 2:
        raise ValueError("context_size must equal pivot_window_size + 2 for v5.1.")
    seq_len = fused_hidden_history.shape[1]
    if seq_len <= 0:
        raise ValueError("fused_hidden_history must contain at least one token.")

    positions = build_v51_context_position_ids(
        seq_len=seq_len,
        pivot_position=pivot_position,
        device=fused_hidden_history.device,
        context_size=context_size,
        pivot_window_size=pivot_window_size,
    ).squeeze(0)
    return fused_hidden_history[:, positions, :]


def build_v51_context_position_ids(
    seq_len: int,
    pivot_position: int,
    device: torch.device,
    context_size: int = 6,
    pivot_window_size: int = 4,
) -> torch.LongTensor:
    """Build [0, 1, pivot-3, pivot-2, pivot-1, pivot] position ids."""
    if context_size != pivot_window_size + 2:
        raise ValueError("context_size must equal pivot_window_size + 2 for v5.1.")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive for v5.1 context positions.")

    sink_positions = torch.tensor([0, 1], device=device, dtype=torch.long)
    window_positions = torch.arange(
        pivot_position - pivot_window_size + 1,
        pivot_position + 1,
        device=device,
        dtype=torch.long,
    )
    positions = torch.cat([sink_positions, window_positions]).clamp(
        min=0, max=seq_len - 1
    )
    return positions.unsqueeze(0)


class FlashMTPDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3FlashMTPDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        flashmtp_config = getattr(config, "flashmtp_config", {}) or {}
        self.chs_concat_mode = flashmtp_config.get("chs_concat_mode", "feature")
        if self.chs_concat_mode != "feature":
            print_on_rank0(
                "FlashMTP v5.1 only supports feature CHS; overriding "
                f"chs_concat_mode={self.chs_concat_mode!r} to 'feature'."
            )
            self.chs_concat_mode = "feature"
            flashmtp_config["chs_concat_mode"] = "feature"
            config.flashmtp_config = flashmtp_config
        self.layers = nn.ModuleList(
            [
                Qwen3FlashMTPDecoderLayer(config, layer_idx, self.chs_concat_mode)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.target_layer_ids = build_target_layer_ids(
            config.num_target_layers, config.num_hidden_layers
        )
        flashmtp_config["target_layer_ids"] = self.target_layer_ids
        config.flashmtp_config = flashmtp_config
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.block_size = config.block_size
        self.mask_token_id = flashmtp_config.get("mask_token_id", None)
        self.context_size = flashmtp_config.get("context_size", 6)
        self.pivot_window_size = flashmtp_config.get("pivot_window_size", 4)
        self._last_decode_stats = {}

        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        print_on_rank0(
            f"self.chs_concat_mode: {self.chs_concat_mode}, "
            f"context_size: {self.context_size}, "
            f"pivot_window_size: {self.pivot_window_size}, "
            f"target_layer_ids: {self.target_layer_ids}"
        )

        self.post_init()

    def get_last_decode_stats(self) -> dict:
        return dict(self._last_decode_stats)

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        context_position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:

        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        if context_position_ids is None:
            context_position_ids = torch.zeros(
                target_hidden.shape[:2],
                dtype=position_ids.dtype,
                device=position_ids.device,
            )
        full_position_ids = torch.cat([context_position_ids, position_ids], dim=1)
        full_hidden_states = torch.cat([target_hidden, hidden_states], dim=1)
        position_embeddings = self.rotary_emb(full_hidden_states, full_position_ids)
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
    ):
        self.eval()
        self._last_decode_stats = {
            "accept_lengths": [],
            "target_total_time": 0.0,
            "draft_total_time": 0.0,
            "steps": 0,
        }
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens

        block_size = self.block_size
        output_ids = torch.full(
            (1, max_length + block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=target.device,
        )
        position_ids = torch.arange(
            output_ids.shape[1], device=target.device
        ).unsqueeze(0)

        past_key_values_target = DynamicCache()

        # Prefill stage
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
        self._last_decode_stats["target_total_time"] += time.perf_counter() - target_start

        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
            output.logits, temperature
        )
        prefill_positions = torch.arange(
            num_input_tokens, device=input_ids.device, dtype=torch.long
        ).unsqueeze(0)
        fused_hidden_history = extract_context_features_at_positions(
            output.hidden_states,
            self.target_layer_ids,
            prefill_positions,
        )
        target_hidden = build_v51_context_feature(
            fused_hidden_history,
            pivot_position=num_input_tokens - 1,
            context_size=self.context_size,
            pivot_window_size=self.pivot_window_size,
        )
        context_position_ids = build_v51_context_position_ids(
            seq_len=fused_hidden_history.shape[1],
            pivot_position=num_input_tokens - 1,
            device=input_ids.device,
            context_size=self.context_size,
            pivot_window_size=self.pivot_window_size,
        )

        # Decode stage
        start = input_ids.shape[1]
        while start < max_length:
            block_output_ids = output_ids[:, start : start + block_size].clone()
            block_position_ids = position_ids[:, start : start + block_size]
            noise_embedding = target.model.embed_tokens(block_output_ids)
            if target.device.type == "cuda":
                torch.cuda.synchronize(target.device)
            draft_start = time.perf_counter()
            draft_logits = target.lm_head(
                self(
                    target_hidden=target_hidden,
                    context_position_ids=context_position_ids,
                    noise_embedding=noise_embedding,
                    position_ids=position_ids[:, start : start + block_size],
                    past_key_values=None,
                    use_cache=False,
                    is_causal=False,
                )[:, -block_size + 1 :, :]
            )
            if target.device.type == "cuda":
                torch.cuda.synchronize(target.device)
            self._last_decode_stats["draft_total_time"] += time.perf_counter() - draft_start
            block_output_ids[:, 1:] = sample(draft_logits)

            if target.device.type == "cuda":
                torch.cuda.synchronize(target.device)
            target_start = time.perf_counter()
            output = target(
                block_output_ids,
                position_ids=block_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
            )
            if target.device.type == "cuda":
                torch.cuda.synchronize(target.device)
            self._last_decode_stats["target_total_time"] += time.perf_counter() - target_start

            posterior = sample(output.logits, temperature)
            acceptance_length = (
                (block_output_ids[:, 1:] == posterior[:, :-1])
                .cumprod(dim=1)
                .sum(dim=1)[0]
                .item()
            )
            output_ids[:, start : start + acceptance_length + 1] = block_output_ids[
                :, : acceptance_length + 1
            ]
            output_ids[:, start + acceptance_length + 1] = posterior[
                :, acceptance_length
            ]
            start += acceptance_length + 1
            past_key_values_target.crop(start)

            accepted_count = acceptance_length + 1
            block_hidden_positions = torch.arange(
                output.hidden_states[0].shape[1],
                device=input_ids.device,
                dtype=torch.long,
            ).unsqueeze(0)
            fused_block_hidden = extract_context_features_at_positions(
                output.hidden_states,
                self.target_layer_ids,
                block_hidden_positions,
            )
            fused_hidden_history = torch.cat(
                [fused_hidden_history, fused_block_hidden[:, :accepted_count, :]],
                dim=1,
            )
            target_hidden = build_v51_context_feature(
                fused_hidden_history,
                pivot_position=start - 1,
                context_size=self.context_size,
                pivot_window_size=self.pivot_window_size,
            )
            context_position_ids = build_v51_context_position_ids(
                seq_len=fused_hidden_history.shape[1],
                pivot_position=start - 1,
                device=input_ids.device,
                context_size=self.context_size,
                pivot_window_size=self.pivot_window_size,
            )
            self._last_decode_stats["accept_lengths"].append(acceptance_length + 1)
            self._last_decode_stats["steps"] += 1
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
