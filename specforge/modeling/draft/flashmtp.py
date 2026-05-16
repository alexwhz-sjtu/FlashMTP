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
) -> torch.Tensor:
    """Return (B, 1, S, H) pivot features for inference."""
    off = _infer_hs_embedding_offset(hidden_states, num_transformer_layers)
    pieces: list[torch.Tensor] = []
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
        pos = torch.arange(s_len, device=x.device, dtype=torch.long).unsqueeze(0).expand(
            bsz * n_blk, -1
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
        self, config: Qwen3Config, layer_idx: int, chs_concat_mode: str, pivot_fuse_mode: str
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
    def __init__(self, config: Qwen3Config, layer_idx: int, chs_concat_mode: str, pivot_fuse_mode: str):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3FlashMTPAttention(
            config=config, layer_idx=layer_idx, chs_concat_mode=chs_concat_mode, pivot_fuse_mode=pivot_fuse_mode
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

        s_layers = len(self.target_layer_ids)
        h = config.hidden_size
        if self.pivot_fuse_mode == "linear_fuse":
            self.fc = nn.Linear(s_layers * h, h, bias=False)
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
            f"target_layer_ids={self.target_layer_ids}"
        )

        self.post_init()

    @property
    def chs_len_per_block(self) -> int:
        return len(self.target_layer_ids) if self.pivot_fuse_mode == "prefix_condition" else 1

    def get_last_decode_stats(self) -> dict:
        return dict(self._last_decode_stats)

    def _fuse_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """(B, N, S, H) -> (B, N, H) for linear/attention, or (B, N*S, H) for prefix."""
        bsz, n_blk, s_len, h = target_hidden.shape
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
        depth_emb = self.layer_depth_embedding(depth_ids).view(1, 1, s_len, h)
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
        rotary_pos = rotary_position_ids if rotary_position_ids is not None else draft_pos
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
        target_hidden = gather_pivot_multilayer_inference(
            output.hidden_states,
            self.target_layer_ids,
            -1,
            self.config.num_target_layers,
        )

        # Decode stage
        acceptance_lengths = []
        start = input_ids.shape[1]
        while start < max_length:
            block_output_ids = output_ids[:, start : start + block_size].clone()
            block_position_ids = position_ids[:, start : start + block_size]
            noise_embedding = target.model.embed_tokens(block_output_ids)
            if target.device.type == "cuda":
                torch.cuda.synchronize(target.device)
            draft_start = time.perf_counter()
            chs = self.chs_len_per_block
            ctx_pos_part = torch.full(
                (1, chs),
                start - 1,
                dtype=torch.long,
                device=target.device,
            )
            full_rotary = torch.cat([ctx_pos_part, block_position_ids], dim=-1)
            draft_logits = target.lm_head(
                self(
                    target_hidden=target_hidden,
                    noise_embedding=noise_embedding,
                    position_ids=block_position_ids,
                    rotary_position_ids=full_rotary,
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
            pivot_index = min(
                acceptance_length, output.hidden_states[0].shape[1] - 1
            )
            target_hidden = gather_pivot_multilayer_inference(
                output.hidden_states,
                self.target_layer_ids,
                pivot_index,
                self.config.num_target_layers,
            )
            acceptance_lengths.append(acceptance_length + 1)
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
