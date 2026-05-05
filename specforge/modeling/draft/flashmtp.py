import time
from typing import Callable, Optional

import torch
from torch import nn
from transformers import DynamicCache
from transformers.cache_utils import Cache
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

try:
    from torch.nn.attention.flex_attention import create_block_mask

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    create_block_mask = None


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


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
    del num_draft_layers
    return list(range(1, num_target_layers, 2))


def build_stage_ranges(block_size: int, num_draft_layers: int) -> list[tuple[int, int]]:
    if block_size == 16 and num_draft_layers == 5:
        return [(0, 2), (2, 4), (4, 8), (8, 12), (12, 16)]
    if num_draft_layers <= 0:
        raise ValueError("num_draft_layers must be positive.")
    base = block_size // num_draft_layers
    remainder = block_size % num_draft_layers
    ranges = []
    start = 0
    for layer_idx in range(num_draft_layers):
        width = base + (1 if layer_idx < remainder else 0)
        end = start + width
        ranges.append((start, end))
        start = end
    return ranges


def _stage_id_from_position(pos, stage_ranges):
    stage_id = torch.zeros_like(pos)
    for idx, (start, _end) in enumerate(stage_ranges):
        stage_id = torch.where(pos >= start, torch.full_like(pos, idx), stage_id)
    return stage_id


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
        
        # Only apply RoPE to draft tokens; CHS slots stay un-rotated.
        if self.chs_concat_mode in ("seq", "feature"):
            k_ctx_part = k[:, :, :ctx_len, :]  
            k_noise_part = k[:, :, ctx_len:, :]   
            q, k_noise_part = apply_rotary_pos_emb(q, k_noise_part, cos, sin)
            k = torch.cat([k_ctx_part, k_noise_part], dim=2)
        else:
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


def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: Optional[list[int]],
) -> torch.Tensor:
    """Extract hidden states from specified layer IDs."""
    offset = 1
    selected_states = []
    for layer_id in layer_ids:
        selected_states.append(hidden_states[layer_id + offset])
    target_hidden = torch.cat(selected_states, dim=-1)
    return target_hidden


def extract_latest_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: Optional[list[int]],
    token_index: int = -1,
    chs_concat_mode: str = "feature",
) -> torch.Tensor:
    """Extract latest token hidden states from specified layers.

    Returns:
        - seq mode: (B, L, H)
        - feature mode: (B, 1, H*L)
    """
    offset = 1
    selected_states = []
    for layer_id in layer_ids:
        layer_hidden = hidden_states[layer_id + offset]
        selected_states.append(layer_hidden[:, token_index, :].unsqueeze(1))

    if chs_concat_mode == "seq":
        return torch.cat(selected_states, dim=1)
    return torch.cat(selected_states, dim=-1)


class FlashMTPDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3FlashMTPDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        flashmtp_config = getattr(config, "flashmtp_config", {}) or {}
        if not hasattr(config, "flashmtp_config") or config.flashmtp_config is None:
            config.flashmtp_config = flashmtp_config
        self.chs_concat_mode = flashmtp_config.get("chs_concat_mode", "feature")
        self.layers = nn.ModuleList(
            [
                Qwen3FlashMTPDecoderLayer(config, layer_idx, self.chs_concat_mode)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        # target_layer_ids: list of layer indices to extract from target model
        self.target_layer_ids = flashmtp_config.get(
            "target_layer_ids",
            build_target_layer_ids(config.num_target_layers, config.num_hidden_layers),
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.block_size = config.block_size
        self.stage_ranges = [
            tuple(stage_range)
            for stage_range in flashmtp_config.get(
                "stage_ranges",
                build_stage_ranges(config.block_size, config.num_hidden_layers),
            )
        ]
        if len(self.stage_ranges) != config.num_hidden_layers:
            raise ValueError(
                "flashmtp_config.stage_ranges must have one range per draft layer."
            )
        if self.stage_ranges[0][0] != 0 or self.stage_ranges[-1][1] != self.block_size:
            raise ValueError("stage_ranges must cover the full draft block.")
        self.mask_token_id = flashmtp_config.get("mask_token_id", None)
        self._last_decode_stats = {}
        stage_head_config = flashmtp_config.get("stage_head", True)
        if isinstance(stage_head_config, str):
            stage_head_config = stage_head_config.lower() in ("true", "1", "yes", "y", "on")
        self.use_stage_heads = bool(stage_head_config)
        self.config.flashmtp_config["stage_head"] = self.use_stage_heads
        object.__setattr__(self, "_shared_stage_lm_head", None)
        if self.use_stage_heads:
            self.stage_lm_heads = nn.ModuleList(
                [
                    nn.Linear(config.hidden_size, config.vocab_size, bias=False)
                    for _ in range(max(config.num_hidden_layers - 1, 0))
                ]
            )

        # For seq concat mode: use Identity (no computation, no parameters)
        # For feature mode: use Linear projection and RMSNorm
        if self.chs_concat_mode == "feature":
            self.fc = nn.Linear(
                len(self.target_layer_ids) * config.hidden_size,
                config.hidden_size,
                bias=False,
            )
            self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.fc = nn.Identity()
            # self.hidden_norm = nn.Identity()
            # maybe need norm
            self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        print_on_rank0(f"self.chs_concat_mode: {self.chs_concat_mode}")
        print_on_rank0(f"FlashMTP stage_ranges: {self.stage_ranges}")
        print_on_rank0(
            "FlashMTP stage heads: "
            + (
                "stage0 shared target lm_head; stage1+ draft-owned heads"
                if self.use_stage_heads
                else "shared frozen target lm_head"
            )
        )

        self.post_init()

    def initialize_stage_lm_heads(self, target_lm_head: nn.Module) -> None:
        """Attach target head for stage0 and initialize trainable stage1+ heads."""
        self.set_shared_stage_lm_head(target_lm_head)
        if not self.use_stage_heads:
            return

        target_weight = target_lm_head.weight.detach()
        for head in self.stage_lm_heads:
            head.weight.data.copy_(
                target_weight.to(device=head.weight.device, dtype=head.weight.dtype)
            )
        self.config.flashmtp_config["stage_ranges"] = [
            list(stage_range) for stage_range in self.stage_ranges
        ]

    def set_shared_stage_lm_head(self, target_lm_head: nn.Module) -> None:
        """Use target lm_head for all stages without registering it in this module."""
        target_lm_head.eval()
        target_lm_head.requires_grad_(False)
        object.__setattr__(self, "_shared_stage_lm_head", target_lm_head)

    def get_last_decode_stats(self) -> dict:
        return dict(self._last_decode_stats)

    def build_inference_attention_mask(
        self, batch_size: int, ctx_len: int, device: torch.device
    ):
        if (
            not FLEX_ATTENTION_AVAILABLE
            or self.config._attn_implementation != "flex_attention"
        ):
            return None

        def mask_mod(b, h, q_idx, kv_idx):
            q_pos = q_idx % self.block_size
            q_stage = _stage_id_from_position(q_pos, self.stage_ranges)

            is_context = kv_idx < ctx_len
            is_draft = kv_idx >= ctx_len
            kv_pos = (kv_idx - ctx_len) % self.block_size
            kv_stage = _stage_id_from_position(kv_pos, self.stage_ranges)
            return is_context | (is_draft & (kv_stage <= q_stage))

        return create_block_mask(
            mask_mod,
            B=batch_size,
            H=None,
            Q_LEN=self.block_size,
            KV_LEN=ctx_len + self.block_size,
            device=device,
        )

    def _stage_logits_from_hidden(
        self, hidden_states: torch.Tensor, stage_idx: int
    ) -> torch.Tensor:
        bsz, q_len, hidden_size = hidden_states.shape
        if q_len % self.block_size != 0:
            raise ValueError(
                f"Draft sequence length {q_len} must be divisible by block_size={self.block_size}."
            )
        n_blocks = q_len // self.block_size
        start, end = self.stage_ranges[stage_idx]
        stage_hidden = hidden_states.view(bsz, n_blocks, self.block_size, hidden_size)[
            :, :, start:end, :
        ]
        stage_hidden = stage_hidden.reshape(bsz, n_blocks * (end - start), hidden_size)
        shared_head = self._shared_stage_lm_head
        if stage_idx == 0 or not self.use_stage_heads:
            if shared_head is None:
                raise RuntimeError(
                    "FlashMTPDraftModel needs a shared target lm_head for stage0. "
                    "Call set_shared_stage_lm_head() before forward/spec_generate."
                )
            return shared_head(stage_hidden)

        if stage_idx - 1 >= len(self.stage_lm_heads):
            raise RuntimeError(
                f"Missing draft-owned lm_head for stage {stage_idx}. "
                f"Expected {len(self.stage_ranges) - 1} trainable heads, got "
                f"{len(self.stage_lm_heads)}."
            )
        return self.stage_lm_heads[stage_idx - 1](stage_hidden)

    def _scatter_stage_logits(
        self,
        stage_logits: list[torch.Tensor],
        bsz: int,
        n_blocks: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        full_logits = torch.empty(
            bsz,
            n_blocks,
            self.block_size,
            self.config.vocab_size,
            device=device,
            dtype=dtype,
        )
        for stage_idx, logits in enumerate(stage_logits):
            start, end = self.stage_ranges[stage_idx]
            full_logits[:, :, start:end, :] = logits.view(
                bsz, n_blocks, end - start, self.config.vocab_size
            )
        return full_logits.view(
            bsz, n_blocks * self.block_size, self.config.vocab_size
        )

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        # position_embeddings = self.rotary_emb(torch.cat([target_hidden, hidden_states], dim=1), position_ids)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        stage_logits = []
        stage_hidden_states = []
        for stage_idx, layer in enumerate(self.layers):
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
            norm_hidden_states = self.norm(hidden_states)
            stage_hidden_states.append(norm_hidden_states)
            stage_logits.append(
                self._stage_logits_from_hidden(norm_hidden_states, stage_idx)
            )

        bsz, q_len = hidden_states.shape[:2]
        n_blocks = q_len // self.block_size
        logits = self._scatter_stage_logits(
            stage_logits,
            bsz=bsz,
            n_blocks=n_blocks,
            device=hidden_states.device,
            dtype=stage_logits[0].dtype,
        )
        return {
            "logits": logits,
            "stage_logits": stage_logits,
            "hidden_states": stage_hidden_states,
            "last_hidden_state": stage_hidden_states[-1],
        }

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
        if not self.use_stage_heads:
            self.set_shared_stage_lm_head(target.lm_head)
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
        target_hidden = extract_latest_context_feature(
            output.hidden_states,
            self.target_layer_ids,
            token_index=-1,
            chs_concat_mode=self.chs_concat_mode,
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
            draft_attn_mask = self.build_inference_attention_mask(
                batch_size=block_output_ids.shape[0],
                ctx_len=target_hidden.shape[1],
                device=block_output_ids.device,
            )
            draft_logits = self(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, start : start + block_size],
                past_key_values=None,
                use_cache=False,
                is_causal=False,
                attention_mask=draft_attn_mask,
            )["logits"][:, 1:block_size, :]
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
            target_hidden = extract_latest_context_feature(
                output.hidden_states,
                self.target_layer_ids,
                token_index=pivot_index,
                chs_concat_mode=self.chs_concat_mode,
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
