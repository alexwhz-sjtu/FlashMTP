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


def flashmtp_slot_group(slot):
    """Map slots to FlashMTP semantic groups: anchor, 1, 2, then chunks of 4."""
    if not torch.is_tensor(slot):
        slot = torch.as_tensor(slot)
    return torch.where(
        slot <= 1,
        slot,
        torch.where(slot < 4, torch.full_like(slot, 2), 3 + (slot - 4) // 4),
    )


def build_flashmtp_prediction_groups(block_size: int) -> list[tuple[int, int]]:
    """Return prediction groups over draft slots: [1], [2,3], [4..7], ..."""
    groups = []
    start = 1
    group_size = 1
    while start < block_size:
        end = min(block_size, start + group_size)
        groups.append((start, end))
        start = end
        group_size = 2 if group_size == 1 else 4
    return groups


def create_flashmtp_single_block_mask(
    batch_size: int,
    block_size: int,
    device: torch.device,
    attention_backend: str,
    dtype: torch.dtype = torch.float32,
) -> Optional[torch.Tensor]:
    """Build inference mask for one pivot plus one FlashMTP draft block.

    KV layout: [Pivot | anchor, draft slots...]
    Q layout:  [anchor, draft slots...]

    Pivot is visible to all draft slots. Draft slots follow block-causal
    semantic groups: anchor, [1], [2,3], [4..7], ...
    """
    q_len = block_size
    kv_len = block_size + 1

    if attention_backend == "flex_attention":
        if not FLEX_ATTENTION_AVAILABLE:
            raise ImportError("flex_attention is required for FlashMTP BlockMask.")

        def mask_mod(b, h, q_idx, kv_idx):
            is_context = kv_idx == 0
            kv_slot = kv_idx - 1
            q_group = flashmtp_slot_group(q_idx)
            kv_group = flashmtp_slot_group(kv_slot)
            same_or_previous_group = kv_group <= q_group
            return is_context | ((kv_idx > 0) & same_or_previous_group)

        return create_block_mask(
            mask_mod,
            B=batch_size,
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )

    q_slots = torch.arange(q_len, device=device).view(q_len, 1)
    kv_slots = torch.arange(kv_len, device=device).view(1, kv_len) - 1
    q_groups = flashmtp_slot_group(q_slots)
    kv_groups = flashmtp_slot_group(kv_slots.clamp(min=0))
    visible = (kv_slots < 0) | (kv_groups <= q_groups)
    mask = torch.zeros(
        (batch_size, 1, q_len, kv_len),
        device=device,
        dtype=dtype,
    )
    return mask.masked_fill(~visible.view(1, 1, q_len, kv_len), torch.finfo(dtype).min)


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
        
        # CHS is positioned at anchor-1 and draft tokens at anchor+k.
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


def extract_latest_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: Optional[list[int]],
    token_index: int = -1,
) -> torch.Tensor:
    """Extract feature-concat hidden states from specified layers."""
    offset = 1
    selected_states = []
    for layer_id in layer_ids:
        layer_hidden = hidden_states[layer_id + offset]
        selected_states.append(layer_hidden[:, token_index, :].unsqueeze(1))
    return torch.cat(selected_states, dim=-1)


class FlashMTPDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3FlashMTPDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        flashmtp_config = getattr(config, "flashmtp_config", {}) or {}
        self.chs_concat_mode = "feature"
        flashmtp_config["chs_concat_mode"] = "feature"
        config.flashmtp_config = flashmtp_config
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
        self.mask_token_id = flashmtp_config.get("mask_token_id", None)
        self._last_decode_stats = {}

        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        print_on_rank0(f"self.chs_concat_mode: {self.chs_concat_mode}")

        self.post_init()

    def get_last_decode_stats(self) -> dict:
        return dict(self._last_decode_stats)

    def fuse_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        return self.hidden_norm(self.fc(target_hidden))

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
            {"id": int(token_id), "token": tokenizer.decode([int(token_id)])}
            for token_id in token_ids.view(-1).tolist()
        ]

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states = noise_embedding
        target_hidden = self.fuse_target_hidden(target_hidden)
        # position_embeddings = self.rotary_emb(torch.cat([target_hidden, hidden_states], dim=1), position_ids)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
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
    def spec_generate_with_profile(
        self,
        target: nn.Module,
        tokenizer,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
        top_k: int = 5,
        print_fn=print,
    ):
        """Profiled speculative generation with per-step token confidences."""
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
        if self.mask_token_id is None:
            raise ValueError(
                "FlashMTPDraftModel.mask_token_id is None. Load a checkpoint with "
                "flashmtp_config['mask_token_id'] or set draft_model.mask_token_id "
                "before calling spec_generate_with_profile()."
            )
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
        anchor_token = output_ids[:, num_input_tokens : num_input_tokens + 1]
        print_fn(
            f"\n[Prefill anchor] pos={num_input_tokens} "
            f"token={self._format_token_list(anchor_token[0], tokenizer)}"
        )

        target_hidden = extract_latest_context_feature(
            output.hidden_states,
            self.target_layer_ids,
            token_index=-1,
        )

        start = input_ids.shape[1]
        prediction_groups = build_flashmtp_prediction_groups(block_size)
        draft_attention_mask = create_flashmtp_single_block_mask(
            batch_size=input_ids.shape[0],
            block_size=block_size,
            device=target.device,
            attention_backend=self.config._attn_implementation,
            dtype=next(self.parameters()).dtype,
        )
        while start < max_length:
            print_fn(f"\n[Spec step {self._last_decode_stats['steps']}] start={start}")
            block_output_ids = output_ids[:, start : start + block_size].clone()
            block_position_ids = position_ids[:, start : start + block_size]
            for group_start, group_end in prediction_groups:
                noise_embedding = target.model.embed_tokens(block_output_ids)
                if target.device.type == "cuda":
                    torch.cuda.synchronize(target.device)
                draft_start = time.perf_counter()
                draft_hidden = self(
                    target_hidden=target_hidden,
                    noise_embedding=noise_embedding,
                    position_ids=position_ids[:, start - 1 : start + block_size],
                    attention_mask=draft_attention_mask,
                    past_key_values=None,
                    use_cache=False,
                    is_causal=False,
                )
                draft_logits = target.lm_head(
                    draft_hidden[:, group_start:group_end, :]
                )
                if target.device.type == "cuda":
                    torch.cuda.synchronize(target.device)
                self._last_decode_stats["draft_total_time"] += (
                    time.perf_counter() - draft_start
                )

                for offset, slot in enumerate(range(group_start, group_end)):
                    topk_entries = self._format_token_topk(
                        draft_logits[0, offset], tokenizer, top_k
                    )
                    print_fn(
                        f"  [Draft slot {slot} | abs_pos={start + slot}] "
                        f"top{top_k}={topk_entries}"
                    )

                block_output_ids[:, group_start:group_end] = sample(
                    draft_logits, temperature
                )

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
            accepted_tokens = block_output_ids[:, : acceptance_length + 1]
            print_fn(
                f"  [Accept] length={acceptance_length + 1} "
                f"tokens={self._format_token_list(accepted_tokens[0], tokenizer)}"
            )

            for target_pos in range(min(acceptance_length + 1, output.logits.shape[1])):
                role = "accepted" if target_pos < acceptance_length else "correction"
                target_token = posterior[:, target_pos : target_pos + 1]
                target_top3 = self._format_token_topk(
                    output.logits[0, target_pos], tokenizer, 3
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
            pivot_index = min(
                acceptance_length, output.hidden_states[0].shape[1] - 1
            )
            target_hidden = extract_latest_context_feature(
                output.hidden_states,
                self.target_layer_ids,
                token_index=pivot_index,
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
        if self.mask_token_id is None:
            raise ValueError(
                "FlashMTPDraftModel.mask_token_id is None. Load a checkpoint with "
                "flashmtp_config['mask_token_id'] or set draft_model.mask_token_id "
                "before calling spec_generate()."
            )
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
        )

        # Decode stage
        acceptance_lengths = []
        start = input_ids.shape[1]
        prediction_groups = build_flashmtp_prediction_groups(block_size)
        draft_attention_mask = create_flashmtp_single_block_mask(
            batch_size=input_ids.shape[0],
            block_size=block_size,
            device=target.device,
            attention_backend=self.config._attn_implementation,
            dtype=next(self.parameters()).dtype,
        )
        while start < max_length:
            block_output_ids = output_ids[:, start : start + block_size].clone()
            block_position_ids = position_ids[:, start : start + block_size]
            for group_start, group_end in prediction_groups:
                noise_embedding = target.model.embed_tokens(block_output_ids)
                if target.device.type == "cuda":
                    torch.cuda.synchronize(target.device)
                draft_start = time.perf_counter()
                draft_hidden = self(
                    target_hidden=target_hidden,
                    noise_embedding=noise_embedding,
                    position_ids=position_ids[:, start - 1 : start + block_size],
                    attention_mask=draft_attention_mask,
                    past_key_values=None,
                    use_cache=False,
                    is_causal=False,
                )
                draft_logits = target.lm_head(
                    draft_hidden[:, group_start:group_end, :]
                )
                if target.device.type == "cuda":
                    torch.cuda.synchronize(target.device)
                self._last_decode_stats["draft_total_time"] += (
                    time.perf_counter() - draft_start
                )
                block_output_ids[:, group_start:group_end] = sample(
                    draft_logits, temperature
                )

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
