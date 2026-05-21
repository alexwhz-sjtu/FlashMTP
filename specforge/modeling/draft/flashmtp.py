import time
from typing import Any, Callable, List, Optional

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

# MDLM-style block fill: unmask counts per draft round (1+2+4+8 == block_size-1 when bs=16).
MDLM_CONFIDENCE_ROUND_COUNTS: Tuple[int, ...] = (1, 2, 4, 8)


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
    """Return (B, 1, S, H) pivot features for inference (``token_index`` along seq dim)."""
    off = _infer_hs_embedding_offset(hidden_states, num_transformer_layers)
    pieces: list[torch.Tensor] = []
    for layer_id in target_layer_ids:
        layer_h = hidden_states[layer_id + off]
        pieces.append(layer_h[:, token_index, :].unsqueeze(1))
    return torch.stack(pieces, dim=2)


def prepare_target_hidden(
    hidden_states: tuple[torch.Tensor, ...] | list[torch.Tensor],
    anchor_positions: torch.Tensor,
    target_layer_ids: list[int],
    num_transformer_layers: int,
) -> torch.Tensor:
    """Gather pivot hidden states for selected transformer layers at anchor-1 -> (B, N, S, H)."""
    context_positions = (anchor_positions - 1).clamp(min=0)
    off = _infer_hs_embedding_offset(hidden_states, num_transformer_layers)
    pieces: list[torch.Tensor] = []
    for layer_id in target_layer_ids:
        layer_hidden = hidden_states[layer_id + off]
        layer_selected = torch.gather(
            layer_hidden,
            dim=1,
            index=context_positions.unsqueeze(-1).expand(
                -1, -1, layer_hidden.size(-1)
            ),
        )
        pieces.append(layer_selected)
    return torch.stack(pieces, dim=2)


class Qwen3FlashMTPAttention(nn.Module):
    """Non-causal self-attn: K/V = fused pivot ctx + draft; RoPE on full K (v1.1 linear_fuse path)."""

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

        cos, sin = position_embeddings

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
        self.chs_concat_mode = "feature"
        flashmtp_config["chs_concat_mode"] = "feature"
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

        flashmtp_config.setdefault("num_middle_layers_n", self.num_middle_layers_n)
        flashmtp_config["target_layer_ids"] = self.target_layer_ids
        self.train_lm_head = bool(flashmtp_config.get("train_lm_head", False))
        flashmtp_config["train_lm_head"] = self.train_lm_head
        self.local_position = bool(flashmtp_config.get("local_position", False))
        flashmtp_config["local_position"] = self.local_position
        if self.train_lm_head:
            self.draft_lm_head = nn.Linear(
                config.hidden_size, config.vocab_size, bias=False
            )
        else:
            self.draft_lm_head = None
        config.flashmtp_config = flashmtp_config

        self.layers = nn.ModuleList(
            [
                Qwen3FlashMTPDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.block_size = config.block_size
        self.mask_token_id = flashmtp_config.get("mask_token_id", None)
        self.sink_num = flashmtp_config.get("sink_num", None)
        self._last_decode_stats: dict = {}

        s_layers = len(self.target_layer_ids)
        h = config.hidden_size
        self.fc = nn.Linear(s_layers * h, h, bias=False)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        print_on_rank0(
            "FlashMTP: linear_fuse (v1.1), "
            f"num_middle_layers_n={self.num_middle_layers_n}, "
            f"target_layer_ids={self.target_layer_ids}, "
            f"train_lm_head={self.train_lm_head}, local_position={self.local_position}"
        )

        self.post_init()

    @property
    def chs_len_per_block(self) -> int:
        return 1

    def _fuse_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """(B, N, S, H) -> (B, N, H) linear fuse + norm (v1.1)."""
        bsz, n_blk, s_len, h = target_hidden.shape
        if s_len != len(self.target_layer_ids):
            raise ValueError(
                f"target_hidden S={s_len} != len(target_layer_ids)={len(self.target_layer_ids)}"
            )
        flat = target_hidden.reshape(bsz, n_blk, s_len * h)
        return self.hidden_norm(self.fc(flat))

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

    def get_last_decode_stats(self) -> dict:
        return dict(self._last_decode_stats) if self._last_decode_stats else {
            "accept_lengths": [],
            "total_time": 0.0,
            "target_total_time": 0.0,
            "draft_total_time": 0.0,
            "draft_forwards": 0,
            "accepted_tokens": 0,
        }

    def _forward_block_draft_logits(
        self,
        target: nn.Module,
        target_hidden: torch.Tensor,
        block_output_ids: torch.LongTensor,
        block_position_ids_for_draft: torch.LongTensor,
        rotary_position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        noise_embedding = target.model.embed_tokens(block_output_ids)
        hidden = self(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=block_position_ids_for_draft,
            rotary_position_ids=rotary_position_ids,
            past_key_values=None,
            use_cache=False,
            is_causal=False,
        )
        tail = hidden[:, -self.block_size + 1 :, :]
        if self.draft_lm_head is not None:
            return self.draft_lm_head(tail)
        return target.lm_head(tail)

    def _confidence_and_token_for_row(
        self, row: torch.Tensor, temperature: float
    ) -> tuple[float, torch.Tensor]:
        if temperature < 1e-5:
            tok = row.argmax(dim=-1)
            conf = row.max().item()
            return conf, tok
        prob = torch.softmax(row / temperature, dim=-1)
        tok = torch.multinomial(prob, num_samples=1).squeeze(-1)
        conf = prob.max().item()
        return conf, tok

    def _fill_block_draft_mdlm_confidence(
        self,
        target: nn.Module,
        target_hidden: torch.Tensor,
        block_output_ids: torch.LongTensor,
        block_position_ids_for_draft: torch.LongTensor,
        rotary_position_ids: Optional[torch.LongTensor],
        temperature: float,
        draft_forwards: list,
        draft_time: list,
    ) -> None:
        """Iterative unmask: rounds unmask 1,2,4,8 positions by highest confidence; tail fill if needed."""
        bs = self.block_size
        masked = {p for p in range(1, bs)}
        for k in MDLM_CONFIDENCE_ROUND_COUNTS:
            if not masked:
                break
            take = min(k, len(masked))
            if take <= 0:
                break
            t0 = time.perf_counter()
            logits = self._forward_block_draft_logits(
                target,
                target_hidden,
                block_output_ids,
                block_position_ids_for_draft,
                rotary_position_ids,
            )
            draft_time[0] += time.perf_counter() - t0
            draft_forwards[0] += 1

            scored: list[tuple[float, int, torch.Tensor]] = []
            for pos in masked:
                li = pos - 1
                conf, tok = self._confidence_and_token_for_row(
                    logits[0, li], temperature
                )
                scored.append((conf, pos, tok))
            scored.sort(key=lambda x: -x[0])
            for _, pos, tok in scored[:take]:
                block_output_ids[0, pos] = tok
                masked.discard(pos)

        if masked:
            t0 = time.perf_counter()
            logits = self._forward_block_draft_logits(
                target,
                target_hidden,
                block_output_ids,
                block_position_ids_for_draft,
                rotary_position_ids,
            )
            draft_time[0] += time.perf_counter() - t0
            draft_forwards[0] += 1
            for pos in list(masked):
                li = pos - 1
                row = logits[0, li : li + 1]
                new_tok = sample(row.unsqueeze(0), temperature)
                block_output_ids[0, pos] = new_tok.view(-1)[0]
            masked.clear()

    @torch.inference_mode()
    def _speculative_generate_impl(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
        accept_lengths_out: Optional[List[int]],
        draft_decode: str,
        spec_trace_fn: Optional[Callable[..., None]] = None,
    ) -> torch.Tensor:
        self.eval()
        if self.mask_token_id is None:
            raise ValueError(
                "mask_token_id is None: set config.flashmtp_config['mask_token_id'] "
                "or config.dflashconfig['mask_token_id'] (training checkpoint)."
            )
        if draft_decode not in ("oneshot", "mdlm"):
            raise ValueError("draft_decode must be 'oneshot' or 'mdlm'")

        dev = input_ids.device
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens
        block_size = self.block_size

        accept_buf: List[int] = [] if accept_lengths_out is None else accept_lengths_out
        target_time = 0.0
        draft_time = 0.0
        draft_forwards = 0
        gen_t0 = time.perf_counter()

        output_ids = torch.full(
            (1, max_length + block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=dev,
        )
        position_ids = torch.arange(output_ids.shape[1], device=dev).unsqueeze(0)
        past_key_values_target = DynamicCache()

        t0 = time.perf_counter()
        output = target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )
        target_time += time.perf_counter() - t0

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

        start = input_ids.shape[1]
        spec_step = 0
        chs = self.chs_len_per_block
        while start < max_length:
            block_output_ids = output_ids[:, start : start + block_size].clone()
            block_position_ids = position_ids[:, start : start + block_size]
            if self.local_position:
                draft_block_pos = torch.arange(
                    1, block_size + 1, device=dev, dtype=torch.long
                ).unsqueeze(0)
            else:
                draft_block_pos = block_position_ids
            if self.local_position:
                ctx_pos_part = torch.zeros(1, chs, dtype=torch.long, device=dev)
            else:
                ctx_pos_part = torch.full(
                    (1, chs),
                    start - 1,
                    dtype=torch.long,
                    device=dev,
                )
            full_rotary = torch.cat([ctx_pos_part, draft_block_pos], dim=-1)
            block_position_ids_for_draft = draft_block_pos

            dfw = [0]
            dtw = [0.0]
            draft_logits: Optional[torch.Tensor] = None
            if draft_decode == "mdlm":
                self._fill_block_draft_mdlm_confidence(
                    target,
                    target_hidden,
                    block_output_ids,
                    block_position_ids_for_draft,
                    full_rotary,
                    temperature,
                    dfw,
                    dtw,
                )
                draft_forwards += dfw[0]
                draft_time += dtw[0]
                if spec_trace_fn is not None:
                    t0 = time.perf_counter()
                    draft_logits = self._forward_block_draft_logits(
                        target,
                        target_hidden,
                        block_output_ids,
                        block_position_ids_for_draft,
                        full_rotary,
                    )
                    draft_time += time.perf_counter() - t0
                    draft_forwards += 1
            else:
                t0 = time.perf_counter()
                draft_logits = self._forward_block_draft_logits(
                    target,
                    target_hidden,
                    block_output_ids,
                    block_position_ids_for_draft,
                    full_rotary,
                )
                draft_time += time.perf_counter() - t0
                draft_forwards += 1
                block_output_ids[:, 1:] = sample(draft_logits, temperature)

            t0 = time.perf_counter()
            output = target(
                block_output_ids,
                position_ids=block_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
            )
            target_time += time.perf_counter() - t0

            posterior = sample(output.logits, temperature)
            acceptance_length = (
                (block_output_ids[:, 1:] == posterior[:, :-1])
                .cumprod(dim=1)
                .sum(dim=1)[0]
                .item()
            )
            accept_len_report = int(acceptance_length) + 1
            accept_buf.append(accept_len_report)
            if spec_trace_fn is not None and draft_logits is not None:
                spec_trace_fn(
                    spec_step,
                    int(start),
                    block_output_ids,
                    draft_logits,
                    output.logits,
                    posterior,
                    int(acceptance_length),
                    int(accept_len_report),
                )
            spec_step += 1
            output_ids[:, start : start + acceptance_length + 1] = block_output_ids[
                :, : acceptance_length + 1
            ]
            output_ids[:, start + acceptance_length + 1] = posterior[
                :, acceptance_length
            ]
            start += acceptance_length + 1
            past_key_values_target.crop(start)
            pivot_index = min(
                int(acceptance_length), output.hidden_states[0].shape[1] - 1
            )
            pivot_index = max(pivot_index, 0)
            target_hidden = gather_pivot_multilayer_inference(
                output.hidden_states,
                self.target_layer_ids,
                pivot_index,
                self.config.num_target_layers,
            )
            if stop_token_ids is not None and any(
                stop_token_id in output_ids[:, num_input_tokens:]
                for stop_token_id in stop_token_ids
            ):
                break

        total_elapsed = time.perf_counter() - gen_t0
        self._last_decode_stats = {
            "accept_lengths": list(accept_buf),
            "total_time": total_elapsed,
            "target_total_time": target_time,
            "draft_total_time": draft_time,
            "draft_forwards": draft_forwards,
            "accepted_tokens": int(sum(accept_buf)) if accept_buf else 0,
        }

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
    def spec_generate_mdlm(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
        accept_lengths_out: Optional[List[int]] = None,
        spec_trace_fn: Optional[Callable[..., None]] = None,
    ):
        """Spec decode with MDLM-style draft: 4 confidence rounds (1,2,4,8) + tail fill, then target verify."""
        return self._speculative_generate_impl(
            target=target,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
            accept_lengths_out=accept_lengths_out,
            draft_decode="mdlm",
            spec_trace_fn=spec_trace_fn,
        )

    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
        accept_lengths_out: Optional[List[int]] = None,
        spec_trace_fn: Optional[Callable[..., None]] = None,
    ):
        """Spec decode: single draft forward fills the whole draft tail, then target verify."""
        return self._speculative_generate_impl(
            target=target,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
            accept_lengths_out=accept_lengths_out,
            draft_decode="oneshot",
            spec_trace_fn=spec_trace_fn,
        )
