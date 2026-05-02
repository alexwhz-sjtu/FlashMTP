import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Union

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


def decode_single_token_for_log(tokenizer, token_id: int) -> str:
    if tokenizer is None:
        return ""
    try:
        return tokenizer.decode([token_id], skip_special_tokens=False)
    except Exception:
        return ""


def append_draft_block_topk_jsonl(
    draft_logits: torch.Tensor,
    tokenizer,
    out_file: Any,
    *,
    temperature: float,
    topk: int,
    decode_step: int,
    acceptance_length: int,
    last_accepted_token: str = "",
    draft_sampled_ids: Optional[torch.Tensor] = None,
    anchor_token_id: Optional[int] = None,
) -> None:
    """Write one JSONL record with draft top-k probabilities for a verify step."""
    if out_file is None or tokenizer is None:
        return
    logits = draft_logits.float().squeeze(0)
    temp = float(temperature)
    probs = torch.softmax(logits / temp, dim=-1) if temp > 0 else torch.softmax(logits, dim=-1)
    k = min(max(1, int(topk)), probs.shape[-1])
    vals, indices = probs.topk(k, dim=-1)

    positions = []
    for pos in range(vals.shape[0]):
        p_row = probs[pos]
        entropy_nats = float(-(p_row * p_row.clamp_min(1e-30).log()).sum().item())
        entries = []
        for j in range(k):
            tid = int(indices[pos, j].item())
            prob = float(vals[pos, j].item())
            entries.append(
                {
                    "token_id": tid,
                    "prob": round(prob, 6),
                    "token": decode_single_token_for_log(tokenizer, tid),
                }
            )
        payload: dict[str, Any] = {
            "position": pos + 1,
            "topk": entries,
            "entropy_nats": round(entropy_nats, 6),
            "entropy_bits": round(entropy_nats / math.log(2.0), 6),
        }
        if draft_sampled_ids is not None:
            ds = draft_sampled_ids
            sel_tid = int(ds[0, pos].item()) if ds.dim() == 2 else int(ds[pos].item())
            payload["selected_token_id"] = sel_tid
            payload["selected_prob"] = round(float(p_row[sel_tid].item()), 6)
            payload["selected_token"] = decode_single_token_for_log(tokenizer, sel_tid)
        positions.append(payload)

    row: dict[str, Any] = {
        "decode_step": int(decode_step),
        "acceptance_length": int(acceptance_length),
        "tokens_committed_from_block": int(acceptance_length) + 1,
        "last_accepted_token": last_accepted_token,
        "positions": positions,
    }
    if anchor_token_id is not None:
        row["anchor_token_id"] = int(anchor_token_id)
        row["anchor_token"] = decode_single_token_for_log(tokenizer, int(anchor_token_id))
    out_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    out_file.flush()


class SpecDebugRecorder:
    """Small JSONL recorder for FlashMTP speculative decoding traces."""

    def __init__(self, debug_dir: Optional[str], tokenizer=None):
        self.tokenizer = tokenizer
        self.debug_path = self._resolve_debug_dir(debug_dir)
        self._run_file: Optional[Path] = None
        self._steps: list[dict[str, Any]] = []

    @staticmethod
    def _resolve_debug_dir(debug_dir: Optional[str]) -> Optional[Path]:
        debug_dir = debug_dir or os.environ.get("FLASHMTP_DEBUG_DIR")
        if not debug_dir:
            return None
        path = Path(debug_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def token_ids_from_tensor(self, tensor: torch.Tensor) -> list[int]:
        if self.debug_path is None:
            return []
        return [int(x) for x in tensor.detach().flatten().cpu().tolist()]

    def start_run(self, **metadata: Any) -> None:
        if self.debug_path is None:
            return
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        self._run_file = self.debug_path / f"flashmtp_spec_generate_{timestamp}.jsonl"
        self._append({"event": "start", **metadata})

    def record_prefill(self, **payload: Any) -> None:
        self._append({"event": "prefill", **payload})

    def add_step(self, **payload: Any) -> None:
        self._steps.append(payload)
        self._append({"event": "step", **payload})

    def dump(self, **summary: Any) -> Optional[str]:
        if self.debug_path is None:
            return None
        self._append({"event": "summary", "steps": self._steps, **summary})
        return str(self._run_file) if self._run_file is not None else None

    def _append(self, payload: dict[str, Any]) -> None:
        if self._run_file is None:
            return
        with self._run_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


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
        self.chs_fusion = CHSQueryFusion(
            config,
            nsrc,
            chs_fusion_layer_idx=chs_fusion_layer_idx,
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.block_size = config.block_size
        self.mask_token_id = flashmtp_config.get("mask_token_id", None)
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
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states = noise_embedding
        if target_hidden is None:
            raise ValueError("target_hidden is required.")
        # (B, N, S, H) before fusion
        target_hidden = self.chs_fusion(target_hidden)
        # RoPE must cover K/V = [context | draft]. Previously only `hidden_states` (draft)
        # was used, so cos/sin length did not match concat(k_ctx, k_noise) in attention.
        bs = self.block_size
        dlen = hidden_states.size(1)
        if dlen % bs != 0:
            raise ValueError(
                f"Draft sequence length {dlen} must be divisible by block_size {bs}."
            )
        n_blocks = dlen // bs
        if target_hidden.size(1) != n_blocks:
            raise ValueError(
                f"Fused target_hidden len {target_hidden.size(1)} != num_blocks {n_blocks}."
            )
        # Block i draft starts at position_ids[:, i * bs] (= anchor i); CHS uses anchor-1.
        first_of_block = position_ids[:, ::bs]
        ctx_position_ids = (first_of_block - 1).clamp(min=0)
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

    def get_last_decode_stats(self) -> dict:
        return getattr(self, "_last_decode_stats", {})

    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
        tokenizer=None,
        debug_dir: Optional[str] = None,
        draft_topk_file=None,
        draft_topk: int = 5,
        accept_lengths_out: Optional[List[int]] = None,
    ):
        self.eval()
        self._last_decode_stats = {
            "accept_lengths": [],
            "target_total_time": 0.0,
            "draft_total_time": 0.0,
            "steps": 0,
            "debug_file": None,
        }
        if self.mask_token_id is None:
            raise ValueError(
                "mask_token_id is None: set config.flashmtp_config['mask_token_id'] "
                "or config.dflashconfig['mask_token_id'] (training checkpoint)."
            )
        debug_recorder = SpecDebugRecorder(debug_dir=debug_dir, tokenizer=tokenizer)
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

        debug_recorder.start_run(
            temperature=float(temperature),
            block_size=int(block_size),
            num_input_tokens=int(num_input_tokens),
            max_new_tokens=int(max_new_tokens),
            prompt_token_ids=debug_recorder.token_ids_from_tensor(input_ids),
        )

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
        first_sampled_ids = sample(output.logits, temperature)
        output_ids[:, num_input_tokens : num_input_tokens + 1] = first_sampled_ids
        debug_recorder.record_prefill(
            first_sampled_token_ids=debug_recorder.token_ids_from_tensor(
                first_sampled_ids
            )
        )
        pre_idx = (num_input_tokens - 1) * torch.ones(1, 1, device=dev, dtype=torch.long)
        pre_idx = pre_idx.clamp(min=0)
        target_hidden = extract_stacked_chs(output.hidden_states, pre_idx)

        acceptance_lengths: list[int] = []
        target_total_time = 0.0
        draft_total_time = 0.0
        steps = 0
        start = input_ids.shape[1]
        while start < max_length:
            block_output_ids = output_ids[:, start : start + block_size].clone()
            block_position_ids = position_ids[:, start : start + block_size]
            noise_embedding = target.model.embed_tokens(block_output_ids)
            context_token_ids = debug_recorder.token_ids_from_tensor(
                output_ids[:, :start]
            )
            block_seed_token_ids = debug_recorder.token_ids_from_tensor(
                block_output_ids[:, :1]
            )
            # 与 OnlineFlashMTPModel._create_position_ids 一致：块内为 [anchor, anchor+1, ...]
            block_position_ids_for_draft = position_ids[
                :, start : start + block_size
            ]
            draft_start_time = time.time()
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
            draft_total_time += time.time() - draft_start_time
            draft_sampled_ids = sample(draft_logits)
            block_output_ids[:, 1:] = draft_sampled_ids

            target_start_time = time.time()
            output = target(
                block_output_ids,
                position_ids=block_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
            )
            target_total_time += time.time() - target_start_time

            posterior = sample(output.logits, temperature)
            acceptance_length = (
                (block_output_ids[:, 1:] == posterior[:, :-1])
                .cumprod(dim=1)
                .sum(dim=1)[0]
                .item()
            )
            # 与 DFlash/eval 一致：连续接受的 draft 位置数 + 块首 seed，记为一步的「接收长度」
            accept_len_report = int(acceptance_length) + 1
            acceptance_lengths.append(accept_len_report)
            if accept_lengths_out is not None:
                accept_lengths_out.append(accept_len_report)
            if draft_topk_file is not None:
                last_accepted_token = ""
                if tokenizer is not None:
                    last_token_id = int(block_output_ids[0, acceptance_length].item())
                    last_accepted_token = decode_single_token_for_log(
                        tokenizer, last_token_id
                    )
                append_draft_block_topk_jsonl(
                    draft_logits,
                    tokenizer,
                    draft_topk_file,
                    temperature=temperature,
                    topk=draft_topk,
                    decode_step=steps + 1,
                    acceptance_length=int(acceptance_length),
                    last_accepted_token=last_accepted_token,
                    draft_sampled_ids=draft_sampled_ids,
                    anchor_token_id=int(block_output_ids[0, 0].item()),
                )
            posterior_token_ids = debug_recorder.token_ids_from_tensor(posterior)
            accepted_token_ids = debug_recorder.token_ids_from_tensor(
                block_output_ids[:, : acceptance_length + 1]
            )
            replacement_token_id = int(posterior[0, acceptance_length].item())
            debug_recorder.add_step(
                step=steps + 1,
                start=int(start),
                block_size=int(block_size),
                context_token_ids=context_token_ids,
                block_seed_token_ids=block_seed_token_ids,
                block_position_ids=debug_recorder.token_ids_from_tensor(
                    block_position_ids
                ),
                draft_sampled_token_ids=debug_recorder.token_ids_from_tensor(
                    draft_sampled_ids
                ),
                posterior_token_ids=posterior_token_ids,
                acceptance_length=int(acceptance_length),
                accepted_token_ids=accepted_token_ids,
                replacement_token_id=replacement_token_id,
            )
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
            steps += 1
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

        debug_file = debug_recorder.dump(
            final_output_token_ids=debug_recorder.token_ids_from_tensor(output_ids),
            accept_lengths=acceptance_lengths,
            target_total_time=target_total_time,
            draft_total_time=draft_total_time,
            steps=steps,
        )
        self._last_decode_stats = {
            "accept_lengths": acceptance_lengths,
            "target_total_time": target_total_time,
            "draft_total_time": draft_total_time,
            "steps": steps,
            "debug_file": debug_file,
        }

        return output_ids
