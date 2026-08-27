from __future__ import annotations

import logging
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from .config import FlashMTPConfig, parse_flashmtp_config

logger = logging.getLogger(__name__)

try:
    from sgl_kernel.flash_attn import flash_attn_varlen_func
except Exception:  # pragma: no cover - exercised only on non-CUDA installations
    flash_attn_varlen_func = None


def _apply_rms_norm(
    norm: RMSNorm,
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
):
    """SGLang's fused RMSNorm is 2-D; preserve leading FlashMTP dimensions."""
    shape = hidden_states.shape
    flat = hidden_states.reshape(-1, shape[-1])
    if residual is None:
        return norm(flat).view(shape)
    normed, updated_residual = norm(
        flat, residual.reshape(-1, residual.shape[-1])
    )
    return normed.view(shape), updated_residual.view(shape)


class FlashMTPAttention(nn.Module):
    def __init__(self, config, layer_id: int) -> None:
        super().__init__()
        del layer_id
        hidden_size = int(config.hidden_size)
        tp_size = int(get_tensor_model_parallel_world_size())
        total_heads = int(config.num_attention_heads)
        total_kv_heads = int(getattr(config, "num_key_value_heads", total_heads))
        head_dim = int(getattr(config, "head_dim", hidden_size // total_heads))
        if total_heads % tp_size != 0:
            raise ValueError(
                f"FlashMTP num_attention_heads={total_heads} is not divisible by tp={tp_size}."
            )
        if total_kv_heads >= tp_size and total_kv_heads % tp_size != 0:
            raise ValueError(
                f"FlashMTP num_key_value_heads={total_kv_heads} is not divisible by tp={tp_size}."
            )
        if total_kv_heads < tp_size and tp_size % total_kv_heads != 0:
            raise ValueError(
                f"FlashMTP tp={tp_size} is not divisible by num_key_value_heads={total_kv_heads}."
            )

        self.num_heads = total_heads // tp_size
        self.num_kv_heads = max(1, total_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * head_dim
        self.kv_size = self.num_kv_heads * head_dim
        self.scaling = head_dim**-0.5

        bias = bool(getattr(config, "attention_bias", False))
        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_dim,
            total_num_heads=total_heads,
            total_num_kv_heads=total_kv_heads,
            bias=bias,
            prefix="qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            total_heads * head_dim,
            hidden_size,
            bias=bias,
            prefix="o_proj",
        )
        self.q_norm = RMSNorm(head_dim, eps=eps)
        self.k_norm = RMSNorm(head_dim, eps=eps)
        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=int(getattr(config, "max_position_embeddings", 32768)),
            base=float(getattr(config, "rope_theta", 1_000_000)),
            rope_scaling=getattr(config, "rope_scaling", None),
            is_neox_style=bool(getattr(config, "rope_is_neox_style", True)),
        )
        self._cu_seqlens: dict[tuple[str, int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_cu_seqlens(
        self, device: torch.device, batch_size: int, query_len: int, kv_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (str(device), batch_size, query_len, kv_len)
        cached = self._cu_seqlens.get(key)
        if cached is None:
            cached = (
                torch.arange(
                    0,
                    (batch_size + 1) * query_len,
                    query_len,
                    device=device,
                    dtype=torch.int32,
                ),
                torch.arange(
                    0,
                    (batch_size + 1) * kv_len,
                    kv_len,
                    device=device,
                    dtype=torch.int32,
                ),
            )
            self._cu_seqlens[key] = cached
        return cached

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_size: int,
        query_len: int,
        kv_len: int,
    ) -> torch.Tensor:
        if flash_attn_varlen_func is not None and q.is_cuda:
            cu_q, cu_k = self._get_cu_seqlens(
                q.device, batch_size, query_len, kv_len
            )
            version = 4 if torch.cuda.get_device_capability(q.device)[0] >= 10 else 3
            return flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=query_len,
                max_seqlen_k=kv_len,
                softmax_scale=self.scaling,
                causal=False,
                ver=version,
            )

        q = q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.num_heads != self.num_kv_heads:
            groups = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False, scale=self.scaling
        )
        return out.transpose(1, 2).reshape(-1, self.num_heads, self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context_states: torch.Tensor,
        local_positions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, block_size, _ = hidden_states.shape
        context_len = int(context_states.shape[1])
        combined = torch.cat([context_states, hidden_states], dim=1)
        qkv, _ = self.qkv_proj(combined.reshape(-1, combined.shape[-1]))
        q_all, k_all, v_all = qkv.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        q = q_all.view(batch_size, context_len + block_size, -1)[:, context_len:]
        q = q.reshape(-1, self.q_size)
        k_ctx = k_all.view(batch_size, context_len + block_size, -1)[:, :context_len]
        k_noise = k_all.view(batch_size, context_len + block_size, -1)[:, context_len:]
        k_noise = k_noise.reshape(-1, self.kv_size)

        # Match the HF prefix_condition order: q norm -> local RoPE for q/noise-k ->
        # concatenate unrotated CHS keys -> k norm.
        q = self.q_norm(q.reshape(-1, self.head_dim)).view(-1, self.q_size)
        q, k_noise = self.rotary_emb(local_positions.reshape(-1), q, k_noise)
        k = torch.cat(
            [k_ctx, k_noise.view(batch_size, block_size, self.kv_size)], dim=1
        ).reshape(-1, self.kv_size)
        k = self.k_norm(k.reshape(-1, self.head_dim)).view(-1, self.kv_size)
        v = v_all.view(batch_size, context_len + block_size, self.kv_size).reshape(
            -1, self.kv_size
        )

        out = self._attention(
            q.view(-1, self.num_heads, self.head_dim),
            k.view(-1, self.num_kv_heads, self.head_dim),
            v.view(-1, self.num_kv_heads, self.head_dim),
            batch_size,
            block_size,
            context_len + block_size,
        )
        out, _ = self.o_proj(out.reshape(-1, self.q_size))
        return out.view(batch_size, block_size, -1)


class FlashMTPMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        intermediate_size = int(config.intermediate_size)
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size, intermediate_size], bias=False
        )
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x, _ = self.gate_up_proj(x.reshape(-1, shape[-1]))
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x.view(shape)


class FlashMTPDecoderLayer(nn.Module):
    def __init__(self, config, layer_id: int) -> None:
        super().__init__()
        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.input_layernorm = RMSNorm(int(config.hidden_size), eps=eps)
        self.self_attn = FlashMTPAttention(config, layer_id)
        self.post_attention_layernorm = RMSNorm(int(config.hidden_size), eps=eps)
        self.mlp = FlashMTPMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context_states: torch.Tensor,
        local_positions: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            normed = _apply_rms_norm(self.input_layernorm, hidden_states)
        else:
            normed, residual = _apply_rms_norm(
                self.input_layernorm, hidden_states, residual
            )
        attn = self.self_attn(normed, context_states, local_positions)
        normed, residual = _apply_rms_norm(
            self.post_attention_layernorm, attn, residual
        )
        return self.mlp(normed), residual


class FlashMTPMarkovHead(nn.Module):
    """Tensor-parallel serial vocabulary head used by v2 FlashMTP checkpoints."""

    def __init__(self, parsed: FlashMTPConfig) -> None:
        super().__init__()
        self.head_type = parsed.markov_head_type
        self.output_mode = parsed.markov_output_mode
        self.rank = parsed.markov_rank
        self.hidden_size = parsed.hidden_size
        self.prev_token_embedding = VocabParallelEmbedding(
            parsed.vocab_size,
            self.rank,
            org_num_embeddings=parsed.vocab_size,
            prefix="markov_head.prev_token_embedding",
        )
        self.output_proj = ParallelLMHead(
            parsed.vocab_size,
            self.rank,
            bias=False,
            org_num_embeddings=parsed.vocab_size,
            prefix="markov_head.output_proj",
        )
        self.gate_proj = None
        self.state_proj = None
        self.state_out_proj = None
        self.hidden_proj = None
        self.hidden_fuse_gate_proj = None
        self.state_hidden_mlp = None
        if self.head_type == "gated":
            self.gate_proj = ReplicatedLinear(
                self.hidden_size + self.rank, self.rank, bias=True,
                prefix="markov_head.gate_proj",
            )
        elif self.head_type == "rnn":
            self.state_proj = ReplicatedLinear(
                2 * self.rank, 2 * self.rank, bias=True,
                prefix="markov_head.state_proj",
            )
            self.hidden_proj = ReplicatedLinear(
                self.hidden_size, self.rank, bias=False,
                prefix="markov_head.hidden_proj",
            )
            self.hidden_fuse_gate_proj = ReplicatedLinear(
                2 * self.rank, self.rank, bias=True,
                prefix="markov_head.hidden_fuse_gate_proj",
            )
            self.state_out_proj = ReplicatedLinear(
                self.rank, self.rank, bias=False,
                prefix="markov_head.state_out_proj",
            )
        elif self.head_type == "rnn_easy":
            self.state_proj = ReplicatedLinear(
                2 * self.rank, 2 * self.rank, bias=True,
                prefix="markov_head.state_proj",
            )
            self.hidden_proj = ReplicatedLinear(
                self.hidden_size, self.rank, bias=False,
                prefix="markov_head.hidden_proj",
            )
            self.state_hidden_mlp = ReplicatedLinear(
                2 * self.rank, self.rank, bias=True,
                prefix="markov_head.state_hidden_mlp",
            )

    @staticmethod
    def _linear(layer: ReplicatedLinear, value: torch.Tensor) -> torch.Tensor:
        return layer(value)[0]

    def initial_state(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        if self.head_type in {"rnn", "rnn_easy"}:
            return hidden_states.new_zeros(hidden_states.shape[0], self.rank)
        return None

    def step(
        self,
        *,
        prev_token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        state: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        prev = self.prev_token_embedding(prev_token_ids.long())
        hidden_latent = (
            self._linear(self.hidden_proj, hidden_states)
            if self.hidden_proj is not None and self.output_mode == "direct"
            else None
        )
        if self.head_type == "vanilla":
            return prev, None
        if self.head_type == "gated":
            assert self.gate_proj is not None
            gate = torch.sigmoid(self._linear(self.gate_proj, torch.cat([hidden_states, prev], -1)))
            return gate * prev, None
        assert self.state_proj is not None
        if state is None:
            state = torch.zeros_like(prev)
        gate_raw, candidate_raw = self._linear(
            self.state_proj, torch.cat([state, prev], -1)
        ).chunk(2, dim=-1)
        gate = torch.sigmoid(gate_raw)
        new_state = gate * state + (1.0 - gate) * torch.tanh(candidate_raw)
        if self.head_type == "rnn_easy":
            if self.output_mode == "direct":
                assert self.state_hidden_mlp is not None and hidden_latent is not None
                latent = self._linear(
                    self.state_hidden_mlp, torch.cat([new_state, hidden_latent], -1)
                )
                return latent, new_state
            return new_state, new_state
        assert self.state_out_proj is not None
        serial = torch.tanh(self._linear(self.state_out_proj, new_state))
        if hidden_latent is None:
            return serial, new_state
        assert self.hidden_fuse_gate_proj is not None
        fused = torch.cat([serial, hidden_latent], -1)
        fuse_gate = torch.sigmoid(self._linear(self.hidden_fuse_gate_proj, fused))
        return fuse_gate * serial + (1.0 - fuse_gate) * hidden_latent, new_state


class FlashMTPDraftModel(nn.Module):
    """SGLang/TP FlashMTP draft model with no attention KV cache."""

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        del quant_config, prefix
        parsed = parse_flashmtp_config(config)
        self.config = config
        self.block_size = parsed.block_size
        self.target_layer_ids = parsed.target_layer_ids
        self.mask_token_id = parsed.mask_token_id
        self.hidden_size = parsed.hidden_size
        self.context_len = parsed.num_context_tokens
        self.include_embedding_chs = parsed.include_embedding_chs
        self.markov_head_type = parsed.markov_head_type
        self.markov_output_mode = parsed.markov_output_mode
        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.layers = nn.ModuleList(
            [FlashMTPDecoderLayer(config, i) for i in range(parsed.num_hidden_layers)]
        )
        self.norm = RMSNorm(parsed.hidden_size, eps=eps)
        self.hidden_norm = RMSNorm(parsed.hidden_size, eps=eps)
        self.layer_depth_embedding = nn.Embedding(
            parsed.num_target_layers, parsed.hidden_size
        )
        self.markov_head = (
            None
            if parsed.markov_head_type == "none"
            else FlashMTPMarkovHead(parsed)
        )
        self.register_buffer(
            "target_layer_ids_tensor",
            torch.tensor(parsed.target_layer_ids, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "local_positions",
            torch.arange(1, parsed.block_size + 1, dtype=torch.long).unsqueeze(0),
            persistent=False,
        )

    def forward(
        self, noise_embedding: torch.Tensor, target_hidden: torch.Tensor
    ) -> torch.Tensor:
        if noise_embedding.ndim != 3 or noise_embedding.shape[1] != self.block_size:
            raise ValueError(
                f"Expected noise_embedding [B, {self.block_size}, H], got {tuple(noise_embedding.shape)}."
            )
        if target_hidden.shape[1:] != (self.context_len, self.hidden_size):
            raise ValueError(
                "FlashMTP pivot shape mismatch: expected "
                f"[B, {self.context_len}, {self.hidden_size}], got {tuple(target_hidden.shape)}."
            )
        depth = self.layer_depth_embedding(self.target_layer_ids_tensor).unsqueeze(0)
        if self.include_embedding_chs:
            raw_embedding = target_hidden[:, :1]
            layer_context = target_hidden[:, 1:] + depth
            target_hidden = torch.cat([raw_embedding, layer_context], dim=1)
        else:
            target_hidden = target_hidden + depth
        context = _apply_rms_norm(self.hidden_norm, target_hidden)
        local_positions = self.local_positions.expand(noise_embedding.shape[0], -1)
        hidden_states = noise_embedding
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, context, local_positions, residual
            )
        if residual is None:
            return _apply_rms_norm(self.norm, hidden_states)
        hidden_states, _ = _apply_rms_norm(self.norm, hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        stacked = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params = dict(self.named_parameters())
        loaded = set()
        for name, weight in weights:
            for param_name, weight_name, shard_id in stacked:
                if f".{weight_name}." not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped not in params:
                    break
                param = params[mapped]
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, weight, shard_id)
                loaded.add(mapped)
                break
            else:
                if name not in params:
                    continue
                param = params[name]
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, weight)
                loaded.add(name)
        missing = set(params) - loaded
        if missing:
            raise RuntimeError(
                "FlashMTP checkpoint did not initialize parameters: "
                + ", ".join(sorted(missing)[:16])
            )


EntryClass = FlashMTPDraftModel
