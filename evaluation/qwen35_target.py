"""Qwen3.5 text target with exact hybrid-cache rollback for FlashMTP evaluation.

Transformers 5.3's multi-token cached GatedDeltaNet path starts from zero.
Use the existing state for cached blocks and retain enough recurrence inputs to
reconstruct an accepted prefix. Only the small delta recurrence is replayed,
not projections, attention, MLPs, or the full prefix. Replay time is timed by
spec_generate. Prefill and single-token baseline use the upstream implementation.
"""
from types import MethodType

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DynamicCache,
    apply_mask_to_padding_states,
)


class RollbackQwen35Cache(Qwen3_5DynamicCache):
    def __init__(self, config):
        super().__init__(config)
        self.pending = {}
        self.block_start = None

    def crop(self, max_length):
        current_length = self.get_seq_length()
        if not 0 <= max_length <= current_length:
            raise ValueError(f"Invalid cache crop {max_length} from {current_length}")
        if max_length < current_length:
            if self.block_start is None or max_length < self.block_start:
                raise ValueError("Cannot roll back before the latest verification block")
            accepted = max_length - self.block_start
            for idx, state in self.pending.items():
                module, conv_history, initial, q, k, v, g, beta = state
                self.conv_states[idx] = conv_history[:, :, accepted:accepted + module.conv_kernel_size].contiguous()
                if accepted == 0:
                    self.recurrent_states[idx] = initial
                else:
                    _, restored = module.recurrent_gated_delta_rule(
                        q[:, :accepted], k[:, :accepted], v[:, :accepted],
                        g=g[:, :accepted], beta=beta[:, :accepted],
                        initial_state=initial, output_final_state=True,
                        use_qk_l2norm_in_kernel=True,
                    )
                    self.recurrent_states[idx] = restored
            for idx in self.transformer_layers:
                self.key_cache[idx] = self.key_cache[idx][:, :, :max_length, :]
                self.value_cache[idx] = self.value_cache[idx][:, :, :max_length, :]
        self.pending.clear()
        self.block_start = None


def _cached_block_forward(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None):
    if not (cache_params is not None and cache_params.has_previous_state and hidden_states.shape[1] > 1):
        return self._flashmtp_original_forward(hidden_states, cache_params, cache_position, attention_mask)
    if not isinstance(cache_params, RollbackQwen35Cache):
        raise TypeError("Qwen3.5 cached block evaluation requires RollbackQwen35Cache")
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch, length, _ = hidden_states.shape
    idx = self.layer_idx
    # The first linear layer runs before the first full-attention layer extends KV.
    if idx == 0:
        cache_params.pending.clear()
        cache_params.block_start = cache_params.get_seq_length()
    raw_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
    history = torch.cat((cache_params.conv_states[idx], raw_qkv), dim=-1)
    if self.causal_conv1d_fn is not None and getattr(self, "_flashmtp_fused_conv", True):
        # Match upstream's fused FP32 convolution + SiLU accumulation. Doing
        # BF16 conv1d followed by SiLU introduces an extra rounding boundary.
        mixed = self.causal_conv1d_fn(
            x=history, weight=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias, activation=self.activation,
        )[:, :, -length:].transpose(1, 2)
    else:
        conv = F.conv1d(history, self.conv1d.weight, self.conv1d.bias, groups=self.conv_dim)
        mixed = F.silu(conv[:, :, -length:]).transpose(1, 2)
    cache_params.conv_states[idx] = history[:, :, -self.conv_kernel_size:].contiguous()
    query, key, value = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    query = query.reshape(batch, length, -1, self.head_k_dim)
    key = key.reshape(batch, length, -1, self.head_k_dim)
    value = value.reshape(batch, length, -1, self.head_v_dim)
    ratio = self.num_v_heads // self.num_k_heads
    if ratio > 1:
        query = query.repeat_interleave(ratio, dim=2)
        key = key.repeat_interleave(ratio, dim=2)
    beta = self.in_proj_b(hidden_states).sigmoid()
    g = -self.A_log.float().exp() * F.softplus(self.in_proj_a(hidden_states).float() + self.dt_bias)
    initial = cache_params.recurrent_states[idx]
    core, final = self.recurrent_gated_delta_rule(
        query, key, value, g=g, beta=beta, initial_state=initial,
        output_final_state=True, use_qk_l2norm_in_kernel=True,
    )
    cache_params.recurrent_states[idx] = final
    cache_params.pending[idx] = (self, history, initial, query, key, value, g, beta)
    z = self.in_proj_z(hidden_states).reshape(-1, self.head_v_dim)
    core = self.norm(core.reshape(-1, self.head_v_dim), z).reshape(batch, length, -1)
    return self.out_proj(core)


def _text_attention_forward(self, *args, **kwargs):
    # HF 5.3 passes 3-axis mRoPE positions to FA2's 2D packed-sequence
    # detector. RoPE is already applied from position_embeddings; supply the
    # single text axis so FA2 cannot mistake axes for packed sequences.
    positions = kwargs.get("position_ids")
    if positions is not None and positions.ndim == 3:
        kwargs["position_ids"] = positions[0]
    return self._flashmtp_original_forward(*args, **kwargs)


def install_qwen35_rollback(target):
    for layer in target.model.layers:
        if layer.layer_type == "linear_attention":
            module = layer.linear_attn
            module._flashmtp_original_forward = module.forward
            module.forward = MethodType(_cached_block_forward, module)
        else:
            module = layer.self_attn
            module._flashmtp_original_forward = module.forward
            module.forward = MethodType(_text_attention_forward, module)
    target.make_inference_cache = lambda: RollbackQwen35Cache(target.config)
    return target
