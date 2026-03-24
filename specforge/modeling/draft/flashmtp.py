"""
FlashMTP: 无KV Cache的快速多token预测草稿模型

核心设计:
1. 小模型不使用KV Cache，每次前向都是独立的
2. target_hidden作为前缀（prefix）输入，不添加位置编码
3. 只需要最后一个干净的token的hidden states
4. 块内双向注意力
"""

from typing import Optional, Callable
from typing_extensions import Unpack, Tuple
import torch
import time
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    Qwen3Config,
    Qwen3PreTrainedModel,
    Qwen3MLP,
    GradientCheckpointingLayer,
    FlashAttentionKwargs,
    rotate_half,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """应用旋转位置编码"""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    """采样函数"""
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: Optional[list[int]] = None,
) -> torch.Tensor:
    """
    从目标模型的hidden states中提取特征

    Args:
        hidden_states: 目标模型各层的hidden states列表
        layer_ids: 要提取的层ID列表，None表示所有层

    Returns:
        target_hidden: [batch, seq_len, sum(hidden_sizes)]
    """
    offset = 1  # 第0层是embedding

    if layer_ids is None:
        # 默认提取所有层
        layer_ids = list(range(len(hidden_states) - offset))

    selected_states = []
    for layer_id in layer_ids:
        selected_states.append(hidden_states[layer_id + offset])

    target_hidden = torch.cat(selected_states, dim=-1)
    return target_hidden


class FlashMTPAttention(nn.Module):
    """
    FlashMTP 注意力机制

    特点:
    1. Q只来自noise_embedding（要生成的token）
    2. K/V来自target_hidden（前缀，无位置编码）+ noise_embedding（有位置编码）
    3. 块内双向注意力
    """

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False  # 非因果注意力（块内双向）

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,      # [bsz, q_len, hidden_size] - noise_embedding
        target_hidden: torch.Tensor,      # [bsz, ctx_len, hidden_size] - 前缀，无位置编码
        position_embeddings: tuple[torch.Tensor, torch.Tensor],  # RoPE
        attention_mask: Optional[torch.Tensor] = None,
        concat_mode: Optional[str] = "feature",
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: noise_embedding [bsz, q_len, hidden_size]
            target_hidden: 前缀特征 [bsz, ctx_len, hidden_size]
            position_embeddings: (cos, sin) for RoPE
            attention_mask: 可选的注意力掩码

        Returns:
            attn_output: [bsz, q_len, hidden_size]
            attn_weights: None或注意力权重
        """
        bsz, q_len, _ = hidden_states.shape
        ctx_len = target_hidden.shape[1]

        # Q投影: 只来自hidden_states（要生成的token）
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)  # [bsz, num_heads, q_len, head_dim]

        # K/V投影: 来自target_hidden（前缀）+ hidden_states（当前token）
        # 注意：target_hidden不添加位置编码
        k_ctx = self.k_proj(target_hidden)  # [bsz, ctx_len, num_kv_heads * head_dim]
        k_noise = self.k_proj(hidden_states)  # [bsz, q_len, num_kv_heads * head_dim]
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)

        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)

        k = self.k_norm(k).transpose(1, 2)  # [bsz, num_kv_heads, ctx_len + q_len, head_dim]
        v = v.transpose(1, 2)  # [bsz, num_kv_heads, ctx_len + q_len, head_dim]

        # 应用RoPE到q和k部分
        cos, sin = position_embeddings

        # print(f"FlashMTPAttention Layer {self.layer_idx}: q shape {q.shape}, k shape {k.shape}, ctx_len {ctx_len}, q_len {q_len}")
        # print(f"ctx_len: {ctx_len}, q_len: {q_len}, position_embeddings cos shape: {cos.shape}, sin shape: {sin.shape}")
        if concat_mode == "feature":
            # ctx_len == 1：对整个k应用RoPE（ctx部分也有位置编码）
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        else:
            # ctx_len > 1：只对q和k的noise部分应用RoPE
            # k[:, :, ctx_len:, :] 是noise部分
            q, k_noise_rope = apply_rotary_pos_emb(q, k[:, :, ctx_len:, :], cos, sin)
            # 重新组合k：ctx部分（无RoPE）+ noise部分（有RoPE）
            k = torch.cat([k[:, :, :ctx_len, :], k_noise_rope], dim=2)

        # Attention计算
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
            **kwargs,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class FlashMTPDecoderLayer(GradientCheckpointingLayer):
    """FlashMTP 解码器层"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = FlashMTPAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        concat_mode: Optional[str] = "feature",
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.FloatTensor:
        """
        Args:
            target_hidden: 前缀特征 [bsz, ctx_len, hidden_size]
            hidden_states: 输入 [bsz, q_len, hidden_size]
            attention_mask: 注意力掩码
            position_ids: 位置ID
            position_embeddings: RoPE嵌入

        Returns:
            hidden_states: [bsz, q_len, hidden_size]
        """
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            concat_mode=concat_mode,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class FlashMTPDraftModel(Qwen3PreTrainedModel):
    """
    FlashMTP 草稿模型

    特点:
    1. 无KV Cache - 每次前向都是独立的
    2. target_hidden作为前缀输入，不添加位置编码
    3. 支持两种拼接方式：
       - 序列维度拼接（默认）：[ctx_len + block_size, hidden_size]
       - 特征维度拼接：保留两种实现供选择
    """

    config_class = Qwen3Config
    _no_split_modules = ["FlashMTPDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config

        # 解码器层
        self.layers = nn.ModuleList([
            FlashMTPDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        ])

        # 配置
        dflash_config = getattr(config, "dflash_config", {}) or {}
        self.target_layer_ids = dflash_config.get("target_layer_ids", None)
        self.block_size = config.block_size
        self.mask_token_id = dflash_config.get("mask_token_id", None)

        # 归一化和位置编码
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)

        # 拼接方式配置 - 必须在fc初始化之前
        self.concat_mode = dflash_config.get("concat_mode", "seq")  # "seq" 或 "feature"

        # 根据concat_mode决定是否创建投影层
        if self.concat_mode == "feature":
            # feature模式：需要投影层将拼接后的特征映射到hidden_size
            num_target_layers = getattr(config, "num_target_layers", 36)
            target_hidden_dim = num_target_layers * config.hidden_size
            self.fc = nn.Linear(target_hidden_dim, config.hidden_size, bias=False)
            self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            # seq模式：不需要投影层，target_hidden直接作为K/V输入
            # 使用Identity避免在forward中判断
            self.fc = nn.Identity()
            self.hidden_norm = nn.Identity()

        self.post_init()

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        concat_mode: Optional[str] = "feature",
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            position_ids: [bsz, seq_len] 位置ID
            attention_mask: 可选的注意力掩码
            noise_embedding: [bsz, block_size, hidden_size] 块token的embedding
            target_hidden: [bsz, ctx_len, hidden_size] 或 [bsz, ctx_len, num_target_layers * hidden_size]
                          取决于concat_mode

        Returns:
            hidden_states: [bsz, block_size, hidden_size]
        """
        # 根据concat_mode处理target_hidden
        # seq模式: fc和hidden_norm都是Identity，直接返回输入
        # feature模式: 通过fc投影并归一化
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        # target_hidden: [bsz, ctx_len, hidden_size]

        # 获取位置编码
        # ctx_len = target_hidden.shape[1]
        ctx_len = target_hidden.shape[1] if target_hidden is not None else 0
        if concat_mode == "feature":
            # ctx 也加 RoPE，拼接到 noise_embedding 前面
            # 需要 target_hidden 和 noise_embedding shape: [bsz, 1, hidden_size], [bsz, block_size, hidden_size]
            full_embedding = torch.cat([target_hidden, noise_embedding], dim=1)  # [bsz, 1+block_size, hidden_size]
            # position_ids 也拼接 ctx 的位置id（假设为 position_ids[:, :1]）
            position_embeddings = self.rotary_emb(full_embedding, position_ids)
        else:
            # 只对 noise_embedding 加 RoPE
            position_embeddings = self.rotary_emb(noise_embedding, position_ids)

        # 通过解码器层
        hidden_states = noise_embedding
        for layer in self.layers:
            hidden_states = layer(
                target_hidden=target_hidden,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                concat_mode=concat_mode,
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
        """
        投机解码生成

        Args:
            target: 目标模型（大模型）
            input_ids: [1, N] prompt tokens
            max_new_tokens: 最大生成token数
            stop_token_ids: 停止token ID列表
            temperature: 采样温度

        Returns:
            output_ids: 生成的token序列
        """
        self.eval()
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens
        block_size = self.block_size
        device = target.device

        # 初始化输出序列
        output_ids = torch.full(
            (1, max_length + block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        output_ids[:, :num_input_tokens] = input_ids

        # 位置编码
        position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)

        # Target模型的KV Cache
        past_key_values_target = None

        # ========== Prefill 阶段 ==========
        output = target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )

        # 采样第一个token（非投机）
        output_ids[:, num_input_tokens:num_input_tokens+1] = sample(output.logits[:, -1:, :], temperature)

        # 提取target_hidden（最后一个token的所有层hidden states）
        if self.target_layer_ids is None:
            # 默认提取所有层
            target_hidden = extract_context_feature(output.hidden_states)
        else:
            target_hidden = extract_context_feature(output.hidden_states, self.target_layer_ids)
        # target_hidden: [1, seq_len, num_target_layers * hidden_size]
        # 只取最后一个位置
        target_hidden = target_hidden[:, -1:, :]  # [1, 1, num_target_layers * hidden_size]

        # ========== Decode 阶段（循环）==========
        acceptance_lengths = []
        target_total_time = 0.0
        draft_total_time = 0.0
        steps = 0
        start = num_input_tokens + 1  # 从第一个生成的token之后开始

        while start < max_length:
            # 构造块输入
            block_output_ids = output_ids[:, start:start+block_size].clone()
            block_position_ids = position_ids[:, start:start+block_size]

            # 通过target模型的embedding层获取noise_embedding
            noise_embedding = target.model.embed_tokens(block_output_ids)

            # 小模型前向（无KV Cache！）
            draft_start_time = time.time()
            draft_hidden = self(
                target_hidden=target_hidden,  # [1, 1, num_target_layers * hidden_size]
                noise_embedding=noise_embedding,  # [1, block_size, hidden_size]
                position_ids=block_position_ids,
            )
            draft_total_time += time.time() - draft_start_time

            # 通过target模型的lm_head得到logits
            # 只取后block_size-1个位置（第一个位置是已知的anchor token）
            draft_logits = target.lm_head(draft_hidden[:, -(block_size-1):, :])

            # 采样B-1个token
            block_output_ids[:, 1:] = sample(draft_logits, temperature)

            # 大模型验证
            target_start_time = time.time()
            output = target(
                block_output_ids,
                position_ids=block_position_ids,
                past_key_values=output.past_key_values if hasattr(output, 'past_key_values') else None,
                use_cache=True,
                output_hidden_states=True,
            )
            target_total_time += time.time() - target_start_time

            # 采样验证结果
            posterior = sample(output.logits, temperature)

            # 接受/拒绝逻辑（最长前缀匹配）
            matches = (block_output_ids[:, 1:] == posterior[:, :-1])
            acceptance_length = matches.cumprod(dim=1).sum(dim=1)[0].item()

            # 更新输出序列
            output_ids[:, start:start+acceptance_length+1] = block_output_ids[:, :acceptance_length+1]
            output_ids[:, start+acceptance_length+1:start+acceptance_length+2] = posterior[:, acceptance_length:acceptance_length+1]

            # 更新状态
            start += acceptance_length + 1
            acceptance_lengths.append(acceptance_length + 1)
            steps += 1

            # 更新target_hidden（从验证结果中提取）
            if self.target_layer_ids is None:
                new_target_hidden = extract_context_feature(output.hidden_states)
            else:
                new_target_hidden = extract_context_feature(output.hidden_states, self.target_layer_ids)
            # 只取已接受token的hidden states的最后一个
            target_hidden = new_target_hidden[:, acceptance_length:acceptance_length+1, :]

            # 检查停止条件
            if stop_token_ids is not None and any(
                stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
            ):
                break

        # 清理输出
        output_ids = output_ids[:, :max_length]
        output_ids = output_ids[:, output_ids[0] != self.mask_token_id]

        if stop_token_ids is not None:
            stop_token_ids_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
            stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids_tensor).nonzero(as_tuple=True)[0]
            if stop_token_indices.numel() > 0:
                output_ids = output_ids[:, :num_input_tokens + stop_token_indices[0] + 1]

        # 保存统计信息
        self._last_decode_stats = {
            "accept_lengths": acceptance_lengths,
            "target_total_time": target_total_time,
            "draft_total_time": draft_total_time,
            "steps": steps,
        }

        return output_ids

    def get_last_decode_stats(self):
        """返回上次生成的统计信息"""
        return getattr(self, "_last_decode_stats", None)
