"""
FlashMTP: 训练封装类

核心特点:
1. 无KV Cache - 每次前向独立
2. target_hidden作为前缀输入，不添加位置编码
3. 支持联合块训练（Joint Block Training）
4. 支持加权损失（Weighted Block Loss）
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from specforge.modeling.draft.flashmtp import FlashMTPDraftModel

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    BlockMask = None
    create_block_mask = None


class OnlineFlashMTPModel(nn.Module):
    """
    FlashMTP 在线训练封装类

    支持:
    1. 随机采样anchor positions
    2. 构造mask块输入（第一个token是真实的，其余是mask）
    3. 联合块训练（多个块一起前向，稀疏掩码隔离）
    4. 加权交叉熵损失
    """

    def __init__(
        self,
        draft_model: FlashMTPDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        num_anchors: int = 512,
        attention_backend: str = "flex_attention",
        loss_decay_gamma: Optional[float] = None,
        concat_mode: str = "seq",  # "seq" 或 "feature"
    ):
        """
        Args:
            draft_model: FlashMTP草稿模型
            target_lm_head: 目标模型的lm_head
            target_embed_tokens: 目标模型的embedding层
            mask_token_id: mask token的ID
            block_size: 块大小（默认16）
            num_anchors: 每个序列采样的锚点数量
            attention_backend: 注意力后端（"flex_attention" 或 "eager"）
            loss_decay_gamma: 损失衰减参数（None时自动设置）
            concat_mode: 拼接方式（"seq" 或 "feature"）
        """
        super().__init__()
        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.num_anchors = num_anchors
        self.attention_backend = attention_backend
        self.concat_mode = concat_mode

        # 自动设置gamma
        if loss_decay_gamma is None:
            if block_size == 16:
                self.loss_decay_gamma = 7.0
            elif block_size == 10:
                self.loss_decay_gamma = 5.0
            else:
                self.loss_decay_gamma = 5.0 + (block_size - 10) * (7.0 - 5.0) / (16 - 10)
        else:
            self.loss_decay_gamma = loss_decay_gamma

        # 预计算位置权重
        k = torch.arange(1, block_size, dtype=torch.float32)  # 从1开始（跳过anchor token）
        position_weights = torch.exp(-(k - 1) / self.loss_decay_gamma)
        position_weights = position_weights / position_weights.mean()  # 归一化
        self.register_buffer('position_weights', position_weights)

        # 缓存Flex Attention的block mask
        self._cached_block_mask: Optional[BlockMask] = None

    def _sample_anchor_positions(
        self,
        seq_len: int,
        loss_mask: torch.Tensor,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        随机采样anchor positions

        Args:
            seq_len: 序列长度
            loss_mask: 损失掩码 [bsz, seq_len]
            device: 设备

        Returns:
            anchor_positions: [bsz, n_anchors] 锚点位置
            keep_mask: [bsz, n_anchors] 有效锚点掩码
        """
        bs = self.block_size
        bsz = loss_mask.shape[0]
        max_anchor = max(seq_len - bs, 0)

        # 找到有效的锚点位置（考虑loss_mask和block边界）
        valid = loss_mask[:, :max_anchor + 1] > 0.5
        valid_counts = valid.sum(dim=1)
        max_n = min(self.num_anchors, int(valid_counts.max().item()))

        if max_n <= 0:
            raise ValueError("No valid anchor positions found. Check your data.")

        # 随机采样
        indices = torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        masked_indices = torch.where(
            valid, indices, torch.tensor(seq_len + 1, device=device)
        )

        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, torch.tensor(2.0, device=device))

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        anchors = gathered[:, :max_n].sort(dim=1).values

        # 创建keep mask
        keep_mask = torch.arange(max_n, device=device).unsqueeze(0) < valid_counts.unsqueeze(1).clamp(max=max_n)
        anchors = torch.where(
            keep_mask, anchors, torch.tensor(0, dtype=torch.long, device=device)
        )

        return anchors, keep_mask

    def _create_position_ids(self, anchor_positions: torch.Tensor) -> torch.Tensor:
        """
        为并行的draft blocks创建绝对位置编码

        Args:
            anchor_positions: [bsz, n_blocks] 锚点位置

        Returns:
            position_ids: [bsz, n_blocks * block_size]
        """
        bsz, n_blocks = anchor_positions.shape
        device = anchor_positions.device
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        pos_ids = anchor_positions.unsqueeze(-1) + offsets
        return pos_ids.view(bsz, -1)

    def _create_noise_embed(
        self,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        创建noise embedding
        每个块的第一个token是真实的（anchor token），其余是mask token

        Args:
            input_ids: [bsz, seq_len]
            anchor_positions: [bsz, n_blocks]
            block_keep_mask: [bsz, n_blocks]

        Returns:
            noise_embedding: [bsz, n_blocks * block_size, hidden_size]
        """
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size
        device = input_ids.device

        # 初始化全部为mask token
        noise_ids = torch.full(
            (bsz, n * bs), self.mask_token_id, dtype=torch.long, device=device
        )

        # 每个块的第一个位置放anchor token
        block_starts = torch.arange(n, device=device) * bs
        block_starts = block_starts.unsqueeze(0).expand(bsz, -1)

        # 获取anchor tokens
        valid_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, valid_anchor_positions)

        # 填充anchor tokens
        flat_batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n)
        noise_ids[flat_batch_idx, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(self.mask_token_id, dtype=torch.long, device=device),
        )

        return self.embed_tokens(noise_ids)

    def _create_dflash_block_mask(
        self,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        target_hidden_len: int,
        device: torch.device,
    ):
        """
        创建DFlash风格的块掩码

        KV: [Context (target_hidden_len tokens) | Block_0 | Block_1 | ... | Block_{n-1}]
        Q:  [Block_0 | Block_1 | ... | Block_{n-1}]

        规则:
        1. 每个块只能看到自己的context token（target_hidden中对应位置的token）
        2. 块内双向注意力
        3. 不同块之间不可见
        """
        def mask_mod(b, h, q_idx, kv_idx):
            q_block_id = q_idx // self.block_size

            # Context部分（target_hidden）- 每个块对应一个独立的context token
            is_context = kv_idx < target_hidden_len
            # 只允许看到本块对应的context token
            mask_context = is_context & (kv_idx == q_block_id)

            # Block部分
            is_draft = kv_idx >= target_hidden_len
            kv_block_id = (kv_idx - target_hidden_len) // self.block_size
            mask_draft = is_draft & (q_block_id == kv_block_id)

            # 检查块是否有效
            is_valid_block = block_keep_mask[b, q_block_id]

            return (mask_context | mask_draft) & is_valid_block

        B, N = anchor_positions.shape
        Q_LEN = N * self.block_size
        KV_LEN = target_hidden_len + N * self.block_size

        if not FLEX_ATTENTION_AVAILABLE:
            # 回退到标准attention mask
            return self._create_standard_block_mask(
                anchor_positions, block_keep_mask, target_hidden_len, device
            )

        return create_block_mask(
            mask_mod, B=B, H=None, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device
        )

    def _create_standard_block_mask(
        self,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        target_hidden_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        创建标准的块注意力掩码（当Flex Attention不可用时的回退）

        返回: [B, 1, Q_LEN, KV_LEN] 的布尔掩码，True表示可以attend
        """
        B, N = anchor_positions.shape
        Q_LEN = N * self.block_size
        KV_LEN = target_hidden_len + N * self.block_size

        # 创建query和key的块ID
        q_block_ids = torch.arange(Q_LEN, device=device) // self.block_size  # [Q_LEN]
        q_block_ids = q_block_ids.unsqueeze(0).expand(B, -1)  # [B, Q_LEN]

        # 创建mask
        mask = torch.zeros(B, 1, Q_LEN, KV_LEN, dtype=torch.bool, device=device)

        for b in range(B):
            for q_idx in range(Q_LEN):
                q_block = q_idx // self.block_size

                # 可以attend到对应的context token
                if q_block < target_hidden_len:
                    mask[b, 0, q_idx, q_block] = True

                # 可以attend到同块的noise token（双向）
                kv_start = target_hidden_len + q_block * self.block_size
                kv_end = kv_start + self.block_size
                mask[b, 0, q_idx, kv_start:kv_end] = True

        # 应用block_keep_mask
        for b in range(B):
            for q_block in range(N):
                if not block_keep_mask[b, q_block]:
                    q_start = q_block * self.block_size
                    q_end = q_start + self.block_size
                    mask[b, 0, q_start:q_end, :] = False

        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        并行块训练前向

        Args:
            input_ids: [bsz, seq_len] 输入token IDs
            hidden_states: [bsz, seq_len, feature_dim] 目标模型的hidden states
            loss_mask: [bsz, seq_len] 损失掩码

        Returns:
            loss: 标量损失
            accuracy: 准确率
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # 采样anchor positions
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )

        # 创建noise embedding
        noise_embedding = self._create_noise_embed(
            input_ids, anchor_positions, block_keep_mask
        )

        # 创建position ids
        position_ids = self._create_position_ids(anchor_positions)

        # 提取target_hidden（每个anchor位置对应的hidden state）
        # target_hidden: [bsz, n_blocks, feature_dim]
        n_blocks = anchor_positions.shape[1]
        target_hidden_list = []
        for b in range(bsz):
            for n in range(n_blocks):
                if block_keep_mask[b, n]:
                    pos = anchor_positions[b, n].item()
                    target_hidden_list.append(hidden_states[b:b+1, pos:pos+1, :])
                else:
                    # 无效块用零填充
                    target_hidden_list.append(torch.zeros(
                        1, 1, hidden_states.shape[-1],
                        dtype=hidden_states.dtype, device=device
                    ))

        # 拼接target_hidden
        # 注意：两种模式都在序列维度(dim=1)拼接，保持n_blocks个独立的ctx
        # - feature模式: [bsz, n_blocks, feature_dim]，后续通过fc投影到hidden_size
        # - seq模式: [bsz, n_blocks, hidden_size]，直接作为K/V输入
        target_hidden = torch.cat(target_hidden_list, dim=1)  # [bsz, n_blocks, feature_dim]

        # 创建attention mask
        target_hidden_len = target_hidden.shape[1]  # n_blocks
        if self.attention_backend == "flex_attention" and FLEX_ATTENTION_AVAILABLE:
            attention_mask = self._create_dflash_block_mask(
                anchor_positions, block_keep_mask, target_hidden_len, device
            )
        else:
            attention_mask = self._create_standard_block_mask(
                anchor_positions, block_keep_mask, target_hidden_len, device
            )

        # 小模型前向
        output_hidden = self.draft_model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
        )

        # 通过lm_head得到logits
        logits = self.lm_head(output_hidden)

        # 构造labels
        label_offsets = torch.arange(0, self.block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )

        # 构造weight mask
        weight_mask = block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        weight_mask = weight_mask * valid_label_mask.float()

        # 跳过第一个位置（anchor token，已知）
        pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
        weight_mask = weight_mask * (pos_in_block > 0).float()

        # 应用原始loss mask
        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        weight_mask = weight_mask * original_loss_mask_gathered

        binary_eval_mask = weight_mask.view(-1)

        # 应用位置权重衰减
        # position_weights: [block_size-1]（跳过第一个位置）
        decay_weights = torch.ones(self.block_size, device=device)
        decay_weights[1:] = self.position_weights  # 第一个位置权重为0（不计算loss）
        weight_mask = weight_mask * decay_weights.view(1, 1, -1)

        # 计算交叉熵损失
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = target_ids.view(-1)
        flat_weights = weight_mask.view(-1)

        loss_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        valid_token_count = flat_weights.sum() + 1e-6
        loss = (loss_per_token * flat_weights).sum() / valid_token_count

        # 计算准确率
        with torch.no_grad():
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & (binary_eval_mask > 0.5)
            actual_token_count = binary_eval_mask.sum() + 1e-6
            accuracy = correct.sum().float() / actual_token_count

        return loss, accuracy
