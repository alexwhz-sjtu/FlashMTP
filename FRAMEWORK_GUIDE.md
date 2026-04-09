* 定义：在投机解码中。小模型会生成多个候选token，大模型并行验证。我们将大模型验证产生的最后一个token称为bonus token。

# FlashMTP

## 一、背景

我正在进行投机解码优化的工作。我利用一个多层并行预测小模型，输入多个mask，一次预测多个token。具体来讲，我利用大模型的隐藏状态当作condition注入小模型中。

## 二、核心设计思想

### 1.1 任务定义

小模型（Draft Model）接收大模型（Target Model）最后一个token的**所有层hidden states拼接**（序列维度或者特征维度，可选，请给出选择接口），预测接下来**B个token**（B=block_size，默认16）。

### 1.2 关键特点

- **无自回归依赖**：小模型预测块内token时，**不依赖**ground truth token的embedding
- **双向注意力**：块内token之间是双向注意力（非因果）

📋 文档核心内容

一、核心设计思想

你的小模型接收大模型最后一个token的所有层hidden states拼接作为输入，预测接下来B个token（block_size，默认16）。关键创新是：预测块内token时不依赖ground truth的embedding，而是用mask token填充。

二、推理流程（带详细图解）

- Prefill阶段：大模型处理prompt，提取 target_hidden（最后一个token的所有层hidden拼接）
- Decode循环：
  a. 构造块输入：[target_hidden, real_token, mask_id, mask_id, ...]，real_token是经过验证的干净的最后一个token（bonus token）。
  b. 小模型通过双向注意力生成B-1个token。[real_token, mask_id, mask_id, ...]作为query。target_hidden（一个或多个不加位置编码）
  c. 大模型验证整个块
  d. 接受/拒绝（最长前缀匹配）
  e.  更新target_hidden

三、训练流程（带详细图解）

- 训练Step：随机采样anchor点 → 构造mask块 → 联合块训练（关键！）→ 加权损失
- 稀疏注意力掩码：确保同序列内多个块之间不互相看到
- WeightedBlockLoss：块内位置指数加权（早期位置权重更高）

## 三、变体

1. 大模型最后一个hiddenstates，所有层沿着feature维度拼接

* 拼接之后通过fc层降维，拼接上噪声块

2. 沿着seq维度拼接

* hiddenstates当作L个位置，拼上噪声块。隐藏状态不加positionid

# v1.1

1. 将L+1个hiddenstates全部包括。即包括最开始的embedding，提供起始点参考。
2. seq拼接时，添加fc层，对每个hs进行等维投影，变换到embedding空间。对每个hs用相同positionid？

# v2

## 核心改进（仅支持feature拼接模式，删除seq模式相关）

1. **问题诊断**：只用 hidden states 会造成局部语义不连续，出现极低接受长度频率显著增加。同时因为没有显式上文，预测上限不高（10/16）。
2. **解决方案**：

   * **显式包含最新一部分上文（使用 KVCache 形式）**：类似于 sliding window，KVCache 用**因果注意力**保证 cache 是无损的
   * **上文信息注入**：bonus token的融合 hidden states 注入到每个noise embedding位置：具体在 feature 维度上拼接再和每一个 noise embedding 拼接，经过 fc、norm 降维到 hidden_dim 作为输入 query
   * **噪声块双向注意力**：噪声块部份是**双向注意力**

## v2 Mask 构造详解

### 布局 (Layout)

```
QKV: [full sequence token| Block_0 | Block_1 | ... | Block_{N-1}]
         └ 因果注意力 ┘         └─ 双向注意力 ─┘
```

### 注意力规则

1. **KVCache 部分（因果注意力）sliding window风格（窗长W）**：

   - token_i 只能看到 token_{i-W+1} 到 token_i（包括自己）
   - 这是标准的自回归因果滑动窗口掩码
   - 保证局部语义连贯
   - 注意是原始token经过KV后缓存的kvcache，不是hiddenstates！
   - bonus token的融合 hidden states 注入到每个noise embedding位置：具体在 feature 维度上拼接再和每一个 noise embedding 拼接，经过 fc、norm 降维到 hidden_dim 作为输入 query
2. **Block 部分（双向注意力 + 全局上文可见）**：

   - Block_i 可以看到**前W个token**（上文信息）
   - Block_i 内部是**双向注意力**（块内所有位置互相可见）
   - Block_i **看不到**其他 Block_j (j ≠ i)
3. **有效性掩码**：

   - 无效的 block (block_keep_mask=False) 的 query 不会参与计算。
4. **训练mask**：

   * 使用一个特殊的注意力掩码（Attention Mask），使得以anchor position t位置为锚点的block只能显示计算 [t−W+1, t-1] 范围内的 token（其中 W 是窗口大小）。
   * 注意：anchor token 本身**不包含**在上文窗口中，因为它作为 bonus token 出现在 Block 的起始位置
   * noise block 内部还是双向注意力

---

## v2 训练工作流程

### 输入构造

```
Layout: [Full Sequence (seq_len tokens) | Block_0 | Block_1 | ... | Block_{N-1}]
        └────── 因果注意力 ──────┘     └──── 双向注意力 ────┘
```

1. **Full Sequence Embeddings**
   - 对输入 `input_ids` 进行 embedding lookup
   - 作为 KVCache 部分，使用因果注意力

2. **Block Embeddings**
   - 每个 Block 包含 `block_size` 个位置
   - 第一个位置是 **bonus token**（即 anchor position 的真实 token）
   - 其余位置是 **MASK token**
   - 通过 `embed_tokens` 获取初始 embedding

3. **Target Hidden States 注入**
   - 从 Target Model 提取 anchor_position-1 处的所有层 hidden states
   - 沿 feature 维度拼接：(B, N, H×L)
   - 通过 `fc + norm` 投影到 hidden_size：(B, N, H)
   - 将投影后的 hidden states 拼接到每个 Block 位置的 embedding 上
   - 通过 `block_input_proj + norm` 降维回 hidden_size

### Attention Mask 规则

```python
# 整体布局索引
Full Sequence: [0, seq_len-1]
Block_i:       [seq_len + i*block_size, seq_len + (i+1)*block_size - 1]

# Case 1: Full Sequence 内部 → 因果注意力
kv_idx <= q_idx

# Case 2: Block_i → Full Sequence → 只能看到上文窗口
window = [anchor_i - W + 1, anchor_i - 1]  # 不包含 anchor_i

# Case 3: Block_i 内部 → 双向注意力
所有位置互相可见（包括 bonus token 和 mask tokens）

# Case 4: Block_i ↔ Block_j (i≠j) → 不可见
完全被屏蔽
```

### 前向传播流程

```python
# 1. 采样锚点
anchor_positions, block_keep_mask = sample_anchors(...)

# 2. 构造输入
full_seq_embeds = embed_tokens(input_ids)                          # (B, seq_len, H)
block_embeds = create_block_embeddings(input_ids, anchors)         # (B, N*bs, H)
noise_embedding = concat([full_seq_embeds, block_embeds])          # (B, seq_len+N*bs, H)

# 3. 准备 target hidden states
target_hidden = prepare_target_hidden(hidden_states, anchors)      # (B, N, H*L)
target_hidden_proj = fc(hidden) + norm(hidden)                     # (B, N, H)

# 4. 注入到 blocks（在 DraftModel 内部完成）
block_embeds = reshape(block_embeds, (B, N, bs, H))
block_embeds = concat([block_embeds, target_hidden_proj_expand], dim=-1)  # (B, N, bs, 2H)
block_embeds = block_input_proj(block_embeds) + block_input_norm(block_embeds)  # (B, N, bs, H)

# 5. 构造完整输入
hidden_states = concat([full_seq_embeds, reshape(block_embeds)])   # (B, seq_len+N*bs, H)

# 6. 应用 attention mask
mask = create_flashmtp_block_mask(seq_len, anchors, ...)

# 7. 通过 Draft Model
output = draft_model(hidden_states, mask, position_ids)

# 8. 只取 Block 部分计算 loss
logits = lm_head(output)  # (B, N*block_size, vocab_size)
```

### 关键设计要点

1. **因果 KVCache**：Full Sequence 部分确保上文信息无损传递
2. **Sliding Window**：Block 只能看到 anchor 前的 W-1 个 token，避免重复看到 bonus token
3. **双向 Block**：块内所有位置互相可见，提升并行预测能力
4. **Hidden States 注入**：通过 feature 拼接+投影，将大模型语义信息注入每个预测位置
