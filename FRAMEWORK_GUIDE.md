* 核心定义：在投机解码中。小模型会生成多个候选token，大模型并行验证

## 一、核心设计思想

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
  a. 构造块输入：[target_hidden, real_token, mask_id, mask_id, ...]，real_token是经过验证的干净的最后一个token（bonus token）
  b. 小模型通过双向注意力生成B-1个token。[real_token, mask_id, mask_id, ...]作为query。target_hidden（一个或多个不加位置编码）
  c. 大模型验证整个块
  d. 接受/拒绝（最长前缀匹配）
  e.  更新target_hidden

三、训练流程（带详细图解）

- 训练Step：随机采样anchor点 → 构造mask块 → 联合块训练（关键！）→ 加权损失
- 稀疏注意力掩码：确保同序列内多个块之间不互相看到
- WeightedBlockLoss：块内位置指数加权（早期位置权重更高）
