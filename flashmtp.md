# FlashMTP：模型结构与训练流程

本文档基于 `scripts/run_training_flashmtp.sh` 与 `scripts/train_flashmtp.py` 的实现，描述论文实验所采用的 **prefix_condition** pivot 融合、**LOCAL_POSITION** 配置、以及草稿模型 **不使用 KV Cache** 的设定。

---

## 1. 总体思路

FlashMTP 是一种面向投机解码（speculative decoding）的草稿模型训练方法。与自回归草稿模型不同，FlashMTP 在每个 anchor 位置一次性并行预测一个长度为 $B$ 的 token 块（block），块内使用**双向注意力**；与 DFlash 等保留草稿 KV Cache 的方法不同，FlashMTP **完全丢弃草稿侧的 KV Cache**，转而依赖目标模型（teacher）在 anchor 前一时刻的多层 hidden states 作为 **Contextual Pivot（上下文枢轴，CHS）**。本文采用 **prefix_condition** 融合：各层 hidden state 以独立 KV 前缀注入草稿 attention（而非压缩为单一向量），配合可学习层深度 embedding 区分 target 层身份；teacher 的最新 hidden state 已编码全部历史信息，足以支撑后续一块 token 的并行预测。

训练数据为目标模型自身生成的响应（regen 数据），以保证草稿分布与验证阶段 target 行为对齐。训练时仅更新草稿模型参数；target 的 embedding 与 lm\_head 冻结共享。

---

## 2. 模型结构

### 2.1 目标模型（Target / Teacher）

- **骨干网络**：Qwen3-8B（`AutoModelForCausalLM`），推理与训练前向均 **冻结**（`eval` + `no_grad`）。
- **作用**：对完整序列做一次前向，输出各 transformer 层的 hidden states；训练时不使用 KV Cache（`use_cache=False`）。
- **Pivot 层选取**：固定取 **首层（layer 0）** 与 **末层（layer $L-1$）**，并在中间等间隔选取 $N$ 层（由 `NUM_MIDDLE_LAYERS_N` 控制），共 $S = N + 2$ 层。默认 $N=5$，即 $S=7$ 层。
- **Pivot 位置**：对每个 anchor 位置 $a$，从各选中层在序列位置 $\max(a-1, 0)$ 处 gather hidden state，得到形状为 $(B_{\text{batch}}, N_{\text{blocks}}, S, H)$ 的多层 pivot 特征。在 **prefix_condition** 模式下，$S$ 层 hidden states **不压缩为单向量**，而是作为 $S$ 条独立的上下文槽位注入草稿 attention（见 §2.2）。

### 2.2 草稿模型（Draft Model）

- **骨干结构**：与 target 同架构族（Qwen3），但层数更少。默认 **5 层**（`NUM_DRAFT_LAYERS=5`），hidden size、head 数等与 target 一致。
- **块大小**：$B = 16$（`BLOCK_SIZE=16`）。
- **无 KV Cache**：草稿前向始终 `past_key_values=None, use_cache=False`；每步仅对当前块做一次完整前向，历史信息完全由 CHS pivot 注入，不在草稿侧累积 K/V。
- **双向注意力**：`is_causal=False`，块内 token 可互相 attend；不同 parallel block 之间互不可见（由 Flex Attention BlockMask 约束）。

#### Contextual Pivot 融合（`PIVOT_FUSE_MODE`）

本文实验采用 **`prefix_condition`**（`PIVOT_FUSE_MODE=prefix_condition`）：将 target 各层 hidden state 作为 **多层 KV 前缀条件** 注入草稿 attention，而非先融合为单一 pivot 向量。

**预处理**（`_fuse_target_hidden`）：

1. 对每个 target 层向量 $h_k$，加上可学习的 **层深度 embedding** `nn.Embedding(num_target_layers, H)`，索引为该层在 target 中的层号 `target_layer_ids[k]`（**不是 RoPE**）。
2. 经 `RMSNorm` 后，将 $(B, N, S, H)$ reshape 为 $(B, N \cdot S, H)$。
3. 每个 parallel block 对应 **`chs_len_per_block = S`** 条上下文槽位（默认 $S=7$），总上下文长度随 pivot 层数线性增长。

**与其它融合模式的区别**（供 ablation 参考）：

| 模式 | 融合方式 | `chs_len_per_block` |
|------|----------|---------------------|
| `linear_fuse` | $S$ 层特征维拼接 → Linear → RMSNorm，压成 1 个 pivot | 1 |
| `attention_fuse` | 层间 attention 融合，取最后位置为 pivot | 1 |
| **`prefix_condition`** | 各层独立保留，加深度 embedding 后作为多条 KV 前缀 | **$S$** |

草稿 **每一层** `Qwen3FlashMTPDecoderLayer` 均传入同一份 `target_hidden`，每层 self-attention 都从这份 KV 前缀读取条件。

#### 注意力注入机制（prefix_condition）

每层 `Qwen3FlashMTPAttention` 将 pivot 映射为额外的 K/V 前缀（CHS），与草稿 token 的 K/V 拼接后做 attention：

- **Q** 来自草稿 noise embedding（anchor 真实 token + 后续 MASK token），**施加 RoPE**。
- **K_ctx / V_ctx**：由 `target_hidden` 经 `k_proj` / `v_proj` 得到，**不参与 RoPE**（仅用层深度 embedding 区分层身份）。
- **K_noise / V_noise**：由草稿序列经 `k_proj` / `v_proj` 得到，**施加 RoPE**。
- **K/V** = `[K_ctx, V_ctx] ∥ [K_noise, V_noise]`（head 维 seq 维拼接）。
- 每个 block 的 query 只能 attend 自己的 **$S$ 个 CHS 槽位** + 自己的 $B$ 个草稿 token（Flex Attention BlockMask 保证块间隔离）。

与 `linear_fuse` / `attention_fuse` 不同：后两者先将多层压成 1 个 pivot，再与草稿 **整块** 一起做 RoPE；**prefix_condition 刻意将前半段 KV 留在 RoPE 之外**，保留各层 target hidden 的独立语义。

#### 位置编码（LOCAL_POSITION 模式）

启用 **`LOCAL_POSITION=true`** 时：

| 分量 | 位置 ID |
|------|---------|
| CHS（pivot 前缀）RoPE | 全 **0**（每个 block 重复 $S$ 次） |
| 草稿块内 token RoPE | 块内局部 **`1, 2, …, B`**（每个 parallel block 重复同一模式） |

草稿不再使用全局绝对位置（`anchor, anchor+1, …`），而是让模型只感知块内相对顺序。**prefix_condition 下 CHS 的 `k_ctx` 仍不参与 RoPE**（实现不变），RoPE 表中的前缀段仅作用于草稿段的 Q / `k_noise`；层间区分完全由可学习深度 embedding 承担。训练与推理时草稿均按此规则构造 `rotary_position_ids`；**target 验证前向仍使用全局 `position_ids`**，与草稿侧解耦。

#### 输出头

默认共享 **冻结的 target `lm_head`** 将草稿最后一层 hidden states 映射为词表 logits。

---

## 3. 训练流程

### 3.1 数据

| 项目 | 配置 |
|------|------|
| 来源 | Nemotron 子集，由 Qwen3-8B **重新生成**的 JSONL（regen 数据） |
| 样本量 | 40,000（`DATA_NUM_SAMPLES=40000`） |
| 模板 | Qwen chat template（`CHAT_TEMPLATE=qwen`） |
| 思考链 | 开启（`ENABLE_THINKING=on`） |
| 最大序列长度 | 4,096（`MAX_LENGTH=4096`） |
| Loss mask | 仅对 assistant 回复段计算损失（prompt / system 等位置 mask 为 0） |

### 3.2 单步训练（Online FlashMTP）

对每个 batch，流程如下：

1. **Target 前向**（冻结）：输入 `input_ids`，得到全部 $L$ 层 hidden states。
2. **Anchor 采样**：在 `loss_mask` 有效区域内随机采样最多 **512** 个 anchor 位置（`NUM_ANCHORS=512`），得到 $N$ 个 parallel block。
3. **构造草稿输入**：
   - 每个 block 第 0 位：anchor 处的**真实 token embedding**；
   - 第 $1 \sim B-1$ 位：**MASK token** embedding。
4. **提取 Pivot**：在 $\max(a-1, 0)$ 处 gather 各选中层的 hidden states；经 prefix_condition 预处理后 reshape 为 $(B, N \cdot S, H)$。
5. **草稿并行前向**：$N$ 个 block 拼成一次前向（草稿序列总长 $N \cdot B$，上下文总长 $N \cdot S$），Flex Attention 保证 block 间隔离；CHS 与草稿 RoPE 按 LOCAL_POSITION 规则构造。
6. **标签与损失**：
   - 位置 $k$（$k \ge 1$）预测 token $y_{a+k}$（同位预测）；
   - 跳过 $k=0$（anchor 本身不参与 CE）；
   - **加权交叉熵**：块内第 $k$ 个预测 token 权重为 $\exp\!\left(-\frac{k-1}{\gamma}\right)$，$\gamma=7$（`LOSS_DECAY_GAMMA=7`），使近 anchor 的 token 权重更高；各 slot 权重由 validity mask 与 decay 决定。
7. **反向传播**：仅更新草稿模型参数；target embedding / lm\_head 不参与梯度。

### 3.3 分布式与优化

| 项目 | 配置 |
|------|------|
| 启动方式 | `torchrun`，8 GPU（`NPROC_PER_NODE=8`） |
| 草稿并行策略 | FSDP（`SHARD_GRAD_OP`），bf16 混合精度 |
| Target | 每卡独立前向（HF backend），不参与 FSDP |
| Batch size | 1 × 1（`BATCH_SIZE=1`, `ACCUMULATION_STEPS=1`） |
| 学习率 | $6 \times 10^{-4}$ |
| 预热 | 总步数 4%（`WARMUP_RATIO=0.04`） |
| 梯度裁剪 | `MAX_GRAD_NORM=1.0` |
| 训练轮数 | 6 epochs（`NUM_EPOCHS=6`） |
| 随机种子 | 42 |
| Attention 后端 | Flex Attention（`ATTENTION_BACKEND=flex_attention`） |
| 日志 / 保存 | 每 50 step 打 log，每 5000 step 存 checkpoint |

---

## 4. 推理时的 KV Cache 策略（与训练一致的设计）

- **Target 模型**：正常使用 KV Cache 做自回归验证（`use_cache=True`），prefill 后逐步 verify 草稿块。
- **草稿模型**：**不使用 KV Cache**。每轮投机步仅对当前 $B$ 个 token 做一次前向；pivot 由 target 在当前接受位置的多层 hidden states 实时 gather，经 prefix_condition 预处理后作为 $S$ 条 KV 前缀注入草稿 attention。
- 该设计与训练一致：草稿从不缓存历史 K/V，完全依赖最新 Contextual Pivot 承载前文语义。

---

## 5. 主要超参数汇总

| 参数 | 符号/环境变量 | 默认值（本文实验） |
|------|---------------|-------------------|
| Target 模型 | `TARGET_MODEL` | Qwen3-8B |
| 草稿层数 | `NUM_DRAFT_LAYERS` | 5 |
| 块大小 | `BLOCK_SIZE` | 16 |
| Pivot 中间层数 | `NUM_MIDDLE_LAYERS_N` | 5（共 7 层 pivot） |
| Pivot 融合 | `PIVOT_FUSE_MODE` | **`prefix_condition`** |
| 局部位置编码 | `LOCAL_POSITION` | **true** |
| Loss 衰减 $\gamma$ | `LOSS_DECAY_GAMMA` | 7 |
| 每序列 anchor 数 | `NUM_ANCHORS` | 512 |
| 最大序列长度 | `MAX_LENGTH` | 4096 |
| 训练样本数 | `DATA_NUM_SAMPLES` | 40000 |
| 训练轮数 | `NUM_EPOCHS` | 6 |
| 学习率 | `LEARNING_RATE` | 6e-4 |
| Batch size（有效） | `BATCH_SIZE` × `ACCUMULATION_STEPS` | 1 |
| GPU 数 | `NPROC_PER_NODE` | 8 |

---

## 6. 与 DFlash 等方法的关键区别（论文表述参考）

1. **无草稿 KV Cache**：不将历史 token 的 K/V 缓存在草稿模型中；每步仅用 target 最新多层 hidden states 作为条件，使草稿参数量与推理状态更小。
2. **prefix_condition 多层前缀**：target 各层 hidden state 不融合为单向量，而是以 $S$ 条独立 KV 前缀注入每层 attention，配合可学习层深度 embedding 区分层身份，CHS 侧不参与 RoPE。
3. **块内双向并行**：一次前向预测 $B-1$ 个 speculative token（slot 0 为 anchor，由 target 提供），而非逐 token 自回归草稿。
4. **LOCAL_POSITION**：草稿 RoPE 与全局序列位置解耦，块内使用固定局部模式 `1…B`，CHS 前缀 RoPE 为 0，便于跨位置泛化块内相对依赖。
5. **单目标 CE 损失**：无 DFlash++ 的辅助对比损失；损失为带指数衰减权重的标准交叉熵。
