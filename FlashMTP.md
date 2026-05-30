# FlashMTP v1.1

## Background

投机解码的加速上限取决于草稿模型在**一次前向**内能并行猜对多少 token。草稿模型参数量小，在 agent 等**长上下文**场景下若仍维护完整 KV cache，既占显存又难以复用目标模型已算好的表征。

我们的前提是：目标模型在**完整前文**上算出的隐状态，已是对历史的充分压缩；草稿侧**不必再展开历史 KV**，只需以该压缩表征为条件，预测紧接着的一小段续写（一个 block）。

DFlash 仍把各位置的融合 hs 当作草稿侧的 cache。

### 定义

* **Contextual Pivot（上下文枢轴）**：目标模型在某一时刻、多层上的隐状态集合，作为「过去 → 未来块」的支点。训练时 Pivot 取 **anchor 前一 token** 处各选定层的 hs（即块起点之前的浓缩点）；**anchor token** 则是块内第一个**干净** token（bonus），与 Pivot 在输入序列上相邻但职责不同。
* **hs**：hidden states 简称，可来自目标模型任意 transformer 层。
* **训练数据**：响应全部由目标大模型生成（可经 regenerate 对齐分布），使块内监督 token 与验证器行为一致。
* **anchor token**：每条样本上随机采样的块起点 token。输入为：**Pivot 条件** + anchor 的 embedding（clean）+ 其余 $B-1$ 个 mask（噪声）embedding；监督从块内**第二个**槽位起，对应 teacher 轨迹上 anchor 之后的续写。

---

## 模型架构

### Prefix condition（前缀条件）

目标模型在 Pivot 位置抽取**若干层** hs（采用首尾加中间间隔采样），沿**序列维**把每一层视为 prefix 中的一个 token，再与草稿块拼接进同一次注意力。

* **语义**：不同深度对前文的「看法」不同；序列化 prefix 让草稿网络在层与层之间逐段读取，而不是一次性看融合向量。
* **注意力角色**：草稿块 token 为 **Q**；prefix + 块内 token 共同构成 **K/V**。块内为**双向**注意力（并行生成），不同训练块之间**不可互看**（仅见本块的 Pivot prefix）。

### Depth embedding（层深嵌入）

多层 hs 若共用同一套位置编码，模型难以区分「来自第几层」。为每层 Pivot 加上**可学习的 layer index embedding**，再展平为 prefix 序列。

* **与 RoPE 的分工**：RoPE 刻画 prefix / 块内的**相对几何**；depth embedding 刻画**信息来自目标模型哪一层**。
* **作用**：避免不同深度的表征在 prefix 序列中被混为同质 token，使层间差异可被草稿模型显式利用。

### Local position id（块内局部位置）

每个投机块是独立语义单元；块内 token 应学习**局部相对次序**，而非绑定全文绝对距离。

* **草稿块**：位置 id 在块内从 **1 到 B** 重复（每个并行训练块一套），与全局序列长度解耦。
* **Pivot prefix**：RoPE 位置均设为 **0**，与块内坐标系对齐，表示「紧贴块起点之前的条件点」。
* **动机**：训练可在较短 `max_length` 上完成，推理时长文档上块内几何不变，减轻绝对位置外推压力。目标模型仍用全局位置算 Pivot；仅草稿侧使用局部 id。

---

## 训练范式（与架构配套）

1. 目标模型对整段响应做一次前向，导出各层 hs（在线）或读缓存（离线）。
2. 在 `loss_mask` 有效区间内**随机采样**多个 anchor，每条序列形成多个并行块，提高样本效率。
3. 每块构造：Pivot 多层 hs → prefix；块内 slot0 = anchor（clean），slot $1..B-1$ = mask embedding。
4. 单次草稿前向 + 块内 CE；**无草稿侧 KV cache**，历史信息仅经 Pivot（及 prefix 序列）注入。

---

## Loss 设计

主目标：让草案在**固定 Pivot** 条件下，对块内续写 token 的预测分布接近目标模型（与投机验证一致）。

### 监督范围

* 块内位置 $=0$（anchor / bonus）**不参与** loss，避免把边界 token 当并行草稿目标。
* 对 $k=1,\ldots,B-1$，用 teacher 轨迹上 **anchor+k** 的 token 作标签（同位预测：看见 anchor 与部分 mask 布局后，预测下一段真实续写）。
* 仅在 `loss_mask` 有效、且未越出序列长度的位置上计算。

### 主损失：块内加权交叉熵

$$
\mathcal{L}_{\text{CE}} = \frac{\sum_{k \in \mathcal{U}} w_k \cdot \mathrm{CE}\bigl(q_\phi(\cdot \mid p),\, y_k\bigr)}{\sum_{k \in \mathcal{U}} w_k}
$$

其中 $p$ 为 Pivot 条件，$\mathcal{U}$ 为块内可监督位置集合，$y_k$ 为 teacher token。logits 默认经**冻结的目标 lm_head** 读出（亦可选用可训练的 draft head 做消融）。

权重：**指数衰减** $w_k \propto \exp(-(k-1)/\gamma)$ | 继承「越靠前越重要」的投机直觉：前几个 slot 猜错会立刻截断接受链。
### 评估代理（非训练 loss）

块内从 slot1 起连续猜对的个数 + 1，在有效块上平均，作为**期望接受长度**的贪婪代理，与推理时逐 token 验证的 streak 行为同向，便于监控是否学到「长串并行命中」。


### 启动脚本
测评/推理启动脚本：/evaluation