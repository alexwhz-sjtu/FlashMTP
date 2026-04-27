# FlashMTP

## Ours core idea

由于隐状态是模型在**完整上下文**下计算得到的，因此它们可以看作对上下文的**浓缩表示**。在预测后续 block 的 token 时，我们只需要**最新的隐状态**即可。

我们提出 FlashMTP：利用**最新的隐状态**并结合**双向注意力的**扩散原理，高效生成草稿 token

## Base structure

与 DFlash 类似。但我们使用**所有层**的 bonus 隐状态。原因在于：在生成隐状态时，各层会关注上下文的不同部分，因为不同层、不同注意力头的模式差异很大。我们沿 **特征 / 序列 维度**把它们拼接起来，并作为条件使用。随后把 bonus 的干净 token 与若干 mask（噪声）拼接起来，**只做一次前向**。其中噪声 block 作为 **Q**，拼接后的序列作为 **KV**。

## v1.1 Improved condition injection

- 为提升模型表达能力与条件信息量，我们把**整条拼接序列**都作为 **Q** 输入模型。这样前缀可以在各层之间被逐步处理，每一层都能得到**不同的前缀表示**。
- 在构造前缀隐状态时，我们**把初始 embedding 也纳入其中**。
- **seq 模式**：各层对应的隐状态使用**相同的位置 id（position id）**。进入attention之前用线性层将其转换到embedding空间。

## v2: Improved structure

加入最新一段kvcache

## v3: Diffusion-based draft model

### 1. previous version

事实上之前的版本并没有基于扩散原理。仅仅是输入B个mask直接映射到干净token

### 2. Consistency distillation [[2511.19269] CDLM: Consistency Diffusion Language Models For Faster Sampling](https://arxiv.org/abs/2511.19269)

base on base structure

### 1. 模型原理（Model Principle）

本方案将**每次 draft 的 B 个 token** 视为一个独立的短序列（长度 L = 16），并将 Inner block size 设置为 B_in = 1。

- **核心思想**：模型在一次前向传播中，只负责预测当前连续的 16 个 token，不预测后续内容。下一次预测属于新的独立 forward。
- **Teacher**：原始自回归模型，用于生成高质量轨迹。
- **Student**：我们要训练的 draft 模型。输入条件是大模型bonus token之前的融合hiddenstates。bonustoken拼接上noiseembedding作为Query
- **训练目标**：让学生模型从任意中间状态 y，稳定地一次性预测出高质量的 16 个 token（即实现可靠的 multi-token draft）。

通过将 B=16 视为完整序列、B_in=1 视为最小块，模型可以同时利用：

- Distillation Loss：学习 Teacher 的正确预测
- Consistency Loss：学会从“部分完成”到“全部完成”的稳定跳跃

这样训练出的模型适合作为投机解码（Speculative Decoding）的 draft 模型，一次 forward 即可输出 16 个 draft token。

两个状态：

- **y**：**当前采样的解码中间状态**（部分完成的草稿）
- **y**：当前 innerblock token 被 unmask  后的状态。B长度内在这之后的token还是mask着的。（inner_block completion states）

### 2. 损失函数（Loss Functions）

#### 2.1 Distillation Loss（蒸馏损失，主损失）

$L_{\text{Distillation}} = \mathbb{E} \frac{1}{|U_y|} \sum_{i \in U_y} D_{KL} \left( p_i^{(T)} \parallel q_\phi(\cdot | y, x)_i \right)$

- $U_y$ ：当前 B=16 个位置中，从 y 到 y* 之间**新被 unmask** 的 token 位置。一次B_in个token（即1）
- ( $p_i^{(T)}$ ：Teacher 使用 hidden state 重建的 softmax 预测。
- **作用**：强制学生在看到部分 token 的情况下，预测出 Teacher 会最终选择的 16 个 token。

#### 2.2 Consistency Loss（一致性损失，关键稳定损失）

$$
L_{\text{Consistency}} = \mathbb{E} \frac{1}{|S_y|} \sum_{i \in S_y} D_{KL} \left( q_\phi^-(\cdot | y^*, x)*i \parallel q*\phi(\cdot | y, x)_i \right)
$$

- $S_y$ ：在 y* 中仍然保持 masked 的位置（即当前 B 个序列中尚未 final 的 token）。
- $q_\phi^-$ ：带 stop-gradient 的目标（防止训练不稳定）。
- **作用**：让学生在 “少信息状态 y” 和 “innerblock completion 状态 y* ” 之间保持预测一致，实现稳定跳跃。

#### 2.3 总损失函数

$$
L = w_{\text{distill}} \cdot L_{\text{Distillation}} + w_{\text{cons}} \cdot L_{\text{Consistency}}
$$

**推荐权重**（B=16, L=1）：

- $w_{\text{distill}}$ = 1.0 
- $w_{\text{cons}} $= 0.6

这里有一个问题，$q_\phi^-$ 需要学生模型自己来预测。但我是从零初始化，因此需要先训练一个多步迭代扩散草稿模型（纯蒸馏损失），再加入一致性损失。上面的两个权重给出调整接口。

## 3. 训练流程（Training Procedure）

### 3.1 准备阶段（Offline Trajectory Collection）

1. 使用 自回归目标模型 模型对大量 prompt 生成轨迹。（已完成）
2. 每条轨迹记录：
  - 人为构建 块内 B 内的 Token 序列轨迹 ($\mathcal{T}_x$。按照从左到右的因果顺序。也就是说，解码轨迹模拟自回归生成的轨迹。
  - 对应 hidden state buffer（用于重建 Teacher logits）*这个在线生成

### 3.2 训练阶段（每一次迭代）

1. 基于base structure，我们已经采样了一个anchor token。那么后面的目标就是预测此锚点块的解码轨迹 $\mathcal{T}_{x}$）。
2. **随机采样中间状态 y**：
  - 在轨迹中随机选择步骤 k。在这里，因为轨迹是因果的，因此可以直接采样一个块内位置p，其之前p个token看作已生成的轨迹，后面的都是mask。
  - y = $\mathcal{T}_x$)
  - y* = $\mathcal{T}_x'$) 即当前B_in中被全部unmask后的状态。这里就是第p+1个token被unmask
3. 计算两个损失（所有计算均限制在当前 B=16 个 token 范围内）：
  - Distillation Loss（在 Uy 上）
  - Consistency Loss（在 Sy 上）
4. 计算总损失并反向传播，更新学生模型参数。
5. 重复上述过程直到收敛。

### 3.3 推理阶段（Inference / Drafting）

- 输入：大模型融合hiddenstates + 当前需要 draft 的 16 个 masked 位置。
- 一次前向传播：模型直接预测当前 16 个 token。

## TODO：

1. 修改loss
2. diffusion based distillation

