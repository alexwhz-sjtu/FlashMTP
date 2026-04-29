# FlashMTP

## Background
我现在在做一个投机解码的工作。

**传统的投机解码**：草稿模型是自回归的太慢了。然而文字之间语义是连贯的，相关的，因此我可以利用双向注意力/扩散原理来进行少次（理想是1次）前向就生成一段长度的候选token。

**KV cache抛弃**： 对于草稿模型，kvcache是冗余的。大模型***生成的最新的隐藏状态***应该是计算了所有历史信息，理论上是对前文的浓缩。因此我将这个作为上下文中枢（Contextual Pivot）可以只使用这个信息就可以预测后面一块内容。此外，大模型不同深度的层关注前文不同的信息，因此我会纳入所有层的hidden states，进行信息提取。


### 核心
我的核心就是去掉kvcache。请不要变动并且相信大模型最新hiddenstates信息足够。并且，对于大模型每层，关注的历史token是不同的，不同层hiddenstates应该已经包含了token的交互，只是我没有利用好，请基于此继续思考

## Preliminary：定义说明，相关原理与关联工作

### 定义
* Contextual Pivot (上下文枢轴)：目标模型最新的融合hidden sstates，它是连接过去（全量历史）与未来（生成块）的支点。
* hs：hidden states的简称。hs可以是任意层的输出hs
* 训练数据：我的训练数据全部是目标大模型生成的响应，这样可以对齐。

### 掩码离散扩散语言模型
- 前向过程（加噪/扩散）： 这是一个人为破坏数据的过程。给定一段文本，我们随机地将一些 Token 替换为特殊的 [MASK] 标签。随着步数 $t$ 的增加，掩码比例逐渐增大，直到 $t=T$ 时，文本变成纯粹的 [MASK] 序列。

- 反向过程（去噪/生成）： 这是模型学习的任务。模型接收一个带有 [MASK] 的序列，目标是预测这些 [MASK] 位置原本的 Token。在推理阶段，我们可以从全 [MASK] 开始，通过一次或多次迭代，不断填补并修正这些位置。

MDLM 的训练本质上是一个多尺度的填空任务。

- 采样掩码比例：对于每一条训练数据，随机选择一个掩码率 $\sigma \in (0, 1]$。这对应了扩散过程中的不同时间步。

- 构造输入：根据该比例将 Token 序列的部分内容替换为 [MASK]。

- 预测与损失：将掩码序列输入模型，要求模型预测所有被掩码位置的原始 Token。损失函数：通常采用交叉熵损失（Cross-Entropy Loss）。

### 连续扩散语言模型 LangFlow
嵌入空间扩散：将 Token 映射到连续的嵌入空间（Embedding Space）进行扩散 。这种方法能避免维度灾难，且更易于进行编辑和少步生成 。

### 基本原理：

1. 嵌入空间扩散 (Embedding-space Diffusion)：LangFlow 在连续向量空间进行加噪：$$z_t = (1 - \sigma_t) x_0 + \sigma_t \epsilon$$ 
$x_0$：原始文本的嵌入向量（Embedding）。     
$\epsilon$：高斯噪声 $\mathcal{N}(0, I)$。  
$\sigma_t$：噪声水平（由噪声调度器控制）。它描述了如何将干净的词向量逐渐变模糊直到变成纯噪声的轨迹。

2. 基于 Bregman 散度的流匹配   
文章证明了在连续流模型中使用交叉熵（Cross-Entropy）的合法性。模型不再试图恢复具体的向量数值，而是通过当前带有噪声的向量 $z_t$，预测 Token 分布：$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} [ -\log p_\theta(x_0 \mid z_t) ]$$

3. 信息均匀调度器 (Information-uniform Scheduler)：  
作者认为文本在不同噪声水平下携带的信息量是不均匀的。调度器应该根据 对数信噪比（log-SNR） $\gamma$ 来设计。作者的目标是让时间步 $t$ 与模型能够还原的信息量（由互信息或交叉熵衡量）成线性关系。  
Gumbel 调度器：作者发现信息增益的分布呈现出一种类似“先慢后快再慢”的特征，这与 Gumbel 分布 的累积分布函数（CDF）高度吻合。噪声水平 $\sigma_t$ 不再是简单的线性函数，而是通过一个受 Gumbel 分布启发的函数来计算：$$\gamma(t) = \text{Gumbel\_CDF}(t; \mu, \beta)$$其中 $\mu$ 和 $\beta$ 是可学习的参数。

### 训练与采样流程   
1. 准备阶段特征提取：  
首先通过一个预训练的嵌入层（或随机初始化后随模型训练）将输入的 Token 序列映射为连续向量 $\mathbf{X} \in \mathbb{R}^{L \times D}$。  
噪声采样：为每个 batch 随机采样一个时间步 $t \in [0, 1]$，并根据调度器计算对应的噪声强度 $\sigma_t$。  
2. 构造扰动输入加噪：根据公式 $z_t = (1-\sigma_t)x_0 + \sigma_t\epsilon$ 混合原始嵌入和高斯噪声。   
  **Self-Conditioning**（关键细节）：在 50% 的训练迭代中，模型先进行一次前向传播得到预测值 $\hat{x}_0$。然后将 $\hat{x}_0$ 重新喂给模型作为额外的条件输入（Conditioning）。作用：这能极大地提升模型在生成时的稳定性和连贯性，是连续扩散模型追平自回归模型的“秘籍”。  
3. 模型前向计算将扰动后的序列 $z_t$（以及可选的 self-conditioning 信息）输入 Transformer 编码器。模型输出每个位置在词表上的概率分布 $P(\mathbf{V} \mid z_t)$。  
4. 损失计算与优化目标函数：直接计算预测分布与真实 Token 之间的 Cross-Entropy Loss。梯度更新：使用 AdamW 等优化器。由于是全并行计算，训练效率远高于自回归模型。

### 相关工作：扩散投机解码 DFlash
DFlash也利用了大模型的hs，但是他保留了kvcache。它间隔的选取了五层大模型的hs，再沿着特征维度拼接，用fc层降维，他的kvcache就是每个token位置对应的大模型的融合hs。推理时，他把所有位置融合hs注入到每层充当kvcache，拼接B个mask，一次前向预测B个token。

训练时也是一次前向计算loss，越靠前的位置loss权重越大。


### 冲程蒸馏 Streak-Distillation（与本项目对齐的说明）

核心：**不显式逐点 KL**，而是最大化投机解码下的**期望接受 streak**；用教师轨迹上的联合质量连接「验证器按前缀接受」与「草案并行预测」。  
本项目里草案仅以 **Pivot $p$** 为条件（无草案侧 KV）；验证仍按完整前文 + 已接受前缀。相对原文只是把 $Q_{\text{diff}}$ 的条件从前缀 $s$ 换成 $p$，细节见下文 v3.3。

## 进度与版本迭代

### v1
v1是基础结构，和DFlash类似，只不过我只用了Contextual Pivot hs。并且，我认为需要充分利用大模型，提取所有层hs可以包含信息流动的pattern，因此我的hs选取了大模型所有层。

效果和dflash相差一个接收长度。

### v1.1
hs的拼接降维使用fc，本质上对于不同层融合权重是固定的。我认为，信息流动是变化的，每层计算的权重是输入相关得（input-aware）。即，对于不同Pivot，权重应该变化。

因此，我用一个轻量attention层进行hs融合。我按照序列维度输入所有hs，用最后一层的hs去attend之前的hs，计算完将其作为草稿模型条件输入。

### v3
我认为，上述的方法，直接让模型学习噪声到数据的映射，过于困难。我采用离散扩散语言模型（MDLM）的训练方法，对草案进行逐步 mask / unmask。我曾保持越靠前 loss 权重越大，但多次迭代并未显著拉长接受长度：距离衰减的权重会让靠前位置置信度偏高、靠后位置偏低，于是按置信度逐步 unmask 的策略收益有限。



### v3.3
我用一个轻量attention层进行hs融合。我按照序列维度输入所有hs，用最后一层的hs去attend之前的hs，计算完将其作为草稿模型条件输入。当前代码应该已经实现。

**Step1 训练侧调整**：先按常规 MDLM 训练（不加位置权重）。噪声调度可以偏向于全mask更多。loss可以使用CE和KL散度组合，两者都可以在脚本中控制权重。同时KL散度可以指定计算预测词表中topk的kl散度。选择all则是全词表。

**Step2 冲程蒸馏（Streak-Distillation）**

训练数据已是目标模型生成的响应，故块内「教师」token 与验证分布 $P$ 同源；下文仍写 $\mathbb{E}_{x\sim P(\cdot|c)}$ 仅为与 streak 文献记号一致，实现上可直接用数据中的续写片段。

---

**1. 记号**

| 记号 | 含义 |
|------|------|
| $p$ | **Contextual Pivot**：当前步多层的 HS 融合，草案 **唯一** 显式条件（替代原文的前缀 $s$）。 $p$是随机采样的anchor position。
| $P$ | 目标/验证侧自回归分布（与数据来源一致），用于接受概率与 streak 目标中的轨迹。 |
| $Q_{\text{diff}}(\cdot \| p)$ | 仅以 $p$ 为条件的离散扩散草案，并行预测块内 token。 |

接受 $\alpha_j$ 仍依赖验证器前缀；草案分支不展开 KV，信息经 $p$ 注入。

---

**2. 要点（与原文同构，条件 $s\to p$）**

1. 优化期望接受长度 $\text{Tokens}_{\text{Draft}}(\gamma, p)$。  
2. 用 $P$ 上的贪婪代理替代不可微的拒绝采样。  
3. 在教师轨迹上抬升 $Q_{\text{diff}}$ 的联合质量，促成长 streak。  
4. $Q_{\text{diff}}(x_j|p)$ 不显式依赖 $x_{1:j-1}$；前缀依赖经验证期望进入目标。

---

**3. 预期接受令牌数（草案条件为 Pivot）**

设块长 $\gamma$，草案采样 $x_{1:\gamma} \sim Q(\cdot \| p)$。记验证器在第 $m$ 步的（条件）接受概率为 $\alpha_m(\cdot)$，其自变量为 **验证前缀**（含真实上文与已接受草案）。自然推广为：

$$
\text{Tokens}_{\text{Draft}}(\gamma, p) = \mathbb{E}_{x_{1:\gamma} \sim Q(\cdot|p)} \left[ \sum_{m=1}^{\gamma} \prod_{j=1}^{m} \alpha_j\bigl(c \circ x_{1:j-1}\bigr) \right]
$$

其中 $c$ 为与 $p$ 对齐的验证前文；$\alpha_j$ 随已接受前缀变，$Q$ 侧条件固定为 $p$。

---

**4. 贪婪接受代理**

$$
\tilde{\alpha}_j(p) \approx \mathbb{E}_{x_{1:j-1} \sim P(\cdot|c)}\, \mathbb{E}_{x_j \sim P(\cdot|c \circ x_{1:j-1})}\bigl[\, Q_{\text{diff}}(x_j \,|\, p) \,\bigr]
$$

$Q_{\text{diff}}(x_j|p)$ 为位置 $j$ 在 pivot 条件下的预测概率（与掩码日程一致即可）。

---

**5. Streak 目标（Pivot 版）**

$$
\mathcal{L}_{\text{streak}}(\theta) = \mathbb{E}_{(p,c)}\;\mathbb{E}_{\,x_{1:\gamma} \sim P(\cdot|c)}\left[ \sum_{m=1}^{\gamma} \prod_{j=1}^{m} q_j\bigl(x_j \,\big|\, p\,;\theta\bigr) \right]
$$

- 每条样本上取对齐的 $(c,p)$；$x_{1:\gamma}$ 为响应中的续写块（形式上等价于 $P(\cdot|c)$ 的轨迹）。  
- $q_j$ 为草案在位置 $j$ 的概率；并行下联合取 $\prod_j q_j(\cdot|p)$。

目标：在固定 $p$ 下拉高与目标续写一致的 **长 streak**，而非只对齐首 token。

---

**小结**：v3 = MDLM 主训练 + **Pivot 条件 streak 蒸馏**；相对标准 Streak-Distillation 仅将 $Q_{\text{diff}}$ 的条件由前缀 $s$ 改为 $p$。

### v4
我希望模型尽可能的利用我的pivot。我在思考使用LangFlow连续扩散原理进行草稿模型构建，之后再使用连续扩散模型的蒸馏方式。


