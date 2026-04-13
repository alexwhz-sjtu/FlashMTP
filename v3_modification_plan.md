# FlashMTP v3 修改方案

## 背景

利用扩散原理构建草稿模型。草稿模型embedding层和lm_head和大模型共享。

## 目标

将现有的 Base Structure 修改为基于 Diffusion 的 v3 结构，实现 Consistency Distillation 训练。

## 当前状态 vs v3 的关键差异


| 特性               | Base Structure (当前) | v3 (目标)                              |
| ---------------- | ------------------- | ------------------------------------ |
| 训练方式             | 单次前向预测所有token       | 模拟扩散过程，从中间状态 y 学习                    |
| 损失函数             | 交叉熵损失               | Distillation Loss + Consistency Loss |
| Block内处理         | 并行预测所有位置            | 模拟自回归轨迹，采样中间状态                       |
| Inner block size | 无 (直接预测整个block)     | B_in = 1                             |


## 核心概念

### 1. 两个关键状态

- **y**: 当前采样的解码中间状态（部分完成的草稿），部分token被mask
- **y***: y经过unmask B_in个token后的状态（inner_block completion state）

### 2. 两个损失

- **Distillation Loss**: 在 U_y（从y到y*新被unmask的位置）上，学生学习teacher的预测
- **Consistency Loss**: 在 S_y（y*中仍然masked的位置）上，保证y和y*的预测一致性

### 3. 训练流程

1. 从轨迹中随机采样块内位置p
2. p之前的token视为已生成，之后的都是mask → 构建状态y
3. 第p+1个token被unmask → 构建状态y*
4. 计算两个损失

---

## 需要修改的文件清单

### 1. `specforge/core/flashmtp.py` - 核心训练逻辑

**修改内容:**

#### 1.1 新增 v3 相关配置参数

```python
class OnlineFlashMTPModel:
    def __init__(...):
        # 新增参数
        self.use_diffusion_training = True  # 是否使用v3扩散训练
        self.w_distill = 1.0  # Distillation loss权重
        self.w_cons = 0.6  # Consistency loss权重
        self.inner_block_size = 1  # B_in = 1
        self.enable_consistency_after_steps = 1000  # 多少步后启用consistency loss
```

#### 1.2 新增状态采样函数

```python
def sample_intermediate_state(
    self,
    input_ids: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    采样中间状态y和y*。

    对于每个block：
    - 在[1, block_size)范围内随机采样一个位置p
    - y: token[0:p]是真实的，token[p:]是mask
    - y*: token[0:p+1]是真实的，token[p+1:]是mask

    返回:
        - y_input_ids: 状态y的input_ids (mask部分用mask_token_id填充)
        - y_star_input_ids: 状态y*的input_ids
        - unmask_positions: 本次被unmask的位置 (用于确定U_y)
    """
```

#### 1.3 新增 Forward 函数（v3版本）

```python
def forward_v3(
    self,
    input_ids: torch.Tensor,
    hidden_states: torch.Tensor,  # Teacher的hidden states
    loss_mask: torch.Tensor,
    global_step: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    v3扩散训练前向传播。

    步骤:
    1. 采样anchor positions
    2. 对每个block采样中间状态y和y*
    3. 分别用draft model预测y和y*状态
    4. 计算Distillation Loss (在U_y上)
    5. 计算Consistency Loss (在S_y上)
    6. 返回总损失
    """
```

#### 1.4 新增损失计算函数

```python
def compute_distillation_loss(
    self,
    student_logits_y: torch.Tensor,  # 学生在状态y的预测
    teacher_logits: torch.Tensor,     # Teacher的预测
    unmask_positions: torch.Tensor,   # U_y: 本次unmask的位置
    block_keep_mask: torch.Tensor,
) -> torch.Tensor:
    """
    L_distill = E[ (1/|U_y|) * sum_{i in U_y} D_KL(p_i^T || q_phi(.|y,x)_i) ]
    """

def compute_consistency_loss(
    self,
    student_logits_y: torch.Tensor,      # 学生在状态y的预测
    student_logits_y_star: torch.Tensor, # 学生在状态y*的预测 (stop-gradient)
    still_masked_positions: torch.Tensor, # S_y: 仍然masked的位置
    block_keep_mask: torch.Tensor,
) -> torch.Tensor:
    """
    L_cons = E[ (1/|S_y|) * sum_{i in S_y} D_KL(q_phi^-(.|y*,x)_i || q_phi(.|y,x)_i) ]

    注意: q_phi^- 是stop-gradient的目标
    """
```

#### 1.5 修改 Teacher Logits 获取

```python
def get_teacher_logits(
    self,
    input_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    anchor_positions: torch.Tensor,
) -> torch.Tensor:
    """
    使用Teacher model的hidden states重建logits。

    对于每个位置，用对应层的hidden state过lm_head。
    """
```

---

### 2. `scripts/train_flashmtp.py` - 训练脚本

**修改内容:**

#### 2.1 新增命令行参数

```python
def parse_args():
    # v3扩散训练相关参数
    parser.add_argument("--use-diffusion-training", action="store_true",
                        help="使用v3扩散训练模式")
    parser.add_argument("--w-distill", type=float, default=1.0,
                        help="Distillation loss权重")
    parser.add_argument("--w-cons", type=float, default=0.6,
                        help="Consistency loss权重")
    parser.add_argument("--enable-consistency-after-steps", type=int, default=1000,
                        help="多少步后开始使用consistency loss")
    parser.add_argument("--inner-block-size", type=int, default=1,
                        help="Inner block size (B_in)，默认为1")
```

#### 2.2 修改训练循环

```python
def main():
    # ...

    for epoch in range(start_epoch, args.num_epochs):
        for step_in_epoch, data in enumerate(progress_bar):
            # ...数据准备...

            # 获取teacher的hidden states (用于重建logits)
            target_output = target_model.generate_flashmtp_data(
                input_ids, attention_mask, loss_mask
            )
            hidden_states = tuple(h.cuda() for h in target_output.hidden_states)

            # v3扩散训练
            if args.use_diffusion_training:
                loss, loss_dict = flashmtp_model.forward_v3(
                    input_ids=input_ids,
                    hidden_states=hidden_states,
                    loss_mask=loss_mask,
                    global_step=global_step,
                )
                # loss_dict 包含: total_loss, distill_loss, cons_loss, distill_acc, cons_acc
            else:
                # 原有base structure训练
                loss, accuracy = flashmtp_model(
                    input_ids=input_ids,
                    hidden_states=hidden_states,
                    loss_mask=loss_mask,
                )

            # 记录metrics (兼容两种模式)
            if args.use_diffusion_training:
                record_metrics_v3(...)
            else:
                record_metrics(...)
```

#### 2.3 新增v3 metrics记录

```python
def record_metrics_v3(
    args,
    loss_dict: dict,  # 包含各分项损失
    global_step: int,
    tracker,
    optimizer,
    mode: str = "train",
):
    """记录v3训练的各项指标"""
    logdict = {
        f"{mode}/total_loss": loss_dict["total_loss"],
        f"{mode}/distill_loss": loss_dict["distill_loss"],
        f"{mode}/cons_loss": loss_dict["cons_loss"],
        f"{mode}/distill_acc": loss_dict["distill_acc"],
        f"{mode}/cons_acc": loss_dict["cons_acc"],
    }
    # ...
```

---

### 3. `specforge/modeling/draft/flashmtp.py` - Draft Model (可选增强)

**当前状态:** 模型结构已经支持双向注意力，不需要大幅修改。

**可选修改:**

- 如果需要支持更复杂的扩散 timestep 嵌入，可以添加 timestep embedding
- 但v3文档说明"每次前向只负责预测当前16个token"，所以当前结构基本满足

---

### 4. `specforge/modeling/target/flashmtp_target_model.py` - Target Model

**修改内容:**

#### 4.1 添加 Teacher Logits 重建函数

```python
def reconstruct_teacher_logits(
    self,
    hidden_states: Tuple[torch.Tensor],  # 各层hidden states
    input_ids: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """
    使用特定层的hidden state重建teacher的logits。

    Args:
        hidden_states: tuple of (batch, seq_len, hidden_size)
        input_ids: 原始input_ids (用于确定layer映射)
        positions: 需要重建logits的位置

    Returns:
        logits: (batch, num_positions, vocab_size)
    """
    # 对于每个位置，使用对应层的hidden state
    # 假设用最后一层或者特定层的hidden state
```

---

### 5. `specforge/core/loss.py` - 损失函数

**新增内容:**

```python
def kl_divergence_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    计算KL散度损失 D_KL(teacher || student)。

    Args:
        student_logits: (batch, seq_len, vocab_size)
        teacher_logits: (batch, seq_len, vocab_size)
        mask: (batch, seq_len) 哪些位置参与计算
        reduction: "mean" or "sum"

    Returns:
        loss: scalar
    """
    # p = softmax(teacher_logits)
    # log_q = log_softmax(student_logits)
    # KL = sum(p * (log p - log q))
```

---

## 关键实现细节

### 1. 状态 y 和 y* 的构建

对于每个block（长度B=16）：

```python
# 在block内随机采样一个位置p (1 <= p < B)
p = random.randint(1, B-1)

# 状态y: 前p个token是真实的，后面是mask
y_tokens = [real_tokens[0:p]] + [MASK] * (B - p)

# 状态y*: 前p+1个token是真实的，后面是mask
y_star_tokens = [real_tokens[0:p+1]] + [MASK] * (B - p - 1)

# U_y: 本次新unmask的位置 = {p}
U_y = [p]

# S_y: 仍然masked的位置 = {p+1, p+2, ..., B-1}
S_y = list(range(p+1, B))
```

### 2. Attention Mask 处理

在v3中，attention mask的构建方式与base structure相同：

- CHS (Context Hidden States) 作为KV
- Block tokens作为Q
- Block内双向可见

### 3. 两阶段训练策略

```python
if global_step < enable_consistency_after_steps:
    # 第一阶段: 只用Distillation Loss
    w_cons = 0.0
else:
    # 第二阶段: 使用两个loss
    w_cons = args.w_cons
```

原因: 学生模型q_phi^-需要产生有意义的预测，否则consistency loss会不稳定。

---

### 问题1: Teacher Logits 的重建方式

**描述:** Distillation Loss需要teacher的logits分布p_i^(T)。当前代码中teacher只返回hidden states。

**修改：**在训练时实时计算。需要确认current code的`target_model.generate_flashmtp_data`是否能支持。

---

### 问题2: Consistency Loss的Stop-Gradient实现

**描述:** Consistency Loss中的q_phi^-需要stop-gradient。

**确认:** PyTorch中可以用`student_logits_y_star.detach()`实现。

---

### 问题3: 是否需要保留Base Structure的兼容

**描述:** 是否需要在代码中保留原有的base structure训练方式？

**建议:** 保留，通过`--use-diffusion-training` flag切换，便于对比实验。

---

### 问题4: Inner Block Size

文档中说明B_in = 1，即每次只unmask 1个token。

**确认:** 这样实现最简单，也符合文档描述。后续如果要支持B_in > 1需要调整。

---

## 实施顺序建议

1. **第一步**: 修改`flashmtp.py`，添加v3前向传播和损失计算
2. **第二步**: 修改`train_flashmtp.py`，添加命令行参数和训练循环
3. **第三步**: 修改`flashmtp_target_model.py`，添加teacher logits重建
4. **第四步**: 添加`loss.py`中的KL散度函数
5. **第五步**: 测试和调试

---

## 文件依赖关系

```
train_flashmtp.py
    ├── OnlineFlashMTPModel (flashmtp.py)
    │   ├── forward_v3()
    │   ├── sample_intermediate_state()
    │   ├── compute_distillation_loss()
    │   └── compute_consistency_loss()
    ├── FlashMTPTargetModel (flashmtp_target_model.py)
    │   └── reconstruct_teacher_logits() [新增]
    └── kl_divergence_loss (loss.py) [新增]
```

---

请确认以上方案后，我将开始具体实现。



下面把这条「固定长度窗口内的局部 \(y,y^\*\) + 段内全双向注意力」说清楚：**记号、采样、\(U_y,S_y\)、损失怎么构造、和「分块」解耦的方式**，以及**能严格说到哪一步、哪一步只能是原理性论证**（避免假证明）。

---

## 1. 你在做什么：对象与「分块」解耦

- **窗口**：在全长序列里选一个连续区间，长度记为 \(B\)（可与现在的 `block_size` 同义）。例如从 anchor \(a\) 起，位置集合  
  \(\mathcal{I}=\{a,a+1,\ldots,a+B-1\}\)。
- **中间状态**：只在这个区间上定义「哪些位置已观测、哪些是 MASK」，得到两个离散状态 \(y,y^\*\)（在 \(\mathcal{I}\) 上与完整序列一致，区间外可视为与训练无关或用 padding/不计算 loss）。
- **与分块的关系**：  
  - **损失里**只依赖 \((y,y^\*)\) 在 \(\mathcal{I}\) 上的模式，**不需要**「多块顺序解码」或「块完成」等故事；\(U_y,S_y\) 都是 \(\mathcal{I}\) 的子集。  
  - **模型里**「分块」若存在，只影响 **注意力图**（以前多块互不看；改成「这一段 draft token 全互看」）。因此：**采样与 loss 构造与是否分块无关；分块只影响注意力归纳偏置和实现效率。**

---

## 2. 采样 \(y\) 与 \(y^\*\)（与 FlashMTP v3 一致的一种具体化）

设 \(\mathcal{I}\) 上对应的真实 token 为 \((x_a,\ldots,x_{a+B-1})\)。采样两个整数：

- \(p \in \{1,\ldots,B-\Delta\}\)（或你们现在的 \(p \in [1, B-\Delta]\)），\(\Delta=\) `inner_block_size` 或随机跳跃长度。
- **状态 \(y\)**：位置 \(a,\ldots,a+p-1\) 用真实 token，\(a+p,\ldots,a+B-1\) 为 MASK。  
- **状态 \(y^\*\)**：位置 \(a,\ldots,a+p+\Delta-1\) 为真实 token，其余为 MASK。

则（与你们代码、CDLM 记号一致）：

- **\(U_y\)**：在 \(\mathcal{I}\) 上「\(y\) 为 MASK 且 \(y^\*\) 非 MASK」的位置，即 **新揭开的 \(\Delta\) 个位置**（若 \(\Delta>1\) 则是一段连续下标）。
- **\(S_y\)**：「\(y\) 与 \(y^\*\) 均为 MASK」的位置，即 **窗口尾部仍未观测** 的 \(B-(p+\Delta)\) 个位置。

这样 **\(U_y \cup S_y \subseteq \mathcal{I}\)**，不会出现「必须依赖后面整块生成」的 \(S_y\)；若 \(p+\Delta=B\)，则 \(S_y=\emptyset\)，一致性项自然为 0（与「\(y^\*\) 填满整个窗口」一致）。

---

## 3. 注意力：从「块隔离」到「整段互看」

- **块隔离（现状）**：不同 block 的 draft 互不可见，块内双向。  
- **整段互看（你的想法）**：**同一窗口 \(\mathcal{I}\)** 内，所有 draft 位置两两可作为 K/V 被看见（再叠上对 CHS/前缀的可见性规则）。  

**含义**：模型在预测任意 MASK 位置时，可以同时利用 **本窗口内所有已填 token 与所有仍为 MASK 的位置上的输入模式**（由 noise embedding 给出），归纳偏置从「局部块」变成「整段联合建模」。这与 **\(U_y,S_y\) 只在 \(\mathcal{I}\) 上定义** 不矛盾：损失仍只在你指定的位置上算 KL。

---

## 4. Loss 构造（与你们 `forward_v3` + CDLM 对齐）

设 \(q_\phi(\cdot \mid y,x)\) 为学生（draft）在条件 \((y,x)\) 下、经共享 `lm_head` 得到的各位置条件分布；Teacher 在位置 \(i\) 的分布为 \(p^{T}_{i}\)（由 target 最后一层 hidden + `lm_head` 得到）。常用写法：

**蒸馏（在 \(U_y\) 上）**  
\[
\mathcal{L}_{\mathrm{distill}}
=
\mathbb{E}\Big[
\frac{1}{|U_y|}\sum_{i\in U_y}
D_{\mathrm{KL}}\!\big(p^{T}_{i}\,\|\,q_\phi(\cdot\mid y,x)_i\big)
\Big].
\]  
**含义**：在「从 \(y\) 到 \(y^\*\) 新露出的观测」上，学生学习 Teacher 的边际预测。

**一致性（在 \(S_y\) 上）**  
\[
\mathcal{L}_{\mathrm{cons}}
=
\mathbb{E}\Big[
\frac{1}{|S_y|}\sum_{i\in S_y}
D_{\mathrm{KL}}\!\big(q_\phi^{-}(\cdot\mid y^\*,x)_i\,\|\,q_\phi(\cdot\mid y,x)_i\big)
\Big],
\]  
其中 \(q_\phi^{-}\) 表示对 \(y^\*\) 分支 **stop-gradient**。  
**含义**：在两边 **都尚未给出真实 token** 的位置上，要求「信息更少」的 \(y\) 与「信息更多」的 \(y^\*\) 给出的预测一致。

**总目标**（与你们一致）  
\[
\mathcal{L} = w_{\mathrm{distill}}\mathcal{L}_{\mathrm{distill}} + w_{\mathrm{cons}}\mathcal{L}_{\mathrm{cons}}
\]  
（若再加标准 CE/DLM 正则，可再加一项，此处略。）

实现上：**掩码只对 \(U_y\) / \(S_y\) 位置累积梯度**；与 CDLM 的差别主要是 **\(y^\*\) 的定义**（你们是「多露 \(\Delta\) token」，CDLM 常取「整块补全」），从而 **\(S_y\) 落在窗口内尾部** 而非「后面所有块」。

---

## 5. 「数学原理」能证明什么、不能证明什么

下面分 **可严格陈述** 与 **原理性目标** 两类，避免伪证明。

### 5.1 可严格陈述（局部、形式化）

1. **KL 的方向**（前向 KL \(D_{\mathrm{KL}}(P\|Q)\)）对固定 \(P\)、变分 \(Q\)：在单纯形上，关于 \(Q\) 的极小化等价于在期望意义下最小化交叉熵 \(\mathbb{E}_{z\sim P}[-\log Q(z)]\)。因此 **蒸馏项**是在 Teacher 固定下，对 \(q_\phi(\cdot|y)\) 做 **加权对数损失** 的一种形式；**单点、固定 Teacher** 时对 \(q\) 的凸性可谈，**对神经网络参数**仍整体非凸。

2. **一致性项 + detach**：只对 \(q_\phi(\cdot|y)\) 反传时，目标是在每个 \(i\in S_y\) 上最小化 \(D_{\mathrm{KL}}(q_{\phi,\mathrm{sg}}(\cdot|y^\*)_i \| q_\phi(\cdot|y)_i)\)，即把 **\(q(\cdot|y)\) 拉向冻结的 \(q(\cdot|y^\*)\)**。这是良定义的凸优化 **在分布空间每个坐标上**；对 \(\phi\) 仍是通常的神经网络非凸问题。

3. **\(U_y\) 与 \(S_y\) 不交**（在你们的构造下通常不交）：两项在不同位置集合上作用，总梯度是各位置贡献之和（再乘权重）。

以上都是 **定义与凸性层面**，不涉及「一定学会任意中间状态一步到达 Teacher」的全局定理。

### 5.2 原理性论证（非唯一最优的严格证明）

1. **轨迹一致性 / 自洽**：若理想生成过程在「信息量单调增加」的路径上，对 **尚未固定的位置** 的条件分布应随信息增加而 **相容**（粗体：同一边缘在不同条件化下的相容性）。一致性损失把这一要求 **松弛为** 有限对 \((y,y^\*)\) 上的 KL，是 **经验近似**，不是某条 ODE 或 Bellman 方程的充要条件。

2. **与 CDLM 的 \(S_y\) 差异**：CDLM 在「当前块已填满」时，\(S_y\) 自然落到后续块；你们在 **固定窗口 + 局部 \(y^\*\)** 下，\(S_y\) 落在 **窗口内尾部**——**同一数学形式**，只是 **\(y^\*\) 相对完整序列的语义**不同，注意力从块隔离改为整段互看，相当于 **改变 \(q_\phi\) 的函数族与归纳偏置**，不改变 KL 公式的合法性。

3. **随机 \(\Delta\)**：若把 \(\Delta\) 随机化，是对 **同一窗口上多组 \((y,y^\*)\) 约束** 的蒙特卡洛覆盖；**没有**一般定理保证「随机 \(\Delta\) 严格优于固定 \(\Delta=1\)」或全局最优。

---

## 6. 小结表

| 维度 | 内容 |
|------|------|
| **思路** | 在固定长度 \(\mathcal{I}\) 上采 \(y,y^\*\)；\(U_y,S_y\subseteq\mathcal{I}\)；与叙事上的「块」无关。 |
| **注意力** | 仅改变 draft 内可见范围：由块隔离 → 段内全双向（+ CHS 规则）。 |
| **Loss** | \(U_y\)：Teacher–Student KL；\(S_y\)：\(q(y^\*)\) 与 \(q(y)\) 的 KL，\(y^\*\) 断梯度。 |
| **可严格说清的** | KL 作为交叉熵极小、单分布意义下的凸性、detach 的优化目标形式。 |
| **不能当作定理的** | 全局最优、与连续一致性模型 ODE 的等价、随机 \(\Delta\) 的排序最优。 |

如果你希望把某一段改成论文里的「命题 + 假设」体例（例如只陈述「在固定 \(\phi\) 的某层输出上 KL 的凸性」），可以指定章节口吻（定理/备注），我可以再压成半页「可放进附录」的版本。