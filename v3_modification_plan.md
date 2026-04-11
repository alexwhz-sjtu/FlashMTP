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