## 项目背景（仓库与问题设定）

本仓库在 [SpecForge](https://github.com/sgl-project/specforge) 训练框架上实现了 **FlashMTP**：一种面向投机解码的**草稿模型**方案。设计目标包括：（1）避免传统草稿的**逐步自回归**，改为在固定块长内用**双向注意力**一次性拟合多个未来 token；（2）草稿侧**不把整段历史的 KV 当作显式依赖**，而是主要依赖目标（大）模型在某一 **anchor 位置** 上产出的**多层 hidden states（CHS, Context Hidden States）** 作为压缩上下文，以降低带宽、利于高并发与纯算力瓶颈场景；（3）训练与推理时，目标模型仍负责提供真实语义与最终验证，草稿负责提出候选块。

与仓库中 **DFlash** 等基线相比：DFlash 典型做法是对目标侧**逐位置**融合/使用更丰富的 hidden 信息；FlashMTP 则明确将「仅 anchor 处多层 HS + 块内 MASK 噪声嵌入」作为草稿条件，更激进地收缩上下文形式，因此若单点 HS 对「被大模型低注意的前文 token」编码不足，接受率会承压——这与下文实验现象一致。

## FlashMTP 代码基本结构（与目录对应）

| 模块 | 路径 | 职责摘要 |
| --- | --- | --- |
| 草稿网络 | `specforge/modeling/draft/flashmtp.py` | `FlashMTPDraftModel`：基于 Qwen3 的堆叠解码层；**非因果**自注意力；`Qwen3FlashMTPAttention` 将 **K/V** 设为 `concat(每块 1 个 CHS 条件, 块内 draft token)`。CHS 由 **`CHSQueryFusion`** 在 **深度维**对 `embed+各解码层`（anchor−1 处 gather 的栈）做非因果自注意力，RoPE 为 `0…S−1`，**取最后一槽输出** 作为单块条件（顶层 HS 的 Q attend 全层；曾用可学习 query，已改为此形式）。`extract_stacked_chs` / `spec_generate` 从 `hidden_states` 取层。 |
| 训练封装 | `specforge/core/flashmtp.py` | `stack_hidden_states_for_positions`：按 anchor 取 **位置 `anchor-1`** 的各层（与 embed）hidden，**stack 为 (B, N, S, H)** 供草稿 `CHSQueryFusion`；`create_flashmtp_block_mask`：**Flex** 块掩码——块 *i* 仅见本块 CHS 与同块内双向 draft，块间互不可见；`OnlineFlashMTPModel`：并行多 anchor、块首为真 token/其余 MASK、**块内 CE**（可配 `loss_decay_gamma`）。 |
| 目标模型后端 | `specforge/modeling/target/flashmtp_target_model.py` | `HFFlashMTPTargetModel` / `SGLangFlashMTPTargetModel`：前向得到各层 hidden，供训练时构造 CHS。 |
| 训练入口 | `scripts/train_flashmtp.py` | 装配 target、draft、`OnlineFlashMTPModel`；`--chs-concat-mode {feature,seq}` 写入 `flashmtp_config`（**`seq` 未实现**，运行时会 `NotImplementedError`）；`--chs-fusion-layer-idx` 等；**须用 `torchrun` 启动** 以提供分布式环境变量。 |

**数据与注意力语义（与实现对齐）**：CHS 在训练里对应「预测块起点 anchor 的上一位置」的层表示（`prepare_target_hidden` 中 `context_positions = anchor - 1`）；草稿 queries 在块内全位置与自身块双向混合，**不能**跨块 attend 到其他块的 CHS 或 token。

**推理备注**：`spec_generate` 中目标模型使用 KV cache 做自回归；草稿侧对块仍作前向，且实现里可对 draft 使用 `DynamicCache` 做步进优化——与「不缓存整段 target 历史到草稿」并不矛盾：草稿的「上下文张量」仍是每步更新后的 `target_hidden` 切片，而非在草稿内维护对无限长 target 的 KV。

**实现补充（v1.4）**：`FlashMTPDraftModel` 中 RoPE 已改为对 **`concat(CHS, draft)` 全长** 计算（CHS 槽位用各块 **首 draft 位置 `−1`** 与 `prepare_target_hidden` 语义对齐），避免仅对 draft 长度假算 `cos/sin` 与 `K` 维数不一致。

---

### 方法论（在做什么、凭什么）

1. **条件来源**：不引入长序列 target-KV，仅用「块起点 anchor 上一位置」的 **embed + 各层 hidden** 堆成深度序列，作为该块的**压缩语义条件**（CHS 管线）。
2. **跨层利用**：在深度维用**非因果**自注意力做 **HS-aware 融合**，以**最顶层**槽位为 readout，使「以顶层语义为锚、向浅层/嵌入层选择性聚合」有显式机制；与「单层线性压特征」或「纯可学习 query」可对照实验。
3. **块内生成**：**MASK + 块内双向** draft 层、Flex block mask 保证块与块间不串信息；监督为块内带 **位置衰减** 的 CE，与「投机解码更关注块首/早期接受」的动机一致。
4. **验证闭环**：大模型对草稿候选做自回归验证；实现上目标侧可 cache、草稿条件每步用最新 **stacked CHS** 更新，与「不缓存整段到草稿内」的设定一致。

---

我现在在做一个投机解码的工作。我认为传统的投机解码，草稿模型自回归太慢了。文字之间语义是连贯的、相关的，因此利用双向注意力在块内做少次前向以生成连续候选；并通过「仅单位置多层 HS + 轻量条件」在工程上收缩草稿侧对长历史的显式依赖。

进一步，对草稿网络而言，不维护整段 **target** 的 KV，有利于带宽与并发布局；大模型在 anchor−1 处的各层表示仍参与残差/注意力堆叠，**方法论上**把「跨 token 的编码」压进有限深度序列里再融合，接受率上仍可能弱于**逐位置**用 HS 的强基线（如 DFlash），属于信息—开销权衡。

### 当前进展

- **代码（v1.4）**  
  - **CHS 融合**：`CHSQueryFusion` 在深度维对 `embed+各层` 做非因果自注意力，**最后一槽（顶层 HS）** 为条件向量；`train_flashmtp.py` 提供 `--chs-concat-mode`（`feature` 为当前实现路径；`seq` 预留且会显式报未实现）。  
  - **RoPE / 注意力**：`K = concat(CHS, draft)` 时，RoPE 在 **`[ctx_pos; draft pos]`** 上整体计算，与块掩码、训练 anchor 一致。  
  - **工程**：`run_training_flashmtp.sh` 用 **`torchrun` 全进程**（含单卡）以满足 `init_distributed`；W&B 在指定 `run_id` 时 **`resume=allow`**，避免新 run 误用 `must` 报错。多卡训练需 **可见多 GPU**（勿将 8 进程全压在 `CUDA_VISIBLE_DEVICES=0` 上，易 OOM）。

- **实验与结论（历史）**  
  在同等数据与配置对比下，早期「单层 CHS 条件」的接收长度仍低于 DFlash 一类逐位置强条件基线；后续方向仍在「更好利用层栈」与「训练 recipe（mask 课程/扩散等）」上迭代。

| Model | Ours (早期基线) | DFlash (参考) |
| --- | --- | --- |
| 接收长度 (16) | 2.7～3.5 | 3.6～4.8 |

**思考与反思**：大模型在 **anchor−1** 处对长上下文的**注意力分配**未必覆盖后续块所需的全部细粒度信息；在坚持「不拉长 target KV、只信多层 HS 栈」的前提下，**融合方式、训练目标与块内可学习先验**仍是可调旋钮。

**开放问题（仍可优化处）**：`target_layer_ids` 子集或重加权、块内/扩散式训练、轻量可学习 per-slot 偏置、与 DFlash 的公平对齐（同数据、同步数、同块长）等，见下节「结合代码的改进反思」。

我现在在想如何改进我的模型原理或者架构。对于草稿模型设计，**可动变量**包括：层融合结构、块长与 anchor 数、损失加权、是否引入与「块未观测位置」兼容的辅助监督等。

### 改进方向和已有尝试
1. hs（hiddenstates）的融合方式改变：我将大模型每层hs沿着序列维度拼接，并且加入了层嵌入提供显示层深度信息。训练保持一次forward unmask全部，但结果上看收益很小

2. 从loss设计入手：从单纯ce，变为ce，kl和mse损失混合，训练保持一次forward unmask全部，效果甚至变差。

3. 从训练方法入手：
* 问题：训练时一次解码可能无法使模型学习到正确的token间就信息交互，虽然是双向注意力，但是在大部分层后续的token都是不确定的，并且最终结果上看块后面部分的token经常是重复/无意义的。
* 尝试：使用离散扩散语言模型训练方法，随机采集噪声程度，预测结果。保留权重衰减。

4. 能否包含更多未来信息。就像我的反思中提到，我能否加入一些轻量可学习模块，或者额外输入比如register，来在大模型生成next hs时，同时学习到包含未来B长度所需要的信息。

我的核心就是去掉kvcache。请不要变动并且相信大模型最新hiddenstates信息足够。并且，对于大模型每层，关注的历史token是不同的，不同层hiddenstates应该已经包含了token的交互，只是我没有利用好。

### 结合代码与上述进展的改进反思（面向实现）

1. **CHS 与 `prepare_target_hidden` 的瓶颈**  
   当前 CHS 严格来自 **单位置 `anchor-1`** 的多层 gather。若接受率被「低注意力的前文 token」拖累，在**不改**「只信最新 HS」的前提下，可优先在**利用方式**上动手：`target_layer_ids` 子集/重加权、层间 gating 或低秩混合（在 `fc` 前后）、或在 `Qwen3FlashMTPAttention` 里对 **k_ctx / v_ctx** 与 **k_noise / v_noise** 做可学习尺度（让草稿网络显式决定信上下文 vs 信块内自洽）。这些都不等同于重新引入整段 target KV。

2. **`chs_concat_mode=seq` vs `feature` 与 RoPE**  
   当前主分支为 **`feature` 式层栈 + `CHSQueryFusion` 单条件 token**；`seq` 若实现需多块 CHS 与 `create_flashmtp_block_mask` 中 `chs_len_per_block` 等一并改。草稿栈内 `Qwen3FlashMTPAttention` 的 **K/V 全长 RoPE** 已在 v1.4 与 **ctx∥draft 拼接** 对齐。可系统做 **同数据、同算力** 的消融，并检查 **anchor 采样** 与 `loss_mask` 的有效交叠。

3. **训练目标 vs 块内靠后位置崩溃**  
   `OnlineFlashMTPModel` 里块内除首位外全为 MASK，再双向一次预测全位置；与现象「块后部重复/无意义」一致时，可尝试：**课程式**提高 MASK 比例或块内**逐步 unmask**、与已有「离散扩散」思路对齐；或在保持 CE 主损失下，**只对前 k 个位置**加权验证接受率（与 `loss_decay_gamma` 同向但目标更贴近投机解码的 early-accept）。

4. **损失与基线**  
   CE+KL+MSE 变差，说明目标模型分布与点估计 HS 监督可能冲突；若继续加辅助项，宜绑定 **同分布目标**（例如仅对 logits 的蒸馏且冻结 target head 温度），避免与块内双向带来的噪声梯度打架。

## Attempt 1（层栈 / HS-aware 融合）

**动机**：沿特征维 `concat+fc` 是**与样本无关的线性层间混合**；而同一 `anchor−1` 上各层对「要预测的块」贡献不同，需要 **input-dependent** 的跨层重加权。  
**做法演进**：先尝试在深度序列末增加 **可学习 query**，做非因果自注意力后取读出口；**当前实现**改为 **不含额外参数槽**：序列仅为 `[embed, h_0,…,h_{L-1}]`，**最后一槽即顶层 target HS**，其注意力输出作为单块 **CHS 条件**（仍以「顶层的 Q 对全深 attend」实现选择性，与「显式可学习 query」非数学等价、归纳偏置不同）。**方法论要点**：不增加 target KV、只在 CHS 头里调节「怎样读层栈」。

## Attempt 2
可否使用连续扩散模型的建模方式，逐步从hs中解析信息。然后通过成熟的蒸馏方式，进行少步蒸馏