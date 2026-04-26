## 项目背景（仓库与问题设定）

本仓库在 [SpecForge](https://github.com/sgl-project/specforge) 训练框架上实现了 **FlashMTP**：一种面向投机解码的**草稿模型**方案。设计目标包括：（1）避免传统草稿的**逐步自回归**，改为在固定块长内用**双向注意力**一次性拟合多个未来 token；（2）草稿侧**不把整段历史的 KV 当作显式依赖**，而是主要依赖目标（大）模型在某一 **anchor 位置** 上产出的**多层 hidden states（CHS, Context Hidden States）** 作为压缩上下文，以降低带宽、利于高并发与纯算力瓶颈场景；（3）训练与推理时，目标模型仍负责提供真实语义与最终验证，草稿负责提出候选块。

与仓库中 **DFlash** 等基线相比：DFlash 典型做法是对目标侧**逐位置**融合/使用更丰富的 hidden 信息；FlashMTP 则明确将「仅 anchor 处多层 HS + 块内 MASK 噪声嵌入」作为草稿条件，更激进地收缩上下文形式，因此若单点 HS 对「被大模型低注意的前文 token」编码不足，接受率会承压——这与下文实验现象一致。

## FlashMTP 代码基本结构（与目录对应）

| 模块 | 路径 | 职责摘要 |
| --- | --- | --- |
| 草稿网络 | `specforge/modeling/draft/flashmtp.py` | `FlashMTPDraftModel`：基于 Qwen3 的堆叠解码层；**非因果**自注意力；`Qwen3FlashMTPAttention` 将 **K/V** 设为 `concat(目标 CHS 投影, 草稿 token 嵌入)`，Q 来自草稿隐状态；支持 `chs_concat_mode`：`seq`（多层沿序列维拼接，CHS 占 L 个 token）与 `feature`（多层沿特征维拼接后经 `fc` 压到 `hidden_size`）；`extract_context_feature` / `spec_generate` 中从目标 `hidden_states` 按 `target_layer_ids` 取层。 |
| 训练封装 | `specforge/core/flashmtp.py` | `prepare_target_hidden`：按 anchor 取 **位置 `anchor-1`** 的各层 hidden，再按 `seq`/`feature` 拼成 CHS；`create_flashmtp_block_mask`：**Flex Attention** 块掩码——块 *i* 仅见自己的 CHS *i* 与同块内双向 draft token，块间互不可见；`OnlineFlashMTPModel`：并行多 anchor、噪声输入（块首为真实 token、其余为 MASK）、**块内 CE**（可配 `loss_decay_gamma` 使越靠前位置权重越大）。 |
| 目标模型后端 | `specforge/modeling/target/flashmtp_target_model.py` | `HFFlashMTPTargetModel` / `SGLangFlashMTPTargetModel`：前向得到各层 hidden，供训练时构造 CHS。 |
| 训练入口 | `scripts/train_flashmtp.py` | 装配 target、draft、`OnlineFlashMTPModel`；`num_target_layers` 与 `flashmtp_config`（含 `chs_concat_mode`、`mask_token_id`、`target_layer_ids`）等超参。 |

**数据与注意力语义（与实现对齐）**：CHS 在训练里对应「预测块起点 anchor 的上一位置」的层表示（`prepare_target_hidden` 中 `context_positions = anchor - 1`）；草稿 queries 在块内全位置与自身块双向混合，**不能**跨块 attend 到其他块的 CHS 或 token。

**推理备注**：`spec_generate` 中目标模型使用 KV cache 做自回归；草稿侧对块仍作前向，且实现里可对 draft 使用 `DynamicCache` 做步进优化——与「不缓存整段 target 历史到草稿」并不矛盾：草稿的「上下文张量」仍是每步更新后的 `target_hidden` 切片，而非在草稿内维护对无限长 target 的 KV。

---

我现在在做一个投机解码的工作。我认为传统的投机解码，草稿模型自回归太慢了。文字之间语义是连贯的，相关的，因此我可以利用双向注意力/扩散原理来进行少次前向就生成一段长度的候选token。

进一步，我认为对于草稿模型，kvcache是冗余的，大模型生成的最新的隐藏状态应该是计算了所有历史信息，因此我应该可以只使用这个信息就可以预测后面一块内容。kvcache的消除对于运侧并发场景和infra很友好，草稿模型没有kvcache，gpu带宽不需要要求高，而且变成纯compute-bound。

### 当前进展
我现在实现了基础结构，把最新的大模型每层hiddenstates拼接，投影成一个embedding，在拼接B个mask，一次forward，双向注意力，直接生成cleantoken，大模型再去验证。训练的时候也是这样训练，loss是每个位置的交叉熵，并且越前面的token loss权重越大。可是基础结构效果不如保持全量的kvcache（之前每个位置的融合hs）。

相同数据量下，对比结果如下：

Model	Ours now	DFlash now
接收长度(16)	2.7~3.5	3.6～4.8

**思考与反思**：大模型生成最新hs时，对于前文也是有注意力分配的，对于不关注的token，包含的语义信息就少了，但在预测后面一整块内容时，也许就要用到之前不关注的token。

我现在在想如何改进我的模型原理或者架构，请帮我分析一下。对于草稿模型设计，那些是变量我可以优化的.

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
   `seq` 模式下对 CHS 的 RoPE 与 `feature` 不同（见 `flashmtp.py` 中 `position_embeddings` 与 `Qwen3FlashMTPAttention` 分支）。若两种模式收益都有限，可系统做一次 **同数据、同算力** 的对比，并检查 **anchor 采样**（`num_anchors`、与 `loss_mask` 的交集）是否让有效监督偏少。

3. **训练目标 vs 块内靠后位置崩溃**  
   `OnlineFlashMTPModel` 里块内除首位外全为 MASK，再双向一次预测全位置；与现象「块后部重复/无意义」一致时，可尝试：**课程式**提高 MASK 比例或块内**逐步 unmask**、与已有「离散扩散」思路对齐；或在保持 CE 主损失下，**只对前 k 个位置**加权验证接受率（与 `loss_decay_gamma` 同向但目标更贴近投机解码的 early-accept）。

4. **损失与基线**  
   CE+KL+MSE 变差，说明目标模型分布与点估计 HS 监督可能冲突；若继续加辅助项，宜绑定 **同分布目标**（例如仅对 logits 的蒸馏且冻结 target head 温度），避免与块内双向带来的噪声梯度打架。

5. **「更多未来信息」与不变约束**  
   Register / 可学习 query 不必然要求大模型改结构：可在 `noise_embedding` 侧加入 **B 个可学习偏置** 或 **per-block 位置编码**，仅增加草稿参数量，表示「在固定 HS 条件下对尚未观测的 B−1 个槽位的先验」——仍不把长历史 KV 接回草稿。

6. **与 DFlash 的公平对比**  
   表观接收长度差可能来自 **CHS 信息量** 与 **训练 recipe** 两方面；建议在文档中固定 **块长、步数、数据管线**，再区分「架构」与「数据/超参」因素，避免单次实验掩盖可改进点。


## Attempt 1
我现在认为沿着feature维度拼接，再通过fc降维，本质上是对每层hs等权相加。但是对于一个位置的hs，不同层的信息融合程度是不一样的。因此我可以使用attention模块来进行hs-aware的信息提取。具体来讲，我将每层hs拼接上一个可学习query，使用这个query进行选择性信息提取。再将这个query作为条件输入小模型。