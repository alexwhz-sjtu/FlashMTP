# FlashMTP

## Background
我现在在做一个投机解码的工作。

**传统的投机解码**：草稿模型是自回归的太慢了。然而文字之间语义是连贯的，相关的，我的目标是进行词组的预测。词组之间是强相关的，因此我利用双向注意力，输入多个mask，希望一次预测多个token出来。

**KV cache抛弃**： 对于草稿模型，kvcache是冗余的。大模型***生成的最新的隐藏状态***应该是计算了所有历史信息，理论上是对前文的浓缩。因此我将这个作为上下文中枢（Contextual Pivot）可以只使用这个信息就可以预测后面一块内容。此外，大模型不同深度的层关注前文不同的信息，因此我会纳入所有层的hidden states，进行信息提取。

### 定义
* Contextual Pivot (上下文枢轴)：目标模型最新的融合hidden sstates，它是连接过去（全量历史）与未来（生成块）的支点。
* hs：hidden states的简称。hs可以是任意层的输出hs
* 训练数据：我的训练数据全部是目标大模型生成的响应，这样可以对齐。
* anchor token：训练时随机采样的位置上的token，训练输入为，预测anchortoken的hs（即pivot），拼接ancho token（clean的）在拼接上B-1个noise embedding。

### 核心
我的核心就是去掉kvcache。请不要变动并且相信大模型最新hiddenstates信息足够。并且，对于大模型每层，关注的历史token是不同的，不同层hiddenstates应该已经包含了token的交互.

### 相关工作：扩散投机解码 DFlash
DFlash也利用了大模型的hs，但是他保留了kvcache。它间隔的选取了五层大模型的hs，再沿着特征维度拼接，用fc层降维，他的kvcache就是每个token位置对应的大模型的融合hs。推理时，他把所有位置融合hs注入到每层充当kvcache，拼接B个mask，一次前向预测B个token。

训练时也是一次前向计算loss，越靠前的位置loss权重越大。

## Exp Version

### 1. 目标与动机

exp version 的目标是在“不为草稿模型维护完整历史 KV cache”的前提下，提高块内并行预测的相关性与可训练性。

已有并行预测方法，token之间无法建立语义关联。第一类工作，类似Medusa。把未来多个 token 交给相互独立的预测头，这会带来一个明显问题：后面位置无法利用前面位置的预测信息，越靠后的 token 越难预测。第二类工作，类似DFlash，利用双向注意力一次性预测很长序列。这看似让token之间互相关联，但实际上，在模型内部，token还未形成清晰语义的向量，token之间看见的都是噪声。

Eagle这类方法，token遵循严格因果关系，后面的token可以注意到前面的token。然而，eagle的草稿模型受限于多次自回归，无法并行，从而带来显著的时间开销。

FlashMTP exp 的思路是把未来 token 按语义块组织起来，让同一块内的 token 可以双向交互，同时让不同语义块之间保持从左到右的因果顺序。这样，在块内，由于token之间语义关联强，一次预测难度较低。而且后面的块可以看到前面确定下来的token，形成清晰语义关联。

因此，exp version 不是简单地一次预测 `B-1` 个互相独立的 mask，而是把一个长度为 `B` 的 draft block 拆成多个语义组chunk，按组逐步补全：

chunk大小可选择，总和要等于block_size

比如，block_size=16, chunk：[4,4,4,4]

- chunk 1: anchor+mask0~2
- chunk 2: mask3~6
- chunk 3: mask7~10
- chunk 4: mask11~14

chunk（组）内 token 可以互相可见；组间只允许看见更早的组（clean的内容）。当配置了 `decode_chunk_sizes` 时，**clean query 对 draft 只 attend clean KV**：更早 chunk 的全部 clean、同 chunk 内 clean 全双向；**不看** mask 流 KV。**mask 流在 slot0（`M0:0`）的 KV 不参与 attention**（anchor 仅由 clean 列 `C0:0` 表示）；mask query 仍按 `create_flashmtp_block_mask` 使用本 chunk 内 `M0:1` 起的 mask 列等。


### 2. 上下文表示：Contextual Pivot

训练和推理都使用目标模型的 hidden states 作为历史上下文压缩表示。

对每个 anchor 位置 `a`，草稿模型使用目标模型在 `a-1` 位置的 hidden states 作为 contextual pivot。实现中会等间隔选择若干目标模型层。

然后把这些层在同一位置的 hidden states 沿 feature 维拼接，再经过草稿模型中的 `fc + RMSNorm` 降维到 draft hidden size。

当前 exp version 固定使用 feature concat，CHS 作为草稿 attention 的上下文 KV token，并且使用 `anchor-1` 的 position id 参与 RoPE。

### 3. 草稿输入结构

对每个采样到的 anchor，训练时构造两条 draft stream：

```text
KV layout:
[CHS_i | Clean_i[0:B] | Mask_i[0:B]]

Q layout:
[Clean_i[0:B] | Mask_i[0:B]]
```

其中：

- `CHS_i` 是 anchor `i` 对应的 contextual pivot。
- `Clean_i[k]` 是真实 token `input_ids[anchor+k]` 的 embedding；越界或无效 block 会替换为 mask token。
- `Mask_i[k]`：在 **slot 0** 处嵌入与 **`Clean_i[0]`（anchor）相同的 token**，其余槽位为 mask token embedding；与推理单流「首位为 anchor」对齐。仍只取 **mask 流** 送入 `lm_head` 算 logits（slot 0 若被 loss 权重关掉则与此前一致）。
- 输出 hidden 会 reshape 成 `(batch, anchors, 2, block_size, hidden)`，只取 mask stream 送入 target `lm_head` 计算 logits。

因此，训练不是只输入 `anchor token + B-1 masks`。当前实现输入的是完整 clean stream 加完整 mask stream。clean stream 提供块内可见的真实条件，mask stream 负责学习对应位置的并行预测。

推理时没有 clean/mask 双流。推理主入口 `FlashMTPDraftModel.spec_generate()` 使用一个单 block：

```text
KV layout: [Pivot | anchor, draft slots...]
Q layout:  [anchor, draft slots...]
```

推理按预测组迭代：每次用当前已经填好的 `block_output_ids` 重新 embedding，预测当前组的 token，并把预测结果写回 block，再进入下一组。

### 4. 块因果注意力

核心 mask 由 `flashmtp_slot_group()` 定义语义组：

```text
slot 0 -> group 0      anchor
slot 1 -> group 1      first token
slot 2-3 -> group 2    two-token group
slot >=4 -> group 3 + floor((slot-4)/4)
```

训练时的 visibility 规则：

- 每个 sampled block 只能看自己的 CHS，不能看其他 anchor 的 CHS。
- 不同 sampled training blocks 之间完全不可见。
- clean query 只能看同一 block 内更早或当前语义组的 clean tokens。
- mask query 可以看更早语义组的 clean tokens，以及当前语义组内的 mask tokens。
- 当前语义组内的 mask tokens 彼此双向可见。
- padding 出来的无效 anchor 由 `block_keep_mask=False` 屏蔽，不参与 attention 和 loss。

推理时的 visibility 规则更直接：

- pivot 对所有 draft slots 可见。
- draft slot 只能看见更早或当前语义组的 draft slots。
- 当前语义组内 token 双向可见。
- 后续语义组不可见。

这保证了训练和推理都遵循同一组语义块顺序：`anchor -> 1 -> 2 -> 4 -> 4 -> ...`。

### 5. 训练目标与 anchor 采样

随机采样anchor token，预测后面B个token。loss计算时anchor token不计入。并且投机解码中，必须保证靠前的token正确，因此，再token-level基础上引入位置降权，越靠后的token loss权重降低。

指标包括 token accuracy 和 `prefix_accuracy`。`prefix_accuracy` 衡量一个 block 从 slot 1 开始连续预测正确的前缀比例，更接近 speculative decoding 中 acceptance length 的训练代理指标。

### 7. 推理流程

推理仍使用目标模型做 prefill 和 verification：

1. 目标模型对 prompt prefill，产生第一个 token 和最新 hidden states。
2. 从目标 hidden states 中抽取 contextual pivot。
3. 草稿模型按预测组 `[1] -> [2,3] -> [4..]` 依次补全一个 block。
4. 目标模型一次验证整个 block。
5. 根据 target posterior 计算 acceptance length。
6. 接受前缀写入输出，未接受处写入 target posterior token。
7. 裁剪 target KV cache 到已接受长度，并用目标模型在 pivot 位置的新 hidden states 更新 contextual pivot。

草稿模型不维护完整历史 KV cache；历史信息仍然通过目标模型的最新 fused hidden states 注入。

### 8. 当前实现边界

- 当前 exp version 的 hidden states 融合方式固定为 feature concat + FC 降维。
- 当前 contextual pivot 是单位置 `anchor-1` / latest accepted pivot，不是多位置 sink/window。
- 当前训练使用 clean/mask 双流；推理使用按语义组逐步填充的单流 block。
- `anchor_chunk_size` 只改变显存与 backward 分块方式，不改变训练目标。

### Extra. Anchor Chunking

当 `--anchor-chunk-size > 0` 且小于 `num_anchors` 时，训练不会一次处理所有 anchors，而是：

1. 本卡先采样完整 anchor set。
2. 通过 `align_anchor_count_across_ranks()` 把不同 rank 的 anchor 数 padding 到相同长度，保证所有 rank 有相同 chunk 数。
3. 预先计算本地所有 anchors 的 `total_valid_token_count`。
4. 按 `anchor_chunk_size` 分块 forward/backward。
5. 每个 chunk 的 loss numerator 除以全 anchor 的 denominator 后 backward。

这样多次 chunk backward 的梯度等价于一次处理所有 anchors 的 token-weighted loss，但显存峰值更低。注意这里不会在不同 GPU 之间共享 anchors；跨 rank 只做长度对齐和日志 all-reduce。
