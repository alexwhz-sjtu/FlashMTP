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

### exp version

背景与动机 (Background & Motivation)

- 克服独立性瓶颈：现有的并行预测框架（如 MEDUSA）通过多个独立的预测头同时生成 Token，但各预测头之间缺乏语义关联，后面的头无法看见前面的头，导致位置越靠后的 Token 预测准确率指数级下降。  

- 语义块并行预测：在具有相似语义的块内，预测难度较低，可以利用双向注意力进行一次前向，并行预测。

- 增强块内语义连贯性：通过引入块状因果注意力，使模型能够在一个语义块内（如一个短语）进行双向建模，同时确保块与块之间维持严格的因果依赖。

模型架构

使用n层transformer（n：1~3），他的输入为pivot hs，anchortoken，B-1个mask。

之后，我进行多部迭代，每步迭代并行预测出一些token块（块符合从左到右顺序）。由于achortoken之后的token比较难而且很关键，因此我第一步只预测他一个。之后，第二部，我预测后面两个，在之后，每步我预测4个。每步预测出来的token都要解码，然后再当作输入拼接到已知条件后。其中，pivot，anchor token各自看作长度为1的块，即他们不能看到后面的token。之后每步预测的token当作一个块，他们可以看见前面的token，块内互相可见，但不能看到后面的块。

在训练和推理时，按照上面的块因果，构造mask进行块内并行训练。

hidden states融合方式保持现有flashmtp方法。

