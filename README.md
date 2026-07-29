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

加入低秩串行头。在保留 FlashMTP backbone 一次并行计算整个 block 的基础上，
串行头通过前一个真实/采样 token 恢复块内自回归依赖。

### 1. 记号与整体数据流

以下使用：

| 符号 | 含义 |
| --- | --- |
| $B$ | batch size |
| $A$ | 每条序列采样的 anchor/block 数 |
| $L$ | `block_size`，包含 slot 0 的已知 anchor |
| $K=L-1$ | 真正需要预测的 draft token 数 |
| $D$ | FlashMTP backbone hidden size |
| $R$ | `markov_rank`，串行头中间维度 |
| $V$ | vocabulary size |

训练时 FlashMTP backbone 的完整输出为：

$$
H_{\mathrm{all}}\in\mathbb{R}^{B\times A\times L\times D}.
$$

slot 0 是已知 anchor，不参与 token 预测。串行头实际接收：

$$
H=H_{\mathrm{all}}[:,:,1:,:]
\in\mathbb{R}^{B\times A\times K\times D}.
$$

Teacher forcing 使用的前驱 token 和监督目标分别是：

$$
X_{\mathrm{prev}}
=[x_{\mathrm{anchor}},x_1,\ldots,x_{K-1}]
\in\mathbb{N}^{B\times A\times K},
$$

$$
Y=[x_1,x_2,\ldots,x_K]
\in\mathbb{N}^{B\times A\times K}.
$$

推理时没有 anchor 维度 $A$，对应张量为
$H\in\mathbb{R}^{B\times K\times D}$。第 $k$ 步生成的 token 会作为
第 $k+1$ 步串行头的输入。

三种 head 共享两个低秩映射：

1. 前驱 token embedding：

$$
E:\mathbb{N}\rightarrow\mathbb{R}^{R},
   \qquad E\in\mathbb{R}^{V\times R}.
$$

2. 词表投影：

$$
W_{\mathrm{out}}:\mathbb{R}^{R}\rightarrow\mathbb{R}^{V},
   \qquad W_{\mathrm{out}}\in\mathbb{R}^{V\times R}.
$$

`markov_rank` 就是这里的 $R$，可以通过训练脚本自由指定。

### 2. Vanilla head

Vanilla head 的串行分支只读取前一个 token ID，不读取当前位置 hidden state：

$$
e_{k-1}=E[x_{k-1}]\in\mathbb{R}^{R},
$$

$$
\ell^{\mathrm{head}}_k
=W_{\mathrm{out}}e_{k-1}
\in\mathbb{R}^{V}.
$$

训练阶段的维度变化为：

```text
previous token IDs       [B, A, K]
    ↓ Embedding(V, R)
previous token features  [B, A, K, R]
    ↓ Linear(R, V)
head logits              [B, A, K, V]
```

Vanilla 没有额外的 hidden/state 中间层。Teacher forcing 时所有前驱 token
已知，因此训练可以沿 $K$ 个位置并行；推理时必须等待上一步采样结果。

需要注意：在 `direct` 模式下，Vanilla 最终分布只依赖前一个 token，
等价于一个低秩 bigram head，不再使用当前位置的 backbone hidden state。

### 3. Gated head

Gated head 同时读取前一个 token embedding 和当前位置 backbone hidden：

$$
e_{k-1}=E[x_{k-1}]\in\mathbb{R}^{R},
\qquad h_k\in\mathbb{R}^{D}.
$$

首先拼接为：

$$
z_k=[h_k;e_{k-1}]
\in\mathbb{R}^{D+R}.
$$

然后产生 $R$ 维 gate：

$$
g_k=\sigma(W_g z_k+b_g)
\in\mathbb{R}^{R},
$$

$$
u_k=g_k\odot e_{k-1}
\in\mathbb{R}^{R}.
$$

在 `direct` 模式下，hidden 还会通过 $W_h: \mathbb{R}^{D}\to\mathbb{R}^{R}$ 直接进入 latent：

$$
h'_k=W_h h_k\in\mathbb{R}^{R},
\qquad
u_k=g_k\odot e_{k-1}+h'_k\in\mathbb{R}^{R}.
$$

$$
\ell^{\mathrm{head}}_k
=W_{\mathrm{out}}u_k
\in\mathbb{R}^{V}.
$$

维度变化为：

```text
previous token IDs       [B, A, K]
    ↓ Embedding(V, R)
token features           [B, A, K, R]

backbone hidden          [B, A, K, D]
    ├─ concat with token features
    │     ↓
    │  gate input          [B, A, K, D + R]
    │     ↓ Linear(D + R, R) + sigmoid
    │  gate                [B, A, K, R]
    │
    └─ Linear(D, R)        # 仅 direct 模式
       hidden latent       [B, A, K, R]

gate * token features + hidden latent
head latent              [B, A, K, R]
    ↓ Linear(R, V)
head logits              [B, A, K, V]
```

Gated head 没有跨位置 recurrent state。Teacher forcing 时所有位置也可以并行。

### 4. RNN head

RNN head 维护一个仅由**前序 token 链**驱动的跨位置 $R$ 维状态：

$$
s_{k-1}\in\mathbb{R}^{R},
\qquad s_0=0.
$$

状态更新只看前一个 token，不看当前位置 hidden：

$$
z^{\mathrm{mem}}_k=[s_{k-1};E[x_{k-1}]]
\in\mathbb{R}^{2R},
$$

$$
[g_k,\widetilde{c}_k]
=W_{\mathrm{state}}z^{\mathrm{mem}}_k
\in\mathbb{R}^{2R},
$$

$$
s_k=g_k\odot s_{k-1}
 +(1-g_k)\odot\tanh(\widetilde{c}_k)
\in\mathbb{R}^{R}.
$$

由状态产生串行低秩表示：

$$
u^{\mathrm{serial}}_k=\tanh(W_{\mathrm{out}}s_k)\in\mathbb{R}^{R}.
$$

在 `direct` 模式下，再把当前并行 hidden 投影后门控融合进输出 latent（**不进 state**）：

$$
h'_k=W_h h_k\in\mathbb{R}^{R},
\qquad
\alpha_k=\sigma(W_{\mathrm{fuse}}[u^{\mathrm{serial}}_k;h'_k]),
$$

$$
u_k=\alpha_k\odot u^{\mathrm{serial}}_k
 +(1-\alpha_k)\odot h'_k.
$$

$$
\ell^{\mathrm{head}}_k
=W_{\mathrm{out}}u_k
\in\mathbb{R}^{V}.
$$

单个位置的维度变化为：

```text
previous state           [B, A, R]
previous token feature   [B, A, R]
    ↓ concatenate
memory input             [B, A, 2R]
    ↓ Linear(2R, 2R)
gate / candidate         2 × [B, A, R]
    ↓ state update
new state                [B, A, R]
    ↓ Linear(R, R)
serial latent            [B, A, R]
current hidden           [B, A, D]        # 仅 direct，且只用于输出
    ↓ Linear(D, R)
hidden latent            [B, A, R]
    ↓ gate 融合
head latent              [B, A, R]
    ↓ Linear(R, V)
head logits              [B, A, V]
```

RNN 训练仍使用真实前驱 token 做 teacher forcing，但由于
$s_k$ 依赖 $s_{k-1}$，必须沿 block 的 $K$ 个位置串行展开。
不同 batch、不同 anchor/block 之间的 state 相互独立，并在每个 block
开始时清零。

### 5. 两种 logits 输出模式

并行 backbone hidden 经过共享 target LM head 或可训练 draft LM head，
得到基础 logits：

$$
\ell^{\mathrm{base}}
\in\mathbb{R}^{B\times A\times K\times V}.
$$

支持两种最终输出方式：

#### `additive`

串行 head 生成 logit bias：

$$
\ell^{\mathrm{final}}_k
=\ell^{\mathrm{base}}_k+\ell^{\mathrm{head}}_k.
$$

维度为：

```text
base logits  [B, A, K, V]
head logits  [B, A, K, V]
    ↓ element-wise addition
final logits [B, A, K, V]
```

#### `direct`

跳过 base LM head，串行 head 直接产生最终 logits。对 `gated` / `rnn`，hidden 会通过
$W_h: \mathbb{R}^{D}\to\mathbb{R}^{R}$ 残差注入 latent，使 head 能直接利用 backbone 上下文：

$$
\ell^{\mathrm{final}}_k=\ell^{\mathrm{head}}_k.
$$

```text
serial latent            [B, A, K, R]
hidden latent            [B, A, K, R]   # gated / rnn only
    ↓ add
head latent              [B, A, K, R]
    ↓ Linear(R, V)
final logits             [B, A, K, V]
```

`direct` 跳过 base LM head，串行 head 直接产生最终 logits。

实际训练中不会长期物化完整的
$[B,A,K,V]$ 张量。模型先保存较小的
$[B,A,K,R]$ head latent，再按 `ce_chunk_size` 分块投影到词表并计算 CE，
从而控制 full-vocabulary logits 的峰值显存。

### 6. 参数规模

所有 head 都包含 token embedding 和输出投影，共约：

$$
2VR
$$

个参数。额外参数为：

| Head | 额外参数量 |
| --- | ---: |
| Vanilla | $0$ |
| Gated | $R(D+R)+R+DR+2R^2+R$ |
| RNN | $7R^2+3R+DR+2R^2+R$ |
| MLP | $2DR+3R^2+7R$ |

例如 Qwen3-8B 使用 $D=4096$、$V=151936$、$R=256$ 时：

| 模块 | 参数量 |
| --- | ---: |
| Token embedding $V\times R$ | 38,895,616 |
| Output projection $R\rightarrow V$ | 38,895,616 |
| Gated 额外部分 | 2,294,272 |
| RNN 额外部分 | 1,508,096 |
| MLP 额外部分 | 2,295,552 |

### 7. 训练接口

Python 训练入口支持：

```bash
--markov-head-type none|vanilla|gated|rnn|mlp
--markov-output-mode additive|direct
--markov-rank 256
```

Shell 启动脚本使用：

```bash
MARKOV_HEAD_TYPE=rnn \
MARKOV_OUTPUT_MODE=additive \
MARKOV_RANK=256 \
bash scripts/run_training_flashmtp.sh --dt h100
```

关闭串行 head、保持原始 FlashMTP 行为：

```bash
MARKOV_HEAD_TYPE=none \
bash scripts/run_training_flashmtp.sh --dt h100
```

> \# git clone the source code
> 
> git clone https://github.com/sgl-project/SpecForge.git
> 
> cd SpecForge
> 
> \# create a new virtual environment
> 
> uv venv -p 3.11
> 
> source .venv/bin/activate
> 
> \# install specforge
> 
> uv pip install -v -e . --prerelease=allow
>
> uv pip install datasets==4.8.3 pyarrow==23.0.1
