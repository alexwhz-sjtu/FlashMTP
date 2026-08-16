# FlashMTP

> **ICLR submission package:** [`ICLR_SUBMISSION_PACKAGE.md`](ICLR_SUBMISSION_PACKAGE.md)  
> **Benchmark results:** [`benchmark_results/SUMMARY.md`](benchmark_results/SUMMARY.md)  
> **Stochastic verification:** [`docs/STOCHASTIC_VERIFICATION.md`](docs/STOCHASTIC_VERIFICATION.md)  
> **Compile profiling:** [`profile/compile_serial_head_profile.md`](profile/compile_serial_head_profile.md)

## Key results (Qwen3-8B, Model B @ temp=0)

| Dataset | Speedup | Accept length |
|---------|--------:|--------------:|
| Math500 | **3.77×** | 5.06 |
| GSM8K | **3.68×** | 4.94 |
| MBPP | 2.93× | 3.97 |
| Macro mean (8 datasets) | **2.43×** | 3.25 |

With `compile_serial_head`: up to **4.01×** on Math500. At temp=1, rejection sampling beats token-match by **+12%** on GSM8K (3.58× vs 3.20×). See [`benchmark_results/SUMMARY.md`](benchmark_results/SUMMARY.md) for full tables.

```bash
# Reproduce summaries from benchmark logs
python scripts/summarize_benchmarks.py --per-run
```

## Ours core idea

当前版本固定使用 pivot-Q：CHS hidden 作为 context KV，anchor 前 `W-1` 个连续 token embedding 放在 draft Q 前部。

FlashMTP 使用 dense Sliding-CHS 和块内双向注意力，一次 backbone 前向生成整个草稿 block 的 hidden，再由低秩串行 head 恢复块内自回归依赖。

## Base structure

draft Q 固定为 `[embed(a-W+1), ..., embed(a-1), embed(a), MASK...]`，CHS 只作为 context KV。CHS 保留 `CHS_NUM_LAYERS` 个均匀选取并加入层深 embedding 的 target hidden，不包含重复的 pivot token embedding；所有 CHS 槽共享 `anchor-1` 的 RoPE position id。window 和 anchor 不监督，window 通过自身的 K/V 投影参与块内双向注意力。

窗口始终是 dense Sliding-CHS。最后一个 window token 与 CHS hidden 都对应 `anchor-1`；local 模式下第一个有效 window token 从 0 开始编号。

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
H_{\mathrm{all}}\in\mathbb{R}^{B\times A\times (L+1)\times D}.
$$

slot 0 是 pivot token（`anchor-1`），slot 1 是已知 anchor，两者都不参与 token 预测。串行头实际接收：

$$
H=H_{\mathrm{all}}[:,:,2:,:]
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

### 3. RNN head

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

### 4. 两种 logits 输出模式

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

跳过 base LM head，串行 head 直接产生最终 logits。对 `rnn` / `rnn_easy`，hidden 会通过
$W_h: \mathbb{R}^{D}\to\mathbb{R}^{R}$ 残差注入 latent，使 head 能直接利用 backbone 上下文：

$$
\ell^{\mathrm{final}}_k=\ell^{\mathrm{head}}_k.
$$

```text
serial latent            [B, A, K, R]
hidden latent            [B, A, K, R]   # rnn / rnn_easy in direct mode
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

### 5. 参数规模

所有 head 都包含 token embedding 和输出投影，共约：

$$
2VR
$$

个参数。额外参数为：

| Head | 额外参数量 |
| --- | ---: |
| Vanilla | $0$ |
| RNN | $7R^2+3R+DR+2R^2+R$ |
| RNN easy | $7R^2+3R+DR+2R^2$ |

例如 Qwen3-8B 使用 $D=4096$、$V=151936$、$R=256$ 时：

| 模块 | 参数量 |
| --- | ---: |
| Token embedding $V\times R$ | 38,895,616 |
| Output projection $R\rightarrow V$ | 38,895,616 |
| RNN 额外部分 | 1,508,096 |
| RNN easy 额外部分 | 1,507,840 |

### 6. 训练接口

Python 训练入口支持：

```bash
--markov-head-type none|vanilla|rnn|rnn_easy
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

## Citation

```bibtex
@article{flashmtp2026,
  title={FlashMTP: Parallel Block Drafting with Low-Rank Markov Heads
         for Fast and Correct Stochastic Speculative Decoding},
  author={...},
  year={2026}
}
```
