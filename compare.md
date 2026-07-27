# DSpark RNN Head vs FlashMTP RNN Head 结构对比

本文对比 **DeepSpec / DSpark**（`/data/DeepSpec/deepspec/modeling/dspark/markov_head.py` 中的 `RNNHead`）与 **FlashMTP_v2**（`specforge/modeling/draft/flashmtp_markov_head.py` 中的 `FlashMTPMarkovHead`，`head_type="rnn"`）的 RNN 串行头。

## 符号约定

| 符号 | 含义 | 典型值（Qwen3-8B） |
|------|------|-------------------|
| $B$ | batch size | — |
| $A$ | 每样本 anchor / block 数 | 512 |
| $K$ | block size（块内 token 数） | 16 |
| $D$ | 并行骨干 hidden 维度 | 4096 |
| $R$ | Markov 低秩维度 `markov_rank` | 256 |
| $V$ | 词表大小 | 151936 |
| $h_k$ | 位置 $k$ 的并行骨干 hidden | $[\text{*}, D]$ |
| $x_{k-1}$ | 前一 token ID | 标量 / $[\text{*}]$ |
| $e_{k-1}$ | 前一 token embedding $E[x_{k-1}]$ | $[\text{*}, R]$ |
| $s_k$ | RNN 状态 | $[\text{*}, R]$ |

---

## 1. DSpark `RNNHead`

**源码：** `deepspec/modeling/dspark/markov_head.py` → `class RNNHead`

### 1.1 输入

| 输入 | 训练形状 | 推理形状 | 说明 |
|------|----------|----------|------|
| `base_logits` | $[B, A, K, V]$ | $[B, K, V]$ | 并行骨干经 draft LM head 得到的基础 logits |
| `token_ids`（prev） | $[B, A, K]$ | 逐步更新 | 位置 $k$ 的前驱 token；$k{=}0$ 为 anchor |
| `hidden_states` $h_k$ | $[B, A, K, D]$ | $[B, K, D]$ | 并行骨干每位置 hidden |
| `state` $s_{k-1}$ | $[B, A, R]$ | $[B, R]$ | 块内初始为 0 |

### 1.2 单步计算（`_rnn_step`）

**联合输入（state 更新与 output 共用）：**

$$
z_k = [\,s_{k-1};\, e_{k-1};\, h_k\,] \in \mathbb{R}^{2R+D}
$$

**联合投影：**

$$
[\tilde{g}_k,\, \tilde{c}_k,\, o_k]
= W_{\mathrm{joint}}\, z_k
\in \mathbb{R}^{3R}
$$

**状态更新（含 $h_k$）：**

$$
g_k = \sigma(\tilde{g}_k),\quad
\tilde{s}_k = \tanh(\tilde{c}_k),\quad
s_k = g_k \odot s_{k-1} + (1-g_k)\odot \tilde{s}_k
$$

**输出（additive，无独立 hidden 融合支路）：**

$$
\mathrm{bias}_k = W_2\big(\tanh(o_k)\big) \in \mathbb{R}^{V}
$$

$$
\ell_k = \ell^{\mathrm{base}}_k + \mathrm{bias}_k
\in \mathbb{R}^{V}
$$

其中 $\ell^{\mathrm{base}}_k = \mathrm{LMHead}(h_k)$，$W_2$ 即 `markov_w2: Linear(R→V)`。

### 1.3 维度流（单步）

```text
previous state s_{k-1}     [*, R]
prev token embed e_{k-1}   [*, R]
current hidden h_k         [*, D]
    ↓ concat
joint input z_k            [*, 2R + D]
    ↓ Linear(2R+D, 3R)
gate / candidate / output  3 × [*, R]
    ↓ GRU-style update (uses h_k)
new state s_k                [*, R]
output branch o_k            [*, R]   (不写入 state)
    ↓ tanh → Linear(R, V)
markov bias                  [*, V]
    ↓ add
final logits                 [*, V]    (+ base_logits)
```

### 1.4 可学习参数

| 模块 | 维度 | 作用 |
|------|------|------|
| `markov_w1` | Embedding $V \times R$ | 前驱 token 嵌入 |
| `joint_proj` | Linear $(2R{+}D) \to 3R$ | 同时驱动 state 更新与 output 支路 |
| `markov_w2` | Linear $R \to V$ | 低秩状态 → 词表 bias |

**特点：** 仅 **additive** 一种输出方式；$h_k$ 同时进入 **state 更新** 和 **base logits**（经 LM head），RNN 支路本身不再单独投影 $h_k$。

---

## 2. FlashMTP `FlashMTPMarkovHead`（RNN）

**源码：** `specforge/modeling/draft/flashmtp_markov_head.py` → `head_type="rnn"`

### 2.1 输入

| 输入 | 训练形状 | 推理形状 | 说明 |
|------|----------|----------|------|
| `hidden_states` $h_k$ | $[B, A, K{-}1, D]$ | $[B, K{-}1, D]$ | 并行骨干 hidden（预测 slot $1..K{-}1$） |
| `prev_token_ids` | $[B, A, K{-}1]$ | 逐步更新 | teacher forcing / 采样得到的前驱 token |
| `state` $s_{k-1}$ | $[B, A, R]$ | $[B, R]$ | 块内初始为 0 |
| `base_logits` | additive 时需要 | additive 时需要 | $\mathrm{LMHead}(h_k)$，$[B, A, K{-}1, V]$ |

> FlashMTP 训练 CE 在 anchor 之后的 $K{-}1$ 个预测位上计算；DSpark 对整块 $K$ 个位置做 `apply_block_logits`（含 anchor 位逻辑由 `prev_token_ids` 构造决定）。

### 2.2 单步计算（`_compute_step_latent`）

**① 记忆支路（仅 token 链，不含 $h_k$）：**

$$
z^{\mathrm{mem}}_k = [\,s_{k-1};\, e_{k-1}\,] \in \mathbb{R}^{2R}
$$

$$
[\tilde{g}_k,\, \tilde{c}_k] = W_{\mathrm{state}}\, z^{\mathrm{mem}}_k \in \mathbb{R}^{2R}
$$

$$
s_k = \sigma(\tilde{g}_k)\odot s_{k-1}
+ \bigl(1-\sigma(\tilde{g}_k)\bigr)\odot \tanh(\tilde{c}_k)
$$

$$
u^{\mathrm{serial}}_k = \tanh\!\big(W_{\mathrm{out}}\, s_k\big) \in \mathbb{R}^{R}
$$

**② 输出支路（仅 direct 模式融合 $h_k$）：**

$$
h'_k = W_h\, h_k \in \mathbb{R}^{R}
$$

$$
\alpha_k = \sigma\!\big(W_{\mathrm{fuse}}\,[u^{\mathrm{serial}}_k;\, h'_k]\big) \in \mathbb{R}^{R}
$$

$$
u_k = \alpha_k \odot u^{\mathrm{serial}}_k
+ (1-\alpha_k)\odot h'_k
$$

**③ 词表 logits：**

| 模式 | 公式 | 输出形状 |
|------|------|----------|
| **additive** | $\ell_k = \mathrm{LMHead}(h_k) + W_{\mathrm{vocab}}\, u^{\mathrm{serial}}_k$ | $[\text{*}, V]$ |
| **direct** | $\ell_k = W_{\mathrm{vocab}}\, u_k$ | $[\text{*}, V]$ |

其中 $W_{\mathrm{vocab}}$ 即 `output_proj: Linear(R→V)`。direct 模式下 **不走** base LM head 作为主 logits。

### 2.3 维度流（单步，direct 模式）

```text
previous state s_{k-1}     [*, R]
prev token embed e_{k-1}   [*, R]
    ↓ concat
memory input               [*, 2R]
    ↓ Linear(2R, 2R)
gate / candidate           2 × [*, R]
    ↓ GRU-style update (no h_k)
new state s_k                [*, R]
    ↓ Linear(R, R) + tanh
serial latent u^serial       [*, R]

current hidden h_k           [*, D]    ← 仅输出侧
    ↓ Linear(D, R)
hidden latent h'             [*, R]
    ↓ gate fuse
head latent u_k              [*, R]
    ↓ Linear(R, V)
head logits ℓ_k              [*, V]
```

### 2.4 可学习参数（RNN）

| 模块 | 维度 | 作用 |
|------|------|------|
| `prev_token_embedding` | Embedding $V \times R$ | 前驱 token 嵌入 |
| `state_proj` | Linear $2R \to 2R$ | **仅 token 记忆** 的 GRU 更新 |
| `state_out_proj` | Linear $R \to R$ | 从 state 读出 serial latent |
| `hidden_proj` | Linear $D \to R$ | 并行 hidden → 低秩（direct） |
| `hidden_fuse_gate_proj` | Linear $2R \to R$ | serial / hidden 门控融合 |
| `output_proj` | Linear $R \to V$ | 低秩 latent → 词表 |

---

## 3. 核心差异一览

| 维度 | DSpark `RNNHead` | FlashMTP RNN |
|------|------------------|---------------|
| **State 更新输入** | $[s_{k-1}; e_{k-1}; h_k]$ | $[s_{k-1}; e_{k-1}]$ **不含** $h_k$ |
| **$h_k$ 的作用** | 进入 state 更新 + base logits | 仅输出侧（direct 下 gate 融合）；additive 下只走 LM head |
| **RNN → 词表路径** | $\tanh(o_k) \xrightarrow{W_2} \mathbb{R}^V$（直接出 bias） | 先得到 $u_k \in \mathbb{R}^R$，再 $W_{\mathrm{vocab}}$ |
| **与 base LM 关系** | 固定 additive：$\ell = \ell^{\mathrm{base}} + \mathrm{bias}$ | additive **或** direct（Markov 独占 logits） |
| **Hidden 融合** | 无独立融合层；$h_k$ 经 LM head 与 bias 相加 | direct 下 $u^{\mathrm{serial}}$ 与 $W_h h_k$ **门控融合** |
| **联合投影** | 一个 `joint_proj (2R+D→3R)` | 拆为 `state_proj (2R→2R)` + 输出侧多层 |
| **块内展开** | `apply_block_logits` 串行 $K$ 步 | `forward_teacher_forcing` 串行 $K{-}1$ 步 |
| **State 语义** | 记忆 + 位置 hidden 混合 | 记忆 **仅编码前序 token 链** |

**设计意图简述：**

- **DSpark：** 一个 GRU cell 同时吃 $(s, e, h)$，RNN 输出作为 LM head 上的 **加性 correction**；并行信息既在 base logits 里，也参与 state 轨迹。
- **FlashMTP：** 刻意 **解耦**——state 只维护 token 历史；$h_k$ 在 direct 模式下通过 **低秩门控** 注入输出，可选完全去掉 base LM head（direct）；另可辅以 `BASE_LM_CE` 监督骨干 hidden。

---

## 4. 参数量对比（$D{=}4096,\, R{=}256$，不含 embedding）

| 模块 | DSpark RNN | FlashMTP RNN |
|------|------------|--------------|
| 状态 / 联合投影 | $(2R{+}D)\times 3R \approx 3.46$M | $2R \times 2R \approx 0.26$M（`state_proj`） |
| State → latent | （含在 joint 的 $o_k$ 支路） | $R^2 \approx 0.07$M（`state_out_proj`） |
| Hidden → latent | — | $D \times R \approx 1.05$M（`hidden_proj`） |
| 融合门控 | — | $2R^2 \approx 0.13$M（`hidden_fuse_gate_proj`） |
| Latent → vocab | $R \times V$（共享 `markov_w2`） | $R \times V$（`output_proj`） |

> 两者共享 $V \times R$ 级 embedding + $R \times V$ 输出投影量级；FlashMTP 把「记忆」与「并行上下文」拆开后，**state 侧更轻**，direct 融合侧额外引入 $D{\to}R$ 与 gate。

---

## 5. AI 绘图提示词（学术风格结构图）

以下提示词可直接交给图像生成 / 绘图 AI（如 GPT-4o、Midjourney、diagram 工具），分别生成两张 **论文插图风格** 的结构图。建议输出：**白底、矢量框图、细线箭头、Times/New Roman 或 LaTeX 风格标注、无 3D 装饰**。

### 图 A：DSpark RNNHead

```
Create an academic neural architecture diagram (white background, clean vector style, suitable for an ML paper figure). Title: "DSpark RNNHead (DeepSpec)".

Layout: left-to-right dataflow for one timestep k inside a draft block.

Input boxes (top row):
- "s_{k-1}" tensor shape [R]  (recurrent state, light blue)
- "x_{k-1}" token ID → "Embedding W1" → "e_{k-1}" [R]  (light green)
- "h_k" parallel backbone hidden [D]  (light orange)

All three concatenate into "z_k = [s_{k-1}; e_{k-1}; h_k]" with dimension [2R+D].

Single shared block: "Linear joint_proj (2R+D → 3R)" splitting into three branches:
- gate g_k [R] with σ
- candidate c_k [R] with tanh
- output o_k [R]

GRU state update box:
"s_k = g_k ⊙ s_{k-1} + (1-g_k) ⊙ c_k"  [R]
Show h_k arrow entering the joint_proj (NOT bypassing to state directly).

Parallel branch from h_k (separate path, dashed orange):
"h_k" → "Draft LM Head" → "base_logits_k" [V]

Output branch:
"tanh(o_k)" [R] → "Linear W2 (R→V)" → "markov_bias_k" [V]

Final summation node:
"ℓ_k = base_logits_k + markov_bias_k" [V]

Feedback arrow: s_k loops to next timestep s_{k-1}.

Caption note: "Within-block teacher forcing; state initialized to 0 per block."

Style: IEEE/NeurIPS figure aesthetic, monochrome boxes with subtle color fills, dimension labels on every arrow, no cartoon icons, sans-serif math labels.
```

### 图 B：FlashMTP RNNHead（direct 模式）

```
Create an academic neural architecture diagram (white background, clean vector style, suitable for an ML paper figure). Title: "FlashMTP RNNHead (direct mode)".

Layout: two parallel pathways merging at the end — Memory Path (top) and Context Path (bottom).

MEMORY PATH (token-only recurrence):
Inputs:
- "s_{k-1}" [R] (light blue)
- "x_{k-1}" → "Embedding E" → "e_{k-1}" [R] (light green)

Concatenate: "z_mem = [s_{k-1}; e_{k-1}]" [2R]
→ "Linear state_proj (2R→2R)" split into gate and candidate
→ GRU update: "s_k = σ(g)⊙s_{k-1} + (1-σ(g))⊙tanh(c)" [R]
→ "Linear state_out_proj (R→R)" + tanh → "u^serial_k" [R]

Explicitly show NO arrow from h_k into the memory path.

CONTEXT PATH (parallel backbone, output only):
"h_k" [D] (light orange) → "Linear hidden_proj (D→R)" → "h'_k" [R]

FUSION block:
Concatenate [u^serial_k; h'_k] [2R]
→ "Linear fuse_gate (2R→R)" + σ → "α_k" [R]
→ element-wise: "u_k = α_k ⊙ u^serial_k + (1-α_k) ⊙ h'_k" [R]

OUTPUT:
"u_k" [R] → "Linear output_proj (R→V)" → "ℓ_k" [V]

Show NO base LM head in this diagram (direct mode). Optional small note: "additive mode: ℓ_k = LMHead(h_k) + W_vocab·u^serial_k".

Feedback: s_k loops to next step.

Style: same as Figure A for consistency; use a dashed box around Memory Path labeled "Token-chain state (no h_k)" and solid box around Context Path labeled "Parallel context (per-position h_k)".

Caption: "State encodes prefix tokens only; backbone hidden fused at output via gated low-rank latent."
```

### 使用建议

1. 两张图使用 **相同配色与字体**，便于并排放入论文或 slide。
2. 若工具支持，在图下方加 **Notation table**：$R{=}256,\, D{=}4096,\, V{=}|\mathcal{V}|$。
3. 对比展示时建议左右并排：左 DSpark（单 joint_proj + additive），右 FlashMTP（解耦 memory / context + direct）。
