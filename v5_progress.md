# FlashMTP v5 修改总结

## 核心动机

v5 的出发点是承认 pure pivot 条件的信息量可能不足，但仍然坚持草稿侧不维护 KV cache。相比 v3.3 只在 `anchor-1` 单点上做多层 hidden fusion，v5 适量引入历史 hidden states，让 pivot 在进入草稿模型前先从历史融合表示中补充上下文信息。

这不是回到 DFlash 的完整历史 KV cache 路线。v5 仍然只把一个 enriched pivot 作为草稿模型的显式条件；历史信息只在进入草稿模型前被轻量 cross-attention 汇入 pivot。

## Hidden State 融合方法

v3.3 使用 depth-axis attention，让最后层 hidden attend 同一位置的所有层 hidden。v5 改为更接近 DFlash 的融合方式：

- 等间隔选取 `num_draft_layers` 个 target model layer。
- 对每个历史位置，将这些层的 hidden states 沿特征维拼接。
- 使用一个 FC 层降维回 draft hidden size。
- 对每个 token 位置都保留这样的 fused history hidden。

这样做的考虑是：不同 target 层包含不同层级的语义和局部/全局信息，但用 input-aware depth attention 只处理单个 pivot 点，不能让 pivot 主动访问历史轨迹。v5 先把每个历史位置压成统一语义空间，再让 pivot 在时间维上选择性读取历史。

## Pivot Cross-Attention

训练中一个 block 的起点为 `anchor = a`。块首 token 是 clean anchor token，但草稿条件使用的是 target model 在 `a-1` 位置的 hidden state，即 pivot。

v5 的 cross-attention 约定：

- query：`fused_hs[a-1]`，即 pivot hidden。
- key/value：`fused_hs[0:a-1]`，即 pivot 之前的历史 hidden，不包含 pivot 自身。
- query RoPE position id：`a-1`。
- key/value RoPE position ids：`0, 1, ..., a-2`。
- 输出通过 residual 保留 pivot 自身信息，得到 enriched pivot。

因此有两层位置编码语义：

- pivot-history cross-attention 内部：建模 pivot 对历史 token 的读取关系。
- draft decoder 内部：enriched pivot 作为唯一 context token，对应位置 `a-1`；draft block 的位置为 `a, a+1, ...`。

这个设计避免训练时使用 `anchor` 位置 target hidden，因为推理时当前 anchor token 虽然 clean，但还没有经过 target forward 得到 hidden state。训练/推理条件因此保持一致。

## MDLM 与 Streak 的关系

MDLM 和 Streak 的损失目标不变。修改集中在草稿模型的条件构造：

- MDLM 仍然做块内随机 mask，监督 `pos_in_block > 0` 的 masked token。
- Streak 仍然使用 LS-RSL 作为主目标，并可用 CE 辅助项。达标后使用 log-rho 上的指数饱和函数，并加入位置相关权重：

$$
\phi_j(\rho_j)=
\begin{cases}
\rho_j, & \rho_j < 1, \\\\
1+\frac{w_j}{\beta}\left(1-\exp(-\beta\log\rho_j)\right), & \rho_j \ge 1,
\end{cases}
$$

其中块内监督位置为 $j=1,2,\dots,\gamma$，反向使用 DFlash 风格的指数衰减，让越靠前的位置达标后权重越小：

$$
w_j=\exp\left(-\frac{\gamma-j}{7}\right)
$$

其中 $\beta>0$ 控制达标后的饱和速度；当前默认 $\beta=2$。最后一个监督位置满足 $w_\gamma=1$，前面位置的达标后梯度按指数减小。对 $\rho_j$ 的导数为：

$$
\frac{\partial \phi_j}{\partial \rho_j}
=
w_j\rho_j^{-(\beta+1)},
\qquad \rho_j \ge 1.
$$

- 两者都不再只传单点 stacked hidden，而是传完整目标 hidden history 和 attention mask。

这样 MDLM 和 Streak 共享同一个 v5 draft 条件路径，避免两阶段训练学到不同的信息接口。

## 推理一致性

推理侧也同步维护完整历史 hidden states：

- prefill 后保留 prompt 的完整 target hidden states。
- 每次验证一个 speculative block 后，只追加 seed token 和已接受 draft token 的 hidden。
- 被拒绝后由 target 采样出的下一个 token 尚未经过 target forward，因此不追加它的 hidden。

下一轮 draft 的 pivot 仍然对应当前 anchor 的前一位置。这样 cross-attention 看到的历史范围与训练中的 `0..a-2` 对齐。

# DFlash部分


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
\text{Accept}\_{\text{L}}(\gamma, p) = \mathbb{E}_{x_{1:\gamma} \sim Q(\cdot|p)} \left[ \sum_{m=1}^{\gamma} \prod_{j=1}^{m} \alpha_j\bigl(c \circ x_{1:j-1}\bigr) \right]
$$

其中 $c$ 为与 $p$ 对齐的验证前文；$\alpha_j$ 随已接受前缀变，$Q$ 侧条件固定为 $p$。

---

**4. 贪婪接受代理**

$$
\tilde{\alpha}_j(p) \approx \mathbb{E}_{x_{1:j-1} \sim P(\cdot|c)}\, \mathbb{E}_{x_j \sim P(\cdot|c \circ x_{1:j-1})}\bigl[\, Q_{\text{diff}}(x_j \,|\, p) \,\bigr]
$$

$Q_{\text{diff}}(x_j|p)$ 为位置 $j$ 在 pivot 条件下的预测概率（与掩码日程一致即可）。

---

**5. Streak 目标**

$$
\mathcal{L}_{\text{streak}}(\theta) = \mathbb{E}_{(p,c)}\;\mathbb{E}_{\,x_{1:\gamma} \sim P(\cdot|c)}\left[ \sum_{m=1}^{\gamma} \prod_{j=1}^{m} q_j\bigl(x_j \,\big|\, p\,;\theta\bigr) \right]
$$

- 每条样本上取对齐的 $(c,p)$；$x_{1:\gamma}$ 为响应中的续写块（形式上等价于 $P(\cdot|c)$ 的轨迹）。  
- $q_j$ 为草案在位置 $j$ 的概率；并行下联合取 $\prod_j q_j(\cdot|p)$。

---

**6. 梯度权重调整**  


我们将原始的 Streak Loss 改造为 Log-Smoothed Relative Streak Loss (LS-RSL)，并在达标后使用 log-rho 上的指数饱和函数。

**A. 目标锚点 (Target Anchoring)**

首先定义每个位置的置信度门槛 $T_j$，防止被大模型的低置信度“带歪”：$$T_j = \max(0.5, p_j)$$  
其中 $p_j$ 是教师模型在 target token 上的概率。

**B. 相对概率映射 (Relative Mapping)**  

定义当前草稿模型的相对置信度 $\rho_j = q_j / T_j$。为了实现“达标后降权但不截断”，我们使用一个带位置权重的分段平滑函数 $\phi_j(\rho_j)$：  

$$
\phi_j(\rho_j)=
\begin{cases}
\rho_j, & \rho_j < 1, \\\\
1+\frac{w_j}{\beta}\left(1-\exp(-\beta\log\rho_j)\right), & \rho_j \ge 1,
\end{cases}
$$

其中块内监督位置为 $j=1,2,\dots,\gamma$。借鉴 DFlash 的指数位置衰减，但方向反过来，让越靠前的位置在达标后权重越小：

$$
w_j=\exp\left(-\frac{\gamma-j}{7}\right)
$$

其中 $\beta>0$ 控制达标后的饱和速度；当前默认 $\beta=2$。因此最后一个监督位置满足 $w_\gamma=1$，前面位置的达标后梯度按指数减小。对 $\rho_j$ 的导数为：

$$
\frac{\partial \phi_j}{\partial \rho_j}
=
w_j\rho_j^{-(\beta+1)},
\qquad \rho_j \ge 1.
$$

**C. 修改后的 Streak Loss**

将 $\phi_j(\rho_j)$ 代入累乘项，构建期望相对长度：$$\mathcal{L} = -\log \left( \sum_{m=1}^{\gamma} \prod_{j=1}^{m} \phi_j(\rho_j) \right)$$

---
**7. 混合loss**  
从头训练时，总 loss 采用逐位置 CE 辅助项打底：

$$
\mathcal{L}_{\text{total}}
= \lambda_{\text{streak}}\mathcal{L}_{\text{conf-streak}}
+ \lambda_{\text{ce}}\mathcal{L}_{\text{CE}}
$$

$\mathcal{L}_{\text{CE}}$ 对块内除 anchor 外的每个有效位置单独计算平均 CE，并同样除以实际参与 CE 的位置数（完整块为 $B-1$），不加位置权重，也不加置信度权重；主要目标仍由 $\mathcal{L}_{\text{conf-streak}}$ 控制。

目标：在固定 $p$ 下拉高与目标续写一致的 **长 streak**，而非只对齐首 token。

> **实现（与上节 clarification 对齐）**：块内序列下标从块起点起算时，**第一个 slot 不参与** streak 与 MDLM 监督；代码用 `pos_in_block>0` 掩码，且 streak 外层对 $\exp(\mathrm{cum}_m)$ 从块内第二槽对应的前缀（$m\ge 1$）起求和，避免仅将首槽 $\log q$ 置零仍残留 $\exp(0)$ 常数项。
