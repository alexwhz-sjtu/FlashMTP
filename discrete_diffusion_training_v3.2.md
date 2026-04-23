# FlashMTP v3.2：离散扩散式草稿训练（更新说明）

本文档说明在 `specforge/core/flashmtp.py` 与 `scripts/train_flashmtp.py` 中新增的 **离散扩散风格** 训练选项，以及与既有「干净前缀补全」训练的关系。

## 1. 背景

`iterative_block_generation_design.md` 中描述的原训练目标是：在块内给定 **干净前缀** + 后缀 `[MASK]`，对后缀做加权 CE（越靠近块首的待预测位置权重越大）。

v3.2 在此基础上增加 **`discrete_diffusion` 模式**：在草稿块内按 **噪声调度** 随机选取若干位置打上 `[MASK]`，其余位置保留 **干净 token**；模型在 teacher 条件（融合 hidden）下 **只恢复被掩码位置**。该设定与离散状态空间中的随机掩码 / 扩散步进目标一致，便于与 CE、对 teacher 分布的 KL、以及表示层的 MSE 联合优化。

## 2. 训练模式切换

| 模式 | 含义 |
|------|------|
| `clean_prefix` | 原有逻辑：每个 anchor 复制为 cold-start（p=1）与 continuation（p≥2）两路，共 `2N` 个并行块；仅对「补全后缀」位置做加权 CE。 |
| `discrete_diffusion` | 每个 anchor **单路** `N` 个块；块内按调度随机掩码；在掩码位置上计算 CE / KL / MSE（可选加权）。 |

命令行：

```bash
--training-mode discrete_diffusion
```

默认仍为 `clean_prefix`，以保持旧脚本行为不变；新实验请显式打开 `discrete_diffusion`。

## 3. 噪声调度与随机掩码

- **`--diffusion-mask-schedule`**  
  - `uniform`：掩码比例 `r ~ U[r_min, r_max]`。  
  - `cosine`：先采样 `u ~ U(0,1)`，再令  
    `r = r_min + (r_max - r_min) · sin²(π u / 2)`，使分布更平滑。

- **`--diffusion-mask-ratio-min` / `--diffusion-mask-ratio-max`**  
  约束每个块上掩码比例的可行区间（默认 `0.1` ~ `1.0`）。

- **块内实现**（`block_size ≥ 2`）：在序列合法位置中随机选取 **至少 1 个掩码、至少 1 个未掩码**，从而在块内同时存在「可见干净 token」与「待恢复位置」。`block_size == 1` 时退化为仅掩码该单 token。

## 4. 三种损失与权重

在 **被掩码且落在 `loss_mask` 内** 的位置上计算（块级再平均）：

1. **CE**（`loss_weight_ce`）：学生分布对真实 token 的交叉熵。  
2. **KL**（`loss_weight_kl`）：`KL(student ‖ teacher)`，其中 teacher 为 **目标模型 `lm_head` 在对应序列位置、最后一层 teacher hidden 上** 的分布（与推理时 target 一致）。  
3. **MSE**（`loss_weight_mse`）：学生草稿 **最后一层输出 hidden** 与 teacher **最后一层 hidden**（由 `target_layer_ids` 中最后一层索引取出）的均方误差。

总损失（块上已聚合为标量后再按 batch 平均）：

\[
\mathcal{L} = w_{\mathrm{ce}} \mathcal{L}_{\mathrm{ce}} + w_{\mathrm{kl}} \mathcal{L}_{\mathrm{kl}} + w_{\mathrm{mse}} \mathcal{L}_{\mathrm{mse}}
\]

对应命令行：

```text
--loss-weight-ce 1.0
--loss-weight-kl 0.0
--loss-weight-mse 0.0
```

将某一项设为 `0` 可关闭该项（KL/MSE 关闭时可减少一次 `lm_head` 全序列前向或 hidden 对齐计算的开销；当前实现中 KL 为 0 时不再计算 teacher logits 聚合）。

## 5. 位置权重（越靠前越大）

与 design 文档中「块内靠前位置更重要」一致，在 `discrete_diffusion` 下对块内下标 `j = 0, …, B-1` 使用与 **块首距离** 相关的指数衰减（`--loss-decay-gamma`）：

- 设 `γ > 0`：未归一化权重 \(w_j \propto \exp(-j/\gamma)\)，仅在 **监督位置**（掩码 ∩ 合法 ∩ loss_mask）上归一化，使每个块内在监督位置上的权重和为 1。  
- `γ` 为空或 `≤ 0`：监督位置上均匀权重。

该权重同时作用于 CE、KL、MSE 的块内加权和（同一套 `tilde_w`）。

## 6. 日志与监控

`discrete_diffusion` 下 `loss_dict` 额外包含：

- `loss_ce_mean` / `loss_kl_mean` / `loss_mse_mean`：各分项在未加权聚合意义下的块均值（便于看量级；总损失仍由 `loss_weight_*` 组合）。  
- `mean_mask_ratio`：有效块上「掩码位置数 / block_size」的平均。

训练脚本在 `training_mode == discrete_diffusion` 时会对上述标量做分布式 `all_reduce` 并写入 tracker；进度条 postfix 会显示 `mask_r`、`ce_m`、`kl_m`、`mse_m`。

## 7. 与 clean_prefix 的兼容说明

- **continuation 相关参数**（`--continuation-loss-weight`、`--continuation-warmup-epochs` 等）仅对 **`clean_prefix`** 生效。  
- **`discrete_diffusion`** 不使用 cold/continuation 双路展开，因此 `mean_prefix_len_*` 与 `lambda2_eff` 在日志中可为 0；请以 `mean_mask_ratio` 与三项 `loss_*_mean` 为准。

## 8. 推荐起步配置示例

```bash
python scripts/train_flashmtp.py \
  --training-mode discrete_diffusion \
  --diffusion-mask-schedule cosine \
  --diffusion-mask-ratio-min 0.1 \
  --diffusion-mask-ratio-max 1.0 \
  --loss-weight-ce 1.0 \
  --loss-weight-kl 0.1 \
  --loss-weight-mse 0.01 \
  --loss-decay-gamma 2.0 \
  ... # 其余数据与模型路径与原先相同
```

可根据验证集调 `w_kl`、`w_mse` 与 `γ`；若仅希望与旧版「仅 CE」接近，可设 `loss-weight-kl=0`、`loss-weight-mse=0`。

---

*文档版本：与 FlashMTP v3.2 代码分支一致；实现位置：`OnlineFlashMTPModel.training_mode`、`_forward_discrete_diffusion`。*
