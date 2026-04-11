# FlashMTP Base 结构：关键代码说明

---

## 1. 设计要点（与 README 的映射）


| README 概念                                                                                                                 | 代码中的体现                                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| 使用**所有层**的 bonus 隐状态，沿序列或特征维拼接                                                                                            | `prepare_target_hidden`（`specforge/core/flashmtp.py` L21）；`target_layer_ids`（`specforge/modeling/draft/flashmtp.py` L237） |
| bonus 干净 token + mask 噪声；**在 `FlashMTPDraftModel.forward` 将 CHS 与噪声在序列维拼接**，整段送入各层；注意力对**整条拼接序列**做 Q/K/V 投影，**输出再裁成仅噪声段** | 拼接 L287；`Qwen3FlashMTPAttention` 中 Q/K/V 投影 L105–107；输出截取 `draft/flashmtp.py` L299                                        |
| **seq 模式**：多层 CHS 在序列维展开，**RoPE 的 `position_ids` 对每层重复同一 context 位置**                                                     | `OnlineFlashMTPModel.forward` L312–320（`repeat_interleave`）                                                               |
| **feature 模式**：多层在特征维拼接，再经 `fc` 映射到 `hidden_size`                                                                         | `prepare_target_hidden` L58–60；`fc` L248–254                                                                              |
| 块内**双向**、多块隔离                                                                                                             | `is_causal = False`（`draft/flashmtp.py` L58）；`create_flashmtp_block_mask`（`core/flashmtp.py` L63）                         |


---

## 2. 文件与职责总览


| 路径                                                                                                           | 职责                                                                              |
| ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| `[specforge/modeling/draft/flashmtp.py](./specforge/modeling/draft/flashmtp.py)`                             | 草稿模型：`Qwen3FlashMTPAttention` / `FlashMTPDraftModel`、`spec_generate`            |
| `[specforge/core/flashmtp.py](./specforge/core/flashmtp.py)`                                                 | 训练封装：`prepare_target_hidden`、`create_flashmtp_block_mask`、`OnlineFlashMTPModel` |
| `[specforge/modeling/target/flashmtp_target_model.py](./specforge/modeling/target/flashmtp_target_model.py)` | Target 后端；`generate_flashmtp_data`（HF/SGLang）                                   |
| `[scripts/train_flashmtp.py](./scripts/train_flashmtp.py)`                                                   | 训练入口：解析参数、`build_models`、FSDP、数据循环                                              |


---

## 3. 训练脚本 `scripts/train_flashmtp.py`

**文件**：`[./scripts/train_flashmtp.py](./scripts/train_flashmtp.py)`

- **CLI**（含 `--chs-concat-mode`、`--flashmtp-loss-type`、`--distill-temperature`、`--kl-topk` 等）→ `parse_args`（L39–161）
- **构建 target + draft、写入 `flashmtp_config`** → `build_models`（L163–214）
- **组装 `OnlineFlashMTPModel`**（加载 target 的 `embed_tokens` / `lm_head`）→ `OnlineFlashMTPModel(...)`（L442–455）
- **FSDP 包装** → `FSDP(...)`（L457–465）
- **每步**：`generate_flashmtp_data` → `flashmtp_model(...)`将离线收集的回答prefill一遍得到全部大模型hidden states（L515–527）

---

## 4. CHS 张量准备：`prepare_target_hidden`

**文件**：`[specforge/core/flashmtp.py](./specforge/core/flashmtp.py)`（**L21** 起）

- **anchor 前一位置 gather**（预测 anchor 处 token 的上下文）→ `context_positions`、`torch.gather`（L39–53）
- **按** `target_layer_ids` **逐层收集**（L45–53）
- **seq**：`torch.cat(..., dim=1)` → `(B, N*L, H)`（L55–57）
- **feature**：`torch.cat(..., dim=-1)` → `(B, N, H*L)`（L58–60）

---

## 5. 训练时块掩码：`create_flashmtp_block_mask`

**文件**：`[specforge/core/flashmtp.py](./specforge/core/flashmtp.py)`（**L63** 起）

- **布局与规则**（docstring L70–94）：序列形如 `[CHS_0|…|CHS_{N-1}|Block_0|…|Block_{N-1}]`；**同一块组内** CHS 与对应 Block **互相可见**，块间不可见；无效块被屏蔽。
- **可见性逻辑** → `flashmtp_mask_mod`（L96–123）：`same_group`、`is_valid`（含 `block_keep_mask`）
- **构造长度并调用 flex mask** → `B, N = ...` 至 `return`（L125–134）

---

## 6. 草稿模型主体：`FlashMTPDraftModel`

**文件**：`[specforge/modeling/draft/flashmtp.py](./specforge/modeling/draft/flashmtp.py)`（类 **L221** 起）

- **推理时从 target 各层 hidden 取特征并拼接** → `extract_context_feature`（L207–218）
- `__init__`：`chs_concat_mode`、`target_layer_ids`、`fc`（L225–267）
- **feature**：`Linear(len(target_layer_ids)*H → H)`（L248–254）
- **seq**：同样使用一层 `Linear(H→H)`（L256–262）；再与噪声段拼接（L287）
- **前向**：`target_hidden = self.fc(target_hidden)`（L284），`torch.cat([target_hidden, noise_embedding], dim=1)`（L287），堆叠层后 **只返回噪声段** `[:, -noise_len:, :]`（L299）

---

## 7. 草稿层注意力：`Qwen3FlashMTPAttention`

**文件**：`[specforge/modeling/draft/flashmtp.py](./specforge/modeling/draft/flashmtp.py)`（类 **L43** 起）

- **输入**：`hidden_states` 在**进入本层前**已是 `[fc(CHS) | noise_embedding]` **在序列维拼接**的一条序列（见 `FlashMTPDraftModel.forward` L287）；`target_hidden` 仍传入层内用于接口兼容，当前实现中 **Q/K/V 均由拼接后的** `hidden_states` **投影**（L105–107）。
- **RoPE**：对**整段** `q`、`k` 施加（L130–132），`position_ids` 由外层 `FlashMTPDraftModel`/`OnlineFlashMTPModel` 构造（seq 与 feature 分支见 core L312–327）。
- **双向（非因果）** → `self.is_causal = False`（L58）
- **输出投影** → `o_proj`（L152–153）

**解码层封装** → `Qwen3FlashMTPDecoderLayer.forward`（L168–204）

---

## 8. 在线训练封装：`OnlineFlashMTPModel`

**文件**：`[specforge/core/flashmtp.py](./specforge/core/flashmtp.py)`（类 **L137** 起）

- **超参数**（`loss_type`、`distill_temperature`、`kl_topk` 等）→ `__init__`（L140–174）
- **随机锚点** → `_sample_anchor_positions`（L176–211）
- **整序列噪声 id 辅助** → `prepare_noise_input`（L213–234）
- **块内 `position_ids`** → `_create_position_ids`（L236–243）
- **并行块噪声 id + `embed_tokens`** → `_create_noise_embed`（L245–273）
- **前向**：锚点、`full_position_ids`、`prepare_target_hidden`、`draft_model`、`lm_head` → `forward`（L286–360）
- **标签与 `weight_mask`、可选 `loss_decay_gamma`**（L362–398）
- **损失**：`loss_type == "kl"` 时为教师 last-hidden + **可选 top-k KL**（L406–452）；否则加权 **CE**（L453–458）
- **准确率**（L460–465）

---

