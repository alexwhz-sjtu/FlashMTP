# FlashMTP Base 结构：关键代码说明

本文档对应 [README.md](./README.md) 中的 **Base structure** 与 **v1.1 Improved condition injection**：说明仓库里与「多锚点并行训练 + 单次前向草稿块 + CHS 条件」直接相关的实现位置与职责，便于对照论文/设计阅读源码。

> **关于预览里点击跳转**  
> Markdown 预览会把链接里的 `#L89` 当成**文件名的一部分**（去找 `flashmtp.py#L89`），从而提示文件不存在。  
> 因此下文链接**只指向真实文件路径**（以 `./` 相对本 MD 所在目录）；**行号写在链接文字里**。打开文件后请用 **Ctrl+G**（Windows/Linux）或 **Cmd+G**（macOS）「转到行」输入括号中的数字即可。

---

## 1. 设计要点（与 README 的映射）


| README 概念                                        | 代码中的体现                                                                                                                                      |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| 使用**所有层**的 bonus 隐状态，沿序列或特征维拼接                   | `[prepare_target_hidden`（L21）](./specforge/core/flashmtp.py)；`[target_layer_ids`（L230）](./specforge/modeling/draft/flashmtp.py)             |
| bonus 干净 token + mask 噪声，**噪声侧作 Q**，拼接序列为 **KV** | `[Qwen3FlashMTPAttention.forward`（L89 起）](./specforge/modeling/draft/flashmtp.py) 中 `q` 与 `k/v` 拼接                                          |
| v1.1：**整条噪声序列作 Q**、CHS 作 KV 前缀                   | `[FlashMTPDraftModel.forward`（L257）](./specforge/modeling/draft/flashmtp.py)                                                                |
| **seq 模式**：RoPE 主要作用在噪声段 K                       | `[chs_concat_mode == "seq"`（L120）](./specforge/modeling/draft/flashmtp.py)；`[full_position_ids` seq 分支（L282）](./specforge/core/flashmtp.py) |
| 块内**双向**、多块隔离                                    | `[is_causal = False`（L58）](./specforge/modeling/draft/flashmtp.py)；`[create_flashmtp_block_mask`（L62）](./specforge/core/flashmtp.py)        |


---

## 2. 文件与职责总览


| 路径                                                                                                           | 职责                                                                              |
| ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| `[specforge/modeling/draft/flashmtp.py](./specforge/modeling/draft/flashmtp.py)`                             | 草稿模型：`Qwen3FlashMTPAttention` / `FlashMTPDraftModel`、`spec_generate`            |
| `[specforge/core/flashmtp.py](./specforge/core/flashmtp.py)`                                                 | 训练封装：`prepare_target_hidden`、`create_flashmtp_block_mask`、`OnlineFlashMTPModel` |
| `[specforge/modeling/target/flashmtp_target_model.py](./specforge/modeling/target/flashmtp_target_model.py)` | Target 后端抽象；SGLang/HF 前向与 `generate_flashmtp_data`                              |
| `[scripts/train_flashmtp.py](./scripts/train_flashmtp.py)`                                                   | 训练入口：构建模型、FSDP、数据 → loss                                                        |


---

## 3. 草稿层注意力：`Qwen3FlashMTPAttention`

**文件**：`[specforge/modeling/draft/flashmtp.py](./specforge/modeling/draft/flashmtp.py)`（类从 **L43** 起）

- **输入分工（Q 来自噪声序列、target_hidden 为 CHS）** → `[forward` 签名与 `q_len` / `ctx_len`（L89–100）](./specforge/modeling/draft/flashmtp.py)
- **Q 投影与归一化** → `[q_proj` → `q_norm` → `transpose`（L102–104）](./specforge/modeling/draft/flashmtp.py)
- **KV：上下文与噪声在序列维拼接** → `[k_ctx`/`v_ctx` 与 `k_noise`/`v_noise`，`torch.cat`（L105–112）](./specforge/modeling/draft/flashmtp.py)
- **seq 模式：仅对噪声段 K 与 Q 施加 RoPE** → `[if self.chs_concat_mode == "seq"`（L120–124）](./specforge/modeling/draft/flashmtp.py)
- **feature 模式：整段 K（含 ctx）与 Q 一起 RoPE** → `[else` 分支（L125–126）](./specforge/modeling/draft/flashmtp.py)
- **KV Cache 更新** → `[past_key_values.update`（L128–130）](./specforge/modeling/draft/flashmtp.py)
- **注意力计算与输出投影** → `[attn_fn` 与 `o_proj`（L132–147）](./specforge/modeling/draft/flashmtp.py)
- **双向（非因果）** → `[self.is_causal = False`（L58）](./specforge/modeling/draft/flashmtp.py)

**解码层封装** → `[Qwen3FlashMTPDecoderLayer.forward`（L162–197）](./specforge/modeling/draft/flashmtp.py)

---

## 4. 草稿模型主体：`FlashMTPDraftModel`

**文件**：`[specforge/modeling/draft/flashmtp.py](./specforge/modeling/draft/flashmtp.py)`（类从 **L214** 起）

- **从 target 的 `hidden_states` 元组按层抽取并特征维拼接（推理用）** → `[extract_context_feature`（L201–211）](./specforge/modeling/draft/flashmtp.py)
- `**__init__`：`chs_concat_mode`、`target_layer_ids`、`fc`/`hidden_norm`** → `[FlashMTPDraftModel.__init__`（L218–252）](./specforge/modeling/draft/flashmtp.py)
- `**feature`：`Linear` 融合多层特征** → `[nn.Linear` 分支（L241–247）](./specforge/modeling/draft/flashmtp.py)
- `**seq`：`Identity` + 仍用 `hidden_norm`** → `[Identity` 分支（L248–252）](./specforge/modeling/draft/flashmtp.py)
- **前向：噪声作 `hidden_states`，CHS 经 `fc`/`hidden_norm`，RoPE 仅对噪声序列** → `[forward`（L257–283）](./specforge/modeling/draft/flashmtp.py)

---

## 5. CHS 张量准备：`prepare_target_hidden`

**文件**：`[specforge/core/flashmtp.py](./specforge/core/flashmtp.py)`（函数从 **L21** 起）

- **锚点前一位置 gather（注释：预测 anchor 处 token）** → `[context_positions` 与 `torch.gather`（L39–52）](./specforge/core/flashmtp.py)
- **按层收集 `selected_states`** → `[for layer_id in target_layer_ids`（L44–53）](./specforge/core/flashmtp.py)
- **seq：序列维拼接 `(B, N*L, H)`** → `[torch.cat(..., dim=1)`（L55–57）](./specforge/core/flashmtp.py)
- **feature：特征维拼接 `(B, N, H*L)`** → `[torch.cat(..., dim=-1)`（L58–60）](./specforge/core/flashmtp.py)

---

## 6. 训练时块掩码：`create_flashmtp_block_mask`

**文件**：`[specforge/core/flashmtp.py](./specforge/core/flashmtp.py)`（函数从 **L62** 起）

- **函数入口与 KV/Q 布局文档字符串** → `[create_flashmtp_block_mask` 与 docstring（L62–93）](./specforge/core/flashmtp.py)
- **可见性规则：`mask_context` / `mask_draft` / `is_valid_block`** → `[flashmtp_mask_mod`（L95–116）](./specforge/core/flashmtp.py)
- `**Q_LEN` / `KV_LEN` 与 `create_block_mask` 调用** → `[B, N = ...` 至 `return`（L118–124）](./specforge/core/flashmtp.py)

---

## 7. 在线训练封装：`OnlineFlashMTPModel`

**文件**：`[specforge/core/flashmtp.py](./specforge/core/flashmtp.py)`（类从 **L127** 起）

- **类定义与超参数（`num_anchors`、`loss_decay_gamma`、`chs_concat_mode` 等）** → `[__init__`（L127–156）](./specforge/core/flashmtp.py)
- **随机锚点采样** → `[_sample_anchor_positions`（L158–193）](./specforge/core/flashmtp.py)
- **按块首为真实 token 的 noise token 序列（辅助接口）** → `[prepare_noise_input`（L195–216）](./specforge/core/flashmtp.py)
- **块内绝对 `position_ids`** → `[_create_position_ids`（L218–225）](./specforge/core/flashmtp.py)
- **块首 bonus + 其余 MASK，再 `embed_tokens`** → `[_create_noise_embed`（L227–255）](./specforge/core/flashmtp.py)
- **训练前向：锚点、噪声 embedding、`full_position_ids`（seq / feature 分支）** → `[forward` 前半（L257–305）](./specforge/core/flashmtp.py)
- **调用 `prepare_target_hidden` 与 `draft_model`** → `[target_hidden` 与 `output_hidden`（L307–321）](./specforge/core/flashmtp.py)
- **共享 target `lm_head` 得 logits** → `[logits = self.lm_head`（L323）](./specforge/core/flashmtp.py)
- **标签 gather、`weight_mask`、可选位置衰减** → `[label_indices` 至 `decay_weights`（L325–361）](./specforge/core/flashmtp.py)
- **加权 CE 与 accuracy** → `[F.cross_entropy` 与 `return`（L363–381）](./specforge/core/flashmtp.py)

---

## 8. 目标模型与训练脚本

### 8.1 `specforge/modeling/target/flashmtp_target_model.py`

**文件**：`[./specforge/modeling/target/flashmtp_target_model.py](./specforge/modeling/target/flashmtp_target_model.py)`

- **训练用输出结构** → `[FlashMTPTargetOutput`（L25–30）](./specforge/modeling/target/flashmtp_target_model.py)
- **抽象基类与 `generate_flashmtp_data` 接口** → `[FlashMTPTargetModel`（L33–64）](./specforge/modeling/target/flashmtp_target_model.py)
- **SGLang：`enable_return_hidden_states`** → `[ServerArgs`（L83–91）](./specforge/modeling/target/flashmtp_target_model.py)
- **默认捕获全层** → `[set_capture_layers`（L116–130）](./specforge/modeling/target/flashmtp_target_model.py)
- **SGLang 前向取 hidden** → `[_extend` 中 `CaptureHiddenMode.FULL`（L171–187）](./specforge/modeling/target/flashmtp_target_model.py)
- **SGLang 路径 `generate_flashmtp_data`** → `[SGLangFlashMTPTargetModel.generate_flashmtp_data`（L196–237）](./specforge/modeling/target/flashmtp_target_model.py)
- **HF 路径 `from_pretrained`** → `[HFFlashMTPTargetModel.from_pretrained`（L245–268）](./specforge/modeling/target/flashmtp_target_model.py)
- **HF 路径 `generate_flashmtp_data`** → `[HFFlashMTPTargetModel.generate_flashmtp_data`（L271–293）](./specforge/modeling/target/flashmtp_target_model.py)
- **后端工厂** → `[get_flashmtp_target_model`（L296–321）](./specforge/modeling/target/flashmtp_target_model.py)

### 8.2 `scripts/train_flashmtp.py`

**文件**：`[./scripts/train_flashmtp.py](./scripts/train_flashmtp.py)`

- **CLI：`--chs-concat-mode` 等** → `[parse_args`（L84–105）](./scripts/train_flashmtp.py)
- **构建 target + draft、写入 `flashmtp_config`** → `[build_models`（L142–193）](./scripts/train_flashmtp.py)
- **组装 `OnlineFlashMTPModel`（target head/embed）** → `[OnlineFlashMTPModel(...)`（L421–431）](./scripts/train_flashmtp.py)
- **FSDP 包装** → `[FSDP(...)`（L433–441）](./scripts/train_flashmtp.py)
- **每步：target 产 hidden → draft 前向与 loss** → `[generate_flashmtp_data` 与 `flashmtp_model`（L490–503）](./scripts/train_flashmtp.py)

---

## 9. 阅读顺序建议（附入口行号）

1. Q/KV 与 RoPE：`[Qwen3FlashMTPAttention.forward`（L89）](./specforge/modeling/draft/flashmtp.py) → `[FlashMTPDraftModel.forward`（L257）](./specforge/modeling/draft/flashmtp.py)
2. 训练张量与掩码：`[prepare_target_hidden`（L21）](./specforge/core/flashmtp.py) → `[create_flashmtp_block_mask`（L62）](./specforge/core/flashmtp.py) → `[OnlineFlashMTPModel.forward`（L257）](./specforge/core/flashmtp.py)
3. 训练主循环：`[main` 中训练循环（L470–503）](./scripts/train_flashmtp.py)
4. 推理演示：`[spec_generate`（L285）](./specforge/modeling/draft/flashmtp.py)

---

## 10. 与 README 后续版本的边界

- **v2 / v3（KV cache 扩展、扩散与 consistency loss）** → 以 [README.md](./README.md) 描述为准；当前实现以 `[OnlineFlashMTPModel` 加权 CE（L363–372）](./specforge/core/flashmtp.py) 为主。
- `**chs_concat_mode` 与维度** → `[--chs-concat-mode`（L90）](./scripts/train_flashmtp.py) 与 `[build_models` 写 config（L172–175）](./scripts/train_flashmtp.py) 需与 `[prepare_target_hidden`（L21）](./specforge/core/flashmtp.py) / `[fc` 分支（L241）](./specforge/modeling/draft/flashmtp.py) 一致。

若你希望把某一函数逐行注释成「伪代码级」文档，可以指定文件与函数名再补一节附录。