# FlashMTP dense Sliding-CHS

当前实现使用固定的 pivot-Q dense SWA（dense Sliding-CHS），checkpoint 架构版本为 `sliding_chs_first_token_window_v5`。

## 条件结构

对 anchor `a`、窗口长度 `W` 和当前 CHS 层数 `S`，每个草稿 block 的条件为：

```text
CHS KV: [CHS_layers(a-1)]
Q: [embed(a-W+1), ..., embed(a-1), embed(a), MASK...]
```

- 历史使用 `a-W+1 .. a-1` 的 token embedding 作为 draft Q，同时通过 Q 的 K/V 投影参与块内双向注意力；CHS 只作为 context KV。最后一个 window token 与 CHS 共享 RoPE id `a-1`。window 与 anchor 均不监督。
- 当前 CHS 只保留 `S` 个均匀选取的 target hidden，全部加层深 embedding，排在 window 之前，并共享 pivot 的 RoPE position id `a-1`（local 模式下使用对应的同一局部位置）。CHS 中不再额外加入 token embedding。
- `local_position` 下，window token 从 anchor 反推其全局位置，再以第一个有效 window token 为局部位置 `0`；最后一个 window token 和 CHS hidden 对应同一个 `anchor-1` 局部位置。
- 每个并行 block 只能访问自己的历史、当前 CHS 和 draft token。
- draft block 内使用双向注意力。

默认采用 target 一致的全局 RoPE。启用 `LOCAL_POSITION=true` 时，只对 draft 的窗口使用局部编号；target 的 position id 和完整 KV cache 始终保持全局位置。

## 对齐

训练和推理保持相同的外部对齐：`BLOCK_SIZE=B`，每轮提出 `B-1` 个 draft token，target 验证窗口最大为 `B`。draft query 在 `[embed(anchor=a), MASK...]` 前拼接 window embedding；window 与 anchor 不监督，预测 `anchor+1..anchor+B-1`。选中的 `S` 个 target hidden states 作为 context KV。

## 推理状态

target 保留完整 KV cache。draft 不提供 cache 接口，也不保留投影后的 K/V 或层状态；推理仅滚动保存下一轮所需的 token embedding，每轮重新注入 window 和当前 CHS，并完整计算草稿 block。

## 配置

| 配置 | 默认值 | 含义 |
|---|---:|---|
| `SLIDING_WINDOW_SIZE` | 64 | dense 历史窗口 W，实际历史槽为 W-1 |
| `CHS_NUM_LAYERS` | 7 | 当前 pivot 保留的 target hidden 层数；CHS 不包含 token embedding |
| `LOCAL_POSITION` | false | draft 是否使用窗口局部 RoPE |
| `BLOCK_SIZE` | 16 | 包含已知 anchor 的草稿 block 宽度 |
| `NUM_DRAFT_LAYERS` | 5 | 草稿 Transformer 层数 |

没有 HISTORY_MODE、BWA stride、pivot fusion 或 left-shift 配置；布局固定为 pivot-Q。

## 串行 head

并行 backbone 后可选 `vanilla`、`rnn` 或 `rnn_easy` 低秩串行 head。训练使用 teacher forcing，推理按 block 内位置依次采样。输出支持：

- `additive`：串行 head logits 加到 base LM-head logits。
- `direct`：低秩 head 直接产生最终 logits，跳过 base LM head。

当 `SLIDING_WINDOW_SIZE > 1` 且串行 head 为 `rnn` / `rnn_easy` 时，会用 `embed(anchor-1)` 先更新一次 recurrent state，再开始预测第一个 draft token。这样预测第一个 MASK 时，state 里已经同时编码了 `anchor-1` 与 `anchor` 两个位置的信息。训练与推理路径一致。

训练 loss 为：

```text
FINAL_CE_WEIGHT * final CE
+ TV_LOSS_WEIGHT * target/draft distribution L1
+ BASE_LM_CE_WEIGHT * optional base-head CE
```
