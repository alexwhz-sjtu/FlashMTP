# FlashMTP dense Sliding-CHS

当前实现使用 dense SWA（dense Sliding-CHS），checkpoint 架构版本为 `sliding_chs_first_token_window_v5`。历史表示支持 `fuse` 和 `token` 两种模式。

## 条件结构

对 anchor `a`、窗口长度 `W` 和当前 CHS 层数 `S`，每个草稿 block 的条件为：

```text
CHS KV:
  fuse:     [CHS_layers(a-1), fuse3(a-W), ..., fuse3(a-2)]
  token:    [CHS_layers(a-1), embed(a-W+1), ..., embed(a-1)]
  pivot_q:  [CHS_layers(a-1)]
Q:
  fuse/token: [embed(a), MASK...]          # 长度 B，块内双向
  pivot_q:    [embed(a-W+1), ..., embed(a-1), embed(a), MASK...]
```

- `fuse`：历史使用 target 首层、中层和末层 hidden，每个位置通过 `Linear(3H,H) + RMSNorm` 融合成一个槽位，作为 context KV。
- `token`：历史直接使用 target token embedding，作为 context KV；最后一个历史 token 和 pivot 的 RoPE position id 均为 `a-1`。
- `pivot_q`：与 `token` 使用相同的 `a-W+1 .. a-1` embedding，但这些槽位作为 draft Q（同时通过 Q 的 K/V 参与块内双向注意力），CHS 仍只作为 context KV。最后一个 window token 与 CHS 仍共享 RoPE id `a-1`。window 与 anchor 均不监督。
- 当前 CHS 只保留 `S` 个均匀选取的 target hidden，全部加层深 embedding，排在 window 之前，并共享 pivot 的 RoPE position id `a-1`（local 模式下使用对应的同一局部位置）。CHS 中不再额外加入 token embedding。
- `token`/`pivot_q` + `local_position` 模式下，window token 从 anchor 反推其全局位置，再以第一个有效 window token 为局部位置 `0`；最后一个 window token 和 CHS hidden 对应同一个 `anchor-1` 局部位置。
- 每个并行 block 只能访问自己的历史、当前 CHS 和 draft token。
- draft block 内使用双向注意力。

默认采用 target 一致的全局 RoPE。启用 `LOCAL_POSITION=true` 时，只对 draft 的窗口使用局部编号；target 的 position id 和完整 KV cache 始终保持全局位置。

## 对齐

训练和推理保持相同的外部对齐：`BLOCK_SIZE=B`，每轮提出 `B-1` 个 draft token，target 验证窗口最大为 `B`。`fuse`/`token` 的 draft query 为 `[embed(anchor=a), MASK...]`；`pivot_q` 在其前面再拼上 window embedding。anchor（以及 `pivot_q` 的 window）位置不监督，预测 `anchor+1..anchor+B-1`。选中的 `S` 个 target hidden states 位于 context 最前面；`fuse`/`token` 的显式 window 紧随其后作为 KV，`pivot_q` 的 window 改为 Q。

## 推理状态

target 保留完整 KV cache。draft 不提供 cache 接口，也不保留投影后的 K/V 或层状态；推理仅滚动保存构造下一轮历史所需的融合 hidden（`fuse`）或 token embedding（`token`/`pivot_q`），每轮重新注入 history 和当前 CHS，并完整计算草稿 block。

## 配置

| 配置 | 默认值 | 含义 |
|---|---:|---|
| `SLIDING_WINDOW_SIZE` | 64 | dense 历史窗口 W，实际历史槽为 W-1 |
| `HISTORY_MODE` | fuse | `fuse` 融合 hidden 作 KV；`token` embedding 作 KV；`pivot_q` embedding 作 Q |
| `CHS_NUM_LAYERS` | 7 | 当前 pivot 保留的 target hidden 层数；CHS 不包含 token embedding |
| `LOCAL_POSITION` | false | draft 是否使用窗口局部 RoPE |
| `BLOCK_SIZE` | 16 | 包含已知 anchor 的草稿 block 宽度 |
| `NUM_DRAFT_LAYERS` | 5 | 草稿 Transformer 层数 |

没有 BWA stride、pivot fusion 或 left-shift 配置。旧配置中的 `history_mode=dense` 会按 `fuse` 兼容加载。

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
