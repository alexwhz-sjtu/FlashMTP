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
- Streak 仍然使用 LS-RSL 作为主目标，并可用 CE 辅助项。
- 两者都不再只传单点 stacked hidden，而是传完整目标 hidden history 和 attention mask。

这样 MDLM 和 Streak 共享同一个 v5 draft 条件路径，避免两阶段训练学到不同的信息接口。

## 推理一致性

推理侧也同步维护完整历史 hidden states：

- prefill 后保留 prompt 的完整 target hidden states。
- 每次验证一个 speculative block 后，只追加 seed token 和已接受 draft token 的 hidden。
- 被拒绝后由 target 采样出的下一个 token 尚未经过 target forward，因此不追加它的 hidden。

下一轮 draft 的 pivot 仍然对应当前 anchor 的前一位置。这样 cross-attention 看到的历史范围与训练中的 `0..a-2` 对齐。

## 当前验证状态

已完成基础可运行性检查：

- 修改后的核心 Python 文件通过 `py_compile`。
- IDE lint 未报告新增错误。
- 使用 `run_v3_3_streak.sh --dt a800` 做 smoke training，已进入 `Streak ep0` 并连续输出多步 loss，说明 v5 Streak 路径能完成实际 forward/backward。

后续建议先按 v5 从头跑 MDLM，再用 v5 MDLM checkpoint 初始化 Streak；旧 v3.3 checkpoint 的 fusion 模块结构不同，不建议直接 strict load。
