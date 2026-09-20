# FlashMTP dense Sliding-CHS 训练

训练入口：`scripts/run_training_flashmtp.sh` → `scripts/train_flashmtp.py`。

```bash
cd /data/wanghanzhen/FlashMTP_v2swa
source .venv/bin/activate
SLIDING_WINDOW_SIZE=512 \
CHS_NUM_LAYERS=12 \
LOCAL_POSITION=false \
HEADPOS=false \
HISTORY_MODE=fuse \
BLOCK_SIZE=8 \
NUM_DRAFT_LAYERS=5 \
NUM_EPOCHS=6 \
NUM_ANCHORS=512 \
MAX_LENGTH=4096 \
BATCH_SIZE=1 \
LOSS_DECAY_GAMMA=4 \
DATA_NUM_SAMPLES=pb_80k \
BASE_LM_CE_DECAY_GAMMA=12 \
LEARNING_RATE=4e-4 \
FINAL_CE_WEIGHT=0.1 \
TV_LOSS_WEIGHT=1.0 \
BASE_LM_CE_WEIGHT=0.06 \
MARKOV_HEAD_TYPE=rnn_easy \
MARKOV_OUTPUT_MODE=direct \
MARKOV_RANK=320 \
TRAIN_DATA_PATH='/data/wanghanzhen/training_data/generated/qwen3-4b/open_perfectblend_80k_qwen3_4b.jsonl' \
MODEL_TAG='Qwen3_4B' \
TARGET_MODEL='/data/wanghanzhen/models/Qwen3-4B' \
bash scripts/run_training_flashmtp.sh --dt h100
```



## Dense SWA 参数


| 环境变量                  | 默认值   | 说明                                                                           |
| --------------------- | ----- | ---------------------------------------------------------------------------- |
| `SLIDING_WINDOW_SIZE` | 64    | dense 窗口 W，使用 anchor 前 W-1 个连续位置                                             |
| `HISTORY_MODE`        | fuse  | `fuse` 使用融合 hidden；`token` 使用 token embedding                                |
| `CHS_NUM_LAYERS`      | 7     | pivot 保留的 target hidden 层数；pivot token embedding 作为 query                    |
| `LOCAL_POSITION`      | false | draft 使用局部或全局 RoPE                                                           |
| `BLOCK_SIZE`          | 16    | draft Q 为已知 anchor + B-1 个 MASK；pivot embedding 位于 CHS 首位，实际 proposal 数为 B-1 |
| `NUM_DRAFT_LAYERS`    | 5     | 草稿 Transformer 层数                                                            |
| `NUM_ANCHORS`         | 512   | 每条训练序列最多采样的 anchor 数                                                         |


窗口布局固定为 dense；历史表示由 `HISTORY_MODE=fuse|token` 控制。`token` 模式下最后一个历史 token 与 pivot 共用 `anchor-1` 的 RoPE position id。

## 串行 head 与 loss


| 环境变量                 | 可选值/含义                                            |
| -------------------- | ------------------------------------------------- |
| `MARKOV_HEAD_TYPE`   | `none` / `vanilla` / `gated` / `rnn` / `rnn_easy` |
| `MARKOV_OUTPUT_MODE` | `additive` / `direct`                             |
| `MARKOV_RANK`        | 低秩 state/embedding 维度                             |
| `HEADPOS`            | `true` 启用相对位置 embedding；默认 `false`                |
| `FINAL_CE_WEIGHT`    | 最终预测 CE 权重                                        |
| `TV_LOSS_WEIGHT`     | target/draft 分布 L1 权重                             |
| `BASE_LM_CE_WEIGHT`  | 可选 base LM-head CE 权重                             |


`HEADPOS=true` 时串行 head 使用 block 内相对位置 embedding。`vanilla` / `gated + additive` 在
previous-token embedding 处注入，`rnn/rnn_easy + direct` 在降维后的 hidden
latent 处注入；训练与推理均使用相同的 `0..block_size-2` slot 编号。设为
`false` 时保持原始无位置 embedding 的行为。

训练时 target 冻结，只捕获 dense 历史所需的首层、中层、末层，以及当前 CHS 和 TV loss 所需层。draft 不使用 KV cache。