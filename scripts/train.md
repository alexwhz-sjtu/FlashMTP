# FlashMTP dense Sliding-CHS 训练

训练入口：`scripts/run_training_flashmtp.sh` → `scripts/train_flashmtp.py`。

```bash
cd /share/dai-sys/wanghanzhen/projects/MTP/FlashMTP_v2swa
source .venv/bin/activate
SLIDING_WINDOW_SIZE=9 \
CHS_NUM_LAYERS=12 \
LOCAL_POSITION=true \
CE_CHUNK_SIZE=4096 \
BLOCK_SIZE=8 \
NUM_DRAFT_LAYERS=5 \
NUM_EPOCHS=8 \
NUM_ANCHORS=512 \
MAX_LENGTH=4096 \
BATCH_SIZE=1 \
LOSS_DECAY_GAMMA=4 \
DATA_NUM_SAMPLES=pb_80k \
BASE_LM_CE_DECAY_GAMMA=12 \
LEARNING_RATE=5e-4 \
FINAL_CE_WEIGHT=0.1 \
TV_LOSS_WEIGHT=1.0 \
BASE_LM_CE_WEIGHT=0.06 \
MARKOV_HEAD_TYPE=vanilla \
MARKOV_OUTPUT_MODE=additive \
MARKOV_RANK=256 \
TRAIN_DATA_PATH='/share/dai-sys/wanghanzhen/projects/MTP/training_data/open_perfectblend_80k_qwen3_8b.jsonl' \
MODEL_TAG='Qwen3_8B' \
TARGET_MODEL=/share/dai-sys/wanghanzhen/models/Qwen/Qwen3-8B \
bash scripts/run_training_flashmtp.sh --dt h100
```

## Dense SWA 参数


| 环境变量                  | 默认值   | 说明                                                                            |
| --------------------- | ----- | ----------------------------------------------------------------------------- |
| `SLIDING_WINDOW_SIZE` | 64    | dense 窗口 W，使用 anchor 前 W-1 个连续位置                                              |
| `CHS_NUM_LAYERS`      | 7     | pivot 保留的 target hidden 层数；CHS 不含 token embedding，排在 window 前                 |
| `LOCAL_POSITION`      | false | draft 使用局部或全局 RoPE                                                            |
| `BLOCK_SIZE`          | 16    | draft Q 为已知 anchor + B-1 个 MASK；pivot embedding 位于 CHS 首位，实际 proposal 数为 B-1  |
| `NUM_DRAFT_LAYERS`    | 5     | 草稿 Transformer 层数                                                             |
| `NUM_ANCHORS`         | 512   | 每条训练序列最多采样的 anchor 数                                                          |


窗口布局固定为 pivot-Q dense SWA：context 只保留 CHS，window embedding 拼到 draft Q 前面：`[embed(a-W+1)..embed(a-1), embed(a), MASK...]`。最后一个 window token 与 CHS hidden 共用 `anchor-1` 的 RoPE position id；local 模式中第一个有效 window token 的 position id 为 0。

## 串行 head 与 loss


| 环境变量                 | 可选值/含义                                            |
| -------------------- | ------------------------------------------------- |
| `MARKOV_HEAD_TYPE`   | `none` / `vanilla` / `gated` / `rnn` / `rnn_easy` |
| `MARKOV_OUTPUT_MODE` | `additive` / `direct`                             |
| `MARKOV_RANK`        | 低秩 state/embedding 维度                             |
| `FINAL_CE_WEIGHT`    | 最终预测 CE 权重                                        |
| `TV_LOSS_WEIGHT`     | target/draft 分布 L1 权重                             |
| `BASE_LM_CE_WEIGHT`  | 可选 base LM-head CE 权重                             |


`SLIDING_WINDOW_SIZE > 1` 且串行 head 为 `rnn` / `rnn_easy` 时，会先用 `embed(anchor-1)` 初始化 recurrent state，再预测第一个 draft token。

训练时 target 冻结，只捕获当前 CHS 所需层；TV loss 直接复用 target prefill logits。draft 不使用 KV cache。
