# FlashMTP dense Sliding-CHS 训练

训练入口：`scripts/run_training_flashmtp.sh` → `scripts/train_flashmtp.py`。

```bash
cd /data/wanghanzhen/FlashMTP_v2swa
source .venv/bin/activate
SLIDING_WINDOW_SIZE=1 \
CHS_NUM_LAYERS=12 \
LOCAL_POSITION=true \
HEADPOS=false \
BLOCK_SIZE=8 \
NUM_DRAFT_LAYERS=5 \
NUM_EPOCHS=6 \
NUM_ANCHORS=512 \
MAX_LENGTH=4096 \
BATCH_SIZE=1 \
LOSS_DECAY_GAMMA=4 \
DATA_NUM_SAMPLES=q3_8b_pb_80k \
BASE_LM_CE_DECAY_GAMMA=12 \
LEARNING_RATE=4e-4 \
FINAL_CE_WEIGHT=0.1 \
TV_LOSS_WEIGHT=1.0 \
BASE_LM_CE_WEIGHT=0.0 \
MARKOV_HEAD_TYPE=rnn_easy \
MARKOV_OUTPUT_MODE=direct \
ENABLE_BACKBONE_CONV=true \
BACKBONE_CONV_MODE=simple_conv \
BACKBONE_CONV_KERNEL_SIZE=2 \
BACKBONE_CONV_GROUP_SIZE=16 \
MARKOV_RANK=512 \
TRAIN_DATA_PATH='/data/wanghanzhen/training_data/generated/qwen3-8b/open_perfectblend_80k_qwen3_8b.jsonl' \
MODEL_TAG='Qwen3_8B' \
TARGET_MODEL='/data/wanghanzhen/models/Qwen3-8B' \
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

### 层间卷积 simple_conv

保留原有训练命令的其它参数，设置 `BACKBONE_CONV_MODE=simple_conv`。
可选模式为 `none`、`full`、`simple_conv`；显式模式优先于旧的
`ENABLE_BACKBONE_CONV`。未设置模式时，旧开关行为保持不变。

```bash
BACKBONE_CONV_MODE=simple_conv \
BACKBONE_CONV_KERNEL_SIZE=2 \
BACKBONE_CONV_GROUP_SIZE=16 \
bash scripts/run_training_flashmtp.sh --dt h100
```

直接调用 `scripts/train_flashmtp.py` 时，添加：
`--backbone-conv-mode simple_conv --conv-kernel-size 2 --conv-group-size 16`。
这里的 `--dt` 沿用原有环境配置选择，不根据 GPU 型号自动切换；数据、模型和输出路径
需沿用自己的训练配置。

`simple_conv` 在每个非末层的 MLP 残差相加之后执行一次单侧动态因果卷积，
最后一层不加。5 层时共 4 次卷积，基础核恒等初始化、动态投影零初始化。
模式保存到 checkpoint 的 `flashmtp_config.backbone_conv_mode`，推理自动读取。
旧 checkpoint 未记录模式时，根据 `backbone_conv_enabled` 恢复 none/full。
恢复训练要求卷积模式相同，不能将已有 full checkpoint 直接作为 simple_conv resume。

可复现的合成 backbone forward 测速（不含 target、LM head 和串行 head）：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python \
  scripts/benchmark_simple_conv_forward.py \
  --config /path/to/checkpoint/config.json \
  --output benchmark_results/simple_conv_forward.json
```
