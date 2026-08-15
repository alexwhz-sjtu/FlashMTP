## FlashMTP 训练启动指南

本文档由 `scripts/train.ipynb` 整理而来，汇总 FlashMTP v1.1 的训练启动命令与常用工作流。

训练入口脚本：`scripts/run_training_flashmtp.sh` → `scripts/train_flashmtp.py`

## 融合模式说明

`PIVOT_FUSE_MODE` 可选值：


| 模式                 | 说明                                 |
| ------------------ | ---------------------------------- |
| `linear_fuse`      | 选定层沿特征维度拼接，Linear 定权融合成 pivot      |
| `attention_fuse`   | Attention 提取信息，最后一层 attend 之前层     |
| `prefix_condition` | 将 HS 拼接在 anchor 和 mask 输入序列最前面（推荐） |


## 串行 Head

训练脚本支持在并行 FlashMTP backbone 后增加低秩串行 head：


| 环境变量                 | 可选值                                                       | 默认值        |
| -------------------- | --------------------------------------------------------- | ---------- |
| `MARKOV_HEAD_TYPE`   | `none` / `vanilla` / `gated` / `rnn` / `rnn_easy` / `mlp` | `none`     |
| `MARKOV_OUTPUT_MODE` | `additive` / `direct`                                     | `additive` |
| `MARKOV_RANK`        | 正整数                                                       | `256`      |
| `FINAL_CE_WEIGHT`    | 最终 CE loss 权重                                             | `1.0`      |
| `TV_LOSS_WEIGHT`     | 串行 head TV loss 权重                                        | `1.0`      |


`additive` 将 head 输出作为 logit bias 加到并行 base logits；`direct`  
直接将 head 输出作为最终 logits。训练使用真实前驱 token 做 teacher forcing，  
推理时按块内位置串行采样。启用串行 head 时，总 loss 包含  
`FINAL_CE_WEIGHT * CE + TV_LOSS_WEIGHT * TV`；TV 是串行最终分布与目标模型  
对应 causal 位置分布之间的词表维 L1 距离，不乘 `1/2`，位置权重复用  
`LOSS_DECAY_GAMMA`，最后按有效位置数平均。

示例：

```plaintext
MARKOV_HEAD_TYPE=rnn MARKOV_OUTPUT_MODE=additive MARKOV_RANK=256 \
FINAL_CE_WEIGHT=1.0 TV_LOSS_WEIGHT=1.0 \
bash scripts/run_training_flashmtp.sh --dt h100
```



## 单机训练

适用于本地 / 单节点 H100 等环境，快速验证配置：

```plaintext
# ["linear_fuse", "attention_fuse", "prefix_condition"]
cd /share/dai-sys/wanghanzhen/projects/MTP/FlashMTP_v2
source .venv/bin/activate
NUM_MIDDLE_LAYERS_N=16 NUM_DRAFT_LAYERS=5 NUM_EPOCHS=8 PIVOT_FUSE_MODE=prefix_condition DATA_NUM_SAMPLES=pb_80k MAX_LENGTH=4096 NUM_ANCHORS=512 BLOCK_SIZE=8 LOCAL_POSITION=true \
LOSS_DECAY_GAMMA=4 BASE_LM_CE_DECAY_GAMMA=12 BASE_LM_CE_WEIGHT=0.06 FINAL_CE_WEIGHT=0.1 TV_LOSS_WEIGHT=1.0 \
MARKOV_HEAD_TYPE=rnn_easy MARKOV_OUTPUT_MODE=additive MARKOV_RANK=256 \
NPROC_PER_NODE=8 TP_SIZE=1 SHARD_DRAFT_BY_TP=1 CE_CHUNK_SIZE=4096 \
LEARNING_RATE=5e-4 \
TRAIN_DATA_PATH="/share/dai-sys/wanghanzhen/projects/MTP/training_data/open_perfectblend_80k_qwen3_8b.jsonl" \
TARGET_MODEL_BACKEND=sglang SGLANG_MEM_FRACTION_STATIC=0.3 \
TARGET_MODEL=/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/models/Qwen/Qwen3-8B \
MODEL_TAG='Qwen3-8B' \
bash scripts/run_training_flashmtp.sh --dt h100
```

预测左移一位

```plaintext
cd /inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/FlashMTP_v2
source .venv/bin/activate
export FLASHINFER_CACHE_DIR=/root/.cache/flashinfer_$(hostname)
LEFT_SHIFT=true \
NUM_MIDDLE_LAYERS_N=16 NUM_DRAFT_LAYERS=5 NUM_EPOCHS=6 PIVOT_FUSE_MODE=prefix_condition DATA_NUM_SAMPLES=2360k_aug3 MAX_LENGTH=30720 NUM_ANCHORS=768 BLOCK_SIZE=8 LOCAL_POSITION=true \
LOSS_DECAY_GAMMA=4 BASE_LM_CE_DECAY_GAMMA=12 BASE_LM_CE_WEIGHT=0.06 FINAL_CE_WEIGHT=0.1 TV_LOSS_WEIGHT=1.0 \
MARKOV_HEAD_TYPE=vanilla MARKOV_OUTPUT_MODE=additive MARKOV_RANK=256 \
NPROC_PER_NODE=8 TP_SIZE=1 SHARD_DRAFT_BY_TP=1 CE_CHUNK_SIZE=6144 \
TRAIN_DATA_PATH="/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/FlashMTP/cache/data/regen_data/qwen3_8b/mixed_2360k_qwen3_8b_nm_pb_swe_aug3.jsonl" \
TARGET_MODEL_BACKEND=sglang SGLANG_MEM_FRACTION_STATIC=0.25 \
TARGET_MODEL=/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/models/Qwen/Qwen3-8B \
MODEL_TAG='Qwen3-8B' \
bash scripts/run_training_flashmtp.sh --dt qz
```

Qwen-14B

```plaintext
# ["linear_fuse", "attention_fuse", "prefix_condition"]
cd /share/dai-sys/wanghanzhen/projects/MTP/FlashMTP_v1.1
source .venv/bin/activate
NUM_MIDDLE_LAYERS_N=16 NUM_DRAFT_LAYERS=5 NUM_EPOCHS=8 PIVOT_FUSE_MODE=prefix_condition DATA_NUM_SAMPLES=40000 MAX_LENGTH=40960 NUM_ANCHORS=1024 BLOCK_SIZE=8 SHARD_DRAFT_BY_TP=2 \
NPROC_PER_NODE=8 TP_SIZE=2 CE_CHUNK_SIZE=8192 \
TRAIN_DATA_PATH="/share/dai-sys/wanghanzhen/projects/MTP/training_data/nemotron_think_off_samples_40000_qwen3_8b_regen.jsonl" \
TARGET_MODEL_BACKEND=sglang SGLANG_MEM_FRACTION_STATIC=0.25 \
TARGET_MODEL=/share/dai-sys/wanghanzhen/models/Qwen/Qwen3-14B \
MODEL_TAG='Qwen3-14B' \
LOSS_DECAY_GAMMA=7 LOCAL_POSITION=true NUM_MIDDLE_LAYERS=5 \
bash scripts/run_training_flashmtp.sh --dt h100
```

**主要参数：**

- `NUM_MIDDLE_LAYERS_N=16`：中间等间隔选取的 target 层数 N
- `PIVOT_FUSE_MODE=prefix_condition`：pivot 融合方式
- `DATA_NUM_SAMPLES=40000`：训练样本数
- `BLOCK_SIZE=16`：草稿块大小
- `LOSS_DECAY_GAMMA=7`：块内 CE 衰减系数
- `LOCAL_POSITION=true`：草稿侧使用块内局部位置编码
- `--dt h100`：设备类型（可选 `qz` / `a800` / `h100`）



## 数据混合与打乱

将新数据与旧 replay 数据混合并 shuffle，用于继续训练场景：

```plaintext
cd /data/wanghanzhen/Projects/MTP/NIPS26/FlashMTP_v1.1

NEW_DATA=/data/wanghanzhen/Projects/MTP/NIPS26/training_data/regen_data/math_code/math_code_aug_off_samples_1w_qwen3_8b_regen.jsonl
OLD_DATA=/data/wanghanzhen/Projects/MTP/NIPS26/training_data/regen_data/tmp/old_replay_50k.jsonl
MIX_DIR=/data/wanghanzhen/Projects/MTP/NIPS26/training_data/regen_data/math_code_continue_mix

mkdir -p ${MIX_DIR}
shuf -n 50000 ${OLD_DATA} > ${MIX_DIR}/old_replay_50k.jsonl
cat ${NEW_DATA} ${MIX_DIR}/old_replay_50k.jsonl | shuf > ${MIX_DIR}/math_code_1w_old_replay_50k.jsonl
```



### 多机

```bash
pkill -9 python3
# A. 释放 GPU
/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/stop_keeper.sh

cd /inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/FlashMTP_v2
source .venv/bin/activate
export FLASHINFER_CACHE_DIR=/root/.cache/flashinfer_$(hostname)
LEFT_SHIFT=false \
NUM_MIDDLE_LAYERS_N=16 NUM_DRAFT_LAYERS=5 NUM_EPOCHS=10 PIVOT_FUSE_MODE=prefix_condition DATA_NUM_SAMPLES=2360k_aug3_qwen3_8b MAX_LENGTH=30720 NUM_ANCHORS=768 BLOCK_SIZE=8 LOCAL_POSITION=true \
LOSS_DECAY_GAMMA=4 BASE_LM_CE_DECAY_GAMMA=12 BASE_LM_CE_WEIGHT=0.06 FINAL_CE_WEIGHT=0.1 TV_LOSS_WEIGHT=1.0 \
MARKOV_HEAD_TYPE=gates MARKOV_OUTPUT_MODE=additive MARKOV_RANK=512 \
LEARNING_RATE=5e-4 \
NPROC_PER_NODE=8 TP_SIZE=1 SHARD_DRAFT_BY_TP=1 CE_CHUNK_SIZE=6144 \
TRAIN_DATA_PATH="/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/FlashMTP/cache/data/regen_data/qwen3_8b/mixed_2360k_qwen3_8b_nm_pb_swe_aug3.jsonl" \
TARGET_MODEL_BACKEND=sglang SGLANG_MEM_FRACTION_STATIC=0.3 \
TARGET_MODEL=/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/models/Qwen/Qwen3-8B \
MODEL_TAG='Qwen3-8B' \
bash scripts/run_training_flashmtp.sh --dt qz > "whz_mtp_logs/train_flashmtp_qz_dist_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

# 训练结束后恢复 GPU
/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/start_keeper.sh
```



## 常用环境变量速查


| 变量                       | 默认值（脚本内）      | 说明                                    |
| ------------------------ | ------------- | ------------------------------------- |
| `NUM_EPOCHS`             | 6             | 训练 epoch 数                            |
| `MAX_LENGTH`             | 4096          | 最大序列长度                                |
| `NUM_DRAFT_LAYERS`       | 5             | 草稿模型层数                                |
| `NUM_MIDDLE_LAYERS_N`    | 5             | target 中间选取层数                         |
| `BLOCK_SIZE`             | —             | 草稿块大小                                 |
| `PIVOT_FUSE_MODE`        | `linear_fuse` | pivot 融合模式                            |
| `CHS_CONCAT_MODE`        | `feature`     | CHS 拼接模式                              |
| `LOCAL_POSITION`         | false         | 块内局部位置编码                              |
| `LOSS_DECAY_GAMMA`       | —             | 最终 CE 块内衰减系数                          |
| `BASE_LM_CE_WEIGHT`      | 0             | 骨干 hidden 经 target lmhead 的辅助 CE 权重 λ |
| `BASE_LM_CE_DECAY_GAMMA` | —             | 辅助 CE 独立衰减系数（不设则均匀权重）                 |
| `CHAT_TEMPLATE`          | —             | 对话模板（`qwen` / `llama3`）               |
| `DATA_NUM_SAMPLES`       | 40000         | 训练样本数                                 |
| `--dt`                   | a800          | 设备类型：`qz` / `a800` / `h100`           |


更多参数说明见 `scripts/run_training_flashmtp.sh` 与项目根目录 `v1.1.md`。