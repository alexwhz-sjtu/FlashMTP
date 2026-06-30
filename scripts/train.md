# FlashMTP 训练启动指南

本文档由 `scripts/train.ipynb` 整理而来，汇总 FlashMTP v1.1 的训练启动命令与常用工作流。

训练入口脚本：`scripts/run_training_flashmtp.sh` → `scripts/train_flashmtp.py`

---

## 融合模式说明

`PIVOT_FUSE_MODE` 可选值：


| 模式                 | 说明                                 |
| ------------------ | ---------------------------------- |
| `linear_fuse`      | 选定层沿特征维度拼接，Linear 定权融合成 pivot      |
| `attention_fuse`   | Attention 提取信息，最后一层 attend 之前层     |
| `prefix_condition` | 将 HS 拼接在 anchor 和 mask 输入序列最前面（推荐） |


---

## 单机训练

适用于本地 / 单节点 H100 等环境，快速验证配置：

```bash
# ["linear_fuse", "attention_fuse", "prefix_condition"]

source .venv/bin/activate
NUM_MIDDLE_LAYERS_N=16 NUM_EPOCHS=6 PIVOT_FUSE_MODE=prefix_condition DATA_NUM_SAMPLES=40000 \
BLOCK_SIZE=16 LOSS_DECAY_GAMMA=7 LOCAL_POSITION=true CAUSAL_MODE=true NUM_MIDDLE_LAYERS=5 \
bash scripts/run_training_flashmtp.sh --dt h100
```

**主要参数：**

- `NUM_MIDDLE_LAYERS_N=16`：中间等间隔选取的 target 层数 N
- `PIVOT_FUSE_MODE=prefix_condition`：pivot 融合方式
- `DATA_NUM_SAMPLES=40000`：训练样本数
- `BLOCK_SIZE=16`：草稿块大小
- `LOSS_DECAY_GAMMA=7`：块内 CE 衰减系数
- `LOCAL_POSITION=true`：草稿侧使用块内局部位置编码
- `W1_MSE=0.1`：MSE 辅助损失权重
- `--dt h100`：设备类型（可选 `qz` / `a800` / `h100`）

---

## 多机训练

### Qwen 模板（900k 混合数据）

```bash
cd /inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/FlashMTP_v1.1 && \
mkdir -p whz_mtp_logs && \
export NUM_DRAFT_LAYERS=5 NUM_EPOCHS=8 PIVOT_FUSE_MODE=prefix_condition DATA_NUM_SAMPLES=40000 CHAT_TEMPLATE=qwen && \
export NUM_MIDDLE_LAYERS_N=16 LOCAL_POSITION=true BLOCK_SIZE=12 LOSS_DECAY_GAMMA=8 && \
export TRAIN_DATA_PATH='/workspace/wanghanzhen/NIPS26/training_data/regen_data/nemotron_40000/nemotron_think_off_samples_40000_qwen3_8b_regen.jsonl' && \
bash scripts/run_training_flashmtp.sh --dt qz \
  > "whz_mtp_logs/train_flashmtp_qz_dist_$(date +%Y%m%d_%H%M%S).log" 2>&1
```

### Llama3 模板（532k ShareGPT + UltraChat）

```bash
cd /inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/FlashMTP_v1.1 && \
mkdir -p whz_mtp_logs && \
export NUM_DRAFT_LAYERS=5 NUM_EPOCHS=8 PIVOT_FUSE_MODE=prefix_condition DATA_NUM_SAMPLES=532000 CHAT_TEMPLATE=llama3 && \
export NUM_MIDDLE_LAYERS_N=16 LOCAL_POSITION=true BLOCK_SIZE=12 LOSS_DECAY_GAMMA=6 MAX_LENGTH=4096 && \
export CACHE_DIR=/data/wanghanzhen/Projects/MTP/NIPS26/FlashMTP_v1.1/cache/data/regen_data/sharegpt_ultrachat_first &&\
export TARGET_MODEL=/data/wanghanzhen/models/Llama-3.1-8B-Instruct &&\
export TRAIN_DATA_PATH='/data/wanghanzhen/Projects/MTP/NIPS26/FlashMTP/cache/data/regen_data/sharegpt_ultrachat/sharegpt_ultrachat_think_off_samples_532k_llama3.1_8b_regen.jsonl' &&\
bash scripts/run_training_flashmtp.sh --dt qz
```

---

## 数据混合与打乱

将新数据与旧 replay 数据混合并 shuffle，用于继续训练场景：

```bash
cd /data/wanghanzhen/Projects/MTP/NIPS26/FlashMTP_v1.1

NEW_DATA=/data/wanghanzhen/Projects/MTP/NIPS26/training_data/regen_data/math_code/math_code_aug_off_samples_1w_qwen3_8b_regen.jsonl
OLD_DATA=/data/wanghanzhen/Projects/MTP/NIPS26/training_data/regen_data/tmp/old_replay_50k.jsonl
MIX_DIR=/data/wanghanzhen/Projects/MTP/NIPS26/training_data/regen_data/math_code_continue_mix

mkdir -p ${MIX_DIR}
shuf -n 50000 ${OLD_DATA} > ${MIX_DIR}/old_replay_50k.jsonl
cat ${NEW_DATA} ${MIX_DIR}/old_replay_50k.jsonl | shuf > ${MIX_DIR}/math_code_1w_old_replay_50k.jsonl
```

**流程：**

1. 从旧 replay 池中随机抽取 50k 条
2. 与新数据拼接
3. 整体 shuffle 后写入混合文件

---

## 继续训练（从 checkpoint 恢复）

从已有 checkpoint 继续训练 math_code 数据：

```bash
cd /data/wanghanzhen/Projects/MTP/NIPS26/FlashMTP_v1.1

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
NPROC_PER_NODE=6 \
TARGET_MODEL_PATH=/data/wanghanzhen/models/Qwen/Qwen3-8B \
CKPT_DIR=/data/wanghanzhen/Projects/MTP/NIPS26/FlashMTP_v1.1/cache/models/flashmtp_qz_prefix_condition_fuse16_feature_sample_2350325_think_off_nlayers5_block_16_ep5 \
TRAIN_DATA_PATH=/data/wanghanzhen/Projects/MTP/NIPS26/training_data/regen_data/math_code/math_code_aug_off_samples_1w_qwen3_8b_regen.jsonl \
OUTPUT_DIR=./cache/models/flashmtp_continue_math_code_1w_replay2w_from_2350325 \
CACHE_DIR=./cache/data/regen_data/math_code_continue_mix_3w_maxlen20480 \
NUM_EPOCHS=8 \
LEARNING_RATE=3e-5 \
WARMUP_RATIO=0.03 \
MAX_GRAD_NORM=1.0 \
MAX_LENGTH=4096 \
NUM_ANCHORS=768 \
NUM_DRAFT_LAYERS=5 \
BLOCK_SIZE=16 \
PIVOT_FUSE_MODE=prefix_condition \
NUM_MIDDLE_LAYERS_N=16 \
CHS_CONCAT_MODE=feature \
LOCAL_POSITION=true \
TRAIN_LM_HEAD=false \
LOSS_DECAY_GAMMA=7 \
SAVE_INTERVAL=5000 \
LOG_INTERVAL=20 \
WANDB_RUN_ID=flashmtp_continue_math_code_1w_ep6 \
WANDB_RUN_NAME=flashmtp_continue_math_code_1w_ep6 \
bash scripts/run_training_flashmtp.sh --dt h100
```

**继续训练关键参数：**


| 参数                                        | 说明                 |
| ----------------------------------------- | ------------------ |
| `CKPT_DIR`                                | 待恢复的 checkpoint 目录 |
| `TARGET_MODEL_PATH`                       | 目标模型路径             |
| `OUTPUT_DIR`                              | 新 run 输出目录         |
| `TRAIN_DATA_PATH`                         | 训练数据 jsonl         |
| `CACHE_DIR`                               | 预处理缓存目录            |
| `CUDA_VISIBLE_DEVICES` / `NPROC_PER_NODE` | GPU 与进程数           |
| `WANDB_RUN_ID` / `WANDB_RUN_NAME`         | WandB 实验标识         |


---

## 常用环境变量速查


| 变量                    | 默认值（脚本内）      | 说明                          |
| --------------------- | ------------- | --------------------------- |
| `NUM_EPOCHS`          | 6             | 训练 epoch 数                  |
| `MAX_LENGTH`          | 4096          | 最大序列长度                      |
| `NUM_DRAFT_LAYERS`    | 5             | 草稿模型层数                      |
| `NUM_MIDDLE_LAYERS_N` | 5             | target 中间选取层数               |
| `BLOCK_SIZE`          | —             | 草稿块大小                       |
| `PIVOT_FUSE_MODE`     | `linear_fuse` | pivot 融合模式                  |
| `CHS_CONCAT_MODE`     | `feature`     | CHS 拼接模式                    |
| `LOCAL_POSITION`      | false         | 块内局部位置编码                    |
| `TRAIN_LM_HEAD`       | false         | 是否单独训练草稿 lm_head            |
| `LOSS_DECAY_GAMMA`    | —             | 块内 CE 衰减系数                  |
| `CHAT_TEMPLATE`       | —             | 对话模板（`qwen` / `llama3`）     |
| `DATA_NUM_SAMPLES`    | 40000         | 训练样本数                       |
| `--dt`                | a800          | 设备类型：`qz` / `a800` / `h100` |


更多参数说明见 `scripts/run_training_flashmtp.sh` 与项目根目录 `v1.1.md`。