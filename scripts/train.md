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

## `LEFT_SHIFT`（默认关闭）

| 环境变量 | 默认值 | 含义 |
| --- | --- | --- |
| `LEFT_SHIFT` | `false` | 关闭 DeepSpec 风格左移监督 |

- **legacy（默认，`LEFT_SHIFT=false`）**：`BLOCK_SIZE` 是 draft block 宽度；slot 0 只做 anchor 上下文，监督 slot `1..B-1`。旧 checkpoint 若 config 里没有 `left_shift` 字段，推理也按 legacy 处理。
- **left_shift（`LEFT_SHIFT=true`）**：`BLOCK_SIZE` 是总跨度（anchor + `B-1` 个 draft）；draft 实际只有 `B-1` 个并行 slot，监督 anchor+1..anchor+(B-1)。

推理时 `left_shift` 从 checkpoint 的 `flashmtp_config` 读取，不需要手动传参；benchmark 日志会打印 `Block alignment: legacy` 或 `left_shift`。

---

## 串行 Head

训练脚本支持在并行 FlashMTP backbone 后增加低秩串行 head：


| 环境变量                 | 可选值                                     | 默认值        |
| -------------------- | --------------------------------------- | ---------- |
| `MARKOV_HEAD_TYPE`   | `none` / `vanilla` / `rnn` / `rnn_easy` | `none`     |
| `MARKOV_OUTPUT_MODE` | `additive` / `direct`                   | `additive` |
| `MARKOV_RANK`        | 正整数                                     | `256`      |
| `FINAL_CE_WEIGHT`    | 最终 CE loss 权重                           | `1.0`      |
| `TV_LOSS_WEIGHT`     | 串行 head TV loss 权重                      | `1.0`      |


`additive` 将 head 输出作为 logit bias 加到并行 base logits；`direct`
直接将 head 输出作为最终 logits。训练使用真实前驱 token 做 teacher forcing，
推理时按块内位置串行采样。启用串行 head 时，总 loss 包含
`FINAL_CE_WEIGHT * CE + TV_LOSS_WEIGHT * TV`；TV 是串行最终分布与目标模型
对应 causal 位置分布之间的词表维 L1 距离，不乘 `1/2`，位置权重复用
`LOSS_DECAY_GAMMA`，最后按有效位置数平均。

示例：

```bash
MARKOV_HEAD_TYPE=rnn MARKOV_OUTPUT_MODE=additive MARKOV_RANK=256 \
FINAL_CE_WEIGHT=1.0 TV_LOSS_WEIGHT=1.0 \
bash scripts/run_training_flashmtp.sh --dt h100
```

---



## 单机训练

适用于本地 / 单节点 H100 等环境，快速验证配置：

```bash
# ["linear_fuse", "attention_fuse", "prefix_condition"]
cd /share/dai-sys/wanghanzhen/projects/MTP/FlashMTP_v2
source .venv/bin/activate
LEFT_SHIFT=false \
NUM_MIDDLE_LAYERS_N=16 NUM_DRAFT_LAYERS=5 NUM_EPOCHS=6 PIVOT_FUSE_MODE=prefix_condition DATA_NUM_SAMPLES=pb_80k MAX_LENGTH=4096 NUM_ANCHORS=512 BLOCK_SIZE=8 LOCAL_POSITION=true \
LOSS_DECAY_GAMMA=4 BASE_LM_CE_DECAY_GAMMA=16 BASE_LM_CE_WEIGHT=0.06 FINAL_CE_WEIGHT=0.1 TV_LOSS_WEIGHT=1.0 \
MARKOV_HEAD_TYPE=rnn_easy MARKOV_OUTPUT_MODE=direct MARKOV_RANK=512 \
NPROC_PER_NODE=8 TP_SIZE=1 SHARD_DRAFT_BY_TP=1 CE_CHUNK_SIZE=8192 \
TRAIN_DATA_PATH="/share/dai-sys/wanghanzhen/projects/MTP/training_data/open_perfectblend_80k_qwen3_8b.jsonl" \
TARGET_MODEL_BACKEND=sglang SGLANG_MEM_FRACTION_STATIC=0.25 \
TARGET_MODEL=/share/dai-sys/wanghanzhen/models/Qwen/Qwen3-8B \
MODEL_TAG='Qwen3-8B' \
bash scripts/run_training_flashmtp.sh --dt h100
```

Qwen-14B

```bash
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

训练脚本通过 `CKPT_DIR` 加载 checkpoint 目录中的 `model.safetensors` + `config.json`，
若存在 `training_state.pt` 则同时恢复 epoch / global_step / optimizer / scheduler。

**checkpoint 目录结构**（两种均支持）：

- 扁平目录：`config.json`、`model.safetensors`、`training_state.pt`（直接作为 `CKPT_DIR`）
- 子目录：`epoch_{e}_step_{s}/` 下含上述文件（`CKPT_DIR` 指向该子目录，或 `RESUME=1` 从 `OUTPUT_DIR` 自动找最新）

**继续训练关键参数：**


| 参数                                        | 说明                                  |
| ----------------------------------------- | ----------------------------------- |
| `CKPT_DIR`                                | 待恢复的 checkpoint 目录                  |
| `TARGET_MODEL`                            | 目标模型路径                              |
| `OUTPUT_DIR`                              | 新 run 输出目录（勿与 `CKPT_DIR` 相同，除非有意覆盖） |
| `TRAIN_DATA_PATH`                         | 训练数据 jsonl                          |
| `CACHE_DIR`                               | 预处理缓存目录                             |
| `CUDA_VISIBLE_DEVICES` / `NPROC_PER_NODE` | GPU 与进程数                            |
| `WANDB_RUN_ID` / `WANDB_RUN_NAME`         | WandB 实验标识                          |


**注意：**

- `MARKOV_HEAD_TYPE` / `MARKOV_OUTPUT_MODE` / `MARKOV_RANK` 必须与 checkpoint 一致（脚本会校验）。
- `LOSS_DECAY_GAMMA`、`BASE_LM_CE_*`、`FINAL_CE_WEIGHT`、`TV_LOSS_WEIGHT` 可在 resume 时覆盖，不影响权重加载。
- `NUM_EPOCHS` 为**总 epoch 数**（非额外 epoch）。从 epoch 5 恢复且 `NUM_EPOCHS=6` 时，仅再跑 1 个 epoch。
- 若 `NUM_EPOCHS` 大于原 run，scheduler 的 `T_max` 仍来自 checkpoint，LR 衰减不会按新总步数重算；需接受原 schedule 尾部或放弃加载 scheduler（手动改学习率）。
- 恢复时**不要**同时设 `RESUME=1` 与 `CKPT_DIR`；`RESUME` 仅从 `OUTPUT_DIR` 找最新 checkpoint。
- 现有 v2 checkpoint 的 `training_state.pt` 在**训练保存时**就只含约 15/65 个参数的 Adam 动量（FSDP + 外部 `BF16Optimizer` 未做跨 rank gather）；扁平目录与 `epoch_*_step_`* 子目录内容一致，并非导出截断。脚本会**尽量部分恢复**兼容的 optimizer state，其余参数动量重新初始化。权重与 scheduler（LR/epoch/step）仍可正确恢复。
- 若不想混用「部分旧动量 + 新初始化动量」，可加 `RESUME_OPTIMIZER=0`（或 `--no-resume-optimizer`），仅恢复 scheduler，Adam 全部重初始化。
- 若原训练 `OUTPUT_DIR` 仍保留 `epoch_*_step_*` 子目录，resume 会优先使用该子目录下的 `training_state.pt`（动量完整度与扁平目录相同）。



### FlashMTP_v2 示例：从 step 50000 继续（rnn_easy / direct / TV=0）

checkpoint：`cache/models/flashmtp_v2_mhrnn_easy_direct_r512_wb_0.2_bgemma_21_qwen3_8b`
（epoch=5, global_step=50000, 原 `NUM_EPOCHS=6`，约剩 1 epoch / ~10000 step）

```bash
cd /share/dai-sys/wanghanzhen/projects/MTP/FlashMTP_v2
source .venv/bin/activate

CKPT_DIR=/share/dai-sys/wanghanzhen/projects/MTP/FlashMTP_v2/cache/models/flashmtp_v2_mhrnn_easy_direct_r512_wb_0.2_bgemma_21_qwen3_8b \
OUTPUT_DIR=./cache/models/flashmtp_v2_mhrnn_easy_direct_r512_wb_0.2_bgemma_21_qwen3_8b_continue \
NUM_MIDDLE_LAYERS_N=16 NUM_DRAFT_LAYERS=5 NUM_EPOCHS=6 PIVOT_FUSE_MODE=prefix_condition \
DATA_NUM_SAMPLES=80000 MAX_LENGTH=4096 NUM_ANCHORS=512 BLOCK_SIZE=16 LOCAL_POSITION=true \
LOSS_DECAY_GAMMA=7 BASE_LM_CE_DECAY_GAMMA=21 BASE_LM_CE_WEIGHT=0.2 FINAL_CE_WEIGHT=1.0 TV_LOSS_WEIGHT=0.0 \
MARKOV_HEAD_TYPE=rnn_easy MARKOV_OUTPUT_MODE=direct MARKOV_RANK=512 \
NPROC_PER_NODE=8 TP_SIZE=1 SHARD_DRAFT_BY_TP=1 CE_CHUNK_SIZE=8192 \
TRAIN_DATA_PATH="/share/dai-sys/wanghanzhen/projects/MTP/training_data/open_perfectblend_80k_qwen3_8b.jsonl" \
TARGET_MODEL_BACKEND=sglang SGLANG_MEM_FRACTION_STATIC=0.25 \
TARGET_MODEL=/share/dai-sys/wanghanzhen/models/Qwen/Qwen3-8B \
MODEL_TAG='Qwen3-8B' \
WANDB_RUN_ID=flashmtp_v2_mhrnn_easy_continue_ep6 \
WANDB_RUN_NAME=flashmtp_v2_mhrnn_easy_continue_ep6 \
bash scripts/run_training_flashmtp.sh --dt h100
```



### v1.1 示例（math_code 继续训练）

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
LOSS_DECAY_GAMMA=7 \
SAVE_INTERVAL=5000 \
LOG_INTERVAL=20 \
WANDB_RUN_ID=flashmtp_continue_math_code_1w_ep6 \
WANDB_RUN_NAME=flashmtp_continue_math_code_1w_ep6 \
bash scripts/run_training_flashmtp.sh --dt h100
```

---



## 常用环境变量速查


| 变量                       | 默认值（脚本内）      | 说明                                     |
| ------------------------ | ------------- | -------------------------------------- |
| `NUM_EPOCHS`             | 6             | 训练 epoch 数                             |
| `MAX_LENGTH`             | 4096          | 最大序列长度                                 |
| `NUM_DRAFT_LAYERS`       | 5             | 草稿模型层数                                 |
| `NUM_MIDDLE_LAYERS_N`    | 5             | target 中间选取层数                          |
| `BLOCK_SIZE`             | —             | 草稿块大小                                  |
| `PIVOT_FUSE_MODE`        | `linear_fuse` | pivot 融合模式                             |
| `CHS_CONCAT_MODE`        | `feature`     | CHS 拼接模式                               |
| `LOCAL_POSITION`         | false         | 块内局部位置编码                               |
| `LOSS_DECAY_GAMMA`       | —             | 最终 CE 块内衰减系数                           |
| `BASE_LM_CE_WEIGHT`      | 0             | 骨干 hidden 经 target lm_head 的辅助 CE 权重 λ |
| `BASE_LM_CE_DECAY_GAMMA` | —             | 辅助 CE 独立衰减系数（不设则均匀权重）                  |
| `CHAT_TEMPLATE`          | —             | 对话模板（`qwen` / `llama3`）                |
| `DATA_NUM_SAMPLES`       | 40000         | 训练样本数                                  |
| `--dt`                   | a800          | 设备类型：`qz` / `a800` / `h100`            |


更多参数说明见 `scripts/run_training_flashmtp.sh` 与项目根目录 `v1.1.md`。