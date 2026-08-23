# FlashMTP 当前训练接口

## Teacher

入口：`run_training_flashmtp_teacher.sh` → `train_flashmtp_teacher.py`。

```text
teacher_loss = FINAL_CE_WEIGHT * final_ce
             + TV_LOSS_WEIGHT * sum(abs(p_final - p_target))
             + BASE_LM_CE_WEIGHT * base_ce
```

位置权重为 `exp(-offset/gamma)`，第一个预测位置 offset 为 0。Target prefill
logits 在 anchor 采样后一次 gather，完整序列张量随即释放。

```bash
cd /share/dai-sys/wanghanzhen/projects/MTP/FlashMTP_v2.3
source .venv/bin/activate
SWA_WINDOW_SIZE=128 \
ANCHOR_GROUP_SIZE=6 \
CHS_NUM_LAYERS=12 \
LOCAL_POSITION=true \
CE_CHUNK_SIZE=6144 \
BLOCK_SIZE=8 \
NUM_DRAFT_LAYERS=5 \
NUM_EPOCHS=8 \
NUM_ANCHORS=768 \
MAX_LENGTH=20480 \
BATCH_SIZE=1 \
LOSS_DECAY_GAMMA=4 \
DATA_NUM_SAMPLES=2360K_aug1_qwen3_8b \
BASE_LM_CE_DECAY_GAMMA=12 \
ACCUMULATION_STEPS=2 \
LEARNING_RATE=5e-4 \
FINAL_CE_WEIGHT=0.1 \
TV_LOSS_WEIGHT=1.0 \
BASE_LM_CE_WEIGHT=0.06 \
MARKOV_HEAD_TYPE=rnn_easy \
MARKOV_OUTPUT_MODE=direct \
MARKOV_RANK=512 \
TRAIN_DATA_PATH='dataset_path' \
MODEL_TAG='Qwen3_8B' \
TARGET_MODEL=/share/dai-sys/wanghanzhen/models/Qwen/Qwen3-8B \
bash scripts/run_training_flashmtp_teacher.sh --dt h100

```


| 变量                   | 含义                                          |
| -------------------- | ------------------------------------------- |
| `SWA_WINDOW_SIZE`    | Teacher 时间窗口 W，包含 W-1 个 fuse 位置和一个 CHS 时间位置 |
| `ANCHOR_GROUP_SIZE`  | Draft Q 中包含 anchor 的真实 token 数 G            |
| `BLOCK_SIZE`         | anchor-inclusive block 大小 B，预测 B-1 个 token  |
| `CHS_NUM_LAYERS`     | `a-1` 处保留的 target hidden 层数                 |
| `NUM_DRAFT_LAYERS`   | 并行 draft Transformer 深度                     |
| `MARKOV_HEAD_TYPE`   | `none`、`vanilla`、`gated`、`rnn` 或 `rnn_easy` |
| `MARKOV_OUTPUT_MODE` | `additive` 或 `direct`                       |
| `MARKOV_RANK`        | 串行头低秩维度                                     |




## Student 两阶段

入口：`run_training_flashmtp_two_stage.sh` → `train_flashmtp_two_stage.py`。
Teacher checkpoint 是 G、CHS、block、draft depth 和串行头结构的权威来源。

`STUDENT_INIT_MODE` 支持 `scratch`（默认）和 `shared_init`。`shared_init`
在 fresh Stage 1 开始前复制 teacher 的 `layers`、`norm`、
`layer_depth_embedding` 和 `context_norm`；teacher-only 历史融合参数与
串行 head 不在此时复制。模式会写入 checkpoint，恢复时自动沿用。

Stage 1：

```text
loss = STAGE1_TV_WEIGHT * weighted_mean(sum(abs(p_student - p_teacher)))
     + STAGE1_HIDDEN_WEIGHT * weighted_mean(SmoothL1(h_student, h_teacher))
```

Teacher 在 `eval/no_grad` 下运行。两者共享 anchors、target hidden、真实 Q embedding、
labels 和有效位置 mask；student 串行头不参与优化。

```bash
cd /share/dai-sys/wanghanzhen/projects/MTP/FlashMTP_v2.3
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/stop_keeper.sh

TARGET_MODEL=/data/wanghanzhen/models/Qwen3-8B \
STUDENT_INIT_MODE=shared_init \
TARGET_MODEL_BACKEND=sglang \
SGLANG_MEM_FRACTION_STATIC=0.25 \
TEACHER_DRAFT_PATH='/data/wanghanzhen/FlashMTP_v2.3/cache/models/flashmtp_v2_3_teacher_maskrow_from1m_2n16g_targettp2_draftdp16_sglang025_swa128_ag6_chs12_a768_block8_d5_rnn_easy_direct_r512_aug1_qwen3_8b_maxlen10240_acc2_lr5e5_4ep/final' \
TRAIN_DATA_PATH='/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/models/Qwen/Qwen3-8B' \
TP_SIZE=2 \
NNODES=3 \
ACCUMULATION_STEPS=1 \
STAGE1_EPOCHS=2 \
STAGE1_LEARNING_RATE=2e-4 \
STAGE1_WARMUP_RATIO=0.02 \
STAGE1_TV_WEIGHT=1.0 \
STAGE1_HIDDEN_WEIGHT=0.0 \
STAGE1_SMOOTH_L1_BETA=1.0 \
STAGE1_LOSS_DECAY_GAMMA=12 \
STAGE2_EPOCHS=4 \
STAGE2_LEARNING_RATE=1e-4 \
STAGE2_WARMUP_RATIO=0.02 \
STAGE2_FINAL_CE_WEIGHT=0.1 \
STAGE2_TV_WEIGHT=1.0 \
STAGE2_BASE_CE_WEIGHT=0.06 \
STAGE2_LOSS_DECAY_GAMMA=4 \
STAGE2_BASE_CE_DECAY_GAMMA=12 \
SHARD_DRAFT_BY_TP=1 \
MAX_LENGTH=10240 \
NUM_ANCHORS=768 \
bash scripts/run_training_flashmtp_two_stage.sh --dt qz > "whz_mtp_logs/train_flashmtp_qz_dist_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/stop_keeper.sh
```

Stage 2 只复制 teacher 串行头，然后释放 teacher 和 Stage 1 optimizer。训练 loss
与 teacher 的三项监督 loss 相同，并创建新的 optimizer/scheduler。


| 变量                                              | 含义                   |
| ----------------------------------------------- | -------------------- |
| `STAGE1_EPOCHS` / `STAGE2_EPOCHS`               | 两阶段独立 epoch 数        |
| `STAGE1_LEARNING_RATE` / `STAGE2_LEARNING_RATE` | 两阶段独立学习率             |
| `STAGE1_WARMUP_RATIO` / `STAGE2_WARMUP_RATIO`   | 两阶段独立 warmup 比例      |
| `STAGE1_TV_WEIGHT` / `STAGE1_HIDDEN_WEIGHT`     | Stage 1 两项 loss 权重   |
| `STAGE1_SMOOTH_L1_BETA`                         | SmoothL1 beta        |
| `STAGE1_LOSS_DECAY_GAMMA`                       | Stage 1 共用位置衰减       |
| `STAGE2_FINAL_CE_WEIGHT`                        | Stage 2 final CE 权重  |
| `STAGE2_TV_WEIGHT`                              | Stage 2 target TV 权重 |
| `STAGE2_BASE_CE_WEIGHT`                         | Stage 2 base CE 权重   |
| `STAGE2_LOSS_DECAY_GAMMA`                       | final CE/TV 位置衰减     |
| `STAGE2_BASE_CE_DECAY_GAMMA`                    | base CE 独立位置衰减       |


通用变量包括 `ACCUMULATION_STEPS`、`NUM_ANCHORS`、`MAX_LENGTH`、
`SAVE_INTERVAL`、`LOG_INTERVAL`、`TARGET_MODEL_BACKEND` 和 `RESUME_FROM`。

当前 teacher/student 训练均使用 `mask_embedding_mode=vocab_row`：MASK ID 必须
对应 target embedding 中已有的一行，不再用词表均值构造 MASK embedding。
多机迁移和完整启动示例见 `docs/STUDENT_TWO_STAGE_PORTING.md`。