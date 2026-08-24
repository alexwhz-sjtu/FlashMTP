# v2.3 Student 两阶段训练迁移与启动

## 1. 迁移内容

将完整的 `FlashMTP_v2.3` 源码目录复制到每台训练机器；不要复制旧机器的
`.venv`。在第三方机器上使用 Python 3.11 重新安装环境：

```bash
cd /path/to/FlashMTP_v2.3
uv venv -p 3.11
source .venv/bin/activate
uv pip install -v -e . --prerelease=allow
python -m unittest discover -s tests -v
```

两机训练时，以下内容必须在两台机器上以相同路径可读：

- target model；
- teacher 的具体 checkpoint；
- JSONL 训练数据；
- 数据缓存目录；
- student 输出目录。

推荐让缓存和输出目录位于共享文件系统。Student checkpoint 包含 rank-local
optimizer shard，而模型和配置只由 global rank 0 写入；没有共享文件系统时，
跨节点恢复会缺少文件。

## 2. 推荐配置

Teacher checkpoint 是 student 结构参数的权威来源。必须指向包含
`config.json` 和 `model.safetensors` 的具体 checkpoint，例如 `final/`，不能只指向
teacher 输出根目录。

当前训练固定使用 v2 MASK 行为：`MASK_TOKEN_ID=151669` 直接读取 target 的
`embedding.weight[151669]`。该 ID 必须小于 target embedding 的实际行数。

```bash
cd /path/to/FlashMTP_v2.3
source .venv/bin/activate

TARGET_MODEL=/shared/models/Qwen3-8B \
TARGET_MODEL_BACKEND=sglang \
SGLANG_MEM_FRACTION_STATIC=0.25 \
TEACHER_DRAFT_PATH=/shared/checkpoints/teacher/final \
STAGE1_TRAIN_DATA_PATH=/shared/data/distillation.jsonl \
STAGE2_TRAIN_DATA_PATH=/shared/data/supervised.jsonl \
STAGE1_EPOCHS=2 \
STAGE1_LEARNING_RATE=5e-4 \
STAGE1_WARMUP_RATIO=0.04 \
STAGE1_TV_WEIGHT=1.0 \
STAGE1_HIDDEN_WEIGHT=1.0 \
STAGE1_SMOOTH_L1_BETA=1.0 \
STAGE1_LOSS_DECAY_GAMMA=4 \
STAGE2_EPOCHS=6 \
STAGE2_LEARNING_RATE=2e-4 \
STAGE2_WARMUP_RATIO=0.04 \
STAGE2_FINAL_CE_WEIGHT=0.1 \
STAGE2_TV_WEIGHT=1.0 \
STAGE2_BASE_CE_WEIGHT=0.06 \
STAGE2_LOSS_DECAY_GAMMA=4 \
STAGE2_BASE_CE_DECAY_GAMMA=12 \
MAX_LENGTH=10240 \
NUM_ANCHORS=768 \
BATCH_SIZE=1 \
ACCUMULATION_STEPS=2 \
STAGE1_BUILD_DATASET_NUM_PROC=32 \
STAGE2_BUILD_DATASET_NUM_PROC=32 \
DATALOADER_NUM_WORKERS=8 \
SAVE_INTERVAL=20000 \
LOG_INTERVAL=50 \
TP_SIZE=2 \
SHARD_DRAFT_BY_TP=1 \
NNODES=2 \
NPROC_PER_NODE=8 \
NODE_RANK=0 \
MASTER_ADDR=10.0.0.10 \
MASTER_PORT=29550 \
PYTHON_BIN=/path/to/FlashMTP_v2.3/.venv/bin/python \
DRY_RUN=1 \
bash scripts/run_training_flashmtp_two_stage.sh --dt qz
```

启动器默认 `MASK_TOKEN_ID=151669`、`STUDENT_INIT_MODE=shared_init` 和
`REPORT_TO=wandb`。它使用 target/data/teacher 结构、两阶段 epoch/LR/loss、长度、
anchor、累积步数、world size、TP 和分片模式生成确定性的 `RUN_TAG`，并据此设置：

- `OUTPUT_DIR=${PROJECT_DIR}/cache/models/${RUN_TAG}`；
- `CACHE_DIR=${PROJECT_DIR}/cache/train/s1<data1>_s2<data2>_l<max_length>_m<mask_id>`；
- `WANDB_PROJECT=flashmtp-training-v2.3-student`；
- `WANDB_NAME=<128 字符内的关键配置摘要>_<RUN_TAG hash>`；
- `WANDB_RUN_ID=<模型/数据/并行摘要>_<RUN_TAG hash>`。

这些值均可用同名环境变量覆盖；`OUTPUT_ROOT` 可单独改变默认输出根目录。相同参数会
得到相同路径和 W&B id，适合多节点共享存储和恢复。相同配置要启动独立实验时设置
`RUN_SUFFIX=<tag>`；不使用 W&B 时设置 `REPORT_TO=none`。

`STAGE1_TRAIN_DATA_PATH` 和 `STAGE2_TRAIN_DATA_PATH` 分别控制蒸馏数据与监督训练
数据。fresh 或 Stage 1 恢复时，在加载完 tokenizer 后会立即并行预处理两套数据：
共享存储的分布式训练由 global rank 0 构建 Stage 1 cache、global rank 1 同时构建
Stage 2 cache；单进程调试则使用两个线程。cache 分别位于 `stage1/`、`stage2/`
命名空间。若两个路径相同，只构建一次共享 cache；旧的 `TRAIN_DATA_PATH` 仍可作为
两个变量的共同回退。`STAGE1_BUILD_DATASET_NUM_PROC` 和
`STAGE2_BUILD_DATASET_NUM_PROC` 可分别设置内部 map 进程数，未设置时都回退到
`BUILD_DATASET_NUM_PROC`。

节点信息沿用 v2 的 `PET_*` 默认值：

| 最终变量 | 默认来源 |
| --- | --- |
| `NNODES` | `PET_NNODES` → `NNODES` → `1` |
| `NODE_RANK` | `PET_NODE_RANK` → `NODE_RANK` → `0` |
| `NPROC_PER_NODE` | `NPROC_PER_NODE` → `PET_NPROC_PER_NODE` → 可见 GPU 数 |
| `MASTER_ADDR` | `MASTER_ADDR` → `PET_MASTER_ADDR` → `127.0.0.1` |
| `MASTER_PORT` | `MASTER_PORT` → `PET_MASTER_PORT` → `29502` |

`--dt qz` 由 shell 启动器消费并默认设置 `WANDB_MODE=offline`，不会传给 Python。
也支持 `--dt a800` 和 `--dt h100`。

先在两台机器分别用各自的 `NODE_RANK=0/1` 执行 `DRY_RUN=1`，确认展开后的
torchrun 命令一致。正式启动时删除 `DRY_RUN=1`，通常先启动 node 1，再立即启动
node 0。`MASTER_ADDR` 必须是 node 0 可被其他节点访问的内网地址；防火墙需放行
`MASTER_PORT`。

`SHARD_DRAFT_BY_TP=1` 时，`BATCH_SIZE=1` 表示每个 draft rank 的本地 batch；
启动器会将 target prefill batch 自动扩成 `TP_SIZE`。同一个 TP 组先对相同的
`TP_SIZE` 个样本做一次 target prefill，然后 TP rank `r` 只保留第 `r` 个样本的
input、anchor、hidden 和 logits。Stage 1 的 teacher/student 共享这个 rank-local
样本，Stage 2 也沿用相同分片。因此 target 使用 TP，而每个 rank 上的 draft
teacher/student 处理不同数据。

该模式要求 `TARGET_MODEL_BACKEND=sglang`、`TP_SIZE>1`，并要求
`NPROC_PER_NODE` 可整除 `TP_SIZE`，防止一个 TP 组跨机器。若显式把
`BATCH_SIZE` 设为其他值，只允许 `1` 或 `TP_SIZE`。

## 3. 恢复

Stage 1 恢复必须同时提供原 teacher：

```bash
RESUME_FROM=/shared/checkpoints/student_two_stage/stage1/epoch_0_step_20000 \
TEACHER_DRAFT_PATH=/shared/checkpoints/teacher/final \
bash scripts/run_training_flashmtp_two_stage.sh
```

从 `transition/` 或 Stage 2 checkpoint 恢复时不再加载 teacher：

```bash
RESUME_FROM=/shared/checkpoints/student_two_stage/stage2/epoch_0_step_20000 \
bash scripts/run_training_flashmtp_two_stage.sh
```

恢复需要保持相同 world size、`TP_SIZE`、`SHARD_DRAFT_BY_TP`、两阶段数据路径和自动命名所用的
关键参数，因为 optimizer state 按 global rank 分片保存，而且 checkpoint 会记录
draft 数据分片模式。原训练若覆盖过 `OUTPUT_DIR`，恢复时也必须传回同一目录。

## 4. 启动前检查

```bash
bash -n scripts/run_training_flashmtp_two_stage.sh
python -m compileall -q scripts specforge tests
python -m unittest discover -s tests -v
```

W&B 在线模式下所有节点都应能读取 `WANDB_API_KEY` 或有效的 `~/.netrc`；
`WANDB_MODE=offline` 不需要 API key，指标保存在本地 W&B 目录。若显存
不足，先降低 `NUM_ANCHORS`，其次降低 `MAX_LENGTH`；不要改变 teacher checkpoint
定义的 block、G、CHS、draft depth 或 Markov head 结构。
