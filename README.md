# FlashMTP

当前代码只实现一套架构：`swa_teacher_pivotq_student_v1`。旧版 v3/v4/v5、
`dense`/`token` history mode、位置开关和 logits chunk 接口均不兼容。

## 模型定义

设 anchor 为 `a`、`BLOCK_SIZE=B`、`SWA_WINDOW_SIZE=W`、
`ANCHOR_GROUP_SIZE=G`。`B` 包含 anchor，模型预测 `a+1...a+B-1`。

```text
swa_teacher Context KV:
  [fuse(a-W), ..., fuse(a-2), CHS_layers(a-1)]
swa_teacher Draft Q:
  [embed(a-G+1), ..., embed(a), MASK × (B-1)]

pivot_q_student Context KV:
  [CHS_layers(a-1)]
pivot_q_student Draft Q:
  [embed(a-G+1), ..., embed(a), MASK × (B-1)]
```

`fuse(t) = RMSNorm(Linear([h_first(t); h_middle(t); h_last(t)]))`。Teacher
使用全局 RoPE；student 固定使用以第一个有效真实 Q token 为 0 的局部 RoPE。
短上下文在左侧补零并由 attention mask 屏蔽。Context KV 在 draft Transformer
每一层重新投影注入。

## 训练

Teacher 独立训练：

```bash
TARGET_MODEL=/path/to/target \
TRAIN_DATA_PATH=/path/to/train.jsonl \
OUTPUT_DIR=/path/to/teacher-output \
SWA_WINDOW_SIZE=32 ANCHOR_GROUP_SIZE=8 BLOCK_SIZE=8 \
bash scripts/run_training_flashmtp_teacher.sh
```

Student 两阶段连续训练：

```bash
TARGET_MODEL=/path/to/target \
TEACHER_DRAFT_PATH=/path/to/teacher-output/final \
STAGE1_TRAIN_DATA_PATH=/path/to/distillation.jsonl \
STAGE2_TRAIN_DATA_PATH=/path/to/supervised.jsonl \
OUTPUT_DIR=/path/to/student-output \
STUDENT_INIT_MODE=shared_init \
STAGE1_EPOCHS=2 STAGE1_LEARNING_RATE=5e-4 \
STAGE2_EPOCHS=6 STAGE2_LEARNING_RATE=2e-4 \
bash scripts/run_training_flashmtp_two_stage.sh
```

Python 入口默认 `STUDENT_INIT_MODE=scratch`，两阶段 shell 启动器默认使用推荐的
`shared_init`。设为 `shared_init` 时，Stage 1 开始前从
teacher 复制并行 backbone、CHS 编码和相关 norm，但不复制历史融合模块
或串行 head。Stage 1 只更新 student 并行 backbone、CHS 编码和相关 norm，以 teacher hidden
的 LM-head TV 距离及 SmoothL1 蒸馏。Stage 2 在 full-param 上下文中只继承 teacher
串行 head，随后释放 teacher，用 target prefill greedy top-1 作为 final CE/base CE
的 label，并结合 target TV 训练完整 student；串行 head 的 teacher forcing 仍使用
训练数据中的原始 token。两阶段分别创建 optimizer 和 cosine/warmup scheduler。

若 student draft 比 teacher 浅，可设置 `STUDENT_INIT_MODE=shared_partial` 和
`STUDENT_NUM_DRAFT_LAYERS=N`。该模式要求 teacher 层数严格大于 student，按首尾
对齐的均匀索引抽取 teacher backbone 层初始化 student；共享 norm 仍完整复制，
历史融合模块不复制，Stage 2 串行 head 仍从 teacher 直接继承。

Stage 1 和 Stage 2 使用独立数据变量、缓存和 dataloader；fresh/Stage 1 启动时会
同时预处理两套数据。若二者相同，也可继续只设置兼容变量 `TRAIN_DATA_PATH`，数据
只会预处理一次。

SGLang target 可设置 `TP_SIZE=N SHARD_DRAFT_BY_TP=1`：每个 TP 组对共享的 `N`
样本做一次 target prefill，组内每个 rank 的 teacher/student draft 各训练其中一个
不同样本。该分片模式同时用于 Stage 1 和 Stage 2。

启动器兼容 v2 的 `PET_*` 多节点环境变量，默认 MASK ID 为 `151669`，并自动生成
包含 teacher 结构、数据、两阶段超参和并行配置的输出目录及 W&B name/id。

详细参数与 loss 定义见 [scripts/train.md](scripts/train.md)。
第三方机器迁移、两机启动与恢复示例见
[docs/STUDENT_TWO_STAGE_PORTING.md](docs/STUDENT_TWO_STAGE_PORTING.md)。

## Checkpoint 与恢复

Student 输出分为 `stage1/`、`transition/`、`stage2/` 和 `final/`。checkpoint
记录模型角色和结构、阶段 epoch/step、全局 step、shared-init/串行头
继承状态及 teacher 标识。

```bash
# Stage 1 恢复需要 teacher
RESUME_FROM=/path/to/stage1-checkpoint \
TEACHER_DRAFT_PATH=/path/to/teacher ... \
bash scripts/run_training_flashmtp_two_stage.sh

# transition/Stage 2 恢复不会加载 teacher
RESUME_FROM=/path/to/transition-or-stage2-checkpoint ... \
bash scripts/run_training_flashmtp_two_stage.sh
```

若使用 W&B，固定 `--wandb-run-id` 可让恢复后的指标继续写入同一 run；指标前缀为
`stage1/*` 和 `stage2/*`，global step 单调递增。

## Logits 数据流

Target 对每批数据只 prefill 一次。监督训练直接从完整 prefill logits gather
`a...a+B-2`，取 greedy top-1 作为 Stage 2 CE label，随后释放完整 logits；不会从
final hidden 再调用 LM head，也没有 logits chunk 或 gradient checkpoint 分支。
Stage 1 不保留 target logits。

## 测试与安装

```bash
uv venv -p 3.11
source .venv/bin/activate
uv pip install -v -e . --prerelease=allow
python -m unittest discover -s tests -v
```
