# DLite

DLite is a Qwen3 speculative-decoding project with teacher training, two-stage
student training, and standalone evaluation. The student predicts a block in
parallel and uses a lightweight `rnn` sequential head to restore token-to-token
dependence inside the block.

The repository contains the complete training stack. The sibling `DLite/`
repository is the smaller inference-only release.

```bash
uv venv -p 3.11
source .venv/bin/activate
uv pip install -v -e . --prerelease=allow
uv pip install datasets==4.8.3 pyarrow==23.0.1
```

## Environment

Python 3.11 and [uv](https://docs.astral.sh/uv/) are required. Create the
virtual environment, install this repository, then pin the dataset stack:

```bash
uv venv -p 3.11
source .venv/bin/activate
uv pip install -v -e . --prerelease=allow
uv pip install datasets==4.8.3 pyarrow==23.0.1
```

Activate `.venv` before running project commands.

FlashAttention is optional. Install the regular dependencies first so PyTorch
is available while FlashAttention is built:

```bash
uv pip install flash-attn --no-build-isolation
```

The supported target backends are `hf` and `sglang`. Attention backend options
provided by PyTorch/Transformers are preserved; FlashAttention falls back to
SDPA when it is unavailable.

## Train

Teacher training:

```bash
TARGET_MODEL=/path/to/Qwen3-8B \
TRAIN_DATA_PATH=/path/to/train.jsonl \
OUTPUT_DIR=/path/to/teacher \
bash scripts/run_training_dlite_teacher.sh --dt h100
```

Two-stage student training:

```bash
TARGET_MODEL=/path/to/Qwen3-8B \
TEACHER_DRAFT_PATH=/path/to/teacher/final \
TRAIN_DATA_PATH=/path/to/train.jsonl \
OUTPUT_DIR=/path/to/student \
STAGE1_EPOCHS=1 STAGE2_EPOCHS=5 \
LEARNING_RATE=5e-4 STAGE1_KL_WEIGHT=1.0 \
bash scripts/run_training_dlite_two_stage.sh --dt h100
```

Both stages consume the same `TRAIN_DATA_PATH`. The student backbone always
starts from random initialization. Stage 1 trains only with weighted KL
distillation; Stage 2 uses the configured CE/TV objectives. The teacher's
sequential head is copied to the student and frozen during Stage 1.

See [docs/TRAINING.md](docs/TRAINING.md) for configuration details.

## Benchmark

```bash
TARGET_MODEL=/path/to/Qwen3-8B \
DRAFT_NAME_OR_PATH=/path/to/student/final \
DATASET=gsm8k MAX_SAMPLES=100 \
bash evaluation/run_benchmark_dlite.sh
```

Checkpoint model settings live under `dlite_config`. The public architecture is
`DLiteDraftModel`; the only sequential-head type is `rnn`. New training uses
`dlite_v2`, while inference remains compatible with both `dlite_v1` and
`dlite_v2` checkpoints.

## License

See [LICENSE](LICENSE).
