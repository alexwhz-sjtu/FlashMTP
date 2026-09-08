#!/usr/bin/env bash
set -euo pipefail

cd /data/wanghanzhen/FlashMTP_v2.3
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

: "${NODE_RANK:?NODE_RANK must be 0 or 1}"

SWA_WINDOW_SIZE=512 \
ANCHOR_GROUP_SIZE=6 \
CHS_NUM_LAYERS=12 \
LOCAL_POSITION=true \
CE_CHUNK_SIZE=6144 \
BLOCK_SIZE=8 \
NUM_DRAFT_LAYERS=10 \
NUM_EPOCHS=6 \
NUM_ANCHORS=768 \
MAX_LENGTH=10240 \
BATCH_SIZE=1 \
LOSS_DECAY_GAMMA=4 \
DATA_NUM_SAMPLES=2360k_aug1_plus_math_code_chat_temp1_n5_2693216 \
BASE_LM_CE_DECAY_GAMMA=12 \
ACCUMULATION_STEPS=2 \
LEARNING_RATE=6e-4 \
FINAL_CE_WEIGHT=0.1 \
TV_LOSS_WEIGHT=1.0 \
BASE_LM_CE_WEIGHT=0.06 \
MARKOV_HEAD_TYPE=rnn_easy \
MARKOV_OUTPUT_MODE=direct \
MARKOV_RANK=512 \
TRAIN_DATA_PATH=/data/wanghanzhen/training_data/mixed_2360k_qwen3_8b_aug1_plus_math_code_chat_temp1_n5_2693216_shuffled.jsonl \
MODEL_TAG=Qwen3_8B \
TARGET_MODEL=/data/wanghanzhen/models/Qwen3-8B \
TARGET_MODEL_BACKEND=sglang \
SGLANG_MEM_FRACTION_STATIC=0.25 \
TP_SIZE=2 \
SHARD_DRAFT_BY_TP=1 \
OUTPUT_DIR=/data/wanghanzhen/FlashMTP_v2.3/cache/models/teacher_aug1_plus_temp1_qwen3_8b_d10_swa512_r512 \
PYTHON_BIN=/data/wanghanzhen/FlashMTP_v2.3/.venv/bin/python \
NNODES=2 \
NPROC_PER_NODE=8 \
MASTER_ADDR=192.168.1.252 \
MASTER_PORT=29568 \
bash scripts/run_training_flashmtp_teacher.sh \
  --cache-dir /data/wanghanzhen/FlashMTP_v2.3/cache/train_2360k_aug1_plus_math_code_chat_temp1_n5_qwen3_8b_maxlen10240 \
  --build-dataset-num-proc 32 \
  --dataloader-num-workers 8 \
  --report-to wandb \
  --wandb-project flashmtp-trainingv2-full \
  --wandb-name teacher-qwen3-8b-d10-swa512-r512-2360k-aug1-plus-mathcodechat-temp1-n5 \
  --wandb-run-id teacher-qwen3-8b-d10-swa512-r512-2360k-aug1-plus-mathcodechat-temp1-n5
