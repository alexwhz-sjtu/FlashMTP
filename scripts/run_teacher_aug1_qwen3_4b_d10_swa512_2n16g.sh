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
DATA_NUM_SAMPLES=2300K_aug1_qwen3_4b \
BASE_LM_CE_DECAY_GAMMA=12 \
ACCUMULATION_STEPS=2 \
LEARNING_RATE=6e-4 \
FINAL_CE_WEIGHT=0.1 \
TV_LOSS_WEIGHT=1.0 \
BASE_LM_CE_WEIGHT=0.06 \
MARKOV_HEAD_TYPE=rnn_easy \
MARKOV_OUTPUT_MODE=direct \
MARKOV_RANK=320 \
TRAIN_DATA_PATH=/data/wanghanzhen/training_data/mixed_2.3M_qwen3_4b_aug1.jsonl \
MODEL_TAG=Qwen3_4B \
TARGET_MODEL=/data/wanghanzhen/models/Qwen3-4B \
TARGET_MODEL_BACKEND=sglang \
SGLANG_MEM_FRACTION_STATIC=0.25 \
TP_SIZE=1 \
SHARD_DRAFT_BY_TP=0 \
OUTPUT_DIR=/data/wanghanzhen/FlashMTP_v2.3/cache/models/flashmtp_v2_3_teacher_2n16g_targettp1_draftdp16_sglang025_swa512_ag6_chs12_a768_block8_d10_rnn_easy_direct_r320_aug1_qwen3_4b_maxlen10240_acc2_lr6e4_ep6 \
PYTHON_BIN=/data/wanghanzhen/FlashMTP_v2.3/.venv/bin/python \
NNODES=2 \
NPROC_PER_NODE=8 \
MASTER_ADDR=192.168.1.249 \
MASTER_PORT=29556 \
bash scripts/run_training_flashmtp_teacher.sh \
  --cache-dir /data/wanghanzhen/FlashMTP_v2.3/cache/train_aug1_qwen3_4b_maxlen10240 \
  --build-dataset-num-proc 32 \
  --dataloader-num-workers 8 \
  --report-to wandb \
  --wandb-project flashmtp-trainingv2-full \
  --wandb-name "${WANDB_NAME:-flashmtp-v2.3-teacher-qwen3-4b-d10-swa512-r320-lr6e4-ep6}" \
  --wandb-run-id "${WANDB_RUN_ID:-flashmtp-v2-3-teacher-qwen3-4b-d10-swa512-r320-lr6e4-ep6}"
