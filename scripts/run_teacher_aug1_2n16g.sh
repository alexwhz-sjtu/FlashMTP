#!/usr/bin/env bash
set -euo pipefail

cd /data/wanghanzhen/FlashMTP_v2.3
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

: "${NODE_RANK:?NODE_RANK must be 0 or 1}"

SWA_WINDOW_SIZE=128 \
ANCHOR_GROUP_SIZE=6 \
CHS_NUM_LAYERS=12 \
LOCAL_POSITION=true \
CE_CHUNK_SIZE=6144 \
BLOCK_SIZE=8 \
NUM_DRAFT_LAYERS=5 \
NUM_EPOCHS=8 \
NUM_ANCHORS=768 \
MAX_LENGTH=10240 \
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
TRAIN_DATA_PATH=/data/wanghanzhen/training_data/mixed_2360k_qwen3_8b_nm_pb_swe_aug1.jsonl \
MODEL_TAG=Qwen3_8B \
TARGET_MODEL=/data/wanghanzhen/models/Qwen3-8B \
TARGET_MODEL_BACKEND=sglang \
SGLANG_MEM_FRACTION_STATIC=0.25 \
TP_SIZE=2 \
SHARD_DRAFT_BY_TP=1 \
OUTPUT_DIR=/data/wanghanzhen/FlashMTP_v2.3/cache/models/flashmtp_v2_3_teacher_2n16g_targettp2_draftdp16_sglang025_swa128_ag6_chs12_a768_block8_d5_rnn_easy_direct_r512_aug1_qwen3_8b_maxlen10240_acc2 \
PYTHON_BIN=/data/wanghanzhen/FlashMTP_v2.3/.venv/bin/python \
NNODES=2 \
NPROC_PER_NODE=8 \
MASTER_ADDR=192.168.1.249 \
MASTER_PORT=29547 \
bash scripts/run_training_flashmtp_teacher.sh \
  --cache-dir /data/wanghanzhen/FlashMTP_v2.3/cache/train_aug1_maxlen10240 \
  --build-dataset-num-proc 32 \
  --dataloader-num-workers 8 \
  --report-to wandb \
  --wandb-project flashmtp-trainingv2-full \
  --wandb-name flashmtp-v2.3-teacher-2n16g-targettp2-draftdp16-sglang025-a768-aug1-qwen3-8b-maxlen10240-acc2 \
  --wandb-run-id flashmtp-v2-3-teacher-2n16g-targettp2-draftdp16-sglang025-a768-aug1-qwen3-8b-maxlen10240-acc2
