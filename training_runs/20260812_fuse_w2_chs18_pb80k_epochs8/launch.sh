#!/usr/bin/env bash
set -euo pipefail

cd /share/dai-sys/wanghanzhen/projects/MTP/FlashMTP_v2swa
source .venv/bin/activate

SLIDING_WINDOW_SIZE=2 \
CHS_NUM_LAYERS=18 \
LOCAL_POSITION=true \
HISTORY_MODE=fuse \
BLOCK_SIZE=8 \
NUM_DRAFT_LAYERS=5 \
NUM_EPOCHS=8 \
NUM_ANCHORS=512 \
MAX_LENGTH=4096 \
BATCH_SIZE=1 \
LOSS_DECAY_GAMMA=4 \
DATA_NUM_SAMPLES=pb_80k \
BASE_LM_CE_DECAY_GAMMA=12 \
LEARNING_RATE=5e-4 \
FINAL_CE_WEIGHT=0.1 \
TV_LOSS_WEIGHT=1.0 \
BASE_LM_CE_WEIGHT=0.0 \
MARKOV_HEAD_TYPE=vanilla \
MARKOV_OUTPUT_MODE=additive \
MARKOV_RANK=256 \
TRAIN_DATA_PATH='/share/dai-sys/wanghanzhen/projects/MTP/training_data/open_perfectblend_80k_qwen3_8b.jsonl' \
MODEL_TAG='Qwen3_8B' \
TARGET_MODEL=/share/dai-sys/wanghanzhen/models/Qwen/Qwen3-8B \
bash scripts/run_training_flashmtp.sh --dt h100
