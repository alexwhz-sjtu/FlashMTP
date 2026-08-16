#!/bin/bash
set -euo pipefail

cd /data/wanghanzhen/FlashMTP_v2swa
source .venv/bin/activate
export PYTHONPATH="/data/wanghanzhen/FlashMTP_v2swa${PYTHONPATH:+:${PYTHONPATH}}"

: "${NODE_RANK:?NODE_RANK must be 0 or 1}"

SLIDING_WINDOW_SIZE=5 \
CHS_NUM_LAYERS=18 \
LOCAL_POSITION=true \
BLOCK_SIZE=8 \
NUM_DRAFT_LAYERS=5 \
NUM_EPOCHS=8 \
NUM_ANCHORS=768 \
ANCHOR_CHUNK_SIZE=128 \
MAX_LENGTH=30720 \
BATCH_SIZE=1 \
LOSS_DECAY_GAMMA=4 \
BASE_LM_CE_DECAY_GAMMA=12 \
LEARNING_RATE=5e-4 \
FINAL_CE_WEIGHT=0.1 \
TV_LOSS_WEIGHT=1.0 \
BASE_LM_CE_WEIGHT=0.06 \
MARKOV_HEAD_TYPE=vanilla \
MARKOV_OUTPUT_MODE=additive \
MARKOV_RANK=256 \
DATA_NUM_SAMPLES=2360k \
TRAIN_DATA_PATH=/data/wanghanzhen/training_data/mixed_2360k_qwen3_8b_nm_pb_swe_aug3.jsonl \
CACHE_DIR=/data/wanghanzhen/FlashMTP_v2/cache/data/regen_data/nemotron_pb_80k \
BUILD_DATASET_NUM_PROC=32 \
MODEL_TAG=Qwen3_8B \
TARGET_MODEL=/data/wanghanzhen/models/Qwen3-8B \
TARGET_MODEL_BACKEND=sglang \
SGLANG_MEM_FRACTION_STATIC=0.3 \
TP_SIZE=2 \
OUTPUT_DIR=/data/wanghanzhen/FlashMTP_v2swa/cache/models/flashmtp_v2swa_2n16g_tp2_ac128_vanilla_additive_r256_mixed2360k_w5_chs18_block8_maxlen30720_ep8 \
WANDB_PROJECT=flashmtp-training-v2 \
WANDB_RUN_ID=flashmtp_v2swa_2n16g_tp2_ac128_vanilla_additive_r256_mixed2360k_w5_chs18_block8_maxlen30720_ep8 \
WANDB_RUN_NAME=flashmtp_v2swa_2n16g_tp2_ac128_vanilla_additive_r256_mixed2360k_w5_chs18_block8_maxlen30720_ep8 \
NNODES=2 \
NPROC_PER_NODE=8 \
MASTER_ADDR=192.168.1.249 \
MASTER_PORT=29531 \
bash scripts/run_training_flashmtp.sh --dt h100
