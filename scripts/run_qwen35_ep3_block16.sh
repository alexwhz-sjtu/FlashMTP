#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR=/data/wanghanzhen/FlashMTP_v2swa/cache/models/Qwen3.5-4B/Flashmtp_v2_qwen3.5_4b_ep3
OUTPUT_DIR="${OUTPUT_DIR:-/data/wanghanzhen/FlashMTP_v2swa/cache/models/Qwen3.5-4B/Flashmtp_v2_qwen3.5_4b_ep3_block16_lr2e-4_gamma7_5ep}"

if [[ -z "${TRAIN_DATA_PATH:-}" || ! -f "${TRAIN_DATA_PATH}" ]]; then
    echo "Set TRAIN_DATA_PATH to an existing training JSONL file." >&2
    exit 1
fi
if [[ ! -f "${CKPT_DIR}/config.json" || ! -f "${CKPT_DIR}/model.safetensors" ]]; then
    echo "Missing checkpoint in ${CKPT_DIR}" >&2
    exit 1
fi
if [[ -d "${OUTPUT_DIR}" && -n "$(ls -A "${OUTPUT_DIR}")" ]]; then
    echo "Output directory already contains files: ${OUTPUT_DIR}" >&2
    exit 1
fi

export CKPT_DIR OUTPUT_DIR TRAIN_DATA_PATH
export LOAD_WEIGHTS_ONLY=1
export TARGET_MODEL=/data/wanghanzhen/models/Qwen/Qwen3.5-4B
export TARGET_MODEL_BACKEND=sglang
export MODEL_TAG=Qwen3.5_4B
export NUM_DRAFT_LAYERS=5
export BLOCK_SIZE=16
export NUM_EPOCHS=5
export LEARNING_RATE=0.0002
export LOSS_DECAY_GAMMA=7

# Keep the ep3 checkpoint's architecture and conditioning layout.
export SLIDING_WINDOW_SIZE=1
export CHS_LAYER_IDS=1,3,7,11,15,19,23,27,30,31
export LOCAL_POSITION=true
export BACKBONE_CONV_MODE=none
export MARKOV_HEAD_TYPE=rnn_easy
export MARKOV_OUTPUT_MODE=direct
export MARKOV_RANK=320

exec bash "${PROJECT_DIR}/scripts/run_training_flashmtp.sh" --dt h100
