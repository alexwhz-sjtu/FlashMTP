#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
DATA_PATH="/share/dai-sys/wanghanzhen/projects/MTP/training_data/generated/qwen3.5-35b-a3b/open_perfectblend_80k_think_off_temp0_maxnew4096.jsonl"
TARGET_PATH="/share/dai-sys/wanghanzhen/models/Qwen/Qwen3.5-35B-A3B"
DRAFT_CONFIG="${PROJECT_DIR}/configs/qwen3.5-35b-a3b-eagle3.json"
EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-80000}"

cd "${PROJECT_DIR}"
source .venv/bin/activate

actual_samples="$(wc -l < "${DATA_PATH}")"
if [ "${actual_samples}" -ne "${EXPECTED_SAMPLES}" ]; then
    echo "错误: 训练数据尚未完整生成 (${actual_samples}/${EXPECTED_SAMPLES}): ${DATA_PATH}" >&2
    exit 2
fi

export NUM_MIDDLE_LAYERS_N=14
export NUM_DRAFT_LAYERS=5
export NUM_EPOCHS=8
export PIVOT_FUSE_MODE=prefix_condition
export DATA_NUM_SAMPLES=pb_80k
export MAX_LENGTH=4096
export NUM_ANCHORS=512
export BLOCK_SIZE=8
export LOCAL_POSITION=true
export LOSS_DECAY_GAMMA=4
export BASE_LM_CE_DECAY_GAMMA=12
export BASE_LM_CE_WEIGHT=0.06
export FINAL_CE_WEIGHT=0.1
export TV_LOSS_WEIGHT=1.0
export MARKOV_HEAD_TYPE=rnn_easy
export MARKOV_OUTPUT_MODE=direct
export MARKOV_RANK=512
export NPROC_PER_NODE=8
export DISAGGREGATE=true
export RANK_TARGET_PER_NODE="${RANK_TARGET_PER_NODE:-6}"
export RANK_DRAFT_PER_NODE="${RANK_DRAFT_PER_NODE:-2}"
export TARGET_TP_SIZE="${TARGET_TP_SIZE:-2}"
export TP_SIZE="${TARGET_TP_SIZE}"
export NODE_BATCH_SIZE="${NODE_BATCH_SIZE:-12}"
export DRAFT_MICRO_BATCH_SIZE="${DRAFT_MICRO_BATCH_SIZE:-6}"
export PIPELINE_DEPTH="${PIPELINE_DEPTH:-2}"
export CE_CHUNK_SIZE=4096
export LEARNING_RATE=5e-4
export TRAIN_DATA_PATH="${DATA_PATH}"
export TARGET_MODEL_BACKEND=sglang
# TP=2 holds roughly 35 GB of BF16 target weights per rank, so the Qwen3-8B
# value 0.3 cannot leave a valid SGLang KV pool.  TP=4 needs only ~17.5 GB/rank.
if [ -z "${SGLANG_MEM_FRACTION_STATIC:-}" ]; then
    if [ "${TARGET_TP_SIZE}" -eq 2 ]; then
        export SGLANG_MEM_FRACTION_STATIC=0.55
    else
        export SGLANG_MEM_FRACTION_STATIC=0.3
    fi
fi
export TARGET_MODEL="${TARGET_PATH}"
export DRAFT_CONFIG_PATH="${DRAFT_CONFIG}"
export CHAT_TEMPLATE=qwen3.5-instruct
export ENABLE_THINKING=off
export MODEL_TAG=Qwen3.5-35B-A3B
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/cache/models/flashmtp_qwen35_35b_a3b_pb80k_disagg_t${RANK_TARGET_PER_NODE}_d${RANK_DRAFT_PER_NODE}_tp${TARGET_TP_SIZE}_prefix_fuse16_nlayers5_block8_rnn_easy_direct_r512_ep8}"

exec bash scripts/run_training_flashmtp.sh --dt h100
