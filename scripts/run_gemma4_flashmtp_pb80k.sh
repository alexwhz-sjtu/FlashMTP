#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
DATA_PATH=/share/dai-sys/wanghanzhen/projects/MTP/training_data/generated/gemma4-12b/open_perfectblend_80k_think_off_temp0_maxnew4096.jsonl
TARGET_PATH=/share/dai-sys/wanghanzhen/models/google/gemma-4-12B-it
EXPECTED_SAMPLES=${EXPECTED_SAMPLES:-80000}

while true; do
  busy=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | awk '$1 > 1024 || $2 > 5 {n++} END {print n+0}')
  if [ "${busy}" -eq 0 ]; then break; fi
  echo "Waiting for all 8 GPUs to become idle (${busy} busy)..."
  sleep 30
done

actual_samples=$(wc -l < "${DATA_PATH}")
error_samples=$(wc -l < "${DATA_PATH%.jsonl}_error.jsonl")
if [ "${actual_samples}" -ne "${EXPECTED_SAMPLES}" ] || [ "${error_samples}" -ne 0 ]; then
  echo "Training data invalid: success=${actual_samples}/${EXPECTED_SAMPLES}, errors=${error_samples}" >&2
  exit 2
fi

export FLASHMTP_VENV=${PROJECT_DIR}/.venv-gemma4
export NUM_MIDDLE_LAYERS_N=14 NUM_DRAFT_LAYERS=5 NUM_EPOCHS=8
export PIVOT_FUSE_MODE=prefix_condition DATA_NUM_SAMPLES=pb_80k
export MAX_LENGTH=4096 NUM_ANCHORS=512 BLOCK_SIZE=8 LOCAL_POSITION=true
export LOSS_DECAY_GAMMA=4 BASE_LM_CE_DECAY_GAMMA=12
export BASE_LM_CE_WEIGHT=0.06 FINAL_CE_WEIGHT=0.1 TV_LOSS_WEIGHT=1.0
export MARKOV_HEAD_TYPE=rnn_easy MARKOV_OUTPUT_MODE=direct MARKOV_RANK=512
export NPROC_PER_NODE=8 TP_SIZE=1 SHARD_DRAFT_BY_TP=1 CE_CHUNK_SIZE=4096
export LEARNING_RATE=5e-4 TRAIN_DATA_PATH=${DATA_PATH}
export TARGET_MODEL_BACKEND=sglang SGLANG_MEM_FRACTION_STATIC=0.4
export SGLANG_ATTENTION_BACKEND=triton
export TARGET_MODEL=${TARGET_PATH} MODEL_TAG=Gemma4-12B
export CHAT_TEMPLATE=gemma4 ENABLE_THINKING=off
export ATTENTION_BACKEND=flex_attention
export OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/cache/models/flashmtp_gemma4_12b_pb80k_prefix_fuse16_nlayers5_block8_rnn_easy_direct_r512_ep8}

cd "${PROJECT_DIR}"
exec bash scripts/run_training_flashmtp.sh --dt h100
