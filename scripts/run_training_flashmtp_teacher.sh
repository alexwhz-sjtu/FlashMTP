#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"

OPTIONAL_ARGS=()
[[ -n "${LOSS_DECAY_GAMMA:-}" ]] && OPTIONAL_ARGS+=(--loss-decay-gamma "${LOSS_DECAY_GAMMA}")
[[ -n "${BASE_LM_CE_DECAY_GAMMA:-}" ]] && OPTIONAL_ARGS+=(--base-lm-ce-decay-gamma "${BASE_LM_CE_DECAY_GAMMA}")
[[ -n "${RESUME_FROM:-}" ]] && OPTIONAL_ARGS+=(--resume-from "${RESUME_FROM}")
[[ -n "${INIT_FROM:-}" ]] && OPTIONAL_ARGS+=(--init-from "${INIT_FROM}")
[[ -n "${CHAT_TEMPLATE:-}" ]] && OPTIONAL_ARGS+=(--chat-template "${CHAT_TEMPLATE}")
if [[ "${SHARD_DRAFT_BY_TP:-0}" == "1" ]]; then
  OPTIONAL_ARGS+=(--shard-draft-by-tp)
else
  OPTIONAL_ARGS+=(--no-shard-draft-by-tp)
fi

TRAIN_BATCH_SIZE="${BATCH_SIZE:-1}"
if [[ "${SHARD_DRAFT_BY_TP:-0}" == "1" && "${TP_SIZE:-1}" -gt 1 ]]; then
  if [[ "${TRAIN_BATCH_SIZE}" -eq 1 ]]; then
    TRAIN_BATCH_SIZE="${TP_SIZE}"
  fi
fi

exec "${PYTHON_BIN}" -m torch.distributed.run \
  --nnodes "${NNODES}" --node_rank "${NODE_RANK}" \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}" \
  -m scripts.train_flashmtp_teacher \
  --target-model-path "${TARGET_MODEL:?set TARGET_MODEL}" \
  --target-model-backend "${TARGET_MODEL_BACKEND:-hf}" \
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.4}" \
  --train-data-path "${TRAIN_DATA_PATH:?set TRAIN_DATA_PATH}" \
  --output-dir "${OUTPUT_DIR:?set OUTPUT_DIR}" \
  --block-size "${BLOCK_SIZE:-8}" \
  --num-draft-layers "${NUM_DRAFT_LAYERS:-5}" \
  --swa-window-size "${SWA_WINDOW_SIZE:-32}" \
  --anchor-group-size "${ANCHOR_GROUP_SIZE:-8}" \
  --chs-num-layers "${CHS_NUM_LAYERS:-7}" \
  --markov-head-type "${MARKOV_HEAD_TYPE:-vanilla}" \
  --markov-output-mode "${MARKOV_OUTPUT_MODE:-additive}" \
  --markov-rank "${MARKOV_RANK:-256}" \
  --num-epochs "${NUM_EPOCHS:-6}" \
  --learning-rate "${LEARNING_RATE:-5e-4}" \
  --warmup-ratio "${WARMUP_RATIO:-0.04}" \
  --final-ce-weight "${FINAL_CE_WEIGHT:-1.0}" \
  --tv-loss-weight "${TV_LOSS_WEIGHT:-1.0}" \
  --base-lm-ce-weight "${BASE_LM_CE_WEIGHT:-0.0}" \
  --markov-teacher-forcing-ratio "${MARKOV_TEACHER_FORCING_RATIO:-1.0}" \
  --batch-size "${TRAIN_BATCH_SIZE}" \
  --max-length "${MAX_LENGTH:-4096}" \
  --num-anchors "${NUM_ANCHORS:-512}" \
  --accumulation-steps "${ACCUMULATION_STEPS:-1}" \
  --log-interval "${LOG_INTERVAL:-50}" \
  --save-interval "${SAVE_INTERVAL:-20000}" \
  --tp-size "${TP_SIZE:-1}" \
  "${OPTIONAL_ARGS[@]}" \
  "$@"
