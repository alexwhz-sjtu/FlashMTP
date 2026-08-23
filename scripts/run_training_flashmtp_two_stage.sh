#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  NPROC_PER_NODE="$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')"
fi
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29502}"
TP_SIZE="${TP_SIZE:-1}"
SHARD_DRAFT_BY_TP="${SHARD_DRAFT_BY_TP:-0}"

is_nonnegative_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

for item in \
  "NNODES:${NNODES}" \
  "NODE_RANK:${NODE_RANK}" \
  "NPROC_PER_NODE:${NPROC_PER_NODE}" \
  "MASTER_PORT:${MASTER_PORT}" \
  "TP_SIZE:${TP_SIZE}"; do
  name="${item%%:*}"
  value="${item#*:}"
  if ! is_nonnegative_integer "${value}"; then
    echo "${name} must be an integer, got ${value}" >&2
    exit 2
  fi
done
if (( NNODES < 1 || NPROC_PER_NODE < 1 || TP_SIZE < 1 )); then
  echo "NNODES, NPROC_PER_NODE, and TP_SIZE must be positive" >&2
  exit 2
fi
if (( NODE_RANK >= NNODES )); then
  echo "NODE_RANK=${NODE_RANK} must be smaller than NNODES=${NNODES}" >&2
  exit 2
fi
if (( (NNODES * NPROC_PER_NODE) % TP_SIZE != 0 )); then
  echo "World size must be divisible by TP_SIZE" >&2
  exit 2
fi
if (( NPROC_PER_NODE % TP_SIZE != 0 )); then
  echo "NPROC_PER_NODE must be divisible by TP_SIZE so TP groups stay within one node" >&2
  exit 2
fi
if (( NNODES > 1 )) && [[ "${MASTER_ADDR}" == "127.0.0.1" || "${MASTER_ADDR}" == "localhost" || "${MASTER_ADDR}" == "0.0.0.0" ]]; then
  echo "Multi-node training requires a reachable MASTER_ADDR, got ${MASTER_ADDR}" >&2
  exit 2
fi

: "${TARGET_MODEL:?set TARGET_MODEL}"
: "${TRAIN_DATA_PATH:?set TRAIN_DATA_PATH}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
: "${STAGE1_EPOCHS:?set STAGE1_EPOCHS}"
: "${STAGE1_LEARNING_RATE:?set STAGE1_LEARNING_RATE}"
: "${STAGE2_EPOCHS:?set STAGE2_EPOCHS}"
: "${STAGE2_LEARNING_RATE:?set STAGE2_LEARNING_RATE}"

if [[ -z "${RESUME_FROM:-}" && -z "${TEACHER_DRAFT_PATH:-}" ]]; then
  echo "Fresh two-stage training requires TEACHER_DRAFT_PATH" >&2
  exit 2
fi

if [[ "${SHARD_DRAFT_BY_TP}" != "0" && "${SHARD_DRAFT_BY_TP}" != "1" ]]; then
  echo "SHARD_DRAFT_BY_TP must be 0 or 1" >&2
  exit 2
fi
TRAIN_BATCH_SIZE="${BATCH_SIZE:-1}"
if ! is_nonnegative_integer "${TRAIN_BATCH_SIZE}" || (( TRAIN_BATCH_SIZE < 1 )); then
  echo "BATCH_SIZE must be a positive integer" >&2
  exit 2
fi
if [[ "${SHARD_DRAFT_BY_TP}" == "1" ]]; then
  if [[ "${TARGET_MODEL_BACKEND:-hf}" != "sglang" ]]; then
    echo "SHARD_DRAFT_BY_TP=1 requires TARGET_MODEL_BACKEND=sglang" >&2
    exit 2
  fi
  if (( TP_SIZE <= 1 )); then
    echo "SHARD_DRAFT_BY_TP=1 requires TP_SIZE > 1" >&2
    exit 2
  fi
  if (( TRAIN_BATCH_SIZE != 1 && TRAIN_BATCH_SIZE != TP_SIZE )); then
    echo "With SHARD_DRAFT_BY_TP=1, BATCH_SIZE must be 1 (auto-expand) or TP_SIZE" >&2
    exit 2
  fi
  # BATCH_SIZE is the shared target-prefill batch.  Each TP rank receives one
  # distinct rank-local sample after prefill.
  TRAIN_BATCH_SIZE="${TP_SIZE}"
fi

OPTIONAL_ARGS=()
[[ -n "${TEACHER_DRAFT_PATH:-}" ]] && OPTIONAL_ARGS+=(--teacher-draft-path "${TEACHER_DRAFT_PATH}")
[[ -n "${STUDENT_INIT_MODE:-}" ]] && OPTIONAL_ARGS+=(--student-init-mode "${STUDENT_INIT_MODE}")
[[ -n "${STAGE1_LOSS_DECAY_GAMMA:-}" ]] && OPTIONAL_ARGS+=(--stage1-loss-decay-gamma "${STAGE1_LOSS_DECAY_GAMMA}")
[[ -n "${STAGE2_LOSS_DECAY_GAMMA:-}" ]] && OPTIONAL_ARGS+=(--stage2-loss-decay-gamma "${STAGE2_LOSS_DECAY_GAMMA}")
[[ -n "${STAGE2_BASE_CE_DECAY_GAMMA:-}" ]] && OPTIONAL_ARGS+=(--stage2-base-ce-decay-gamma "${STAGE2_BASE_CE_DECAY_GAMMA}")
[[ -n "${RESUME_FROM:-}" ]] && OPTIONAL_ARGS+=(--resume-from "${RESUME_FROM}")
[[ -n "${MASK_TOKEN_ID:-}" ]] && OPTIONAL_ARGS+=(--mask-token-id "${MASK_TOKEN_ID}")
[[ -n "${CHAT_TEMPLATE:-}" ]] && OPTIONAL_ARGS+=(--chat-template "${CHAT_TEMPLATE}")
[[ -n "${SGLANG_ATTENTION_BACKEND:-}" ]] && OPTIONAL_ARGS+=(--sglang-attention-backend "${SGLANG_ATTENTION_BACKEND}")
[[ -n "${SGLANG_CONTEXT_LENGTH:-}" ]] && OPTIONAL_ARGS+=(--sglang-context-length "${SGLANG_CONTEXT_LENGTH}")
[[ -n "${SGLANG_MAX_RUNNING_REQUESTS:-}" ]] && OPTIONAL_ARGS+=(--sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}")
[[ -n "${SGLANG_MAX_TOTAL_TOKENS:-}" ]] && OPTIONAL_ARGS+=(--sglang-max-total-tokens "${SGLANG_MAX_TOTAL_TOKENS}")
[[ "${SHARD_DRAFT_BY_TP}" == "1" ]] && OPTIONAL_ARGS+=(--shard-draft-by-tp)
[[ "${IS_PREFORMATTED:-0}" == "1" ]] && OPTIONAL_ARGS+=(--is-preformatted)
[[ "${TRUST_REMOTE_CODE:-0}" == "1" ]] && OPTIONAL_ARGS+=(--trust-remote-code)

CMD=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --nnodes "${NNODES}" --node_rank "${NODE_RANK}"
  --nproc_per_node "${NPROC_PER_NODE}"
  --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}"
  -m scripts.train_flashmtp_two_stage
  --target-model-path "${TARGET_MODEL}"
  --target-model-backend "${TARGET_MODEL_BACKEND:-hf}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.4}"
  --train-data-path "${TRAIN_DATA_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --stage1-epochs "${STAGE1_EPOCHS}"
  --stage1-learning-rate "${STAGE1_LEARNING_RATE}"
  --stage1-warmup-ratio "${STAGE1_WARMUP_RATIO:-0.04}"
  --stage1-tv-weight "${STAGE1_TV_WEIGHT:-1.0}"
  --stage1-hidden-weight "${STAGE1_HIDDEN_WEIGHT:-1.0}"
  --stage1-smooth-l1-beta "${STAGE1_SMOOTH_L1_BETA:-1.0}"
  --stage2-epochs "${STAGE2_EPOCHS}"
  --stage2-learning-rate "${STAGE2_LEARNING_RATE}"
  --stage2-warmup-ratio "${STAGE2_WARMUP_RATIO:-0.04}"
  --stage2-final-ce-weight "${STAGE2_FINAL_CE_WEIGHT:-1.0}"
  --stage2-tv-weight "${STAGE2_TV_WEIGHT:-1.0}"
  --stage2-base-ce-weight "${STAGE2_BASE_CE_WEIGHT:-0.0}"
  --batch-size "${TRAIN_BATCH_SIZE}"
  --max-length "${MAX_LENGTH:-4096}"
  --num-anchors "${NUM_ANCHORS:-512}"
  --accumulation-steps "${ACCUMULATION_STEPS:-1}"
  --max-grad-norm "${MAX_GRAD_NORM:-1.0}"
  --seed "${SEED:-42}"
  --dist-timeout "${DIST_TIMEOUT:-1200}"
  --cache-dir "${CACHE_DIR:-./cache/train}"
  --build-dataset-num-proc "${BUILD_DATASET_NUM_PROC:-8}"
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS:-8}"
  --log-interval "${LOG_INTERVAL:-50}"
  --save-interval "${SAVE_INTERVAL:-20000}"
  --tp-size "${TP_SIZE}"
  "${OPTIONAL_ARGS[@]}"
  "$@"
)

printf 'Launching two-stage FlashMTP student:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

VISIBLE_GPUS="$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')"
if ! is_nonnegative_integer "${VISIBLE_GPUS}" || (( VISIBLE_GPUS < NPROC_PER_NODE )); then
  echo "Requested ${NPROC_PER_NODE} processes but only ${VISIBLE_GPUS} CUDA devices are visible" >&2
  exit 2
fi

exec "${CMD[@]}"
