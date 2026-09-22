#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PASSTHROUGH_ARGS=()
DT="${DT:-}"
while (( $# > 0 )); do
  case "$1" in
    --dt)
      if (( $# < 2 )); then
        echo "--dt requires qz, a800, or h100" >&2
        exit 2
      fi
      DT="$2"
      shift 2
      ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac
done
if [[ -n "${DT}" ]]; then
  case "${DT}" in
    qz|a800|h100) ;;
    *) echo "--dt must be qz, a800, or h100; got ${DT}" >&2; exit 2 ;;
  esac
  if [[ "${DT}" == "qz" ]]; then
    export WANDB_MODE="${WANDB_MODE:-offline}"
  fi
fi

if [[ -z "${PYTHON_BIN:-}" && -x "${PROJECT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

: "${TARGET_MODEL:?set TARGET_MODEL}"
: "${TRAIN_DATA_PATH:?set TRAIN_DATA_PATH}"

NNODES="${PET_NNODES:-${NNODES:-1}}"
NODE_RANK="${PET_NODE_RANK:-${NODE_RANK:-0}}"
NPROC_PER_NODE="${PET_NPROC_PER_NODE:-${NPROC_PER_NODE:-8}}"
MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-29501}}"
export MASTER_ADDR MASTER_PORT

TARGET_MODEL_BACKEND="${TARGET_MODEL_BACKEND:-hf}"
BLOCK_SIZE="${BLOCK_SIZE:-8}"
NUM_DRAFT_LAYERS="${NUM_DRAFT_LAYERS:-5}"
SWA_WINDOW_SIZE="${SWA_WINDOW_SIZE:-32}"
ANCHOR_GROUP_SIZE="${ANCHOR_GROUP_SIZE:-8}"
CHS_NUM_LAYERS="${CHS_NUM_LAYERS:-7}"
MARKOV_HEAD_TYPE="${MARKOV_HEAD_TYPE:-vanilla}"
MARKOV_OUTPUT_MODE="${MARKOV_OUTPUT_MODE:-additive}"
MARKOV_RANK="${MARKOV_RANK:-256}"
NUM_EPOCHS="${NUM_EPOCHS:-6}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.04}"
FINAL_CE_WEIGHT="${FINAL_CE_WEIGHT:-1.0}"
TV_LOSS_WEIGHT="${TV_LOSS_WEIGHT:-1.0}"
BASE_LM_CE_WEIGHT="${BASE_LM_CE_WEIGHT:-0.0}"
MARKOV_TEACHER_FORCING_RATIO="${MARKOV_TEACHER_FORCING_RATIO:-1.0}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
NUM_ANCHORS="${NUM_ANCHORS:-512}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"
TP_SIZE="${TP_SIZE:-1}"
SHARD_DRAFT_BY_TP="${SHARD_DRAFT_BY_TP:-0}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
SAVE_INTERVAL="${SAVE_INTERVAL:-20000}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.4}"
LOSS_DECAY_GAMMA="${LOSS_DECAY_GAMMA:-}"
BASE_LM_CE_DECAY_GAMMA="${BASE_LM_CE_DECAY_GAMMA:-}"

TRAIN_BATCH_SIZE="${BATCH_SIZE:-1}"
if [[ "${SHARD_DRAFT_BY_TP}" == "1" && "${TP_SIZE}" -gt 1 && "${TRAIN_BATCH_SIZE}" -eq 1 ]]; then
  TRAIN_BATCH_SIZE="${TP_SIZE}"
fi
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

slug() {
  local value="$1"
  local limit="$2"
  value="$(printf '%s' "${value}" | sed -E 's/[^[:alnum:]_.-]+/-/g; s/^-+//; s/-+$//; s/-+/-/g')"
  [[ -n "${value}" ]] || value="na"
  printf '%.*s' "${limit}" "${value}"
}

if [[ -n "${MODEL_TAG:-}" ]]; then
  MODEL_TAG="$(slug "${MODEL_TAG}" 24)"
else
  TARGET_BASENAME="${TARGET_MODEL%/}"
  TARGET_BASENAME="${TARGET_BASENAME##*/}"
  MODEL_TAG="$(slug "${TARGET_BASENAME}" 24)"
fi

if [[ -n "${DATA_NUM_SAMPLES:-}" ]]; then
  DATA_TAG="$(slug "${DATA_NUM_SAMPLES}" 24)"
else
  DATA_BASENAME="${TRAIN_DATA_PATH%/}"
  DATA_BASENAME="${DATA_BASENAME##*/}"
  DATA_BASENAME="${DATA_BASENAME%.jsonl}"
  DATA_TAG="$(slug "${DATA_BASENAME}" 28)"
fi

DT_TAG=""
[[ -n "${DT}" ]] && DT_TAG="${DT}_"

RUN_TAG="flashmtp_v23_teacher_${DT_TAG}${MODEL_TAG}_${DATA_TAG}_${NNODES}n${WORLD_SIZE}g_tp${TP_SIZE}_swa${SWA_WINDOW_SIZE}_ag${ANCHOR_GROUP_SIZE}_chs${CHS_NUM_LAYERS}_a${NUM_ANCHORS}_block${BLOCK_SIZE}_d${NUM_DRAFT_LAYERS}_${MARKOV_HEAD_TYPE}_${MARKOV_OUTPUT_MODE}_r${MARKOV_RANK}_maxlen${MAX_LENGTH}_ep${NUM_EPOCHS}"
if [[ -n "${RUN_SUFFIX:-}" ]]; then
  RUN_TAG="$(slug "${RUN_TAG}" 210)_$(slug "${RUN_SUFFIX}" 24)"
else
  RUN_TAG="$(slug "${RUN_TAG}" 240)"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/cache/models}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_TAG}}"

OPTIONAL_ARGS=()
[[ -n "${LOSS_DECAY_GAMMA}" ]] && OPTIONAL_ARGS+=(--loss-decay-gamma "${LOSS_DECAY_GAMMA}")
[[ -n "${BASE_LM_CE_DECAY_GAMMA}" ]] && OPTIONAL_ARGS+=(--base-lm-ce-decay-gamma "${BASE_LM_CE_DECAY_GAMMA}")
[[ -n "${RESUME_FROM:-}" ]] && OPTIONAL_ARGS+=(--resume-from "${RESUME_FROM}")
[[ -n "${INIT_FROM:-}" ]] && OPTIONAL_ARGS+=(--init-from "${INIT_FROM}")
[[ -n "${CHAT_TEMPLATE:-}" ]] && OPTIONAL_ARGS+=(--chat-template "${CHAT_TEMPLATE}")
[[ -n "${MASK_TOKEN_ID:-}" ]] && OPTIONAL_ARGS+=(--mask-token-id "${MASK_TOKEN_ID}")
[[ -n "${CACHE_DIR:-}" ]] && OPTIONAL_ARGS+=(--cache-dir "${CACHE_DIR}")
if [[ "${SHARD_DRAFT_BY_TP}" == "1" ]]; then
  OPTIONAL_ARGS+=(--shard-draft-by-tp)
else
  OPTIONAL_ARGS+=(--no-shard-draft-by-tp)
fi

REPORT_TO="${REPORT_TO:-}"
if [[ -n "${REPORT_TO}" ]]; then
  OPTIONAL_ARGS+=(--report-to "${REPORT_TO}")
  if [[ "${REPORT_TO}" == "wandb" ]]; then
    RUN_HASH="$("${PYTHON_BIN}" -c 'import hashlib, sys; print(hashlib.sha1(sys.argv[1].encode()).hexdigest()[:8])' "${RUN_TAG}")"
    WANDB_PROJECT="${WANDB_PROJECT:-flashmtp-training-v2.3-teacher}"
    WANDB_NAME="${WANDB_RUN_NAME:-${WANDB_NAME:-$(slug "v23t_${DT_TAG}${MODEL_TAG}_${DATA_TAG}_swa${SWA_WINDOW_SIZE}_ag${ANCHOR_GROUP_SIZE}_d${NUM_DRAFT_LAYERS}_${MARKOV_HEAD_TYPE}_${MARKOV_OUTPUT_MODE}_r${MARKOV_RANK}" 110)_${RUN_HASH}}}"
    WANDB_RUN_ID="${WANDB_RUN_ID:-$(slug "v23t-${DT_TAG}${MODEL_TAG}-${DATA_TAG}-${RUN_HASH}" 64)}"
    OPTIONAL_ARGS+=(
      --wandb-project "${WANDB_PROJECT}"
      --wandb-name "${WANDB_NAME}"
      --wandb-run-id "${WANDB_RUN_ID}"
    )
  fi
fi

CMD=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --nnodes "${NNODES}" --node_rank "${NODE_RANK}"
  --nproc_per_node "${NPROC_PER_NODE}"
  --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}"
  -m scripts.train_flashmtp_teacher
  --target-model-path "${TARGET_MODEL}"
  --target-model-backend "${TARGET_MODEL_BACKEND}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
  --train-data-path "${TRAIN_DATA_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --block-size "${BLOCK_SIZE}"
  --num-draft-layers "${NUM_DRAFT_LAYERS}"
  --swa-window-size "${SWA_WINDOW_SIZE}"
  --anchor-group-size "${ANCHOR_GROUP_SIZE}"
  --chs-num-layers "${CHS_NUM_LAYERS}"
  --markov-head-type "${MARKOV_HEAD_TYPE}"
  --markov-output-mode "${MARKOV_OUTPUT_MODE}"
  --markov-rank "${MARKOV_RANK}"
  --num-epochs "${NUM_EPOCHS}"
  --learning-rate "${LEARNING_RATE}"
  --warmup-ratio "${WARMUP_RATIO}"
  --final-ce-weight "${FINAL_CE_WEIGHT}"
  --tv-loss-weight "${TV_LOSS_WEIGHT}"
  --base-lm-ce-weight "${BASE_LM_CE_WEIGHT}"
  --markov-teacher-forcing-ratio "${MARKOV_TEACHER_FORCING_RATIO}"
  --batch-size "${TRAIN_BATCH_SIZE}"
  --max-length "${MAX_LENGTH}"
  --num-anchors "${NUM_ANCHORS}"
  --accumulation-steps "${ACCUMULATION_STEPS}"
  --log-interval "${LOG_INTERVAL}"
  --save-interval "${SAVE_INTERVAL}"
  --tp-size "${TP_SIZE}"
  "${OPTIONAL_ARGS[@]}"
  "${PASSTHROUGH_ARGS[@]}"
)

printf 'FlashMTP v2.3 teacher output: %s\n' "${OUTPUT_DIR}"
printf 'Launching teacher:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

mkdir -p "${OUTPUT_DIR}"
exec "${CMD[@]}"
