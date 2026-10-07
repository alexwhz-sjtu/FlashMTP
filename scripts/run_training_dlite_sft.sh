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
  [[ "${DT}" == "qz" ]] && export WANDB_MODE="${WANDB_MODE:-offline}"
fi

if [[ -z "${PYTHON_BIN:-}" && -x "${PROJECT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
PYTHON_EXECUTABLE="$(command -v "${PYTHON_BIN}" 2>/dev/null || printf '%s' "${PYTHON_BIN}")"
export PATH="$(cd "$(dirname "${PYTHON_EXECUTABLE}")" && pwd):${PATH}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi

: "${TARGET_MODEL:?set TARGET_MODEL}"
if [[ -n "${TRAIN_DATA_PATH:-}" && -n "${TRAIN_HIDDEN_STATES_PATH:-}" ]] || \
   [[ -z "${TRAIN_DATA_PATH:-}" && -z "${TRAIN_HIDDEN_STATES_PATH:-}" ]]; then
  echo "Set exactly one of TRAIN_DATA_PATH or TRAIN_HIDDEN_STATES_PATH" >&2
  exit 2
fi
TRAIN_INPUT_PATH="${TRAIN_HIDDEN_STATES_PATH:-${TRAIN_DATA_PATH:-}}"

NNODES="${PET_NNODES:-${NNODES:-1}}"
NODE_RANK="${PET_NODE_RANK:-${NODE_RANK:-0}}"
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  NPROC_PER_NODE="${PET_NPROC_PER_NODE:-}"
fi
if [[ -z "${NPROC_PER_NODE}" ]]; then
  NPROC_PER_NODE="$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')"
fi
MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-29503}}"
export MASTER_ADDR MASTER_PORT

if [[ -z "${TARGET_MODEL_BACKEND:-}" ]]; then
  case "${TARGET_MODEL%/}" in
    *Qwen3.5-*) TARGET_MODEL_BACKEND="sglang" ;;
    *) TARGET_MODEL_BACKEND="hf" ;;
  esac
fi
if [[ -z "${SGLANG_ATTENTION_BACKEND:-}" && "${TARGET_MODEL%/}" == *Qwen3.5-* ]]; then
  SGLANG_ATTENTION_BACKEND="fa3"
fi
if [[ -z "${CHAT_TEMPLATE:-}" && "${TARGET_MODEL%/}" == *Qwen3.5-* ]]; then
  CHAT_TEMPLATE="qwen3.5"
fi
DLITE_VERSION="${DLITE_VERSION:-dlite_v2}"
LOCAL_POSITION="${LOCAL_POSITION:-true}"
BLOCK_SIZE="${BLOCK_SIZE:-8}"
NUM_DRAFT_LAYERS="${NUM_DRAFT_LAYERS:-5}"
CHS_NUM_LAYERS="${CHS_NUM_LAYERS:-7}"
TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-0,1,3,7,11,15,19,23,27,29,30,31}"
if [[ -n "${TARGET_LAYER_IDS}" ]]; then
  IFS=',' read -r -a _TARGET_LAYER_ID_ARRAY <<< "${TARGET_LAYER_IDS}"
  CHS_NUM_LAYERS="${#_TARGET_LAYER_ID_ARRAY[@]}"
fi
SEQUENTIAL_HEAD="${SEQUENTIAL_HEAD:-rnn}"
SEQUENTIAL_RANK="${SEQUENTIAL_RANK:-256}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
LEARNING_RATE="${LEARNING_RATE:-4e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.04}"
FINAL_CE_WEIGHT="${FINAL_CE_WEIGHT:-0.1}"
TV_LOSS_WEIGHT="${TV_LOSS_WEIGHT:-1.0}"
BASE_LM_CE_WEIGHT="${BASE_LM_CE_WEIGHT:-0.0}"
LOSS_DECAY_GAMMA="${LOSS_DECAY_GAMMA:-}"
BASE_LM_CE_DECAY_GAMMA="${BASE_LM_CE_DECAY_GAMMA:-}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
NUM_ANCHORS="${NUM_ANCHORS:-512}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"
TP_SIZE="${TP_SIZE:-1}"
SHARD_DRAFT_BY_TP="${SHARD_DRAFT_BY_TP:-0}"
if [[ -n "${TRAIN_HIDDEN_STATES_PATH:-}" && ( "${TP_SIZE}" != "1" || "${SHARD_DRAFT_BY_TP}" != "0" ) ]]; then
  echo "Offline regen_full training requires TP_SIZE=1 and SHARD_DRAFT_BY_TP=0" >&2
  exit 2
fi
REQUESTED_BATCH_SIZE="${BATCH_SIZE:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
SAVE_INTERVAL="${SAVE_INTERVAL:-20000}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.4}"

is_nonnegative_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

for item in \
  "NNODES:${NNODES}" \
  "NODE_RANK:${NODE_RANK}" \
  "NPROC_PER_NODE:${NPROC_PER_NODE}" \
  "MASTER_PORT:${MASTER_PORT}" \
  "TP_SIZE:${TP_SIZE}" \
  "BATCH_SIZE:${REQUESTED_BATCH_SIZE}"; do
  name="${item%%:*}"
  value="${item#*:}"
  if ! is_nonnegative_integer "${value}"; then
    echo "${name} must be an integer, got ${value}" >&2
    exit 2
  fi
done
if (( NNODES < 1 || NPROC_PER_NODE < 1 || TP_SIZE < 1 || REQUESTED_BATCH_SIZE < 1 )); then
  echo "NNODES, NPROC_PER_NODE, TP_SIZE, and BATCH_SIZE must be positive" >&2
  exit 2
fi
if (( NODE_RANK >= NNODES )); then
  echo "NODE_RANK=${NODE_RANK} must be smaller than NNODES=${NNODES}" >&2
  exit 2
fi
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
if (( WORLD_SIZE % TP_SIZE != 0 || NPROC_PER_NODE % TP_SIZE != 0 )); then
  echo "World size and NPROC_PER_NODE must both be divisible by TP_SIZE" >&2
  exit 2
fi
if (( NNODES > 1 )) && [[ "${MASTER_ADDR}" == "127.0.0.1" || "${MASTER_ADDR}" == "localhost" || "${MASTER_ADDR}" == "0.0.0.0" ]]; then
  echo "Multi-node training requires a reachable MASTER_ADDR" >&2
  exit 2
fi
if [[ "${SHARD_DRAFT_BY_TP}" != "0" && "${SHARD_DRAFT_BY_TP}" != "1" ]]; then
  echo "SHARD_DRAFT_BY_TP must be 0 or 1" >&2
  exit 2
fi
case "${LOCAL_POSITION,,}" in
  true|1|yes|on) ;;
  *) echo "DLite student SFT requires LOCAL_POSITION=true" >&2; exit 2 ;;
esac

TRAIN_BATCH_SIZE="${REQUESTED_BATCH_SIZE}"
if [[ "${SHARD_DRAFT_BY_TP}" == "1" ]]; then
  if [[ "${TARGET_MODEL_BACKEND}" != "sglang" || "${TP_SIZE}" -le 1 ]]; then
    echo "SHARD_DRAFT_BY_TP=1 requires sglang and TP_SIZE > 1" >&2
    exit 2
  fi
  if (( TRAIN_BATCH_SIZE != 1 && TRAIN_BATCH_SIZE != TP_SIZE )); then
    echo "With SHARD_DRAFT_BY_TP=1, BATCH_SIZE must be 1 or TP_SIZE" >&2
    exit 2
  fi
  TRAIN_BATCH_SIZE="${TP_SIZE}"
fi

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
  MODEL_TAG="$(slug "${TARGET_BASENAME##*/}" 24)"
fi
if [[ -n "${DATA_NUM_SAMPLES:-}" ]]; then
  DATA_TAG="$(slug "${DATA_NUM_SAMPLES}" 28)"
else
  DATA_BASENAME="${TRAIN_INPUT_PATH%/}"
  DATA_BASENAME="${DATA_BASENAME##*/}"
  DATA_TAG="$(slug "${DATA_BASENAME%.jsonl}" 28)"
fi
DT_TAG=""
[[ -n "${DT}" ]] && DT_TAG="${DT}_"

RUN_TAG="${DLITE_VERSION}_sft_${DT_TAG}${MODEL_TAG}_${DATA_TAG}_${NNODES}n${WORLD_SIZE}g_tp${TP_SIZE}_sh${SHARD_DRAFT_BY_TP}_chs${CHS_NUM_LAYERS}_a${NUM_ANCHORS}_block${BLOCK_SIZE}_d${NUM_DRAFT_LAYERS}_${SEQUENTIAL_HEAD}_r${SEQUENTIAL_RANK}_maxlen${MAX_LENGTH}_ep${NUM_EPOCHS}_lr${LEARNING_RATE}_ce${FINAL_CE_WEIGHT}_tv${TV_LOSS_WEIGHT}_base${BASE_LM_CE_WEIGHT}"
if [[ -n "${RUN_SUFFIX:-}" ]]; then
  RUN_TAG="$(slug "${RUN_TAG}" 210)_$(slug "${RUN_SUFFIX}" 24)"
else
  RUN_TAG="$(slug "${RUN_TAG}" 240)"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/cache/models}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_TAG}}"
CACHE_DIR="${CACHE_DIR:-${PROJECT_DIR}/cache/train/${DATA_TAG}_l${MAX_LENGTH}_m${MASK_TOKEN_ID:-auto}}"
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-dlite-training}"
RUN_HASH="$("${PYTHON_BIN}" -c 'import hashlib, sys; print(hashlib.sha1(sys.argv[1].encode()).hexdigest()[:8])' "${RUN_TAG}")"
WANDB_NAME="${WANDB_RUN_NAME:-${WANDB_NAME:-$(slug "${DLITE_VERSION}_sft_${DT_TAG}${MODEL_TAG}_${DATA_TAG}_d${NUM_DRAFT_LAYERS}_${SEQUENTIAL_HEAD}_r${SEQUENTIAL_RANK}" 110)_${RUN_HASH}}}"
WANDB_RUN_ID="${WANDB_RUN_ID:-$(slug "${DLITE_VERSION}-sft-${DT_TAG}${MODEL_TAG}-${DATA_TAG}-${RUN_HASH}" 64)}"

OPTIONAL_ARGS=(--local-position)
if [[ -n "${TRAIN_HIDDEN_STATES_PATH:-}" ]]; then
  OPTIONAL_ARGS+=(--train-hidden-states-path "${TRAIN_HIDDEN_STATES_PATH}")
else
  OPTIONAL_ARGS+=(--train-data-path "${TRAIN_DATA_PATH}")
fi
[[ -n "${LOSS_DECAY_GAMMA}" ]] && OPTIONAL_ARGS+=(--loss-decay-gamma "${LOSS_DECAY_GAMMA}")
[[ -n "${BASE_LM_CE_DECAY_GAMMA}" ]] && OPTIONAL_ARGS+=(--base-lm-ce-decay-gamma "${BASE_LM_CE_DECAY_GAMMA}")
[[ -n "${RESUME_FROM:-}" ]] && OPTIONAL_ARGS+=(--resume-from "${RESUME_FROM}")
[[ -n "${INIT_FROM:-}" ]] && OPTIONAL_ARGS+=(--init-from "${INIT_FROM}")
[[ -n "${MASK_TOKEN_ID:-}" ]] && OPTIONAL_ARGS+=(--mask-token-id "${MASK_TOKEN_ID}")
[[ -n "${EMBEDDING_KEY:-}" ]] && OPTIONAL_ARGS+=(--embedding-key "${EMBEDDING_KEY}")
[[ -n "${LM_HEAD_KEY:-}" ]] && OPTIONAL_ARGS+=(--lm-head-key "${LM_HEAD_KEY}")
[[ -n "${CHAT_TEMPLATE:-}" ]] && OPTIONAL_ARGS+=(--chat-template "${CHAT_TEMPLATE}")
[[ -n "${TARGET_LAYER_IDS}" ]] && OPTIONAL_ARGS+=(--target-layer-ids "${TARGET_LAYER_IDS}")
[[ -n "${SGLANG_ATTENTION_BACKEND:-}" ]] && OPTIONAL_ARGS+=(--sglang-attention-backend "${SGLANG_ATTENTION_BACKEND}")
[[ -n "${SGLANG_CONTEXT_LENGTH:-}" ]] && OPTIONAL_ARGS+=(--sglang-context-length "${SGLANG_CONTEXT_LENGTH}")
[[ -n "${SGLANG_MAX_RUNNING_REQUESTS:-}" ]] && OPTIONAL_ARGS+=(--sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}")
[[ -n "${SGLANG_MAX_TOTAL_TOKENS:-}" ]] && OPTIONAL_ARGS+=(--sglang-max-total-tokens "${SGLANG_MAX_TOTAL_TOKENS}")
[[ "${SHARD_DRAFT_BY_TP}" == "1" ]] && OPTIONAL_ARGS+=(--shard-draft-by-tp)
[[ "${SHARD_DRAFT_BY_TP}" == "0" ]] && OPTIONAL_ARGS+=(--no-shard-draft-by-tp)
[[ "${IS_PREFORMATTED:-0}" == "1" ]] && OPTIONAL_ARGS+=(--is-preformatted)
[[ "${PAD_TO_MAX_LENGTH:-0}" == "1" ]] && OPTIONAL_ARGS+=(--pad-to-max-length)
[[ "${TRUST_REMOTE_CODE:-0}" == "1" ]] && OPTIONAL_ARGS+=(--trust-remote-code)
OPTIONAL_ARGS+=(--report-to "${REPORT_TO}")
if [[ "${REPORT_TO}" == "wandb" ]]; then
  OPTIONAL_ARGS+=(
    --wandb-project "${WANDB_PROJECT}"
    --wandb-name "${WANDB_NAME}"
    --wandb-run-id "${WANDB_RUN_ID}"
  )
fi

CMD=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --nnodes "${NNODES}" --node_rank "${NODE_RANK}"
  --nproc_per_node "${NPROC_PER_NODE}"
  --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}"
  -m scripts.train_dlite_sft
  --target-model-path "${TARGET_MODEL}"
  --target-model-backend "${TARGET_MODEL_BACKEND}"
  --dlite-version "${DLITE_VERSION}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
  --output-dir "${OUTPUT_DIR}"
  --block-size "${BLOCK_SIZE}"
  --num-draft-layers "${NUM_DRAFT_LAYERS}"
  --chs-num-layers "${CHS_NUM_LAYERS}"
  --sequential-head "${SEQUENTIAL_HEAD}"
  --sequential-rank "${SEQUENTIAL_RANK}"
  --num-epochs "${NUM_EPOCHS}"
  --learning-rate "${LEARNING_RATE}"
  --warmup-ratio "${WARMUP_RATIO}"
  --final-ce-weight "${FINAL_CE_WEIGHT}"
  --tv-loss-weight "${TV_LOSS_WEIGHT}"
  --base-lm-ce-weight "${BASE_LM_CE_WEIGHT}"
  --batch-size "${TRAIN_BATCH_SIZE}"
  --max-length "${MAX_LENGTH}"
  --num-anchors "${NUM_ANCHORS}"
  --accumulation-steps "${ACCUMULATION_STEPS}"
  --max-grad-norm "${MAX_GRAD_NORM:-1.0}"
  --seed "${SEED:-42}"
  --dist-timeout "${DIST_TIMEOUT:-1200}"
  --cache-dir "${CACHE_DIR}"
  --build-dataset-num-proc "${BUILD_DATASET_NUM_PROC:-8}"
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS:-8}"
  --log-interval "${LOG_INTERVAL}"
  --save-interval "${SAVE_INTERVAL}"
  --tp-size "${TP_SIZE}"
  "${OPTIONAL_ARGS[@]}"
  "${PASSTHROUGH_ARGS[@]}"
)

printf 'DLite v2.3 SFT config: nodes=%s rank=%s gpus/node=%s world=%s tp=%s local_position=true\n' \
  "${NNODES}" "${NODE_RANK}" "${NPROC_PER_NODE}" "${WORLD_SIZE}" "${TP_SIZE}"
printf 'Output directory: %s\nTraining dataset: %s\nDataset cache: %s\n' \
  "${OUTPUT_DIR}" "${TRAIN_INPUT_PATH}" "${CACHE_DIR}"
if [[ "${REPORT_TO}" == "wandb" ]]; then
  printf 'W&B project: %s\nW&B name: %s\nW&B run id: %s\n' \
    "${WANDB_PROJECT}" "${WANDB_NAME}" "${WANDB_RUN_ID}"
fi
printf 'Launching direct DLite student SFT:'
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
