#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PASSTHROUGH_ARGS=()
DT="${DT:-a800}"
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
case "${DT}" in
  qz|a800|h100) ;;
  *) echo "--dt must be qz, a800, or h100; got ${DT}" >&2; exit 2 ;;
esac
if [[ "${DT}" == "qz" ]]; then
  export WANDB_MODE="${WANDB_MODE:-offline}"
fi

if [[ -z "${PYTHON_BIN:-}" && -x "${PROJECT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  NPROC_PER_NODE="${PET_NPROC_PER_NODE:-}"
fi
if [[ -z "${NPROC_PER_NODE}" ]]; then
  NPROC_PER_NODE="$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')"
fi
NNODES="${PET_NNODES:-${NNODES:-1}}"
NODE_RANK="${PET_NODE_RANK:-${NODE_RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-29502}}"
TP_SIZE="${TP_SIZE:-1}"
SHARD_DRAFT_BY_TP="${SHARD_DRAFT_BY_TP:-0}"
export MASTER_ADDR MASTER_PORT

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
STAGE1_TRAIN_DATA_PATH="${STAGE1_TRAIN_DATA_PATH:-${TRAIN_DATA_PATH:-}}"
STAGE2_TRAIN_DATA_PATH="${STAGE2_TRAIN_DATA_PATH:-${TRAIN_DATA_PATH:-}}"
: "${STAGE1_TRAIN_DATA_PATH:?set STAGE1_TRAIN_DATA_PATH (or legacy TRAIN_DATA_PATH)}"
: "${STAGE2_TRAIN_DATA_PATH:?set STAGE2_TRAIN_DATA_PATH (or legacy TRAIN_DATA_PATH)}"
: "${STAGE1_EPOCHS:?set STAGE1_EPOCHS}"
: "${STAGE2_EPOCHS:?set STAGE2_EPOCHS}"

LEARNING_RATE="${LEARNING_RATE:-${STAGE1_LEARNING_RATE:-}}"
: "${LEARNING_RATE:?set LEARNING_RATE (legacy STAGE1_LEARNING_RATE is accepted)}"
if [[ -n "${STAGE2_LEARNING_RATE:-}" && "${STAGE2_LEARNING_RATE}" != "${LEARNING_RATE}" ]]; then
  echo "Ignoring STAGE2_LEARNING_RATE=${STAGE2_LEARNING_RATE}; both stages use LEARNING_RATE=${LEARNING_RATE}." >&2
fi

if [[ -z "${RESUME_FROM:-}" && -z "${TEACHER_DRAFT_PATH:-}" ]]; then
  echo "Fresh two-stage training requires TEACHER_DRAFT_PATH" >&2
  exit 2
fi

TARGET_MODEL_BACKEND="${TARGET_MODEL_BACKEND:-hf}"
MASK_TOKEN_ID="${MASK_TOKEN_ID:-151669}"
STUDENT_INIT_MODE="${STUDENT_INIT_MODE:-shared_init}"
STUDENT_NUM_DRAFT_LAYERS="${STUDENT_NUM_DRAFT_LAYERS:-${NUM_DRAFT_LAYERS:-}}"
WARMUP_RATIO="${WARMUP_RATIO:-${STAGE1_WARMUP_RATIO:-0.04}}"
if [[ -n "${STAGE2_WARMUP_RATIO:-}" && "${STAGE2_WARMUP_RATIO}" != "${WARMUP_RATIO}" ]]; then
  echo "Ignoring STAGE2_WARMUP_RATIO=${STAGE2_WARMUP_RATIO}; warmup runs once with WARMUP_RATIO=${WARMUP_RATIO}." >&2
fi
STAGE1_KL_WEIGHT="${STAGE1_KL_WEIGHT:-${STAGE1_TV_WEIGHT:-1.0}}"
STAGE1_HIDDEN_WEIGHT="${STAGE1_HIDDEN_WEIGHT:-1.0}"
STAGE1_SMOOTH_L1_BETA="${STAGE1_SMOOTH_L1_BETA:-1.0}"
STAGE1_LOSS_DECAY_GAMMA="${STAGE1_LOSS_DECAY_GAMMA:-}"
STAGE2_FINAL_CE_WEIGHT="${STAGE2_FINAL_CE_WEIGHT:-1.0}"
STAGE2_TV_WEIGHT="${STAGE2_TV_WEIGHT:-1.0}"
STAGE2_BASE_CE_WEIGHT="${STAGE2_BASE_CE_WEIGHT:-0.0}"
STAGE2_LOSS_DECAY_GAMMA="${STAGE2_LOSS_DECAY_GAMMA:-}"
STAGE2_BASE_CE_DECAY_GAMMA="${STAGE2_BASE_CE_DECAY_GAMMA:-}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
NUM_ANCHORS="${NUM_ANCHORS:-512}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"
REQUESTED_BATCH_SIZE="${BATCH_SIZE:-1}"
NAME_TARGET_BATCH="${REQUESTED_BATCH_SIZE}"
NAME_DRAFT_BATCH="${REQUESTED_BATCH_SIZE}"
if [[ "${SHARD_DRAFT_BY_TP}" == "1" ]]; then
  NAME_TARGET_BATCH="${TP_SIZE}"
  NAME_DRAFT_BATCH="1"
fi
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

if [[ -n "${STUDENT_NUM_DRAFT_LAYERS}" ]]; then
  if ! is_nonnegative_integer "${STUDENT_NUM_DRAFT_LAYERS}" || (( STUDENT_NUM_DRAFT_LAYERS < 1 )); then
    echo "STUDENT_NUM_DRAFT_LAYERS must be a positive integer" >&2
    exit 2
  fi
fi
if [[ -z "${RESUME_FROM:-}" && "${STUDENT_INIT_MODE}" == "shared_partial" && -z "${STUDENT_NUM_DRAFT_LAYERS}" ]]; then
  echo "Fresh shared_partial training requires STUDENT_NUM_DRAFT_LAYERS" >&2
  exit 2
fi
STUDENT_DEPTH_TAG=""
if [[ "${STUDENT_INIT_MODE}" == "shared_partial" ]]; then
  STUDENT_DEPTH_TAG="_sd${STUDENT_NUM_DRAFT_LAYERS:-checkpoint}"
fi

slug() {
  local value="$1"
  local limit="$2"
  value="$(printf '%s' "${value}" | sed -E 's/[^[:alnum:]_.-]+/-/g; s/^-+//; s/-+$//')"
  [[ -n "${value}" ]] || value="na"
  printf '%.*s' "${limit}" "${value}"
}

TARGET_BASENAME="${TARGET_MODEL%/}"
TARGET_BASENAME="${TARGET_BASENAME##*/}"
TARGET_TAG="$(slug "${TARGET_BASENAME}" 14)"
STAGE1_DATA_BASENAME="${STAGE1_TRAIN_DATA_PATH%/}"
STAGE1_DATA_BASENAME="${STAGE1_DATA_BASENAME##*/}"
STAGE1_DATA_BASENAME="${STAGE1_DATA_BASENAME%.jsonl}"
STAGE1_DATA_TAG="$(slug "${STAGE1_DATA_BASENAME}" 18)"
STAGE2_DATA_BASENAME="${STAGE2_TRAIN_DATA_PATH%/}"
STAGE2_DATA_BASENAME="${STAGE2_DATA_BASENAME##*/}"
STAGE2_DATA_BASENAME="${STAGE2_DATA_BASENAME%.jsonl}"
STAGE2_DATA_TAG="$(slug "${STAGE2_DATA_BASENAME}" 18)"
DATA_TAG="s1${STAGE1_DATA_TAG}_s2${STAGE2_DATA_TAG}"

TEACHER_SOURCE="${TEACHER_DRAFT_PATH:-${RESUME_FROM:-checkpoint}}"
TEACHER_SOURCE="${TEACHER_SOURCE%/}"
TEACHER_BASENAME="${TEACHER_SOURCE##*/}"
case "${TEACHER_BASENAME}" in
  final|transition|epoch_*_step_*)
    TEACHER_PARENT="${TEACHER_SOURCE%/*}"
    TEACHER_BASENAME="${TEACHER_PARENT##*/}"
    ;;
esac
TEACHER_TAG="$(slug "${TEACHER_BASENAME}" 28)"
TEACHER_WANDB_TAG="$(slug "${TEACHER_BASENAME}" 24)"
TEACHER_CONFIG="${TEACHER_DRAFT_PATH:-}/config.json"
if [[ -n "${TEACHER_DRAFT_PATH:-}" && -f "${TEACHER_CONFIG}" ]]; then
  TEACHER_ARCH_META="$("${PYTHON_BIN}" -c '
import json, sys
c = json.load(open(sys.argv[1], encoding="utf-8"))
f = c.get("flashmtp_config") or {}
v = lambda key, default="x": f.get(key, c.get(key, default))
full = "swa{}_ag{}_chs{}_b{}_d{}_{}_{}_r{}".format(
    v("swa_window_size"), v("anchor_group_size"), v("chs_num_layers"),
    c.get("block_size", "x"), c.get("num_hidden_layers", "x"),
    v("markov_head_type"), v("markov_output_mode"), v("markov_rank"),
)
short = "s{}g{}c{}b{}d{}-{}-{}-r{}".format(
    v("swa_window_size"), v("anchor_group_size"), v("chs_num_layers"),
    c.get("block_size", "x"), c.get("num_hidden_layers", "x"),
    v("markov_head_type"), v("markov_output_mode"), v("markov_rank"),
)
print(full + "|" + short)
' "${TEACHER_CONFIG}")"
  TEACHER_TAG="$(slug "${TEACHER_ARCH_META%%|*}" 64)"
  TEACHER_WANDB_TAG="$(slug "${TEACHER_ARCH_META#*|}" 40)"
fi

RUN_TAG="v23s_${DT}_${TARGET_TAG}_${DATA_TAG}_ws${WORLD_SIZE}_tp${TP_SIZE}_sh${SHARD_DRAFT_BY_TP}_tb${NAME_TARGET_BATCH}_db${NAME_DRAFT_BATCH}_${TEACHER_TAG}_i${STUDENT_INIT_MODE}${STUDENT_DEPTH_TAG}_m${MASK_TOKEN_ID}_e${STAGE1_EPOCHS}+${STAGE2_EPOCHS}_lr${LEARNING_RATE}_w${WARMUP_RATIO}_kl${STAGE1_KL_WEIGHT}_h${STAGE1_HIDDEN_WEIGHT}_g${STAGE1_LOSS_DECAY_GAMMA:-none}_ce${STAGE2_FINAL_CE_WEIGHT}_tv${STAGE2_TV_WEIGHT}_b${STAGE2_BASE_CE_WEIGHT}_g${STAGE2_LOSS_DECAY_GAMMA:-none}_bg${STAGE2_BASE_CE_DECAY_GAMMA:-none}_L${MAX_LENGTH}_A${NUM_ANCHORS}_ac${ACCUMULATION_STEPS}"
if [[ -n "${RUN_SUFFIX:-}" ]]; then
  RUN_TAG="$(slug "${RUN_TAG}" 210)_$(slug "${RUN_SUFFIX}" 24)"
else
  RUN_TAG="$(slug "${RUN_TAG}" 240)"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/cache/models}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_TAG}}"
CACHE_DIR="${CACHE_DIR:-${PROJECT_DIR}/cache/train/${DATA_TAG}_l${MAX_LENGTH}_m${MASK_TOKEN_ID}}"
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-flashmtp-training-v2.3-student}"
RUN_HASH="$("${PYTHON_BIN}" -c 'import hashlib, sys; print(hashlib.sha1(sys.argv[1].encode()).hexdigest()[:8])' "${RUN_TAG}")"
WANDB_DEFAULT_BASE="v23s_${DT}_${TARGET_TAG}_${DATA_TAG}_ws${WORLD_SIZE}tp${TP_SIZE}sh${SHARD_DRAFT_BY_TP}_${TEACHER_WANDB_TAG}_e${STAGE1_EPOCHS}+${STAGE2_EPOCHS}lr${LEARNING_RATE}_L${MAX_LENGTH}A${NUM_ANCHORS}"
WANDB_DEFAULT_NAME="$(slug "${WANDB_DEFAULT_BASE}" 118)_${RUN_HASH}"
WANDB_DEFAULT_ID="v23s-${DT}-${TARGET_TAG}-${DATA_TAG}-ws${WORLD_SIZE}-tp${TP_SIZE}-sh${SHARD_DRAFT_BY_TP}-${RUN_HASH}"
WANDB_NAME="${WANDB_RUN_NAME:-${WANDB_NAME:-${WANDB_DEFAULT_NAME}}}"
WANDB_RUN_ID="${WANDB_RUN_ID:-${WANDB_DEFAULT_ID}}"

if [[ "${SHARD_DRAFT_BY_TP}" != "0" && "${SHARD_DRAFT_BY_TP}" != "1" ]]; then
  echo "SHARD_DRAFT_BY_TP must be 0 or 1" >&2
  exit 2
fi
TRAIN_BATCH_SIZE="${REQUESTED_BATCH_SIZE}"
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
[[ -n "${STUDENT_NUM_DRAFT_LAYERS}" ]] && OPTIONAL_ARGS+=(--student-num-draft-layers "${STUDENT_NUM_DRAFT_LAYERS}")
[[ -n "${STAGE1_LOSS_DECAY_GAMMA:-}" ]] && OPTIONAL_ARGS+=(--stage1-loss-decay-gamma "${STAGE1_LOSS_DECAY_GAMMA}")
[[ -n "${STAGE2_LOSS_DECAY_GAMMA:-}" ]] && OPTIONAL_ARGS+=(--stage2-loss-decay-gamma "${STAGE2_LOSS_DECAY_GAMMA}")
[[ -n "${STAGE2_BASE_CE_DECAY_GAMMA:-}" ]] && OPTIONAL_ARGS+=(--stage2-base-ce-decay-gamma "${STAGE2_BASE_CE_DECAY_GAMMA}")
[[ -n "${STAGE1_BUILD_DATASET_NUM_PROC:-}" ]] && OPTIONAL_ARGS+=(--stage1-build-dataset-num-proc "${STAGE1_BUILD_DATASET_NUM_PROC}")
[[ -n "${STAGE2_BUILD_DATASET_NUM_PROC:-}" ]] && OPTIONAL_ARGS+=(--stage2-build-dataset-num-proc "${STAGE2_BUILD_DATASET_NUM_PROC}")
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
  -m scripts.train_flashmtp_two_stage
  --target-model-path "${TARGET_MODEL}"
  --target-model-backend "${TARGET_MODEL_BACKEND}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.4}"
  --stage1-train-data-path "${STAGE1_TRAIN_DATA_PATH}"
  --stage2-train-data-path "${STAGE2_TRAIN_DATA_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --stage1-epochs "${STAGE1_EPOCHS}"
  --learning-rate "${LEARNING_RATE}"
  --warmup-ratio "${WARMUP_RATIO}"
  --stage1-kl-weight "${STAGE1_KL_WEIGHT}"
  --stage1-hidden-weight "${STAGE1_HIDDEN_WEIGHT}"
  --stage1-smooth-l1-beta "${STAGE1_SMOOTH_L1_BETA}"
  --stage2-epochs "${STAGE2_EPOCHS}"
  --stage2-final-ce-weight "${STAGE2_FINAL_CE_WEIGHT}"
  --stage2-tv-weight "${STAGE2_TV_WEIGHT}"
  --stage2-base-ce-weight "${STAGE2_BASE_CE_WEIGHT}"
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
  --log-interval "${LOG_INTERVAL:-50}"
  --save-interval "${SAVE_INTERVAL:-20000}"
  --tp-size "${TP_SIZE}"
  "${OPTIONAL_ARGS[@]}"
  "${PASSTHROUGH_ARGS[@]}"
)

printf 'FlashMTP v2.3 two-stage config: dt=%s nodes=%s rank=%s gpus/node=%s world=%s tp=%s\n' \
  "${DT}" "${NNODES}" "${NODE_RANK}" "${NPROC_PER_NODE}" "${WORLD_SIZE}" "${TP_SIZE}"
printf 'Output directory: %s\n' "${OUTPUT_DIR}"
printf 'Stage 1 dataset: %s\nStage 2 dataset: %s\n' \
  "${STAGE1_TRAIN_DATA_PATH}" "${STAGE2_TRAIN_DATA_PATH}"
printf 'Dataset cache: %s\n' "${CACHE_DIR}"
printf 'MASK token id: %s\n' "${MASK_TOKEN_ID}"
if [[ "${REPORT_TO}" == "wandb" ]]; then
  printf 'W&B project: %s\nW&B name: %s\nW&B run id: %s\n' \
    "${WANDB_PROJECT}" "${WANDB_NAME}" "${WANDB_RUN_ID}"
fi
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
