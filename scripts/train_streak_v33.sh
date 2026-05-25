#!/usr/bin/env bash
# FlashMTP v3.3 — 阶段二：Streak（train_flashmtp_streak.py）
#   STREAK_FROM_SCRATCH=1：草案随机初始化
#   STREAK_FROM_SCRATCH=0：从 MDLM_INIT_CKPT（epoch_*_step_*）加载草案权重
# 用法:
#   ./scripts/train_streak_v33.sh
#   STREAK_FROM_SCRATCH=0 MDLM_INIT_CKPT=/path/to/epoch_1_step_1000 ./scripts/train_streak_v33.sh --dt qz
#   STREAK_RAW_PROBS=1 ./scripts/train_streak_v33.sh   # streak 主项仅用草案 log q，无教师锚点 / log_phi
#   [额外参数透传给 train_flashmtp_streak.py]
set -euo pipefail

# =============================================================================
# 仅本脚本：环境与数据（按机器修改；与 MDLM 脚本应对齐）
# =============================================================================
DT="${DT:-a800}"

DATA_NUM_SAMPLES="${DATA_NUM_SAMPLES:-40000}"
ENABLE_THINKING="${ENABLE_THINKING:-off}"

# qz
TRAIN_DATA_PATH_QZ="/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/FlashMTP/cache/data/regen_data/nemotron_${DATA_NUM_SAMPLES}/nemotron_think_${ENABLE_THINKING}_samples_${DATA_NUM_SAMPLES}_qwen3_8b_regen.jsonl"
TARGET_MODEL_QZ="/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/models/Qwen/Qwen3-8B"
CACHE_ROOT_QZ="./cache/data/regen_data/nemotron_${DATA_NUM_SAMPLES}"

#h100
TRAIN_DATA_PATH_H100="../training_data/regen_data/nemotron_${DATA_NUM_SAMPLES}/nemotron_think_${ENABLE_THINKING}_samples_${DATA_NUM_SAMPLES}_qwen3_8b_regen.jsonl"
TARGET_MODEL_H100="${WHZ_DIR:-/data/wanghanzhen}/models/Qwen/Qwen3-8B"
CACHE_ROOT_H100="./cache/data/regen_data/nemotron_${DATA_NUM_SAMPLES}"

#a800
TRAIN_DATA_PATH_A800="/share/wanghanzhen/SpeculativeDecoding/NIPS26/FlashMTP_v1.1/cache/data/regen_data/nemotron_40000/nemotron_think_on_samples_40000_qwen3_8b_regen.jsonl"
TARGET_MODEL_A800="/share/public/public_models/Qwen3-8B"
CACHE_ROOT_A800="./cache/data/regen_data/nemotron_40000"

# =============================================================================
# 仅本脚本：Streak 初始化策略
# =============================================================================
# 1=随机初始化；0=加载 MDLM_INIT_CKPT。可在运行前 export STREAK_FROM_SCRATCH=0
STREAK_FROM_SCRATCH="${STREAK_FROM_SCRATCH:-1}"
# 当 STREAK_FROM_SCRATCH=0 时必填：MDLM 的 epoch_*_step_* 目录
MDLM_INIT_CKPT="${MDLM_INIT_CKPT:-}"

# =============================================================================
# 仅本脚本：Streak 训练超参（可与 MDLM 脚本设不同学习率/epoch 等）
# =============================================================================
NUM_EPOCHS="12"
MAX_LENGTH="4096"
NUM_DRAFT_LAYERS="5"
BLOCK_SIZE="16"
NUM_ANCHORS="512"
NUM_MIDDLE_LAYERS_N="${NUM_MIDDLE_LAYERS_N:-0}"
BATCH_SIZE="1"
ACCUMULATION_STEPS="1"
TP_SIZE="1"
DIST_TIMEOUT="3600"
CHAT_TEMPLATE="qwen"
ATTENTION_BACKEND="flex_attention"
TARGET_MODEL_BACKEND="hf"

LEARNING_RATE="6e-4"
WARMUP_RATIO="0.04"
MAX_GRAD_NORM="1.0"

STREAK_WEIGHT="1.0"
STREAK_CE_WEIGHT="0.1"
LOG_PROB_MIN="-40.0"
# 1= streak 主项直接用草案 log q（无教师锚点 / 无 log_phi）；0= 默认 LS-RSL
STREAK_RAW_PROBS="${STREAK_RAW_PROBS:-0}"

SAVE_INTERVAL="10000"
LOG_INTERVAL="50"
EVAL_INTERVAL="1000"

DATALOADER_NUM_WORKERS="8"
BUILD_DATASET_NUM_PROC="8"

REPORT_TO="wandb"
WANDB_PROJECT="flashmtp-training-exp"
WANDB_DIR="./wandb"
WANDB_RUN_NAME=""
WANDB_RUN_ID=""

IS_PREFORMATTED=""

STREAK_OUTPUT_DIR="${STREAK_OUTPUT_DIR:-}"

# =============================================================================
# 仅本脚本：分布式（与 MDLM 连续同机跑时，可把 STREAK_MASTER_PORT 改成未占用端口）
# =============================================================================
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29512}"
STREAK_MASTER_PORT="${STREAK_MASTER_PORT:-}" # 空：单机时自动用 MASTER_PORT+1

# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
export ROOT
if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

PY_EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dt)
      DT="$2"
      shift 2
      ;;
    *)
      PY_EXTRA+=("$1")
      shift
      ;;
  esac
done
if [[ "$DT" != "qz" && "$DT" != "a800" && "$DT" != "h100" ]]; then
  echo "错误: --dt 须为 qz / a800 / h100" >&2
  exit 1
fi

if [[ -n "${PET_NPROC_PER_NODE:-}" ]]; then NPROC_PER_NODE="${PET_NPROC_PER_NODE}"; fi
NNODES="${PET_NNODES:-${NNODES}}"
NODE_RANK="${PET_NODE_RANK:-${NODE_RANK}}"
MASTER_ADDR="${PET_MASTER_ADDR:-${MASTER_ADDR}}"
MASTER_PORT="${PET_MASTER_PORT:-${MASTER_PORT}}"
export CUDA_VISIBLE_DEVICES NPROC_PER_NODE NNODES NODE_RANK MASTER_ADDR MASTER_PORT

if [[ "${NNODES}" -gt 1 ]] 2>/dev/null && { [[ "${MASTER_ADDR}" == "127.0.0.1" ]] || [[ "${MASTER_ADDR}" == "localhost" ]]; }; then
  echo "错误: 多机 (NNODES=${NNODES}) 须设置 MASTER_ADDR / PET_MASTER_ADDR 为可互通地址。" >&2
  exit 1
fi

if [[ "$DT" == "qz" ]]; then
  export WANDB_MODE="${WANDB_MODE:-offline}"
  TRAIN_DATA_PATH="${TRAIN_DATA_PATH_QZ}"
  TARGET_MODEL="${TARGET_MODEL_QZ}"
  CACHE_ROOT="${CACHE_ROOT_QZ}"
elif [[ "$DT" == "h100" ]]; then
  TRAIN_DATA_PATH="${TRAIN_DATA_PATH_H100}"
  TARGET_MODEL="${TARGET_MODEL_H100}"
  CACHE_ROOT="${CACHE_ROOT_H100}"
else
  DATA_NUM_SAMPLES="40000"
  TRAIN_DATA_PATH="${TRAIN_DATA_PATH_A800}"
  TARGET_MODEL="${TARGET_MODEL_A800}"
  CACHE_ROOT="${CACHE_ROOT_A800}"
fi

STAMP_BASE="v33_${DT}_nlayers${NUM_DRAFT_LAYERS}_nmiddle${NUM_MIDDLE_LAYERS_N}_bs${BLOCK_SIZE}_samples${DATA_NUM_SAMPLES}_think_${ENABLE_THINKING}_maxlen${MAX_LENGTH}_kl_na_eps${NUM_EPOCHS}"
STREAK_STAMP="${STAMP_BASE}_wst${STREAK_WEIGHT}_wce${STREAK_CE_WEIGHT}"
if [[ "${STREAK_RAW_PROBS}" == "1" ]]; then
  STREAK_STAMP="${STREAK_STAMP}_rawstreak"
fi
if [[ -z "${STREAK_OUTPUT_DIR}" ]]; then
  STREAK_OUTPUT_DIR="./cache/models/flashmtp_streak_${STREAK_STAMP}"
fi

INIT_ARGS=()
if [[ "${STREAK_FROM_SCRATCH}" == "1" ]]; then
  echo "Streak: STREAK_FROM_SCRATCH=1，不使用 --init-ckpt"
else
  if [[ -z "${MDLM_INIT_CKPT}" || ! -d "${MDLM_INIT_CKPT}" ]]; then
    echo "错误: STREAK_FROM_SCRATCH=0 时需要有效的 MDLM_INIT_CKPT 目录（epoch_*_step_*）" >&2
    exit 1
  fi
  INIT_ARGS=(--init-ckpt "${MDLM_INIT_CKPT}")
  echo "Streak: 从 MDLM 权重初始化: ${MDLM_INIT_CKPT}"
fi

if [[ "${REPORT_TO}" == "wandb" ]]; then
  _t="${WANDB_RUN_TIME_TAG:-$(date +%Y%m%d_%H%M%S)}"
  export WANDB_RUN_TIME_TAG="$_t"
  _suffix="_n${NNODES}_t${_t}"
  if [[ -z "${WANDB_RUN_NAME}" ]]; then
    WANDB_RUN_NAME="v33_${DT}_streak_${STREAK_STAMP}_t${_t}"
  fi
  if [[ -z "${WANDB_RUN_ID}" ]]; then
    WANDB_RUN_ID="v33_${DT}_streak_${STREAK_STAMP}${_suffix}"
  fi
fi

mkdir -p "${STREAK_OUTPUT_DIR}" "${CACHE_ROOT}"

if [[ "${NNODES}" -eq 1 ]] 2>/dev/null; then
  TORCH_MASTER_PORT="${STREAK_MASTER_PORT:-$((MASTER_PORT + 1))}"
else
  TORCH_MASTER_PORT="${STREAK_MASTER_PORT:-${MASTER_PORT}}"
fi

echo "=========================================="
echo "FlashMTP v3.3 — Streak"
echo "  DT=${DT}  NNODES=${NNODES}  MASTER=${MASTER_ADDR}:${TORCH_MASTER_PORT}"
echo "  目标模型: ${TARGET_MODEL}"
echo "  训练数据: ${TRAIN_DATA_PATH}"
echo "  输出: ${STREAK_OUTPUT_DIR}"
echo "  STREAK_FROM_SCRATCH=${STREAK_FROM_SCRATCH}"
echo "  STREAK_RAW_PROBS=${STREAK_RAW_PROBS} (1=草案 log-q streak，无教师锚点)"
echo "  NUM_EPOCHS=${NUM_EPOCHS}  LR=${LEARNING_RATE}"
echo "=========================================="

TORCHRUN="${ROOT}/.venv/bin/torchrun"
if [[ ! -x "${TORCHRUN}" ]]; then TORCHRUN=torchrun; fi

TORCH_CMD=(
  "${TORCHRUN}"
  --nproc_per_node="${NPROC_PER_NODE}"
)
if [[ "${NNODES}" -gt 1 ]] 2>/dev/null; then
  TORCH_CMD+=(
    --nnodes="${NNODES}"
    --node_rank="${NODE_RANK}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${TORCH_MASTER_PORT}"
  )
else
  TORCH_CMD+=(--master_port="${TORCH_MASTER_PORT}")
fi

COMMON=(
  --target-model-path "${TARGET_MODEL}"
  --target-model-backend "${TARGET_MODEL_BACKEND}"
  --train-data-path "${TRAIN_DATA_PATH}"
  --chat-template "${CHAT_TEMPLATE}"
  --num-draft-layers "${NUM_DRAFT_LAYERS}"
  --num-middle-layers-n "${NUM_MIDDLE_LAYERS_N}"
  --block-size "${BLOCK_SIZE}"
  --num-anchors "${NUM_ANCHORS}"
  --max-length "${MAX_LENGTH}"
  --num-epochs "${NUM_EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --accumulation-steps "${ACCUMULATION_STEPS}"
  --tp-size "${TP_SIZE}"
  --dist-timeout "${DIST_TIMEOUT}"
  --attention-backend "${ATTENTION_BACKEND}"
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS}"
  --build-dataset-num-proc "${BUILD_DATASET_NUM_PROC}"
  --save-interval "${SAVE_INTERVAL}"
  --log-interval "${LOG_INTERVAL}"
  --eval-interval "${EVAL_INTERVAL}"
  --report-to "${REPORT_TO}"
  --warmup-ratio "${WARMUP_RATIO}"
  --max-grad-norm "${MAX_GRAD_NORM}"
)
if [[ "${REPORT_TO}" == "wandb" ]]; then
  COMMON+=(--wandb-project "${WANDB_PROJECT}" --wandb-name "${WANDB_RUN_NAME}" --wandb-run-id "${WANDB_RUN_ID}")
fi
[[ -n "${IS_PREFORMATTED}" ]] && COMMON+=(--is-preformatted)

STREAK_PY_EXTRA=()
if [[ "${STREAK_RAW_PROBS}" == "1" ]]; then
  STREAK_PY_EXTRA+=(--streak-raw-probs)
fi

exec "${TORCH_CMD[@]}" "${ROOT}/scripts/train_flashmtp_streak.py" \
  "${COMMON[@]}" \
  --output-dir "${STREAK_OUTPUT_DIR}" \
  --cache-dir "${CACHE_ROOT}/streak_process" \
  --learning-rate "${LEARNING_RATE}" \
  --streak-weight "${STREAK_WEIGHT}" \
  --streak-ce-weight "${STREAK_CE_WEIGHT}" \
  --log-prob-min "${LOG_PROB_MIN}" \
  "${STREAK_PY_EXTRA[@]}" \
  "${INIT_ARGS[@]}" \
  "${PY_EXTRA[@]}"
