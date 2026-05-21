#!/usr/bin/env bash
# FlashMTP v3.3 — 阶段一：从零开始 MDLM 训练（train_flashmtp_mdlm.py）
# 用法: ./scripts/train_mdlm_v33.sh [--dt qz|a800|h100] [额外参数透传给 train_flashmtp_mdlm.py]
set -euo pipefail

# =============================================================================
# 仅本脚本：环境与数据（按机器修改）
# =============================================================================
# qz | a800 | h100；未 export 时默认 a800；也可用 ./scripts/train_mdlm_v33.sh --dt qz 覆盖
DT="${DT:-a800}"

# 数据样本量（用于默认 jsonl / cache 路径拼接）
DATA_NUM_SAMPLES="40000"
ENABLE_THINKING="off" # 默认路径里的 think_on / think_off

# 以下在 DT=qz / h100 时生效；DT=a800 时脚本内写死 a800 默认路径（与旧 run_v3_3_lib 一致）
TRAIN_DATA_PATH_QZ="/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/FlashMTP/cache/data/regen_data/nemotron_${DATA_NUM_SAMPLES}/nemotron_think_${ENABLE_THINKING}_samples_${DATA_NUM_SAMPLES}_qwen3_8b_regen.jsonl"
TARGET_MODEL_QZ="/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/models/Qwen/Qwen3-8B"
CACHE_ROOT_QZ="./cache/data/regen_data/nemotron_${DATA_NUM_SAMPLES}"

TRAIN_DATA_PATH_H100="../training_data/regen_data/nemotron_${DATA_NUM_SAMPLES}/nemotron_think_${ENABLE_THINKING}_samples_${DATA_NUM_SAMPLES}_qwen3_8b_regen.jsonl"
TARGET_MODEL_H100="${WHZ_DIR:-$HOME}/models/Qwen/Qwen3-8B"
CACHE_ROOT_H100="./cache/data/regen_data/nemotron_${DATA_NUM_SAMPLES}"

TRAIN_DATA_PATH_A800="/share/wanghanzhen/SpeculativeDecoding/NIPS26/FlashMTP_v1.1/cache/data/regen_data/nemotron_40000/nemotron_think_on_samples_40000_qwen3_8b_regen.jsonl"
TARGET_MODEL_A800="/share/public/public_models/Qwen3-8B"
CACHE_ROOT_A800="./cache/data/regen_data/nemotron_40000"

# =============================================================================
# 仅本脚本：MDLM 训练超参
# =============================================================================
NUM_EPOCHS="6"
MAX_LENGTH="4096"
NUM_DRAFT_LAYERS="5"
BLOCK_SIZE="16"
NUM_ANCHORS="512"
# v1.1 风格 target_layer_ids：首尾 + 中间均匀层数（build_ablation_target_layer_ids）
NUM_MIDDLE_LAYERS_N="${NUM_MIDDLE_LAYERS_N:-5}"
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

MASK_RATIO_MIN="0.05"
MASK_RATIO_MAX="1.0"
KL_WEIGHT="0.1"
KL_TOPK="10"
CE_WEIGHT="1.0"

SAVE_INTERVAL="10000"
LOG_INTERVAL="50"
EVAL_INTERVAL="1000"

DATALOADER_NUM_WORKERS="8"
BUILD_DATASET_NUM_PROC="8"

REPORT_TO="wandb" # none | wandb 等
WANDB_PROJECT="flashmtp_training_exp"
WANDB_DIR="./wandb"
WANDB_RUN_NAME=""  # 空则自动生成
WANDB_RUN_ID=""   # 空则自动生成

IS_PREFORMATTED="" # 非空则加 --is-preformatted

# 输出目录（留空则按 STAMP 自动生成）
MDLM_OUTPUT_DIR="${MDLM_OUTPUT_DIR:-}"

# =============================================================================
# 仅本脚本：分布式 / torchrun
# =============================================================================
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}" # 可被 PET_NPROC_PER_NODE 覆盖
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29512}"

# =============================================================================
# 运行时：仓库根目录、venv、PYTHONPATH
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

# --- CLI：--dt + 透传 python ---
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

# --- 路径 ---
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

STAMP="v33_${DT}_nlayers${NUM_DRAFT_LAYERS}_nmiddle${NUM_MIDDLE_LAYERS_N}_bs${BLOCK_SIZE}_samples${DATA_NUM_SAMPLES}_think_${ENABLE_THINKING}_maxlen${MAX_LENGTH}_kl_${KL_WEIGHT}_epmdlm${NUM_EPOCHS}"
if [[ -z "${MDLM_OUTPUT_DIR}" ]]; then
  MDLM_OUTPUT_DIR="./cache/models/flashmtp_mdlm_${STAMP}"
fi

# --- W&B 默认名 ---
if [[ "${REPORT_TO}" == "wandb" ]]; then
  _t="${WANDB_RUN_TIME_TAG:-$(date +%Y%m%d_%H%M%S)}"
  export WANDB_RUN_TIME_TAG="$_t"
  _suffix="_n${NNODES}_t${_t}"
  if [[ -z "${WANDB_RUN_NAME}" ]]; then
    WANDB_RUN_NAME="v33_${DT}_mdlm_${STAMP}_t${_t}"
  fi
  if [[ -z "${WANDB_RUN_ID}" ]]; then
    WANDB_RUN_ID="v33_${DT}_mdlm_${STAMP}${_suffix}"
  fi
fi

mkdir -p "${MDLM_OUTPUT_DIR}" "${CACHE_ROOT}"

echo "=========================================="
echo "FlashMTP v3.3 — MDLM（从零）"
echo "  DT=${DT}  NNODES=${NNODES}  MASTER=${MASTER_ADDR}:${MASTER_PORT}"
echo "  目标模型: ${TARGET_MODEL}"
echo "  训练数据: ${TRAIN_DATA_PATH}"
echo "  输出: ${MDLM_OUTPUT_DIR}"
echo "  CACHE_ROOT: ${CACHE_ROOT}"
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
    --master_port="${MASTER_PORT}"
  )
else
  TORCH_CMD+=(--master_port="${MASTER_PORT}")
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

exec "${TORCH_CMD[@]}" "${ROOT}/scripts/train_flashmtp_mdlm.py" \
  "${COMMON[@]}" \
  --output-dir "${MDLM_OUTPUT_DIR}" \
  --cache-dir "${CACHE_ROOT}/mdlm_process" \
  --learning-rate "${LEARNING_RATE}" \
  --mask-ratio-min "${MASK_RATIO_MIN}" \
  --mask-ratio-max "${MASK_RATIO_MAX}" \
  --kl-weight "${KL_WEIGHT}" \
  --kl-topk "${KL_TOPK}" \
  --ce-weight "${CE_WEIGHT}" \
  "${PY_EXTRA[@]}"
