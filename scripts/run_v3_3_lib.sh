#!/usr/bin/env bash
# v3.3 公共环境：多机、--dt 路径、STAMP、W&B 默认名、torchrun 参数。
# 用法：在 cd 到仓库根目录并激活 .venv 后:  source "$ROOT/scripts/run_v3_3_lib.sh"
# 依次调用: v33_parse_cli "$@" ; v33_export_common_training_env ; v33_export_paths_for_dt ; v33_build_torchrun ; v33_wandb_defaults <mdlm|streak>
#set +u  # 调用方若使用 set -u，请在 source 后按需处理默认值

v33_parse_cli() {
  V33_PY_EXTRA=()
  local seen_dt=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dt)
        DT="$2"
        seen_dt=1
        shift 2
        ;;
      *)
        V33_PY_EXTRA+=("$1")
        shift
        ;;
    esac
  done
  if [[ "$seen_dt" -eq 0 ]]; then
    DT="${DT:-a800}"
  fi
  if [[ "$DT" != "qz" && "$DT" != "a800" ]]; then
    echo "错误: --dt 须为 qz 或 a800" >&2
    return 1
  fi
  export DT
  return 0
}

v33_export_common_training_env() {
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  if [ -n "${PET_NPROC_PER_NODE:-}" ]; then
    NPROC_PER_NODE="${PET_NPROC_PER_NODE}"
  else
    NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
  fi
  export NPROC_PER_NODE

  NNODES="${PET_NNODES:-${NNODES:-1}}"
  NODE_RANK="${PET_NODE_RANK:-${NODE_RANK:-0}}"
  MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:-127.0.0.1}}"
  MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-29512}}"
  if [ "${NNODES}" -gt 1 ] 2>/dev/null && { [ "${MASTER_ADDR}" = "127.0.0.1" ] || [ "${MASTER_ADDR}" = "localhost" ]; }; then
    echo "错误: 多机 (NNODES=${NNODES}) 须设置 MASTER_ADDR 或 PET_MASTER_ADDR 为可互通地址。" >&2
    return 1
  fi
  export NNODES NODE_RANK MASTER_ADDR MASTER_PORT

  # --- Epoch：MDLM / Streak 可分开设；未设则回落到 NUM_EPOCHS ---
  NUM_EPOCHS_MDLM=12
  NUM_EPOCHS_STREAK=6
  NUM_EPOCHS="${NUM_EPOCHS:-6}"
  MAX_LENGTH="${MAX_LENGTH:-4096}"
  DATA_NUM_SAMPLES="${DATA_NUM_SAMPLES:-400000}" # 仅用于默认 jsonl / CACHE_ROOT 路径拼接

  ENABLE_THINKING="${ENABLE_THINKING:-on}"         # 默认训练集路径里的 think_on / think_off
  NUM_DRAFT_LAYERS="${NUM_DRAFT_LAYERS:-5}"       # 草案 Transformer 层数
  BLOCK_SIZE="${BLOCK_SIZE:-16}"                  # 每块推测 token 数，需与推理一致
  NUM_ANCHORS="${NUM_ANCHORS:-512}"               # 每条序列最多采样的锚点块数（上限还受序列长约束）
  BATCH_SIZE="${BATCH_SIZE:-1}"

  ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"
  TP_SIZE="${TP_SIZE:-1}"                         # 目标模型张量并行（HF 后端时用）
  DIST_TIMEOUT="${DIST_TIMEOUT:-3600}"
  CHAT_TEMPLATE="${CHAT_TEMPLATE:-qwen3-thinking}" # 与预处理 / loss_mask 规则一致
  REPORT_TO="${REPORT_TO:-wandb}"                 # none | wandb 等，见 specforge.tracker

  # --- MDLM 阶段专用（仅 train_flashmtp_mdlm.py）---
  # MASK_RATIO_*：每个块内对「可监督位置」独立 Bernoulli 掩码的阈值范围；在 [MIN,MAX] 上均匀抽一档作为本块掩码率。
  #   调高 MIN/MAX → 更高掩码比例，更偏「难填空」；MAX=1 允许接近全掩码（仍保证至少一个被掩位置，见 Python 内逻辑）。
  MASK_RATIO_MIN="${MASK_RATIO_MIN:-0.1}"
  MASK_RATIO_MAX="${MASK_RATIO_MAX:-1.0}"
  # KL_*：草案 logits 相对目标模型 teacher_logits 的 KL（HF 且 forward 带 logits 时才有 teacher）。
  # KL_WEIGHT=0 关闭；>0 时与 CE 加权；KL_TOPK=0 为全词表 KL，>0 时在 teacher 的 top-k 子空间上做 KL（省算力、近似蒸馏）。
  KL_WEIGHT="${KL_WEIGHT:-0.2}"
  KL_TOPK="${KL_TOPK:-500}"
  SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"   # 步数；过大则几乎只有 epoch 结束才存 checkpoint
  LOG_INTERVAL="${LOG_INTERVAL:-50}"

  DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
  BUILD_DATASET_NUM_PROC="${BUILD_DATASET_NUM_PROC:-8}"
  ATTENTION_BACKEND="${ATTENTION_BACKEND:-flex_attention}" # 草案 attention：需与 flex BlockMask 一致时用 flex_attention
  TARGET_MODEL_BACKEND="${TARGET_MODEL_BACKEND:-hf}"       # hf | sglang
  LEARNING_RATE_MDLM="${LEARNING_RATE_MDLM:-6e-4}"
  LEARNING_RATE_STREAK="${LEARNING_RATE_STREAK:-6e-4}"     # Streak 阶段通常可略低于 MDLM
  STREAK_WEIGHT="${STREAK_WEIGHT:-1.0}"                    # conf-streak 主 loss 系数
  STREAK_CE_WEIGHT="${STREAK_CE_WEIGHT:-0.1}"              # 逐位置 CE 辅助项系数

  export NUM_EPOCHS NUM_EPOCHS_MDLM NUM_EPOCHS_STREAK MAX_LENGTH DATA_NUM_SAMPLES ENABLE_THINKING NUM_DRAFT_LAYERS BLOCK_SIZE NUM_ANCHORS
  export BATCH_SIZE ACCUMULATION_STEPS TP_SIZE DIST_TIMEOUT CHAT_TEMPLATE REPORT_TO
  export MASK_RATIO_MIN MASK_RATIO_MAX KL_WEIGHT KL_TOPK SAVE_INTERVAL LOG_INTERVAL
  export DATALOADER_NUM_WORKERS BUILD_DATASET_NUM_PROC ATTENTION_BACKEND TARGET_MODEL_BACKEND
  export LEARNING_RATE_MDLM LEARNING_RATE_STREAK STREAK_WEIGHT STREAK_CE_WEIGHT
  return 0
}

v33_export_paths_for_dt() {
  if [ "$DT" = "qz" ]; then
    export WANDB_MODE="${WANDB_MODE:-offline}"
    TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/FlashMTP/cache/data/regen_data/nemotron_${DATA_NUM_SAMPLES}/nemotron_think_${ENABLE_THINKING}_samples_${DATA_NUM_SAMPLES}_qwen3_8b_regen.jsonl}"
    TARGET_MODEL="${TARGET_MODEL:-/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/models/Qwen/Qwen3-8B}"
    CACHE_ROOT="${CACHE_ROOT:-./cache/data/regen_data/nemotron_${DATA_NUM_SAMPLES}}"
  else
    export DATA_NUM_SAMPLES=40000
    TRAIN_DATA_PATH="/share/wanghanzhen/SpeculativeDecoding/NIPS26/FlashMTP_v1.1/cache/data/regen_data/nemotron_40000/nemotron_think_on_samples_40000_qwen3_8b_regen.jsonl"
    TARGET_MODEL="${TARGET_MODEL:-/share/public/public_models/Qwen3-8B}"
    CACHE_ROOT="${CACHE_ROOT:-./cache/data/regen_data/nemotron_40000}"
  fi

  STAMP="v33_${DT}_nlayers${NUM_DRAFT_LAYERS}_bs${BLOCK_SIZE}_samples${DATA_NUM_SAMPLES}_think_${ENABLE_THINKING}_maxlen${MAX_LENGTH}_kl_${KL_WEIGHT}_epm${NUM_EPOCHS_MDLM}_eps${NUM_EPOCHS_STREAK}"
  STREAK_STAMP="${STAMP}_wst${STREAK_WEIGHT}_wce${STREAK_CE_WEIGHT}"
  export STAMP
  export STREAK_STAMP

  export FLASHMTP_TARGET="${FLASHMTP_TARGET:-$TARGET_MODEL}"
  export FLASHMTP_V33_TRAIN="${FLASHMTP_V33_TRAIN:-$TRAIN_DATA_PATH}"
  export FLASHMTP_V33_MDLM_OUT="${FLASHMTP_V33_MDLM_OUT:-${OUTPUT_DIR_MDLM:-./cache/models/flashmtp_mdlm_${STAMP}}}"
  export FLASHMTP_V33_STREAK_OUT="${FLASHMTP_V33_STREAK_OUT:-${OUTPUT_DIR_STREAK:-./cache/models/flashmtp_streak_${STREAK_STAMP}}}"
  export TRAIN_DATA_PATH TARGET_MODEL CACHE_ROOT
  return 0
}

v33_build_torchrun() {
  if [[ -z "${TORCHRUN:-}" ]]; then
    local _root="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
    if [[ -x "${_root}/.venv/bin/torchrun" ]]; then
      TORCHRUN="${_root}/.venv/bin/torchrun"
    else
      TORCHRUN=torchrun
    fi
  fi
  export TORCHRUN
  local _mp="${V33_MASTER_PORT_OVERRIDE:-$MASTER_PORT}"
  V33_TORCHRUN=(
    "$TORCHRUN"
    --nproc_per_node="$NPROC_PER_NODE"
  )
  if [ "${NNODES}" -gt 1 ] 2>/dev/null; then
    V33_TORCHRUN+=(
      --nnodes="$NNODES"
      --node_rank="$NODE_RANK"
      --master_addr="$MASTER_ADDR"
      --master_port="$_mp"
    )
  else
    V33_TORCHRUN+=(--master_port="$_mp")
  fi
  return 0
}

# 仅在对应阶段未显式设置时使用默认 W&B 名，避免覆盖用户 export。
v33_wandb_defaults() {
  local phase="$1"
  if [ "${REPORT_TO}" != "wandb" ]; then
    return 0
  fi
  export WANDB_PROJECT="${WANDB_PROJECT:-flashmtp_v3.3}"
  export WANDB_DIR="${WANDB_DIR:-./wandb}"
  local _time_tag="${WANDB_RUN_TIME_TAG:-$(date +%Y%m%d_%H%M%S)}"
  export WANDB_RUN_TIME_TAG="$_time_tag"
  local _name_suffix="_t${_time_tag}"
  local _suffix="_n${NNODES}_t${_time_tag}"
  case "$phase" in
    mdlm)
      if [[ -z "${WANDB_RUN_NAME:-}" ]]; then
        export WANDB_RUN_NAME="v33_${DT}_mdlm_${STAMP}${_name_suffix}"
      fi
      if [[ -z "${WANDB_RUN_ID:-}" ]]; then
        export WANDB_RUN_ID="v33_${DT}_mdlm_${STAMP}${_suffix}"
      fi
      ;;
    streak)
      if [[ -z "${WANDB_RUN_NAME:-}" ]]; then
        export WANDB_RUN_NAME="v33_${DT}_streak_${STREAK_STAMP:-$STAMP}${_name_suffix}"
      fi
      if [[ -z "${WANDB_RUN_ID:-}" ]]; then
        export WANDB_RUN_ID="v33_${DT}_streak_${STREAK_STAMP:-$STAMP}${_suffix}"
      fi
      ;;
    *)
      echo "v33_wandb_defaults: phase 须为 mdlm 或 streak" >&2
      return 1
      ;;
  esac
  return 0
}

v33_common_py_args() {
  local phase="${1:?v33_common_py_args: 需要参数 mdlm 或 streak}"
  local _ne
  case "$phase" in
    mdlm) _ne="$NUM_EPOCHS_MDLM" ;;
    streak) _ne="$NUM_EPOCHS_STREAK" ;;
    *)
      echo "v33_common_py_args: phase 须为 mdlm / streak， got: $phase" >&2
      return 1
      ;;
  esac
  # shellcheck disable=SC2034
  V33_COMMON_PY=(
    --target-model-path "$TARGET_MODEL"
    --target-model-backend "$TARGET_MODEL_BACKEND"
    --train-data-path "$FLASHMTP_V33_TRAIN"
    --chat-template "$CHAT_TEMPLATE"
    --num-draft-layers "$NUM_DRAFT_LAYERS"
    --block-size "$BLOCK_SIZE"
    --num-anchors "$NUM_ANCHORS"
    --max-length "$MAX_LENGTH"
    --num-epochs "$_ne"
    --batch-size "$BATCH_SIZE"
    --accumulation-steps "$ACCUMULATION_STEPS"
    --tp-size "$TP_SIZE"
    --dist-timeout "$DIST_TIMEOUT"
    --attention-backend "$ATTENTION_BACKEND"
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS"
    --build-dataset-num-proc "$BUILD_DATASET_NUM_PROC"
    --save-interval "$SAVE_INTERVAL"
    --log-interval "$LOG_INTERVAL"
    --report-to "$REPORT_TO"
  )
  if [ "${REPORT_TO}" = "wandb" ]; then
    V33_COMMON_PY+=(--wandb-project "$WANDB_PROJECT")
    [ -n "${WANDB_RUN_NAME:-}" ] && V33_COMMON_PY+=(--wandb-name "$WANDB_RUN_NAME")
    [ -n "${WANDB_RUN_ID:-}" ] && V33_COMMON_PY+=(--wandb-run-id "$WANDB_RUN_ID")
  fi
  [ -n "${IS_PREFORMATTED:-}" ] && V33_COMMON_PY+=(--is-preformatted)
  return 0
}

v33_print_banner() {
  local title="${1:-FlashMTP v3.3}"
  echo "=========================================="
  echo "$title"
  echo "  DT=$DT  NNODES=$NNODES  NODE_RANK=$NODE_RANK"
  echo "  MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"
  echo "  目标模型: $TARGET_MODEL"
  echo "  训练数据: $FLASHMTP_V33_TRAIN"
  echo "  STAMP=$STAMP"
  echo "  NUM_EPOCHS_MDLM=$NUM_EPOCHS_MDLM  NUM_EPOCHS_STREAK=$NUM_EPOCHS_STREAK"
  echo "  MDLM_OUT=$FLASHMTP_V33_MDLM_OUT"
  echo "  STREAK_OUT=$FLASHMTP_V33_STREAK_OUT"
  echo "  CACHE_ROOT=$CACHE_ROOT"
  if [ "${REPORT_TO}" = "wandb" ]; then
    echo "  W&B project=$WANDB_PROJECT run_name=${WANDB_RUN_NAME:-} run_id=${WANDB_RUN_ID:-}"
  fi
  echo "=========================================="
}
