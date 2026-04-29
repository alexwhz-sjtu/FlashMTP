#!/bin/bash
# FlashMTP v3.3 训练入口（单脚本两阶段；逻辑与 run_v3_3_lib + run_training_flashmtp 路径一致，默认 a800）
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
if [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
  source "${PROJECT_DIR}/.venv/bin/activate"
fi
cd "${PROJECT_DIR}"

# shellcheck source=run_v3_3_lib.sh
source "${PROJECT_DIR}/scripts/run_v3_3_lib.sh"
v33_parse_cli "$@" || exit $?

v33_export_common_training_env || exit $?
v33_export_paths_for_dt || exit $?

mkdir -p "$FLASHMTP_V33_MDLM_OUT" "$FLASHMTP_V33_STREAK_OUT" "$CACHE_ROOT"

export ROOT="$PROJECT_DIR"
unset WANDB_RUN_NAME WANDB_RUN_ID || true
v33_wandb_defaults mdlm
v33_print_banner "FlashMTP v3.3（MDLM → Streak）一体化"

TORCHRUN="${PROJECT_DIR}/.venv/bin/torchrun"
if [[ ! -x "$TORCHRUN" ]]; then
  TORCHRUN=torchrun
fi
export TORCHRUN PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

echo "==> Phase 1: MDLM"
v33_common_py_args mdlm
unset V33_MASTER_PORT_OVERRIDE
v33_build_torchrun

"${V33_TORCHRUN[@]}" "${PROJECT_DIR}/scripts/train_flashmtp_mdlm.py" \
  "${V33_COMMON_PY[@]}" \
  --output-dir "$FLASHMTP_V33_MDLM_OUT" \
  --cache-dir "$CACHE_ROOT/mdlm_process" \
  --learning-rate "$LEARNING_RATE_MDLM" \
  --mask-ratio-min "$MASK_RATIO_MIN" \
  --mask-ratio-max "$MASK_RATIO_MAX" \
  --kl-weight "$KL_WEIGHT" \
  --kl-topk "$KL_TOPK" \
  "${V33_PY_EXTRA[@]}"

last="$(ls -d "$FLASHMTP_V33_MDLM_OUT"/epoch_* 2>/dev/null | sort -V | tail -1 || true)"
if [[ -z "$last" || ! -d "$last" ]]; then
  echo "未找到 MDLM checkpoint: $FLASHMTP_V33_MDLM_OUT" >&2
  exit 1
fi
export FLASHMTP_V33_INIT_CKPT="$last"

echo "==> Phase 2: Streak (init $FLASHMTP_V33_INIT_CKPT)"
unset WANDB_RUN_NAME WANDB_RUN_ID || true
v33_wandb_defaults streak
v33_common_py_args streak

if [ "${NNODES}" -eq 1 ] 2>/dev/null; then
  export V33_MASTER_PORT_OVERRIDE="${STREAK_MASTER_PORT:-$((MASTER_PORT + 1))}"
else
  export V33_MASTER_PORT_OVERRIDE="${STREAK_MASTER_PORT:-$MASTER_PORT}"
fi
v33_build_torchrun
unset V33_MASTER_PORT_OVERRIDE

"${V33_TORCHRUN[@]}" "${PROJECT_DIR}/scripts/train_flashmtp_streak.py" \
  "${V33_COMMON_PY[@]}" \
  --output-dir "$FLASHMTP_V33_STREAK_OUT" \
  --cache-dir "$CACHE_ROOT/streak_process" \
  --learning-rate "$LEARNING_RATE_STREAK" \
  --init-ckpt "$FLASHMTP_V33_INIT_CKPT" \
  "${V33_PY_EXTRA[@]}"

echo "完成。Streak 模型目录: $FLASHMTP_V33_STREAK_OUT"
