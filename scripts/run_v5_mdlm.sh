#!/usr/bin/env bash
# v3.3 阶段一：MDLM（可单独运行；参数与 run_v3_3_training_flashmtp.sh 对齐）
# 用法: ./scripts/run_v3_3_mdlm.sh [--dt qz|a800] [其它参数透传给 train_flashmtp_mdlm.py]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
export ROOT
if [ -f "${ROOT}/.venv/bin/activate" ]; then
  source "${ROOT}/.venv/bin/activate"
fi
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

# shellcheck source=run_v3_3_lib.sh
source "$ROOT/scripts/run_v3_3_lib.sh"
v33_parse_cli "$@" || exit $?
set +u
v33_export_common_training_env || exit $?
v33_export_paths_for_dt || exit $?
mkdir -p "$FLASHMTP_V33_MDLM_OUT" "$CACHE_ROOT"
v33_wandb_defaults mdlm || exit $?
v33_common_py_args mdlm
unset V33_MASTER_PORT_OVERRIDE
v33_build_torchrun || exit $?
v33_print_banner "FlashMTP v3.3 — Phase 1 MDLM"

exec "${V33_TORCHRUN[@]}" "$ROOT/scripts/train_flashmtp_mdlm.py" \
  "${V33_COMMON_PY[@]}" \
  --output-dir "$FLASHMTP_V33_MDLM_OUT" \
  --cache-dir "$CACHE_ROOT/mdlm_process" \
  --learning-rate "$LEARNING_RATE_MDLM" \
  --mask-ratio-min "$MASK_RATIO_MIN" \
  --mask-ratio-max "$MASK_RATIO_MAX" \
  --kl-weight "$KL_WEIGHT" \
  --kl-topk "$KL_TOPK" \
  "${V33_PY_EXTRA[@]}"
