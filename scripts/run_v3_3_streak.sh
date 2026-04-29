#!/usr/bin/env bash
# v3.3 阶段二：Streak（需 FLASHMTP_V33_INIT_CKPT；环境与 run_v3_3_training_flashmtp.sh 对齐）
# 用法: ./scripts/run_v3_3_streak.sh [--dt qz|a800] [其它参数透传]
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

INIT_CKPT="${FLASHMTP_V33_INIT_CKPT:?请设置 FLASHMTP_V33_INIT_CKPT 为 MDLM 的 epoch_*_step_* 目录}"

mkdir -p "$FLASHMTP_V33_STREAK_OUT" "$CACHE_ROOT"
v33_wandb_defaults streak || exit $?
v33_common_py_args streak

# 单机时默认同机端口 +1，避免紧接 MDLM 后端口仍占用；多机或与 MDLM 非连续跑时可设 STREAK_MASTER_PORT
if [ "${NNODES}" -eq 1 ] 2>/dev/null; then
  export V33_MASTER_PORT_OVERRIDE="${STREAK_MASTER_PORT:-$((MASTER_PORT + 1))}"
else
  export V33_MASTER_PORT_OVERRIDE="${STREAK_MASTER_PORT:-$MASTER_PORT}"
fi
v33_build_torchrun || exit $?
unset V33_MASTER_PORT_OVERRIDE

v33_print_banner "FlashMTP v3.3 — Phase 2 Streak"

exec "${V33_TORCHRUN[@]}" "$ROOT/scripts/train_flashmtp_streak.py" \
  "${V33_COMMON_PY[@]}" \
  --output-dir "$FLASHMTP_V33_STREAK_OUT" \
  --cache-dir "$CACHE_ROOT/streak_process" \
  --learning-rate "$LEARNING_RATE_STREAK" \
  --init-ckpt "$INIT_CKPT" \
  "${V33_PY_EXTRA[@]}"
