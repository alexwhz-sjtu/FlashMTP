#!/usr/bin/env bash
# v3.3 阶段二：Streak（默认从 MDLM 加载 FLASHMTP_V33_INIT_CKPT；可与一体化脚本共用环境）
# STREAK_FROM_SCRATCH=1：不加载 Phase-1 权重，草案随机初始化。
# STREAK_WEIGHT：conf-streak 主 loss 系数（默认 1.0）。
# STREAK_CE_WEIGHT：块内除 anchor 外的逐位置平均 CE 辅助项系数（默认 0.2；例如 0.1）。
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
source "$ROOT/scripts/run_v5_lib.sh"
v33_parse_cli "$@" || exit $?
set +u
v33_export_common_training_env || exit $?
v33_export_paths_for_dt || exit $?

export FLASHMTP_V33_INIT_CKPT="${FLASHMTP_V33_INIT_CKPT:-/share/wanghanzhen/SpeculativeDecoding/NIPS26/FlashMTP_v1.4/cache/models/FlashMTP_v1.4_sample_400000_think_on_qwen3_8b_maxlen4096_epochs12_nnodes4}"

STREAK_FROM_SCRATCH="${STREAK_FROM_SCRATCH:-1}"
STREAK_WEIGHT="${STREAK_WEIGHT:-1.0}"
STREAK_CE_WEIGHT="${STREAK_CE_WEIGHT:-0.2}"
STREAK_INIT_ARGS=()
if [[ "${STREAK_FROM_SCRATCH}" == "1" ]]; then
  echo "Streak: STREAK_FROM_SCRATCH=1，不加载 --init-ckpt"
else
  INIT_CKPT="${FLASHMTP_V33_INIT_CKPT:?请设置 FLASHMTP_V33_INIT_CKPT 为 MDLM 的 epoch_*_step_* 目录，或设 STREAK_FROM_SCRATCH=1}"
  STREAK_INIT_ARGS=(--init-ckpt "$INIT_CKPT")
fi

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
  --streak-weight "$STREAK_WEIGHT" \
  --streak-ce-weight "$STREAK_CE_WEIGHT" \
  "${STREAK_INIT_ARGS[@]}" \
  "${V33_PY_EXTRA[@]}"
