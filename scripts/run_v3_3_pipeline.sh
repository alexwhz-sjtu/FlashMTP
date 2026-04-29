#!/usr/bin/env bash
# 无缝衔接：MDLM → 最新 checkpoint → Streak（与 run_v3_3_training_flashmtp.sh 同套 dt/路径/W&B 约定）
# 用法: ./scripts/run_v3_3_pipeline.sh [--dt qz|a800] [透传到两个训练脚本的额外 python 参数]
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
mkdir -p "$FLASHMTP_V33_MDLM_OUT" "$FLASHMTP_V33_STREAK_OUT" "$CACHE_ROOT"

unset WANDB_RUN_NAME WANDB_RUN_ID || true
v33_wandb_defaults mdlm || exit $?
v33_print_banner "FlashMTP v3.3 Pipeline（MDLM → Streak）"

export DT
echo "==> Phase 1: MDLM"
bash "$ROOT/scripts/run_v3_3_mdlm.sh" "${V33_PY_EXTRA[@]}"

last="$(ls -d "$FLASHMTP_V33_MDLM_OUT"/epoch_* 2>/dev/null | sort -V | tail -1 || true)"
if [[ -z "$last" || ! -d "$last" ]]; then
  echo "未找到 MDLM checkpoint，请检查 $FLASHMTP_V33_MDLM_OUT" >&2
  exit 1
fi
export FLASHMTP_V33_INIT_CKPT="$last"
echo "==> Phase 2: Streak (init from $FLASHMTP_V33_INIT_CKPT)"

unset WANDB_RUN_NAME WANDB_RUN_ID || true
v33_wandb_defaults streak || exit $?

bash "$ROOT/scripts/run_v3_3_streak.sh" "${V33_PY_EXTRA[@]}"
