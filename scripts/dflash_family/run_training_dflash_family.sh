#!/usr/bin/env bash
set -euo pipefail

ALGORITHM="${1:?usage: run_training_dflash_family.sh ALGORITHM [trainer args...]}"
shift
case "${ALGORITHM}" in dflash|dflash2|dspark) ;; *) echo "unsupported algorithm: ${ALGORITHM}" >&2; exit 2 ;; esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
: "${TARGET_MODEL:?set TARGET_MODEL}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"

DATA_ARGS=()
if [[ -n "${TRAIN_HIDDEN_STATES:-}" ]]; then
  DATA_ARGS+=(--train-hidden-states-path "${TRAIN_HIDDEN_STATES}")
else
  : "${TRAIN_DATA:?set TRAIN_DATA or TRAIN_HIDDEN_STATES}"
  DATA_ARGS+=(--train-data-path "${TRAIN_DATA}")
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"
CMD=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --nproc_per_node "${NPROC_PER_NODE}"
  --master_port "${MASTER_PORT}"
  -m "scripts.dflash_family.train_${ALGORITHM}"
  --target-model-path "${TARGET_MODEL}"
  --output-dir "${OUTPUT_DIR}"
  "${DATA_ARGS[@]}"
  "$@"
)
printf '%s training:' "${ALGORITHM}"
printf ' %q' "${CMD[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" == "1" ]]; then exit 0; fi
exec "${CMD[@]}"
