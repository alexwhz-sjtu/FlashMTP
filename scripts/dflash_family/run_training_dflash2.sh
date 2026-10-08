#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -eq 1 && "$1" == *.y*ml ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"
  PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
  exec "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/config/launch_dflash_family.py" \
    dflash2 "$1"
fi

exec "$(dirname "${BASH_SOURCE[0]}")/run_training_dflash_family.sh" dflash2 "$@"
