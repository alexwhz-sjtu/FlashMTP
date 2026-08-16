#!/bin/bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"

TRAIN_LOG="${TRAIN_LOG:?TRAIN_LOG is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
MODEL_LABEL="${MODEL_LABEL:?MODEL_LABEL is required}"
EXPECTED_FINAL_STEP="${EXPECTED_FINAL_STEP:-79640}"
TARGET_MODEL="${TARGET_MODEL:-/data/wanghanzhen/models/Qwen3-8B}"
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
WATCH_STATE="${WATCH_STATE:-${PROJECT_DIR}/logs/watch_${MODEL_LABEL}.state}"

mkdir -p "$(dirname "${WATCH_STATE}")"

write_state() {
    printf 'state=%s\ntime=%s\n%s\n' \
        "$1" "$(date --iso-8601=seconds)" "${2:-}" > "${WATCH_STATE}"
}

training_is_running() {
    ps -eo args= | grep -F "${PROJECT_DIR}/.venv/bin/python -m torch.distributed.run" \
        | grep -F -- "--output-dir ${OUTPUT_DIR}" >/dev/null
}

all_requested_gpus_idle() {
    local requested=",${GPU_LIST},"
    nvidia-smi \
        --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits \
        | awk -F',' -v requested="${requested}" '
            {
                gsub(/[[:space:]]/, "", $1)
                gsub(/[[:space:]]/, "", $2)
                gsub(/[[:space:]]/, "", $3)
                if (index(requested, "," $1 ",") && ($2 > 1024 || $3 > 10)) {
                    busy = 1
                }
            }
            END { exit busy }
        '
}

write_state "waiting_for_training" "output_dir=${OUTPUT_DIR}"
while ! grep -aq '训练完成！' "${TRAIN_LOG}" 2>/dev/null; do
    if ! training_is_running; then
        write_state "training_failed" "success_marker_missing=1"
        exit 10
    fi
    sleep "${CHECK_INTERVAL}"
done

latest_checkpoint="$({
    find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -type d \
        -name 'epoch_*_step_*' -printf '%f\n'
} | sort -V | tail -1)"

if [ -z "${latest_checkpoint}" ]; then
    write_state "checkpoint_missing" "output_dir=${OUTPUT_DIR}"
    exit 11
fi

latest_step="${latest_checkpoint##*_step_}"
if [ -n "${EXPECTED_FINAL_STEP}" ] && [ "${latest_step}" != "${EXPECTED_FINAL_STEP}" ]; then
    write_state "unexpected_final_step" \
        "latest_checkpoint=${latest_checkpoint} expected_step=${EXPECTED_FINAL_STEP}"
    exit 12
fi

draft_path="${OUTPUT_DIR}/${latest_checkpoint}"
write_state "waiting_for_idle_gpus" "draft_path=${draft_path}"
while ! all_requested_gpus_idle; do
    sleep "${CHECK_INTERVAL}"
done

run_root="${PROJECT_DIR}/benchmark_results/${MODEL_LABEL}_$(date +%Y%m%d_%H%M%S)"
write_state "benchmark_running" "draft_path=${draft_path}\nrun_root=${run_root}"

benchmark_exit=0
TARGET_MODEL="${TARGET_MODEL}" \
DRAFT_PATH="${draft_path}" \
MODEL_LABEL="${MODEL_LABEL}" \
RUN_ROOT="${run_root}" \
GPU_LIST="${GPU_LIST}" \
MAX_SAMPLES=50 \
MAX_NEW_TOKENS=512 \
BATCH_SIZE=1 \
TEMPERATURE=0 \
BLOCK_SIZE=8 \
VERIFY_BLOCK=8 \
bash scripts/run_step390000_benchmarks.sh || benchmark_exit=$?

if [ "${benchmark_exit}" -eq 0 ]; then
    write_state "benchmark_completed" "draft_path=${draft_path}\nrun_root=${run_root}"
else
    write_state "benchmark_failed" \
        "exit_code=${benchmark_exit}\ndraft_path=${draft_path}\nrun_root=${run_root}"
fi
exit "${benchmark_exit}"
