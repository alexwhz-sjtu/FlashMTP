#!/bin/bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"

RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to the existing compile_rejection result directory.}"
PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
TARGET_MODEL="${TARGET_MODEL:-/share/dai-sys/wanghanzhen/models/Qwen/Qwen3-8B}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MAX_SAMPLES="${MAX_SAMPLES:-50}"

RUN_NAMES=(
    "rnn_ce0.1_tv1.0_base0.0_temp1_rejection_compile"
    "rnn_easy_base0.2_temp0_compile"
    "rnn_easy_base0.2_temp0_compile"
)
MODEL_PATHS=(
    "${PROJECT_DIR}/cache/models/flashmtp_v2_mhrnn_direct_r512_ce0.1_tv1.0_wb_0.0_bgemma_21_qwen3_8b"
    "${PROJECT_DIR}/cache/models/flashmtp_v2_mhrnn_easy_direct_r512_wb_0.2_bgemma_21_qwen3_8b"
    "${PROJECT_DIR}/cache/models/flashmtp_v2_mhrnn_easy_direct_r512_wb_0.2_bgemma_21_qwen3_8b"
)
DATASETS=(
    "mt-bench"
    "mt-bench"
    "longbench_v2_64000_32000_multi_document_qa"
)
TEMPERATURES=("1" "0" "0")
VERIFICATION_MODES=("rejection" "match" "match")
GPUS=("0" "1" "2")

run_one() {
    local task_index="$1"
    local run_name="${RUN_NAMES[$task_index]}"
    local draft_path="${MODEL_PATHS[$task_index]}"
    local dataset="${DATASETS[$task_index]}"
    local temperature="${TEMPERATURES[$task_index]}"
    local verification_mode="${VERIFICATION_MODES[$task_index]}"
    local gpu="${GPUS[$task_index]}"
    local log_path="${RUN_ROOT}/logs/${run_name}/${dataset}.log"
    local status_path="${RUN_ROOT}/status/${run_name}/${dataset}.status"

    mkdir -p "$(dirname "${log_path}")" "$(dirname "${status_path}")"
    {
        printf '[RETRY START] %s gpu=%s run=%s dataset=%s\n' \
            "$(date --iso-8601=seconds)" "${gpu}" "${run_name}" "${dataset}"
        printf '[CONFIG] temperature=%s verification=%s compile_serial_head=true\n' \
            "${temperature}" "${verification_mode}"
    } > "${log_path}"
    printf 'running\n' > "${status_path}"

    CUDA_VISIBLE_DEVICES="${gpu}" \
        PYTHONUNBUFFERED=1 NO_COLOR=1 COLUMNS=200 \
        "${PYTHON_BIN}" evaluation/benchmark.py \
        --model-name-or-path "${TARGET_MODEL}" \
        --draft-name-or-path "${draft_path}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --max-samples "${MAX_SAMPLES}" \
        --dataset "${dataset}" \
        --batch-size 1 \
        --block-size 16 \
        --verify-block 16 \
        --temperature "${temperature}" \
        --stochastic-verification-mode "${verification_mode}" \
        --compile-serial-head \
        >> "${log_path}" 2>&1
    local exit_code=$?

    if [ "${exit_code}" -eq 0 ]; then
        printf 'completed\nfinished_at=%s\n' \
            "$(date --iso-8601=seconds)" > "${status_path}"
    else
        printf 'failed exit_code=%s\nfinished_at=%s\n' \
            "${exit_code}" "$(date --iso-8601=seconds)" > "${status_path}"
    fi
    return "${exit_code}"
}

pids=()
for task_index in "${!RUN_NAMES[@]}"; do
    run_one "${task_index}" &
    pids+=("$!")
done

overall_exit=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        overall_exit=1
    fi
done

"${PYTHON_BIN}" scripts/summarize_three_model_speedups.py "${RUN_ROOT}"
exit "${overall_exit}"
