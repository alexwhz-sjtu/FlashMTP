#!/bin/bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"

if [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${PROJECT_DIR}/.venv/bin/activate"
fi

PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
TARGET_MODEL="${TARGET_MODEL:-${WHZ_HOME:-/share/dai-sys/wanghanzhen}/models/Qwen/Qwen3-8B}"
DRAFT_PATH="${DRAFT_PATH:-${PROJECT_DIR}/cache/models/flashmtp_h100_prefix_condition_fuse18_sample_pb_80k_nlayers5_block_8_mhrnn_easy_direct_r512_ce0.1_tv1.0_wb_0.0_bgemma_12_maxlen4096_epochs6_Qwen3-8B/epoch_6_step_59730}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/benchmark_results/fuse18_block8_step59730_$(date +%Y%m%d_%H%M%S)}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MAX_SAMPLES="${MAX_SAMPLES:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TEMPERATURE="${TEMPERATURE:-0}"
BLOCK_SIZE="${BLOCK_SIZE:-8}"
VERIFY_BLOCK="${VERIFY_BLOCK:-8}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"

DATASETS=(
    "gsm8k"
    "math500"
    "aime25"
    "humaneval"
    "mbpp"
    "livecodebench"
    "mt-bench"
    "alpaca"
    "longbench_v2_64000_32000_multi_document_qa"
)

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/status" "${RUN_ROOT}/workers"

{
    printf 'target_model=%s\n' "${TARGET_MODEL}"
    printf 'draft_path=%s\n' "${DRAFT_PATH}"
    printf 'gpu_list=%s\n' "${GPU_LIST}"
    printf 'max_new_tokens=%s\n' "${MAX_NEW_TOKENS}"
    printf 'max_samples=%s\n' "${MAX_SAMPLES}"
    printf 'batch_size=%s\n' "${BATCH_SIZE}"
    printf 'block_size=%s\n' "${BLOCK_SIZE}"
    printf 'verify_block=%s\n' "${VERIFY_BLOCK}"
    printf 'temperature=%s\n' "${TEMPERATURE}"
    printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
} > "${RUN_ROOT}/run_config.txt"

{
    printf 'dataset\trequested_samples\tgpu\tlog_path\tstatus_path\n'
    for dataset_index in "${!DATASETS[@]}"; do
        gpu="${GPUS[$((dataset_index % ${#GPUS[@]}))]}"
        dataset="${DATASETS[$dataset_index]}"
        log_path="${RUN_ROOT}/logs/${dataset}.log"
        status_path="${RUN_ROOT}/status/${dataset}.status"
        printf '%s\t%s\t%s\t%s\t%s\n' \
            "${dataset}" "${MAX_SAMPLES}" "${gpu}" "${log_path}" "${status_path}"
    done
} > "${RUN_ROOT}/manifest.tsv"

run_worker() {
    local worker_gpu="$1"
    local worker_log="${RUN_ROOT}/workers/gpu_${worker_gpu}.log"

    for dataset_index in "${!DATASETS[@]}"; do
        local assigned_gpu="${GPUS[$((dataset_index % ${#GPUS[@]}))]}"
        if [ "${assigned_gpu}" != "${worker_gpu}" ]; then
            continue
        fi

        local dataset="${DATASETS[$dataset_index]}"
        local log_path="${RUN_ROOT}/logs/${dataset}.log"
        local status_path="${RUN_ROOT}/status/${dataset}.status"

        {
            printf '[START] %s gpu=%s dataset=%s\n' \
                "$(date --iso-8601=seconds)" "${worker_gpu}" "${dataset}"
            printf '[COMMAND] CUDA_VISIBLE_DEVICES=%s %s evaluation/benchmark.py ' \
                "${worker_gpu}" "${PYTHON_BIN}"
            printf '%q ' \
                --model-name-or-path "${TARGET_MODEL}" \
                --draft-name-or-path "${DRAFT_PATH}" \
                --max-new-tokens "${MAX_NEW_TOKENS}" \
                --max-samples "${MAX_SAMPLES}" \
                --dataset "${dataset}" \
                --batch-size "${BATCH_SIZE}" \
                --block-size "${BLOCK_SIZE}" \
                --verify-block "${VERIFY_BLOCK}" \
                --temperature "${TEMPERATURE}"
            printf '\n'
        } > "${log_path}"
        printf 'running\n' > "${status_path}"

        CUDA_VISIBLE_DEVICES="${worker_gpu}" \
            PYTHONUNBUFFERED=1 NO_COLOR=1 COLUMNS=200 \
            "${PYTHON_BIN}" evaluation/benchmark.py \
            --model-name-or-path "${TARGET_MODEL}" \
            --draft-name-or-path "${DRAFT_PATH}" \
            --max-new-tokens "${MAX_NEW_TOKENS}" \
            --max-samples "${MAX_SAMPLES}" \
            --dataset "${dataset}" \
            --batch-size "${BATCH_SIZE}" \
            --block-size "${BLOCK_SIZE}" \
            --verify-block "${VERIFY_BLOCK}" \
            --temperature "${TEMPERATURE}" \
            >> "${log_path}" 2>&1
        local exit_code=$?

        if [ "${exit_code}" -eq 0 ]; then
            printf 'completed\nfinished_at=%s\n' \
                "$(date --iso-8601=seconds)" > "${status_path}"
            printf '[DONE] %s dataset=%s\n' \
                "$(date --iso-8601=seconds)" "${dataset}" >> "${worker_log}"
        else
            printf 'failed exit_code=%s\nfinished_at=%s\n' \
                "${exit_code}" "$(date --iso-8601=seconds)" > "${status_path}"
            printf '[FAIL] %s exit=%s dataset=%s\n' \
                "$(date --iso-8601=seconds)" "${exit_code}" "${dataset}" >> "${worker_log}"
        fi
    done
}

printf 'RUN_ROOT=%s\n' "${RUN_ROOT}"
worker_pids=()
for gpu in "${GPUS[@]}"; do
    run_worker "${gpu}" &
    worker_pids+=("$!")
done

overall_exit=0
for pid in "${worker_pids[@]}"; do
    if ! wait "${pid}"; then
        overall_exit=1
    fi
done

if [ -x "${PROJECT_DIR}/scripts/summarize_benchmarks.py" ] || [ -f "${PROJECT_DIR}/scripts/summarize_benchmarks.py" ]; then
    "${PYTHON_BIN}" scripts/summarize_benchmarks.py "${RUN_ROOT}" --per-run \
        > "${RUN_ROOT}/summary_generation.log" 2>&1 || true
fi

printf 'finished_at=%s\noverall_exit=%s\n' \
    "$(date --iso-8601=seconds)" "${overall_exit}" >> "${RUN_ROOT}/run_config.txt"
exit "${overall_exit}"
