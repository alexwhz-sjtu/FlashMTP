#!/bin/bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
TARGET_MODEL="${TARGET_MODEL:-/share/dai-sys/wanghanzhen/models/Qwen/Qwen3-8B}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/benchmark_results/compile_rejection_$(date +%Y%m%d_%H%M%S)}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MAX_SAMPLES="${MAX_SAMPLES:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
VERIFY_BLOCK="${VERIFY_BLOCK:-16}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"

RUN_NAMES=(
    "rnn_ce0.1_tv1.0_base0.0_temp0_compile"
    "rnn_ce0.1_tv1.0_base0.0_temp1_rejection_compile"
    "rnn_easy_base0.2_temp0_compile"
)
MODEL_PATHS=(
    "${PROJECT_DIR}/cache/models/flashmtp_v2_mhrnn_direct_r512_ce0.1_tv1.0_wb_0.0_bgemma_21_qwen3_8b"
    "${PROJECT_DIR}/cache/models/flashmtp_v2_mhrnn_direct_r512_ce0.1_tv1.0_wb_0.0_bgemma_21_qwen3_8b"
    "${PROJECT_DIR}/cache/models/flashmtp_v2_mhrnn_easy_direct_r512_wb_0.2_bgemma_21_qwen3_8b"
)
TEMPERATURES=("0" "1" "0")
VERIFICATION_MODES=("match" "rejection" "match")
DATASETS=(
    "alpaca"
    "gsm8k"
    "mbpp"
    "aime25"
    "math500"
    "mt-bench"
    "longbench_v2_64000_32000_multi_document_qa"
    "longbench_v2_64000_32000_in_context_learning"
)

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/status" "${RUN_ROOT}/workers"

{
    printf 'model\ttemperature\tdataset\trequested_samples\tgpu\tdraft_path\tlog_path\tstatus_path\n'
    for run_index in "${!RUN_NAMES[@]}"; do
        for dataset_index in "${!DATASETS[@]}"; do
            gpu_index=$((dataset_index % ${#GPUS[@]}))
            gpu="${GPUS[$gpu_index]}"
            run_name="${RUN_NAMES[$run_index]}"
            dataset="${DATASETS[$dataset_index]}"
            draft_path="${MODEL_PATHS[$run_index]}"
            temperature="${TEMPERATURES[$run_index]}"
            log_path="${RUN_ROOT}/logs/${run_name}/${dataset}.log"
            status_path="${RUN_ROOT}/status/${run_name}/${dataset}.status"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${run_name}" "${temperature}" "${dataset}" "${MAX_SAMPLES}" \
                "${gpu}" "${draft_path}" "${log_path}" "${status_path}"
        done
    done
} > "${RUN_ROOT}/manifest.tsv"

{
    printf 'target_model=%s\n' "${TARGET_MODEL}"
    printf 'gpu_list=%s\n' "${GPU_LIST}"
    printf 'max_new_tokens=%s\n' "${MAX_NEW_TOKENS}"
    printf 'max_samples=%s\n' "${MAX_SAMPLES}"
    printf 'batch_size=%s\n' "${BATCH_SIZE}"
    printf 'block_size=%s\n' "${BLOCK_SIZE}"
    printf 'verify_block=%s\n' "${VERIFY_BLOCK}"
    printf 'compile_serial_head=true\n'
    printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
} > "${RUN_ROOT}/run_config.txt"

run_worker() {
    local worker_gpu="$1"
    local worker_log="${RUN_ROOT}/workers/gpu_${worker_gpu}.log"

    for run_index in "${!RUN_NAMES[@]}"; do
        local run_name="${RUN_NAMES[$run_index]}"
        local draft_path="${MODEL_PATHS[$run_index]}"
        local temperature="${TEMPERATURES[$run_index]}"
        local verification_mode="${VERIFICATION_MODES[$run_index]}"
        for dataset_index in "${!DATASETS[@]}"; do
            local gpu_index=$((dataset_index % ${#GPUS[@]}))
            local assigned_gpu="${GPUS[$gpu_index]}"
            if [ "${assigned_gpu}" != "${worker_gpu}" ]; then
                continue
            fi

            local dataset="${DATASETS[$dataset_index]}"
            local log_dir="${RUN_ROOT}/logs/${run_name}"
            local status_dir="${RUN_ROOT}/status/${run_name}"
            local log_path="${log_dir}/${dataset}.log"
            local status_path="${status_dir}/${dataset}.status"
            mkdir -p "${log_dir}" "${status_dir}"

            {
                printf '[START] %s gpu=%s run=%s dataset=%s\n' \
                    "$(date --iso-8601=seconds)" "${worker_gpu}" "${run_name}" \
                    "${dataset}"
                printf '[CONFIG] temperature=%s verification=%s compile_serial_head=true\n' \
                    "${temperature}" "${verification_mode}"
            } > "${log_path}"
            printf 'running\n' > "${status_path}"
            printf '[START] %s run=%s dataset=%s\n' \
                "$(date --iso-8601=seconds)" "${run_name}" "${dataset}" \
                >> "${worker_log}"

            CUDA_VISIBLE_DEVICES="${worker_gpu}" \
                PYTHONUNBUFFERED=1 NO_COLOR=1 COLUMNS=200 \
                "${PYTHON_BIN}" evaluation/benchmark.py \
                --model-name-or-path "${TARGET_MODEL}" \
                --draft-name-or-path "${draft_path}" \
                --max-new-tokens "${MAX_NEW_TOKENS}" \
                --max-samples "${MAX_SAMPLES}" \
                --dataset "${dataset}" \
                --batch-size "${BATCH_SIZE}" \
                --block-size "${BLOCK_SIZE}" \
                --verify-block "${VERIFY_BLOCK}" \
                --temperature "${temperature}" \
                --stochastic-verification-mode "${verification_mode}" \
                --compile-serial-head \
                >> "${log_path}" 2>&1
            local exit_code=$?

            if [ "${exit_code}" -eq 0 ]; then
                printf 'completed\nfinished_at=%s\n' \
                    "$(date --iso-8601=seconds)" > "${status_path}"
                printf '[DONE] %s run=%s dataset=%s\n' \
                    "$(date --iso-8601=seconds)" "${run_name}" "${dataset}" \
                    >> "${worker_log}"
            else
                printf 'failed exit_code=%s\nfinished_at=%s\n' \
                    "${exit_code}" "$(date --iso-8601=seconds)" > "${status_path}"
                printf '[FAIL] %s exit=%s run=%s dataset=%s\n' \
                    "$(date --iso-8601=seconds)" "${exit_code}" "${run_name}" \
                    "${dataset}" >> "${worker_log}"
            fi
        done
    done
}

printf 'RUN_ROOT=%s\n' "${RUN_ROOT}"
worker_pids=()
for gpu in "${GPUS[@]}"; do
    run_worker "${gpu}" &
    worker_pids+=("$!")
done

overall_exit=0
for worker_pid in "${worker_pids[@]}"; do
    if ! wait "${worker_pid}"; then
        overall_exit=1
    fi
done

"${PYTHON_BIN}" scripts/summarize_three_model_speedups.py "${RUN_ROOT}"
printf 'finished_at=%s\n' "$(date --iso-8601=seconds)" >> "${RUN_ROOT}/run_config.txt"
exit "${overall_exit}"
