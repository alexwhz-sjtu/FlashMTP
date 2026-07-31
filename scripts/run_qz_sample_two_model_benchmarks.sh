#!/bin/bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
TARGET_MODEL="${TARGET_MODEL:-/share/dai-sys/wanghanzhen/models/Qwen/Qwen3-8B}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/benchmark_results/qz_sample_two_models_$(date +%Y%m%d_%H%M%S)}"
GPU_LIST="${GPU_LIST:-4,5,6,7}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MAX_SAMPLES="${MAX_SAMPLES:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
VERIFY_BLOCK="${VERIFY_BLOCK:-16}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"

MODEL_NAMES=(
    "ce0.02_tv1.0"
    "ce0.1_tv1.0_wb0.04"
)
MODEL_PATHS=(
    "${PROJECT_DIR}/cache/models/flashmtp_qz_sample_80000_think_off_nlayers5_block_16_mhrnn_direct_r512_ce0.02_tv1.0_maxlen4096_epochs6_Qwen3-8B"
    "${PROJECT_DIR}/cache/models/flashmtp_qz_sample_80000_think_off_nlayers5_block_16_mhrnn_direct_r512_ce0.1_tv1.0_wb0.04_maxlen4096_epochs6_Qwen3-8B"
)
DATASETS=(
    "alpaca"
    "mt-bench"
    "gsm8k"
    "math500"
    "aime25"
    "mbpp"
    "livecodebench"
)
TEMPERATURES=("0" "1")

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/status" "${RUN_ROOT}/workers"

{
    printf 'model\ttemperature\tverification\tdataset\trequested_samples\tgpu\tdraft_path\tlog_path\tstatus_path\n'
    combo_index=0
    for model_index in "${!MODEL_NAMES[@]}"; do
        for temperature in "${TEMPERATURES[@]}"; do
            if [ "${temperature}" = "0" ]; then
                verification="match"
            else
                verification="rejection"
            fi
            for dataset_index in "${!DATASETS[@]}"; do
                gpu_index=$(((dataset_index + combo_index) % ${#GPUS[@]}))
                gpu="${GPUS[$gpu_index]}"
                model="${MODEL_NAMES[$model_index]}"
                dataset="${DATASETS[$dataset_index]}"
                draft_path="${MODEL_PATHS[$model_index]}"
                log_path="${RUN_ROOT}/logs/${model}/temperature_${temperature}/${dataset}.log"
                status_path="${RUN_ROOT}/status/${model}/temperature_${temperature}/${dataset}.status"
                printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "${model}" "${temperature}" "${verification}" "${dataset}" \
                    "${MAX_SAMPLES}" "${gpu}" "${draft_path}" "${log_path}" \
                    "${status_path}"
            done
            combo_index=$((combo_index + 1))
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
    printf 'temperature_0_verification=match\n'
    printf 'temperature_1_verification=rejection\n'
    printf 'compile_serial_head=false\n'
    printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
} > "${RUN_ROOT}/run_config.txt"

run_worker() {
    local worker_gpu="$1"
    local worker_log="${RUN_ROOT}/workers/gpu_${worker_gpu}.log"
    local combo_index=0

    for model_index in "${!MODEL_NAMES[@]}"; do
        local model="${MODEL_NAMES[$model_index]}"
        local draft_path="${MODEL_PATHS[$model_index]}"
        for temperature in "${TEMPERATURES[@]}"; do
            local verification="match"
            local extra_args=()
            if [ "${temperature}" = "1" ]; then
                verification="rejection"
                extra_args=(--stochastic-verification-mode rejection)
            fi

            for dataset_index in "${!DATASETS[@]}"; do
                local gpu_index=$(((dataset_index + combo_index) % ${#GPUS[@]}))
                local assigned_gpu="${GPUS[$gpu_index]}"
                if [ "${assigned_gpu}" != "${worker_gpu}" ]; then
                    continue
                fi

                local dataset="${DATASETS[$dataset_index]}"
                local log_dir="${RUN_ROOT}/logs/${model}/temperature_${temperature}"
                local status_dir="${RUN_ROOT}/status/${model}/temperature_${temperature}"
                local log_path="${log_dir}/${dataset}.log"
                local status_path="${status_dir}/${dataset}.status"
                mkdir -p "${log_dir}" "${status_dir}"

                {
                    printf '[START] %s gpu=%s model=%s temperature=%s verification=%s dataset=%s\n' \
                        "$(date --iso-8601=seconds)" "${worker_gpu}" "${model}" \
                        "${temperature}" "${verification}" "${dataset}"
                    printf '[COMMAND] CUDA_VISIBLE_DEVICES=%s %s evaluation/benchmark.py ' \
                        "${worker_gpu}" "${PYTHON_BIN}"
                    printf '%q ' \
                        --model-name-or-path "${TARGET_MODEL}" \
                        --draft-name-or-path "${draft_path}" \
                        --max-new-tokens "${MAX_NEW_TOKENS}" \
                        --max-samples "${MAX_SAMPLES}" \
                        --dataset "${dataset}" \
                        --batch-size "${BATCH_SIZE}" \
                        --block-size "${BLOCK_SIZE}" \
                        --verify-block "${VERIFY_BLOCK}" \
                        --temperature "${temperature}" \
                        "${extra_args[@]}"
                    printf '\n'
                } > "${log_path}"
                printf 'running\n' > "${status_path}"
                printf '[START] %s model=%s temp=%s dataset=%s\n' \
                    "$(date --iso-8601=seconds)" "${model}" "${temperature}" \
                    "${dataset}" >> "${worker_log}"

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
                    "${extra_args[@]}" \
                    >> "${log_path}" 2>&1
                local exit_code=$?

                if [ "${exit_code}" -eq 0 ]; then
                    printf 'completed\nfinished_at=%s\n' \
                        "$(date --iso-8601=seconds)" > "${status_path}"
                    printf '[DONE] %s model=%s temp=%s dataset=%s\n' \
                        "$(date --iso-8601=seconds)" "${model}" "${temperature}" \
                        "${dataset}" >> "${worker_log}"
                else
                    printf 'failed exit_code=%s\nfinished_at=%s\n' \
                        "${exit_code}" "$(date --iso-8601=seconds)" > "${status_path}"
                    printf '[FAIL] %s exit=%s model=%s temp=%s dataset=%s\n' \
                        "$(date --iso-8601=seconds)" "${exit_code}" "${model}" \
                        "${temperature}" "${dataset}" >> "${worker_log}"
                fi
            done
            combo_index=$((combo_index + 1))
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

"${PYTHON_BIN}" scripts/summarize_benchmarks.py "${RUN_ROOT}" --per-run
printf 'finished_at=%s\n' "$(date --iso-8601=seconds)" >> "${RUN_ROOT}/run_config.txt"
printf 'overall_exit=%s\n' "${overall_exit}" >> "${RUN_ROOT}/run_config.txt"
exit "${overall_exit}"
