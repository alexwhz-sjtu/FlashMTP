#!/bin/bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
TARGET_MODEL="/share/dai-sys/wanghanzhen/models/Qwen/Qwen3-4B"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/benchmark_results/ep8_ep10_long_short_20260813}"
TMP_ROOT="${RUN_ROOT}/tmp"
MAX_SAMPLES=50
MAX_NEW_TOKENS=512
BATCH_SIZE=1
TEMPERATURE=0
GPUS=(0 1 2 3)

MODEL_NAMES=(ep10 ep8)
MODEL_PATHS=(
    "/share/dai-sys/wanghanzhen/projects/MTP/FlashMTP_models/FlashMTP_v2_mhrnn_easy_direct_r512_qwen3_4b_ep10"
    "/share/dai-sys/wanghanzhen/projects/MTP/FlashMTP_models/FlashMTP_v2_mhrnn_easy_direct_r512_qwen3_4b_ep8"
)
DATASETS=(
    longbench_v2_64000_32000_single_document_qa
    longbench_v2_64000_32000_multi_document_qa
    longbench_v2_64000_32000_long_dialogue
    longbench_v2_64000_32000_structured_data
    longbench_v2_64000_32000_in_context_learning
    longbench_v2_64000_32000_code_repo
    alpaca
    gsm8k
    math500
    mbpp
    livecodebench
    humaneval
    mt-bench
    aime25
)

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/status" "${RUN_ROOT}/workers" "${TMP_ROOT}"

{
    printf 'model\tdataset\tgpu\tdraft_path\tlog_path\tstatus_path\n'
    task_index=0
    for model_index in "${!MODEL_NAMES[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            gpu="${GPUS[$((task_index % ${#GPUS[@]}))]}"
            model="${MODEL_NAMES[$model_index]}"
            draft_path="${MODEL_PATHS[$model_index]}"
            log_path="${RUN_ROOT}/logs/${model}/${dataset}.log"
            status_path="${RUN_ROOT}/status/${model}/${dataset}.status"
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${model}" "${dataset}" "${gpu}" "${draft_path}" "${log_path}" "${status_path}"
            task_index=$((task_index + 1))
        done
    done
} > "${RUN_ROOT}/manifest.tsv"

{
    printf 'target_model=%s\n' "${TARGET_MODEL}"
    printf 'gpu_list=%s\n' "${GPUS[*]}"
    printf 'max_samples=%s\n' "${MAX_SAMPLES}"
    printf 'max_new_tokens=%s\n' "${MAX_NEW_TOKENS}"
    printf 'batch_size=%s\n' "${BATCH_SIZE}"
    printf 'temperature=%s\n' "${TEMPERATURE}"
    printf 'benchmark_script=%s\n' "${PROJECT_DIR}/evaluation/benchmark.py"
    printf 'reference_script=%s\n' "/share/dai-sys/wanghanzhen/projects/MTP/FlashMTP_v2swa/evaluation/benchmark.py"
    printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
} > "${RUN_ROOT}/run_config.txt"

run_worker() {
    local worker_gpu="$1"
    local worker_log="${RUN_ROOT}/workers/gpu_${worker_gpu}.log"
    local task_index=0

    : > "${worker_log}"
    for model_index in "${!MODEL_NAMES[@]}"; do
        local model="${MODEL_NAMES[$model_index]}"
        local draft_path="${MODEL_PATHS[$model_index]}"
        for dataset in "${DATASETS[@]}"; do
            local assigned_gpu="${GPUS[$((task_index % ${#GPUS[@]}))]}"
            task_index=$((task_index + 1))
            if [ "${assigned_gpu}" != "${worker_gpu}" ]; then
                continue
            fi

            local log_dir="${RUN_ROOT}/logs/${model}"
            local status_dir="${RUN_ROOT}/status/${model}"
            local log_path="${log_dir}/${dataset}.log"
            local status_path="${status_dir}/${dataset}.status"
            mkdir -p "${log_dir}" "${status_dir}"

            {
                printf '[START] %s gpu=%s model=%s dataset=%s\n' \
                    "$(date --iso-8601=seconds)" "${worker_gpu}" "${model}" "${dataset}"
                printf '[COMMAND] CUDA_VISIBLE_DEVICES=%s %q evaluation/benchmark.py ' \
                    "${worker_gpu}" "${PYTHON_BIN}"
                printf '%q ' \
                    --model-name-or-path "${TARGET_MODEL}" \
                    --draft-name-or-path "${draft_path}" \
                    --dataset "${dataset}" \
                    --max-samples "${MAX_SAMPLES}" \
                    --max-new-tokens "${MAX_NEW_TOKENS}" \
                    --batch-size "${BATCH_SIZE}" \
                    --temperature "${TEMPERATURE}"
                printf '\n'
            } > "${log_path}"
            printf 'running\nstarted_at=%s\ngpu=%s\n' \
                "$(date --iso-8601=seconds)" "${worker_gpu}" > "${status_path}"
            printf '[START] %s model=%s dataset=%s\n' \
                "$(date --iso-8601=seconds)" "${model}" "${dataset}" >> "${worker_log}"

            env CUDA_VISIBLE_DEVICES="${worker_gpu}" \
                TMPDIR="${TMP_ROOT}" \
                PYTHONUNBUFFERED=1 NO_COLOR=1 COLUMNS=200 \
                "${PYTHON_BIN}" evaluation/benchmark.py \
                --model-name-or-path "${TARGET_MODEL}" \
                --draft-name-or-path "${draft_path}" \
                --dataset "${dataset}" \
                --max-samples "${MAX_SAMPLES}" \
                --max-new-tokens "${MAX_NEW_TOKENS}" \
                --batch-size "${BATCH_SIZE}" \
                --temperature "${TEMPERATURE}" \
                >> "${log_path}" 2>&1
            local exit_code=$?

            if [ "${exit_code}" -eq 0 ]; then
                printf 'completed\nfinished_at=%s\ngpu=%s\n' \
                    "$(date --iso-8601=seconds)" "${worker_gpu}" > "${status_path}"
                printf '[DONE] %s model=%s dataset=%s\n' \
                    "$(date --iso-8601=seconds)" "${model}" "${dataset}" >> "${worker_log}"
            else
                printf 'failed\nexit_code=%s\nfinished_at=%s\ngpu=%s\n' \
                    "${exit_code}" "$(date --iso-8601=seconds)" "${worker_gpu}" > "${status_path}"
                printf '[FAIL] %s exit=%s model=%s dataset=%s\n' \
                    "$(date --iso-8601=seconds)" "${exit_code}" "${model}" "${dataset}" >> "${worker_log}"
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

printf 'finished_at=%s\n' "$(date --iso-8601=seconds)" >> "${RUN_ROOT}/run_config.txt"
exit "${overall_exit}"
