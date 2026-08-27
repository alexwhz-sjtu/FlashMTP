#!/bin/bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
TARGET_MODEL="${TARGET_MODEL:-/share/dai-sys/models/qwen3-4b}"
DRAFT_MODEL="${DRAFT_MODEL:-/share/dai-sys/wanghanzhen/projects/MTP/FlashMTP_models/FlashMTP_v2_mhvanilla_additive_r256_qwen3_4b_ep6}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/benchmark_results/FlashMTP_v2_mhvanilla_additive_r256_qwen3_4b_ep6_$(date +%Y%m%d_%H%M%S)}"
TMP_ROOT="${RUN_ROOT}/tmp"
MAX_SAMPLES="${MAX_SAMPLES:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TEMPERATURE="${TEMPERATURE:-0}"
GPU_LIST="${GPU_LIST:-6,7}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
MAX_IDLE_MEMORY_MIB="${MAX_IDLE_MEMORY_MIB:-70000}"

QUEUES=(
    "longbench_v2_64000_32000_single_document_qa longbench_v2_64000_32000_multi_document_qa longbench_v2_64000_32000_long_dialogue alpaca gsm8k math500 mbpp"
    "longbench_v2_64000_32000_structured_data longbench_v2_64000_32000_in_context_learning longbench_v2_64000_32000_code_repo livecodebench humaneval mt-bench aime25"
)

for gpu in "${GPUS[@]}"; do
    used_memory="$(nvidia-smi --id="${gpu}" --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{sum += $1} END {print sum + 0}')"
    free_memory=$((81559 - used_memory))
    if [ "${free_memory}" -lt 20000 ]; then
        printf 'Refusing to start: GPU %s has only ~%s MiB free (used %s MiB).\n' \
            "${gpu}" "${free_memory}" "${used_memory}" >&2
        exit 2
    fi
done

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/status" "${RUN_ROOT}/workers" "${TMP_ROOT}"

{
    printf 'target_model=%s\n' "${TARGET_MODEL}"
    printf 'draft_path=%s\n' "${DRAFT_MODEL}"
    printf 'gpu_list=%s\n' "${GPU_LIST}"
    printf 'max_samples=%s\n' "${MAX_SAMPLES}"
    printf 'max_new_tokens=%s\n' "${MAX_NEW_TOKENS}"
    printf 'batch_size=%s\n' "${BATCH_SIZE}"
    printf 'temperature=%s\n' "${TEMPERATURE}"
    printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
} > "${RUN_ROOT}/run_config.txt"

run_worker() {
    local queue_index="$1"
    local worker_gpu="${GPUS[$queue_index]}"
    local worker_log="${RUN_ROOT}/workers/gpu_${worker_gpu}.log"
    local dataset

    : > "${worker_log}"
    for dataset in ${QUEUES[$queue_index]}; do
        local log_path="${RUN_ROOT}/logs/${dataset}.log"
        local status_path="${RUN_ROOT}/status/${dataset}.status"

        {
            printf '[START] %s gpu=%s dataset=%s\n' \
                "$(date --iso-8601=seconds)" "${worker_gpu}" "${dataset}"
            printf '[COMMAND] CUDA_VISIBLE_DEVICES=%s %q evaluation/benchmark.py ' \
                "${worker_gpu}" "${PYTHON_BIN}"
            printf '%q ' \
                --model-name-or-path "${TARGET_MODEL}" \
                --draft-name-or-path "${DRAFT_MODEL}" \
                --dataset "${dataset}" \
                --max-samples "${MAX_SAMPLES}" \
                --max-new-tokens "${MAX_NEW_TOKENS}" \
                --batch-size "${BATCH_SIZE}" \
                --temperature "${TEMPERATURE}"
            printf '\n'
        } > "${log_path}"
        printf 'running\nstarted_at=%s\ngpu=%s\n' \
            "$(date --iso-8601=seconds)" "${worker_gpu}" > "${status_path}"
        printf '[START] %s dataset=%s\n' \
            "$(date --iso-8601=seconds)" "${dataset}" >> "${worker_log}"

        env CUDA_VISIBLE_DEVICES="${worker_gpu}" \
            TMPDIR="${TMP_ROOT}" \
            PYTHONUNBUFFERED=1 NO_COLOR=1 COLUMNS=200 \
            "${PYTHON_BIN}" evaluation/benchmark.py \
            --model-name-or-path "${TARGET_MODEL}" \
            --draft-name-or-path "${DRAFT_MODEL}" \
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
            printf '[DONE] %s dataset=%s\n' \
                "$(date --iso-8601=seconds)" "${dataset}" >> "${worker_log}"
        else
            printf 'failed\nexit_code=%s\nfinished_at=%s\ngpu=%s\n' \
                "${exit_code}" "$(date --iso-8601=seconds)" "${worker_gpu}" > "${status_path}"
            printf '[FAIL] %s exit=%s dataset=%s\n' \
                "$(date --iso-8601=seconds)" "${exit_code}" "${dataset}" >> "${worker_log}"
        fi
    done
}

printf 'RUN_ROOT=%s\n' "${RUN_ROOT}"
worker_pids=()
for queue_index in "${!QUEUES[@]}"; do
    run_worker "${queue_index}" &
    worker_pids+=("$!")
done

overall_exit=0
for worker_pid in "${worker_pids[@]}"; do
    if ! wait "${worker_pid}"; then
        overall_exit=1
    fi
done

"${PYTHON_BIN}" scripts/summarize_ep8_ep10_benchmark.py "${RUN_ROOT}" || overall_exit=1
printf 'finished_at=%s\noverall_exit=%s\n' \
    "$(date --iso-8601=seconds)" "${overall_exit}" >> "${RUN_ROOT}/run_config.txt"
exit "${overall_exit}"
