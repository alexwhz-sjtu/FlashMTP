#!/bin/bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
TARGET_MODEL="${TARGET_MODEL:-/data/wanghanzhen/models/Qwen3-8B}"
DRAFT_PATH="${DRAFT_PATH:-${PROJECT_DIR}/cache/models/flashmtp_v2swa_2n16g_tp2_ac128_vanilla_additive_r256_mixed2360k_w5_chs18_block8_maxlen30720_ep8/epoch_2_step_390000}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/benchmark_results/step390000_$(date +%Y%m%d_%H%M%S)}"
MODEL_LABEL="${MODEL_LABEL:-step390000}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MAX_SAMPLES="${MAX_SAMPLES:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TEMPERATURE="${TEMPERATURE:-0}"
BLOCK_SIZE="${BLOCK_SIZE:-8}"
VERIFY_BLOCK="${VERIFY_BLOCK:-8}"
MAX_IDLE_MEMORY_MIB="${MAX_IDLE_MEMORY_MIB:-1024}"
MAX_IDLE_UTILIZATION="${MAX_IDLE_UTILIZATION:-10}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"

DATASETS=(
    "alpaca"
    "gsm8k"
    "math500"
    "mbpp"
    "livecodebench"
    "humaneval"
    "mt-bench"
    "aime25"
    "longbench_v2_64000_32000_single_document_qa"
    "longbench_v2_64000_32000_multi_document_qa"
    "longbench_v2_64000_32000_long_dialogue"
    "longbench_v2_64000_32000_structured_data"
    "longbench_v2_64000_32000_in_context_learning"
    "longbench_v2_64000_32000_code_repo"
)

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Missing Python environment: ${PYTHON_BIN}" >&2
    exit 2
fi
if [ ! -d "${TARGET_MODEL}" ]; then
    echo "Missing target model: ${TARGET_MODEL}" >&2
    exit 2
fi
if [ ! -d "${DRAFT_PATH}" ]; then
    echo "Missing draft checkpoint: ${DRAFT_PATH}" >&2
    exit 2
fi

declare -A GPU_MEMORY
declare -A GPU_UTILIZATION
while IFS=',' read -r gpu_index gpu_util gpu_memory; do
    gpu_index="${gpu_index//[[:space:]]/}"
    gpu_util="${gpu_util//[[:space:]]/}"
    gpu_memory="${gpu_memory//[[:space:]]/}"
    GPU_MEMORY["${gpu_index}"]="${gpu_memory}"
    GPU_UTILIZATION["${gpu_index}"]="${gpu_util}"
done < <(
    nvidia-smi \
        --query-gpu=index,utilization.gpu,memory.used \
        --format=csv,noheader,nounits
)

for gpu in "${GPUS[@]}"; do
    if [ -z "${GPU_MEMORY[${gpu}]+x}" ]; then
        echo "GPU ${gpu} is not present" >&2
        exit 2
    fi
    if [ "${GPU_MEMORY[${gpu}]}" -gt "${MAX_IDLE_MEMORY_MIB}" ] || \
       [ "${GPU_UTILIZATION[${gpu}]}" -gt "${MAX_IDLE_UTILIZATION}" ]; then
        echo "GPU ${gpu} is busy: memory=${GPU_MEMORY[${gpu}]} MiB, utilization=${GPU_UTILIZATION[${gpu}]}%" >&2
        echo "Refusing to start benchmarks while a requested GPU is occupied." >&2
        exit 3
    fi
done

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
    printf 'model\ttemperature\tdataset\trequested_samples\tgpu\tdraft_path\tlog_path\tstatus_path\n'
    for dataset_index in "${!DATASETS[@]}"; do
        gpu="${GPUS[$((dataset_index % ${#GPUS[@]}))]}"
        dataset="${DATASETS[$dataset_index]}"
        log_path="${RUN_ROOT}/logs/${dataset}.log"
        status_path="${RUN_ROOT}/status/${dataset}.status"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${MODEL_LABEL}" "${TEMPERATURE}" "${dataset}" "${MAX_SAMPLES}" \
            "${gpu}" "${DRAFT_PATH}" "${log_path}" "${status_path}"
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

        printf 'running\nstarted_at=%s\n' "$(date --iso-8601=seconds)" > "${status_path}"
        {
            printf '[START] %s gpu=%s dataset=%s\n' \
                "$(date --iso-8601=seconds)" "${worker_gpu}" "${dataset}"
            printf '[COMMAND] CUDA_VISIBLE_DEVICES=%s %q evaluation/benchmark.py ' \
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
        else
            printf 'failed exit_code=%s\nfinished_at=%s\n' \
                "${exit_code}" "$(date --iso-8601=seconds)" > "${status_path}"
        fi
        printf '%s dataset=%s exit=%s\n' \
            "$(date --iso-8601=seconds)" "${dataset}" "${exit_code}" >> "${worker_log}"
    done
}

echo "RUN_ROOT=${RUN_ROOT}"
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

"${PYTHON_BIN}" scripts/summarize_benchmarks.py \
    "${RUN_ROOT}" --verify-block "${VERIFY_BLOCK}" --per-run \
    > "${RUN_ROOT}/summary_generation.log" 2>&1 || overall_exit=1

printf 'finished_at=%s\noverall_exit=%s\n' \
    "$(date --iso-8601=seconds)" "${overall_exit}" >> "${RUN_ROOT}/run_config.txt"
exit "${overall_exit}"
