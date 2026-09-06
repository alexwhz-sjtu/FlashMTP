#!/bin/bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
TARGET_MODEL="${TARGET_MODEL:-/share/dai-sys/wanghanzhen/models/Qwen/Qwen3-4B}"
DRAFT_PATH="${DRAFT_PATH:-${PROJECT_DIR}/cache/models/flashmtp_v2swa_w5_qwen3_4b_ep10}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/benchmark_results/flashmtp_v2swa_w5_qwen3_4b_ep10_longmix_20k_40k_$(date +%Y%m%d_%H%M%S)}"
MODEL_LABEL="${MODEL_LABEL:-flashmtp_v2swa_w5_qwen3_4b_ep10}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MAX_SAMPLES="${MAX_SAMPLES:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TEMPERATURE="${TEMPERATURE:-0}"
BLOCK_SIZE="${BLOCK_SIZE:-8}"
VERIFY_BLOCK="${VERIFY_BLOCK:-8}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
DATASETS=(
    "hotpotwikiqa_mixup"
    "lic_mixup"
    "multifieldqa_en_mixup"
    "swe_bench_20k_40k"
)

if [ "${#GPUS[@]}" -lt "${#DATASETS[@]}" ]; then
    echo "This runner needs at least ${#DATASETS[@]} GPUs" >&2
    exit 2
fi
for required in "${PYTHON_BIN}" "${TARGET_MODEL}" "${DRAFT_PATH}"; do
    if [ ! -e "${required}" ]; then
        echo "Missing required path: ${required}" >&2
        exit 2
    fi
done

declare -A GPU_MEMORY GPU_UTILIZATION
while IFS=',' read -r gpu_index gpu_util gpu_memory; do
    gpu_index="${gpu_index//[[:space:]]/}"
    gpu_util="${gpu_util//[[:space:]]/}"
    gpu_memory="${gpu_memory//[[:space:]]/}"
    GPU_MEMORY["${gpu_index}"]="${gpu_memory}"
    GPU_UTILIZATION["${gpu_index}"]="${gpu_util}"
done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits)
for index in "${!DATASETS[@]}"; do
    gpu="${GPUS[$index]}"
    if [ -z "${GPU_MEMORY[${gpu}]+x}" ] || [ "${GPU_MEMORY[${gpu}]:-999999}" -gt 1024 ] || [ "${GPU_UTILIZATION[${gpu}]:-999}" -gt 10 ]; then
        echo "GPU ${gpu} is unavailable or busy" >&2
        exit 3
    fi
done

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/status"
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
    printf 'compile_serial_head=false\n'
    printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
} > "${RUN_ROOT}/run_config.txt"

printf 'model\ttemperature\tdataset\trequested_samples\tgpu\tdraft_path\tlog_path\tstatus_path\n' > "${RUN_ROOT}/manifest.tsv"
pids=()
for index in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$index]}"
    gpu="${GPUS[$index]}"
    log_path="${RUN_ROOT}/logs/${dataset}.log"
    status_path="${RUN_ROOT}/status/${dataset}.status"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${MODEL_LABEL}" "${TEMPERATURE}" "${dataset}" "${MAX_SAMPLES}" \
        "${gpu}" "${DRAFT_PATH}" "${log_path}" "${status_path}" >> "${RUN_ROOT}/manifest.tsv"
    printf 'running\nstarted_at=%s\n' "$(date --iso-8601=seconds)" > "${status_path}"
    (
        printf '[START] %s gpu=%s dataset=%s compile_serial_head=false\n' \
            "$(date --iso-8601=seconds)" "${gpu}" "${dataset}"
        CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 NO_COLOR=1 COLUMNS=200 \
            "${PYTHON_BIN}" evaluation/benchmark.py \
            --model-name-or-path "${TARGET_MODEL}" \
            --draft-name-or-path "${DRAFT_PATH}" \
            --max-new-tokens "${MAX_NEW_TOKENS}" \
            --max-samples "${MAX_SAMPLES}" \
            --dataset "${dataset}" \
            --batch-size "${BATCH_SIZE}" \
            --block-size "${BLOCK_SIZE}" \
            --verify-block "${VERIFY_BLOCK}" \
            --temperature "${TEMPERATURE}"
        exit_code=$?
        if [ "${exit_code}" -eq 0 ]; then
            printf 'completed\nfinished_at=%s\n' "$(date --iso-8601=seconds)" > "${status_path}"
        else
            printf 'failed exit_code=%s\nfinished_at=%s\n' "${exit_code}" "$(date --iso-8601=seconds)" > "${status_path}"
        fi
        exit "${exit_code}"
    ) > "${log_path}" 2>&1 &
    pids+=("$!")
done

overall_exit=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        overall_exit=1
    fi
done
"${PYTHON_BIN}" scripts/summarize_benchmarks.py "${RUN_ROOT}" --verify-block "${VERIFY_BLOCK}" --per-run \
    > "${RUN_ROOT}/summary_generation.log" 2>&1 || overall_exit=1
printf 'finished_at=%s\noverall_exit=%s\n' "$(date --iso-8601=seconds)" "${overall_exit}" >> "${RUN_ROOT}/run_config.txt"
echo "RUN_ROOT=${RUN_ROOT}"
exit "${overall_exit}"
