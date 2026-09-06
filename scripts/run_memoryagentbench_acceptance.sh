#!/bin/bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
TARGET_MODEL="${TARGET_MODEL:-/share/dai-sys/wanghanzhen/models/Qwen/Qwen3-4B}"
DRAFT_PATH="${DRAFT_PATH:-${PROJECT_DIR}/cache/models/flashmtp_v2swa_w5_qwen3_4b_ep10}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/benchmark_results/memoryagentbench_acceptance_$(date +%Y%m%d_%H%M%S)}"
GPU_LIST="${GPU_LIST:-0,3,4}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
BLOCK_SIZE="${BLOCK_SIZE:-8}"
VERIFY_BLOCK="${VERIFY_BLOCK:-8}"
ROPE_FACTOR="${ROPE_FACTOR:-4.0}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
CATEGORIES=(factconsolidation_64k eventqa_64k detectiveqa_free)
if [ "${#GPUS[@]}" -lt "${#CATEGORIES[@]}" ]; then
    echo "Need at least ${#CATEGORIES[@]} GPUs in GPU_LIST" >&2
    exit 2
fi

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/results" "${RUN_ROOT}/summaries" "${RUN_ROOT}/status"
{
    printf 'target_model=%s\n' "${TARGET_MODEL}"
    printf 'draft_path=%s\n' "${DRAFT_PATH}"
    printf 'temperature=%s\n' "${TEMPERATURE}"
    printf 'max_new_tokens=%s\n' "${MAX_NEW_TOKENS}"
    printf 'block_size=%s\n' "${BLOCK_SIZE}"
    printf 'verify_block=%s\n' "${VERIFY_BLOCK}"
    printf 'rope_scaling=yarn\nrope_factor=%s\n' "${ROPE_FACTOR}"
    printf 'gpu_list=%s\nstarted_at=%s\n' "${GPU_LIST}" "$(date --iso-8601=seconds)"
} > "${RUN_ROOT}/run_config.txt"

pids=()
for index in "${!CATEGORIES[@]}"; do
    category="${CATEGORIES[$index]}"
    gpu="${GPUS[$index]}"
    log_path="${RUN_ROOT}/logs/${category}.log"
    output_path="${RUN_ROOT}/results/${category}.jsonl"
    summary_path="${RUN_ROOT}/summaries/${category}.json"
    status_path="${RUN_ROOT}/status/${category}.status"
    printf 'running\nstarted_at=%s\n' "$(date --iso-8601=seconds)" > "${status_path}"
    (
        CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 NO_COLOR=1 COLUMNS=200 \
            "${PYTHON_BIN}" evaluation/memoryagentbench_acceptance.py \
            --category "${category}" \
            --model-name-or-path "${TARGET_MODEL}" \
            --draft-name-or-path "${DRAFT_PATH}" \
            --output-jsonl "${output_path}" \
            --summary-json "${summary_path}" \
            --max-new-tokens "${MAX_NEW_TOKENS}" \
            --temperature "${TEMPERATURE}" \
            --block-size "${BLOCK_SIZE}" \
            --verify-block "${VERIFY_BLOCK}" \
            --rope-scaling yarn \
            --rope-factor "${ROPE_FACTOR}"
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
printf 'finished_at=%s\noverall_exit=%s\n' "$(date --iso-8601=seconds)" "${overall_exit}" >> "${RUN_ROOT}/run_config.txt"
echo "RUN_ROOT=${RUN_ROOT}"
exit "${overall_exit}"
