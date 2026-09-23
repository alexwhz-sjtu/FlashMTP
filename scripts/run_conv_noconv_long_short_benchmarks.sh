#!/bin/bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
TARGET_MODEL="${TARGET_MODEL:-/data/wanghanzhen/models/Qwen3-8B}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/benchmark_results/conv_noconv_long_short_$(date +%Y%m%d_%H%M%S)}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TEMPERATURE="${TEMPERATURE:-0}"
BLOCK_SIZE="${BLOCK_SIZE:-8}"
VERIFY_BLOCK="${VERIFY_BLOCK:-8}"
LONG_MAX_SAMPLES="${LONG_MAX_SAMPLES:-50}"

MODEL_NAMES=(
    "convk2g16"
    "noconv"
)
MODEL_PATHS=(
    "${PROJECT_DIR}/cache/models/flashmtp_h100_swa_w1_chs12_chsfirst_tokenwindow_globalpos_convk2g16_sample_q3_8b_pb_80k_wb_0.0_nlayers5_block_8_mhrnn_easy_direct_r512_ce0.1_tv1.0_maxlen4096_epochs6_Qwen3_8B/epoch_6_step_59730"
    "${PROJECT_DIR}/cache/models/flashmtp_h100_swa_w1_chs12_chsfirst_tokenwindow_globalpos_noconv_sample_q3_8b_pb_80k_wb_0.0_nlayers5_block_8_mhrnn_easy_direct_r512_ce0.1_tv1.0_maxlen4096_epochs6_Qwen3_8B/epoch_6_step_59730"
)

SHORT_DATASETS=(
    "gsm8k"
    "math500"
    "aime25"
    "humaneval"
    "mbpp"
    "livecodebench"
    "mt-bench"
    "alpaca"
)
LONG_DATASETS=(
    "longbench_v2_64000_32000_single_document_qa"
    "longbench_v2_64000_32000_multi_document_qa"
    "longbench_v2_64000_32000_long_dialogue"
    "longbench_v2_64000_32000_structured_data"
    "longbench_v2_64000_32000_in_context_learning"
    "longbench_v2_64000_32000_code_repo"
)
DATASETS=("${SHORT_DATASETS[@]}" "${LONG_DATASETS[@]}")

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [ "${#GPUS[@]}" -lt 1 ]; then
    echo "GPU_LIST must contain at least one GPU" >&2
    exit 2
fi

for required in "${PYTHON_BIN}" "${TARGET_MODEL}" "${MODEL_PATHS[@]}"; do
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
for gpu in "${GPUS[@]}"; do
    if [ -z "${GPU_MEMORY[${gpu}]+x}" ] || \
       [ "${GPU_MEMORY[${gpu}]:-999999}" -gt 1024 ] || \
       [ "${GPU_UTILIZATION[${gpu}]:-999}" -gt 10 ]; then
        echo "GPU ${gpu} is unavailable or busy" >&2
        exit 3
    fi
done

is_short_dataset() {
    local query="$1"
    local item
    for item in "${SHORT_DATASETS[@]}"; do
        if [ "${query}" = "${item}" ]; then
            return 0
        fi
    done
    return 1
}

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/status" "${RUN_ROOT}/workers"
{
    printf 'target_model=%s\n' "${TARGET_MODEL}"
    printf 'gpu_list=%s\n' "${GPU_LIST}"
    printf 'max_new_tokens=%s\n' "${MAX_NEW_TOKENS}"
    printf 'short_samples=benchmark defaults\n'
    printf 'long_max_samples=%s\n' "${LONG_MAX_SAMPLES}"
    printf 'batch_size=%s\n' "${BATCH_SIZE}"
    printf 'block_size=%s\n' "${BLOCK_SIZE}"
    printf 'verify_block=%s\n' "${VERIFY_BLOCK}"
    printf 'temperature=%s\n' "${TEMPERATURE}"
    printf 'verification_mode=match\n'
    printf 'compile_serial_head=false\n'
    printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
} > "${RUN_ROOT}/run_config.txt"

{
    printf 'model\ttemperature\tverification\tdataset\trequested_samples\tgpu\tdraft_path\tlog_path\tstatus_path\n'
    task_index=0
    for model_index in "${!MODEL_NAMES[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            gpu="${GPUS[$((task_index % ${#GPUS[@]}))]}"
            requested_samples="${LONG_MAX_SAMPLES}"
            if is_short_dataset "${dataset}"; then
                requested_samples="auto"
            fi
            model="${MODEL_NAMES[$model_index]}"
            draft_path="${MODEL_PATHS[$model_index]}"
            log_path="${RUN_ROOT}/logs/${model}/${dataset}.log"
            status_path="${RUN_ROOT}/status/${model}/${dataset}.status"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${model}" "${TEMPERATURE}" "match" "${dataset}" \
                "${requested_samples}" "${gpu}" "${draft_path}" \
                "${log_path}" "${status_path}"
            task_index=$((task_index + 1))
        done
    done
} > "${RUN_ROOT}/manifest.tsv"

run_worker() {
    local worker_gpu="$1"
    local worker_log="${RUN_ROOT}/workers/gpu_${worker_gpu}.log"
    local task_index=0
    local model_index dataset assigned_gpu model draft_path log_path status_path

    for model_index in "${!MODEL_NAMES[@]}"; do
        model="${MODEL_NAMES[$model_index]}"
        draft_path="${MODEL_PATHS[$model_index]}"
        for dataset in "${DATASETS[@]}"; do
            assigned_gpu="${GPUS[$((task_index % ${#GPUS[@]}))]}"
            task_index=$((task_index + 1))
            if [ "${assigned_gpu}" != "${worker_gpu}" ]; then
                continue
            fi

            log_path="${RUN_ROOT}/logs/${model}/${dataset}.log"
            status_path="${RUN_ROOT}/status/${model}/${dataset}.status"
            mkdir -p "$(dirname "${log_path}")" "$(dirname "${status_path}")"
            benchmark_args=(
                --model-name-or-path "${TARGET_MODEL}"
                --draft-name-or-path "${draft_path}"
                --max-new-tokens "${MAX_NEW_TOKENS}"
                --dataset "${dataset}"
                --batch-size "${BATCH_SIZE}"
                --block-size "${BLOCK_SIZE}"
                --verify-block "${VERIFY_BLOCK}"
                --temperature "${TEMPERATURE}"
            )
            if ! is_short_dataset "${dataset}"; then
                benchmark_args+=(--max-samples "${LONG_MAX_SAMPLES}")
            fi

            {
                printf '[START] %s gpu=%s model=%s dataset=%s\n' \
                    "$(date --iso-8601=seconds)" "${worker_gpu}" "${model}" "${dataset}"
                printf '[COMMAND] CUDA_VISIBLE_DEVICES=%s %q evaluation/benchmark.py ' \
                    "${worker_gpu}" "${PYTHON_BIN}"
                printf '%q ' "${benchmark_args[@]}"
                printf '\n'
            } > "${log_path}"
            printf 'running\nstarted_at=%s\n' "$(date --iso-8601=seconds)" > "${status_path}"
            printf '[START] %s model=%s dataset=%s\n' \
                "$(date --iso-8601=seconds)" "${model}" "${dataset}" >> "${worker_log}"

            CUDA_VISIBLE_DEVICES="${worker_gpu}" \
                NCCL_NVLS_ENABLE=0 PYTHONUNBUFFERED=1 NO_COLOR=1 COLUMNS=200 \
                "${PYTHON_BIN}" evaluation/benchmark.py \
                "${benchmark_args[@]}" >> "${log_path}" 2>&1
            exit_code=$?

            if [ "${exit_code}" -eq 0 ]; then
                printf 'completed\nfinished_at=%s\n' "$(date --iso-8601=seconds)" > "${status_path}"
                printf '[DONE] %s model=%s dataset=%s\n' \
                    "$(date --iso-8601=seconds)" "${model}" "${dataset}" >> "${worker_log}"
            else
                printf 'failed exit_code=%s\nfinished_at=%s\n' \
                    "${exit_code}" "$(date --iso-8601=seconds)" > "${status_path}"
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

"${PYTHON_BIN}" scripts/summarize_benchmarks.py "${RUN_ROOT}" \
    --verify-block "${VERIFY_BLOCK}" --per-run \
    > "${RUN_ROOT}/summary_generation.log" 2>&1 || overall_exit=1
printf 'finished_at=%s\noverall_exit=%s\n' \
    "$(date --iso-8601=seconds)" "${overall_exit}" >> "${RUN_ROOT}/run_config.txt"
printf 'RUN_ROOT=%s\n' "${RUN_ROOT}"
exit "${overall_exit}"
