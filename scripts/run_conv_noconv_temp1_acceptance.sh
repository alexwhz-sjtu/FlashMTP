#!/bin/bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
TARGET_MODEL="${TARGET_MODEL:-/data/wanghanzhen/models/Qwen3-8B}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/benchmark_results/conv_noconv_temp1_acceptance_$(date +%Y%m%d_%H%M%S)}"
GPU_LIST="${GPU_LIST:-0,2,5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
BLOCK_SIZE="${BLOCK_SIZE:-8}"
VERIFY_BLOCK="${VERIFY_BLOCK:-8}"

MODEL_NAMES=("convk2g16" "noconv")
MODEL_PATHS=(
    "${PROJECT_DIR}/cache/models/flashmtp_h100_swa_w1_chs12_chsfirst_tokenwindow_globalpos_convk2g16_sample_q3_8b_pb_80k_wb_0.0_nlayers5_block_8_mhrnn_easy_direct_r512_ce0.1_tv1.0_maxlen4096_epochs6_Qwen3_8B/epoch_6_step_59730"
    "${PROJECT_DIR}/cache/models/flashmtp_h100_swa_w1_chs12_chsfirst_tokenwindow_globalpos_noconv_sample_q3_8b_pb_80k_wb_0.0_nlayers5_block_8_mhrnn_easy_direct_r512_ce0.1_tv1.0_maxlen4096_epochs6_Qwen3_8B/epoch_6_step_59730"
)
DATASETS=("alpaca" "gsm8k" "mbpp")

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [ "${#GPUS[@]}" -ne "${#DATASETS[@]}" ]; then
    echo "GPU_LIST must contain exactly ${#DATASETS[@]} GPUs" >&2
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

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/status" "${RUN_ROOT}/workers"
{
    printf 'target_model=%s\n' "${TARGET_MODEL}"
    printf 'gpu_list=%s\n' "${GPU_LIST}"
    printf 'max_new_tokens=%s\n' "${MAX_NEW_TOKENS}"
    printf 'samples=benchmark defaults\n'
    printf 'batch_size=1\nblock_size=%s\nverify_block=%s\n' "${BLOCK_SIZE}" "${VERIFY_BLOCK}"
    printf 'temperature=1\nverification_mode=rejection\ncompile_serial_head=false\n'
    printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
} > "${RUN_ROOT}/run_config.txt"

{
    printf 'model\ttemperature\tverification\tdataset\trequested_samples\tgpu\tdraft_path\tlog_path\tstatus_path\n'
    for dataset_index in "${!DATASETS[@]}"; do
        dataset="${DATASETS[$dataset_index]}"
        gpu="${GPUS[$dataset_index]}"
        for model_index in "${!MODEL_NAMES[@]}"; do
            model="${MODEL_NAMES[$model_index]}"
            draft_path="${MODEL_PATHS[$model_index]}"
            printf '%s\t1\trejection\t%s\tauto\t%s\t%s\t%s\t%s\n' \
                "${model}" "${dataset}" "${gpu}" "${draft_path}" \
                "${RUN_ROOT}/logs/${model}/${dataset}.log" \
                "${RUN_ROOT}/status/${model}/${dataset}.status"
        done
    done
} > "${RUN_ROOT}/manifest.tsv"

run_dataset_pair() {
    local dataset_index="$1"
    local dataset="${DATASETS[$dataset_index]}"
    local gpu="${GPUS[$dataset_index]}"
    local worker_log="${RUN_ROOT}/workers/gpu_${gpu}.log"
    local model_index model draft_path log_path status_path exit_code

    for model_index in "${!MODEL_NAMES[@]}"; do
        model="${MODEL_NAMES[$model_index]}"
        draft_path="${MODEL_PATHS[$model_index]}"
        log_path="${RUN_ROOT}/logs/${model}/${dataset}.log"
        status_path="${RUN_ROOT}/status/${model}/${dataset}.status"
        mkdir -p "$(dirname "${log_path}")" "$(dirname "${status_path}")"
        printf 'running\nstarted_at=%s\n' "$(date --iso-8601=seconds)" > "${status_path}"
        printf '[START] %s model=%s dataset=%s gpu=%s\n' \
            "$(date --iso-8601=seconds)" "${model}" "${dataset}" "${gpu}" >> "${worker_log}"
        {
            printf '[START] %s gpu=%s model=%s dataset=%s temperature=1 verification=rejection\n' \
                "$(date --iso-8601=seconds)" "${gpu}" "${model}" "${dataset}"
            CUDA_VISIBLE_DEVICES="${gpu}" NCCL_NVLS_ENABLE=0 \
                PYTHONUNBUFFERED=1 NO_COLOR=1 COLUMNS=200 \
                "${PYTHON_BIN}" evaluation/benchmark.py \
                --model-name-or-path "${TARGET_MODEL}" \
                --draft-name-or-path "${draft_path}" \
                --max-new-tokens "${MAX_NEW_TOKENS}" \
                --dataset "${dataset}" \
                --batch-size 1 \
                --block-size "${BLOCK_SIZE}" \
                --verify-block "${VERIFY_BLOCK}" \
                --temperature 1 \
                --stochastic-verification-mode rejection
        } > "${log_path}" 2>&1
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
}

printf 'RUN_ROOT=%s\n' "${RUN_ROOT}"
pids=()
for dataset_index in "${!DATASETS[@]}"; do
    run_dataset_pair "${dataset_index}" &
    pids+=("$!")
done
overall_exit=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
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
