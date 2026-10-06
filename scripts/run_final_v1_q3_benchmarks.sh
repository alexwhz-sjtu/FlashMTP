#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-/data/wanghanzhen/FlashMTP_v2.3/.venv/bin/python}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/benchmark_results/final_v1_q3_$(date +%Y%m%d_%H%M%S)}"
TEMPERATURE=0
MAX_NEW_TOKENS=512
VERIFY_BLOCK=8

MODELS=(final_v1_q3_4b final_v1_q3_8b)
TARGETS=(/data/wanghanzhen/models/Qwen3-4B /data/wanghanzhen/models/Qwen3-8B)
DRAFTS=(
  "${PROJECT_ROOT}/cache/models/final_v1_q3_4b"
  "${PROJECT_ROOT}/cache/models/final_v1_q3_8b"
)
GPUS=(0 1)

DATASETS=(
  alpaca gsm8k math500 mbpp livecodebench humaneval mt-bench aime25
  longbench_v2_64000_32000_single_document_qa
  longbench_v2_64000_32000_multi_document_qa
  longbench_v2_64000_32000_long_dialogue
  longbench_v2_64000_32000_in_context_learning
  longbench_v2_64000_32000_code_repo
)
SAMPLES=(128 128 128 128 128 164 80 30 50 50 50 50 50)

if [[ ! -x "${PYTHON_BIN}" ]]; then
  printf 'Python environment not found: %s\n' "${PYTHON_BIN}" >&2
  exit 2
fi
for draft in "${DRAFTS[@]}"; do
  if [[ ! -s "${draft}/model.safetensors" ]] || find "${draft}" -maxdepth 1 -name '*.incomplete' -print -quit | grep -q .; then
    printf 'Draft checkpoint is incomplete: %s\n' "${draft}" >&2
    exit 2
  fi
done

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/status" "${RUN_ROOT}/workers"
{
  printf 'temperature=%s\n' "${TEMPERATURE}"
  printf 'max_new_tokens=%s\n' "${MAX_NEW_TOKENS}"
  printf 'verify_block=%s\n' "${VERIFY_BLOCK}"
  printf 'gpu_assignment=%s:%s,%s:%s\n' "${MODELS[0]}" "${GPUS[0]}" "${MODELS[1]}" "${GPUS[1]}"
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
} > "${RUN_ROOT}/run_config.txt"

{
  printf 'model\ttemperature\tdataset\trequested_samples\tgpu\tdraft_path\tlog_path\tstatus_path\n'
  for model_i in "${!MODELS[@]}"; do
    for data_i in "${!DATASETS[@]}"; do
      model="${MODELS[$model_i]}"
      dataset="${DATASETS[$data_i]}"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${model}" "${TEMPERATURE}" "${dataset}" "${SAMPLES[$data_i]}" \
        "${GPUS[$model_i]}" "${DRAFTS[$model_i]}" \
        "${RUN_ROOT}/logs/${model}/${dataset}.log" \
        "${RUN_ROOT}/status/${model}/${dataset}.status"
    done
  done
} > "${RUN_ROOT}/manifest.tsv"

run_model_worker() {
  local model_i="$1"
  local model="${MODELS[$model_i]}"
  local target="${TARGETS[$model_i]}"
  local draft="${DRAFTS[$model_i]}"
  local gpu="${GPUS[$model_i]}"
  local failed=0

  mkdir -p "${RUN_ROOT}/logs/${model}" "${RUN_ROOT}/status/${model}"
  for data_i in "${!DATASETS[@]}"; do
    local dataset="${DATASETS[$data_i]}"
    local samples="${SAMPLES[$data_i]}"
    local log="${RUN_ROOT}/logs/${model}/${dataset}.log"
    local status="${RUN_ROOT}/status/${model}/${dataset}.status"
    printf 'running\nstarted_at=%s\n' "$(date --iso-8601=seconds)" > "${status}"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 NO_COLOR=1 COLUMNS=200 \
      "${PYTHON_BIN}" evaluation/benchmark.py \
        --model-name-or-path "${target}" \
        --draft-name-or-path "${draft}" \
        --dataset "${dataset}" \
        --max-samples "${samples}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --batch-size 1 \
        --verify-block "${VERIFY_BLOCK}" \
        --temperature "${TEMPERATURE}" > "${log}" 2>&1
    local rc=$?
    if [[ ${rc} -eq 0 ]]; then
      printf 'completed\nfinished_at=%s\n' "$(date --iso-8601=seconds)" > "${status}"
    else
      printf 'failed exit_code=%s\nfinished_at=%s\n' "${rc}" "$(date --iso-8601=seconds)" > "${status}"
      failed=1
    fi
  done
  return "${failed}"
}

printf 'RUN_ROOT=%s\n' "${RUN_ROOT}"
pids=()
for model_i in "${!MODELS[@]}"; do
  run_model_worker "${model_i}" > "${RUN_ROOT}/workers/${MODELS[$model_i]}.log" 2>&1 &
  pids+=("$!")
done

overall=0
for pid in "${pids[@]}"; do
  wait "${pid}" || overall=1
done
printf 'finished_at=%s\noverall_exit=%s\n' "$(date --iso-8601=seconds)" "${overall}" >> "${RUN_ROOT}/run_config.txt"
printf 'RUN_ROOT=%s\n' "${RUN_ROOT}"
exit "${overall}"
