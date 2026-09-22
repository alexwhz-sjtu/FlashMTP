#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
TARGET_MODEL="${TARGET_MODEL:-/data/wanghanzhen/models/Qwen3-8B}"
DRAFT_PATH="${DRAFT_PATH:-${PROJECT_ROOT}/cache/models/Flashmtp_v2.3_new_qwen3_8b_teacher_d10_swa512}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/benchmark_results/Flashmtp_v2.3_new_qwen3_8b_teacher_d10_swa512_$(date +%Y%m%d_%H%M%S)}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
VERIFY_BLOCK="${VERIFY_BLOCK:-8}"
TEMPERATURE="${TEMPERATURE:-0}"

DATASETS=(
  alpaca gsm8k math500 mbpp livecodebench humaneval mt-bench aime25
  longbench_v2_64000_32000_single_document_qa
  longbench_v2_64000_32000_multi_document_qa
  longbench_v2_64000_32000_long_dialogue
  longbench_v2_64000_32000_structured_data
  longbench_v2_64000_32000_in_context_learning
  longbench_v2_64000_32000_code_repo
)
SAMPLES=(128 128 128 128 128 164 80 30 50 50 50 50 50 50)

if [[ ! -x "${PYTHON_BIN}" || ! -d "${TARGET_MODEL}" || ! -d "${DRAFT_PATH}" ]]; then
  echo "Missing Python, target model, or draft checkpoint" >&2
  exit 2
fi

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/status" "${RUN_ROOT}/workers"
{
  printf 'target_model=%s\n' "${TARGET_MODEL}"
  printf 'draft_path=%s\n' "${DRAFT_PATH}"
  printf 'gpu_list=%s\n' "${GPU_LIST}"
  printf 'max_new_tokens=%s\n' "${MAX_NEW_TOKENS}"
  printf 'verify_block=%s\n' "${VERIFY_BLOCK}"
  printf 'temperature=%s\n' "${TEMPERATURE}"
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
} > "${RUN_ROOT}/run_config.txt"
{
  printf 'model\ttemperature\tdataset\trequested_samples\tgpu\tdraft_path\tlog_path\tstatus_path\n'
  for i in "${!DATASETS[@]}"; do
    gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
    dataset="${DATASETS[$i]}"
    printf 'Flashmtp_v2.3_new_qwen3_8b_teacher_d10_swa512\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${TEMPERATURE}" "${dataset}" "${SAMPLES[$i]}" "${gpu}" "${DRAFT_PATH}" \
      "${RUN_ROOT}/logs/${dataset}.log" "${RUN_ROOT}/status/${dataset}.status"
  done
} > "${RUN_ROOT}/manifest.tsv"

run_one() {
  local gpu="$1" dataset="$2" samples="$3" log="$4" status="$5"
  printf 'running\nstarted_at=%s\n' "$(date --iso-8601=seconds)" > "${status}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 NO_COLOR=1 COLUMNS=200 \
    "${PYTHON_BIN}" evaluation/benchmark.py \
      --model-name-or-path "${TARGET_MODEL}" \
      --draft-name-or-path "${DRAFT_PATH}" \
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
  fi
  return ${rc}
}

worker() {
  local gpu="$1" failed=0
  for i in "${!DATASETS[@]}"; do
    [[ "${GPUS[$((i % ${#GPUS[@]}))]}" == "${gpu}" ]] || continue
    run_one "${gpu}" "${DATASETS[$i]}" "${SAMPLES[$i]}" \
      "${RUN_ROOT}/logs/${DATASETS[$i]}.log" "${RUN_ROOT}/status/${DATASETS[$i]}.status" || failed=1
  done
  return ${failed}
}

pids=()
for gpu in "${GPUS[@]}"; do worker "${gpu}" > "${RUN_ROOT}/workers/gpu_${gpu}.log" 2>&1 & pids+=("$!"); done
overall=0
for pid in "${pids[@]}"; do wait "${pid}" || overall=1; done

"${PYTHON_BIN}" scripts/summarize_benchmarks.py "${RUN_ROOT}" --verify-block "${VERIFY_BLOCK}" --per-run \
  > "${RUN_ROOT}/summary_generation.log" 2>&1 || overall=1
printf 'finished_at=%s\noverall_exit=%s\n' "$(date --iso-8601=seconds)" "${overall}" >> "${RUN_ROOT}/run_config.txt"
echo "RUN_ROOT=${RUN_ROOT}"
exit "${overall}"
