#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT=/share/dai-sys/wanghanzhen/projects/MTP/FlashMTP_v2.3
DRAFT_PATH=/share/dai-sys/wanghanzhen/projects/MTP/FlashMTP_v2swa/cache/models/remote_teacher_v23_epoch5_step740000
TARGET_MODEL=/share/dai-sys/wanghanzhen/models/Qwen/Qwen3-8B
RUN_ROOT=${PROJECT_ROOT}/benchmark_results/teacher_epoch5_step740000_$(date +%Y%m%d_%H%M%S)

datasets=(
  alpaca
  gsm8k
  math500
  mbpp
  livecodebench
  humaneval
  mt-bench
  aime25
  longbench_v2_64000_32000_single_document_qa
  longbench_v2_64000_32000_multi_document_qa
  longbench_v2_64000_32000_long_dialogue
  longbench_v2_64000_32000_structured_data
  longbench_v2_64000_32000_in_context_learning
  longbench_v2_64000_32000_code_repo
)

mkdir -p "${RUN_ROOT}"
printf '%s\n' "${RUN_ROOT}" > "${PROJECT_ROOT}/benchmark_results/latest_teacher_step740k_run.txt"

run_dataset() {
  local gpu=$1
  local dataset=$2
  local log=${RUN_ROOT}/${dataset}.log
  echo "[$(date '+%F %T')] gpu=${gpu} dataset=${dataset} start" | tee -a "${RUN_ROOT}/status.log"
  (
    cd "${PROJECT_ROOT}"
    CUDA_VISIBLE_DEVICES=${gpu} \
    MASTER_PORT=$((29600 + gpu)) \
    NPROC_PER_NODE=1 \
    TARGET_MODEL="${TARGET_MODEL}" \
    DRAFT_NAME_OR_PATH="${DRAFT_PATH}" \
    DATASET="${dataset}" \
    MAX_SAMPLES=50 \
    MAX_NEW_TOKENS=512 \
    BATCH_SIZE=1 \
    TEMPERATURE=0 \
    bash evaluation/run_benchmark_flashmtp.sh --dt h100
  ) >"${log}" 2>&1
  local status=$?
  echo "[$(date '+%F %T')] gpu=${gpu} dataset=${dataset} exit=${status}" | tee -a "${RUN_ROOT}/status.log"
  return "${status}"
}

worker() {
  local gpu=$1
  local index
  local failed=0
  for ((index=gpu; index<${#datasets[@]}; index+=8)); do
    run_dataset "${gpu}" "${datasets[index]}" || failed=1
  done
  return "${failed}"
}

pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  worker "${gpu}" &
  pids+=("$!")
done

overall=0
for pid in "${pids[@]}"; do
  wait "${pid}" || overall=1
done

echo "[$(date '+%F %T')] all_done exit=${overall}" | tee -a "${RUN_ROOT}/status.log"
exit "${overall}"
