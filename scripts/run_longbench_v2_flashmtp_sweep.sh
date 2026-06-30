#!/usr/bin/env bash
# Run FlashMTP_v1.1 evaluation/benchmark.py on LongBench_v2 JSON shards (same paths as dflash),
# excluding the 64000–128000 context bucket (longbench_v2_128000_64000 only).
#
# GPU scheduling: same as dflash — one synchronous job per GPU, round-robin partition, flock per GPU.
#
# Usage:
#   bash scripts/run_longbench_v2_flashmtp_sweep.sh
#
# Env:
#   WHZ_DIR             (default: /data/wanghanzhen)
#   SAMPLES             (default: 50)
#   DT                  (default: qz) — tag for log naming only
#   MAX_PARALLEL_GPUS   (default: length of GPU_IDS)
#   GPU_IDS             (default: "4 5 6 7") — physical GPU indices, one worker each
#   FLASHMTP_ROOT       (default: parent of scripts/)
#   RUN_ID              (optional; default: timestamp) — log subdir lb2_flashmtp_sweep_${RUN_ID}
#   GPU_FLOCK_DIR       (default: /tmp/flashmtp_longbench_v2_gpu_flocks)
#   DRAFT_PATH          (default: cache/models/flashmtp_qz_prefix_condition_fuse_middle_16_feature_sample_900000_think_off_nlayers5_block_16_gamma_7_maxlen4096_epochs8_tlmh0_lp1 under FLASHMTP_ROOT)
#   PYTHON_BIN          (default: FLASHMTP_ROOT/.venv/bin/python)

set -euo pipefail

FLASHMTP_ROOT="${FLASHMTP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${FLASHMTP_ROOT}"


# 自动激活虚拟环境
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
if [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
    source "${PROJECT_DIR}/.venv/bin/activate"
fi

# Used by worker subshells; must be set before any reference (script uses set -u).
PYTHON_BIN="${PYTHON_BIN:-${FLASHMTP_ROOT}/.venv/bin/python}"

WHZ_DIR="${WHZ_DIR:-/data/wanghanzhen}"
SAMPLES="${SAMPLES:-50}"
DT="${DT:-qz}"
GPU_IDS="${GPU_IDS:-4 5 6 7}"
GPU_FLOCK_DIR="${GPU_FLOCK_DIR:-/tmp/flashmtp_longbench_v2_gpu_flocks}"
DRAFT_PATH="${DRAFT_PATH:-${FLASHMTP_ROOT}/cache/models/flashmtp_qz_prefix_condition_fuse_middle_16_feature_sample_900000_think_off_nlayers5_block_16_gamma_7_maxlen4096_epochs8_tlmh0_lp1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
MAX_PARALLEL_GPUS="${MAX_PARALLEL_GPUS:-${#GPU_ID_ARRAY[@]}}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOGDIR="${LOGDIR:-${FLASHMTP_ROOT}/log/lb2_flashmtp_sweep_${RUN_ID}}"
mkdir -p "${LOGDIR}"
mkdir -p "${GPU_FLOCK_DIR}"

sanitize() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_'
}

# Same base paths as dflash/scripts/run_longbench_v2_all_shards.sh;
# exclude only the 64000–128000 context bucket (longbench_v2_128000_64000).
ALL_SHARDS=(
  "/data/wanghanzhen/datasets/LongBench_v2/longbench_v2_40000_20480"
  "/data/wanghanzhen/datasets/LongBench_v2/longbench_v2_64000_16384"
  "/data/wanghanzhen/datasets/LongBench_v2/longbench_v2_64000_32000"
  "/data/wanghanzhen/datasets/LongBench_v2/longbench_v2_128000_64000"
)
SHARDS=()
for shard in "${ALL_SHARDS[@]}"; do
  base=$(basename "${shard}")
  if [[ "${base}" == "longbench_v2_128000_64000" ]]; then
    echo "SKIP shard (64000–128000 context bucket): ${shard}"
    continue
  fi
  SHARDS+=("${shard}")
done

MASTER_LOG="${LOGDIR}/_master.log"
RESULTS_TSV="${LOGDIR}/_results.tsv"

ngpu_detected="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
if [[ -z "${ngpu_detected}" || "${ngpu_detected}" -lt 1 ]]; then
  ngpu_detected=1
fi
if [[ "${MAX_PARALLEL_GPUS}" -lt 1 ]]; then
  MAX_PARALLEL_GPUS=1
fi
if [[ "${MAX_PARALLEL_GPUS}" -gt "${ngpu_detected}" ]]; then
  echo "Clamping MAX_PARALLEL_GPUS from ${MAX_PARALLEL_GPUS} to ${ngpu_detected} (detected GPUs)" | tee -a "${MASTER_LOG}"
  MAX_PARALLEL_GPUS="${ngpu_detected}"
fi

{
  echo "=== LongBench v2 FlashMTP v1.1 sweep ==="
  echo "RUN_ID=${RUN_ID}"
  echo "LOGDIR=${LOGDIR}"
  echo "WHZ_DIR=${WHZ_DIR}"
  echo "SAMPLES=${SAMPLES} DT=${DT} MAX_PARALLEL_GPUS=${MAX_PARALLEL_GPUS} GPU_IDS=${GPU_IDS}"
  echo "GPU_FLOCK_DIR=${GPU_FLOCK_DIR}"
  echo "PYTHON_BIN=${PYTHON_BIN}"
  echo "DRAFT_PATH=${DRAFT_PATH}"
  echo "Detected GPUs: ${ngpu_detected}"
  echo "Started at $(date -Is)"
} | tee -a "${MASTER_LOG}"

echo -e "exit_code\tgpu\tshard\tjson\tlogfile" > "${RESULTS_TSV}"

JOBS=()
for shard in "${SHARDS[@]}"; do
  if [[ ! -d "${shard}" ]]; then
    echo "MISSING SHARD DIR: ${shard}" | tee -a "${MASTER_LOG}"
    exit 1
  fi
  shard_base=$(basename "${shard}")
  shopt -s nullglob
  files=("${shard}"/*.json)
  shopt -u nullglob
  if [[ "${#files[@]}" -eq 0 ]]; then
    echo "No JSON under ${shard}" | tee -a "${MASTER_LOG}"
    exit 1
  fi
  for f in "${files[@]}"; do
    n_samples="$("${PYTHON_BIN}" -c "import json; print(len(json.load(open('${f}'))))")"
    if [[ "${n_samples}" -eq 0 ]]; then
      echo "SKIP empty dataset (${n_samples} samples): ${f}" | tee -a "${MASTER_LOG}"
      continue
    fi
    JOBS+=("${shard_base}|${f}")
  done
done

echo "Total jobs: ${#JOBS[@]}" | tee -a "${MASTER_LOG}"

run_worker() {
  local worker_idx="$1"
  local gpu="${GPU_ID_ARRAY[$worker_idx]}"
  local lockfile="${GPU_FLOCK_DIR}/cuda${gpu}.lock"
  touch "${lockfile}" 2>/dev/null || true

  local i entry shard_base f base slug logfile ec
  for ((i = worker_idx; i < ${#JOBS[@]}; i += MAX_PARALLEL_GPUS)); do
    entry="${JOBS[$i]}"
    shard_base="${entry%%|*}"
    f="${entry#*|}"
    base=$(basename "$f" .json)
    slug=$(sanitize "${base}")
    logfile="${LOGDIR}/${shard_base}__${slug}_flashmtp_${DT}_${SAMPLES}.log"
    if [[ -f "${logfile}" ]]; then
      echo "[GPU ${gpu}] SKIP existing log: ${logfile}" | tee -a "${MASTER_LOG}"
      continue
    fi
    echo "[GPU ${gpu}] START (${i}/${#JOBS[@]}) ${f}" | tee -a "${MASTER_LOG}"

    set +e
    {
      echo "---- $(date -Is) START ${f} ----"
      flock "${lockfile}" bash -c "
        export CUDA_VISIBLE_DEVICES=\"${gpu}\"
        exec \"${PYTHON_BIN}\" evaluation/benchmark.py \
          --model-name-or-path \"${WHZ_DIR}/models/Qwen/Qwen3-8B\" \
          --draft-name-or-path \"${DRAFT_PATH}\" \
          --dataset \"${f}\" \
          --max-samples \"${SAMPLES}\"
      "
      ec=$?
      echo "---- $(date -Is) END exit=${ec} ----"
    } >>"${logfile}" 2>&1
    set -e

    echo -e "${ec}\t${gpu}\t${shard_base}\t${f}\t${logfile}" >> "${RESULTS_TSV}"
    echo "[GPU ${gpu}] END exit=${ec} log=${logfile}" | tee -a "${MASTER_LOG}"
  done
}

pids=()
for ((w = 0; w < MAX_PARALLEL_GPUS; w++)); do
  if ((w < ${#JOBS[@]})); then
    run_worker "${w}" &
    pids+=("$!")
  fi
done

ec_all=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    ec_all=1
  fi
done

{
  echo "Finished at $(date -Is)"
  echo "=== summary lines from per-job logs ==="
  shopt -s nullglob
  for lf in "${LOGDIR}"/*__*_flashmtp_"${DT}"_"${SAMPLES}".log; do
    grep -H "Decoding speedup\|Average Acceptance length:\|Total elapsed time" "${lf}" 2>/dev/null || true
  done
  shopt -u nullglob
} | tee -a "${MASTER_LOG}"

if [[ "${ec_all}" -ne 0 ]]; then
  echo "One or more workers exited non-zero. See ${RESULTS_TSV}" | tee -a "${MASTER_LOG}"
  echo "FAILED_WORKERS ${ec_all} $(date -Is)" > "${LOGDIR}/_SWEEP_DONE"
  exit "${ec_all}"
fi

echo "Sweep complete. LOGDIR=${LOGDIR}" | tee -a "${MASTER_LOG}"
echo "OK $(date -Is)" > "${LOGDIR}/_SWEEP_DONE"
