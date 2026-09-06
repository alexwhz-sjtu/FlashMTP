#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEEPSPEC_ROOT="${DEEPSPEC_ROOT:-/share/dai-sys/wanghanzhen/projects/MTP/DeepSpec}"
MTP_ROOT="${MTP_ROOT:-/share/dai-sys/wanghanzhen/projects/MTP}"
TARGET_MODEL="${TARGET_MODEL:-/share/dai-sys/wanghanzhen/models/Qwen/Qwen3-4B}"
FLASH_DRAFT="${FLASH_DRAFT:-${PROJECT_DIR}/cache/models/flashmtp_v2swa_w5_qwen3_4b_ep10}"
DFLASH_DRAFT="${DFLASH_DRAFT:-/share/dai-sys/wanghanzhen/models/deepseek-ai/dflash_qwen3_4b_block7}"
DSPARK_DRAFT="${DSPARK_DRAFT:-/share/dai-sys/wanghanzhen/models/deepseek-ai/dspark_qwen3_4b_block7}"
DEEPSPEC_PYTHON="${DEEPSPEC_PYTHON:-/share/dai-sys/wanghanzhen/envs/distspec/bin/python}"
DATASET="${DATASET:-${PROJECT_DIR}/benchmark_results/agentlongbench_48k_96k/selected_50.jsonl}"
DATASET_PATH_FILE="${DATASET_PATH_FILE:-${PROJECT_DIR}/benchmark_results/agentlongbench_48k_96k/dataset_path.json}"
RUN_DIR="${RUN_DIR:-${PROJECT_DIR}/benchmark_results/agentlongbench_48k_96k/run_$(date +%Y%m%d_%H%M%S)}"
MAX_SAMPLES="${MAX_SAMPLES:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
FLASH_GPU="${FLASH_GPU:-0}"
DFLASH_GPU="${DFLASH_GPU:-1}"
DSPARK_GPU="${DSPARK_GPU:-2}"
DFLASH_PORT="${DFLASH_PORT:-18011}"

mkdir -p "${RUN_DIR}"
printf '%s\n' "${RUN_DIR}" > "${PROJECT_DIR}/benchmark_results/agentlongbench_48k_96k/latest_run.txt"

run_flashmtp() {
  cd "${PROJECT_DIR}"
  CUDA_VISIBLE_DEVICES="${FLASH_GPU}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PROJECT_DIR}/.venv/bin/python" -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port 29601 \
    evaluation/benchmark.py \
    --model-name-or-path "${TARGET_MODEL}" \
    --draft-name-or-path "${FLASH_DRAFT}" \
    --dataset "${DATASET}" \
    --max-samples "${MAX_SAMPLES}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature 0
}

run_deepspec() {
  local gpu="$1"
  local draft="$2"
  local output_json="$3"
  local master_port="$4"
  cd "${DEEPSPEC_ROOT}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  MASTER_ADDR=127.0.0.1 \
  MASTER_PORT="${master_port}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${DEEPSPEC_PYTHON}" benchmark.py \
    --model-name-or-path "${TARGET_MODEL}" \
    --draft-name-or-path "${draft}" \
    --dataset-path-file "${DATASET_PATH_FILE}" \
    --dataset agentlongbench_48k_96k \
    --max-samples "${MAX_SAMPLES}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature 0 \
    --no-measure-speedup \
    --output-json "${output_json}"
}

run_dflash() {
  local metrics_path="${RUN_DIR}/dflash_qwen3_4b_metrics.jsonl"
  local responses_path="${RUN_DIR}/dflash_qwen3_4b_responses.jsonl"
  local server_log="${RUN_DIR}/dflash_server.log"
  local server_pid
  cd "${MTP_ROOT}"
  CUDA_VISIBLE_DEVICES="${DFLASH_GPU}" \
  PYTHONPATH="${MTP_ROOT}/dflash_eval" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${DEEPSPEC_PYTHON}" agent_long_context_server.py \
    --backend dflash \
    --device cuda:0 \
    --port "${DFLASH_PORT}" \
    --draft-path "${DFLASH_DRAFT}" \
    --metrics-jsonl "${metrics_path}" \
    --max-output-tokens "${MAX_NEW_TOKENS}" \
    --rope-factor 1 \
    --original-max-position-embeddings 40960 \
    > "${server_log}" 2>&1 &
  server_pid=$!

  local ready=0
  for _ in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:${DFLASH_PORT}/health" >/dev/null; then
      ready=1
      break
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      break
    fi
    sleep 2
  done
  if (( ready == 0 )); then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
    return 1
  fi

  cd "${PROJECT_DIR}"
  "${PROJECT_DIR}/.venv/bin/python" scripts/run_raw_prompt_acceptance_client.py \
    --server-url "http://127.0.0.1:${DFLASH_PORT}" \
    --dataset "${DATASET}" \
    --output "${responses_path}" \
    --run-id agentlongbench-48k96k-dflash \
    --model dflash-qwen3-4b \
    --max-samples "${MAX_SAMPLES}" \
    --max-tokens "${MAX_NEW_TOKENS}"
  local client_status=$?
  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
  return "${client_status}"
}

set +e
run_flashmtp > "${RUN_DIR}/flashmtp_v2swa.log" 2>&1 &
flash_pid=$!
run_dflash > "${RUN_DIR}/dflash_qwen3_4b.log" 2>&1 &
dflash_pid=$!
run_deepspec "${DSPARK_GPU}" "${DSPARK_DRAFT}" \
  "${RUN_DIR}/dspark_qwen3_4b.json" 29603 \
  > "${RUN_DIR}/dspark_qwen3_4b.log" 2>&1 &
dspark_pid=$!

wait "${flash_pid}"
flash_status=$?
wait "${dflash_pid}"
dflash_status=$?
wait "${dspark_pid}"
dspark_status=$?
set -e

printf 'flashmtp_v2swa\t%s\n' "${flash_status}" > "${RUN_DIR}/status.tsv"
printf 'dflash_qwen3_4b\t%s\n' "${dflash_status}" >> "${RUN_DIR}/status.tsv"
printf 'dspark_qwen3_4b\t%s\n' "${dspark_status}" >> "${RUN_DIR}/status.tsv"
printf 'run_dir=%s\n' "${RUN_DIR}"
cat "${RUN_DIR}/status.tsv"

if (( flash_status != 0 || dflash_status != 0 || dspark_status != 0 )); then
  exit 1
fi

"${PROJECT_DIR}/.venv/bin/python" \
  "${PROJECT_DIR}/scripts/summarize_agentlongbench_acceptance.py" \
  --run-dir "${RUN_DIR}" \
  --manifest "${DATASET}"
