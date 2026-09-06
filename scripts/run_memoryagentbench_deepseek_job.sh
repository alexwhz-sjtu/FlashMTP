#!/usr/bin/env bash
set -euo pipefail

if (( $# != 5 )); then
  echo "usage: $0 BACKEND CATEGORY GPU PORT RUN_ROOT" >&2
  exit 2
fi

BACKEND="$1"
CATEGORY="$2"
GPU="$3"
PORT="$4"
RUN_ROOT="$5"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MTP_ROOT="$(dirname "${PROJECT_DIR}")"
DEEPSPEC_PYTHON="${DEEPSPEC_PYTHON:-/share/dai-sys/wanghanzhen/envs/distspec/bin/python}"
CLIENT_PYTHON="${PROJECT_DIR}/.venv/bin/python"
TARGET_MODEL="${TARGET_MODEL:-/share/dai-sys/wanghanzhen/models/Qwen/Qwen3-4B}"
if [[ "${BACKEND}" == "dflash" ]]; then
  DRAFT_MODEL="${DFLASH_DRAFT:-/share/dai-sys/wanghanzhen/models/deepseek-ai/dflash_qwen3_4b_block7}"
elif [[ "${BACKEND}" == "dspark" ]]; then
  DRAFT_MODEL="${DSPARK_DRAFT:-/share/dai-sys/wanghanzhen/models/deepseek-ai/dspark_qwen3_4b_block7}"
else
  echo "backend must be dflash or dspark" >&2
  exit 2
fi

JOB_DIR="${RUN_ROOT}/${BACKEND}/${CATEGORY}"
mkdir -p "${JOB_DIR}"
METRICS="${JOB_DIR}/metrics.jsonl"
RESPONSES="${JOB_DIR}/responses.jsonl"
SERVER_LOG="${JOB_DIR}/server.log"
CLIENT_LOG="${JOB_DIR}/client.log"
STATUS="${JOB_DIR}/status.txt"
rm -f "${METRICS}" "${RESPONSES}"
printf 'running\nstarted_at=%s\ngpu=%s\nport=%s\n' \
  "$(date --iso-8601=seconds)" "${GPU}" "${PORT}" > "${STATUS}"

server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

cd "${MTP_ROOT}"
CUDA_VISIBLE_DEVICES="${GPU}" \
PYTHONPATH="${MTP_ROOT}/dflash_eval:${MTP_ROOT}/DeepSpec" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${DEEPSPEC_PYTHON}" agent_long_context_server.py \
  --backend "${BACKEND}" \
  --device cuda:0 \
  --port "${PORT}" \
  --target-path "${TARGET_MODEL}" \
  --draft-path "${DRAFT_MODEL}" \
  --served-model-name "${BACKEND}-qwen3-4b-deepseek" \
  --metrics-jsonl "${METRICS}" \
  --context-limit 163840 \
  --max-output-tokens "${MAX_NEW_TOKENS:-512}" \
  --rope-factor 4 \
  --original-max-position-embeddings 40960 \
  > "${SERVER_LOG}" 2>&1 &
server_pid=$!

ready=0
for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null; then
    ready=1
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    break
  fi
  sleep 2
done
if (( ready == 0 )); then
  printf 'failed_server_start\nfinished_at=%s\n' "$(date --iso-8601=seconds)" >> "${STATUS}"
  exit 1
fi

cd "${PROJECT_DIR}"
client_args=(
  evaluation/memoryagentbench_deepseek_client.py
  --category "${CATEGORY}"
  --server-url "http://127.0.0.1:${PORT}"
  --tokenizer-path "${TARGET_MODEL}"
  --model "${BACKEND}-qwen3-4b-deepseek"
  --output-jsonl "${RESPONSES}"
  --max-new-tokens "${MAX_NEW_TOKENS:-512}"
  --temperature 0
)
if [[ -n "${REQUEST_LIMIT:-}" ]]; then
  client_args+=(--request-limit "${REQUEST_LIMIT}")
fi
PYTHONUNBUFFERED=1 "${CLIENT_PYTHON}" "${client_args[@]}" > "${CLIENT_LOG}" 2>&1

printf 'completed\nfinished_at=%s\n' "$(date --iso-8601=seconds)" >> "${STATUS}"
