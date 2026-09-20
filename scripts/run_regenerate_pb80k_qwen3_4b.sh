#!/usr/bin/env bash
# Regenerate open_perfectblend_80k prompts with Qwen3-4B.
# Keeps the source JSONL schema: id / conversations / source / status.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/wanghanzhen/FlashMTP_v2swa/.venv/bin/python}"
VENV_BIN="$(dirname "${PYTHON_BIN}")"
GENERATOR_SCRIPT="${GENERATOR_SCRIPT:-/data/wanghanzhen/qwen38_remote/regenerate_train_data.py}"
MODEL_PATH="${MODEL_PATH:-/data/wanghanzhen/models/Qwen3-4B}"
INPUT_FILE="${INPUT_FILE:-/data/wanghanzhen/training_data/generated/qwen3-8b/open_perfectblend_80k_qwen3_8b.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-/data/wanghanzhen/training_data/generated/qwen3-4b/open_perfectblend_80k_qwen3_4b.jsonl}"
LOG_DIR="${LOG_DIR:-/data/wanghanzhen/training_data/generated/qwen3-4b/logs/open_perfectblend_80k_temp0}"
GPUS="${GPUS:-1,2,3,4,5,6,7}"
BASE_PORT="${BASE_PORT:-32100}"
CONCURRENCY="${CONCURRENCY:-48}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-16384}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-12288}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_RETRY_PASSES="${MAX_RETRY_PASSES:-3}"

export PATH="${VENV_BIN}:${PATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"

mkdir -p "$(dirname "${OUTPUT_FILE}")" "${LOG_DIR}"
printf '%s\n' "$$" >"${LOG_DIR}/orchestrator.pid"
printf 'starting\n' >"${LOG_DIR}/status"

IFS=',' read -r -a GPU_LIST <<<"${GPUS}"
server_pids=()
server_addresses=()

cleanup() {
  local pid
  for pid in "${server_pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  for pid in "${server_pids[@]:-}"; do
    wait "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

if [ ! -s "${INPUT_FILE}" ]; then
  printf 'failed: input is missing or empty: %s\n' "${INPUT_FILE}" | tee "${LOG_DIR}/status"
  exit 1
fi

for i in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[${i}]}"
  port=$((BASE_PORT + gpu))
  if ss -ltn "sport = :${port}" | grep -q LISTEN; then
    printf 'failed: port %s is already in use\n' "${port}" | tee "${LOG_DIR}/status"
    exit 1
  fi
  address="127.0.0.1:${port}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host 127.0.0.1 \
    --port "${port}" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tp-size 1 \
    --attention-backend fa3 \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --context-length "${CONTEXT_LENGTH}" \
    --max-running-requests "${CONCURRENCY}" \
    --cuda-graph-max-bs "${CONCURRENCY}" \
    --disable-radix-cache \
    >"${LOG_DIR}/server_gpu${gpu}.log" 2>&1 &
  pid=$!
  server_pids+=("${pid}")
  server_addresses+=("${address}")

  ready=0
  for _ in $(seq 1 180); do
    if curl -fsS "http://${address}/health" >/dev/null 2>&1 || \
       curl -fsS "http://${address}/model_info" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      printf 'failed: SGLang server GPU=%s address=%s\n' "${gpu}" "${address}" | tee "${LOG_DIR}/status"
      tail -100 "${LOG_DIR}/server_gpu${gpu}.log" >&2
      exit 1
    fi
    sleep 2
  done
  if [ "${ready}" -ne 1 ]; then
    printf 'failed: timed out waiting for %s\n' "${address}" | tee "${LOG_DIR}/status"
    exit 1
  fi
  printf 'SGLang server ready: GPU=%s address=%s\n' "${gpu}" "${address}"
done

printf '%s\n' "${server_pids[@]}" >"${LOG_DIR}/server.pids"
printf '%s\n' "${server_addresses[@]}" >"${LOG_DIR}/server.addresses"

error_file="${OUTPUT_FILE%.jsonl}_error.jsonl"
expected_inputs=$(awk 'NF {n++} END {print n+0}' "${INPUT_FILE}")

run_generation() {
  local -a regen_args
  regen_args=(
    --model "${MODEL_PATH}"
    --model-type qwen
    --input-file-path "${INPUT_FILE}"
    --output-file-path "${OUTPUT_FILE}"
    --server-address "${server_addresses[@]}"
    --concurrency "${CONCURRENCY}"
    --temperature "${TEMPERATURE}"
    --max-tokens "${MAX_TOKENS}"
    --max-input-tokens "${MAX_INPUT_TOKENS}"
    --num-generations-per-sample 1
  )
  if [ -s "${OUTPUT_FILE}" ] || [ -s "${error_file}" ]; then
    regen_args+=(--resume --retry-errors)
  fi
  "${PYTHON_BIN}" -u "${GENERATOR_SCRIPT}" "${regen_args[@]}"
}

printf 'generating: expected_inputs=%s temperature=%s max_tokens=%s gpus=%s\n' \
  "${expected_inputs}" "${TEMPERATURE}" "${MAX_TOKENS}" "${GPUS}" \
  | tee "${LOG_DIR}/status"

cd /data/wanghanzhen/FlashMTP_v2swa
for ((retry_pass = 0; retry_pass <= MAX_RETRY_PASSES; retry_pass++)); do
  run_generation 2>&1 | tee -a "${LOG_DIR}/regenerate.log"
  successes=$(awk 'NF {n++} END {print n+0}' "${OUTPUT_FILE}" 2>/dev/null || echo 0)
  errors=$(awk 'NF {n++} END {print n+0}' "${error_file}" 2>/dev/null || echo 0)
  if [ "${successes}" -ge "${expected_inputs}" ] && [ "${errors}" -eq 0 ]; then
    printf 'complete: success=%s expected=%s errors=0\n' \
      "${successes}" "${expected_inputs}" | tee "${LOG_DIR}/status"
    exit 0
  fi
  printf 'retry_pass=%s success=%s expected=%s errors=%s\n' \
    "${retry_pass}" "${successes}" "${expected_inputs}" "${errors}" \
    | tee "${LOG_DIR}/status"
done

printf 'failed: success=%s expected=%s errors=%s\n' \
  "${successes}" "${expected_inputs}" "${errors}" | tee "${LOG_DIR}/status"
exit 2
