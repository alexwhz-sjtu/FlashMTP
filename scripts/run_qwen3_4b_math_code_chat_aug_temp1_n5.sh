#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/share/dai-sys/wanghanzhen/projects/MTP
FLASHMTP_ROOT=${PROJECT_ROOT}/FlashMTP_v2
PYTHON=/share/dai-sys/wanghanzhen/envs/mtp-sglang/bin/python
MODEL=/share/dai-sys/wanghanzhen/models/Qwen/Qwen3-4B
INPUT=${INPUT:-${PROJECT_ROOT}/training_data/sampled_origin/math_code_chat_aug.jsonl}
OUTPUT=${OUTPUT:-${PROJECT_ROOT}/training_data/generated/qwen3-4b/math_code_chat_aug_think_off_temp1.0_topp0.9_n5_maxnew4096.jsonl}
LOG_DIR=${LOG_DIR:-${PROJECT_ROOT}/training_data/generated/qwen3-4b/logs/math_code_chat_aug_temp1.0_topp0.9_n5}
BASE_PORT=${BASE_PORT:-48000}
SERVER_CONCURRENCY=${SERVER_CONCURRENCY:-32}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.90}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-40960}
MAX_TOKENS=${MAX_TOKENS:-4096}
MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS:-32768}
NUM_GENERATIONS=${NUM_GENERATIONS:-5}
MAX_RETRY_PASSES=${MAX_RETRY_PASSES:-3}

mkdir -p "$(dirname "${OUTPUT}")" "${LOG_DIR}"
printf '%s\n' "$$" >"${LOG_DIR}/orchestrator.pid"
printf 'starting\n' >"${LOG_DIR}/status"

server_pids=()
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

export PATH="$(dirname "${PYTHON}"):${PATH}"
export TOKENIZERS_PARALLELISM=false
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"

if [ ! -s "${INPUT}" ]; then
  printf 'failed: input is missing or empty: %s\n' "${INPUT}" | tee "${LOG_DIR}/status"
  exit 1
fi

busy_gpus=$(nvidia-smi --query-gpu=memory.used,utilization.gpu \
  --format=csv,noheader,nounits | awk '$1 > 1024 || $2 > 5 {n++} END {print n+0}')
if [ "${busy_gpus}" -ne 0 ]; then
  printf 'failed: expected 8 idle GPUs, found %s busy\n' "${busy_gpus}" | tee "${LOG_DIR}/status"
  exit 1
fi

server_addresses=()
for gpu in 0 1 2 3 4 5 6 7; do
  port=$((BASE_PORT + gpu))
  server_addresses+=("127.0.0.1:${port}")
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m sglang.launch_server \
    --model-path "${MODEL}" \
    --host 127.0.0.1 \
    --port "${port}" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tp-size 1 \
    --attention-backend fa3 \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --context-length "${CONTEXT_LENGTH}" \
    --max-running-requests "${SERVER_CONCURRENCY}" \
    --cuda-graph-max-bs "${SERVER_CONCURRENCY}" \
    --mamba-scheduler-strategy no_buffer \
    --disable-radix-cache \
    >"${LOG_DIR}/server_gpu${gpu}.log" 2>&1 &
  server_pids+=("$!")
done
printf '%s\n' "${server_pids[@]}" >"${LOG_DIR}/server.pids"

deadline=$((SECONDS + 1200))
for index in "${!server_addresses[@]}"; do
  address=${server_addresses[${index}]}
  pid=${server_pids[${index}]}
  until curl -fsS "http://${address}/model_info" >/dev/null 2>&1; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      printf 'failed: SGLang server GPU=%s address=%s\n' "${index}" "${address}" | tee "${LOG_DIR}/status"
      tail -100 "${LOG_DIR}/server_gpu${index}.log" >&2
      exit 1
    fi
    if (( SECONDS >= deadline )); then
      printf 'failed: timed out waiting for %s\n' "${address}" | tee "${LOG_DIR}/status"
      exit 1
    fi
    sleep 2
  done
  printf 'SGLang server ready: GPU=%s, address=%s\n' "${index}" "${address}"
done

expected_inputs=$(awk 'NF {n++} END {print n+0}' "${INPUT}")
expected_outputs=$((expected_inputs * NUM_GENERATIONS))
error_file=${OUTPUT%.jsonl}_error.jsonl

run_generation() {
  local -a regen_args
  regen_args=(
    --model "${MODEL}"
    --model-type qwen
    --input-file-path "${INPUT}"
    --output-file-path "${OUTPUT}"
    --server-address "${server_addresses[@]}"
    --concurrency "${SERVER_CONCURRENCY}"
    --temperature 1.0
    --top-p 0.9
    --max-tokens "${MAX_TOKENS}"
    --max-input-tokens "${MAX_INPUT_TOKENS}"
    --num-generations-per-sample "${NUM_GENERATIONS}"
  )
  if [ -s "${OUTPUT}" ] || [ -s "${error_file}" ]; then
    regen_args+=(--resume --retry-errors)
  fi
  "${PYTHON}" -u scripts/regenerate_train_data.py "${regen_args[@]}"
}

cd "${FLASHMTP_ROOT}"
printf 'generating: expected_outputs=%s\n' "${expected_outputs}" >"${LOG_DIR}/status"
for ((retry_pass = 0; retry_pass <= MAX_RETRY_PASSES; retry_pass++)); do
  run_generation 2>&1 | tee -a "${LOG_DIR}/regenerate.log"
  successes=$(awk 'NF {n++} END {print n+0}' "${OUTPUT}")
  errors=$(awk 'NF {n++} END {print n+0}' "${error_file}")
  if [ "${successes}" -eq "${expected_outputs}" ] && [ "${errors}" -eq 0 ]; then
    printf 'complete: success=%s expected=%s errors=0\n' \
      "${successes}" "${expected_outputs}" | tee "${LOG_DIR}/status"
    exit 0
  fi
  printf 'retry_pass=%s success=%s expected=%s errors=%s\n' \
    "${retry_pass}" "${successes}" "${expected_outputs}" "${errors}" \
    | tee "${LOG_DIR}/status"
done

printf 'failed: success=%s expected=%s errors=%s\n' \
  "${successes}" "${expected_outputs}" "${errors}" | tee "${LOG_DIR}/status"
exit 2
