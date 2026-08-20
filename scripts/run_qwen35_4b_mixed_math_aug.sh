#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/share/dai-sys/wanghanzhen/projects/MTP
FLASHMTP_ROOT=${PROJECT_ROOT}/FlashMTP_v2
PYTHON=/share/dai-sys/wanghanzhen/envs/mtp-sglang/bin/python
MODEL=/share/dai-sys/wanghanzhen/models/Qwen/Qwen3.5-4B
INPUT_DIR=${PROJECT_ROOT}/training_data/sampled_origin
OUTPUT_DIR=${PROJECT_ROOT}/training_data/generated/qwen3.5-4b
LOG_DIR=${OUTPUT_DIR}/logs

MIXED_INPUT=${INPUT_DIR}/mixed_2,350,325.jsonl
MATH_INPUT=${INPUT_DIR}/math_code_aug.jsonl
MIXED_OUTPUT=${OUTPUT_DIR}/mixed_2350325_think_off_temp0_maxnew4096.jsonl
MATH_OUTPUT=${OUTPUT_DIR}/math_code_aug_think_off_temp0_maxnew4096.jsonl

BASE_PORT=${BASE_PORT:-46000}
SERVER_CONCURRENCY=${SERVER_CONCURRENCY:-32}
MIXED_CONCURRENCY=${MIXED_CONCURRENCY:-16}
MATH_CONCURRENCY=${MATH_CONCURRENCY:-4}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.90}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-262144}
MAX_TOKENS=${MAX_TOKENS:-4096}

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# The OpenAI client constructs transports for ALL_PROXY even for localhost.
# These jobs only talk to the eight local SGLang endpoints.
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
export NO_PROXY=127.0.0.1,localhost,::1
export no_proxy="${NO_PROXY}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export TOKENIZERS_PARALLELISM=false

server_pids=()
regen_pids=()
cleanup() {
  local pid
  for pid in "${regen_pids[@]}" "${server_pids[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  for pid in "${regen_pids[@]}" "${server_pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

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

deadline=$((SECONDS + 1200))
for index in "${!server_addresses[@]}"; do
  address=${server_addresses[${index}]}
  pid=${server_pids[${index}]}
  until curl -fsS "http://${address}/model_info" >/dev/null 2>&1; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "SGLang server failed: GPU=${index}, address=${address}" >&2
      tail -100 "${LOG_DIR}/server_gpu${index}.log" >&2
      exit 1
    fi
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for SGLang server: ${address}" >&2
      exit 1
    fi
    sleep 2
  done
  echo "SGLang server ready: GPU=${index}, address=${address}"
done

common_args=(
  --model "${MODEL}"
  --model-type qwen
  --server-address "${server_addresses[@]}"
  --temperature 0
  --top-p 1
  --top-k 1
  --max-tokens "${MAX_TOKENS}"
)

math_args=(
  "${common_args[@]}"
  --input-file-path "${MATH_INPUT}"
  --output-file-path "${MATH_OUTPUT}"
  --concurrency "${MATH_CONCURRENCY}"
)
if [[ -s "${MATH_OUTPUT}" ]]; then
  math_args+=(--resume)
fi

mixed_args=(
  "${common_args[@]}"
  --input-file-path "${MIXED_INPUT}"
  --output-file-path "${MIXED_OUTPUT}"
  --concurrency "${MIXED_CONCURRENCY}"
)
if [[ -s "${MIXED_OUTPUT}" ]]; then
  mixed_args+=(--resume)
fi

# Run the two datasets independently so each has its own output/error/log files.
"${PYTHON}" -u "${FLASHMTP_ROOT}/scripts/regenerate_train_data.py" \
  "${math_args[@]}" >"${LOG_DIR}/regenerate_math_code_aug.log" 2>&1 &
math_pid=$!
regen_pids+=("${math_pid}")

"${PYTHON}" -u "${FLASHMTP_ROOT}/scripts/regenerate_train_data.py" \
  "${mixed_args[@]}" >"${LOG_DIR}/regenerate_mixed.log" 2>&1 &
mixed_pid=$!
regen_pids+=("${mixed_pid}")

set +e
wait "${math_pid}"
math_status=$?
if (( math_status != 0 )); then
  echo "math_code_aug generation failed with status ${math_status}" >&2
  exit "${math_status}"
fi
echo "math_code_aug generation complete: ${MATH_OUTPUT}"

wait "${mixed_pid}"
mixed_status=$?
set -e
if (( mixed_status != 0 )); then
  echo "mixed generation failed with status ${mixed_status}" >&2
  exit "${mixed_status}"
fi
echo "mixed generation complete: ${MIXED_OUTPUT}"

wc -l \
  "${MATH_OUTPUT}" "${MATH_OUTPUT%.jsonl}_error.jsonl" \
  "${MIXED_OUTPUT}" "${MIXED_OUTPUT%.jsonl}_error.jsonl"
