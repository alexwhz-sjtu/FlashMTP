#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/share/dai-sys/wanghanzhen/projects/MTP
FLASHMTP_ROOT=${PROJECT_ROOT}/FlashMTP_v2
PYTHON=/share/dai-sys/wanghanzhen/envs/mtp-sglang/bin/python
MODEL=/share/dai-sys/wanghanzhen/models/Qwen/Qwen3.5-35B-A3B
INPUT=${PROJECT_ROOT}/training_data/sampled_origin/open_perfectblend_80k_prompts.jsonl
OUTPUT=${PROJECT_ROOT}/training_data/generated/qwen3.5-35b-a3b/open_perfectblend_80k_think_off_temp0_maxnew4096.jsonl
LOG_DIR=${PROJECT_ROOT}/training_data/generated/qwen3.5-35b-a3b/logs
BASE_PORT=${BASE_PORT:-42000}
SERVER_CONCURRENCY=${SERVER_CONCURRENCY:-16}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.90}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-262144}

mkdir -p "$(dirname "${OUTPUT}")" "${LOG_DIR}"

server_pids=()
cleanup() {
  local pid
  for pid in "${server_pids[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  for pid in "${server_pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

export PATH="$(dirname "${PYTHON}"):${PATH}"
export TOKENIZERS_PARALLELISM=false
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

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

regen_args=(
  --model "${MODEL}"
  --model-type qwen
  --input-file-path "${INPUT}"
  --output-file-path "${OUTPUT}"
  --server-address "${server_addresses[@]}"
  --concurrency "${SERVER_CONCURRENCY}"
  --temperature 0
  --top-p 1
  --top-k 1
  --max-tokens 4096
)
if [[ -s "${OUTPUT}" ]]; then
  regen_args+=(--resume)
fi

cd "${FLASHMTP_ROOT}"
"${PYTHON}" -u scripts/regenerate_train_data.py "${regen_args[@]}" \
  2>&1 | tee "${LOG_DIR}/regenerate.log"

echo "Generation complete: ${OUTPUT}"
wc -l "${OUTPUT}" "${OUTPUT%.jsonl}_error.jsonl"
