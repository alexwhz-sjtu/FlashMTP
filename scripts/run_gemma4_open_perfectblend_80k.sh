#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/share/dai-sys/wanghanzhen/projects/MTP
FLASHMTP_ROOT=${PROJECT_ROOT}/FlashMTP_v2
PYTHON=${FLASHMTP_ROOT}/.venv-gemma4/bin/python
MODEL=/share/dai-sys/wanghanzhen/models/google/gemma-4-12B-it
INPUT=${PROJECT_ROOT}/training_data/sampled_origin/open_perfectblend_80k_prompts.jsonl
OUTPUT=${OUTPUT:-${PROJECT_ROOT}/training_data/generated/gemma4-12b/open_perfectblend_80k_think_off_temp0_maxnew4096.jsonl}
LOG_DIR=${LOG_DIR:-${PROJECT_ROOT}/training_data/generated/gemma4-12b/logs}
BASE_PORT=${BASE_PORT:-43000}
SERVER_CONCURRENCY=${SERVER_CONCURRENCY:-16}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-65536}
NUM_SAMPLES=${NUM_SAMPLES:-}
EXPECTED_SAMPLES=${EXPECTED_SAMPLES:-80000}

wait_for_all_gpus() {
  while true; do
    local busy
    busy=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | awk '$1 > 1024 || $2 > 5 {n++} END {print n+0}')
    if [ "${busy}" -eq 0 ]; then return 0; fi
    echo "Waiting for all 8 GPUs to become idle (${busy} busy)..."
    sleep 30
  done
}

wait_for_all_gpus
mkdir -p "$(dirname "${OUTPUT}")" "${LOG_DIR}"

server_pids=()
cleanup() {
  local pid
  for pid in "${server_pids[@]:-}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${server_pids[@]:-}"; do wait "${pid}" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

export PATH="$(dirname "${PYTHON}"):${PATH}"
export TOKENIZERS_PARALLELISM=false
# Every request made by this wrapper is local.  httpx still constructs a SOCKS
# transport when both upper- and lower-case proxy variables are present, even
# with NO_PROXY configured, so remove proxies in this subprocess entirely.
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
server_addresses=()
for gpu in 0 1 2 3 4 5 6 7; do
  port=$((BASE_PORT + gpu))
  server_addresses+=("127.0.0.1:${port}")
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m sglang.launch_server \
    --model-path "${MODEL}" --host 127.0.0.1 --port "${port}" \
    --dtype bfloat16 --tp-size 1 --attention-backend triton \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --context-length "${CONTEXT_LENGTH}" \
    --max-running-requests "${SERVER_CONCURRENCY}" \
    --disable-cuda-graph --disable-radix-cache \
    >"${LOG_DIR}/server_gpu${gpu}.log" 2>&1 &
  server_pids+=("$!")
done

deadline=$((SECONDS + 1200))
for index in "${!server_addresses[@]}"; do
  address=${server_addresses[${index}]}; pid=${server_pids[${index}]}
  until curl -fsS "http://${address}/model_info" >/dev/null 2>&1; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      tail -100 "${LOG_DIR}/server_gpu${index}.log" >&2; exit 1
    fi
    if (( SECONDS >= deadline )); then echo "Timed out: ${address}" >&2; exit 1; fi
    sleep 2
  done
done

regen_args=(
  --model "${MODEL}" --model-type gemma4
  --input-file-path "${INPUT}" --output-file-path "${OUTPUT}"
  --server-address "${server_addresses[@]}" --concurrency "${SERVER_CONCURRENCY}"
  --temperature 0 --top-p 1 --top-k 1 --max-tokens 4096
)
if [ -n "${NUM_SAMPLES}" ]; then regen_args+=(--num-samples "${NUM_SAMPLES}"); fi
if [ -s "${OUTPUT}" ]; then regen_args+=(--resume --retry-errors); fi

cd "${FLASHMTP_ROOT}"
"${PYTHON}" -u scripts/regenerate_train_data.py "${regen_args[@]}" \
  2>&1 | tee "${LOG_DIR}/regenerate.log"

actual=$(wc -l < "${OUTPUT}")
errors=$(wc -l < "${OUTPUT%.jsonl}_error.jsonl")
expected=${NUM_SAMPLES:-${EXPECTED_SAMPLES}}
if [ "${actual}" -ne "${expected}" ] || [ "${errors}" -ne 0 ]; then
  echo "Generation incomplete: success=${actual}/${expected}, errors=${errors}" >&2
  exit 2
fi
