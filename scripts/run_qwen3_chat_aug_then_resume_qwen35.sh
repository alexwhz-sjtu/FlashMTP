#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/share/dai-sys/wanghanzhen/projects/MTP
CURRENT_ROOT=${PROJECT_ROOT}/FlashMTP_v2swa
REGEN_SCRIPT=${PROJECT_ROOT}/FlashMTP_v2/scripts/regenerate_train_data.py
PYTHON=/share/dai-sys/wanghanzhen/envs/mtp-sglang/bin/python
INPUT=${PROJECT_ROOT}/training_data/sampled_origin/chat_aug.jsonl

BASE_PORT=${BASE_PORT:-47000}
SERVER_CONCURRENCY=${SERVER_CONCURRENCY:-32}
CLIENT_CONCURRENCY=${CLIENT_CONCURRENCY:-16}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.90}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-40960}
MAX_TOKENS=${MAX_TOKENS:-4096}

unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
export NO_PROXY=127.0.0.1,localhost,::1
export no_proxy="${NO_PROXY}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export TOKENIZERS_PARALLELISM=false

server_pids=()

stop_servers() {
  local pid deadline alive
  for pid in "${server_pids[@]}"; do
    kill -TERM -- "-${pid}" 2>/dev/null || true
  done

  deadline=$((SECONDS + 30))
  while (( SECONDS < deadline )); do
    alive=0
    for pid in "${server_pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        alive=1
        break
      fi
    done
    (( alive == 0 )) && break
    sleep 1
  done

  for pid in "${server_pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -KILL -- "-${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${server_pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  server_pids=()
}

cleanup() {
  stop_servers
}
trap cleanup EXIT INT TERM

run_model() {
  local model_name=$1
  local model_path=$2
  local output_dir=${PROJECT_ROOT}/training_data/generated/${model_name}
  local log_dir=${output_dir}/logs
  local output=${output_dir}/chat_aug_think_off_temp0_maxnew4096.jsonl
  local port address pid deadline index gpu
  local -a server_addresses regen_args

  mkdir -p "${output_dir}" "${log_dir}"
  server_pids=()
  server_addresses=()

  echo "Starting ${model_name}: ${model_path}"
  for gpu in 0 1 2 3 4 5 6 7; do
    port=$((BASE_PORT + gpu))
    server_addresses+=("127.0.0.1:${port}")
    setsid env CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m sglang.launch_server \
      --model-path "${model_path}" \
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
      --disable-radix-cache \
      >"${log_dir}/server_gpu${gpu}.log" 2>&1 &
    server_pids+=("$!")
  done

  deadline=$((SECONDS + 1200))
  for index in "${!server_addresses[@]}"; do
    address=${server_addresses[${index}]}
    pid=${server_pids[${index}]}
    until curl -fsS "http://${address}/model_info" >/dev/null 2>&1; do
      if ! kill -0 "${pid}" 2>/dev/null; then
        echo "SGLang server failed: model=${model_name}, GPU=${index}, address=${address}" >&2
        tail -100 "${log_dir}/server_gpu${index}.log" >&2
        return 1
      fi
      if (( SECONDS >= deadline )); then
        echo "Timed out waiting for SGLang server: ${address}" >&2
        return 1
      fi
      sleep 2
    done
    echo "SGLang server ready: model=${model_name}, GPU=${index}, address=${address}"
  done

  regen_args=(
    --model "${model_path}"
    --model-type qwen
    --input-file-path "${INPUT}"
    --output-file-path "${output}"
    --server-address "${server_addresses[@]}"
    --concurrency "${CLIENT_CONCURRENCY}"
    --temperature 0
    --top-p 1
    --top-k 1
    --max-tokens "${MAX_TOKENS}"
  )
  if [[ -s "${output}" ]]; then
    regen_args+=(--resume)
  fi

  "${PYTHON}" -u "${REGEN_SCRIPT}" "${regen_args[@]}" \
    >"${log_dir}/regenerate_chat_aug.log" 2>&1

  echo "Generation complete: ${model_name}"
  wc -l "${output}" "${output%.jsonl}_error.jsonl"
  stop_servers
}

run_model qwen3-8b /share/dai-sys/wanghanzhen/models/Qwen/Qwen3-8B
run_model qwen3-4b /share/dai-sys/wanghanzhen/models/Qwen/Qwen3-4B

echo "Qwen3 chat_aug generations complete; resuming Qwen3.5-4B generation"
trap - EXIT INT TERM
exec bash "${CURRENT_ROOT}/scripts/run_qwen35_4b_mixed_math_aug.sh"
