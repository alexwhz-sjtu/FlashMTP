#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="/data/wanghanzhen/FlashMTP_v2.3"
SGLANG_ROOT="/data/zhaotianlang/sglang-flashmtp"
RUNNER="/data/wanghanzhen/FlashMTP_v2swa/sglang_bench_20260919/run_matrix.py"
OUT="${PROJECT_ROOT}/sglang_bench_ep10_20260920/by_dataset_128_no_overlap"
TARGET="/data/wanghanzhen/models/Qwen3-8B"
FLASH="${PROJECT_ROOT}/cache/models/full/Flashmtp_v2.3_new_qwen3_8b_teacher_d10_swa512_ep10"
DSPARK="/data/wanghanzhen/models/dspark_qwen3_8b_block7"

datasets=(alpaca livecodebench mt-bench)
mkdir -p "${OUT}/workers"

pids=()
for i in "${!datasets[@]}"; do
  dataset="${datasets[$i]}"
  port=$((42000 + i))
  (
    source "${SGLANG_ROOT}/.venv/bin/activate"
    CUDA_VISIBLE_DEVICES="${i}" "${SGLANG_ROOT}/.venv/bin/python" "${RUNNER}" \
      --dataset "${dataset}" \
      --methods flashmtp,dspark \
      --concurrencies 1,8,16,32,64 \
      --target-model "${TARGET}" \
      --flash-model "${FLASH}" \
      --dspark-model "${DSPARK}" \
      --max-samples 128 \
      --max-new-tokens 512 \
      --port "${port}" \
      --output-dir "${OUT}"
  ) > "${OUT}/workers/${dataset}.log" 2>&1 &
  pids+=("$!")
done

overall=0
for pid in "${pids[@]}"; do
  wait "${pid}" || overall=1
done

exit "${overall}"
