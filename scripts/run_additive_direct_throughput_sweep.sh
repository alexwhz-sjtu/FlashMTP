#!/bin/bash
# End-to-end throughput sweep: additive r256 vs direct r512
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"

if [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
    source "${PROJECT_DIR}/.venv/bin/activate"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export TARGET_MODEL="${TARGET_MODEL:-/data/wanghanzhen/models/Qwen3-8B}"
export BLOCK_SIZE="${BLOCK_SIZE:-16}"
export MAX_SAMPLES="${MAX_SAMPLES:-50}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
export TEMPERATURE="${TEMPERATURE:-0.0}"
export MASTER_PORT="${MASTER_PORT:-29512}"

ADDITIVE_CKPT="${ADDITIVE_CKPT:-${PROJECT_DIR}/cache/models/flashmtp_h100_prefix_condition_fuse18_sample_80000_nlayers5_block_16_mhrnn_additive_r256_wb_0.0_maxlen4096_epochs6_Qwen3-8B/epoch_6_step_59496}"
DIRECT_CKPT="${DIRECT_CKPT:-${PROJECT_DIR}/cache/models/flashmtp_h100_prefix_condition_fuse18_sample_80000_nlayers5_block_16_mhrnn_direct_r512_wb_0.2_bgemma_21_maxlen4096_epochs6_Qwen3-8B/epoch_6_step_59496}"

LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/log/throughput_sweep_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${LOG_DIR}"
SUMMARY_CSV="${LOG_DIR}/summary.csv"
echo "model,dataset,batch_size,throughput_ratio,token_weighted_speedup,avg_accept_length,baseline_s_per_token,flashmtp_s_per_token,elapsed_s,log_file" > "${SUMMARY_CSV}"

run_one() {
    local model_label="$1"
    local draft_ckpt="$2"
    local dataset="$3"
    local batch_size="$4"
    local log_file="${LOG_DIR}/${model_label}_${dataset}_b${batch_size}.log"
    local start_ts end_ts elapsed

    echo "========================================"
    echo "RUN ${model_label} | ${dataset} | batch=${batch_size}"
    echo "Log: ${log_file}"
    echo "========================================"

    start_ts=$(date +%s)
    set +e
    torchrun --nproc_per_node 1 --master_port "${MASTER_PORT}" evaluation/benchmark.py \
        --model-name-or-path "${TARGET_MODEL}" \
        --draft-name-or-path "${draft_ckpt}" \
        --block-size "${BLOCK_SIZE}" \
        --dataset "${dataset}" \
        --max-samples "${MAX_SAMPLES}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --batch-size "${batch_size}" \
        --temperature "${TEMPERATURE}" \
        2>&1 | tee "${log_file}"
    local status=${PIPESTATUS[0]}
    set -e
    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))

    if [ "${status}" -ne 0 ]; then
        echo "FAILED status=${status}" | tee -a "${log_file}"
        echo "${model_label},${dataset},${batch_size},,,,,,${elapsed},${log_file}" >> "${SUMMARY_CSV}"
        return "${status}"
    fi

    python3 - "${log_file}" "${model_label}" "${dataset}" "${batch_size}" "${elapsed}" "${SUMMARY_CSV}" <<'PY'
import re
import sys
from pathlib import Path

log_file, model_label, dataset, batch_size, elapsed, summary_csv = sys.argv[1:7]
text = Path(log_file).read_text()
m = re.search(
    r"=== Overall \(batch_size=\d+\) ===\s+"
    r"turns: (\d+)\s+"
    r"token-weighted speedup: ([0-9.]+)x \| throughput ratio: ([0-9.]+)x.*?\n"
    r"\s+decode s/token baseline=([0-9.]+) flashmtp=([0-9.]+)\s+.*?\n"
    r".*?average acceptance length: ([0-9.]+)",
    text,
    flags=re.S,
)
if not m:
    print(f"WARN: could not parse summary from {log_file}")
    with open(summary_csv, "a") as f:
        f.write(f"{model_label},{dataset},{batch_size},,,,,,{elapsed},{log_file}\n")
    sys.exit(0)

turns, tw_speedup, throughput_ratio, baseline_s, flashmtp_s, avg_accept = m.groups()
with open(summary_csv, "a") as f:
    f.write(
        f"{model_label},{dataset},{batch_size},{throughput_ratio},{tw_speedup},"
        f"{avg_accept},{baseline_s},{flashmtp_s},{elapsed},{log_file}\n"
    )
print(
    f"Parsed: throughput_ratio={throughput_ratio}x avg_accept={avg_accept} elapsed={elapsed}s"
)
PY
}

for dataset in gsm8k alpaca; do
    for batch_size in 8 16 32; do
        run_one additive_r256 "${ADDITIVE_CKPT}" "${dataset}" "${batch_size}" || true
        run_one direct_r512 "${DIRECT_CKPT}" "${dataset}" "${batch_size}" || true
        MASTER_PORT=$((MASTER_PORT + 1))
    done
done

echo ""
echo "=== FINAL SUMMARY ==="
column -t -s, "${SUMMARY_CSV}" || cat "${SUMMARY_CSV}"
