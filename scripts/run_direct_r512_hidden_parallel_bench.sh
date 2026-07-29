#!/bin/bash
# 8-GPU parallel benchmark: direct r512 + hidden_proj parallel, batch 1 & 32
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"

if [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
    source "${PROJECT_DIR}/.venv/bin/activate"
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TARGET_MODEL="${TARGET_MODEL:-/data/wanghanzhen/models/Qwen3-8B}"
export DRAFT_CKPT="${DRAFT_CKPT:-${PROJECT_DIR}/cache/models/flashmtp_h100_prefix_condition_fuse18_sample_80000_nlayers5_block_16_mhrnn_direct_r512_wb_0.2_bgemma_21_maxlen4096_epochs6_Qwen3-8B/epoch_6_step_59496}"
export BLOCK_SIZE="${BLOCK_SIZE:-16}"
export MAX_SAMPLES="${MAX_SAMPLES:-50}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
export TEMPERATURE="${TEMPERATURE:-0.0}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/log/direct_r512_hidden_parallel_${TIMESTAMP}}"
mkdir -p "${LOG_DIR}"

SUMMARY_CSV="${LOG_DIR}/summary.csv"
echo "gpu,dataset,batch_size,throughput_ratio,token_weighted_speedup,avg_accept_length,baseline_s_per_token,flashmtp_s_per_token,elapsed_s,log_file,status" > "${SUMMARY_CSV}"

# Verify hidden parallel optimization present
if ! grep -q "_precompute_hidden_latents" "${PROJECT_DIR}/specforge/modeling/draft/flashmtp_markov_head.py"; then
    echo "ERROR: _precompute_hidden_latents not found in flashmtp_markov_head.py"
    exit 1
fi
echo "Confirmed: hidden_proj parallel optimization active (_precompute_hidden_latents)"

run_job() {
    local gpu="$1"
    local dataset="$2"
    local batch_size="$3"
    local master_port="$4"
    local log_file="${LOG_DIR}/${dataset}_b${batch_size}_gpu${gpu}.log"
    local start_ts end_ts elapsed status=0

    echo "[GPU${gpu}] START ${dataset} batch=${batch_size} -> ${log_file}"
    start_ts=$(date +%s)

    set +e
    env CUDA_VISIBLE_DEVICES="${gpu}" \
        torchrun --nproc_per_node 1 --master_port "${master_port}" \
        evaluation/benchmark.py \
        --model-name-or-path "${TARGET_MODEL}" \
        --draft-name-or-path "${DRAFT_CKPT}" \
        --block-size "${BLOCK_SIZE}" \
        --dataset "${dataset}" \
        --max-samples "${MAX_SAMPLES}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --batch-size "${batch_size}" \
        --temperature "${TEMPERATURE}" \
        2>&1 | tee "${log_file}"
    status=${PIPESTATUS[0]}
    set -e

    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))

    python3 - "${log_file}" "${gpu}" "${dataset}" "${batch_size}" "${elapsed}" "${status}" "${SUMMARY_CSV}" <<'PY'
import re, sys
from pathlib import Path

log_file, gpu, dataset, batch_size, elapsed, status, summary_csv = sys.argv[1:8]
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
stat = "ok" if int(status) == 0 and m else "fail"
with open(summary_csv, "a") as f:
    if m:
        turns, tw, tr, bl, fm, acc = m.groups()
        f.write(f"{gpu},{dataset},{batch_size},{tr},{tw},{acc},{bl},{fm},{elapsed},{log_file},{stat}\n")
        print(f"[GPU{gpu}] DONE {dataset} b={batch_size}: throughput={tr}x accept={acc} elapsed={elapsed}s")
    else:
        f.write(f"{gpu},{dataset},{batch_size},,,,,,{elapsed},{log_file},{stat}\n")
        print(f"[GPU{gpu}] FAILED {dataset} b={batch_size} status={status}")
PY
}

# Launch 8 jobs in parallel (one per GPU)
run_job 0 gsm8k 1 29600 &
run_job 1 gsm8k 32 29601 &
run_job 2 alpaca 1 29602 &
run_job 3 alpaca 32 29603 &
run_job 4 mt-bench 1 29604 &
run_job 5 mt-bench 32 29605 &
run_job 6 livecodebench 1 29606 &
run_job 7 livecodebench 32 29607 &

wait

echo ""
echo "=== ALL JOBS COMPLETE ==="
echo "Log dir: ${LOG_DIR}"
column -t -s, "${SUMMARY_CSV}" || cat "${SUMMARY_CSV}"
