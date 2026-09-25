#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="${RUN_ROOT:-$ROOT/benchmark_results/Flashmtp_v2_qwen3.5_4b_ep3_20260924/full}"
GPU="${GPU:-0}"
mkdir -p "$OUT"
exec 9>"$OUT/run.lock"
flock -n 9 || { echo 'This run is already active'; exit 1; }
export CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 NO_COLOR=1
unset CUDA_LAUNCH_BLOCKING
PY="$ROOT/.venv-qwen35-eval/bin/python"
"$PY" -m pip freeze --local > "$OUT/environment.txt"
git diff > "$OUT/source.patch"
cp evaluation/qwen35_target.py "$OUT/qwen35_target.py"
printf 'running\n' > "$OUT/status"
trap 'printf "failed\n" > "$OUT/status"' ERR
for dataset in gsm8k math500 aime25 humaneval mbpp livecodebench mt-bench alpaca \
    longbench_v2_64000_32000_single_document_qa \
    longbench_v2_64000_32000_multi_document_qa \
    longbench_v2_64000_32000_long_dialogue \
    longbench_v2_64000_32000_structured_data \
    longbench_v2_64000_32000_in_context_learning \
    longbench_v2_64000_32000_code_repo; do
    if [[ -s "$OUT/$dataset.json" ]]; then continue; fi
    printf '%s START %s\n' "$(date -Is)" "$dataset"
    printf '%s\n' "$dataset" > "$OUT/current_dataset"
    "$PY" evaluation/benchmark.py \
        --model-name-or-path /data/wanghanzhen/models/Qwen/Qwen3.5-4B \
        --draft-name-or-path "$ROOT/cache/models/Qwen3.5-4B/Flashmtp_v2_qwen3.5_4b_ep3" \
        --dataset "$dataset" --max-new-tokens 512 --temperature 0 \
        --output-json "$OUT/$dataset.json" > "$OUT/$dataset.log" 2>&1
    printf '%s DONE %s\n' "$(date -Is)" "$dataset"
done
printf 'completed\n' > "$OUT/status"
