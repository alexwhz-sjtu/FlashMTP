#!/usr/bin/env bash
set -uo pipefail
cd /data/wanghanzhen/FlashMTP_v2swa
run=/data/wanghanzhen/FlashMTP_v2swa/diagnostics/paired_eval_20260923
pids=()
for rank in 0 1 2 3 4 5 6 7; do
 CUDA_VISIBLE_DEVICES="$rank" OMP_NUM_THREADS=4 .venv/bin/python -u "$run/paired_eval.py" worker --run "$run" --rank "$rank" --world 8 > "$run/worker_$rank.log" 2>&1 &
 pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
 wait "$pid" || status=1
done
printf '%s\n' "$status" > "$run/exit_status.txt"
if [ "$status" = 0 ]; then
 .venv/bin/python "$run/paired_eval.py" summarize --run "$run" > "$run/summary_generation.log" 2>&1
fi
exit "$status"
