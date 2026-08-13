#!/bin/bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"

RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/benchmark_results/ep8_ep10_long_short_20260813}"
MANIFEST="${RUN_ROOT}/manifest.tsv"
PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
TARGET_MODEL="/share/dai-sys/wanghanzhen/models/Qwen/Qwen3-4B"
TMP_ROOT="${RUN_ROOT}/tmp"
CLAIM_ROOT="${RUN_ROOT}/claims"
MAX_SAMPLES=50
MAX_NEW_TOKENS=512
BATCH_SIZE=1
TEMPERATURE=0

if [ "$#" -lt 1 ]; then
    echo "usage: $0 GPU [GPU ...]" >&2
    exit 2
fi
if [ ! -f "${MANIFEST}" ]; then
    echo "missing manifest: ${MANIFEST}" >&2
    exit 2
fi

mkdir -p "${TMP_ROOT}" "${CLAIM_ROOT}" "${RUN_ROOT}/dynamic_workers"

status_state() {
    local status_path="$1"
    if [ -f "${status_path}" ]; then
        head -1 "${status_path}"
    else
        printf 'pending\n'
    fi
}

all_terminal() {
    local model dataset gpu draft_path log_path status_path
    while IFS=$'\t' read -r model dataset gpu draft_path log_path status_path; do
        if [ "${model}" = model ]; then
            continue
        fi
        case "$(status_state "${status_path}")" in
            completed|failed) ;;
            *) return 1 ;;
        esac
    done < "${MANIFEST}"
    return 0
}

run_worker() {
    local worker_gpu="$1"
    local worker_log="${RUN_ROOT}/dynamic_workers/gpu_${worker_gpu}.log"
    : > "${worker_log}"

    while true; do
        local claimed=0
        local model dataset manifest_gpu draft_path log_path status_path
        while IFS=$'\t' read -r model dataset manifest_gpu draft_path log_path status_path; do
            if [ "${model}" = model ]; then
                continue
            fi
            local state
            state="$(status_state "${status_path}")"
            if [ "${state}" != pending ]; then
                continue
            fi

            local claim_dir="${CLAIM_ROOT}/${model}/${dataset}"
            mkdir -p "$(dirname "${claim_dir}")"
            if ! mkdir "${claim_dir}" 2>/dev/null; then
                continue
            fi
            claimed=1

            mkdir -p "$(dirname "${log_path}")" "$(dirname "${status_path}")"
            {
                printf '[START] %s gpu=%s model=%s dataset=%s scheduler=dynamic\n' \
                    "$(date --iso-8601=seconds)" "${worker_gpu}" "${model}" "${dataset}"
                printf '[COMMAND] CUDA_VISIBLE_DEVICES=%s %q evaluation/benchmark.py ' \
                    "${worker_gpu}" "${PYTHON_BIN}"
                printf '%q ' \
                    --model-name-or-path "${TARGET_MODEL}" \
                    --draft-name-or-path "${draft_path}" \
                    --dataset "${dataset}" \
                    --max-samples "${MAX_SAMPLES}" \
                    --max-new-tokens "${MAX_NEW_TOKENS}" \
                    --batch-size "${BATCH_SIZE}" \
                    --temperature "${TEMPERATURE}"
                printf '\n'
            } > "${log_path}"
            printf 'running\nstarted_at=%s\ngpu=%s\nscheduler=dynamic\n' \
                "$(date --iso-8601=seconds)" "${worker_gpu}" > "${status_path}"
            printf '[START] %s model=%s dataset=%s\n' \
                "$(date --iso-8601=seconds)" "${model}" "${dataset}" >> "${worker_log}"

            env CUDA_VISIBLE_DEVICES="${worker_gpu}" \
                TMPDIR="${TMP_ROOT}" \
                PYTHONUNBUFFERED=1 NO_COLOR=1 COLUMNS=200 \
                "${PYTHON_BIN}" evaluation/benchmark.py \
                --model-name-or-path "${TARGET_MODEL}" \
                --draft-name-or-path "${draft_path}" \
                --dataset "${dataset}" \
                --max-samples "${MAX_SAMPLES}" \
                --max-new-tokens "${MAX_NEW_TOKENS}" \
                --batch-size "${BATCH_SIZE}" \
                --temperature "${TEMPERATURE}" \
                >> "${log_path}" 2>&1
            local exit_code=$?

            if [ "${exit_code}" -eq 0 ]; then
                printf 'completed\nfinished_at=%s\ngpu=%s\nscheduler=dynamic\n' \
                    "$(date --iso-8601=seconds)" "${worker_gpu}" > "${status_path}"
                printf '[DONE] %s model=%s dataset=%s\n' \
                    "$(date --iso-8601=seconds)" "${model}" "${dataset}" >> "${worker_log}"
            else
                printf 'failed\nexit_code=%s\nfinished_at=%s\ngpu=%s\nscheduler=dynamic\n' \
                    "${exit_code}" "$(date --iso-8601=seconds)" "${worker_gpu}" > "${status_path}"
                printf '[FAIL] %s exit=%s model=%s dataset=%s\n' \
                    "$(date --iso-8601=seconds)" "${exit_code}" "${model}" "${dataset}" >> "${worker_log}"
            fi
            break
        done < "${MANIFEST}"

        if [ "${claimed}" -eq 1 ]; then
            continue
        fi
        if all_terminal; then
            printf '[EXIT] %s all tasks terminal\n' "$(date --iso-8601=seconds)" >> "${worker_log}"
            return
        fi
        sleep 10
    done
}

pids=()
for gpu in "$@"; do
    run_worker "${gpu}" &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "${pid}"
done
