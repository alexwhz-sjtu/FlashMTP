#!/bin/bash
# FlashMTP 当前架构推理 / benchmark 启动脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
if [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
    source "${PROJECT_DIR}/.venv/bin/activate"
fi

cd "${PROJECT_DIR}"

# The remote .venv may reuse dependencies from another checkout. Always import
# this checkout's specforge package.
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dt) DT="$2"; shift 2 ;;
        *) shift ;;
    esac
done
DT="${DT:-a800}"
if [[ "$DT" != "qz" && "$DT" != "a800" && "$DT" != "h100" ]]; then
    echo "错误: --dt 须为 qz、a800 或 h100"
    exit 1
fi

# ========================================
# 主要推理参数（与训练脚本同名变量对齐）
# ========================================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29502}"

VERIFY_BLOCK="${VERIFY_BLOCK:-}"
MODEL_TAG="${MODEL_TAG:-Qwen3_8B}"

# 串行 head 配置保存在 checkpoint 的 flashmtp_config 中；以下仅用于日志/路径推导
MARKOV_HEAD_TYPE="${MARKOV_HEAD_TYPE:-}"
MARKOV_OUTPUT_MODE="${MARKOV_OUTPUT_MODE:-}"
MARKOV_RANK="${MARKOV_RANK:-}"

# Benchmark 数据集与采样
DATASET="${DATASET:-gsm8k}"
MAX_SAMPLES="${MAX_SAMPLES:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TEMPERATURE="${TEMPERATURE:-0.0}"

if [ "$DT" = "qz" ]; then
    TARGET_MODEL="${TARGET_MODEL:-/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/models/Qwen/Qwen3-8B}"
elif [ "$DT" = "h100" ]; then
    TARGET_MODEL="${TARGET_MODEL:-$WHZ_HOME/models/Qwen/Qwen3-8B}"
else
    TARGET_MODEL="${TARGET_MODEL:-/share/public/public_models/Qwen3-8B}"
fi

# 草稿 checkpoint：优先 DRAFT_NAME_OR_PATH / DRAFT_MODEL，否则需手动指定
DRAFT_NAME_OR_PATH="${DRAFT_NAME_OR_PATH:-${DRAFT_MODEL:-}}"

if [ -z "${DRAFT_NAME_OR_PATH}" ]; then
    echo "错误: 请设置 DRAFT_NAME_OR_PATH 或 DRAFT_MODEL 指向训练输出的 checkpoint 目录"
    echo "示例:"
    echo "  DRAFT_NAME_OR_PATH=./cache/models/flashmtp_h100_.../epoch_6_step_29844 \\"
    echo "  DATASET=gsm8k bash evaluation/run_benchmark_flashmtp.sh --dt h100"
    exit 1
fi

echo "=========================================="
echo "FlashMTP Benchmark 启动脚本"
echo "=========================================="
echo "运行环境: --dt ${DT}"
echo "目标模型: ${TARGET_MODEL}"
echo "草稿模型: ${DRAFT_NAME_OR_PATH}"
echo "数据集: ${DATASET} (max_samples=${MAX_SAMPLES})"
echo "块大小: 由 checkpoint 固定"
echo "verify_block: ${VERIFY_BLOCK:-与 block_size 相同}"
echo "draft position mode: checkpoint role fixed (teacher global / student local)"
echo "串行 head (checkpoint): type=${MARKOV_HEAD_TYPE:-auto} mode=${MARKOV_OUTPUT_MODE:-auto} rank=${MARKOV_RANK:-auto}"
echo "max_new_tokens=${MAX_NEW_TOKENS} batch_size=${BATCH_SIZE} temperature=${TEMPERATURE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "=========================================="

OPTIONAL_ARGS=""
if [ -n "${VERIFY_BLOCK}" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --verify-block ${VERIFY_BLOCK}"
fi
LAUNCHER=(
    "${PROJECT_DIR}/.venv/bin/python"
    -m
    torch.distributed.run
    --nproc_per_node "${NPROC_PER_NODE}"
    --master_port "${MASTER_PORT}"
)

"${LAUNCHER[@]}" evaluation/benchmark.py \
    --model-name-or-path "${TARGET_MODEL}" \
    --draft-name-or-path "${DRAFT_NAME_OR_PATH}" \
    --dataset "${DATASET}" \
    --max-samples "${MAX_SAMPLES}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --batch-size "${BATCH_SIZE}" \
    --temperature "${TEMPERATURE}" \
    ${OPTIONAL_ARGS}
