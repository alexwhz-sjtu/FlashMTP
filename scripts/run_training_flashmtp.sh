#!/bin/bash
# FlashMTP 训练启动脚本（单目标 CE，无 DFlash++ 的 L_dflash/L_con 等多损失）

set -e

# Fail loudly for old launch files instead of silently running them under a
# fuse/token experiment name with the fixed pivot-Q implementation.
if [[ -n "${HISTORY_MODE:-}" && "${HISTORY_MODE}" != "pivot_q" ]]; then
    echo "错误: fuse/token 历史模式已删除；当前布局固定为 pivot_q"
    exit 1
fi

# 自动激活虚拟环境
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
if [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
    source "${PROJECT_DIR}/.venv/bin/activate"
fi

cd "${PROJECT_DIR}"

# The remote .venv may be shared with another checkout. Ensure torchrun workers
# import this checkout's specforge package rather than the environment's copy.
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"


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
# 主要训练参数
# ========================================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${PET_NPROC_PER_NODE:-8}}"

NUM_EPOCHS="${NUM_EPOCHS:-6}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
SLIDING_WINDOW_SIZE="${SLIDING_WINDOW_SIZE:-64}"
CHS_NUM_LAYERS="${CHS_NUM_LAYERS:-7}"
CHS_LAYOUT_TAG="chsfirst_tokenwindow"
if [[ -n "${DRAFT_INPUT_MODE:-}" && "${DRAFT_INPUT_MODE}" != "legacy" ]]; then
    echo "错误: DRAFT_INPUT_MODE 已移除，backbone 固定为 anchor+MASK query"
    exit 1
fi
LOCAL_POSITION="${LOCAL_POSITION:-false}"
if [[ "${LOCAL_POSITION}" == "1" || "${LOCAL_POSITION}" == "true" ]]; then
    POSITION_TAG="localpos"
else
    POSITION_TAG="globalpos"
fi
NUM_ANCHORS="${NUM_ANCHORS:-512}"
ANCHOR_CHUNK_SIZE="${ANCHOR_CHUNK_SIZE:-0}"

# 恢复训练
RESUME="${RESUME:-}"
CKPT_DIR="${CKPT_DIR:-}"
RESUME_OPTIMIZER="${RESUME_OPTIMIZER:-1}"
LOAD_WEIGHTS_ONLY="${LOAD_WEIGHTS_ONLY:-0}"

# ========================================
# 主要数据集参数
# ========================================
# 数据特征参数
DATA_NUM_SAMPLES="${DATA_NUM_SAMPLES:-40000}"
ENABLE_THINKING="${ENABLE_THINKING:-off}"

# 草稿层数：默认目录名/ WandB id/ run name 中均带 nlayers${NUM_DRAFT_LAYERS}
NUM_DRAFT_LAYERS="${NUM_DRAFT_LAYERS:-5}"

# 低秩串行 head：none | vanilla | gated | rnn | rnn_easy
MARKOV_HEAD_TYPE="${MARKOV_HEAD_TYPE:-none}"
if [[ "$MARKOV_HEAD_TYPE" == "mlp" ]]; then
    echo "错误: MARKOV_HEAD_TYPE=mlp 已不再支持，请使用 none | vanilla | gated | rnn | rnn_easy"
    exit 1
fi
# additive: 修正并行 base logits；direct: head 直接产生最终 logits
MARKOV_OUTPUT_MODE="${MARKOV_OUTPUT_MODE:-additive}"
if [[ "$MARKOV_HEAD_TYPE" == "gated" && "$MARKOV_OUTPUT_MODE" == "direct" ]]; then
    echo "错误: MARKOV_HEAD_TYPE=gated 仅支持 MARKOV_OUTPUT_MODE=additive"
    exit 1
fi
MARKOV_RANK="${MARKOV_RANK:-256}"
FINAL_CE_WEIGHT="${FINAL_CE_WEIGHT:-1.0}"
TV_LOSS_WEIGHT="${TV_LOSS_WEIGHT:-1.0}"
MARKOV_TAG="mh${MARKOV_HEAD_TYPE}_${MARKOV_OUTPUT_MODE}_r${MARKOV_RANK}_ce${FINAL_CE_WEIGHT}_tv${TV_LOSS_WEIGHT}"

# ========================================
# 默认参数（通常不需要修改）
# ========================================

# GPU 设置
NNODES="${PET_NNODES:-${NNODES:-1}}"
NODE_RANK="${PET_NODE_RANK:-${NODE_RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-29501}}"

if [ "${NNODES}" -gt 1 ] 2>/dev/null && { [ "${MASTER_ADDR}" = "127.0.0.1" ] || [ "${MASTER_ADDR}" = "localhost" ]; }; then
    echo "错误: 多机训练 (NNODES=${NNODES}) 须设置 MASTER_ADDR 或 PET_MASTER_ADDR 为可互通的主节点地址。" >&2
    exit 1
fi
export MASTER_ADDR
export MASTER_PORT
TP_SIZE="${TP_SIZE:-1}"
DIST_TIMEOUT="${DIST_TIMEOUT:-1200}"

# 模型参数（OUTPUT_DIR 依赖 BLOCK_SIZE，须早于 dt 分支）
BLOCK_SIZE="${BLOCK_SIZE:-16}"
MODEL_TAG="${MODEL_TAG:-Qwen3_8B}"

if [ "$DT" = "qz" ]; then
    export WANDB_MODE=offline
    TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/FlashMTP/cache/data/regen_data/nemotron_${DATA_NUM_SAMPLES}/nemotron_think_${ENABLE_THINKING}_samples_${DATA_NUM_SAMPLES}_qwen3_8b_regen.jsonl}"
    OUTPUT_DIR="${OUTPUT_DIR:-./cache/models/flashmtp_qz_swa_w${SLIDING_WINDOW_SIZE}_chs${CHS_NUM_LAYERS}_${CHS_LAYOUT_TAG}_${POSITION_TAG}_sample_${DATA_NUM_SAMPLES}_wb_${BASE_LM_CE_WEIGHT}_nlayers${NUM_DRAFT_LAYERS}_block_${BLOCK_SIZE}_${MARKOV_TAG}_maxlen${MAX_LENGTH}_epochs${NUM_EPOCHS}_${MODEL_TAG}}"
    TARGET_MODEL="${TARGET_MODEL:-/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/models/Qwen/Qwen3-8B}"
elif [ "$DT" = "h100" ]; then
    TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/share/dai-sys/wanghanzhen/projects/MTP/training_data/nemotron_think_off_samples_40000_qwen3_8b_regen.jsonl}"
    OUTPUT_DIR="${OUTPUT_DIR:-./cache/models/flashmtp_h100_swa_w${SLIDING_WINDOW_SIZE}_chs${CHS_NUM_LAYERS}_${CHS_LAYOUT_TAG}_${POSITION_TAG}_sample_${DATA_NUM_SAMPLES}_wb_${BASE_LM_CE_WEIGHT}_nlayers${NUM_DRAFT_LAYERS}_block_${BLOCK_SIZE}_${MARKOV_TAG}_maxlen${MAX_LENGTH}_epochs${NUM_EPOCHS}_${MODEL_TAG}}"
    TARGET_MODEL="${TARGET_MODEL:-$WHZ_HOME/models/Qwen/Qwen3-8B}"
else
    TRAIN_DATA_PATH="/share/wanghanzhen/SpeculativeDecoding/NIPS26/FlashMTP_v1.1/cache/data/regen_data/nemotron_40000/nemotron_think_on_samples_40000_qwen3_8b_regen.jsonl"
    OUTPUT_DIR="${OUTPUT_DIR:-./cache/models/flashmtp_a800_swa_w${SLIDING_WINDOW_SIZE}_chs${CHS_NUM_LAYERS}_${CHS_LAYOUT_TAG}_${POSITION_TAG}_nemotron_40000_think_on_nlayers${NUM_DRAFT_LAYERS}_${MARKOV_TAG}_maxlen${MAX_LENGTH}_epochs${NUM_EPOCHS}}"
    TARGET_MODEL="${TARGET_MODEL:-/share/public/public_models/Qwen3-8B}"
fi


TARGET_MODEL_BACKEND="${TARGET_MODEL_BACKEND:-hf}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.25}"
SGLANG_MAX_TOTAL_TOKENS="${SGLANG_MAX_TOTAL_TOKENS:-}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-}"
CE_CHUNK_SIZE="${CE_CHUNK_SIZE:-2048}"

# 训练参数
BATCH_SIZE="${BATCH_SIZE:-1}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"
SHARD_DRAFT_BY_TP="${SHARD_DRAFT_BY_TP:-1}"
# Per DP rank: target sees TRAIN_BATCH_SIZE samples; each TP rank trains one draft slice.
TRAIN_BATCH_SIZE="${BATCH_SIZE}"
if [ "${TP_SIZE}" -gt 1 ] && [ "${SHARD_DRAFT_BY_TP}" = "1" ]; then
    if [ "${BATCH_SIZE}" -eq 1 ]; then
        TRAIN_BATCH_SIZE="${TP_SIZE}"
    fi
fi
LEARNING_RATE="${LEARNING_RATE:-6e-4}"
MARKOV_LR_MULTIPLIER="${MARKOV_LR_MULTIPLIER:-1.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.04}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

EVAL_DATA_PATH="${EVAL_DATA_PATH:-}"
CACHE_DIR="${CACHE_DIR:-./cache/data/regen_data/nemotron_${DATA_NUM_SAMPLES}}"

ATTENTION_BACKEND="${ATTENTION_BACKEND:-flex_attention}"
LOSS_DECAY_GAMMA="${LOSS_DECAY_GAMMA:-7}"
BASE_LM_CE_WEIGHT="${BASE_LM_CE_WEIGHT:-0}"
BASE_LM_CE_DECAY_GAMMA="${BASE_LM_CE_DECAY_GAMMA:-}"

# 日志和保存间隔
LOG_INTERVAL="${LOG_INTERVAL:-50}"
SAVE_INTERVAL="${SAVE_INTERVAL:-20000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"

# Tracker 参数
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-flashmtp-training-v2new}"
WANDB_DIR="${WANDB_DIR:-./wandb}"  # 离线日志保存目录
# 含 dt / 草稿层数 / 样本量 / 拼接方式；run id 与默认 OUTPUT_DIR 中 nlayers* 可对照
# WandB Name/Id 上限 128 字符；超长时保留前缀并追加 8 位哈希，避免训练在 wandb.init 处失败
clip_wandb_id() {
    local value="$1"
    local max_len=128
    if [ "${#value}" -le "${max_len}" ]; then
        printf '%s' "${value}"
        return
    fi
    local digest
    digest="$(printf '%s' "${value}" | sha1sum | cut -c1-8)"
    local keep=$((max_len - 9))
    printf '%s_%s' "${value:0:${keep}}" "${digest}"
}
WANDB_RUN_ID="${WANDB_RUN_ID:-flashmtp_swa_w${SLIDING_WINDOW_SIZE}_chs${CHS_NUM_LAYERS}_${CHS_LAYOUT_TAG}_${POSITION_TAG}_wb_${BASE_LM_CE_WEIGHT}_block_${BLOCK_SIZE}_${MARKOV_TAG}_n${DATA_NUM_SAMPLES}_epochs${NUM_EPOCHS}_${MODEL_TAG}}"
WANDB_NAME="${WANDB_RUN_NAME:-flashmtp_swa_w${SLIDING_WINDOW_SIZE}_chs${CHS_NUM_LAYERS}_${CHS_LAYOUT_TAG}_${POSITION_TAG}_wb_${BASE_LM_CE_WEIGHT}_block_${BLOCK_SIZE}_${MARKOV_TAG}_maxlen${MAX_LENGTH}_ep${NUM_EPOCHS}_${MODEL_TAG}}"
WANDB_RUN_ID="$(clip_wandb_id "${WANDB_RUN_ID}")"
WANDB_NAME="$(clip_wandb_id "${WANDB_NAME}")"
if [ -n "${WANDB_RUN_NAME}" ]; then
    WANDB_RUN_NAME="$(clip_wandb_id "${WANDB_RUN_NAME}")"
fi

# 数据参数
CHAT_TEMPLATE="${CHAT_TEMPLATE:-qwen}"
IS_PREFORMATTED="${IS_PREFORMATTED:-}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
BUILD_DATASET_NUM_PROC="${BUILD_DATASET_NUM_PROC:-8}"


# ========================================
# 显示配置
# ========================================
echo "=========================================="
echo "FlashMTP 训练启动脚本"
echo "=========================================="
echo "运行环境: --dt ${DT} (qz | a800 | h100)"
echo "数据特征:"
echo "  样本数量: ${DATA_NUM_SAMPLES}"
echo "  思考模式: ${ENABLE_THINKING}"
echo "  滑动窗口: W=${SLIDING_WINDOW_SIZE}"
echo "  当前 CHS: pivot embedding + S=${CHS_NUM_LAYERS} 个 hidden 层"
echo "  位置与对齐: draft ${POSITION_TAG}，target 全局位置，anchor query 不监督"
echo "------------------------------------------"
echo "目标模型: ${TARGET_MODEL}"
echo "目标模型后端: ${TARGET_MODEL_BACKEND}"
echo "训练数据: ${TRAIN_DATA_PATH}"
echo "评估数据: ${EVAL_DATA_PATH:-无}"
echo "输出目录: ${OUTPUT_DIR}"
echo "缓存目录: ${CACHE_DIR}"
echo "------------------------------------------"
echo "模型配置:"
echo "  草稿模型层数: ${NUM_DRAFT_LAYERS}"
echo "  块大小: ${BLOCK_SIZE}"
echo "  滑动窗口大小: ${SLIDING_WINDOW_SIZE}"
echo "  CHS hidden 层数: ${CHS_NUM_LAYERS}"
echo "  锚点数量: ${NUM_ANCHORS}"
echo "  锚点执行分块: ${ANCHOR_CHUNK_SIZE} (0=关闭)"
echo "  Attention后端: ${ATTENTION_BACKEND}"
echo "  Loss衰减Gamma: ${LOSS_DECAY_GAMMA:-未设置(不启用)}"
echo "  最终CE权重: ${FINAL_CE_WEIGHT}"
echo "  串行Head TV权重: ${TV_LOSS_WEIGHT}"
echo "  Base LM CE权重: ${BASE_LM_CE_WEIGHT}"
echo "  Base LM CE衰减Gamma: ${BASE_LM_CE_DECAY_GAMMA:-未设置(均匀权重)}"
echo "  串行Head: ${MARKOV_HEAD_TYPE}"
echo "  Head输出模式: ${MARKOV_OUTPUT_MODE}"
echo "  Markov维度: ${MARKOV_RANK}"
echo "------------------------------------------"
echo "训练配置:"
echo "  训练轮数: ${NUM_EPOCHS}"
echo "  批大小: ${TRAIN_BATCH_SIZE} x ${ACCUMULATION_STEPS} = $((TRAIN_BATCH_SIZE * ACCUMULATION_STEPS)) (per DP rank)"
if [ "${TP_SIZE}" -gt 1 ] && [ "${SHARD_DRAFT_BY_TP}" = "1" ]; then
    echo "  shard-draft-by-tp: on (${TP_SIZE} target samples / TP group, 1 draft sample / TP rank)"
fi
echo "  Backbone学习率: ${LEARNING_RATE}"
echo "  Markov Head学习率倍率: ${MARKOV_LR_MULTIPLIER}"
echo "  最大长度: ${MAX_LENGTH}"
echo "  预热比例: ${WARMUP_RATIO}"
echo "  梯度裁剪: ${MAX_GRAD_NORM}"
echo "------------------------------------------"
echo "分布式配置:"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "  NNODES: ${NNODES}"
echo "  NODE_RANK: ${NODE_RANK}"
echo "  MASTER_ADDR: ${MASTER_ADDR}"
echo "  MASTER_PORT: ${MASTER_PORT}"
echo "  TP_SIZE: ${TP_SIZE}"
echo "------------------------------------------"
echo "Tracker: ${REPORT_TO}"
if [ "${REPORT_TO}" = "wandb" ]; then
    echo "  WandB目录: ${WANDB_DIR}"
    if [ -n "${WANDB_RUN_NAME}" ]; then
        echo "  WandB run 名称: ${WANDB_RUN_NAME}"
    fi
    if [ -n "${WANDB_RUN_ID}" ]; then
        echo "  WandB run id: ${WANDB_RUN_ID} (离线: offline-run-${WANDB_RUN_ID})"
    fi
fi
echo "=========================================="
echo ""

# 如果输出目录已存在，自动添加数字后缀
original_output_dir="${OUTPUT_DIR}"
suffix=1
while [ "${NNODES}" -le 1 ] 2>/dev/null && [ -d "${OUTPUT_DIR}" ] && [ -n "$(ls -A "${OUTPUT_DIR}" 2>/dev/null)" ] && [ -z "${CKPT_DIR}" ] && [ -z "${RESUME}" ]; do
    OUTPUT_DIR="${original_output_dir}_${suffix}"
    suffix=$((suffix + 1))
done
if [ "${OUTPUT_DIR}" != "${original_output_dir}" ]; then
    echo "警告: 输出目录 ${original_output_dir} 已存在且非空，自动切换到: ${OUTPUT_DIR}"
fi

# 创建输出目录
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CACHE_DIR}
mkdir -p ${WANDB_DIR}

# ========================================
# 训练
# ========================================
echo ""
echo "==> 开始训练 FlashMTP"
echo ""

# train_flashmtp.py 始终 init_distributed()，需 torchrun 提供 RANK/WORLD_SIZE/LOCAL_RANK
LAUNCHER=(
    "${PROJECT_DIR}/.venv/bin/python"
    -m
    torch.distributed.run
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --nproc_per_node "${NPROC_PER_NODE}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)

# 构建可选参数
OPTIONAL_ARGS=""

if [ -n "${EVAL_DATA_PATH}" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --eval-data-path ${EVAL_DATA_PATH}"
fi

if [ -n "${LOSS_DECAY_GAMMA}" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --loss-decay-gamma ${LOSS_DECAY_GAMMA}"
fi

if [[ "${LOCAL_POSITION}" == "1" || "${LOCAL_POSITION}" == "true" ]]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --local-position"
fi

if awk "BEGIN {exit !(${BASE_LM_CE_WEIGHT} > 0)}"; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --base-lm-ce-weight ${BASE_LM_CE_WEIGHT}"
    if [ -n "${BASE_LM_CE_DECAY_GAMMA}" ]; then
        OPTIONAL_ARGS="${OPTIONAL_ARGS} --base-lm-ce-decay-gamma ${BASE_LM_CE_DECAY_GAMMA}"
    fi
fi

if [ -n "${IS_PREFORMATTED}" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --is-preformatted"
fi

if [ -n "${RESUME}" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --resume"
fi

if [ -n "${CKPT_DIR}" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --ckpt-dir ${CKPT_DIR}"
fi

if [ "${RESUME_OPTIMIZER}" = "0" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --no-resume-optimizer"
fi

if [ "${LOAD_WEIGHTS_ONLY}" = "1" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --load-weights-only"
fi

if [ "${REPORT_TO}" != "none" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --report-to ${REPORT_TO}"
    if [ "${REPORT_TO}" = "wandb" ] && [ -n "${WANDB_PROJECT}" ]; then
        OPTIONAL_ARGS="${OPTIONAL_ARGS} --wandb-project ${WANDB_PROJECT}"
    fi
    if [ -n "${WANDB_RUN_NAME}" ]; then
        OPTIONAL_ARGS="${OPTIONAL_ARGS} --wandb-name ${WANDB_RUN_NAME}"
    elif [ -n "${WANDB_NAME}" ]; then
        OPTIONAL_ARGS="${OPTIONAL_ARGS} --wandb-name ${WANDB_NAME}"
    fi
    if [ -n "${WANDB_RUN_ID}" ]; then
        OPTIONAL_ARGS="${OPTIONAL_ARGS} --wandb-run-id ${WANDB_RUN_ID}"
    fi
fi

if [ "${TARGET_MODEL_BACKEND}" = "sglang" ]; then
    # SGLang profiles KV pool as: free_mem_after_weights - pre_load_mem * (1 - mem_fraction).
    # With ~14B weights on 80GB H100, mem_fraction < ~0.21 yields negative KV capacity.
    if awk "BEGIN {exit !(${SGLANG_MEM_FRACTION_STATIC} < 0.22)}"; then
        echo "WARNING: SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC} is too low for"
        echo "  SGLang KV profiling (need >=0.22 for MAX_LENGTH=${MAX_LENGTH})."
        echo "  Override with SGLANG_MEM_FRACTION_STATIC=0.25 (script default)."
    fi
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION_STATIC}"
    if [ -z "${SGLANG_MAX_TOTAL_TOKENS}" ]; then
        SGLANG_MAX_TOTAL_TOKENS=$((TRAIN_BATCH_SIZE * MAX_LENGTH))
    fi
    if [ -z "${SGLANG_MAX_RUNNING_REQUESTS}" ]; then
        SGLANG_MAX_RUNNING_REQUESTS=${TRAIN_BATCH_SIZE}
    fi
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --sglang-max-total-tokens ${SGLANG_MAX_TOTAL_TOKENS}"
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --sglang-max-running-requests ${SGLANG_MAX_RUNNING_REQUESTS}"
fi

if [ "${TP_SIZE}" -gt 1 ] && [ "${SHARD_DRAFT_BY_TP}" = "1" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --shard-draft-by-tp"
else
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --no-shard-draft-by-tp"
fi

OPTIONAL_ARGS="${OPTIONAL_ARGS} --ce-chunk-size ${CE_CHUNK_SIZE}"

# 运行训练
EXIT_CODE=0
"${LAUNCHER[@]}" ./scripts/train_flashmtp.py \
    --target-model-path ${TARGET_MODEL} \
    --target-model-backend ${TARGET_MODEL_BACKEND} \
    --train-data-path "${TRAIN_DATA_PATH}" \
    --output-dir ${OUTPUT_DIR} \
    --cache-dir ${CACHE_DIR} \
    --num-draft-layers ${NUM_DRAFT_LAYERS} \
    --block-size ${BLOCK_SIZE} \
    --num-anchors ${NUM_ANCHORS} \
    --anchor-chunk-size ${ANCHOR_CHUNK_SIZE} \
    --attention-backend ${ATTENTION_BACKEND} \
    --learning-rate ${LEARNING_RATE} \
    --markov-lr-multiplier ${MARKOV_LR_MULTIPLIER} \
    --warmup-ratio ${WARMUP_RATIO} \
    --num-epochs ${NUM_EPOCHS} \
    --batch-size ${TRAIN_BATCH_SIZE} \
    --accumulation-steps ${ACCUMULATION_STEPS} \
    --max-grad-norm ${MAX_GRAD_NORM} \
    --max-length ${MAX_LENGTH} \
    --log-interval ${LOG_INTERVAL} \
    --save-interval ${SAVE_INTERVAL} \
    --eval-interval ${EVAL_INTERVAL} \
    --chat-template ${CHAT_TEMPLATE} \
    --dataloader-num-workers ${DATALOADER_NUM_WORKERS} \
    --build-dataset-num-proc ${BUILD_DATASET_NUM_PROC} \
    --tp-size ${TP_SIZE} \
    --dist-timeout ${DIST_TIMEOUT} \
    --sliding-window-size ${SLIDING_WINDOW_SIZE} \
    --chs-num-layers ${CHS_NUM_LAYERS} \
    --markov-head-type ${MARKOV_HEAD_TYPE} \
    --markov-output-mode ${MARKOV_OUTPUT_MODE} \
    --markov-rank ${MARKOV_RANK} \
    --final-ce-weight ${FINAL_CE_WEIGHT} \
    --tv-loss-weight ${TV_LOSS_WEIGHT} \
    --seed 42 \
    ${OPTIONAL_ARGS} 2>&1 || EXIT_CODE=$?

# 检查训练是否成功
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "=========================================="
    echo "训练失败 (退出码: $EXIT_CODE)"
    echo "=========================================="
    exit $EXIT_CODE
fi

# ========================================
# 训练完成
# ========================================
echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo "模型保存在: ${OUTPUT_DIR}"
echo ""
echo "使用示例："
echo "  from specforge.modeling.draft.flashmtp import FlashMTPDraftModel"
echo "  draft_model = FlashMTPDraftModel.from_pretrained('${OUTPUT_DIR}/epoch_${NUM_EPOCHS}_step_<step>')"
echo ""
echo "运行推理："
echo "  DRAFT_NAME_OR_PATH=${OUTPUT_DIR}/epoch_${NUM_EPOCHS}_step_<step> DATASET=gsm8k \\"
echo "  bash evaluation/run_benchmark_flashmtp.sh --dt ${DT}"
echo "=========================================="
