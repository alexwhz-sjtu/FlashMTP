#!/bin/bash
# DFlash 训练启动脚本

set -e

# 自动激活虚拟环境
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
if [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
    source "${PROJECT_DIR}/.venv/bin/activate"
fi

# ========================================
# 主要训练参数
# ========================================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

NUM_EPOCHS="${NUM_EPOCHS:-6}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
CHS_CONCAT_MODE="${CHS_CONCAT_MODE:-feature}"
NUM_ANCHORS="${NUM_ANCHORS:-512}"

# 恢复训练
RESUME="${RESUME:-}"
CKPT_DIR="${CKPT_DIR:-}"

# ========================================
# 主要数据集参数
# ========================================
# 数据特征参数
DATA_NUM_SAMPLES="${DATA_NUM_SAMPLES:-400000}"
ENABLE_THINKING="${ENABLE_THINKING:-on}"

# ========================================
# 默认参数（通常不需要修改）
# ========================================

# GPU 设置
MASTER_PORT="${MASTER_PORT:-29501}"
TP_SIZE="${TP_SIZE:-1}"
DIST_TIMEOUT="${DIST_TIMEOUT:-3600}"

# 目标模型路径
TARGET_MODEL="${TARGET_MODEL:-$WHZ_DIR/models/Qwen/Qwen3-8B}"
TARGET_MODEL_BACKEND="${TARGET_MODEL_BACKEND:-hf}"

# 训练参数
BATCH_SIZE="${BATCH_SIZE:-1}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-6e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.04}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

# 数据目录
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/FlashMTP/cache/data/regen_data/nemotron_400000_len_4096/nemotron_think_400000_train_regen.jsonl}"

# TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-./cache/data/regen_data/nemotron_${DATA_NUM_SAMPLES}/nemotron_think_${ENABLE_THINKING}_samples_${DATA_NUM_SAMPLES}_qwen3_8b_regen.jsonl}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-./cache/models/flashmtp_${CHS_CONCAT_MODE}_sample_${DATA_NUM_SAMPLES}_think_${ENABLE_THINKING}_qwen3_8b_maxlen${MAX_LENGTH}}"
CACHE_DIR="${CACHE_DIR:-./cache/data/regen_data/nemotron_${DATA_NUM_SAMPLES}}"

# 模型参数
NUM_DRAFT_LAYERS="${NUM_DRAFT_LAYERS:-5}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flex_attention}"
LOSS_DECAY_GAMMA="${LOSS_DECAY_GAMMA:-7}"

# 日志和保存间隔
LOG_INTERVAL="${LOG_INTERVAL:-50}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"

# Tracker 参数
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-flashmtp-training}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_DIR="${WANDB_DIR:-./wandb}"  # 离线日志保存目录
WANDB_RUN_ID="${WANDB_RUN_ID:-flashmtp_${DATA_NUM_SAMPLES}_${CHS_CONCAT_MODE}}"   # 离线子目录名称 (如: my_run_001，生成 offline-run-my_run_001)

# 数据参数
CHAT_TEMPLATE="${CHAT_TEMPLATE:-qwen3-thinking}"
IS_PREFORMATTED="${IS_PREFORMATTED:-}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
BUILD_DATASET_NUM_PROC="${BUILD_DATASET_NUM_PROC:-8}"


# ========================================
# 显示配置
# ========================================
echo "=========================================="
echo "FlashMTP 训练启动脚本"
echo "=========================================="
echo "数据特征:"
echo "  样本数量: ${DATA_NUM_SAMPLES}"
echo "  思考模式: ${ENABLE_THINKING}"
echo "  数据子目录: ${CHS_CONCAT_MODE}"
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
echo "  锚点数量: ${NUM_ANCHORS}"
echo "  Attention后端: ${ATTENTION_BACKEND}"
echo "  Loss衰减Gamma: ${LOSS_DECAY_GAMMA:-未设置(不启用)}"
echo "------------------------------------------"
echo "训练配置:"
echo "  训练轮数: ${NUM_EPOCHS}"
echo "  批大小: ${BATCH_SIZE} x ${ACCUMULATION_STEPS} = $((BATCH_SIZE * ACCUMULATION_STEPS))"
echo "  学习率: ${LEARNING_RATE}"
echo "  最大长度: ${MAX_LENGTH}"
echo "  预热比例: ${WARMUP_RATIO}"
echo "  梯度裁剪: ${MAX_GRAD_NORM}"
echo "------------------------------------------"
echo "分布式配置:"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "  TP_SIZE: ${TP_SIZE}"
echo "------------------------------------------"
echo "Tracker: ${REPORT_TO}"
if [ "${REPORT_TO}" = "wandb" ]; then
    echo "  WandB目录: ${WANDB_DIR}"
    if [ -n "${WANDB_RUN_ID}" ]; then
        echo "  WandB运行ID: ${WANDB_RUN_ID} (离线子目录: offline-run-${WANDB_RUN_ID})"
    fi
fi
echo "=========================================="
echo ""

# 如果输出目录已存在，自动添加数字后缀
original_output_dir="${OUTPUT_DIR}"
suffix=1
while [ -d "${OUTPUT_DIR}" ] && [ -n "$(ls -A "${OUTPUT_DIR}" 2>/dev/null)" ]; do
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

if [ "${NPROC_PER_NODE}" -gt 1 ]; then
    LAUNCHER=(torchrun --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}")
else
    LAUNCHER=(python)
fi

# 构建可选参数
OPTIONAL_ARGS=""

if [ -n "${EVAL_DATA_PATH}" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --eval-data-path ${EVAL_DATA_PATH}"
fi

if [ -n "${LOSS_DECAY_GAMMA}" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --loss-decay-gamma ${LOSS_DECAY_GAMMA}"
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

if [ "${REPORT_TO}" != "none" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --report-to ${REPORT_TO}"
    if [ "${REPORT_TO}" = "wandb" ] && [ -n "${WANDB_PROJECT}" ]; then
        OPTIONAL_ARGS="${OPTIONAL_ARGS} --wandb-project ${WANDB_PROJECT}"
    fi
    if [ -n "${WANDB_RUN_NAME}" ]; then
        OPTIONAL_ARGS="${OPTIONAL_ARGS} --wandb-run-name ${WANDB_RUN_NAME}"
    fi
    if [ -n "${WANDB_RUN_ID}" ]; then
        OPTIONAL_ARGS="${OPTIONAL_ARGS} --wandb-run-id ${WANDB_RUN_ID}"
    fi
fi

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
    --attention-backend ${ATTENTION_BACKEND} \
    --learning-rate ${LEARNING_RATE} \
    --warmup-ratio ${WARMUP_RATIO} \
    --num-epochs ${NUM_EPOCHS} \
    --batch-size ${BATCH_SIZE} \
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
    --chs-concat-mode ${CHS_CONCAT_MODE} \
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
echo "  from specforge.modeling.draft.dflash import DFlashDraftModel"
echo "  draft_model = DFlashDraftModel.from_pretrained('${OUTPUT_DIR}/epoch_${NUM_EPOCHS}_step_<step>')"
echo ""
echo "运行推理："
echo "  python benchmark.py --draft-model ${OUTPUT_DIR}/epoch_${NUM_EPOCHS}_step_<step>"
echo "=========================================="
