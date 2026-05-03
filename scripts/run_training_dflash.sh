#!/bin/bash
# DFlash 训练启动脚本

set -e

# 自动激活虚拟环境
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
if [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
    source "${PROJECT_DIR}/.venv/bin/activate"
fi

cd "${PROJECT_DIR}"


while [[ $# -gt 0 ]]; do
    case $1 in
        --dt) DT="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        *) shift ;;
    esac
done
if [[ "$DT" != "qz" && "$DT" != "a800" ]]; then
    echo "错误: --dt 须为 qz 或 a800"
    exit 1
fi

# ========================================
# 主要训练参数
# ========================================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
if [ -n "${PET_NPROC_PER_NODE}" ]; then
    NPROC_PER_NODE="${PET_NPROC_PER_NODE}"
else
    NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
fi
NNODES="${PET_NNODES:-${NNODES:-1}}"
NODE_RANK="${PET_NODE_RANK:-${NODE_RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-29502}}"

if [ "${NNODES}" -gt 1 ] 2>/dev/null && { [ "${MASTER_ADDR}" = "127.0.0.1" ] || [ "${MASTER_ADDR}" = "localhost" ]; }; then
    echo "错误: 多机训练 (NNODES=${NNODES}) 须设置 MASTER_ADDR 或 PET_MASTER_ADDR 为可互通的主节点地址。" >&2
    exit 1
fi
export MASTER_ADDR
export MASTER_PORT

NUM_EPOCHS="${NUM_EPOCHS:-12}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
NUM_ANCHORS="${NUM_ANCHORS:-512}"
NUM_DRAFT_LAYERS="${NUM_DRAFT_LAYERS:-5}"

# ========================================
# 主要数据集参数
# ========================================
DATA_NUM_SAMPLES="${DATA_NUM_SAMPLES:-}"
ENABLE_THINKING="${ENABLE_THINKING:-on}"

# ========================================
# 默认参数（通常不需要修改）
# ========================================
TP_SIZE="${TP_SIZE:-1}"
DIST_TIMEOUT="${DIST_TIMEOUT:-3600}"

if [ "$DT" = "qz" ]; then
    DATA_NUM_SAMPLES="${DATA_NUM_SAMPLES:-400000}"
    export WANDB_MODE=offline
    TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/FlashMTP/cache/data/regen_data/nemotron_${DATA_NUM_SAMPLES}/nemotron_think_${ENABLE_THINKING}_samples_${DATA_NUM_SAMPLES}_qwen3_8b_regen.jsonl}"
    OUTPUT_DIR="${OUTPUT_DIR:-./cache/models/dflash_lsrsl_sample_${DATA_NUM_SAMPLES}_think_${ENABLE_THINKING}_nlayers${NUM_DRAFT_LAYERS}_maxlen${MAX_LENGTH}_epochs${NUM_EPOCHS}_nnodes${NNODES}}"
    TARGET_MODEL="${TARGET_MODEL:-/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/models/Qwen/Qwen3-8B}"
else
    DATA_NUM_SAMPLES="${DATA_NUM_SAMPLES:-40000}"
    TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/share/wanghanzhen/SpeculativeDecoding/NIPS26/FlashMTP_v1.1/cache/data/regen_data/nemotron_${DATA_NUM_SAMPLES}/nemotron_think_${ENABLE_THINKING}_samples_${DATA_NUM_SAMPLES}_qwen3_8b_regen.jsonl}"
    OUTPUT_DIR="${OUTPUT_DIR:-./cache/models/dflash_lsrsl_sample_${DATA_NUM_SAMPLES}_think_${ENABLE_THINKING}_nlayers${NUM_DRAFT_LAYERS}_maxlen${MAX_LENGTH}_epochs${NUM_EPOCHS}}"
    TARGET_MODEL="${TARGET_MODEL:-/share/public/public_models/Qwen3-8B}"
fi

TARGET_MODEL_BACKEND="${TARGET_MODEL_BACKEND:-hf}"
RESUME="${RESUME:-}"
CKPT_DIR="${CKPT_DIR:-}"

# 训练参数
BATCH_SIZE="${BATCH_SIZE:-1}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-6e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.04}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

EVAL_DATA_PATH="${EVAL_DATA_PATH:-}"
CACHE_DIR="${CACHE_DIR:-./cache/data/regen_data/nemotron_${DATA_NUM_SAMPLES}}"

# 模型参数
BLOCK_SIZE="${BLOCK_SIZE:-16}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flex_attention}"
STREAK_WEIGHT="${STREAK_WEIGHT:-1.0}"
CE_WEIGHT="${CE_WEIGHT:-0.2}"
STREAK_DECAY_GAMMA="${STREAK_DECAY_GAMMA:-7.0}"

# 日志和保存间隔
LOG_INTERVAL="${LOG_INTERVAL:-50}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"

# Tracker 参数
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-flashmtp-training-exp}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_DIR="${WANDB_DIR:-./wandb}"  # 离线日志保存目录
WANDB_RUN_ID="${WANDB_RUN_ID:-dflash_lsrsl_${DATA_NUM_SAMPLES}}"

# 数据参数
CHAT_TEMPLATE="${CHAT_TEMPLATE:-qwen3-thinking}"
IS_PREFORMATTED="${IS_PREFORMATTED:-}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
BUILD_DATASET_NUM_PROC="${BUILD_DATASET_NUM_PROC:-8}"


# ========================================
# 显示配置
# ========================================
echo "=========================================="
echo "DFlash 训练启动脚本"
echo "=========================================="
echo "数据特征:"
echo "  样本数量: ${DATA_NUM_SAMPLES}"
echo "  思考模式: ${ENABLE_THINKING}"
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
echo "  Streak loss权重: ${STREAK_WEIGHT}"
echo "  CE loss权重: ${CE_WEIGHT}"
echo "  Streak达标后衰减Gamma: ${STREAK_DECAY_GAMMA}"
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
echo "  NNODES: ${NNODES}"
echo "  NODE_RANK: ${NODE_RANK}"
echo "  MASTER_ADDR: ${MASTER_ADDR}"
echo "  MASTER_PORT: ${MASTER_PORT}"
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
echo "==> 开始训练 DFlash"
echo ""

# train_dflash.py 始终 init_distributed()，需 torchrun 提供 RANK/WORLD_SIZE/LOCAL_RANK
LAUNCHER=(
    torchrun
    --nproc_per_node "${NPROC_PER_NODE}"
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)

# 构建可选参数
OPTIONAL_ARGS=""

if [ -n "${EVAL_DATA_PATH}" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --eval-data-path ${EVAL_DATA_PATH}"
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
"${LAUNCHER[@]}" ./scripts/train_dflash.py \
    --target-model-path ${TARGET_MODEL} \
    --target-model-backend ${TARGET_MODEL_BACKEND} \
    --train-data-path "${TRAIN_DATA_PATH}" \
    --output-dir ${OUTPUT_DIR} \
    --cache-dir ${CACHE_DIR} \
    --num-draft-layers ${NUM_DRAFT_LAYERS} \
    --block-size ${BLOCK_SIZE} \
    --num-anchors ${NUM_ANCHORS} \
    --attention-backend ${ATTENTION_BACKEND} \
    --streak-weight ${STREAK_WEIGHT} \
    --ce-weight ${CE_WEIGHT} \
    --streak-decay-gamma ${STREAK_DECAY_GAMMA} \
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
