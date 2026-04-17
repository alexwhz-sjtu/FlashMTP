#!/bin/bash
# DFlash 训练启动脚本

set -e

# 解析命令行参数
DT="a800"  # 默认值为 a800
while [[ $# -gt 0 ]]; do
    case $1 in
        --dt)
            DT="$2"
            shift 2
            ;;
        *)
            # 保留其他参数供后续使用（如果有的话）
            shift
            ;;
    esac
done

# 验证 dt 参数
if [[ "$DT" != "qz" && "$DT" != "a800" ]]; then
    echo "错误: --dt 参数必须是 'qz' 或 'a800'"
    exit 1
fi

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

NUM_EPOCHS="${NUM_EPOCHS:-10}"
MAX_LENGTH="${MAX_LENGTH:-10240}"
CHS_CONCAT_MODE="${CHS_CONCAT_MODE:-feature}"
# CHS 滑动窗口：取 anchor 之前最近 W 个 target 位置的 hidden（含 position_ids）；1 与旧版单点上下文一致
CHS_WINDOW_SIZE="${CHS_WINDOW_SIZE:-64}"
NUM_ANCHORS="${NUM_ANCHORS:-1024}"

# 恢复训练
RESUME="${RESUME:-}"
CKPT_DIR="${CKPT_DIR:-}"

# ========================================
# 主要数据集参数
# ========================================
# 数据特征参数
DATA_NUM_SAMPLES="${DATA_NUM_SAMPLES:-40000}"
ENABLE_THINKING="${ENABLE_THINKING:-on}"

# ========================================
# 默认参数（通常不需要修改）
# ========================================

# GPU 设置
MASTER_PORT="${MASTER_PORT:-29501}"
TP_SIZE="${TP_SIZE:-1}"
DIST_TIMEOUT="${DIST_TIMEOUT:-3600}"

# 训练参数
BATCH_SIZE="${BATCH_SIZE:-1}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-6e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.04}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

# 损失: ce=交叉熵; kl=相对目标模型 last-hidden 的 KL 蒸馏（需 HF 等返回完整 hidden_states）
# ce_kl= CE 与 top-k KL 加权: 总损失 = CE_LOSS_WEIGHT * CE + KL_LOSS_WEIGHT * KL（系数为 0 的分支不计算）
FLASHMTP_LOSS_TYPE="${FLASHMTP_LOSS_TYPE:-ce}"
CE_LOSS_WEIGHT="${CE_LOSS_WEIGHT:-1}"
KL_LOSS_WEIGHT="${KL_LOSS_WEIGHT:-0.2}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-1.0}"
# KL 蒸馏时只对齐教师 top-k logits；<=0 表示全词表
KL_TOPK="${KL_TOPK:-20}"

# 数据目录 - 根据 --dt 参数选择配置
if [ "$DT" = "qz" ]; then
    # qz 配置
    TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/FlashMTP/cache/data/regen_data/nemotron_${DATA_NUM_SAMPLES}/nemotron_think_${ENABLE_THINKING}_samples_${DATA_NUM_SAMPLES}_qwen3_8b_regen.jsonl}"
    OUTPUT_DIR="${OUTPUT_DIR:-./cache/models/flashmtp_v1.3_swa_${CHS_WINDOW_SIZE}_loss_${FLASHMTP_LOSS_TYPE}_${CHS_CONCAT_MODE}_sample_${DATA_NUM_SAMPLES}_think_${ENABLE_THINKING}_qwen3_8b_maxlen${MAX_LENGTH}_epoch${NUM_EPOCHS}}"
    TARGET_MODEL="${TARGET_MODEL:-$WHZ_DIR/models/Qwen/Qwen3-8B}"
else
    # a800 配置（默认）
    TRAIN_DATA_PATH="/share/wanghanzhen/SpeculativeDecoding/NIPS26/FlashMTP_v1.1/cache/data/regen_data/nemotron_40000/nemotron_think_on_samples_40000_qwen3_8b_regen.jsonl"
    OUTPUT_DIR="./cache/models/flashmtp_v1.3_swa_${CHS_WINDOW_SIZE}_loss_${FLASHMTP_LOSS_TYPE}_${CHS_CONCAT_MODE}_sample_${DATA_NUM_SAMPLES}_think_${ENABLE_THINKING}_qwen3_8b_maxlen${MAX_LENGTH}_epoch${NUM_EPOCHS}}"
    TARGET_MODEL="${TARGET_MODEL:-/share/public/public_models/Qwen3-8B}"
fi


TARGET_MODEL_BACKEND="${TARGET_MODEL_BACKEND:-hf}"

EVAL_DATA_PATH="${EVAL_DATA_PATH:-}"
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
WANDB_PROJECT="${WANDB_PROJECT:-flashmtp_training}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_DIR="${WANDB_DIR:-./wandb}"
# offline: 仅本地写入 ${WANDB_DIR}，无需 API key；上线同步: WANDB_MODE=online 并配置密钥
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_RUN_ID="${WANDB_RUN_ID:-flashmtp_v1.3_loss_${FLASHMTP_LOSS_TYPE}_${CHS_CONCAT_MODE}_sample_${DATA_NUM_SAMPLES}_think_${ENABLE_THINKING}_qwen3_8b_maxlen${MAX_LENGTH}_epoch${NUM_EPOCHS}}"

export WANDB_DIR
export WANDB_MODE

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
echo "运行环境: ${DT}"
echo "------------------------------------------"
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
echo "  CHS 窗口 W: ${CHS_WINDOW_SIZE}"
echo "  锚点数量: ${NUM_ANCHORS}"
echo "  Attention后端: ${ATTENTION_BACKEND}"
echo "  Loss衰减Gamma: ${LOSS_DECAY_GAMMA:-未设置(不启用)}"
echo "  损失类型: ${FLASHMTP_LOSS_TYPE} (ce|kl|ce_kl)"
echo "  CE 系数: ${CE_LOSS_WEIGHT} (ce_kl 时有效；纯 kl 时不使用)"
echo "  KL 系数 w: ${KL_LOSS_WEIGHT} (ce_kl 时有效；纯 ce 时不使用)"
echo "  蒸馏温度T: ${DISTILL_TEMPERATURE} (kl / ce_kl 中 KL 项有效)"
echo "  KL top-k: ${KL_TOPK} (kl / ce_kl 中 KL 项；<=0 为全词表)"
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
    echo "  WANDB_MODE: ${WANDB_MODE}"
    echo "  WandB目录: ${WANDB_DIR}"
    if [ -n "${WANDB_RUN_ID}" ]; then
        echo "  WandB运行ID: ${WANDB_RUN_ID}"
    fi
    if [ "${WANDB_MODE}" = "offline" ]; then
        echo "  (离线) 结束后可: wandb sync ${WANDB_DIR}/offline-run-*"
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

# train_flashmtp 会 init_process_group，需 torchrun 注入 RANK/WORLD_SIZE（含单卡）
LAUNCHER=(torchrun --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}")

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
    --chs-window-size ${CHS_WINDOW_SIZE} \
    --flashmtp-loss-type ${FLASHMTP_LOSS_TYPE} \
    --ce-loss-weight ${CE_LOSS_WEIGHT} \
    --kl-loss-weight ${KL_LOSS_WEIGHT} \
    --distill-temperature ${DISTILL_TEMPERATURE} \
    --kl-topk ${KL_TOPK} \
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
