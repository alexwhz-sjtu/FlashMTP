#!/bin/bash
# FlashMTP 训练启动脚本（单目标 CE，无 DFlash++ 的 L_dflash/L_con 等多损失）

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
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

NUM_EPOCHS="${NUM_EPOCHS:-6}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
CHS_CONCAT_MODE="${CHS_CONCAT_MODE:-feature}"
PIVOT_FUSE_MODE="${PIVOT_FUSE_MODE:-prefix_condition}"
NUM_MIDDLE_LAYERS_N="${NUM_MIDDLE_LAYERS_N:-5}"
NUM_ANCHORS="${NUM_ANCHORS:-512}"

# 恢复训练
RESUME="${RESUME:-}"
CKPT_DIR="${CKPT_DIR:-}"
INIT_CKPT_DIR="${INIT_CKPT_DIR:-}"
if [ -n "${INIT_CKPT_DIR}" ] && { [ -n "${CKPT_DIR}" ] || [ -n "${RESUME}" ]; }; then
    echo "错误: INIT_CKPT_DIR 是只加载权重的新训练入口，不能和 CKPT_DIR/RESUME 同时使用"
    exit 1
fi

# ========================================
# 主要数据集参数
# ========================================
# 数据特征参数
DATA_NUM_SAMPLES="${DATA_NUM_SAMPLES:-40000}"
ENABLE_THINKING="${ENABLE_THINKING:-off}"

# 草稿层数：默认目录名/ WandB id/ run name 中均带 nlayers${NUM_DRAFT_LAYERS}
NUM_DRAFT_LAYERS="${NUM_DRAFT_LAYERS:-5}"

# 草稿块内 position_ids：CHS RoPE 前缀全 0，draft 为 1..block_size（默认 false 为全局 anchor 位置）
LOCAL_POSITION="${LOCAL_POSITION:-true}"
LOCAL_POSITION_TAG="lp0"
case "$(echo "${LOCAL_POSITION}" | tr '[:upper:]' '[:lower:]')" in
    true|1|yes) LOCAL_POSITION_TAG="lp1" ;;
esac

# Teacher-match loss cap：p_draft/max(p_teacher,0.5)>1 时将该位置 CE 权重压到块尾槽（默认关闭）
LOSS_TEACHER_MATCH_CAP="${LOSS_TEACHER_MATCH_CAP:-false}"
TEACHER_MATCH_CAP_TAG="tmc0"
case "$(echo "${LOSS_TEACHER_MATCH_CAP}" | tr '[:upper:]' '[:lower:]')" in
    true|1|yes) TEACHER_MATCH_CAP_TAG="tmc1" ;;
esac

# 首个预测 token：draft 末层 hidden 与 target 末层 hidden 的 MSE，权重 w1_mse（0 关闭）
W1_MSE="${W1_MSE:-0}"
W1_MSE_TAG="w1mse0"
if awk "BEGIN {exit !(${W1_MSE} > 0)}"; then
    W1_MSE_TAG="w1mse${W1_MSE}"
fi

# DFlash 两阶段蒸馏：none | stage1 | stage2（默认关闭）
DFLASH_TEACHER_PATH="${DFLASH_TEACHER_PATH:-}"
DFLASH_DISTILL_STAGE="${DFLASH_DISTILL_STAGE:-none}"
DFLASH_DISTILL_WEIGHT="${DFLASH_DISTILL_WEIGHT:-1.0}"
DFLASH_DISTILL_TEMPERATURE="${DFLASH_DISTILL_TEMPERATURE:-2.0}"
DFLASH_DISTILL_TOP_K="${DFLASH_DISTILL_TOP_K:-128}"
DFLASH_STAGE2_CE_GATE="${DFLASH_STAGE2_CE_GATE:-all}"
BASE_LOSS_DECAY_GAMMA="${LOSS_DECAY_GAMMA:-7}"
DFLASH_STAGE1_LOSS_DECAY_GAMMA="${DFLASH_STAGE1_LOSS_DECAY_GAMMA:-${LOSS_DECAY_GAMMA:-14}}"
DFLASH_STAGE2_LOSS_DECAY_GAMMA="${DFLASH_STAGE2_LOSS_DECAY_GAMMA:-${LOSS_DECAY_GAMMA:-7}}"
case "${DFLASH_DISTILL_STAGE}" in
    stage1)
        EFFECTIVE_LOSS_DECAY_GAMMA="${DFLASH_STAGE1_LOSS_DECAY_GAMMA}"
        ;;
    stage2)
        EFFECTIVE_LOSS_DECAY_GAMMA="${DFLASH_STAGE2_LOSS_DECAY_GAMMA}"
        ;;
    *)
        EFFECTIVE_LOSS_DECAY_GAMMA="${BASE_LOSS_DECAY_GAMMA}"
        ;;
esac

DFLASH_DISTILL_TAG="dnone"
if [ "${DFLASH_DISTILL_STAGE}" != "none" ]; then
    DFLASH_DISTILL_TAG="d${DFLASH_DISTILL_STAGE}_dklw${DFLASH_DISTILL_WEIGHT}_top${DFLASH_DISTILL_TOP_K}_g${EFFECTIVE_LOSS_DECAY_GAMMA}"
    if [ "${DFLASH_DISTILL_STAGE}" = "stage2" ]; then
        DFLASH_DISTILL_TAG="${DFLASH_DISTILL_TAG}_ce${DFLASH_STAGE2_CE_GATE}"
    fi
fi

# ========================================
# 默认参数（通常不需要修改）
# ========================================

# GPU 设置
MASTER_PORT="${MASTER_PORT:-29501}"
TP_SIZE="${TP_SIZE:-1}"
DIST_TIMEOUT="${DIST_TIMEOUT:-3600}"

# 模型参数（OUTPUT_DIR 依赖 BLOCK_SIZE，须早于 dt 分支）
BLOCK_SIZE="${BLOCK_SIZE:-16}"

if [ "$DT" = "qz" ]; then
    # export NNODES=2
    # export NODE_RANK=${RANK:-0}
    export WANDB_MODE=offline
    TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/FlashMTP/cache/data/regen_data/nemotron_${DATA_NUM_SAMPLES}/nemotron_think_${ENABLE_THINKING}_samples_${DATA_NUM_SAMPLES}_qwen3_8b_regen.jsonl}"
    OUTPUT_DIR="${OUTPUT_DIR:-./cache/models/flashmtp_qz_${PIVOT_FUSE_MODE}_fuse${NUM_MIDDLE_LAYERS_N}_${CHS_CONCAT_MODE}_sample_${DATA_NUM_SAMPLES}_think_${ENABLE_THINKING}_nlayers${NUM_DRAFT_LAYERS}_block_${BLOCK_SIZE}_maxlen${MAX_LENGTH}_epochs${NUM_EPOCHS}_${LOCAL_POSITION_TAG}_${TEACHER_MATCH_CAP_TAG}_${W1_MSE_TAG}_${DFLASH_DISTILL_TAG}}"
    TARGET_MODEL="${TARGET_MODEL:-/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/models/Qwen/Qwen3-8B}"
elif [ "$DT" = "h100" ]; then
    TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-../training_data/regen_data/nemotron_${DATA_NUM_SAMPLES}/nemotron_think_${ENABLE_THINKING}_samples_${DATA_NUM_SAMPLES}_qwen3_8b_regen.jsonl}"
    OUTPUT_DIR="${OUTPUT_DIR:-./cache/models/flashmtp_h100_${PIVOT_FUSE_MODE}_fuse$((NUM_MIDDLE_LAYERS_N + 2))_sample_${DATA_NUM_SAMPLES}_think_${ENABLE_THINKING}_nlayers${NUM_DRAFT_LAYERS}_block_${BLOCK_SIZE}_maxlen${MAX_LENGTH}_epochs${NUM_EPOCHS}_${LOCAL_POSITION_TAG}_${TEACHER_MATCH_CAP_TAG}_${W1_MSE_TAG}_${DFLASH_DISTILL_TAG}}"
    TARGET_MODEL="${TARGET_MODEL:-$WHZ_DIR/models/Qwen/Qwen3-8B}"
else
    TRAIN_DATA_PATH="/share/wanghanzhen/SpeculativeDecoding/NIPS26/FlashMTP_v1.1/cache/data/regen_data/nemotron_40000/nemotron_think_on_samples_40000_qwen3_8b_regen.jsonl"
    OUTPUT_DIR="${OUTPUT_DIR:-./cache/models/flashmtp_a800_${PIVOT_FUSE_MODE}_fuse${NUM_MIDDLE_LAYERS_N}_nemotron_40000_think_on_nlayers${NUM_DRAFT_LAYERS}_maxlen${MAX_LENGTH}_epochs${NUM_EPOCHS}_${LOCAL_POSITION_TAG}_${TEACHER_MATCH_CAP_TAG}_${W1_MSE_TAG}_${DFLASH_DISTILL_TAG}}"
    TARGET_MODEL="${TARGET_MODEL:-/share/public/public_models/Qwen3-8B}"
fi


TARGET_MODEL_BACKEND="${TARGET_MODEL_BACKEND:-hf}"

# 训练参数
BATCH_SIZE="${BATCH_SIZE:-1}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-6e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.04}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

EVAL_DATA_PATH="${EVAL_DATA_PATH:-}"
CACHE_DIR="${CACHE_DIR:-./cache/data/regen_data/nemotron_${DATA_NUM_SAMPLES}}"

ATTENTION_BACKEND="${ATTENTION_BACKEND:-flex_attention}"
# teacher-match cap 开关见上方 LOSS_TEACHER_MATCH_CAP（默认 false）

# 日志和保存间隔
LOG_INTERVAL="${LOG_INTERVAL:-50}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"

# Tracker 参数
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-flashmtp-training-exp}"
WANDB_DIR="${WANDB_DIR:-./wandb}"  # 离线日志保存目录
# 含 dt / 草稿层数 / 样本量 / 拼接方式；run id 与默认 OUTPUT_DIR 中 nlayers* 可对照
WANDB_RUN_ID="${WANDB_RUN_ID:-flashmtp_${DT}_${PIVOT_FUSE_MODE}_n${NUM_MIDDLE_LAYERS_N}_nlayers${NUM_DRAFT_LAYERS}_block_${BLOCK_SIZE}_n${DATA_NUM_SAMPLES}_${CHS_CONCAT_MODE}_epochs${NUM_EPOCHS}_${LOCAL_POSITION_TAG}_${TEACHER_MATCH_CAP_TAG}_${W1_MSE_TAG}_${DFLASH_DISTILL_TAG}}"
WANDB_NAME="${WANDB_RUN_NAME:-flashmtp_${DT}_${PIVOT_FUSE_MODE}_n${NUM_MIDDLE_LAYERS_N}_nlayers${NUM_DRAFT_LAYERS}_maxlen${MAX_LENGTH}_ep${NUM_EPOCHS}_${CHS_CONCAT_MODE}_${LOCAL_POSITION_TAG}_${TEACHER_MATCH_CAP_TAG}_${W1_MSE_TAG}_${DFLASH_DISTILL_TAG}}"

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
echo "  数据子目录: ${CHS_CONCAT_MODE}"
echo "  Pivot 融合: ${PIVOT_FUSE_MODE} (中间层数 N=${NUM_MIDDLE_LAYERS_N})"
echo "  local_position: ${LOCAL_POSITION} (tag ${LOCAL_POSITION_TAG}; draft 1..block, CHS rope 0)"
echo "  loss_teacher_match_cap: ${LOSS_TEACHER_MATCH_CAP} (tag ${TEACHER_MATCH_CAP_TAG})"
echo "  w1_mse: ${W1_MSE} (tag ${W1_MSE_TAG}; first-pred hidden MSE weight)"
echo "  dflash_distill_stage: ${DFLASH_DISTILL_STAGE} (tag ${DFLASH_DISTILL_TAG})"
if [ "${DFLASH_DISTILL_STAGE}" != "none" ]; then
    echo "  dflash_teacher_path: ${DFLASH_TEACHER_PATH}"
    echo "  dflash_distill: weight=${DFLASH_DISTILL_WEIGHT}, temperature=${DFLASH_DISTILL_TEMPERATURE}, top_k=${DFLASH_DISTILL_TOP_K}"
    echo "  dflash_stage2_ce_gate: ${DFLASH_STAGE2_CE_GATE}"
    echo "  dflash_stage1_loss_decay_gamma: ${DFLASH_STAGE1_LOSS_DECAY_GAMMA}"
    echo "  dflash_stage2_loss_decay_gamma: ${DFLASH_STAGE2_LOSS_DECAY_GAMMA}"
fi
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
echo "  Loss衰减Gamma: ${EFFECTIVE_LOSS_DECAY_GAMMA:-未设置(不启用)}"
echo "------------------------------------------"
echo "训练配置:"
echo "  训练轮数: ${NUM_EPOCHS}"
echo "  批大小: ${BATCH_SIZE} x ${ACCUMULATION_STEPS} = $((BATCH_SIZE * ACCUMULATION_STEPS))"
echo "  学习率: ${LEARNING_RATE}"
echo "  最大长度: ${MAX_LENGTH}"
echo "  预热比例: ${WARMUP_RATIO}"
echo "  梯度裁剪: ${MAX_GRAD_NORM}"
if [ -n "${INIT_CKPT_DIR}" ]; then
    echo "  初始化权重: ${INIT_CKPT_DIR} (不恢复 optimizer/scheduler/step)"
fi
if [ -n "${CKPT_DIR}" ]; then
    echo "  恢复 checkpoint: ${CKPT_DIR} (恢复 scheduler/step)"
fi
echo "------------------------------------------"
echo "分布式配置:"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  NPROC_PER_NODE: ${NPROC_PER_NODE}"
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

# train_flashmtp.py 始终 init_distributed()，需 torchrun 提供 RANK/WORLD_SIZE/LOCAL_RANK
LAUNCHER=(torchrun --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}")

# 构建可选参数
OPTIONAL_ARGS=""

if [ -n "${EVAL_DATA_PATH}" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --eval-data-path ${EVAL_DATA_PATH}"
fi

if [ -n "${EFFECTIVE_LOSS_DECAY_GAMMA}" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --loss-decay-gamma ${EFFECTIVE_LOSS_DECAY_GAMMA}"
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

if [ -n "${INIT_CKPT_DIR}" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --init-ckpt-dir ${INIT_CKPT_DIR}"
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

if [ "${LOCAL_POSITION_TAG}" = "lp1" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --local-position"
fi

if [ "${TEACHER_MATCH_CAP_TAG}" = "tmc1" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --loss-teacher-match-cap"
fi

if awk "BEGIN {exit !(${W1_MSE} > 0)}"; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --w1-mse ${W1_MSE}"
fi

if [ "${DFLASH_DISTILL_STAGE}" != "none" ]; then
    if [ -z "${DFLASH_TEACHER_PATH}" ]; then
        echo "错误: DFLASH_DISTILL_STAGE=${DFLASH_DISTILL_STAGE} 时必须设置 DFLASH_TEACHER_PATH"
        exit 1
    fi
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --dflash-teacher-path ${DFLASH_TEACHER_PATH}"
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --dflash-distill-stage ${DFLASH_DISTILL_STAGE}"
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --dflash-distill-weight ${DFLASH_DISTILL_WEIGHT}"
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --dflash-distill-temperature ${DFLASH_DISTILL_TEMPERATURE}"
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --dflash-distill-top-k ${DFLASH_DISTILL_TOP_K}"
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --dflash-stage2-ce-gate ${DFLASH_STAGE2_CE_GATE}"
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
    --pivot-fuse-mode ${PIVOT_FUSE_MODE} \
    --num-middle-layers-n ${NUM_MIDDLE_LAYERS_N} \
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
echo "  python benchmark.py --draft-model ${OUTPUT_DIR}/epoch_${NUM_EPOCHS}_step_<step>"
echo "=========================================="
