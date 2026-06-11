#!/bin/bash
# FlashMTP 训练启动脚本（单目标 CE + 可选 DFlash 蒸馏）

set -e

# ========================================
# 环境初始化
# ========================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
if [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
    source "${PROJECT_DIR}/.venv/bin/activate"
fi
cd "${PROJECT_DIR}"

# ========================================
# 命令行参数
# ========================================
# --dt: 运行环境 (qz | a800 | h100)，决定默认数据/模型路径
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
# 分布式 / GPU
# ========================================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29501}"
TP_SIZE="${TP_SIZE:-1}"
DIST_TIMEOUT="${DIST_TIMEOUT:-3600}"

# ========================================
# 模型结构
# ========================================
BLOCK_SIZE="${BLOCK_SIZE:-16}"
NUM_DRAFT_LAYERS="${NUM_DRAFT_LAYERS:-5}"
NUM_ANCHORS="${NUM_ANCHORS:-512}"
NUM_MIDDLE_LAYERS_N="${NUM_MIDDLE_LAYERS_N:-5}"
PIVOT_FUSE_MODE="${PIVOT_FUSE_MODE:-prefix_condition}"
CHS_CONCAT_MODE="${CHS_CONCAT_MODE:-feature}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flex_attention}"
TARGET_MODEL_BACKEND="${TARGET_MODEL_BACKEND:-hf}"

# draft 块内 position_ids：CHS RoPE 前缀全 0，draft 为 1..block_size
LOCAL_POSITION="${LOCAL_POSITION:-true}"
LOCAL_POSITION_TAG="lp0"
case "$(echo "${LOCAL_POSITION}" | tr '[:upper:]' '[:lower:]')" in
    true|1|yes) LOCAL_POSITION_TAG="lp1" ;;
esac

# ========================================
# 数据集
# ========================================
DATA_NUM_SAMPLES="${DATA_NUM_SAMPLES:-40000}"
ENABLE_THINKING="${ENABLE_THINKING:-off}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-qwen}"
IS_PREFORMATTED="${IS_PREFORMATTED:-}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
BUILD_DATASET_NUM_PROC="${BUILD_DATASET_NUM_PROC:-8}"
CACHE_DIR="${CACHE_DIR:-./cache/data/regen_data/nemotron_${DATA_NUM_SAMPLES}}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-}"

# ========================================
# 训练超参
# ========================================
NUM_EPOCHS="${NUM_EPOCHS:-6}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-6e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.04}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"


# ========================================
# Checkpoint / 恢复
# ========================================
RESUME="${RESUME:-}"
CKPT_DIR="${CKPT_DIR:-}"
INIT_CKPT_DIR="${INIT_CKPT_DIR:-}"
if [ -n "${INIT_CKPT_DIR}" ] && { [ -n "${CKPT_DIR}" ] || [ -n "${RESUME}" ]; }; then
    echo "错误: INIT_CKPT_DIR 是只加载权重的新训练入口，不能和 CKPT_DIR/RESUME 同时使用"
    exit 1
fi

# ========================================
# DFlash 蒸馏（设置 DFLASH_TEACHER_PATH 即启用）
# ========================================
DFLASH_TEACHER_PATH="${DFLASH_TEACHER_PATH:-}"                         # DFlash teacher checkpoint 路径

# 初始权重
DFLASH_DISTILL_WEIGHT="${DFLASH_DISTILL_WEIGHT:-1.0}"                  # KL 蒸馏损失权重
DFLASH_DISTILL_TEMPERATURE="${DFLASH_DISTILL_TEMPERATURE:-1.0}"        # KL 温度，越大 teacher 分布越平滑
DFLASH_DISTILL_TOP_K="${DFLASH_DISTILL_TOP_K:-128}"                    # KL 候选 token 数：teacher top-k + true label

# CE first-error 后衰减 gamma（first-error 位置权重 1.0，之后 exp(-(k-k_err)/γ)）；0/空表示不衰减
LOSS_DECAY_GAMMA="${LOSS_DECAY_GAMMA-7}"
DFLASH_DISTILL_DECAY_GAMMA="${DFLASH_DISTILL_DECAY_GAMMA:-0}"          # KL 位置衰减 gamma，0/空表示不开

DFLASH_DISTILL_POS_MODE="${DFLASH_DISTILL_POS_MODE:-prefix}"           # distill 位置：prefix / all
DFLASH_CE_PREFIX_WEIGHT="${DFLASH_CE_PREFIX_WEIGHT:-0.0}"              # student 正确前缀（first error 之前）的 CE 权重，0 表示交给 KL
DFLASH_CE_NORM="${DFLASH_CE_NORM:-block}"                              # CE 归一化：block 每块等贡献 / global 早错 block 权重更大

# 中间衰减
DFLASH_MILESTONE_EPOCH="${DFLASH_MILESTONE_EPOCH:-0.0}"                # DFlash 蒸馏中余弦切换开始的 epoch
DFLASH_CE_WEIGHT="${DFLASH_CE_WEIGHT:-0.8}"                            # milestone 后 CE 最终目标权重
DFLASH_DISTILL_MIN_SCALE="${DFLASH_DISTILL_MIN_SCALE:-0.2}"            # KL 余弦衰减的最小保留比例，0 表示降到 0
DFLASH_CE_MIN_SCALE="${DFLASH_CE_MIN_SCALE:-0.0}"                      # CE 初始/最小保留比例，0 表示 milestone 前 CE=0

# 用于 OUTPUT_DIR / WandB run id 的蒸馏配置摘要
DFLASH_DISTILL_TAG="dnone"
if [ -n "${DFLASH_TEACHER_PATH}" ]; then
    DFLASH_DISTILL_TAG="klw${DFLASH_DISTILL_WEIGHT}_top${DFLASH_DISTILL_TOP_K}_ceg${LOSS_DECAY_GAMMA:-none}_dkg${DFLASH_DISTILL_DECAY_GAMMA:-none}_dpos${DFLASH_DISTILL_POS_MODE}_cepw${DFLASH_CE_PREFIX_WEIGHT}_cenorm${DFLASH_CE_NORM}"
fi

# ========================================
# 日志 / 保存 / Tracker
# ========================================
LOG_INTERVAL="${LOG_INTERVAL:-50}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-flashmtp-training-exp}"
WANDB_DIR="${WANDB_DIR:-./wandb}"

# ========================================
# 环境相关默认路径 (--dt)
# ========================================
RUN_SUFFIX="${DFLASH_DISTILL_TAG}"

if [ "$DT" = "qz" ]; then
    export WANDB_MODE=offline
    TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/FlashMTP/cache/data/regen_data/nemotron_${DATA_NUM_SAMPLES}/nemotron_think_${ENABLE_THINKING}_samples_${DATA_NUM_SAMPLES}_qwen3_8b_regen.jsonl}"
    OUTPUT_DIR="${OUTPUT_DIR:-./cache/models/flashmtp_fuse${NUM_MIDDLE_LAYERS_N}_sample_${DATA_NUM_SAMPLES}_think_${ENABLE_THINKING}_nlayers${NUM_DRAFT_LAYERS}_block_${BLOCK_SIZE}_maxlen${MAX_LENGTH}_ep${NUM_EPOCHS}_${RUN_SUFFIX}}"
    TARGET_MODEL="${TARGET_MODEL:-/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/models/Qwen/Qwen3-8B}"
elif [ "$DT" = "h100" ]; then
    TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-../training_data/regen_data/nemotron_${DATA_NUM_SAMPLES}/nemotron_think_${ENABLE_THINKING}_samples_${DATA_NUM_SAMPLES}_qwen3_8b_regen.jsonl}"
    OUTPUT_DIR="${OUTPUT_DIR:-./cache/models/flashmtp_h100_${PIVOT_FUSE_MODE}_fuse$((NUM_MIDDLE_LAYERS_N + 2))_sample_${DATA_NUM_SAMPLES}_think_${ENABLE_THINKING}_nlayers${NUM_DRAFT_LAYERS}_block_${BLOCK_SIZE}_maxlen${MAX_LENGTH}_epochs${NUM_EPOCHS}_${RUN_SUFFIX}}"
    TARGET_MODEL="${TARGET_MODEL:-$WHZ_DIR/models/Qwen/Qwen3-8B}"
else
    TRAIN_DATA_PATH="/share/wanghanzhen/SpeculativeDecoding/NIPS26/FlashMTP_v1.1/cache/data/regen_data/nemotron_40000/nemotron_think_on_samples_40000_qwen3_8b_regen.jsonl"
    OUTPUT_DIR="${OUTPUT_DIR:-./cache/models/flashmtp_a800_${PIVOT_FUSE_MODE}_fuse${NUM_MIDDLE_LAYERS_N}_nemotron_40000_think_on_nlayers${NUM_DRAFT_LAYERS}_maxlen${MAX_LENGTH}_epochs${NUM_EPOCHS}_${RUN_SUFFIX}}"
    TARGET_MODEL="${TARGET_MODEL:-/share/public/public_models/Qwen3-8B}"
fi

WANDB_RUN_ID="${WANDB_RUN_ID:-flashmtp_n${NUM_MIDDLE_LAYERS_N}_nlayers${NUM_DRAFT_LAYERS}_block_${BLOCK_SIZE}_n${DATA_NUM_SAMPLES}_ep${NUM_EPOCHS}_${RUN_SUFFIX}}"
WANDB_NAME="${WANDB_RUN_NAME:-flashmtp_${DT}_n${NUM_MIDDLE_LAYERS_N}_nlayers${NUM_DRAFT_LAYERS}_maxlen${MAX_LENGTH}_ep${NUM_EPOCHS}_${RUN_SUFFIX}}"

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
echo "  local_position: ${LOCAL_POSITION} (tag ${LOCAL_POSITION_TAG})"
echo "  dflash_distill: $([ -n "${DFLASH_TEACHER_PATH}" ] && echo enabled || echo disabled) (tag ${DFLASH_DISTILL_TAG})"
if [ -n "${DFLASH_TEACHER_PATH}" ]; then
    echo "  dflash_teacher_path: ${DFLASH_TEACHER_PATH}"
    echo "  dflash_distill: weight=${DFLASH_DISTILL_WEIGHT}, temperature=${DFLASH_DISTILL_TEMPERATURE}, top_k=${DFLASH_DISTILL_TOP_K}"
    echo "  dflash_distill_decay_gamma: ${DFLASH_DISTILL_DECAY_GAMMA:-未设置(不启用)}"
    echo "  dflash_distill_pos_mode: ${DFLASH_DISTILL_POS_MODE}"
    echo "  dflash_ce_weight: ${DFLASH_CE_WEIGHT}"
    echo "  dflash_ce_prefix_weight: ${DFLASH_CE_PREFIX_WEIGHT}"
    echo "  dflash_ce_norm: ${DFLASH_CE_NORM}"
    echo "  dflash_milestone_epoch: ${DFLASH_MILESTONE_EPOCH}"
    echo "  dflash_distill_min_scale: ${DFLASH_DISTILL_MIN_SCALE}"
    echo "  dflash_ce_min_scale: ${DFLASH_CE_MIN_SCALE}"
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
echo "  CE Loss衰减Gamma: ${LOSS_DECAY_GAMMA:-未设置(不启用)}"
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

# 输出目录冲突时自动追加数字后缀
original_output_dir="${OUTPUT_DIR}"
suffix=1
while [ -d "${OUTPUT_DIR}" ] && [ -n "$(ls -A "${OUTPUT_DIR}" 2>/dev/null)" ]; do
    OUTPUT_DIR="${original_output_dir}_${suffix}"
    suffix=$((suffix + 1))
done
if [ "${OUTPUT_DIR}" != "${original_output_dir}" ]; then
    echo "警告: 输出目录 ${original_output_dir} 已存在且非空，自动切换到: ${OUTPUT_DIR}"
fi

mkdir -p "${OUTPUT_DIR}" "${CACHE_DIR}" "${WANDB_DIR}"

# ========================================
# 构建可选 CLI 参数
# ========================================
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

if [ -n "${DFLASH_TEACHER_PATH}" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --dflash-teacher-path ${DFLASH_TEACHER_PATH}"
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --dflash-distill-weight ${DFLASH_DISTILL_WEIGHT}"
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --dflash-distill-temperature ${DFLASH_DISTILL_TEMPERATURE}"
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --dflash-distill-top-k ${DFLASH_DISTILL_TOP_K}"
    if [ -n "${DFLASH_DISTILL_DECAY_GAMMA}" ]; then
        OPTIONAL_ARGS="${OPTIONAL_ARGS} --dflash-distill-decay-gamma ${DFLASH_DISTILL_DECAY_GAMMA}"
    fi
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --dflash-distill-pos-mode ${DFLASH_DISTILL_POS_MODE}"
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --dflash-ce-weight ${DFLASH_CE_WEIGHT}"
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --dflash-ce-prefix-weight ${DFLASH_CE_PREFIX_WEIGHT}"
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --dflash-ce-norm ${DFLASH_CE_NORM}"
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --dflash-milestone-epoch ${DFLASH_MILESTONE_EPOCH}"
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --dflash-distill-min-scale ${DFLASH_DISTILL_MIN_SCALE}"
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --dflash-ce-min-scale ${DFLASH_CE_MIN_SCALE}"
fi

# ========================================
# 启动训练
# ========================================
echo ""
echo "==> 开始训练 FlashMTP"
echo ""

LAUNCHER=(torchrun --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}")

EXIT_CODE=0
"${LAUNCHER[@]}" ./scripts/train_flashmtp.py \
    --target-model-path "${TARGET_MODEL}" \
    --target-model-backend "${TARGET_MODEL_BACKEND}" \
    --train-data-path "${TRAIN_DATA_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --cache-dir "${CACHE_DIR}" \
    --num-draft-layers "${NUM_DRAFT_LAYERS}" \
    --block-size "${BLOCK_SIZE}" \
    --num-anchors "${NUM_ANCHORS}" \
    --attention-backend "${ATTENTION_BACKEND}" \
    --learning-rate "${LEARNING_RATE}" \
    --warmup-ratio "${WARMUP_RATIO}" \
    --num-epochs "${NUM_EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --accumulation-steps "${ACCUMULATION_STEPS}" \
    --max-grad-norm "${MAX_GRAD_NORM}" \
    --max-length "${MAX_LENGTH}" \
    --log-interval "${LOG_INTERVAL}" \
    --save-interval "${SAVE_INTERVAL}" \
    --eval-interval "${EVAL_INTERVAL}" \
    --chat-template "${CHAT_TEMPLATE}" \
    --dataloader-num-workers "${DATALOADER_NUM_WORKERS}" \
    --build-dataset-num-proc "${BUILD_DATASET_NUM_PROC}" \
    --tp-size "${TP_SIZE}" \
    --dist-timeout "${DIST_TIMEOUT}" \
    --chs-concat-mode "${CHS_CONCAT_MODE}" \
    --pivot-fuse-mode "${PIVOT_FUSE_MODE}" \
    --num-middle-layers-n "${NUM_MIDDLE_LAYERS_N}" \
    --seed 42 \
    ${OPTIONAL_ARGS} 2>&1 || EXIT_CODE=$?

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
