#!/bin/bash
# FlashMTP 训练启动脚本（单目标 CE loss）
#
# 用法:
#   bash scripts/run_training_flashmtp.sh              # 默认 a800 环境
#   bash scripts/run_training_flashmtp.sh --dt qz    # 指定集群
#   NUM_EPOCHS=8 LOCAL_POSITION=true bash scripts/run_training_flashmtp.sh
#
# 可通过环境变量覆盖下方各块的默认值。

set -e

# ---------------------------------------------------------------------------
# 初始化
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
if [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
    source "${PROJECT_DIR}/.venv/bin/activate"
fi
cd "${PROJECT_DIR}"

# 解析脚本参数（其余未知参数忽略）
while [[ $# -gt 0 ]]; do
    case $1 in
        --dt) DT="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# ---------------------------------------------------------------------------
# 1. 运行环境（集群 / 机器类型）
#    qz: 启智离线 WandB；h100: WHZ 开发机；a800: 默认 A800 共享路径
# ---------------------------------------------------------------------------
DT="${DT:-a800}"
if [[ "$DT" != "qz" && "$DT" != "a800" && "$DT" != "h100" ]]; then
    echo "错误: --dt 须为 qz、a800 或 h100"
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. 分布式
# ---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29501}"
TP_SIZE="${TP_SIZE:-1}"
DIST_TIMEOUT="${DIST_TIMEOUT:-3600}"

# ---------------------------------------------------------------------------
# 3. 模型结构（draft / teacher pivot / block）
# ---------------------------------------------------------------------------
NUM_DRAFT_LAYERS="${NUM_DRAFT_LAYERS:-5}"       # draft transformer 层数
NUM_MIDDLE_LAYERS_N="${NUM_MIDDLE_LAYERS_N:-5}"  # teacher 中间层数（首尾各 1 层 + N 中间层）
BLOCK_SIZE="${BLOCK_SIZE:-16}"                   # 每块 speculative token 数
NUM_ANCHORS="${NUM_ANCHORS:-512}"                # 每条序列随机采样的 anchor 块数
CHS_CONCAT_MODE="${CHS_CONCAT_MODE:-feature}"    # CHS 拼接方式（当前固定 feature）
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flex_attention}"

# local_position=true: draft 用块内 1..block_size，CHS prefix RoPE 全 0
# local_position=false: draft 用全局 anchor 位置（默认）
LOCAL_POSITION="${LOCAL_POSITION:-false}"
LOCAL_POSITION_TAG="lp0"
case "$(echo "${LOCAL_POSITION}" | tr '[:upper:]' '[:lower:]')" in
    true|1|yes) LOCAL_POSITION_TAG="lp1" ;;
esac

# ---------------------------------------------------------------------------
# 4. 数据
# ---------------------------------------------------------------------------
DATA_NUM_SAMPLES="${DATA_NUM_SAMPLES:-40000}"
ENABLE_THINKING="${ENABLE_THINKING:-off}"        # on | off，影响 regen 数据文件名
MAX_LENGTH="${MAX_LENGTH:-4096}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-qwen}"
IS_PREFORMATTED="${IS_PREFORMATTED:-}"           # 非空则传 --is-preformatted
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
BUILD_DATASET_NUM_PROC="${BUILD_DATASET_NUM_PROC:-8}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-}"             # 留空则不评估

# ---------------------------------------------------------------------------
# 5. 训练超参
# ---------------------------------------------------------------------------
NUM_EPOCHS="${NUM_EPOCHS:-6}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-6e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.04}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
LOSS_DECAY_GAMMA="${LOSS_DECAY_GAMMA:-7}"        # 块内 CE 指数衰减 γ；留空则不启用

# ---------------------------------------------------------------------------
# 6. Checkpoint / 恢复
# ---------------------------------------------------------------------------
RESUME="${RESUME:-}"                             # 非空则传 --resume
CKPT_DIR="${CKPT_DIR:-}"                         # 指定 checkpoint 目录恢复权重
LOG_INTERVAL="${LOG_INTERVAL:-50}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"

# ---------------------------------------------------------------------------
# 7. 实验追踪（WandB / none）
# ---------------------------------------------------------------------------
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-flashmtp-training-exp}"
WANDB_DIR="${WANDB_DIR:-./wandb}"

# run 标识后缀（qz/h100 的 OUTPUT_DIR 与 WandB 共用）
RUN_SUFFIX="nlayers${NUM_DRAFT_LAYERS}_block_${BLOCK_SIZE}_epochs${NUM_EPOCHS}_${LOCAL_POSITION_TAG}"
WANDB_SUFFIX="n${NUM_MIDDLE_LAYERS_N}_nlayers${NUM_DRAFT_LAYERS}_block_${BLOCK_SIZE}_n${DATA_NUM_SAMPLES}_${CHS_CONCAT_MODE}_epochs${NUM_EPOCHS}_${LOCAL_POSITION_TAG}"

# ---------------------------------------------------------------------------
# 8. 路径（依赖 DT）
# ---------------------------------------------------------------------------
TARGET_MODEL_BACKEND="${TARGET_MODEL_BACKEND:-hf}"
CACHE_DIR="${CACHE_DIR:-./cache/data/regen_data/nemotron_${DATA_NUM_SAMPLES}}"

if [ "$DT" = "qz" ]; then
    export WANDB_MODE=offline
    TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/FlashMTP/cache/data/regen_data/nemotron_${DATA_NUM_SAMPLES}/nemotron_think_${ENABLE_THINKING}_samples_${DATA_NUM_SAMPLES}_qwen3_8b_regen.jsonl}"
    TARGET_MODEL="${TARGET_MODEL:-/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/models/Qwen/Qwen3-8B}"
    OUTPUT_DIR="${OUTPUT_DIR:-./cache/models/flashmtp_qz_fuse${NUM_MIDDLE_LAYERS_N}_${CHS_CONCAT_MODE}_sample_${DATA_NUM_SAMPLES}_think_${ENABLE_THINKING}_${RUN_SUFFIX}_maxlen${MAX_LENGTH}}"
elif [ "$DT" = "h100" ]; then
    TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-../training_data/regen_data/nemotron_${DATA_NUM_SAMPLES}/nemotron_think_${ENABLE_THINKING}_samples_${DATA_NUM_SAMPLES}_qwen3_8b_regen.jsonl}"
    TARGET_MODEL="${TARGET_MODEL:-$WHZ_DIR/models/Qwen/Qwen3-8B}"
    OUTPUT_DIR="${OUTPUT_DIR:-./cache/models/flashmtp_h100_fuse$((NUM_MIDDLE_LAYERS_N + 2))_sample_${DATA_NUM_SAMPLES}_think_${ENABLE_THINKING}_${RUN_SUFFIX}_maxlen${MAX_LENGTH}}"
else
    # a800 默认固定 nemotron_40000 think_on 数据
    TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/share/wanghanzhen/SpeculativeDecoding/NIPS26/FlashMTP_v1.1/cache/data/regen_data/nemotron_40000/nemotron_think_on_samples_40000_qwen3_8b_regen.jsonl}"
    TARGET_MODEL="${TARGET_MODEL:-/share/public/public_models/Qwen3-8B}"
    OUTPUT_DIR="${OUTPUT_DIR:-./cache/models/flashmtp_a800_fuse${NUM_MIDDLE_LAYERS_N}_nemotron_40000_think_on_nlayers${NUM_DRAFT_LAYERS}_maxlen${MAX_LENGTH}_epochs${NUM_EPOCHS}_${LOCAL_POSITION_TAG}}"
fi

WANDB_RUN_ID="${WANDB_RUN_ID:-flashmtp_${DT}_${WANDB_SUFFIX}}"
WANDB_NAME="${WANDB_RUN_NAME:-flashmtp_${DT}_n${NUM_MIDDLE_LAYERS_N}_nlayers${NUM_DRAFT_LAYERS}_maxlen${MAX_LENGTH}_ep${NUM_EPOCHS}_${CHS_CONCAT_MODE}_${LOCAL_POSITION_TAG}}"

# ---------------------------------------------------------------------------
# 打印配置摘要
# ---------------------------------------------------------------------------
echo "=========================================="
echo "FlashMTP 训练"
echo "=========================================="
echo "[环境] dt=${DT}  GPUs=${CUDA_VISIBLE_DEVICES}  nproc=${NPROC_PER_NODE}  tp=${TP_SIZE}"
echo "[模型] draft_layers=${NUM_DRAFT_LAYERS}  middle_N=${NUM_MIDDLE_LAYERS_N}  block=${BLOCK_SIZE}  anchors=${NUM_ANCHORS}"
echo "       attn=${ATTENTION_BACKEND}  loss_gamma=${LOSS_DECAY_GAMMA:-off}  local_position=${LOCAL_POSITION} (${LOCAL_POSITION_TAG})"
echo "[数据] samples=${DATA_NUM_SAMPLES}  think=${ENABLE_THINKING}  maxlen=${MAX_LENGTH}  template=${CHAT_TEMPLATE}"
echo "       train=${TRAIN_DATA_PATH}"
echo "       eval=${EVAL_DATA_PATH:-（无）}"
echo "[训练] epochs=${NUM_EPOCHS}  batch=${BATCH_SIZE}x${ACCUMULATION_STEPS}  lr=${LEARNING_RATE}  warmup=${WARMUP_RATIO}"
echo "[输出] ${OUTPUT_DIR}"
echo "[目标] ${TARGET_MODEL} (${TARGET_MODEL_BACKEND})"
if [ "${REPORT_TO}" = "wandb" ]; then
    echo "[WandB] project=${WANDB_PROJECT}  id=${WANDB_RUN_ID}  dir=${WANDB_DIR}"
fi
echo "=========================================="
echo ""

# 输出目录冲突时自动追加后缀
original_output_dir="${OUTPUT_DIR}"
suffix=1
while [ -d "${OUTPUT_DIR}" ] && [ -n "$(ls -A "${OUTPUT_DIR}" 2>/dev/null)" ]; do
    OUTPUT_DIR="${original_output_dir}_${suffix}"
    suffix=$((suffix + 1))
done
if [ "${OUTPUT_DIR}" != "${original_output_dir}" ]; then
    echo "警告: ${original_output_dir} 已存在且非空，切换到 ${OUTPUT_DIR}"
fi

mkdir -p "${OUTPUT_DIR}" "${CACHE_DIR}" "${WANDB_DIR}"

# ---------------------------------------------------------------------------
# 启动训练
# ---------------------------------------------------------------------------
echo "==> 开始训练 FlashMTP"
echo ""

LAUNCHER=(torchrun --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}")

OPTIONAL_ARGS=""
[ -n "${EVAL_DATA_PATH}" ]    && OPTIONAL_ARGS+=" --eval-data-path ${EVAL_DATA_PATH}"
[ -n "${LOSS_DECAY_GAMMA}" ]  && OPTIONAL_ARGS+=" --loss-decay-gamma ${LOSS_DECAY_GAMMA}"
[ -n "${IS_PREFORMATTED}" ]   && OPTIONAL_ARGS+=" --is-preformatted"
[ -n "${RESUME}" ]            && OPTIONAL_ARGS+=" --resume"
[ -n "${CKPT_DIR}" ]          && OPTIONAL_ARGS+=" --ckpt-dir ${CKPT_DIR}"
[ "${LOCAL_POSITION_TAG}" = "lp1" ] && OPTIONAL_ARGS+=" --local-position"

if [ "${REPORT_TO}" != "none" ]; then
    OPTIONAL_ARGS+=" --report-to ${REPORT_TO}"
    [ "${REPORT_TO}" = "wandb" ] && [ -n "${WANDB_PROJECT}" ] && OPTIONAL_ARGS+=" --wandb-project ${WANDB_PROJECT}"
    [ -n "${WANDB_RUN_NAME}" ]   && OPTIONAL_ARGS+=" --wandb-run-name ${WANDB_RUN_NAME}"
    [ -n "${WANDB_RUN_ID}" ]     && OPTIONAL_ARGS+=" --wandb-run-id ${WANDB_RUN_ID}"
fi

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
    --num-middle-layers-n "${NUM_MIDDLE_LAYERS_N}" \
    --seed 42 \
    ${OPTIONAL_ARGS} 2>&1 || EXIT_CODE=$?

if [ "${EXIT_CODE}" -ne 0 ]; then
    echo ""
    echo "训练失败 (exit=${EXIT_CODE})"
    exit "${EXIT_CODE}"
fi

echo ""
echo "=========================================="
echo "训练完成"
echo "模型目录: ${OUTPUT_DIR}"
echo "推理示例: python evaluation/benchmark.py --draft-model ${OUTPUT_DIR}/epoch_${NUM_EPOCHS}_step_<step>"
echo "=========================================="
