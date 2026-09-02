#!/bin/bash
# FlashMTP 训练启动脚本（单目标 CE，无 DFlash++ 的 L_dflash/L_con 等多损失）

set -e

# 自动激活虚拟环境；默认保持旧 Qwen 环境，Gemma4 wrapper 显式覆盖。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
FLASHMTP_VENV="${FLASHMTP_VENV:-${PROJECT_DIR}/.venv}"
if [ -f "${FLASHMTP_VENV}/bin/activate" ]; then
    source "${FLASHMTP_VENV}/bin/activate"
fi

cd "${PROJECT_DIR}"

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
CHS_CONCAT_MODE="${CHS_CONCAT_MODE:-feature}"
PIVOT_FUSE_MODE="${PIVOT_FUSE_MODE:-linear_fuse}"
NUM_MIDDLE_LAYERS_N="${NUM_MIDDLE_LAYERS_N:-5}"
NUM_ANCHORS="${NUM_ANCHORS:-512}"
TEMP_ROLLOUT="${TEMP_ROLLOUT:-false}"
TEMP_ROLLOUT_PROJECTION_CHUNK_SIZE="${TEMP_ROLLOUT_PROJECTION_CHUNK_SIZE:-0}"
TEMP_ROLLOUT_ENABLED=0
case "$(echo "${TEMP_ROLLOUT}" | tr '[:upper:]' '[:lower:]')" in
    true|1|yes) TEMP_ROLLOUT_ENABLED=1 ;;
esac

# 恢复训练
RESUME="${RESUME:-}"
CKPT_DIR="${CKPT_DIR:-}"
RESUME_OPTIMIZER="${RESUME_OPTIMIZER:-1}"

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
FINAL_FORWARD_KL_WEIGHT="${FINAL_FORWARD_KL_WEIGHT:-0.0}"
TV_LOSS_WEIGHT="${TV_LOSS_WEIGHT:-1.0}"
BASE_LM_FORWARD_KL_WEIGHT="${BASE_LM_FORWARD_KL_WEIGHT:-0.0}"
FORWARD_KL_TAG=""
if awk "BEGIN {exit !((${FINAL_FORWARD_KL_WEIGHT} > 0) || (${BASE_LM_FORWARD_KL_WEIGHT} > 0))}"; then
    FORWARD_KL_TAG="_fklf${FINAL_FORWARD_KL_WEIGHT}_fklb${BASE_LM_FORWARD_KL_WEIGHT}"
fi
MARKOV_TAG="mh${MARKOV_HEAD_TYPE}_${MARKOV_OUTPUT_MODE}_r${MARKOV_RANK}_ce${FINAL_CE_WEIGHT}_tv${TV_LOSS_WEIGHT}${FORWARD_KL_TAG}"

# 草稿块内 position_ids：CHS RoPE 前缀全 0，draft 为 1..block_size（默认 false 为全局 anchor 位置）
LOCAL_POSITION="${LOCAL_POSITION:-false}"
LOCAL_POSITION_TAG="lp0"
case "$(echo "${LOCAL_POSITION}" | tr '[:upper:]' '[:lower:]')" in
    true|1|yes) LOCAL_POSITION_TAG="lp1" ;;
esac

# DeepSpec-style alignment: slot 0 predicts anchor+1 and all B slots are supervised.
LEFT_SHIFT="${LEFT_SHIFT:-false}"
LEFT_SHIFT_TAG="leftshift0"
case "$(echo "${LEFT_SHIFT}" | tr '[:upper:]' '[:lower:]')" in
    true|1|yes) LEFT_SHIFT_TAG="leftshift1" ;;
esac

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
DIST_TIMEOUT="${DIST_TIMEOUT:-120}"

# 模型参数（OUTPUT_DIR 依赖 BLOCK_SIZE，须早于 dt 分支）
BLOCK_SIZE="${BLOCK_SIZE:-16}"
MODEL_TAG="${MODEL_TAG:-'Qwen3_8B'}"

if [ "$DT" = "qz" ]; then
    export WANDB_MODE=offline
    TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/FlashMTP/cache/data/regen_data/nemotron_${DATA_NUM_SAMPLES}/nemotron_think_${ENABLE_THINKING}_samples_${DATA_NUM_SAMPLES}_qwen3_8b_regen.jsonl}"
    OUTPUT_DIR="${OUTPUT_DIR:-./cache/models/flashmtp_qz_${PIVOT_FUSE_MODE}_fuse${NUM_MIDDLE_LAYERS_N}_${CHS_CONCAT_MODE}_sample_${DATA_NUM_SAMPLES}_think_${ENABLE_THINKING}_nlayers${NUM_DRAFT_LAYERS}_block_${BLOCK_SIZE}_${LEFT_SHIFT_TAG}_${MARKOV_TAG}_wb_${BASE_LM_CE_WEIGHT}_bgemma_${BASE_LM_CE_DECAY_GAMMA}_maxlen${MAX_LENGTH}_epochs${NUM_EPOCHS}_${MODEL_TAG}}"
    TARGET_MODEL="${TARGET_MODEL:-/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/models/Qwen/Qwen3-8B}"
elif [ "$DT" = "h100" ]; then
    TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/share/dai-sys/wanghanzhen/projects/MTP/training_data/nemotron_think_off_samples_40000_qwen3_8b_regen.jsonl}"
    OUTPUT_DIR="${OUTPUT_DIR:-./cache/models/flashmtp_h100_${PIVOT_FUSE_MODE}_fuse$((NUM_MIDDLE_LAYERS_N + 2))_sample_${DATA_NUM_SAMPLES}_nlayers${NUM_DRAFT_LAYERS}_block_${BLOCK_SIZE}_${LEFT_SHIFT_TAG}_${MARKOV_TAG}_wb_${BASE_LM_CE_WEIGHT}_bgemma_${BASE_LM_CE_DECAY_GAMMA}_maxlen${MAX_LENGTH}_epochs${NUM_EPOCHS}_${MODEL_TAG}}"
    TARGET_MODEL="${TARGET_MODEL:-$WHZ_HOME/models/Qwen/Qwen3-8B}"
else
    TRAIN_DATA_PATH="/share/wanghanzhen/SpeculativeDecoding/NIPS26/FlashMTP_v1.1/cache/data/regen_data/nemotron_40000/nemotron_think_on_samples_40000_qwen3_8b_regen.jsonl"
    OUTPUT_DIR="${OUTPUT_DIR:-./cache/models/flashmtp_a800_${PIVOT_FUSE_MODE}_fuse${NUM_MIDDLE_LAYERS_N}_nemotron_40000_think_on_nlayers${NUM_DRAFT_LAYERS}_${LEFT_SHIFT_TAG}_${MARKOV_TAG}_maxlen${MAX_LENGTH}_epochs${NUM_EPOCHS}_${LOCAL_POSITION_TAG}}"
    TARGET_MODEL="${TARGET_MODEL:-/share/public/public_models/Qwen3-8B}"
fi


if [ "${TEMP_ROLLOUT_ENABLED}" = "1" ]; then
    TARGET_MODEL_BACKEND="${TARGET_MODEL_BACKEND:-sglang}"
else
    TARGET_MODEL_BACKEND="${TARGET_MODEL_BACKEND:-hf}"
fi
DRAFT_CONFIG_PATH="${DRAFT_CONFIG_PATH:-}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.25}"
SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-flashinfer}"
SGLANG_MAX_TOTAL_TOKENS="${SGLANG_MAX_TOTAL_TOKENS:-}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-}"
CE_CHUNK_SIZE="${CE_CHUNK_SIZE:-2048}"

# 训练参数
BATCH_SIZE="${BATCH_SIZE:-1}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"
SHARD_DRAFT_BY_TP="${SHARD_DRAFT_BY_TP:-1}"
# Per DP rank: target sees TRAIN_BATCH_SIZE samples; each TP rank trains one draft slice.
TRAIN_BATCH_SIZE="${BATCH_SIZE}"
if [ "${TEMP_ROLLOUT_ENABLED}" = "1" ]; then
    TRAIN_BATCH_SIZE=1
    SHARD_DRAFT_BY_TP=0
elif [ "${TP_SIZE}" -gt 1 ] && [ "${SHARD_DRAFT_BY_TP}" = "1" ]; then
    if [ "${BATCH_SIZE}" -eq 1 ]; then
        TRAIN_BATCH_SIZE="${TP_SIZE}"
    fi
fi
LEARNING_RATE="${LEARNING_RATE:-6e-4}"
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
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50000}"

# Tracker 参数
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-flashmtp-training-v2new}"
WANDB_DIR="${WANDB_DIR:-./wandb}"  # 离线日志保存目录
# 含 dt / 草稿层数 / 样本量 / 拼接方式；run id 与默认 OUTPUT_DIR 中 nlayers* 可对照。
# W&B rejects names longer than 128 characters. Keep a readable prefix and a
# stable hash suffix so fully automatic names remain valid and collision-safe.
shorten_wandb_value() {
    local value="$1"
    if [ "${#value}" -le 128 ]; then
        printf '%s' "${value}"
        return
    fi
    local digest
    digest="$(printf '%s' "${value}" | sha256sum | cut -c1-16)"
    printf '%s-%s' "${value:0:111}" "${digest}"
}

WANDB_RUN_ID="${WANDB_RUN_ID:-flashmtp_v2_n${NUM_MIDDLE_LAYERS_N}_nlayers${NUM_DRAFT_LAYERS}_block_${BLOCK_SIZE}_${LEFT_SHIFT_TAG}_${MARKOV_TAG}_wb_${BASE_LM_CE_WEIGHT}_bgemma_${BASE_LM_CE_DECAY_GAMMA}_n${DATA_NUM_SAMPLES}_epochs${NUM_EPOCHS}_${MODEL_TAG}}"
WANDB_NAME="${WANDB_RUN_NAME:-flashmtp_v2_n${NUM_MIDDLE_LAYERS_N}_nlayers${NUM_DRAFT_LAYERS}_block_${BLOCK_SIZE}_${LEFT_SHIFT_TAG}_${MARKOV_TAG}_wb_${BASE_LM_CE_WEIGHT}_bgemma_${BASE_LM_CE_DECAY_GAMMA}_maxlen${MAX_LENGTH}_ep${NUM_EPOCHS}_${MODEL_TAG}}"
WANDB_RUN_ID="$(shorten_wandb_value "${WANDB_RUN_ID}")"
WANDB_NAME="$(shorten_wandb_value "${WANDB_NAME}")"

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
if [ "${LEFT_SHIFT_TAG}" = "leftshift1" ]; then
    echo "  left_shift: true (tag ${LEFT_SHIFT_TAG}; block_size = anchor + B-1 drafts, total span)"
else
    echo "  left_shift: false (tag ${LEFT_SHIFT_TAG}; legacy mode, block_size = draft block width)"
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
echo "  temp-rollout: ${TEMP_ROLLOUT} (greedy target branch labels)"
echo "  Attention后端: ${ATTENTION_BACKEND}"
echo "  Loss衰减Gamma: ${LOSS_DECAY_GAMMA:-未设置(不启用)}"
echo "  最终CE权重: ${FINAL_CE_WEIGHT}"
echo "  Final forward KL权重: ${FINAL_FORWARD_KL_WEIGHT}"
echo "  串行Head TV权重: ${TV_LOSS_WEIGHT}"
echo "  Base LM CE权重: ${BASE_LM_CE_WEIGHT}"
echo "  Base LM forward KL权重: ${BASE_LM_FORWARD_KL_WEIGHT}"
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
# A resume must keep the exact original output directory; otherwise the latest
# epoch_*_step_* checkpoint cannot be discovered. Only fresh single-node runs
# receive an automatic suffix.
while [ "${NNODES}" -le 1 ] 2>/dev/null && [ -z "${RESUME}" ] && [ -d "${OUTPUT_DIR}" ] && [ -n "$(ls -A "${OUTPUT_DIR}" 2>/dev/null)" ]; do
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
    torchrun
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --nproc_per_node "${NPROC_PER_NODE}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)

# 构建可选参数
OPTIONAL_ARGS=""

if [ -n "${DRAFT_CONFIG_PATH}" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --draft-config-path ${DRAFT_CONFIG_PATH}"
fi

if [ -n "${EVAL_DATA_PATH}" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --eval-data-path ${EVAL_DATA_PATH}"
fi

if [ -n "${LOSS_DECAY_GAMMA}" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --loss-decay-gamma ${LOSS_DECAY_GAMMA}"
fi

if awk "BEGIN {exit !(${BASE_LM_CE_WEIGHT} > 0)}"; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --base-lm-ce-weight ${BASE_LM_CE_WEIGHT}"
fi
if awk "BEGIN {exit !((${BASE_LM_CE_WEIGHT} > 0) || (${BASE_LM_FORWARD_KL_WEIGHT} > 0))}" \
    && [ -n "${BASE_LM_CE_DECAY_GAMMA}" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --base-lm-ce-decay-gamma ${BASE_LM_CE_DECAY_GAMMA}"
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

if [ "${LOCAL_POSITION_TAG}" = "lp1" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --local-position"
fi

case "$(echo "${LEFT_SHIFT}" | tr '[:upper:]' '[:lower:]')" in
    true|1|yes) OPTIONAL_ARGS="${OPTIONAL_ARGS} --left-shift" ;;
esac

if [ "${TARGET_MODEL_BACKEND}" = "sglang" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --sglang-attention-backend ${SGLANG_ATTENTION_BACKEND}"
    # SGLang profiles KV pool as: free_mem_after_weights - pre_load_mem * (1 - mem_fraction).
    # With ~14B weights on 80GB H100, mem_fraction < ~0.21 yields negative KV capacity.
    if awk "BEGIN {exit !(${SGLANG_MEM_FRACTION_STATIC} < 0.22)}"; then
        echo "WARNING: SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC} is too low for"
        echo "  SGLang KV profiling (need >=0.22 for MAX_LENGTH=${MAX_LENGTH})."
        echo "  Override with SGLANG_MEM_FRACTION_STATIC=0.25 (script default)."
    fi
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION_STATIC}"
    if [ -z "${SGLANG_MAX_TOTAL_TOKENS}" ]; then
        if [ "${TEMP_ROLLOUT_ENABLED}" = "1" ]; then
            SGLANG_MAX_TOTAL_TOKENS=$((MAX_LENGTH + NUM_ANCHORS * (BLOCK_SIZE - 1)))
        else
            SGLANG_MAX_TOTAL_TOKENS=$((TRAIN_BATCH_SIZE * MAX_LENGTH))
        fi
    fi
    if [ -z "${SGLANG_MAX_RUNNING_REQUESTS}" ]; then
        if [ "${TEMP_ROLLOUT_ENABLED}" = "1" ]; then
            SGLANG_MAX_RUNNING_REQUESTS=$((NUM_ANCHORS + TRAIN_BATCH_SIZE))
        else
            SGLANG_MAX_RUNNING_REQUESTS=${TRAIN_BATCH_SIZE}
        fi
    fi
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --sglang-max-total-tokens ${SGLANG_MAX_TOTAL_TOKENS}"
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --sglang-max-running-requests ${SGLANG_MAX_RUNNING_REQUESTS}"
fi

if [ "${TEMP_ROLLOUT_ENABLED}" = "1" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --temp-rollout --temp-rollout-projection-chunk-size ${TEMP_ROLLOUT_PROJECTION_CHUNK_SIZE}"
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
    --attention-backend ${ATTENTION_BACKEND} \
    --learning-rate ${LEARNING_RATE} \
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
    --chs-concat-mode ${CHS_CONCAT_MODE} \
    --pivot-fuse-mode ${PIVOT_FUSE_MODE} \
    --num-middle-layers-n ${NUM_MIDDLE_LAYERS_N} \
    --markov-head-type ${MARKOV_HEAD_TYPE} \
    --markov-output-mode ${MARKOV_OUTPUT_MODE} \
    --markov-rank ${MARKOV_RANK} \
    --final-ce-weight ${FINAL_CE_WEIGHT} \
    --final-forward-kl-weight ${FINAL_FORWARD_KL_WEIGHT} \
    --tv-loss-weight ${TV_LOSS_WEIGHT} \
    --base-lm-forward-kl-weight ${BASE_LM_FORWARD_KL_WEIGHT} \
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
