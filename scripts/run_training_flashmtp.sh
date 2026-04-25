#!/bin/bash
# FlashMTP 训练：与 scripts/train_flashmtp.py 参数一致；用法: bash scripts/run_training_flashmtp.sh [--dt qz|a800]
#
# 分布式：支持环境变量
#   MASTER_ADDR, MASTER_PORT, PET_MASTER_ADDR, PET_MASTER_PORT
#   PET_NPROC_PER_NODE, PET_NNODES, PET_NODE_RANK
#   (WORLD_SIZE / RANK 由 torchrun 或训练进程设置，仅作日志)
# 单机多卡：默认 NNODES=1, MASTER=127.0.0.1:29501
# 多机：各节点同 MASTER/PORT/NNODES，异 NODE_RANK（或 PET_NNODES / PET_NODE_RANK）
set -e

DT="a800"
while [[ $# -gt 0 ]]; do
    case $1 in
        --dt) DT="$2"; shift 2 ;;
        *) shift ;;
    esac
done
if [[ "$DT" != "qz" && "$DT" != "a800" ]]; then
    echo "错误: --dt 须为 qz 或 a800"
    exit 1
fi

# a800：更保守锚点数，减轻显存峰值
if [ "$DT" = "a800" ]; then
    MAX_LENGTH="${MAX_LENGTH:-4096}"
    NUM_ANCHORS="${NUM_ANCHORS:-512}"
else
    MAX_LENGTH="${MAX_LENGTH:-10240}"
    NUM_ANCHORS="${NUM_ANCHORS:-1024}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
if [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
    source "${PROJECT_DIR}/.venv/bin/activate"
fi
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# ========================================
# 主要训练参数
# ========================================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
if [ -n "${PET_NPROC_PER_NODE:-}" ]; then
    NPROC_PER_NODE="${PET_NPROC_PER_NODE}"
else
    # 常见错误：4 张卡仍起 8 进程 -> NCCL / local rank 与 GPU 不符；若 nproc 大于可见卡数则下调
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        _cvd="${CUDA_VISIBLE_DEVICES// /}"
        if [[ "${_cvd}" =~ ^[0-9,]+$ ]]; then
            _ng=$(echo "${_cvd}" | awk -F',' '{print NF}')
            if [ "${_ng}" -ge 1 ] 2>/dev/null && [ "${NPROC_PER_NODE}" -gt "${_ng}" ] 2>/dev/null; then
                echo "提示: NPROC_PER_NODE=${NPROC_PER_NODE} 大于 CUDA_VISIBLE_DEVICES 中 GPU 数=${_ng}，已自动改为 ${_ng}（可设置 PET_NPROC_PER_NODE 显式覆盖）"
                NPROC_PER_NODE="${_ng}"
            fi
        fi
    fi
fi
NUM_EPOCHS="${NUM_EPOCHS:-10}"
RESUME="${RESUME:-}"
CKPT_DIR="${CKPT_DIR:-}"

# ========================================
# 主要数据集参数
# ========================================
DATA_NUM_SAMPLES="${DATA_NUM_SAMPLES:-400000}"
ENABLE_THINKING="${ENABLE_THINKING:-on}"
NUM_DRAFT_LAYERS="${NUM_DRAFT_LAYERS:-5}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"

# ========================================
# 模型参数
# ========================================
# Teacher 条件窗长 W：每个块在 mask 内最多 attend anchor 之前 W 个 token 的 target hidden；W=1 等价于仅用 anchor-1
CONTEXT_WINDOW_SIZE="${CONTEXT_WINDOW_SIZE:-1}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flex_attention}"
DIFFUSION_MASK_SCHEDULE="${DIFFUSION_MASK_SCHEDULE:-mask_high}" # ("uniform", "cosine", "mask_high")
LOSS_DECAY_GAMMA="${LOSS_DECAY_GAMMA:-7}"
LOSS_WEIGHT_CE="${LOSS_WEIGHT_CE:-1.0}"
LOSS_WEIGHT_KL="${LOSS_WEIGHT_KL:-0.6}"
LOSS_WEIGHT_MSE="${LOSS_WEIGHT_MSE:-0.2}"
LOSS_KL_TOPK="${LOSS_KL_TOPK:-0}"

# ========================================
# 训练参数
# ========================================
# 离散扩散单任务 loss（CE + 可选 KL/MSE），anchor token 永远 clean 且不计入 loss
BATCH_SIZE="${BATCH_SIZE:-1}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-6e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.04}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

# 分布式 rendezvous：MASTER_ADDR / MASTER_PORT
# - 未显式设置 MASTER_ADDR 时：优先 PET_MASTER_ADDR，再 SLURM / PBS 首节点，再否则 127.0.0.1
# - 未显式设置 MASTER_PORT 时：优先 PET_MASTER_PORT，再 29501
if [ -z "${MASTER_ADDR:-}" ]; then
    if [ -n "${PET_MASTER_ADDR:-}" ]; then
        MASTER_ADDR="${PET_MASTER_ADDR}"
    else
        _NLIST="${SLURM_STEP_NODELIST:-${SLURM_JOB_NODELIST:-}}"
        if [ -n "${_NLIST}" ] && command -v scontrol >/dev/null 2>&1; then
            MASTER_ADDR=$(scontrol show hostnames "${_NLIST}" 2>/dev/null | head -1) || true
        fi
        if [ -z "${MASTER_ADDR:-}" ] && [ -n "${PBS_NODEFILE:-}" ] && [ -f "${PBS_NODEFILE}" ]; then
            MASTER_ADDR=$(head -1 "${PBS_NODEFILE}" | tr -d '\r\n' || true)
        fi
        if [ -z "${MASTER_ADDR:-}" ]; then
            MASTER_ADDR=127.0.0.1
        fi
    fi
    unset _NLIST
fi
if [ -z "${MASTER_PORT:-}" ]; then
    if [ -n "${PET_MASTER_PORT:-}" ]; then
        MASTER_PORT="${PET_MASTER_PORT}"
    else
        MASTER_PORT=29501
    fi
fi
export MASTER_ADDR
export MASTER_PORT

# K8s/部分集群用 Pod 主机名作 MASTER 时，c10d 会尝试解析 IPv6 并告警
#   "The IPv6 network addresses of (hostname, port) cannot be retrieved (gai error: -2)"
# 强制走 IPv4 通常可消除告警并避免偶发卡死；多机时仍建议将 PET_MASTER_ADDR 设为
# rank0 可联通的**数字 IPv4**（或 Headless Service 的 ClusterIP）而非无 AAAA 的 DNS 名。
if [ -z "${NCCL_SOCKET_FAMILY:-}" ]; then
    export NCCL_SOCKET_FAMILY=AF_INET
fi

# 多机：NNODES / NODE_RANK
if [ -z "${NNODES:-}" ]; then
    if [ -n "${PET_NNODES:-}" ]; then
        NNODES="${PET_NNODES}"
    elif [ -n "${SLURM_JOB_NUM_NODES:-}" ]; then
        NNODES="${SLURM_JOB_NUM_NODES}"
    else
        NNODES=1
    fi
fi
if [ -z "${NNODES}" ] || [ "${NNODES}" = "0" ]; then
    NNODES=1
fi

if [ -z "${NODE_RANK:-}" ]; then
    if [ -n "${PET_NODE_RANK:-}" ]; then
        NODE_RANK="${PET_NODE_RANK}"
    elif [ -n "${SLURM_NODEID:-}" ]; then
        NODE_RANK="${SLURM_NODEID}"
    else
        NODE_RANK=0
    fi
fi

TP_SIZE="${TP_SIZE:-1}"
DIST_TIMEOUT="${DIST_TIMEOUT:-3600}"

TARGET_MODEL_BACKEND="${TARGET_MODEL_BACKEND:-hf}"
LW_CE_TAG="${LOSS_WEIGHT_CE//./p}"
LW_KL_TAG="${LOSS_WEIGHT_KL//./p}"
LW_MSE_TAG="${LOSS_WEIGHT_MSE//./p}"
if [ "${LOSS_KL_TOPK}" -gt 0 ]; then
    KL_TOPK_TAG="k${LOSS_KL_TOPK}"
else
    KL_TOPK_TAG="kall"
fi
LOSS_TAG="ce${LW_CE_TAG}_kl${LW_KL_TAG}_${KL_TOPK_TAG}_mse${LW_MSE_TAG}"

if [ "$DT" = "qz" ]; then
    TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/FlashMTP/cache/data/regen_data/nemotron_${DATA_NUM_SAMPLES}/nemotron_think_${ENABLE_THINKING}_samples_${DATA_NUM_SAMPLES}_qwen3_8b_regen.jsonl}"
    OUTPUT_DIR="${OUTPUT_DIR:-./cache/models/flashmtp_v3.2_${LOSS_TAG}_sample_${DATA_NUM_SAMPLES}_think_${ENABLE_THINKING}_qwen3_8b_maxlen${MAX_LENGTH}_epochs${NUM_EPOCHS}_dist}"
    TARGET_MODEL="${TARGET_MODEL:-$WHZ_DIR/models/Qwen/Qwen3-8B}"
    export WANDB_MODE=offline
else
    TRAIN_DATA_PATH="/share/wanghanzhen/SpeculativeDecoding/NIPS26/FlashMTP_v1.1/cache/data/regen_data/nemotron_40000/nemotron_think_on_samples_40000_qwen3_8b_regen.jsonl"
    OUTPUT_DIR="${OUTPUT_DIR:-./cache/models/flashmtp_v3.2_${LOSS_TAG}_nemotron_think_on_samples_40000_qwen3_8b}"
    TARGET_MODEL="${TARGET_MODEL:-/share/public/public_models/Qwen3-8B}"
fi

EVAL_DATA_PATH="${EVAL_DATA_PATH:-}"
CACHE_DIR="${CACHE_DIR:-./cache/data/regen_data/nemotron_${DATA_NUM_SAMPLES}}"

LOG_INTERVAL="${LOG_INTERVAL:-50}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"

REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-flashmtp_v3.2-training}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_DIR="${WANDB_DIR:-./wandb}"
WANDB_RUN_ID="${WANDB_RUN_ID:-flashmtp_v3.2_${LOSS_TAG}_${DATA_NUM_SAMPLES}_epochs${NUM_EPOCHS}_dist}"

CHAT_TEMPLATE="${CHAT_TEMPLATE:-qwen3-thinking}"
IS_PREFORMATTED="${IS_PREFORMATTED:-}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
BUILD_DATASET_NUM_PROC="${BUILD_DATASET_NUM_PROC:-8}"


echo "=========================================="
echo "FlashMTP 训练 (${DT})"
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
echo "  条件窗长 W (context-window-size): ${CONTEXT_WINDOW_SIZE}"
echo "  锚点数量: ${NUM_ANCHORS}"
echo "  Attention后端: ${ATTENTION_BACKEND}"
echo "  Diffusion mask schedule: ${DIFFUSION_MASK_SCHEDULE}"
echo "  Loss衰减Gamma: ${LOSS_DECAY_GAMMA:-未设置(不启用)} (w∝exp(-j/γ)，j=块内位置, anchor j=0不监督)"
echo "  Loss权重: CE=${LOSS_WEIGHT_CE}, KL=${LOSS_WEIGHT_KL}, MSE=${LOSS_WEIGHT_MSE}"
echo "  KL词表范围: ${LOSS_KL_TOPK} (0=全词表, >0=teacher top-k)"
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
echo "  PET_MASTER_ADDR: ${PET_MASTER_ADDR:-<未设置>}"
echo "  PET_MASTER_PORT: ${PET_MASTER_PORT:-<未设置>}"
echo "  PET_NNODES: ${PET_NNODES:-<未设置>}"
echo "  PET_NPROC_PER_NODE: ${PET_NPROC_PER_NODE:-<未设置>}"
echo "  PET_NODE_RANK: ${PET_NODE_RANK:-<未设置>}"
echo "  实际使用 MASTER_ADDR: ${MASTER_ADDR}"
echo "  实际使用 MASTER_PORT: ${MASTER_PORT}"
echo "  NNODES: ${NNODES}"
echo "  NODE_RANK: ${NODE_RANK}"
echo "  期望 WORLD_SIZE: $((NNODES * NPROC_PER_NODE)) (RANK/WORLD_SIZE 在 torchrun/子进程内)"
echo "  当前壳层 RANK/WORLD_SIZE: RANK=${RANK:-<未置>}, WORLD_SIZE=${WORLD_SIZE:-<未置>}"
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

original_output_dir="${OUTPUT_DIR}"
suffix=1
while [ -d "${OUTPUT_DIR}" ] && [ -n "$(ls -A "${OUTPUT_DIR}" 2>/dev/null)" ]; do
    OUTPUT_DIR="${original_output_dir}_${suffix}"
    suffix=$((suffix + 1))
done
if [ "${OUTPUT_DIR}" != "${original_output_dir}" ]; then
    echo "警告: 输出目录 ${original_output_dir} 已存在且非空，自动切换到: ${OUTPUT_DIR}"
fi

mkdir -p ${OUTPUT_DIR}
mkdir -p ${CACHE_DIR}
mkdir -p ${WANDB_DIR}

echo ""
echo "==> train_flashmtp.py (torchrun)"
LAUNCHER=(
    torchrun
    --nproc_per_node "${NPROC_PER_NODE}"
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)
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

set +e
"${LAUNCHER[@]}" ./scripts/train_flashmtp.py \
    --target-model-path ${TARGET_MODEL} \
    --target-model-backend ${TARGET_MODEL_BACKEND} \
    --train-data-path "${TRAIN_DATA_PATH}" \
    --output-dir ${OUTPUT_DIR} \
    --cache-dir ${CACHE_DIR} \
    --num-draft-layers ${NUM_DRAFT_LAYERS} \
    --block-size ${BLOCK_SIZE} \
    --num-anchors ${NUM_ANCHORS} \
    --context-window-size ${CONTEXT_WINDOW_SIZE} \
    --attention-backend ${ATTENTION_BACKEND} \
    --diffusion-mask-schedule ${DIFFUSION_MASK_SCHEDULE} \
    --loss-weight-ce ${LOSS_WEIGHT_CE} \
    --loss-weight-kl ${LOSS_WEIGHT_KL} \
    --loss-weight-mse ${LOSS_WEIGHT_MSE} \
    --loss-kl-topk ${LOSS_KL_TOPK} \
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
    ${OPTIONAL_ARGS}
EXIT_CODE=$?
set -e

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "=========================================="
    echo "训练失败 (退出码: $EXIT_CODE)"
    echo "=========================================="
    exit $EXIT_CODE
fi

echo ""
echo "=========================================="
echo "训练完成"
echo "=========================================="
echo "保存路径: ${OUTPUT_DIR}  |  加载: FlashMTPDraftModel.from_pretrained('<epoch_dir>')"
echo "=========================================="
