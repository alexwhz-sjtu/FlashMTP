#!/bin/bash
# DFlash 训练启动脚本
#
# 分布式环境变量（调度器 / torchrun 常见）本脚本会识别并用于构建 torchrun：
#   MASTER_ADDR, MASTER_PORT
#   PET_MASTER_ADDR, PET_MASTER_PORT
#   PET_NPROC_PER_NODE, PET_NNODES, PET_NODE_RANK
#   WORLD_SIZE, RANK（仅用于日志展示；由 torchrun/训练进程在运行时设置）
# 单机多卡：不设置或 NNODES=1、MASTER_ADDR=127.0.0.1 即可（脚本默认）。
# 多机多卡：在每台机器上执行本脚本，并设置同一 MASTER_ADDR/MASTER_PORT、
# 相同 NNODES、不同的 NODE_RANK（或 PET_NNODES / PET_NODE_RANK 等效变量）。

set -e

# 自动激活虚拟环境
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
if [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
    source "${PROJECT_DIR}/.venv/bin/activate"
fi

# ========================================
# 配置参数
# ========================================

# GPU / 进程数：PET_NPROC_PER_NODE 优先于本脚本的 NPROC_PER_NODE
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
if [ -n "${PET_NPROC_PER_NODE:-}" ]; then
    NPROC_PER_NODE="${PET_NPROC_PER_NODE}"
fi

# 分布式 rendezvous：MASTER_ADDR / MASTER_PORT
# - 未显式设置 MASTER_ADDR 时：优先 PET_MASTER_ADDR，再 SLURM / PBS 首节点，再否则 127.0.0.1
# - 未显式设置 MASTER_PORT 时：优先 PET_MASTER_PORT，再 MASTER_PORT，再 29501
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

# 目标模型路径
TARGET_MODEL="${TARGET_MODEL:-$WHZ_DIR/models/Qwen/Qwen3-8B}"
TARGET_MODEL_BACKEND="${TARGET_MODEL_BACKEND:-hf}"  # hf 或 sglang

# 训练参数
NUM_EPOCHS="${NUM_EPOCHS:-6}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-6e-4}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
WARMUP_RATIO="${WARMUP_RATIO:-0.04}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

# 数据特征参数（用于自动构建数据路径）
DATA_NUM_SAMPLES="${DATA_NUM_SAMPLES:-400000}"
ENABLE_THINKING="${ENABLE_THINKING:-on}"

# 构建数据子目录名: n{N|all}_think_{on|off}
DATASET_BASE_DIR="${DATASET_BASE_DIR:-./cache/dataset}"
if [ "${ENABLE_THINKING}" = "on" ] || [ "${ENABLE_THINKING}" = "true" ] || [ "${ENABLE_THINKING}" = "1" ]; then
    THINK_STR="on"
else
    THINK_STR="off"
fi
DATA_SUBDIR="n${DATA_NUM_SAMPLES}_think_${THINK_STR}"

# 数据目录（支持通过 TRAIN_DATA_PATH 直接指定，否则自动构建）
TRAIN_DATA_PATH="./cache/data/regen_data/nemotron_400000/nemotron_think_on_samples_400000_qwen3_8b_regen.jsonl"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-./cache/models/dflash_sample_400000_think_on_qwen3_8b_maxlen${MAX_LENGTH}}"
CACHE_DIR="./cache/data/regen_data/nemotron_400000"

# 模型参数
NUM_DRAFT_LAYERS="${NUM_DRAFT_LAYERS:-5}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
NUM_ANCHORS="${NUM_ANCHORS:-512}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flex_attention}"
LOSS_DECAY_GAMMA="${LOSS_DECAY_GAMMA:-7}"  # 建议: block_size=16用7, 10用5, 8用4

# 日志和保存间隔
LOG_INTERVAL="${LOG_INTERVAL:-50}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10000}"

# Tracker 参数
REPORT_TO="${REPORT_TO:-wandb}"  # none, wandb, tensorboard
WANDB_PROJECT="${WANDB_PROJECT:-flashmtp-training}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_DIR="${WANDB_DIR:-./wandb}"  # 离线日志保存目录
WANDB_RUN_ID="${WANDB_RUN_ID:-dflash_400000}"   # 离线子目录名称 (如: my_run_001，生成 offline-run-my_run_001)

# 分布式参数
TP_SIZE="${TP_SIZE:-1}"
DIST_TIMEOUT="${DIST_TIMEOUT:-30}"

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
echo "  思考模式: ${THINK_STR}"
echo "  数据子目录: ${DATA_SUBDIR}"
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
echo "  PET_MASTER_ADDR: ${PET_MASTER_ADDR:-<未设置>}"
echo "  PET_MASTER_PORT: ${PET_MASTER_PORT:-<未设置>}"
echo "  实际使用 MASTER_ADDR: ${MASTER_ADDR}"
echo "  实际使用 MASTER_PORT: ${MASTER_PORT}"
echo "  PET_NNODES: ${PET_NNODES:-<未设置>}"
echo "  PET_NPROC_PER_NODE: ${PET_NPROC_PER_NODE:-<未设置>}"
echo "  PET_NODE_RANK: ${PET_NODE_RANK:-<未设置>}"
echo "  NNODES: ${NNODES}"
echo "  NODE_RANK: ${NODE_RANK}"
echo "  期望 WORLD_SIZE (未启动前): $((NNODES * NPROC_PER_NODE))"
echo "  运行时 RANK / WORLD_SIZE: 由 torchrun/训练进程设置; 当前: RANK=${RANK:-<未启动>}, WORLD_SIZE=${WORLD_SIZE:-<未启动>}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "  TP_SIZE: ${TP_SIZE}"
echo "------------------------------------------"
echo "Tracker: ${REPORT_TO}"
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

# ========================================
# 训练
# ========================================
echo ""
echo "==> 开始训练 DFlash"
echo ""

# torchrun：FSDP/NCCL 需正确 RANK/WORLD_SIZE；单卡/单进程也使用
#   torchrun --nproc_per_node=1 --nnodes=1 以保证 init_process_group 与多机时一致
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
fi

"${LAUNCHER[@]}"    ./scripts/train_dflash.py \
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
    --seed 42 \
    ${OPTIONAL_ARGS}

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
echo "  draft_model = DFlashDraftModel.from_pretrained('${OUTPUT_DIR}/epoch_6_step_<step>')"
echo ""
echo "运行推理："
echo "  python benchmark.py --draft-model ${OUTPUT_DIR}/epoch_6_step_<step>"
echo "=========================================="
