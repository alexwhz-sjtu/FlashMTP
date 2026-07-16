#!/bin/bash
# 调试版训练脚本 - 解决卡死问题

cd /inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/FlashMTP_v1.3
mkdir -p whz_mtp_logs

# NCCL调试和超时设置
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0  # 根据实际集群网络接口修改
export NCCL_TIMEOUT=1800        # 30分钟NCCL超时

# PyTorch分布式设置
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_BLOCKING_WAIT=1  # 让NCCL错误立即抛出

# CUDA设置
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 减少数据加载worker数量，避免资源竞争
DATALOADER_NUM_WORKERS=2
BUILD_DATASET_NUM_PROC=4

# 日志间隔缩短，方便观察是否卡住
LOG_INTERVAL=10
SAVE_INTERVAL=1000

LOCAL_POSITION=true \
NUM_EPOCHS=6 \
BLOCK_SIZE=16 \
NUM_MIDDLE_LAYERS_N=16 \
DATA_NUM_SAMPLES=2.3m \
MAX_LENGTH=20480 \
NUM_ANCHORS=768 \
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS} \
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC} \
DFLASH_TEACHER_PATH='/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/FlashMTP/cache/models/dflash_mix_sample_14_think_off_qwen3_8b_maxlen20480_nnodes4/epoch_5_step_290000' \
TRAIN_DATA_PATH='/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/FlashMTP/cache/data/regen_data/mixed/mixed_h100_think_off_samples_2350325_qwen3_8b_regen.jsonl' \
DFLASH_ALIGN_MODE=final DFLASH_MILESTONE_EPOCH=2 \
LOSS_DECAY_GAMMA=7 DFLASH_DISTILL_DECAY_GAMMA=14 \
DFLASH_CE_POS_MODE=prefix DFLASH_CE_WEIGHT=0.6 DFLASH_CE_MIN_SCALE=0.0 \
DFLASH_DISTILL_POS_MODE=all DFLASH_DISTILL_WEIGHT=1.0 DFLASH_DISTILL_MIN_SCALE=0.4 \
DFLASH_DISTILL_TEMPERATURE=1.0 DFLASH_DISTILL_TOP_K=64 \
LOG_INTERVAL=${LOG_INTERVAL} \
SAVE_INTERVAL=${SAVE_INTERVAL} \
DIST_TIMEOUT=3600 \
bash scripts/run_training_flashmtp.sh --dt qz \
   > "whz_mtp_logs/train_flashmtp_debug_$(date +%Y%m%d_%H%M%S).log" 2>&1
