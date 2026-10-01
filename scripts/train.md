## DLite training recipe

```bash
/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/stop_keeper.sh  

cd /inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/FlashMTP_dlite

NODE_JIT_CACHE="/tmp/flashmtp-jit-$(hostname)"
export TVM_FFI_CACHE_DIR="${NODE_JIT_CACHE}/tvm-ffi"
export TORCHINDUCTOR_CACHE_DIR="${NODE_JIT_CACHE}/torchinductor"
export TRITON_CACHE_DIR="${NODE_JIT_CACHE}/triton"
source .venv/bin/activate

CHS_NUM_LAYERS=14 \
BLOCK_SIZE=8 \
NUM_DRAFT_LAYERS=5 \
TARGET_MODEL_BACKEND=sglang \
SGLANG_MEM_FRACTION_STATIC=0.3 \
NUM_EPOCHS=10 \
NUM_ANCHORS=768 \
MAX_LENGTH=10240 \
BATCH_SIZE=1 \
LOSS_DECAY_GAMMA=4 \
DATA_NUM_SAMPLES=full_aug1_chinese_q3_4b \
BASE_LM_CE_DECAY_GAMMA=12 \
ACCUMULATION_STEPS=1 \
LEARNING_RATE=4e-4 \
FINAL_CE_WEIGHT=0.1 \
TV_LOSS_WEIGHT=1.0 \
BASE_LM_CE_WEIGHT=0.0 \
SEQUENTIAL_HEAD=rnn \
SEQUENTIAL_RANK=320 \
SHARD_DRAFT_BY_TP=0 \
TP_SIZE=1 \
TRAIN_DATA_PATH='/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/FlashMTP/cache/data/regen_data/qwen3_4b/mixed_2.3M_qwen3_4b_aug1_math_code_chat_temp1_chinese.jsonl' \
MODEL_TAG='Qwen3_4B' \
TARGET_MODEL='/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/models/Qwen/Qwen3-4B' \
bash scripts/run_training_dlite_sft.sh --dt qz > "whz_mtp_logs/train_dlite_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

# ========== 关键修改：等待训练完成 ==========
TRAIN_PID=$!
echo "训练已启动，PID: $TRAIN_PID，等待训练完成..."

if wait "$TRAIN_PID"; then
  TRAIN_EXIT=0
else
  TRAIN_EXIT=$?
fi

echo "训练已结束，退出码: $TRAIN_EXIT"

/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/start_keeper.sh

exit "$TRAIN_EXIT"

```
