## DLite training recipe

```bash
cd /data/wanghanzhen/projects/SpecDecoding/FlashMTP_v2.3
source .venv/bin/activate

SWA_WINDOW_SIZE=1 \
CHS_NUM_LAYERS=14 \
TARGET_MODEL_BACKEND=sglang \
SGLANG_MEM_FRACTION_STATIC=0.3 \
LOCAL_POSITION=true \
BLOCK_SIZE=8 \
NUM_DRAFT_LAYERS=5 \
NUM_EPOCHS=6 \
NUM_ANCHORS=512 \
MAX_LENGTH=4096 \
BATCH_SIZE=1 \
LOSS_DECAY_GAMMA=4 \
DATA_NUM_SAMPLES=pb_80k_qwen3_4b \
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
TRAIN_DATA_PATH='/data/wanghanzhen/training_data/generated/qwen3-4b/open_perfectblend_80k_qwen3_4b.jsonl' \
MODEL_TAG='Qwen3_4B' \
TARGET_MODEL='/data/wanghanzhen/models/Qwen3-4B' \
bash scripts/run_training_dlite_teacher.sh
```
