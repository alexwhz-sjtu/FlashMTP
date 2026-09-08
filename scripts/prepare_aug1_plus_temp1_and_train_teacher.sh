#!/usr/bin/env bash
set -euo pipefail

XIAOYUAN_HOST=192.168.1.249
BASE_DATA=/data/wanghanzhen/training_data/mixed_2360k_qwen3_8b_nm_pb_swe_aug1.jsonl
TEMP1_DATA=/data/wanghanzhen/training_data/generated/qwen3-4b/math_code_chat_aug_think_off_temp1.0_topp0.9_n5_maxnew4096.jsonl
MIXED_DATA=/data/wanghanzhen/training_data/mixed_2360k_qwen3_8b_aug1_plus_math_code_chat_temp1_n5_2693216_shuffled.jsonl
CACHE_ARCHIVE=/data/wanghanzhen/archives/FlashMTP_v2.3_dataset_caches_from_xiaoyuan
STATE_DIR=/data/wanghanzhen/FlashMTP_v2.3/cache/automation
TRAIN_SCRIPT=/data/wanghanzhen/FlashMTP_v2.3/scripts/run_teacher_aug1_plus_temp1_qwen3_8b_d10_2n16g.sh

mkdir -p "$STATE_DIR"
while [ ! -f "$TEMP1_DATA" ] || [ ! -d "$CACHE_ARCHIVE" ]; do
  printf '%s waiting temp1=%s cache_archive=%s\n' \
    "$(date '+%F %T')" "$([ -f "$TEMP1_DATA" ] && echo ready || echo pending)" \
    "$([ -d "$CACHE_ARCHIVE" ] && echo ready || echo pending)"
  sleep 60
done

base_count=$(wc -l < "$BASE_DATA")
temp1_count=$(wc -l < "$TEMP1_DATA")
if [ "$base_count" -ne 2358681 ] || [ "$temp1_count" -ne 334535 ]; then
  printf 'input_count_mismatch base=%s temp1=%s\n' "$base_count" "$temp1_count" \
    > "$STATE_DIR/aug1_plus_temp1_teacher.state"
  exit 2
fi

cache_names=(
  train_aug1_maxlen10240
  train_aug1_maxlen20480
  train_aug1_qwen3_4b_maxlen10240
  train_math_code_chat_qwen3_8b_n5_maxlen10240
)
for cache_name in "${cache_names[@]}"; do
  remote_path="/data/wanghanzhen/FlashMTP_v2.3/cache/$cache_name"
  archive_path="$CACHE_ARCHIVE/$cache_name"
  remote_bytes=$(ssh -o BatchMode=yes -o ConnectTimeout=15 root@"$XIAOYUAN_HOST" \
    "du -sb '$remote_path' | cut -f1")
  archive_bytes=$(du -sb "$archive_path" | cut -f1)
  if [ "$remote_bytes" != "$archive_bytes" ]; then
    printf 'cache_archive_mismatch name=%s remote=%s archive=%s\n' \
      "$cache_name" "$remote_bytes" "$archive_bytes" \
      > "$STATE_DIR/aug1_plus_temp1_teacher.state"
    exit 3
  fi
done

ssh -o BatchMode=yes -o ConnectTimeout=15 root@"$XIAOYUAN_HOST" \
  'rm -rf -- /data/wanghanzhen/FlashMTP_v2.3/cache/train_aug1_maxlen10240 /data/wanghanzhen/FlashMTP_v2.3/cache/train_aug1_maxlen20480 /data/wanghanzhen/FlashMTP_v2.3/cache/train_aug1_qwen3_4b_maxlen10240 /data/wanghanzhen/FlashMTP_v2.3/cache/train_math_code_chat_qwen3_8b_n5_maxlen10240'

test ! -e "$MIXED_DATA"
/data/wanghanzhen/FlashMTP_v2.3/.venv/bin/python \
  /data/wanghanzhen/FlashMTP_v2.3/scripts/mix_data.py \
  --inputs "$BASE_DATA" "$TEMP1_DATA" --output "$MIXED_DATA" --seed 42

mixed_count=$(wc -l < "$MIXED_DATA")
if [ "$mixed_count" -ne 2693216 ]; then
  printf 'mixed_count_mismatch actual=%s expected=2693216\n' "$mixed_count" \
    > "$STATE_DIR/aug1_plus_temp1_teacher.state"
  exit 4
fi

remote_tmp="${MIXED_DATA}.uploading"
ssh -o BatchMode=yes -o ConnectTimeout=15 root@"$XIAOYUAN_HOST" \
  "test ! -e '$remote_tmp'; test ! -e '$MIXED_DATA'"
scp -p "$MIXED_DATA" root@"$XIAOYUAN_HOST":"$remote_tmp"
ssh -o BatchMode=yes -o ConnectTimeout=15 root@"$XIAOYUAN_HOST" \
  "mv '$remote_tmp' '$MIXED_DATA'; test \"\$(wc -l < '$MIXED_DATA')\" -eq 2693216"

if ss -ltn "sport = :29568" | grep -q LISTEN; then
  printf 'master_port_in_use port=29568\n' > "$STATE_DIR/aug1_plus_temp1_teacher.state"
  exit 5
fi

ssh -o BatchMode=yes -o ConnectTimeout=15 root@"$XIAOYUAN_HOST" \
  "mkdir -p /data/wanghanzhen/FlashMTP_v2.3/logs; nohup env NODE_RANK=1 bash '$TRAIN_SCRIPT' > /data/wanghanzhen/FlashMTP_v2.3/logs/teacher_aug1_plus_temp1_qwen3_8b_node1.log 2>&1 < /dev/null & echo \$!"
sleep 3
mkdir -p /data/wanghanzhen/FlashMTP_v2.3/logs
nohup env NODE_RANK=0 bash "$TRAIN_SCRIPT" \
  > /data/wanghanzhen/FlashMTP_v2.3/logs/teacher_aug1_plus_temp1_qwen3_8b_node0.log \
  2>&1 < /dev/null &
printf 'training_started rows=%s node0_pid=%s\n' "$mixed_count" "$!" \
  > "$STATE_DIR/aug1_plus_temp1_teacher.state"
