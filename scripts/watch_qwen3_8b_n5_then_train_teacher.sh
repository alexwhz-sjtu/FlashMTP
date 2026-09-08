#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="192.168.1.252"
EXPECTED_PER_SHARD=167410
LOCAL_LAUNCHER_PID="${LOCAL_LAUNCHER_PID:?set local generation launcher PID}"
REMOTE_LAUNCHER_PID="${REMOTE_LAUNCHER_PID:?set remote generation launcher PID}"
DATA_DIR=/data/wanghanzhen/training_data/generated/qwen3-8b
PART0="$DATA_DIR/math_code_chat_aug_part_00_think_off_temp1.0_topp0.9_n5_maxnew4096.jsonl"
PART1="$DATA_DIR/math_code_chat_aug_part_01_think_off_temp1.0_topp0.9_n5_maxnew4096.jsonl"
ERROR0="${PART0%.jsonl}_error.jsonl"
ERROR1="${PART1%.jsonl}_error.jsonl"
MERGED="$DATA_DIR/math_code_chat_aug_think_off_temp1.0_topp0.9_n5_maxnew4096.jsonl"
TRAIN_SCRIPT=/data/wanghanzhen/FlashMTP_v2.3/scripts/run_teacher_math_code_chat_qwen3_8b_d10_2n16g.sh
STATE_DIR=/data/wanghanzhen/FlashMTP_v2.3/cache/automation
mkdir -p "$STATE_DIR"

while kill -0 "$LOCAL_LAUNCHER_PID" 2>/dev/null || \
      ssh -o BatchMode=yes -o ConnectTimeout=15 root@"$REMOTE_HOST" \
        "kill -0 $REMOTE_LAUNCHER_PID" 2>/dev/null; do
  local_count=$([ -f "$PART0" ] && wc -l < "$PART0" || echo 0)
  remote_count=$(ssh -o BatchMode=yes -o ConnectTimeout=15 root@"$REMOTE_HOST" \
    "f='$PART1'; [ -f \"\$f\" ] && wc -l < \"\$f\" || echo 0")
  printf '%s generating local=%s remote=%s\n' "$(date '+%F %T')" "$local_count" "$remote_count"
  sleep 120
done

local_success=$([ -f "$PART0" ] && wc -l < "$PART0" || echo 0)
local_errors=$([ -f "$ERROR0" ] && wc -l < "$ERROR0" || echo 0)
read -r remote_success remote_errors < <(
  ssh -o BatchMode=yes -o ConnectTimeout=15 root@"$REMOTE_HOST" \
    "s='$PART1'; e='$ERROR1'; printf '%s %s\n' \"\$([ -f \"\$s\" ] && wc -l < \"\$s\" || echo 0)\" \"\$([ -f \"\$e\" ] && wc -l < \"\$e\" || echo 0)\""
)

if [ $((local_success + local_errors)) -ne "$EXPECTED_PER_SHARD" ] || \
   [ $((remote_success + remote_errors)) -ne "$EXPECTED_PER_SHARD" ]; then
  printf 'generation_incomplete local=%s+%s remote=%s+%s expected=%s\n' \
    "$local_success" "$local_errors" "$remote_success" "$remote_errors" \
    "$EXPECTED_PER_SHARD" > "$STATE_DIR/qwen3_8b_n5_then_train.state"
  exit 2
fi

test ! -e "$PART1"
test ! -e "$MERGED"
scp -p root@"$REMOTE_HOST":"$PART1" "$PART1"
/data/wanghanzhen/FlashMTP_v2.3/.venv/bin/python \
  /data/wanghanzhen/FlashMTP_v2.3/scripts/mix_data.py \
  --inputs "$PART0" "$PART1" --output "$MERGED" --seed 42
scp -p "$MERGED" root@"$REMOTE_HOST":"$MERGED"

merged_count=$(wc -l < "$MERGED")
if [ "$merged_count" -ne $((local_success + remote_success)) ]; then
  printf 'merge_count_mismatch merged=%s expected=%s\n' \
    "$merged_count" $((local_success + remote_success)) \
    > "$STATE_DIR/qwen3_8b_n5_then_train.state"
  exit 3
fi

if ss -ltn "sport = :29567" | grep -q LISTEN; then
  printf 'master_port_in_use port=29567\n' > "$STATE_DIR/qwen3_8b_n5_then_train.state"
  exit 4
fi

ssh -o BatchMode=yes -o ConnectTimeout=15 root@"$REMOTE_HOST" \
  "mkdir -p /data/wanghanzhen/FlashMTP_v2.3/logs; nohup env NODE_RANK=1 bash '$TRAIN_SCRIPT' > /data/wanghanzhen/FlashMTP_v2.3/logs/teacher_math_code_chat_qwen3_8b_d10_r512_node1.log 2>&1 < /dev/null & echo \$!"
sleep 3
mkdir -p /data/wanghanzhen/FlashMTP_v2.3/logs
nohup env NODE_RANK=0 bash "$TRAIN_SCRIPT" \
  > /data/wanghanzhen/FlashMTP_v2.3/logs/teacher_math_code_chat_qwen3_8b_d10_r512_node0.log \
  2>&1 < /dev/null &
printf 'training_started merged=%s local_errors=%s remote_errors=%s node0_pid=%s\n' \
  "$merged_count" "$local_errors" "$remote_errors" "$!" \
  > "$STATE_DIR/qwen3_8b_n5_then_train.state"
