#!/usr/bin/env bash
set -euo pipefail

TAU2_ROOT="${TAU2_ROOT:-/share/dai-sys/wanghanzhen/datasets/tau2-bench}"
FLASHMTP_SERVER_URL="${FLASHMTP_SERVER_URL:-http://127.0.0.1:18001}"
TAU2_DOMAIN="${TAU2_DOMAIN:-retail}"
TAU2_TASK_IDS="${TAU2_TASK_IDS:-0}"
TAU2_MAX_STEPS="${TAU2_MAX_STEPS:-200}"
TAU2_TIMEOUT="${TAU2_TIMEOUT:-1800}"
AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-512}"
USER_MAX_TOKENS="${USER_MAX_TOKENS:-256}"
RUN_ID="${RUN_ID:-tau2-${TAU2_DOMAIN}-$(date +%Y%m%d-%H%M%S)}"
SAVE_TO="${SAVE_TO:-${RUN_ID}}"
AGENT_MODEL="${AGENT_MODEL:-Qwen3-4B-FlashMTP-v2swa-agent}"
USER_MODEL="${USER_MODEL:-Qwen3-4B-FlashMTP-v2swa-user}"

if [[ ! -x "${TAU2_ROOT}/.venv/bin/tau2" ]]; then
  echo "tau2 environment is missing: run 'uv sync --frozen' in ${TAU2_ROOT}" >&2
  exit 1
fi

health_json="$(curl -fsS "${FLASHMTP_SERVER_URL}/health")"
python3 -c '
import json, sys
health = json.loads(sys.argv[1])
expected = 4 * 40960
actual = int(health.get("context_limit", 0))
rope = health.get("rope_scaling") or {}
if actual != expected or rope.get("rope_type") != "yarn" or float(rope.get("factor", 0)) != 4.0:
    raise SystemExit(f"server is not configured for YaRN 4x40960: {health}")
' "${health_json}"

task_tag="${TAU2_DOMAIN}:${TAU2_TASK_IDS// /,}"
curl -fsS -X POST "${FLASHMTP_SERVER_URL}/admin/config" \
  -H 'Content-Type: application/json' \
  -d "{\"decode_mode\":\"flashmtp\",\"tags\":{\"harness\":\"tau2\",\"run_id\":\"${RUN_ID}\",\"task\":\"${task_tag}\"}}" \
  >/dev/null

agent_args="{\"api_base\":\"${FLASHMTP_SERVER_URL}/v1\",\"api_key\":\"EMPTY\",\"temperature\":0.0,\"max_tokens\":${AGENT_MAX_TOKENS}}"
user_args="{\"api_base\":\"${FLASHMTP_SERVER_URL}/v1\",\"api_key\":\"EMPTY\",\"temperature\":0.0,\"max_tokens\":${USER_MAX_TOKENS}}"
task_args=()
if [[ -n "${TAU2_TASK_IDS}" ]]; then
  read -r -a task_ids <<<"${TAU2_TASK_IDS}"
  task_args=(--task-ids "${task_ids[@]}")
fi

echo "run_id=${RUN_ID} domain=${TAU2_DOMAIN} tasks=${TAU2_TASK_IDS:-all}"
cd "${TAU2_ROOT}"
env -u ALL_PROXY -u HTTPS_PROXY -u HTTP_PROXY \
  -u all_proxy -u https_proxy -u http_proxy \
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  .venv/bin/tau2 run \
    --domain "${TAU2_DOMAIN}" \
    --agent llm_agent \
    --agent-llm "openai/${AGENT_MODEL}" \
    --agent-llm-args "${agent_args}" \
    --user user_simulator \
    --user-llm "openai/${USER_MODEL}" \
    --user-llm-args "${user_args}" \
    "${task_args[@]}" \
    --num-trials 1 \
    --max-steps "${TAU2_MAX_STEPS}" \
    --max-concurrency 1 \
    --timeout "${TAU2_TIMEOUT}" \
    --save-to "${SAVE_TO}" \
    --log-level INFO

echo "agent metrics filter: --run-id ${RUN_ID} --model ${AGENT_MODEL}"
