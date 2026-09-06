# tau2 harness integration

The local OpenAI-compatible server exposes the FlashMTP v2swa draft checkpoint
with the Qwen3-4B target. It records one JSONL row for every model request,
including the prompt length at request start, the complete speculative acceptance
sequence, generation/decode time, and throughput.

Start the server on an available GPU:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m evaluation.openai_server \
  --device cuda:0 \
  --port 18001 \
  --metrics-jsonl benchmark_results/tau2_v2swa_yarn4x_metrics.jsonl \
  --context-limit 163840 \
  --rope-scaling yarn \
  --rope-factor 4 \
  --original-max-position-embeddings 40960
```

Install tau2 without rewriting its currently stale lock file, then run a task:

```bash
cd /share/dai-sys/wanghanzhen/datasets/tau2-bench
uv sync --frozen

cd /share/dai-sys/wanghanzhen/projects/MTP/FlashMTP_v2swa
TAU2_DOMAIN=retail TAU2_TASK_IDS='0' TAU2_MAX_STEPS=200 \
  scripts/run_tau2_harness.sh
```

The server is batch-1, so the wrapper fixes tau2 concurrency at one. In the
default setup the same local model is used as both the agent and the user
simulator. They use different request model names, so agent metrics can be
selected exactly:

```bash
.venv/bin/python scripts/summarize_agent_metrics.py \
  benchmark_results/tau2_v2swa_yarn4x_metrics.jsonl \
  --run-id RUN_ID \
  --model Qwen3-4B-FlashMTP-v2swa-agent \
  --csv benchmark_results/RUN_ID-agent-turns.csv
```

`context_tokens_at_turn_start` is the fully rendered prompt length for that
agent request, including system policy, tools, conversation history, and tool
results. `accept_lengths` contains emitted tokens per speculative verification
step, including one target anchor/correction token; subtract one to obtain the
number of accepted draft tokens. `generation_tokens_per_s` includes prefill and
decode, while `decode_tokens_per_s` uses the model-reported decode interval.

The endpoint enforces a 163,840-token combined prompt/output budget. This is a
runtime YaRN extension (`4 * 40960`), not evidence that the checkpoint was
trained or quality-validated at that length.
