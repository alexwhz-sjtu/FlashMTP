# SkillsBench agent-harness evaluation

This adapter serves the v2swa draft and its Qwen3-4B target through OpenAI Chat
Completions. It supports full message history, Qwen tool calls, streaming, and a
target-only baseline. Every request writes the context length at the start of
that agent turn and the complete FlashMTP acceptance sequence to JSONL.

## Start the model server

From the FlashMTP repository:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m evaluation.openai_server \
  --device cuda:0 \
  --port 18000 \
  --metrics-jsonl benchmark_results/skillsbench_request_metrics.jsonl
```

The defaults point to:

- target: `/share/dai-sys/wanghanzhen/models/Qwen/Qwen3-4B`
- draft: `/share/dai-sys/wanghanzhen/projects/MTP/FlashMTP_v2swa/cache/models/flashmtp_v2swa_w5_qwen3_4b_ep10`
- context limit: 163,840 tokens via 4× YaRN (`original_max=40960`)
- per-request output cap: 2,048 tokens

The server exposes `/health`, `/v1/models`, `/v1/chat/completions`,
`/metrics/requests`, and `/admin/config`. Generation is serialized because the
current FlashMTP implementation is batch-1 and stores acceptance statistics on
the draft model.

## Run multiple harnesses

From the SkillsBench repository, after `uv sync --locked`:

```bash
.venv/bin/python tools/run_flashmtp_harnesses.py \
  --server-url http://127.0.0.1:18000 \
  --task dialogue-parser \
  --harness opencode \
  --harness pi-acp \
  --mode flashmtp \
  --mode target
```

These are also the runner defaults. The local BenchFlow installation is patched
to execute the current OpenCode native binary directly and to create its
`flashmtp` custom-provider configuration before `opencode acp --pure` starts.

The runner automatically uses the Docker bridge gateway for requests originating
inside task containers. Override it with `--container-server-url` when Docker is
remote or uses a nonstandard network. Runs are sequential so `/admin/config` can
tag every request with its task, harness, mode, and run ID.

Multi-turn support does not need a separate switch: the harness sends the full
`messages` history after every assistant tool call and tool result. The server
re-renders and tokenizes that history, so `context_tokens_at_turn_start` is the
actual Qwen-tokenizer length seen by the model for each turn.

## Inspect context and acceptance lengths

```bash
.venv/bin/python scripts/summarize_agent_metrics.py \
  benchmark_results/skillsbench_request_metrics.jsonl \
  --csv benchmark_results/skillsbench_request_metrics.csv
```

Important fields:

- `turn_index`: number of user messages in the request.
- `context_tokens_at_turn_start`: complete rendered prompt length, including the
  system prompt, skill text supplied by the harness, tool schemas, earlier
  assistant calls, and tool results.
- `accept_lengths`: emitted tokens per speculative verification step. The
  existing FlashMTP metric includes one target anchor/correction token per step.
- `draft_match_lengths`: `accept_lengths - 1`, the number of draft tokens that
  actually matched in each step.
- `generation_wall_time_s`: prefill plus decode plus server-side generation
  overhead. Compare target/FlashMTP only when prompts and outputs match.
- `decode_wall_time_s`: decode-only timing from the model implementation.

## Long context

The checkpoint's native `max_position_embeddings` is 40,960. The server applies
4× YaRN at load time to both target and draft configs, giving a declared runtime
limit of 163,840. This makes 64K requests executable, but it does not prove model
quality at that length: the checkpoint was not trained at 163K. Use the logged
acceptance and output-consistency measurements to validate the extended regime.

Disable the override for native-context comparisons with:

```bash
--rope-scaling none --context-limit 40960
```
