"""OpenAI-compatible, batch-1 serving for FlashMTP agent harness evaluation.

The server intentionally serializes generation. FlashMTP's current decode path is
batch-1 and stores the most recent acceptance statistics on the draft model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import threading
import time
import uuid
from collections import Counter, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import DynamicCache

from evaluation.model_loading import load_flashmtp_benchmark_models
from specforge.modeling.draft.flashmtp import sample


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
DEFAULT_TARGET = "/share/dai-sys/wanghanzhen/models/Qwen/Qwen3-4B"
DEFAULT_DRAFT = (
    "/share/dai-sys/wanghanzhen/projects/MTP/FlashMTP_v2swa/cache/models/"
    "flashmtp_v2swa_w5_qwen3_4b_ep10"
)


def parse_tool_calls(text: str) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert Qwen's tagged tool calls into OpenAI Chat Completions objects."""
    calls: list[dict[str, Any]] = []
    for match in TOOL_CALL_RE.finditer(text):
        try:
            item = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict) or not item.get("name"):
            continue
        arguments = item.get("arguments", {})
        if isinstance(arguments, str):
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError:
                parsed_arguments = arguments
        else:
            parsed_arguments = arguments
        arguments_json = (
            parsed_arguments
            if isinstance(parsed_arguments, str)
            else json.dumps(parsed_arguments, ensure_ascii=False, separators=(",", ":"))
        )
        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": str(item["name"]), "arguments": arguments_json},
            }
        )
    content = TOOL_CALL_RE.sub("", text).strip()
    return (content or None), calls


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cuda_time(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


@torch.inference_mode()
def target_generate(
    target: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    stop_token_ids: list[int],
    temperature: float,
) -> SimpleNamespace:
    """Greedy/sampled target-only baseline with decode timing excluding prefill."""
    num_input_tokens = int(input_ids.shape[1])
    max_length = num_input_tokens + max_new_tokens
    output_ids = torch.empty((1, max_length), dtype=torch.long, device=input_ids.device)
    output_ids[:, :num_input_tokens] = input_ids
    position_ids = torch.arange(max_length, device=input_ids.device).unsqueeze(0)
    past_key_values = DynamicCache()

    prefill_start = _cuda_time(input_ids.device)
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=False,
    )
    next_token = sample(output.logits, temperature)
    prefill_wall_time = _cuda_time(input_ids.device) - prefill_start
    decode_start = _cuda_time(input_ids.device)
    start = num_input_tokens
    stop_set = set(stop_token_ids)
    while start < max_length:
        output_ids[:, start : start + 1] = next_token
        start += 1
        if int(next_token.item()) in stop_set or start >= max_length:
            break
        output = target(
            next_token,
            position_ids=position_ids[:, start - 1 : start],
            past_key_values=past_key_values,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=False,
        )
        next_token = sample(output.logits, temperature)
    decode_wall_time = _cuda_time(input_ids.device) - decode_start
    return SimpleNamespace(
        output_ids=output_ids[:, :start],
        decode_wall_time=decode_wall_time,
        prefill_wall_time=prefill_wall_time,
        accept_lengths=[],
        steps=0,
    )


class MetricsStore:
    def __init__(self, path: Path, retain: int = 1000) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: deque[dict[str, Any]] = deque(maxlen=retain)
        self.lock = threading.Lock()

    def append(self, row: dict[str, Any]) -> None:
        line = json.dumps(row, ensure_ascii=False)
        with self.lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            self.rows.append(row)

    def latest(self, limit: int) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.rows)[-max(0, limit) :]


class Runtime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device)
        self.target, self.draft, self.tokenizer, self.draft_config = (
            load_flashmtp_benchmark_models(args, self.device)
        )
        self.target_attention_backend = str(
            getattr(self.target.config, "_attn_implementation", "unknown")
        )
        self.draft_attention_backend = str(
            getattr(self.draft.config, "_attn_implementation", "unknown")
        )
        self.tokenizer.model_max_length = args.context_limit
        model_context_limit = int(self.target.config.max_position_embeddings)
        if args.context_limit > model_context_limit:
            raise ValueError(
                f"context_limit={args.context_limit} exceeds configured model limit "
                f"{model_context_limit}"
            )
        self.stop_token_ids = sorted(
            {
                int(token_id)
                for token_id in (
                    self.tokenizer.eos_token_id,
                    self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                )
                if token_id is not None and int(token_id) >= 0
            }
        )
        self.lock = asyncio.Lock()
        self.metrics = MetricsStore(Path(args.metrics_jsonl))
        self.tags: dict[str, str] = {}
        self.run_request_counts: Counter[str] = Counter()
        self.decode_mode = args.decode_mode

    def render(self, body: dict[str, Any]) -> tuple[str, torch.Tensor]:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="messages must be a non-empty list")
        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        tools = body.get("tools")
        if tools and body.get("tool_choice") != "none":
            template_kwargs["tools"] = tools
        try:
            prompt = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid chat payload: {exc}") from exc
        return prompt, self.tokenize_prompt(prompt)

    def tokenize_prompt(self, prompt: str) -> torch.Tensor:
        """Tokenize an already-rendered prompt for the legacy Completions API."""
        return self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

    def generate_sync(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        decode_mode: str,
    ) -> SimpleNamespace:
        started = _cuda_time(self.device)
        if decode_mode == "target":
            result = target_generate(
                self.target, input_ids, max_new_tokens, self.stop_token_ids, temperature
            )
        else:
            output_ids = self.draft.spec_generate(
                target=self.target,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                stop_token_ids=self.stop_token_ids,
                temperature=temperature,
                decode_timing_after_first_token=False,
                verify_block_size=self.args.verify_block,
                stochastic_verification_mode=self.args.stochastic_verification_mode,
                compile_serial_head=self.args.compile_serial_head,
            )
            stats = self.draft.get_last_decode_stats()
            result = SimpleNamespace(
                output_ids=output_ids,
                decode_wall_time=float(stats.get("decode_wall_time", 0.0)),
                prefill_wall_time=None,
                accept_lengths=[int(x) for x in stats.get("accept_lengths", [])],
                steps=int(stats.get("steps", 0)),
            )
        result.generation_wall_time = _cuda_time(self.device) - started
        return result


def _completion_chunks(
    response_id: str,
    model: str,
    created: int,
    content: str | None,
    tool_calls: list[dict[str, Any]],
    finish_reason: str,
    usage: dict[str, int],
    include_usage: bool,
) -> Iterator[str]:
    base = {"id": response_id, "object": "chat.completion.chunk", "created": created, "model": model}

    def event(payload: dict[str, Any]) -> str:
        return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

    yield event({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})
    if content:
        yield event({**base, "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]})
    if tool_calls:
        deltas = [dict(call, index=index) for index, call in enumerate(tool_calls)]
        yield event({**base, "choices": [{"index": 0, "delta": {"tool_calls": deltas}, "finish_reason": None}]})
    yield event({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]})
    if include_usage:
        yield event({**base, "choices": [], "usage": usage})
    yield "data: [DONE]\n\n"


def create_app(args: argparse.Namespace) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = Runtime(args)
        yield

    app = FastAPI(title="FlashMTP OpenAI-compatible server", lifespan=lifespan)

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        runtime: Runtime = request.app.state.runtime
        return {
            "status": "ok",
            "model": args.served_model_name,
            "decode_mode": runtime.decode_mode,
            "context_limit": args.context_limit,
            "rope_scaling": runtime.target.config.rope_scaling,
            "target_attention_backend": runtime.target_attention_backend,
            "draft_attention_backend": runtime.draft_attention_backend,
            "server_max_output_tokens": args.max_output_tokens,
        }

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": args.served_model_name, "object": "model", "owned_by": "local"}]}

    @app.post("/admin/config")
    async def admin_config(request: Request) -> dict[str, Any]:
        runtime: Runtime = request.app.state.runtime
        body = await request.json()
        if "decode_mode" in body:
            if body["decode_mode"] not in {"flashmtp", "target"}:
                raise HTTPException(status_code=400, detail="decode_mode must be flashmtp or target")
            runtime.decode_mode = body["decode_mode"]
        if "tags" in body:
            if not isinstance(body["tags"], dict):
                raise HTTPException(status_code=400, detail="tags must be an object")
            runtime.tags = {str(k): str(v) for k, v in body["tags"].items()}
            if run_id := runtime.tags.get("run_id"):
                runtime.run_request_counts[run_id] = 0
        return {"decode_mode": runtime.decode_mode, "tags": runtime.tags}

    @app.get("/metrics/requests")
    async def request_metrics(request: Request, limit: int = 100) -> dict[str, Any]:
        runtime: Runtime = request.app.state.runtime
        return {"data": runtime.metrics.latest(min(max(limit, 0), 1000))}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        runtime: Runtime = request.app.state.runtime
        body = await request.json()
        model = str(body.get("model") or args.served_model_name)
        _, input_ids = runtime.render(body)
        input_tokens = int(input_ids.shape[1])
        requested_tokens = int(body.get("max_completion_tokens") or body.get("max_tokens") or args.max_output_tokens)
        if args.ignore_request_output_limit:
            max_new_tokens = min(args.max_output_tokens, args.context_limit - input_tokens)
        else:
            max_new_tokens = min(requested_tokens, args.max_output_tokens, args.context_limit - input_tokens)
        if max_new_tokens <= 0:
            raise HTTPException(
                status_code=400,
                detail=f"context has {input_tokens} tokens; limit is {args.context_limit}",
            )
        temperature = float(body.get("temperature") or 0.0)
        if temperature < 0:
            raise HTTPException(status_code=400, detail="temperature must be non-negative")
        if temperature > 0 and args.stochastic_verification_mode == "match":
            temperature = 0.0

        request_id = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())
        messages = body["messages"]
        turn_index = sum(1 for message in messages if message.get("role") == "user")
        decode_mode = runtime.decode_mode
        tags = dict(runtime.tags)
        wall_start = time.perf_counter()
        run_key = tags.get("run_id", "untagged")
        async with runtime.lock:
            runtime.run_request_counts[run_key] += 1
            request_index_within_run = runtime.run_request_counts[run_key]
            result = await asyncio.to_thread(
                runtime.generate_sync, input_ids, max_new_tokens, temperature, decode_mode
            )
        request_wall_time = time.perf_counter() - wall_start

        generated_ids = result.output_ids[0, input_tokens:]
        output_tokens = int(generated_ids.numel())
        text = runtime.tokenizer.decode(generated_ids, skip_special_tokens=True)
        content, tool_calls = parse_tool_calls(text)
        hit_length = output_tokens >= max_new_tokens and (
            not output_tokens or int(generated_ids[-1]) not in runtime.stop_token_ids
        )
        finish_reason = "tool_calls" if tool_calls else ("length" if hit_length else "stop")
        usage = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        accepts = result.accept_lengths
        draft_matches = [max(0, value - 1) for value in accepts]
        accept_histogram = {str(k): v for k, v in sorted(Counter(accepts).items())}
        decode_time = float(result.decode_wall_time)
        generation_time = float(result.generation_wall_time)
        row: dict[str, Any] = {
            "timestamp": _utc_now(),
            "request_id": request_id,
            "model": model,
            "decode_mode": decode_mode,
            "request_index_within_run": request_index_within_run,
            "context_limit": args.context_limit,
            "rope_scaling": runtime.target.config.rope_scaling,
            "target_attention_backend": runtime.target_attention_backend,
            "draft_attention_backend": runtime.draft_attention_backend,
            "temperature": temperature,
            "turn_index": turn_index,
            "is_multi_turn": turn_index > 1,
            "message_count": len(messages),
            "message_roles": [str(message.get("role", "")) for message in messages],
            "tool_definition_count": len(body.get("tools") or []),
            "tool_call_count": len(tool_calls),
            "context_tokens_at_turn_start": input_tokens,
            "requested_max_output_tokens": requested_tokens,
            "effective_max_output_tokens": max_new_tokens,
            "request_output_limit_ignored": args.ignore_request_output_limit,
            "output_tokens": output_tokens,
            "response_text": text,
            "total_tokens": input_tokens + output_tokens,
            "finish_reason": finish_reason,
            "request_wall_time_s": request_wall_time,
            "generation_wall_time_s": generation_time,
            "decode_wall_time_s": decode_time,
            "prefill_wall_time_s": result.prefill_wall_time,
            "prefill_and_overhead_wall_time_s": max(0.0, generation_time - decode_time),
            "generation_tokens_per_s": output_tokens / max(generation_time, 1e-9),
            "decode_tokens_per_s": output_tokens / max(decode_time, 1e-9),
            "accept_lengths": accepts,
            "accept_length_semantics": (
                "emitted tokens per speculative step, including one target anchor/correction token"
                if accepts
                else None
            ),
            "draft_match_lengths": draft_matches,
            "average_accept_length": sum(accepts) / len(accepts) if accepts else None,
            "average_draft_matches": sum(draft_matches) / len(draft_matches) if draft_matches else None,
            "accept_length_histogram": accept_histogram,
            "speculative_steps": int(result.steps),
            **tags,
        }
        runtime.metrics.append(row)

        message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        response = {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "message": message, "logprobs": None, "finish_reason": finish_reason}],
            "usage": usage,
        }
        if body.get("stream"):
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
            return StreamingResponse(
                _completion_chunks(
                    request_id, model, created, content, tool_calls, finish_reason, usage, include_usage
                ),
                media_type="text/event-stream",
            )
        return JSONResponse(response)

    @app.post("/v1/completions")
    async def completions(request: Request):
        """Serve raw prompts used by BFCL's local OSS model handlers."""
        runtime: Runtime = request.app.state.runtime
        body = await request.json()
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise HTTPException(
                status_code=400,
                detail="prompt must be a non-empty string; batched prompts are unsupported",
            )

        model = str(body.get("model") or args.served_model_name)
        input_ids = runtime.tokenize_prompt(prompt)
        input_tokens = int(input_ids.shape[1])
        requested_tokens = int(body.get("max_tokens") or args.max_output_tokens)
        max_new_tokens = min(
            requested_tokens,
            args.max_output_tokens,
            args.context_limit - input_tokens,
        )
        if max_new_tokens <= 0:
            raise HTTPException(
                status_code=400,
                detail=f"context has {input_tokens} tokens; limit is {args.context_limit}",
            )
        temperature = float(body.get("temperature") or 0.0)
        if temperature < 0:
            raise HTTPException(status_code=400, detail="temperature must be non-negative")
        if temperature > 0 and args.stochastic_verification_mode == "match":
            temperature = 0.0

        request_id = "cmpl-" + uuid.uuid4().hex
        created = int(time.time())
        decode_mode = runtime.decode_mode
        tags = dict(runtime.tags)
        wall_start = time.perf_counter()
        run_key = tags.get("run_id", "untagged")
        async with runtime.lock:
            runtime.run_request_counts[run_key] += 1
            request_index_within_run = runtime.run_request_counts[run_key]
            result = await asyncio.to_thread(
                runtime.generate_sync,
                input_ids,
                max_new_tokens,
                temperature,
                decode_mode,
            )
        request_wall_time = time.perf_counter() - wall_start

        generated_ids = result.output_ids[0, input_tokens:]
        output_tokens = int(generated_ids.numel())
        text = runtime.tokenizer.decode(generated_ids, skip_special_tokens=True)
        hit_length = output_tokens >= max_new_tokens and (
            not output_tokens or int(generated_ids[-1]) not in runtime.stop_token_ids
        )
        finish_reason = "length" if hit_length else "stop"
        usage = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        accepts = result.accept_lengths
        draft_matches = [max(0, value - 1) for value in accepts]
        parsed_tool_calls = parse_tool_calls(text)[1]
        decode_time = float(result.decode_wall_time)
        generation_time = float(result.generation_wall_time)
        row: dict[str, Any] = {
            "timestamp": _utc_now(),
            "request_id": request_id,
            "api": "completions",
            "model": model,
            "decode_mode": decode_mode,
            "request_index_within_run": request_index_within_run,
            "context_limit": args.context_limit,
            "rope_scaling": runtime.target.config.rope_scaling,
            "target_attention_backend": runtime.target_attention_backend,
            "draft_attention_backend": runtime.draft_attention_backend,
            "temperature": temperature,
            "turn_index": len(re.findall(r"<\\|im_start\\|>user\\n", prompt)),
            "is_multi_turn": prompt.count("<|im_start|>user\n") > 1,
            "message_count": None,
            "message_roles": [],
            "tool_definition_count": prompt.count('"name":'),
            "tool_call_count": len(parsed_tool_calls),
            "context_tokens_at_turn_start": input_tokens,
            "requested_max_output_tokens": requested_tokens,
            "effective_max_output_tokens": max_new_tokens,
            "output_tokens": output_tokens,
            "response_text": text,
            "total_tokens": input_tokens + output_tokens,
            "finish_reason": finish_reason,
            "request_wall_time_s": request_wall_time,
            "generation_wall_time_s": generation_time,
            "decode_wall_time_s": decode_time,
            "prefill_wall_time_s": result.prefill_wall_time,
            "prefill_and_overhead_wall_time_s": max(0.0, generation_time - decode_time),
            "generation_tokens_per_s": output_tokens / max(generation_time, 1e-9),
            "decode_tokens_per_s": output_tokens / max(decode_time, 1e-9),
            "accept_lengths": accepts,
            "accept_length_semantics": (
                "emitted tokens per speculative step, including one target anchor/correction token"
                if accepts
                else None
            ),
            "draft_match_lengths": draft_matches,
            "average_accept_length": sum(accepts) / len(accepts) if accepts else None,
            "average_draft_matches": (
                sum(draft_matches) / len(draft_matches) if draft_matches else None
            ),
            "accept_length_histogram": {
                str(k): v for k, v in sorted(Counter(accepts).items())
            },
            "speculative_steps": int(result.steps),
            **tags,
        }
        runtime.metrics.append(row)

        response = {
            "id": request_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "text": text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
        }
        return JSONResponse(response)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", default=DEFAULT_TARGET)
    parser.add_argument("--draft-name-or-path", default=DEFAULT_DRAFT)
    parser.add_argument("--served-model-name", default="Qwen3-4B-FlashMTP-v2swa")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--metrics-jsonl", default="benchmark_results/skillsbench_request_metrics.jsonl")
    parser.add_argument("--decode-mode", choices=("flashmtp", "target"), default="flashmtp")
    parser.add_argument("--context-limit", type=int, default=163840)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument(
        "--ignore-request-output-limit",
        action="store_true",
        help="Generate up to the server/context limit even when the client sends max_tokens.",
    )
    parser.add_argument("--rope-scaling", choices=("none", "yarn"), default="yarn")
    parser.add_argument("--rope-factor", type=float, default=4.0)
    parser.add_argument("--original-max-position-embeddings", type=int, default=40960)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--verify-block", type=int, default=None)
    parser.add_argument("--mask-token-id", type=int, default=None)
    parser.add_argument("--local-position", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--compile-serial-head", action="store_true")
    parser.add_argument(
        "--stochastic-verification-mode", choices=("match", "rejection"), default="match"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uvicorn.run(create_app(args), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
