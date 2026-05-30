"""Baseline and FlashMTP speculative generation for benchmarking."""

from __future__ import annotations

import time
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from specforge.modeling.draft.flashmtp import FlashMTPDraftModel, sample


def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def decode_weight(run) -> int:
    n = int(getattr(run, "num_tokens_for_decode_rate", run.num_output_tokens))
    return int(run.batch_size) * max(n, 1)


def decode_wall_seconds(run) -> float:
    return float(run.time_per_output_token) * decode_weight(run)


@torch.inference_mode()
def run_benchmark_warmup(
    target: AutoModelForCausalLM,
    draft_model: FlashMTPDraftModel,
    tokenizer: AutoTokenizer,
    block_size: int,
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    stop_token_ids: list[int],
) -> None:
    warmup_prompt = [{"role": "user", "content": "Warmup."}]
    input_text = tokenizer.apply_chat_template(
        warmup_prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    if batch_size > 1:
        input_ids = input_ids.expand(batch_size, -1).contiguous()
    warmup_new_tokens = min(16, max_new_tokens)

    target_generate(
        target=target,
        input_ids=input_ids,
        max_new_tokens=warmup_new_tokens,
        stop_token_ids=stop_token_ids,
        temperature=temperature,
        decode_timing_after_first_token=False,
    )
    flashmtp_generate(
        model=draft_model,
        target=target,
        input_ids=input_ids,
        max_new_tokens=warmup_new_tokens,
        block_size=block_size,
        stop_token_ids=stop_token_ids,
        temperature=temperature,
        decode_timing_after_first_token=False,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cuda_time()


@torch.inference_mode()
def target_generate(
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    decode_timing_after_first_token: bool = False,
) -> SimpleNamespace:
    bsz = input_ids.shape[0]
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    output_ids = torch.empty((bsz, max_length), dtype=torch.long, device=input_ids.device)
    output_ids[:, :num_input_tokens] = input_ids
    position_ids = torch.arange(max_length, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
    past_key_values = DynamicCache()
    stop_tensor = (
        torch.tensor(stop_token_ids, device=input_ids.device, dtype=torch.long)
        if stop_token_ids
        else None
    )

    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=False,
    )
    next_token = sample(output.logits, temperature)
    time_to_first_token = (cuda_time() - prefill_start) / bsz

    decode_start: float | None = None if decode_timing_after_first_token else cuda_time()
    start = num_input_tokens
    while start < max_length:
        output_ids[:, start : start + 1] = next_token
        start += 1
        if decode_timing_after_first_token and decode_start is None:
            decode_start = cuda_time()
        if stop_tensor is not None and torch.all(torch.isin(next_token.squeeze(-1), stop_tensor)):
            break
        if start >= max_length:
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

    output_ids = output_ids[:, :start]
    num_output_tokens = output_ids.shape[1] - num_input_tokens
    if decode_start is None:
        decode_start = cuda_time()
    total_decode_time = cuda_time() - decode_start
    rate_tokens = max(num_output_tokens - (1 if decode_timing_after_first_token else 0), 1)
    time_per_output_token = total_decode_time / (bsz * rate_tokens)

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        num_tokens_for_decode_rate=rate_tokens,
        batch_size=bsz,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        throughput_tokens_per_sec=(bsz * rate_tokens) / max(total_decode_time, 1e-9),
        decode_wall_time=total_decode_time,
        acceptance_lengths=[1] * num_output_tokens,
    )


@torch.inference_mode()
def flashmtp_generate(
    model: FlashMTPDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    decode_timing_after_first_token: bool = False,
) -> SimpleNamespace:
    original_block_size = model.block_size
    model.block_size = block_size
    try:
        output_ids = model.spec_generate(
            target=target,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
            decode_timing_after_first_token=decode_timing_after_first_token,
        )
    finally:
        model.block_size = original_block_size

    stats = model.get_last_decode_stats()
    bsz = int(input_ids.shape[0])
    num_input_tokens = input_ids.shape[1]
    num_output_tokens = output_ids.shape[1] - num_input_tokens
    decode_wall_time = float(stats.get("decode_wall_time", 0.0))
    rate_tokens = max(num_output_tokens - (1 if decode_timing_after_first_token else 0), 1)
    time_per_output_token = decode_wall_time / (bsz * max(rate_tokens, 1))

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        num_tokens_for_decode_rate=rate_tokens,
        batch_size=bsz,
        time_to_first_token=0.0,
        time_per_output_token=time_per_output_token,
        throughput_tokens_per_sec=(bsz * rate_tokens) / max(decode_wall_time, 1e-9),
        decode_wall_time=decode_wall_time,
        acceptance_lengths=stats.get("accept_lengths", []),
    )
