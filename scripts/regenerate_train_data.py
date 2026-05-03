"""
This script will re-generate the dataset from target model,
which better aligns the draft model with the target model’s output distribution.
It accepts preformatted conversation JSONL, CodeAlpaca JSONL, and Orca Math parquet.

Output files are organized by (dataset_name, enable_thinking, samples, model):
  cache/data/regen_data/{dataset}_think_{on|off}_samples_{count}_{model}_regen.jsonl

max_tokens is set large by default (32768) to collect complete data.
Truncation is handled later during training via --max-length.

Usage:
1. Set up one or more SGLang servers for the target model.

python3 -m sglang.launch_server \
	--model meta-llama/Llama-3.1-8B-Instruct \
	--mem-fraction-static 0.75 \
	--cuda-graph-max-bs 128 \
	--tp 1 \
	--trust-remote-code \
	--host 0.0.0.0 \
	--port 30000 \
	--dtype bfloat16


2. Regenerate the dataset using the `regenerate_train_data.py` script.
python scripts/regenerate_train_data.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --concurrency 64 \
    --num-samples 50000 \
    --enable-thinking \
    --server-address localhost:30000 \
    --temperature 0.8 \
    --input-file-path ./cache/dataset/sharegpt_train.jsonl

Output will be saved to:
  ./cache/data/regen_data/sharegpt_think_on_samples_50000_meta-llama_Llama-3.1-8B-Instruct_regen.jsonl

You can also explicitly specify the output path:
    --output-file-path ./cache/dataset/custom_output.jsonl
"""

import argparse
import importlib
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterator, List, Optional

from openai import OpenAI
from tqdm import tqdm


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Re-generate training data using sglang model server"
    )

    # model related arguments
    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model", type=str, required=True)
    model_group.add_argument(
        "--is-reasoning-model",
        action="store_true",
        help="Whether the model is a reasoning model",
    )
    model_group.add_argument(
        "--is-gpt-oss",
        action="store_true",
        help="Whether the model is a GPT-OSS model",
    )
    model_group.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking mode for the model (affects enable_thinking in chat_template_kwargs)",
    )

    # sampling params
    sampling_params_group = parser.add_argument_group("sampling parameters")
    sampling_params_group.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sglang model server",
    )
    sampling_params_group.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling top_p",
    )
    sampling_params_group.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling value sent via extra_body",
    )
    sampling_params_group.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Mapped to presence_penalty in the OpenAI API",
    )
    sampling_params_group.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Maximum number of tokens per generation (default: 32768, set large to collect complete data)",
    )

    # optimization
    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="The number of requests to send to a single server concurrently, the total number of concurrent requests is concurrency * number of server addresses",
    )

    # data related arguments
    data_group = parser.add_argument_group("data")
    data_group.add_argument(
        "--input-file-path",
        type=str,
        required=True,
        help="Path to the input file (conversation JSONL, CodeAlpaca JSONL, or Orca Math parquet)",
    )
    data_group.add_argument(
        "--output-file-path", type=str, default=None,
        help="Path to the output file. If not provided, auto-generated as "
             "./cache/data/regen_data/{dataset}_think_{on|off}_samples_{count}_{model}_regen.jsonl"
    )
    data_group.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="The number of samples to regenerate, if not provided, all samples will be regenerated",
    )
    data_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file, skip already processed samples",
    )

    # sglang server
    server_group = parser.add_argument_group("sglang server")
    server_group.add_argument(
        "--server-address",
        type=str,
        nargs="+",
        help="Server address and port for sglang model server",
    )
    return parser.parse_args()


def get_random_reasoning_effort() -> str:
    """Get a random reasoning effort level for the model with weighted probabilities."""
    # usage example: https://huggingface.co/openai/gpt-oss-20b/discussions/28
    # Reasoning effort levels with weights: LOW(4), MEDIUM(4), HIGH(2)
    reasoning_efforts = [
        "low",
        "medium",
        "high",
    ]
    weights = [4, 4, 2]
    return random.choices(reasoning_efforts, weights=weights, k=1)[0]


def extract_samples_from_filename(input_file_path: str) -> str:
    """Extract sample count from input filename.

    Examples:
        nemotron_4000.jsonl -> 4000
        train_2000.jsonl -> 2000
    """
    import re
    basename = os.path.basename(input_file_path)
    if basename.endswith(".parquet"):
        return "all"
    match = re.search(r'(\d+)', basename)
    return match.group(1) if match else "all"


def extract_dataset_name(input_file_path: str) -> str:
    """Extract dataset name from input filename.

    Examples:
        nemotron_4000.jsonl -> nemotron
        train_data.jsonl -> train_data
    """
    basename = os.path.basename(input_file_path)
    # Remove extension and any _{number} suffix
    import re
    name = re.sub(r'\.(jsonl?|parquet)$', '', basename)
    if name.startswith("train-") and os.path.basename(os.path.dirname(input_file_path)):
        name = os.path.basename(os.path.dirname(input_file_path))
    name = re.sub(r'_\d+$', '', name)
    return name


def build_output_filename(input_file_path: str, num_samples, enable_thinking: bool, model_name: str) -> str:
    """Build output filename from data collection features.

    Format: {dataset}_think_{on|off}_samples_{count}_{model}_regen.jsonl
    Examples:
        nemotron_think_on_samples_2000_qwen3_8b_regen.jsonl
        sharegpt_think_off_samples_all_llama3_1_regen.jsonl
    """
    dataset_name = extract_dataset_name(input_file_path)
    # Use explicit num_samples if provided, otherwise extract from filename
    if num_samples is not None:
        samples_str = str(num_samples)
    else:
        samples_str = extract_samples_from_filename(input_file_path)
    think_str = "on" if enable_thinking else "off"
    # Clean model name (remove path separators)
    model_name_clean = model_name.replace('/', '_').replace('\\', '_')
    return f"{dataset_name}_think_{think_str}_samples_{samples_str}_{model_name_clean}_regen.jsonl"


def resolve_output_path(args):
    """Resolve the output file path from args.

    If --output-file-path is provided, use it directly.
    Otherwise, auto-generate based on input filename, num_samples, enable_thinking, and model.
    """
    if args.output_file_path is not None:
        return args.output_file_path
    filename = build_output_filename(args.input_file_path, args.num_samples, args.enable_thinking, args.model)
    return os.path.join("./cache/data/regen_data", filename)


def add_number_suffix_if_exists(file_path: str, related_file_path: Optional[str] = None) -> str:
    """Return a non-existing file path by appending _1, _2, ... before the extension."""
    if not os.path.exists(file_path) and (
        related_file_path is None or not os.path.exists(related_file_path)
    ):
        return file_path

    directory = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    stem, extension = os.path.splitext(filename)

    suffix = 1
    while True:
        candidate = os.path.join(directory, f"{stem}_{suffix}{extension}")
        candidate_related = (
            candidate.replace(".jsonl", "_error.jsonl")
            if related_file_path is not None
            else None
        )
        if not os.path.exists(candidate) and (
            candidate_related is None or not os.path.exists(candidate_related)
        ):
            return candidate
        suffix += 1


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def normalize_codealpaca_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert CodeAlpaca instruction/input rows to the conversation schema."""
    instruction = clean_text(record.get("instruction"))
    input_text = clean_text(record.get("input"))
    if not instruction:
        raise ValueError("CodeAlpaca record is missing instruction")

    if input_text:
        user_content = f"{instruction}\n\nInput:\n{input_text}"
    else:
        user_content = instruction
    return {"conversations": [{"role": "user", "content": user_content}]}


def normalize_orca_math_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Orca Math question/answer rows to the conversation schema."""
    question = clean_text(record.get("question"))
    if not question:
        raise ValueError("Orca Math record is missing question")
    return {"conversations": [{"role": "user", "content": question}]}


def normalize_input_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize supported input dataset rows to the training conversation schema."""
    if "conversations" in record:
        return record
    if "instruction" in record:
        return normalize_codealpaca_record(record)
    if "question" in record:
        return normalize_orca_math_record(record)
    raise ValueError(f"Unsupported input record fields: {sorted(record.keys())}")


def iter_jsonl_records(input_file_path: str) -> Iterator[Dict[str, Any]]:
    with open(input_file_path, "r") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            yield normalize_input_record(json.loads(line))


def iter_parquet_records(input_file_path: str) -> Iterator[Dict[str, Any]]:
    try:
        pq = importlib.import_module("pyarrow.parquet")
    except ImportError as exc:
        raise ImportError(
            "Reading parquet input requires pyarrow. Install pyarrow or run in the project venv."
        ) from exc

    parquet_file = pq.ParquetFile(input_file_path)
    for batch in parquet_file.iter_batches(batch_size=1024):
        for row in batch.to_pylist():
            yield normalize_input_record(row)


def iter_input_records(input_file_path: str) -> Iterator[Dict[str, Any]]:
    if input_file_path.endswith(".parquet"):
        return iter_parquet_records(input_file_path)
    return iter_jsonl_records(input_file_path)


def count_input_records(input_file_path: str) -> int:
    if input_file_path.endswith(".parquet"):
        try:
            pq = importlib.import_module("pyarrow.parquet")
        except ImportError as exc:
            raise ImportError(
                "Reading parquet input requires pyarrow. Install pyarrow or run in the project venv."
            ) from exc
        return pq.ParquetFile(input_file_path).metadata.num_rows
    with open(input_file_path, "r") as input_file:
        return sum(1 for line in input_file if line.strip())


def compute_context_length(conversations: List[Dict[str, Any]]) -> int:
    """
    This is a rough estimate of the context length measured in untokenized
    tokens.
    """
    length = 0
    for message in conversations:
        content = message.get("content")
        if isinstance(content, str):
            # {"role": "assistant", "content": "Hi, how can I help?"}
            length += len(content.split())
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        length += len(text.split())
    return length


def build_query_kwargs(args, messages, max_tokens=None):
    effective_max_tokens = max_tokens if max_tokens is not None else args.max_tokens

    query_kwargs = dict(
        model=args.model,
        messages=messages,
        max_tokens=effective_max_tokens,
        temperature=args.temperature,
        stream=False,
    )
    if args.top_p is not None:
        query_kwargs["top_p"] = args.top_p
    if args.repetition_penalty is not None:
        query_kwargs["presence_penalty"] = args.repetition_penalty

    extra_body = {"chat_template_kwargs": {"enable_thinking": args.enable_thinking}}
    
    if args.top_k is not None:
        extra_body["top_k"] = args.top_k
    if extra_body:
        query_kwargs["extra_body"] = extra_body
    if args.is_gpt_oss:
        query_kwargs["reasoning_effort"] = get_random_reasoning_effort()
    return query_kwargs


def call_sglang(
    args,
    server_address: str,
    data: List[Dict[str, Any]],
    max_tokens=None,
) -> str:
    """Send a batch of prompts to sglang /v1/completions."""
    client = OpenAI(base_url=f"http://{server_address}/v1", api_key="None")

    messages = data["conversations"]
    regenerated_messages = []

    # ignore data which starts with an assistant message
    if messages[0]["role"] == "assistant":
        data["status"] = "error"
        data["error"] = "Data starts with an assistant message"
        return data

    for message in messages:
        if message["role"] == "system":
            regenerated_messages.append(message)
        elif message["role"] == "assistant":
            continue
        elif message["role"] == "user":
            regenerated_messages.append(message)

            query_kwargs = build_query_kwargs(args, regenerated_messages, max_tokens)

            try:
                resp = client.chat.completions.create(**query_kwargs)
            except Exception as e:
                data["status"] = "error"
                data["error"] = str(e)
                return data
            response_text = resp.choices[0].message.content
            resp_msg = {
                "role": "assistant",
                "content": response_text,
            }
            if args.is_reasoning_model:
                resp_msg["thinking"] = resp.choices[0].message.reasoning_content
                print(f"Reasoning content: {resp_msg['thinking']}")
            regenerated_messages.append(resp_msg)
        else:
            data["status"] = "error"
            data["error"] = f"Invalid message role: {message['role']}"
            return data
    data["conversations"] = regenerated_messages
    data["status"] = "success"
    return data


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Resolve output file path (auto-generate if not explicitly provided)
    output_file_path = resolve_output_path(args)
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    error_file_path = output_file_path.replace(".jsonl", "_error.jsonl")
    if not args.resume:
        original_output_file_path = output_file_path
        output_file_path = add_number_suffix_if_exists(output_file_path, error_file_path)
        if output_file_path != original_output_file_path:
            print(f"Output file exists, using: {output_file_path}")
    print(output_file_path)

    # Validate parameters
    if not (0.0 <= args.temperature <= 1.0):
        raise ValueError("Temperature must be between 0.0 and 1.0")

    if args.max_tokens <= 0:
        raise ValueError("Max tokens must be greater than 0")

    print(f"Configuration:")
    print(f"  Model path: {args.model}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Temperature: {args.temperature}")
    print(f"  API URL: {args.server_address}")
    print(f"  Input file: {args.input_file_path}")
    print(f"  Output file: {output_file_path}")
    print(f"  Resume mode: {args.resume}")
    print("-" * 50)
    total_records = count_input_records(args.input_file_path)

    skip_lines = 0
    error_file_path = output_file_path.replace(".jsonl", "_error.jsonl")

    if args.resume and os.path.exists(output_file_path):
        existing_success = sum(1 for _ in open(output_file_path))
        existing_error = 0
        if os.path.exists(error_file_path):
            existing_error = sum(1 for _ in open(error_file_path))
        skip_lines = existing_success + existing_error
        print(f"Resume mode enabled:")
        print(f"  Found {existing_success} successful samples in output file")
        print(f"  Found {existing_error} error samples in error file")
        print(f"  Skipping first {skip_lines} input samples")
        print("-" * 50)

        if skip_lines >= total_records:
            print(f"All {total_records} samples already processed. Nothing to do.")
            return

    # test all server addresses
    valid_server_addresses = []
    for server_address in args.server_address:
        dummy_data = dict(
            conversations=[{"role": "user", "content": "Hello, how are you?"}]
        )
        result = call_sglang(
            args,
            server_address,
            dummy_data,
            max_tokens=1,
        )
        if result is not None:
            valid_server_addresses.append(server_address)
        else:
            print(f"Server {server_address} is not available")

    if len(valid_server_addresses) == 0:
        raise ValueError("No server address is available")
    print(
        f"Using {len(valid_server_addresses)} server addresses: {valid_server_addresses}"
    )
    print("-" * 50)

    # Determine file open mode based on resume flag
    file_mode = "a" if (args.resume and skip_lines > 0) else "w"
    print(
        f"Regenerating dataset and saving the output to {output_file_path} and error log to {error_file_path}"
    )
    print(
        f"File open mode: {file_mode} ({'append' if file_mode == 'a' else 'overwrite'})"
    )
    print("-" * 50)
    context_token_sum = 0
    context_token_min = None
    context_token_max = 0
    success_samples = 0
    error_samples = 0

    # Create progress bar
    with (
        open(output_file_path, file_mode) as output_file_handle,
        open(error_file_path, file_mode) as error_file_handle,
    ):
        executor = ThreadPoolExecutor(
            max_workers=args.concurrency * len(valid_server_addresses)
        )
        waiting_queue = {
            server_address: [] for server_address in valid_server_addresses
        }
        input_records = iter_input_records(args.input_file_path)
        pbar = tqdm(total=total_records, desc="Processing", initial=skip_lines)
        start_server_index = 0

        if skip_lines > 0:
            print(f"Skipping {skip_lines} already processed samples...")
            for _ in range(skip_lines):
                next(input_records, None)
            print(f"Resuming from sample {skip_lines + 1}")

        for data in input_records:
            if (
                args.num_samples is not None
                and success_samples + error_samples >= args.num_samples
            ):
                break

            # find server address with the least waiting requests
            server_address = valid_server_addresses[start_server_index]
            start_server_index = (start_server_index + 1) % len(valid_server_addresses)

            # submit prompt to sglang
            while len(waiting_queue[server_address]) >= args.concurrency:
                finished_on_request = False
                # check if any future is done, if so, write the result to the output file
                for req_future in waiting_queue[server_address]:
                    if req_future.done():
                        regen_data = req_future.result()

                        if regen_data["status"] == "error":
                            error_file_handle.write(
                                json.dumps(regen_data, ensure_ascii=False) + "\n"
                            )
                            error_samples += 1
                        else:
                            ctx_len = compute_context_length(
                                regen_data.get("conversations", [])
                            )
                            context_token_sum += ctx_len
                            if context_token_min is None:
                                context_token_min = ctx_len
                            else:
                                context_token_min = min(context_token_min, ctx_len)
                            context_token_max = max(context_token_max, ctx_len)

                            output_file_handle.write(
                                json.dumps(regen_data, ensure_ascii=False) + "\n"
                            )
                            success_samples += 1
                        waiting_queue[server_address].remove(req_future)
                        finished_on_request = True

                if finished_on_request:
                    break

            req_future = executor.submit(
                call_sglang,
                args,
                server_address,
                data,
            )
            waiting_queue[server_address].append(req_future)
            pbar.update(1)

        # deal with all the remaining requests
        for server_address, waiting_queue_items in waiting_queue.items():
            for req_future in waiting_queue_items:
                regen_data = req_future.result()
                if regen_data["status"] == "error":
                    error_file_handle.write(
                        json.dumps(regen_data, ensure_ascii=False) + "\n"
                    )
                    error_samples += 1
                else:
                    ctx_len = compute_context_length(
                        regen_data.get("conversations", [])
                    )
                    context_token_sum += ctx_len
                    if context_token_min is None:
                        context_token_min = ctx_len
                    else:
                        context_token_min = min(context_token_min, ctx_len)
                    context_token_max = max(context_token_max, ctx_len)

                    output_file_handle.write(
                        json.dumps(regen_data, ensure_ascii=False) + "\n"
                    )
                    success_samples += 1

    print(f"\nProcessing completed!")
    if success_samples > 0:
        avg_len = context_token_sum / success_samples
        print("Context length statistics (token count over conversations):")
        print(f"Number of successful examples: {success_samples}")
        print(f"Shortest context length: {context_token_min}")
        print(f"Longest context length: {context_token_max}")
        print(f"Average context length: {avg_len:.2f}")
    else:
        print("No successful examples to compute context length statistics.")

    total_processed = success_samples + error_samples
    if skip_lines > 0:
        print(f"\nResume processing completed!")
        print(f"  Previously processed: {skip_lines}")
        print(
            f"  Newly processed: {total_processed} ({success_samples} success, {error_samples} failed)"
        )
        print(f"  Total: {skip_lines + total_processed}")
    else:
        print(
            f"\nProcessing completed! {success_samples} samples regenerated, {error_samples} samples failed."
        )


if __name__ == "__main__":
    main()
