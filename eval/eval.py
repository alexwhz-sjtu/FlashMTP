import torch
import json
import argparse
import numpy as np
import os
import sys
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from specforge.modeling.draft.flashmtp import FlashMTPDraftModel

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def load_mtbench101_questions(question_file, begin=None, end=None):
    """Load questions from mtbench101.jsonl format"""
    questions = []
    with open(question_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # Convert mtbench101 format to standard format
                turns = []
                for hist in data['history']:
                    turns.append(hist['user'])
                
                question = {
                    'question_id': data['id'],
                    'category': data.get('task', 'mtbench101'),
                    'turns': turns,
                    'history': data['history']  # Keep original history for reference
                }
                questions.append(question)
    
    if begin is not None and end is not None:
        questions = questions[begin:end]
    
    return questions


def multi_turn_dialogue(
    draft_model,
    target_model,
    tokenizer,
    turns,
    max_new_tokens=4096,
    temperature=0.0,
    log_file=None,
    thinking=False,
    debug_dir=None,
    draft_topk_file=None,
    draft_topk=5,
):
    """
    Execute multi-turn dialogue with the model
    
    Args:
        draft_model: The FlashMTP draft model instance
        target_model: The target model instance
        tokenizer: The tokenizer
        turns: List of user inputs (strings)
        max_new_tokens: Maximum new tokens to generate
        temperature: Temperature for generation
        log_file: File object to write logs to (optional)
    
    Returns:
        List of assistant responses and their statistics
    """
    conversation_history = []
    responses = []
    turn_stats = []
    response_lengths = []
    device = next(draft_model.parameters()).device
    
    for turn_idx, user_input in enumerate(turns):
        if log_file:
            log_file.write(f"Turn {turn_idx + 1}: {user_input[:50]}...\n")
            log_file.flush()
        
        # Add user message to conversation history
        conversation_history.append({"role": "user", "content": "Answer the following question as detailed as possible: " + user_input})
        
        # Apply chat template with full conversation history
        text = tokenizer.apply_chat_template(
            conversation_history,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking
        )
        
        # Tokenize and move to GPU
        input_ids = tokenizer([text], return_tensors="pt").input_ids.to(device)
        
        # Generate response
        output_ids = draft_model.spec_generate(
            target=target_model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=[tokenizer.eos_token_id],
            temperature=temperature,
            debug_dir=debug_dir,
            tokenizer=tokenizer,
            draft_topk_file=draft_topk_file,
            draft_topk=draft_topk,
        )
        
        # Decode output (remove input tokens)
        output = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Add assistant response to conversation history for next turn
        conversation_history.append({"role": "assistant", "content": output})
        responses.append(output)
        
        # Calculate response length in tokens
        response_token_ids = tokenizer([output]).input_ids[0]
        response_length = len(response_token_ids)
        response_lengths.append(response_length)
        get_stats_fn = getattr(draft_model, "get_last_decode_stats", None)
        stats = get_stats_fn() if callable(get_stats_fn) else None
        stats = stats or {
            "accept_lengths": [],
            "target_total_time": 0.0,
            "draft_total_time": 0.0,
            "steps": 0,
        }
        
        # Add computed metrics to stats
        stats["response_length"] = response_length
        accept_lengths = stats.get("accept_lengths", [])
        stats["mean_accept_length"] = float(np.mean(accept_lengths)) if accept_lengths else 0.0
        
        # Calculate throughput (tokens per second)
        total_time = stats.get("target_total_time", 0.0) + stats.get("draft_total_time", 0.0)
        if total_time > 0 and response_length > 0:
            stats["throughput"] = response_length / total_time
        else:
            stats["throughput"] = 0.0
        
        turn_stats.append(stats)

        if log_file:
            log_file.write(f"Assistant: {output[:100]}...\n")
            log_file.write(f"  Response Length: {response_length} tokens\n")
            log_file.write(f"  Accept Lengths: {stats['accept_lengths']}\n")
            log_file.write(f"  Mean Accept Length: {stats['mean_accept_length']:.4f}\n")
            log_file.write(f"  Target Time: {stats['target_total_time']:.4f}s | Draft Time: {stats['draft_total_time']:.4f}s\n")
            log_file.write(f"  Throughput: {stats['throughput']:.2f} tokens/sec\n\n")
            log_file.flush()
    
    return responses, turn_stats


def summarize_question_stats(turn_stats, responses=None):
    all_accept_lengths = []
    response_lengths = []
    mean_accept_lengths = []
    throughputs = []
    target_total = 0.0
    draft_total = 0.0
    total_steps = 0
    total_response_length = 0

    for one_turn in turn_stats:
        all_accept_lengths.extend(one_turn.get("accept_lengths", []))
        response_lengths.append(one_turn.get("response_length", 0))
        mean_accept_lengths.append(one_turn.get("mean_accept_length", 0.0))
        throughputs.append(one_turn.get("throughput", 0.0))
        target_total += one_turn.get("target_total_time", 0.0)
        draft_total += one_turn.get("draft_total_time", 0.0)
        total_steps += one_turn.get("steps", 0)
        total_response_length += one_turn.get("response_length", 0)

    mean_accept = float(np.mean(all_accept_lengths)) if all_accept_lengths else 0.0
    mean_response_length = float(np.mean(response_lengths)) if response_lengths else 0.0

    mean_per_turn_accept = float(np.mean(mean_accept_lengths)) if mean_accept_lengths else 0.0
    mean_throughput = float(np.mean(throughputs)) if throughputs else 0.0
    total_time = target_total + draft_total
    overall_throughput = total_response_length / total_time if total_time > 0 else 0.0
    
    return {
        "all_accept_lengths": all_accept_lengths,
        "mean_accept_length": mean_accept,
        "response_lengths": response_lengths,
        "mean_response_length": mean_response_length,
        "total_response_length": total_response_length,
        "mean_per_turn_accept_length": mean_per_turn_accept,
        "mean_throughput": mean_throughput,
        "overall_throughput": overall_throughput,
        "target_total_time": target_total,
        "draft_total_time": draft_total,
        "total_time": total_time,
        "total_steps": total_steps,
    }


def main():
    parser = argparse.ArgumentParser()
    
    # Predefined FlashMTP model pairs
    qwen3_8b_flashmtp_draft = str(
        ROOT
        / "cache/models/FlashMTP_v1.4_sample_400000_think_on_qwen3_8b_maxlen4096_epochs12_nnodes4"
    )
    MODEL_PAIRS = {
        "qwen3-8b": {
            "target": "/share/public/public_models/Qwen3-8B",
            "draft": qwen3_8b_flashmtp_draft,
        },
    }
    
    parser.add_argument("--model-pair", type=str, default="qwen3-8b", 
                        choices=list(MODEL_PAIRS.keys()),
                        help="Choose a predefined FlashMTP model pair.")
    parser.add_argument("--target-model-path", type=str, default=None,
                        help="Override target model path (if not using predefined pairs)")
    parser.add_argument("--draft-model-path", type=str, default=None,
                        help="Override draft model path (if not using predefined pairs)")
    parser.add_argument("--question-file", type=str, default="/share/wanghanzhen/SpeculativeDecoding/NIPS26/dataset/mtbench101/question.jsonl")
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-dir", type=str, default="./model_answer")
    parser.add_argument("--debug-dir", type=str, default="./debug")
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument(
        "--draft-topk-file",
        type=str,
        default=None,
        help="若指定路径，则每个 decode_step 写一个 JSON 对象：positions 内每位含 top-k、熵，以及草稿该步采样 token 的 selected_prob（softmax 置信度）",
    )
    parser.add_argument(
        "--draft-topk",
        type=int,
        default=5,
        help="与 --draft-topk-file 配合：每个块内位置保留的 top 概率个数，默认 5",
    )

    parser.add_argument("--thinking", type=bool, default=False)
    args = parser.parse_args()
    
    # Use predefined model pair or custom paths
    if args.target_model_path is None:
        args.target_model_path = MODEL_PAIRS[args.model_pair]["target"]
    if args.draft_model_path is None:
        args.draft_model_path = MODEL_PAIRS[args.model_pair]["draft"]

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.debug_dir, exist_ok=True)
    
    # Extract model and dataset names for output filename
    target_model_name = os.path.basename(args.target_model_path)
    draft_model_name = os.path.basename(args.draft_model_path)
    dataset_name = os.path.basename(os.path.dirname(args.question_file))
    
    # Create output filename
    output_file = os.path.join(
        args.output_dir,
        f"{target_model_name}-{'think' if args.thinking else ''}-FlashMTP-{dataset_name}-temperature-{args.temperature}-max_length_{args.max_length}.jsonl"
    )
    log_file_path = os.path.join(
        args.output_dir,
        f"{target_model_name}-{'think' if args.thinking else ''}-FlashMTP-{dataset_name}-temperature-{args.temperature}-max_length_{args.max_length}.log"
    )

    # Load models (force local FlashMTP implementation)
    draft_model = FlashMTPDraftModel.from_pretrained(
        args.draft_model_path,
        dtype="auto",
        device_map="cuda:0",
        trust_remote_code=True,
    ).eval()
    
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model_path,
        dtype="auto",
        device_map="cuda:0",
        trust_remote_code=True,
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path, trust_remote_code=True)

    questions = load_mtbench101_questions(args.question_file, begin=args.begin, end=args.end)
    
    draft_topk_fp = None
    if args.draft_topk_file:
        _dtk_dir = os.path.dirname(os.path.abspath(args.draft_topk_file))
        if _dtk_dir:
            os.makedirs(_dtk_dir, exist_ok=True)
        draft_topk_fp = open(args.draft_topk_file, "w", encoding="utf-8")

    # Open output and log files
    try:
        with open(output_file, 'w', encoding='utf-8') as f_out, \
             open(log_file_path, 'w', encoding='utf-8') as f_log:

            f_log.write(f"Loaded {len(questions)} questions from {args.question_file}\n")
            f_log.write(f"Target Model: {args.target_model_path}\n")
            f_log.write(f"Draft Model: {args.draft_model_path}\n")
            f_log.write(f"Dataset: {dataset_name}\n")
            f_log.write(f"Temperature: {args.temperature}\n")
            f_log.write(f"Max Length: {args.max_length}\n\n")
            f_log.flush()

            for idx, question in enumerate(questions):
                f_log.write("\n" + "=" * 80 + "\n")
                f_log.write(f"Question {idx + 1}/{len(questions)} | ID: {question['question_id']} | Category: {question['category']}\n")
                f_log.write("=" * 80 + "\n")
                f_log.flush()

                responses, turn_stats = multi_turn_dialogue(
                    draft_model,
                    target_model,
                    tokenizer,
                    question['turns'],
                    max_new_tokens=args.max_length,
                    temperature=args.temperature,
                    log_file=f_log,
                    thinking=args.thinking,
                    debug_dir=args.debug_dir if args.debug else None,
                    draft_topk_file=draft_topk_fp,
                    draft_topk=args.draft_topk,
                )

                summary = summarize_question_stats(turn_stats, responses)
                f_log.write("-" * 80 + "\n")
                f_log.write("Question Summary\n")
                f_log.write(f"Response Lengths (per turn): {summary['response_lengths']}\n")
                f_log.write(f"Mean Response Length: {summary['mean_response_length']:.1f} tokens\n")
                f_log.write(f"Total Response Length: {summary['total_response_length']} tokens\n")
                f_log.write(f"\n")
                f_log.write(f"Accept Lengths (all verifications): {summary['all_accept_lengths']}\n")
                f_log.write(f"Overall Mean Accept Length: {summary['mean_accept_length']:.4f}\n")
                f_log.write(f"Mean Per-Turn Accept Length: {summary['mean_per_turn_accept_length']:.4f}\n")
                f_log.write(f"\n")
                f_log.write(f"Target Total Time: {summary['target_total_time']:.4f}s\n")
                f_log.write(f"Draft Total Time: {summary['draft_total_time']:.4f}s\n")
                f_log.write(f"Total Time: {summary['total_time']:.4f}s\n")
                f_log.write(f"\n")
                f_log.write(f"Mean Throughput (per turn): {summary['mean_throughput']:.2f} tokens/sec\n")
                f_log.write(f"Overall Throughput: {summary['overall_throughput']:.2f} tokens/sec\n")
                f_log.write(f"Total Verification Steps: {summary['total_steps']}\n")
                f_log.write("-" * 80 + "\n")
                f_log.flush()

                # Save result to JSONL file
                result = {
                    "question_id": question['question_id'],
                    "category": question['category'],
                    "turns": question['turns'],
                    "responses": responses,
                    "statistics": {
                        "turn_stats": turn_stats,
                        "summary": summary
                    }
                }
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                f_out.flush()

            f_log.write(f"\n\nResults saved to: {output_file}\n")
            f_log.write(f"Logs saved to: {log_file_path}\n")
            if args.draft_topk_file:
                f_log.write(f"Draft top-k JSONL: {args.draft_topk_file}\n")
    finally:
        if draft_topk_fp is not None:
            draft_topk_fp.close()


if __name__ == "__main__":
    main()