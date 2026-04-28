#!/usr/bin/env python3
"""FlashMTP 推理：统计写入 flashmtp_accept_lengths.json，各题回复写入 flashmtp_answers.jsonl。"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_TARGET_MODEL = "/share/public/public_models/Qwen3-8B"
DEFAULT_DRAFT_MODEL = str(
    ROOT
    / "cache/models/FlashMTP_v1.4_sample_400000_think_on_qwen3_8b_maxlen4096_epochs12_nnodes4"
)
DEFAULT_OUTPUT_DIR = str(ROOT / "model_answer")

from specforge.modeling.draft.flashmtp import FlashMTPDraftModel


def load_mtbench101_questions(
    question_file: str, begin: Optional[int], end: Optional[int]
):
    questions = []
    with open(question_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                turns = [hist["user"] for hist in data["history"]]
                questions.append(
                    {
                        "question_id": data["id"],
                        "category": data.get("task", "mtbench101"),
                        "turns": turns,
                    }
                )
    if begin is not None and end is not None:
        questions = questions[begin:end]
    elif begin is not None:
        questions = questions[begin:]
    return questions


def main():
    parser = argparse.ArgumentParser(
        description="FlashMTP 评估：输出每题 accept_lengths 与平均接收长度到 model_answer"
    )
    parser.add_argument(
        "--target-model-path",
        type=str,
        default=DEFAULT_TARGET_MODEL,
    )
    parser.add_argument(
        "--draft-model-path",
        type=str,
        default=DEFAULT_DRAFT_MODEL,
    )
    parser.add_argument(
        "--question-file",
        type=str,
        default="/share/wanghanzhen/SpeculativeDecoding/NIPS26/dataset/mtbench101/question.jsonl",
    )
    parser.add_argument("--begin", type=int, default=2)
    parser.add_argument("--end", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="默认: 仓库下 model_answer/",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="flashmtp_accept_lengths.json",
        help="接收长度统计写入 output-dir 下的文件名",
    )
    parser.add_argument(
        "--answers-file",
        type=str,
        default="flashmtp_answers.jsonl",
        help="各题多轮回复写入 output-dir 下的 JSONL（与统计文件分离）",
    )
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    hf_logging.set_verbosity_error()

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_file
    answers_path = out_dir / args.answers_file

    draft_model = FlashMTPDraftModel.from_pretrained(
        args.draft_model_path,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    ).eval()

    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model_path,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path, trust_remote_code=True
    )
    device = next(target_model.parameters()).device

    questions = load_mtbench101_questions(
        args.question_file, begin=args.begin, end=args.end
    )

    per_question: list[dict] = []
    all_lengths: list[int] = []
    answer_rows: list[dict] = []

    for question in questions:
        qid = question["question_id"]
        accept_lengths: list[int] = []
        conversation_history: list[dict] = []
        turn_responses: list[str] = []

        for user_input in question["turns"]:
            conversation_history.append(
                {
                    "role": "user",
                    "content": "Answer the following question as detailed as possible: "
                    + user_input,
                }
            )
            text = tokenizer.apply_chat_template(
                conversation_history,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=args.thinking,
            )
            input_ids = tokenizer([text], return_tensors="pt").input_ids.to(device)
            turn_lengths: list[int] = []
            output_ids = draft_model.spec_generate(
                target=target_model,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
                accept_lengths_out=turn_lengths,
            )
            accept_lengths.extend(turn_lengths)
            reply = tokenizer.decode(
                output_ids[0][input_ids.shape[1] :], skip_special_tokens=True
            ).strip()
            conversation_history.append({"role": "assistant", "content": reply})
            turn_responses.append(reply)

        all_lengths.extend(accept_lengths)
        mean_q = (
            float(statistics.mean(accept_lengths))
            if accept_lengths
            else 0.0
        )
        per_question.append(
            {
                "question_id": qid,
                "accept_lengths": accept_lengths,
                "mean_accept_length": mean_q,
            }
        )
        answer_rows.append(
            {
                "question_id": qid,
                "category": question["category"],
                "turns": question["turns"],
                "responses": turn_responses,
            }
        )

    overall_mean = (
        float(statistics.mean(all_lengths)) if all_lengths else 0.0
    )
    payload = {
        "per_question": per_question,
        "overall_mean_accept_length": overall_mean,
        "total_spec_decode_steps": len(all_lengths),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    with open(answers_path, "w", encoding="utf-8") as fa:
        for row in answer_rows:
            fa.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


if __name__ == "__main__":
    main()
