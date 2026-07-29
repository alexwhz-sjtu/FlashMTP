#!/usr/bin/env python3
"""Single (model, batch) profile worker; prints JSON after ===JSON_RESULT===."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.profile_markov_head_timing import profile_one_mode
from scripts.profile_spec_step_breakdown import profile_steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-ckpt", required=True)
    parser.add_argument("--target-model-path", default="/data/wanghanzhen/models/Qwen3-8B")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--prompt", default="Solve step by step: What is 17 * 23?")
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--timed-steps", type=int, default=100)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path, trust_remote_code=True)
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to("cuda")

    head = profile_one_mode(
        draft_ckpt=args.draft_ckpt,
        target_path=args.target_model_path,
        input_ids=input_ids,
        batch_size=args.batch_size,
        num_warmup_steps=args.warmup_steps,
        num_timed_steps=args.timed_steps,
    )
    head_dict = asdict(head)
    del head
    gc.collect()
    torch.cuda.empty_cache()

    step = profile_steps(
        draft_ckpt=args.draft_ckpt,
        target_path=args.target_model_path,
        input_ids=input_ids,
        batch_size=args.batch_size,
        num_warmup=args.warmup_steps,
        num_timed=args.timed_steps,
    )
    step_dict = asdict(step)

    row = {
        "batch_size": args.batch_size,
        "draft_ckpt": args.draft_ckpt,
        "prompt_len": int(input_ids.shape[1]),
        # draft micro-profile (markov script)
        "draft_backbone_ms": head_dict["draft_backbone_avg_ms"],
        "target_lm_head_ms": head_dict["target_lm_head_avg_ms"],
        "serial_head_ms": head_dict["markov_serial_head_avg_ms"],
        "head_path_ms": head_dict["total_head_path_avg_ms"],
        "draft_total_ms": head_dict["draft_backbone_avg_ms"] + head_dict["total_head_path_avg_ms"],
        # full spec step (includes verify; may differ slightly on draft parts)
        "spec_draft_backbone_ms": step_dict["draft_backbone_ms"],
        "spec_target_lm_head_ms": step_dict["target_lm_head_ms"],
        "spec_serial_head_ms": step_dict["serial_head_ms"],
        "spec_draft_total_ms": step_dict["draft_total_ms"],
        "target_verify_ms": step_dict["target_verify_ms"],
        "step_total_ms": step_dict["step_total_ms"],
        "verify_fraction": step_dict["verify_fraction"],
        "mode": step_dict["mode"],
        "markov_rank": step_dict["markov_rank"],
        "num_warmup": args.warmup_steps,
        "num_timed": args.timed_steps,
    }
    print("===JSON_RESULT===")
    print(json.dumps(row))


if __name__ == "__main__":
    main()
