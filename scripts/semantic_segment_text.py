#!/usr/bin/env python3
"""使用 Qwen3（或同结构）tokenizer + embed_tokens，按相邻 token 余弦不相似度做语义切分示例。"""
import argparse
import json

import torch
from transformers import AutoTokenizer

from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.text_semantic_segment import semantic_segment_text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, default="/share/public/public_models/Qwen3-8B", help="本地或 Hub 上的 Qwen3 权重路径")
    p.add_argument("--text", type=str, default="Time is not merely the mechanical ticking of numbers on a clock; it is a river, silently eroding the banks of memory. We drift within it, sometimes flowing downstream with ease and comfort, other times rowing upstream with difficulty yet determination. Every present moment is the sum of all past choices and the starting point for all future possibilities. Do not be anxious about its passage, for it is precisely these moments that shape the unique you.", help="直接输入一段文字")
    p.add_argument("--text-file", type=str, default="", help="从文件读取 UTF-8 文本（与 --text 二选一）")
    p.add_argument("--percentile", type=float, default=20.0, help="只保留高于该分位的局部峰值断点")
    p.add_argument("--min-segment-tokens", type=int, default=1, help="相邻断点在 token 缝隙下标上最小间隔")
    p.add_argument("--smooth-window", type=int, default=3, help="不相似度一维平滑窗口（奇数，<=1 关闭）")
    p.add_argument("--add-special-tokens", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
    args = p.parse_args()

    if args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = args.text
    if not text.strip():
        raise SystemExit("请提供 --text 或 --text-file")

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )
    target = TargetEmbeddingsAndHead.from_pretrained(
        args.model_path,
        embed_key="model.embed_tokens.weight",
        lm_head_key="lm_head.weight",
        device=args.device,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )

    segments, dissim, gaps = semantic_segment_text(
        text,
        tokenizer,
        target.embed_tokens,
        device=args.device,
        add_special_tokens=args.add_special_tokens,
        percentile=args.percentile,
        min_segment_tokens=args.min_segment_tokens,
        smooth_window=args.smooth_window,
    )

    out = {
        "num_tokens": len(tokenizer.encode(text, add_special_tokens=args.add_special_tokens)),
        "gap_indices": gaps,
        "dissimilarity_preview": dissim[: min(32, dissim.numel())].tolist(),
        "segments": segments,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
