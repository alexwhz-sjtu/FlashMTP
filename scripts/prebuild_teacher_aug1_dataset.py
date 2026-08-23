#!/usr/bin/env python3
"""Build the node-local processed dataset cache before distributed training."""

import hashlib
import os

from datasets import load_dataset
from transformers import AutoTokenizer

from specforge.data import build_eagle3_dataset


DATA_PATH = "/data/wanghanzhen/training_data/mixed_2360k_qwen3_8b_nm_pb_swe_aug1.jsonl"
MODEL_PATH = "/data/wanghanzhen/models/Qwen3-8B"
CACHE_ROOT = "/data/wanghanzhen/FlashMTP_v2.3/cache/train_aug1_maxlen10240"
MAX_LENGTH = 10240
CHAT_TEMPLATE = "qwen"


def main() -> None:
    cache_key = hashlib.md5(
        f"{DATA_PATH}-{MAX_LENGTH}-{CHAT_TEMPLATE}-{MODEL_PATH}".encode()
    ).hexdigest()
    raw = load_dataset("json", data_files=DATA_PATH)["train"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    dataset = build_eagle3_dataset(
        dataset=raw,
        tokenizer=tokenizer,
        chat_template=CHAT_TEMPLATE,
        max_length=MAX_LENGTH,
        is_preformatted=False,
        cache_dir=os.path.join(CACHE_ROOT, "processed_dataset"),
        cache_key=cache_key,
        num_proc=32,
    )
    dataset = dataset.filter(lambda value: value["loss_mask"].sum() >= 16)
    print(f"PREBUILD_COMPLETE samples={len(dataset)} cache_key={cache_key}")


if __name__ == "__main__":
    main()
