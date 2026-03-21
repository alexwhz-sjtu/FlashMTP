"""Convert Nemotron parquet files to jsonl for regenerate_train_data.py.

Output schema:
{
	"id": str,
	"conversations": [
		{"role": str, "content": str}
	]
}

It supports balanced random sampling across task types (column: category).
Output filename is fixed to: Nemotron_{num_samples}.jsonl
"""

import argparse
import glob
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

import pyarrow.parquet as pq
from tqdm import tqdm


VALID_ROLES = {"system", "user", "assistant"}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument("--input-dir", type=str, required=True)
	parser.add_argument("--output-dir", type=str, required=True)
	parser.add_argument("--num-samples", type=int, required=True)
	parser.add_argument("--seed", type=int, default=42)
	return parser.parse_args()


def iter_rows(parquet_path: str) -> Iterable[Dict[str, Any]]:
	pf = pq.ParquetFile(parquet_path)
	for batch in pf.iter_batches():
		for row in batch.to_pylist():
			yield row


def normalize_conversations(messages: Any) -> Optional[List[Dict[str, str]]]:
	if not isinstance(messages, list):
		return None

	convs: List[Dict[str, str]] = []
	for m in messages:
		if not isinstance(m, dict):
			continue
		role = str(m.get("role", "")).strip().lower()
		if role not in VALID_ROLES:
			continue
		content = m.get("content", "")
		if not isinstance(content, str):
			content = json.dumps(content, ensure_ascii=False)
		if role == "system" and content.strip() == "":
			continue
		convs.append({"role": role, "content": content})

	if not convs:
		return None

	# regenerate_train_data expects the first non-system role to be user.
	start = 0
	if convs[0]["role"] == "system":
		start = 1
	if start >= len(convs) or convs[start]["role"] != "user":
		return None

	has_assistant = any(x["role"] == "assistant" for x in convs)
	if not has_assistant:
		return None
	return convs


def main() -> None:
	args = parse_args()
	random.seed(args.seed)

	parquet_files = sorted(glob.glob(os.path.join(args.input_dir, "*.parquet")))
	if not parquet_files:
		raise FileNotFoundError(f"No parquet files found in {args.input_dir}")

	buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
	sample_key = 0

	for parquet_path in tqdm(parquet_files, desc="Reading parquet"):
		for row in iter_rows(parquet_path):
			task_type = row.get("category")
			if not isinstance(task_type, str) or not task_type:
				continue

			convs = normalize_conversations(row.get("messages"))
			if convs is None:
				continue

			buckets[task_type].append({"_key": sample_key, "conversations": convs})
			sample_key += 1

	task_types = sorted([k for k, v in buckets.items() if len(v) > 0])
	if not task_types:
		raise RuntimeError("No valid samples found after filtering")

	# Balanced allocation: average random sampling among task types.
	base = args.num_samples // len(task_types)
	rem = args.num_samples % len(task_types)

	selected: List[Dict[str, Any]] = []
	for i, t in enumerate(task_types):
		need = base + (1 if i < rem else 0)
		pool = buckets[t]
		if len(pool) <= need:
			chosen = pool
		else:
			chosen = random.sample(pool, need)
		selected.extend(chosen)

	# If total is still short (some tasks have too few samples), fill from leftovers.
	if len(selected) < args.num_samples:
		used_keys = {x["_key"] for x in selected}
		leftovers: List[Dict[str, Any]] = []
		for t in task_types:
			for item in buckets[t]:
				if item["_key"] not in used_keys:
					leftovers.append(item)
		random.shuffle(leftovers)
		need_more = args.num_samples - len(selected)
		selected.extend(leftovers[:need_more])

	random.shuffle(selected)
	if len(selected) > args.num_samples:
		selected = selected[: args.num_samples]

	os.makedirs(args.output_dir, exist_ok=True)
	output_path = os.path.join(args.output_dir, f"Nemotron_{args.num_samples}.jsonl")
	with open(output_path, "w", encoding="utf-8") as f:
		for idx, row in enumerate(selected):
			out_row = {
				"id": idx,
				"conversations": row["conversations"],
			}
			f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

	print(f"task_types: {task_types}")
	print(f"written: {len(selected)}")
	print(f"output: {output_path}")


if __name__ == "__main__":
	main()
