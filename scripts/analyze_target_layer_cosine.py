#!/usr/bin/env python3
"""Measure cosine similarity between each target layer's input and output."""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F
from transformers import AutoTokenizer

from datasets import load_dataset
from specforge.data import build_eagle3_dataset
from specforge.distributed import init_distributed
from specforge.modeling.config_utils import get_pretrained_config_dict
from specforge.modeling.target.flashmtp_target_model import get_flashmtp_target_model
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead


@dataclass
class RunningStats:
    total: float = 0.0
    total_sq: float = 0.0
    count: int = 0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().float()
        if values.numel() == 0:
            return
        self.total += values.sum().item()
        self.total_sq += values.square().sum().item()
        self.count += values.numel()
        self.minimum = min(self.minimum, values.min().item())
        self.maximum = max(self.maximum, values.max().item())

    def as_dict(self, prefix: str) -> dict[str, float | int | None]:
        if self.count == 0:
            return {
                f"{prefix}_mean": None,
                f"{prefix}_std": None,
                f"{prefix}_min": None,
                f"{prefix}_max": None,
                f"{prefix}_tokens": 0,
            }
        mean = self.total / self.count
        variance = max(0.0, self.total_sq / self.count - mean * mean)
        return {
            f"{prefix}_mean": mean,
            f"{prefix}_std": math.sqrt(variance),
            f"{prefix}_min": self.minimum,
            f"{prefix}_max": self.maximum,
            f"{prefix}_tokens": self.count,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--chat-template", default="qwen")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--is-preformatted", action="store_true")
    parser.add_argument("--attention-backend", default="fa3")
    parser.add_argument("--mem-fraction-static", type=float, default=0.25)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--dist-timeout", type=int, default=30)
    parser.add_argument(
        "--output-dir",
        default="evaluation_runs/qwen35_layer_input_output_cosine",
    )
    return parser.parse_args()


def get_layer_types(model_path: str) -> list[str]:
    raw_config = get_pretrained_config_dict(model_path)
    text_config = raw_config.get("text_config", raw_config)
    num_layers = int(text_config["num_hidden_layers"])
    layer_types = text_config.get("layer_types")
    if layer_types is None:
        return ["unknown"] * num_layers
    if len(layer_types) != num_layers:
        raise ValueError(
            f"layer_types has {len(layer_types)} entries for {num_layers} layers."
        )
    return [str(layer_type) for layer_type in layer_types]


def normalize_sample_tensor(value: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(value)
    while value.ndim > 1 and value.shape[0] == 1:
        value = value.squeeze(0)
    if value.ndim != 1:
        raise ValueError(
            f"Expected a 1-D token tensor, got shape {tuple(value.shape)}."
        )
    return value.unsqueeze(0).cuda(non_blocking=True)


def hidden_states_by_layer(
    hidden_states: tuple[torch.Tensor, ...] | dict[int, torch.Tensor],
    num_layers: int,
) -> dict[int, torch.Tensor]:
    if isinstance(hidden_states, dict):
        result = hidden_states
    else:
        if len(hidden_states) != num_layers:
            raise ValueError(
                f"Expected {num_layers} captured layers, got {len(hidden_states)}."
            )
        result = dict(enumerate(hidden_states))
    missing = sorted(set(range(num_layers)) - set(result))
    if missing:
        raise ValueError(f"Target backend did not return layers {missing}.")
    return result


def format_optional(value: float | int | None, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def write_results(
    output_dir: Path,
    rows: list[dict[str, object]],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "layer_cosine.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    full_loss_means = [
        float(row["loss_mean"])
        for row in rows
        if row["layer_type"] == "full_attention" and row["loss_mean"] is not None
    ]
    full_loss_means_without_final = [
        float(row["loss_mean"])
        for row in rows[:-1]
        if row["layer_type"] == "full_attention" and row["loss_mean"] is not None
    ]
    linear_loss_means_without_first = [
        float(row["loss_mean"])
        for row in rows[1:]
        if row["layer_type"] == "linear_attention" and row["loss_mean"] is not None
    ]
    markdown = [
        "# Target layer input/output cosine similarity",
        "",
        f"- Model: `{args.target_model_path}`",
        f"- Dataset: `{args.data_path}`",
        f"- Samples: {args.num_samples}",
        f"- Maximum sequence length: {args.max_length}",
        "- Layer IDs are 0-based.",
        "- `loss_mean` covers supervised assistant tokens only.",
        "",
        "## Group summary",
        "",
    ]
    if full_loss_means:
        markdown.append(
            f"- Full-attention layers, loss-token mean: "
            f"{sum(full_loss_means) / len(full_loss_means):.6f}"
        )
    if full_loss_means_without_final:
        markdown.append(
            "- Full-attention layers excluding the final layer, loss-token mean: "
            f"{sum(full_loss_means_without_final) / len(full_loss_means_without_final):.6f}"
        )
    if linear_loss_means_without_first:
        markdown.append(
            "- Linear-attention layers excluding layer 0, loss-token mean: "
            f"{sum(linear_loss_means_without_first) / len(linear_loss_means_without_first):.6f}"
        )
    markdown.extend(
        [
            "",
            "Layer 0 compares token embeddings with the first block output; all later "
            "layers compare the previous block output with the current block output.",
            "",
            "## Per-layer results",
            "",
            "| layer | type | all-token mean | std | loss-token mean | loss tokens |",
            "| ---: | :--- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        markdown.append(
            "| {layer_id} | {layer_type} | {all_mean} | {all_std} | "
            "{loss_mean} | {loss_tokens} |".format(
                layer_id=row["layer_id"],
                layer_type=row["layer_type"],
                all_mean=format_optional(row["all_mean"]),
                all_std=format_optional(row["all_std"]),
                loss_mean=format_optional(row["loss_mean"]),
                loss_tokens=row["loss_tokens"],
            )
        )
    (output_dir / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be at least 1.")
    if args.max_length < 1:
        raise ValueError("--max-length must be at least 1.")

    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    rank = dist.get_rank()
    layer_types = get_layer_types(args.target_model_path)
    num_layers = len(layer_types)

    target = get_flashmtp_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend="sglang",
        torch_dtype=torch.bfloat16,
        attention_backend=args.attention_backend,
        mem_fraction_static=args.mem_fraction_static,
        context_length=args.max_length,
        enable_torch_compile=False,
        max_running_requests=1,
        max_total_tokens=args.max_length,
        ep_size=1,
    )
    target.set_capture_layers(list(range(num_layers)))
    embedding = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model_path,
        device="cuda",
        dtype=torch.bfloat16,
    ).embed_tokens

    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    raw_dataset = load_dataset("json", data_files=args.data_path, split="train")
    sample_count = min(args.num_samples, len(raw_dataset))
    raw_dataset = raw_dataset.shuffle(seed=args.seed).select(range(sample_count))
    dataset = build_eagle3_dataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        shuffle_seed=args.seed,
        num_proc=1,
        is_preformatted=args.is_preformatted,
    )

    all_stats = [RunningStats() for _ in range(num_layers)]
    loss_stats = [RunningStats() for _ in range(num_layers)]
    for sample_index, sample in enumerate(dataset):
        input_ids = normalize_sample_tensor(sample["input_ids"]).long()
        attention_mask = normalize_sample_tensor(sample["attention_mask"]).long()
        loss_mask = normalize_sample_tensor(sample["loss_mask"]).long()
        output = target.generate_flashmtp_data(input_ids, attention_mask, loss_mask)
        captured = hidden_states_by_layer(output.hidden_states, num_layers)

        previous = embedding(input_ids)
        valid = attention_mask.bool()
        supervised = valid & loss_mask.bool()
        for layer_id in range(num_layers):
            current = captured[layer_id]
            cosine = F.cosine_similarity(previous.float(), current.float(), dim=-1)
            all_stats[layer_id].update(cosine[valid])
            loss_stats[layer_id].update(cosine[supervised])
            previous = current
        if rank == 0:
            print(
                f"Processed sample {sample_index + 1}/{len(dataset)} "
                f"({input_ids.shape[1]} tokens)",
                flush=True,
            )

    if rank == 0:
        rows: list[dict[str, object]] = []
        for layer_id, layer_type in enumerate(layer_types):
            rows.append(
                {
                    "layer_id": layer_id,
                    "layer_number": layer_id + 1,
                    "layer_type": layer_type,
                    **all_stats[layer_id].as_dict("all"),
                    **loss_stats[layer_id].as_dict("loss"),
                }
            )
        output_dir = Path(args.output_dir)
        write_results(output_dir, rows, args)
        print(f"Wrote {output_dir / 'summary.md'}")
        print(f"Wrote {output_dir / 'layer_cosine.csv'}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
