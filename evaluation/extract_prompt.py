#!/usr/bin/env python3
"""Extract the input prompt for a specific sample from a benchmark dataset."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import benchmark as bench


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name or path")
    parser.add_argument("--max-samples", type=int, default=50,
                        help="Number of samples selected by benchmark (seed=0).")
    parser.add_argument("--sample-idx", type=int, default=14,
                        help="Zero-based sample index to extract.")
    args = parser.parse_args()

    dataset = bench.load_benchmark_dataset(args.dataset)
    dataset = bench.select_max_samples(dataset, args.max_samples)

    if args.sample_idx < 0 or args.sample_idx >= len(dataset):
        print(f"Error: sample index {args.sample_idx} out of range [0, {len(dataset)})", file=sys.stderr)
        sys.exit(1)

    instance = dataset[args.sample_idx]
    print(f"Dataset: {args.dataset}")
    print(f"Max samples: {args.max_samples}")
    print(f"Sample index: {args.sample_idx}")
    print(f"Number of turns: {len(instance['turns'])}")
    print("=" * 80)
    for turn_idx, turn in enumerate(instance["turns"]):
        print(f"\n--- Turn {turn_idx} ---\n")
        print(turn)
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
