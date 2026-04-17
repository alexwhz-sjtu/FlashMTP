#!/usr/bin/env python3
# no use in training, don't need to run this script

"""演示 FlashMTP 训练中「干净前缀长度」的采样分布（与训练代码一致）。

与 ``specforge.core.flashmtp.OnlineFlashMTPModel.sample_clean_prefix_lengths`` 的
构造逻辑对齐：对每个有效 block 独立采样整数 p ∈ {{0, …, block_size-1}}。

- ``prefix_len_sample_bias`` ∈ (0, 1]：P(p=k) ∝ r^k，r<1 时更偏向小 p；
- ``prefix_len_sample_bias == 1.0``：在 {{0,…,B-1}} 上均匀分布。

默认与当前训练入口一致：``block_size=16``（``train_flashmtp.py --block-size``），
``prefix_len_sample_bias=0.6``（``OnlineFlashMTPModel`` 构造时未传入则使用该默认值）。

用法示例::

    python scripts/demo_clean_prefix_sampling.py --num-samples 10000 --seed 42
"""

from __future__ import annotations

import argparse
from collections import Counter

import torch


def sample_clean_prefix_lengths(
    num_samples: int,
    block_size: int,
    prefix_len_sample_bias: float,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """与训练时相同的单变量采样（等价于 block_keep_mask 全为 True 时的一行）。

    Args:
        num_samples: 采样次数（每个样本对应一个 block 的一次前缀长度）。
        block_size: 块大小 B；p 的取值范围为 [0, B-1]。
        prefix_len_sample_bias: 记为 r；r>=1 时为均匀；否则为截断几何质量。

    Returns:
        shape ``(num_samples,)`` 的 int64 张量。
    """
    if not (0.0 < prefix_len_sample_bias <= 1.0):
        raise ValueError(
            "prefix_len_sample_bias must be in (0, 1]; use 1.0 for uniform sampling"
        )
    dev = device or torch.device("cpu")
    bs = block_size
    r = prefix_len_sample_bias
    if r >= 1.0:
        p = torch.randint(
            0,
            bs,
            (num_samples,),
            device=dev,
            dtype=torch.long,
            generator=generator,
        )
    else:
        idx = torch.arange(bs, device=dev, dtype=torch.float32)
        probs = torch.pow(torch.tensor(r, device=dev, dtype=torch.float32), idx)
        probs = probs / probs.sum()
        p = torch.multinomial(
            probs,
            num_samples=num_samples,
            replacement=True,
            generator=generator,
        )
    return p


def theoretical_probs(block_size: int, prefix_len_sample_bias: float) -> list[float]:
    """返回 k=0..B-1 的理论概率（与 multinomial 使用的 probs 一致）。"""
    bs = block_size
    r = prefix_len_sample_bias
    if r >= 1.0:
        return [1.0 / bs] * bs
    idx = list(range(bs))
    raw = [r**k for k in idx]
    s = sum(raw)
    return [x / s for x in raw]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, required=True, help="采样次数")
    parser.add_argument("--block-size", type=int, default=16, help="块大小 B（默认与训练一致）")
    parser.add_argument(
        "--prefix-len-sample-bias",
        type=float,
        default=0.6,
        help="r，训练时 OnlineFlashMTPModel 默认 0.6；1.0 为均匀",
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子（可复现）")
    parser.add_argument(
        "--list",
        action="store_true",
        help="打印每一次采样的具体数值（样本多时不建议）",
    )
    args = parser.parse_args()

    gen = None
    if args.seed is not None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(args.seed)

    p = sample_clean_prefix_lengths(
        args.num_samples,
        args.block_size,
        args.prefix_len_sample_bias,
        generator=gen,
    )
    values = p.tolist()

    print(f"block_size={args.block_size}, prefix_len_sample_bias={args.prefix_len_sample_bias}")
    print(f"num_samples={args.num_samples}, seed={args.seed}")
    print(f"取值范围: 整数 p ∈ [0, {args.block_size - 1}]（每个有效 block 独立采样）")
    print()

    probs = theoretical_probs(args.block_size, args.prefix_len_sample_bias)
    print("理论概率 P(p=k):")
    for k, pr in enumerate(probs):
        print(f"  p={k:2d}: {pr:.6f}")
    print()

    ctr = Counter(values)
    print("经验频数 / 比例:")
    for k in range(args.block_size):
        c = ctr.get(k, 0)
        ratio = c / args.num_samples
        print(f"  p={k:2d}: {c:6d}  ({ratio:.4f})")

    mean = sum(values) / len(values)
    exp_mean = sum(k * probs[k] for k in range(args.block_size))
    print()
    print(f"样本均值: {mean:.4f}  理论期望: {exp_mean:.4f}")

    if args.list:
        print()
        print("各次采样值:", values)


if __name__ == "__main__":
    main()
