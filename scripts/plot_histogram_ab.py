#!/usr/bin/env python3
"""Plot overlaid histograms for A/B percentage distributions."""

import argparse
from pathlib import Path


HISTOGRAM_A = [
    "12.2%",
    "13.8%",
    "13.7%",
    "10.4%",
    "8.5%",
    "6.8%",
    "6.5%",
    "5.4%",
    "4.3%",
    "3.6%",
    "2.6%",
    "2.4%",
    "2.1%",
    "1.6%",
    "1.3%",
    "4.9%",
]

HISTOGRAM_B = [
    "12.4%",
    "12.6%",
    "12.8%",
    "9.7%",
    "8.3%",
    "6.5%",
    "6.5%",
    "5.1%",
    "4.7%",
    "3.3%",
    "3.6%",
    "2.5%",
    "2.2%",
    "1.7%",
    "1.8%",
    "6.4%",
]


def parse_percentages(values):
    return [float(value.strip().rstrip("%")) for value in values]


def save_plot(output_path: Path, bar_width: float, bar_gap: float):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise SystemExit("matplotlib and numpy are required to draw the figure.") from exc

    hist_a = np.array(parse_percentages(HISTOGRAM_A))
    hist_b = np.array(parse_percentages(HISTOGRAM_B))
    labels = [str(index + 1) for index in range(len(hist_a))]
    bin_step = bar_width + bar_gap
    x = np.arange(len(labels)) * bin_step

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(12, 5.8), dpi=180)

    ax.bar(
        x,
        hist_a,
        width=bar_width,
        label="A",
        color="#4C78A8",
        alpha=1.0,
        edgecolor="white",
        linewidth=0.8,
        zorder=2,
    )
    ax.bar(
        x,
        hist_b,
        width=bar_width,
        label="B",
        color="#F58518",
        alpha=0.5,
        edgecolor="#C75D00",
        linewidth=0.8,
        zorder=3,
    )

    ax.set_title("Histogram A vs B", pad=14, fontsize=15, weight="bold")
    ax.set_xlabel("Bin")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(x[0] - bar_width / 2 - bar_gap, x[-1] + bar_width / 2 + bar_gap)
    ax.set_ylim(0, max(hist_a.max(), hist_b.max()) + 2.0)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=1)
    ax.legend(frameon=False, ncols=2, loc="upper right")

    for xpos, value_a, value_b in zip(x, hist_a, hist_b):
        ax.text(
            xpos,
            max(value_a, value_b) + 0.18,
            f"A {value_a:.1f}%\nB {value_b:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.4,
            linespacing=0.95,
        )

    ax.annotate(
        "B has more mass in the right tail",
        xy=(x[-1], hist_b[-1]),
        xytext=(x[10], 12.7),
        arrowprops=dict(arrowstyle="->", color="#C75D00", lw=1.4),
        color="#9A4A00",
        fontsize=10,
        ha="left",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot overlaid A/B histogram percentage distributions."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("FlashMTP_exp/assets/histogram_A_B.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--bar-width",
        type=float,
        default=0.9,
        help="Width of each bar.",
    )
    parser.add_argument(
        "--bar-gap",
        type=float,
        default=0.1,
        help="Gap between neighboring bar edges.",
    )
    args = parser.parse_args()

    if args.bar_width <= 0:
        raise SystemExit("--bar-width must be positive")
    if args.bar_gap < 0:
        raise SystemExit("--bar-gap must be non-negative")

    save_plot(args.output, args.bar_width, args.bar_gap)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
