#!/usr/bin/env python3
"""Generate line charts from FlashMTP profile sweep summary.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_NOTO_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

BATCH_SIZES = [1, 8, 32, 64]
MODELS = {
    "additive_r256": {"label": "Additive r256", "color": "#2563eb", "marker": "o"},
    "direct_r512": {"label": "Direct r512", "color": "#dc2626", "marker": "s"},
}

COMPONENTS = {
    "spec_draft_backbone_ms": {"label": "Backbone", "color": "#3b82f6"},
    "spec_target_lm_head_ms": {"label": "Target LM Head", "color": "#f59e0b"},
    "spec_serial_head_ms": {"label": "Serial Head", "color": "#10b981"},
}


def _load_summary(path: Path) -> dict[str, dict[int, dict]]:
    """Return {model_label: {batch_size: record}}."""
    records = json.loads(path.read_text())
    grouped: dict[str, dict[int, dict]] = {}
    for rec in records:
        label = rec["label"]
        grouped.setdefault(label, {})[rec["batch_size"]] = rec
    return grouped


def _series(grouped: dict, model: str, key: str) -> list[float]:
    return [grouped[model][bs][key] for bs in BATCH_SIZES]


def _draft_fraction(rec: dict) -> float:
    return rec["spec_draft_total_ms"] / rec["step_total_ms"] * 100.0


def _setup_style() -> None:
    if Path(_NOTO_CJK).exists():
        fm.fontManager.addfont(_NOTO_CJK)
        cjk_family = fm.FontProperties(fname=_NOTO_CJK).get_name()
    else:
        cjk_family = "DejaVu Sans"

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "font.family": cjk_family,
            "axes.unicode_minus": False,
        }
    )


def plot_draft_component_breakdown(grouped: dict, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    fig.suptitle(
        "Draft 组件分解 / Draft Component Breakdown (spec-step)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    for ax, (model_key, style) in zip(axes, MODELS.items()):
        for comp_key, comp_style in COMPONENTS.items():
            y = _series(grouped, model_key, comp_key)
            ax.plot(
                BATCH_SIZES,
                y,
                marker=style["marker"],
                color=comp_style["color"],
                linewidth=2,
                markersize=7,
                label=comp_style["label"],
            )
        ax.set_title(style["label"])
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Time (ms)")
        ax.set_xticks(BATCH_SIZES)
        ax.legend(loc="upper left")

    fig.tight_layout()
    out = out_dir / "01_draft_component_breakdown.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_draft_total(grouped: dict, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_key, style in MODELS.items():
        y = _series(grouped, model_key, "spec_draft_total_ms")
        ax.plot(
            BATCH_SIZES,
            y,
            marker=style["marker"],
            color=style["color"],
            linewidth=2,
            markersize=8,
            label=style["label"],
        )
    ax.set_title("Draft 总耗时 vs Batch / Draft Total vs Batch (spec-step)")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("spec_draft_total_ms (ms)")
    ax.set_xticks(BATCH_SIZES)
    ax.legend()
    fig.tight_layout()
    out = out_dir / "02_draft_total_vs_batch.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_step_total(grouped: dict, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_key, style in MODELS.items():
        y = _series(grouped, model_key, "step_total_ms")
        ax.plot(
            BATCH_SIZES,
            y,
            marker=style["marker"],
            color=style["color"],
            linewidth=2,
            markersize=8,
            label=style["label"],
        )
    ax.set_title("Spec Step 总耗时 vs Batch / Step Total vs Batch")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("step_total_ms (ms)")
    ax.set_xticks(BATCH_SIZES)
    ax.legend()
    fig.tight_layout()
    out = out_dir / "03_step_total_vs_batch.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_target_verify(grouped: dict, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_key, style in MODELS.items():
        y = _series(grouped, model_key, "target_verify_ms")
        ax.plot(
            BATCH_SIZES,
            y,
            marker=style["marker"],
            color=style["color"],
            linewidth=2,
            markersize=8,
            label=style["label"],
        )
    ax.set_title("Target Verify 耗时 vs Batch / Target Verify vs Batch")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("target_verify_ms (ms)")
    ax.set_xticks(BATCH_SIZES)
    ax.legend()
    fig.tight_layout()
    out = out_dir / "04_target_verify_vs_batch.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_fractions(grouped: dict, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_key, style in MODELS.items():
        verify_pct = [grouped[model_key][bs]["verify_fraction"] * 100 for bs in BATCH_SIZES]
        draft_pct = [_draft_fraction(grouped[model_key][bs]) for bs in BATCH_SIZES]
        ax.plot(
            BATCH_SIZES,
            verify_pct,
            marker=style["marker"],
            color=style["color"],
            linewidth=2,
            markersize=8,
            linestyle="-",
            label=f"{style['label']} — Verify %",
        )
        ax.plot(
            BATCH_SIZES,
            draft_pct,
            marker=style["marker"],
            color=style["color"],
            linewidth=2,
            markersize=8,
            linestyle="--",
            alpha=0.75,
            label=f"{style['label']} — Draft %",
        )
    ax.set_title("Draft / Verify 占比 vs Batch / Fraction vs Batch (%)")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(BATCH_SIZES)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    out = out_dir / "05_draft_verify_fraction_vs_batch.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_additive_lm_head(grouped: dict, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    y = _series(grouped, "additive_r256", "spec_target_lm_head_ms")
    ax.plot(
        BATCH_SIZES,
        y,
        marker="o",
        color="#f59e0b",
        linewidth=2,
        markersize=8,
        label="Additive r256 — spec_target_lm_head_ms",
    )
    ax.set_title("Additive LM Head 开销 vs Batch / LM Head Overhead (additive only)")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("spec_target_lm_head_ms (ms)")
    ax.set_xticks(BATCH_SIZES)
    ax.legend()
    fig.tight_layout()
    out = out_dir / "06_additive_lm_head_overhead.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot profile sweep summary charts.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "log/local_profile_sweep_20260729_105058/summary.json",
        help="Path to summary.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for PNGs (default: <summary_dir>/plots)",
    )
    args = parser.parse_args()

    summary_path = args.summary.resolve()
    out_dir = (args.out_dir or summary_path.parent / "plots").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped = _load_summary(summary_path)
    _setup_style()

    plots = [
        plot_draft_component_breakdown(grouped, out_dir),
        plot_draft_total(grouped, out_dir),
        plot_step_total(grouped, out_dir),
        plot_target_verify(grouped, out_dir),
        plot_fractions(grouped, out_dir),
        plot_additive_lm_head(grouped, out_dir),
    ]

    print(f"Saved {len(plots)} plots to {out_dir}:")
    for p in plots:
        print(f"  {p}")


if __name__ == "__main__":
    main()
