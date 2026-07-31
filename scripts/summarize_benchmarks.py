#!/usr/bin/env python3
"""Parse FlashMTP benchmark logs into consolidated CSV/JSON summaries.

Supports run directories produced by:
  - scripts/run_three_model_speedup_benchmarks.sh
  - scripts/run_compile_rejection_benchmarks.sh

Each run root must contain manifest.tsv plus logs/ and status/.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PATTERNS = {
    "turns": re.compile(r"turns:\s*(\d+)"),
    "speedup": re.compile(r"token-weighted speedup:\s*([0-9.]+)x"),
    "throughput_ratio": re.compile(r"throughput ratio:\s*([0-9.]+)x"),
    "unweighted_speedup": re.compile(r"unweighted speedup:\s*([0-9.]+)x"),
    "decode_times": re.compile(
        r"decode s/token baseline=([0-9.eE+-]+)\s+flashmtp=([0-9.eE+-]+)"
    ),
    "acceptance": re.compile(r"average acceptance length:\s*([0-9.]+)"),
    "elapsed": re.compile(r"Total elapsed time:\s*([0-9.]+)s"),
    "verification_mode": re.compile(
        r"verification_mode=(\w+)|verification=(\w+)"
    ),
    "compile_serial_head": re.compile(r"compile_serial_head=(true|false)"),
}


def _last_match(pattern: re.Pattern[str], text: str) -> str | None:
    matches = pattern.findall(text)
    if not matches:
        return None
    value = matches[-1]
    if isinstance(value, tuple):
        return next((part for part in value if part), None)
    return value


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _draft_accept_rate(avg_accept_length: float | None, verify_block: int) -> float | None:
    """Fraction of proposed draft tokens accepted before the bonus token."""
    if avg_accept_length is None or verify_block <= 1:
        return None
    return max(0.0, min(1.0, (avg_accept_length - 1.0) / (verify_block - 1)))


@dataclass
class BenchmarkRow:
    run_id: str
    model: str
    temperature: str
    verification_mode: str
    compile_serial_head: str
    dataset: str
    requested_samples: str
    gpu: str
    status: str
    turns: str
    token_weighted_speedup: str
    throughput_ratio: str
    unweighted_speedup: str
    baseline_s_per_token: str
    flashmtp_s_per_token: str
    average_acceptance_length: str
    draft_accept_rate: str
    elapsed_seconds: str
    draft_path: str
    log_path: str
    status_path: str


def _infer_run_metadata(run_root: Path) -> dict[str, str]:
    run_id = run_root.name
    verification_mode = "match"
    compile_serial_head = "false"
    if "rejection" in run_id:
        verification_mode = "rejection"
    if "compile" in run_id:
        compile_serial_head = "true"

    config_path = run_root / "run_config.txt"
    if config_path.exists():
        for line in config_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("compile_serial_head="):
                compile_serial_head = line.split("=", 1)[1].strip()

    return {
        "run_id": run_id,
        "verification_mode_default": verification_mode,
        "compile_serial_head_default": compile_serial_head,
    }


def parse_log(
    log_path: Path,
    *,
    verify_block: int,
    verification_mode_default: str,
    compile_default: str,
) -> dict[str, str]:
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    decode_times = _last_match(PATTERNS["decode_times"], text)
    avg_accept = _last_match(PATTERNS["acceptance"], text)
    accept_rate = _draft_accept_rate(_to_float(avg_accept), verify_block)
    verification_mode = (
        _last_match(PATTERNS["verification_mode"], text)
        or verification_mode_default
    )
    compile_flag = _last_match(PATTERNS["compile_serial_head"], text) or compile_default
    return {
        "turns": _last_match(PATTERNS["turns"], text) or "",
        "token_weighted_speedup": _last_match(PATTERNS["speedup"], text) or "",
        "throughput_ratio": _last_match(PATTERNS["throughput_ratio"], text) or "",
        "unweighted_speedup": _last_match(PATTERNS["unweighted_speedup"], text)
        or "",
        "baseline_s_per_token": decode_times[0] if decode_times else "",
        "flashmtp_s_per_token": decode_times[1] if decode_times else "",
        "average_acceptance_length": avg_accept or "",
        "draft_accept_rate": f"{accept_rate:.4f}" if accept_rate is not None else "",
        "elapsed_seconds": _last_match(PATTERNS["elapsed"], text) or "",
        "verification_mode": verification_mode,
        "compile_serial_head": compile_flag,
    }


def summarize_run(run_root: Path, *, verify_block: int = 16) -> list[BenchmarkRow]:
    run_root = run_root.resolve()
    manifest_path = run_root / "manifest.tsv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    meta = _infer_run_metadata(run_root)
    rows: list[BenchmarkRow] = []
    with manifest_path.open(encoding="utf-8") as manifest_file:
        for item in csv.DictReader(manifest_file, delimiter="\t"):
            log_path = Path(item["log_path"])
            status_path = Path(item["status_path"])
            status = (
                status_path.read_text(encoding="utf-8", errors="replace").strip()
                if status_path.exists()
                else "missing"
            )
            status_line = status.splitlines()[0] if status else "unknown"
            parsed = parse_log(
                log_path,
                verify_block=verify_block,
                verification_mode_default=meta["verification_mode_default"],
                compile_default=meta["compile_serial_head_default"],
            )
            rows.append(
                BenchmarkRow(
                    run_id=meta["run_id"],
                    model=item["model"],
                    temperature=item["temperature"],
                    verification_mode=parsed.get(
                        "verification_mode", meta["verification_mode_default"]
                    ),
                    compile_serial_head=parsed.get(
                        "compile_serial_head", meta["compile_serial_head_default"]
                    ),
                    dataset=item["dataset"],
                    requested_samples=item["requested_samples"],
                    gpu=item["gpu"],
                    status=status_line,
                    turns=parsed.get("turns", ""),
                    token_weighted_speedup=parsed.get("token_weighted_speedup", ""),
                    throughput_ratio=parsed.get("throughput_ratio", ""),
                    unweighted_speedup=parsed.get("unweighted_speedup", ""),
                    baseline_s_per_token=parsed.get("baseline_s_per_token", ""),
                    flashmtp_s_per_token=parsed.get("flashmtp_s_per_token", ""),
                    average_acceptance_length=parsed.get(
                        "average_acceptance_length", ""
                    ),
                    draft_accept_rate=parsed.get("draft_accept_rate", ""),
                    elapsed_seconds=parsed.get("elapsed_seconds", ""),
                    draft_path=item.get("draft_path", ""),
                    log_path=str(log_path),
                    status_path=str(status_path),
                )
            )
    return rows


def _write_csv(path: Path, rows: list[BenchmarkRow]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _aggregate_json(rows: list[BenchmarkRow]) -> dict[str, Any]:
    completed = [row for row in rows if row.status == "completed"]
    by_run: dict[str, list[BenchmarkRow]] = {}
    for row in rows:
        by_run.setdefault(row.run_id, []).append(row)

    def _mean_speedup(subset: list[BenchmarkRow]) -> float | None:
        values = [
            float(row.token_weighted_speedup)
            for row in subset
            if row.token_weighted_speedup
        ]
        return sum(values) / len(values) if values else None

    run_summaries = {}
    for run_id, run_rows in by_run.items():
        done = [row for row in run_rows if row.status == "completed"]
        run_summaries[run_id] = {
            "total": len(run_rows),
            "completed": len(done),
            "failed": sum(row.status.startswith("failed") for row in run_rows),
            "incomplete": len(run_rows)
            - len(done)
            - sum(row.status.startswith("failed") for row in run_rows),
            "macro_mean_speedup": _mean_speedup(done),
            "datasets": {
                row.dataset: {
                    "speedup": row.token_weighted_speedup,
                    "accept_length": row.average_acceptance_length,
                    "draft_accept_rate": row.draft_accept_rate,
                    "status": row.status,
                }
                for row in run_rows
            },
        }

    return {
        "total_rows": len(rows),
        "completed_rows": len(completed),
        "macro_mean_speedup_all_completed": _mean_speedup(completed),
        "runs": run_summaries,
        "rows": [asdict(row) for row in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_roots",
        nargs="*",
        type=Path,
        help="Benchmark run directories (default: all under benchmark_results/)",
    )
    parser.add_argument(
        "--benchmark-results-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "benchmark_results",
    )
    parser.add_argument("--verify-block", type=int, default=16)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Consolidated CSV path (default: benchmark_results/consolidated_summary.csv)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Consolidated JSON path (default: benchmark_results/consolidated_summary.json)",
    )
    parser.add_argument(
        "--per-run",
        action="store_true",
        help="Also refresh summary.csv inside each run root",
    )
    args = parser.parse_args()

    if args.run_roots:
        run_roots = [path.resolve() for path in args.run_roots]
    else:
        run_roots = sorted(
            path
            for path in args.benchmark_results_dir.iterdir()
            if path.is_dir() and (path / "manifest.tsv").exists()
        )

    all_rows: list[BenchmarkRow] = []
    for run_root in run_roots:
        rows = summarize_run(run_root, verify_block=args.verify_block)
        all_rows.extend(rows)
        if args.per_run:
            _write_csv(run_root / "summary.csv", rows)
            print(
                f"Updated {run_root / 'summary.csv'}: "
                f"completed={sum(r.status == 'completed' for r in rows)} "
                f"total={len(rows)}"
            )

    out_csv = args.output_csv or (
        args.benchmark_results_dir / "consolidated_summary.csv"
    )
    out_json = args.output_json or (
        args.benchmark_results_dir / "consolidated_summary.json"
    )
    if all_rows:
        _write_csv(out_csv, all_rows)
        out_json.write_text(
            json.dumps(_aggregate_json(all_rows), indent=2) + "\n", encoding="utf-8"
        )

    completed = sum(row.status == "completed" for row in all_rows)
    failed = sum(row.status.startswith("failed") for row in all_rows)
    print(
        f"Wrote {out_csv} and {out_json}: "
        f"runs={len(run_roots)} rows={len(all_rows)} "
        f"completed={completed} failed={failed} "
        f"incomplete={len(all_rows) - completed - failed}"
    )


if __name__ == "__main__":
    main()
