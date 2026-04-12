from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval_metrics.config import CURRENT_METRICS_ROOT, FIGURES_ROOT


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _pretty_scenario(name: str, run_dir: str) -> str:
    text = str(name or "").strip().lower()
    if text == "lane_keep_5min":
        return "Lane keep"
    if text == "highway_overtake":
        return "Highway overtake"
    fallback = str(run_dir or "").replace("_eval", "").replace("_", " ").strip()
    return fallback.title() or "Scenario"


def _load_drive_summary(path_text: str) -> dict[str, object]:
    path = Path(str(path_text or "").strip())
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    full_route = payload.get("full_route", {})
    return full_route if isinstance(full_route, dict) else {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot report-ready end-to-end latency ECDFs from staged latency joins."
    )
    parser.add_argument(
        "--run-index",
        type=Path,
        default=CURRENT_METRICS_ROOT / "carla_run_index.csv",
        help="Run index CSV from gather_current_metrics.py",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=FIGURES_ROOT / "latency_ecdf.png",
        help="Path to save the PNG figure.",
    )
    parser.add_argument(
        "--title",
        default="End-to-End Latency Distribution",
        help="Optional figure title.",
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    run_rows = _load_csv_rows(Path(args.run_index))

    grouped: dict[str, list[float]] = defaultdict(list)
    run_counts: dict[str, int] = defaultdict(int)
    for run_row in run_rows:
        latency_csv = str(run_row.get("latency_csv") or "").strip()
        if not latency_csv:
            continue
        latency_rows = _load_csv_rows(Path(latency_csv))
        values = [
            _to_float(row.get("end_to_end_latency_ms"))
            for row in latency_rows
        ]
        values = [float(value) for value in values if value is not None]
        if not values:
            continue
        drive_summary = _load_drive_summary(str(run_row.get("drive_json") or ""))
        scenario_name = _pretty_scenario(
            str(drive_summary.get("scenario_name") or ""),
            str(run_row.get("run_dir") or ""),
        )
        grouped[scenario_name].extend(values)
        run_counts[scenario_name] += 1

    if not grouped:
        raise ValueError(
            "No populated latency_joined.csv files were found in the staged current_metrics run index."
        )

    colors = {
        "Lane keep": "#1d4ed8",
        "Highway overtake": "#dc2626",
    }

    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    summary_lines = []
    global_max = 0.0
    for scenario_name in sorted(grouped):
        values = np.sort(np.asarray(grouped[scenario_name], dtype=float))
        y = np.arange(1, values.size + 1, dtype=float) / float(values.size)
        color = colors.get(scenario_name, None)
        ax.step(
            values,
            y,
            where="post",
            linewidth=2.4,
            color=color,
            label=f"{scenario_name} (runs={run_counts[scenario_name]}, latency samples={values.size})",
        )
        mean_ms = float(values.mean())
        p95_ms = float(np.percentile(values, 95))
        global_max = max(global_max, float(values.max()))
        ax.axvline(mean_ms, ymin=0.0, ymax=0.93, linestyle="--", linewidth=1.6, color=color, alpha=0.8)
        ax.axvline(p95_ms, ymin=0.0, ymax=0.93, linestyle=":", linewidth=2.0, color=color, alpha=0.95)
        summary_lines.append(
            f"{scenario_name}: mean {mean_ms:.1f} ms, p95 {p95_ms:.1f} ms"
        )

    ax.set_title(args.title.strip() or "End-to-End Latency Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Empirical cumulative probability")
    ax.set_ylim(0.0, 1.01)
    ax.set_xlim(0.0, global_max * 1.03 if global_max > 0 else 1.0)
    ax.grid(alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    ax.text(
        0.985,
        0.985,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.95},
    )

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=220, bbox_inches="tight")
    print(f"Saved figure to {args.output_png}")


if __name__ == "__main__":
    main()
