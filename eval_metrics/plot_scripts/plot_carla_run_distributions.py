from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval_metrics.config import CURRENT_METRICS_ROOT, FIGURES_ROOT


METRICS = [
    ("completion_time_s", "Completion Time (s)"),
    ("mean_velocity_mps", "Mean Velocity (m/s)"),
    ("lane_offset_mean_m", "Lane Offset Mean (m)"),
    ("steering_angle_mean_rad", "Steering Angle Mean (rad)"),
    ("steering_entropy", "Steering Entropy"),
    ("lane_error_rmse_m", "Lane Error RMSE (m)"),
    ("lane_invasions", "Lane Invasions"),
]


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


def _load_drive_summary(path_text: str) -> dict[str, object]:
    path = Path(str(path_text or "").strip())
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    full_route = payload.get("full_route", {})
    return full_route if isinstance(full_route, dict) else {}


def _pretty_scenario(name: str, run_dir: str) -> str:
    text = str(name or "").strip().lower()
    if text == "lane_keep_5min":
        return "Lane keep"
    if text == "highway_overtake":
        return "Highway overtake"
    fallback = str(run_dir or "").replace("_eval", "").replace("_", " ").strip()
    return fallback.title() or "Scenario"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot report-ready CARLA run distributions from staged drive summaries."
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
        default=FIGURES_ROOT / "carla_run_distributions.png",
        help="Path to save the PNG figure.",
    )
    parser.add_argument(
        "--title",
        default="CARLA Run-Level Performance",
        help="Optional figure title.",
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    run_rows = _load_csv_rows(Path(args.run_index))

    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for run_row in run_rows:
        drive_summary = _load_drive_summary(str(run_row.get("drive_json") or ""))
        if not drive_summary:
            continue
        scenario_name = _pretty_scenario(
            str(drive_summary.get("scenario_name") or ""),
            str(run_row.get("run_dir") or ""),
        )
        metric_row = {}
        for metric_key, _ in METRICS:
            metric_value = _to_float(drive_summary.get(metric_key))
            if metric_value is not None:
                metric_row[metric_key] = float(metric_value)
        if metric_row:
            grouped[scenario_name].append(metric_row)

    if not grouped:
        raise ValueError("No drive metrics were found in the staged current_metrics run index.")

    scenario_names = sorted(grouped)
    colors = {
        "Lane keep": "#1d4ed8",
        "Highway overtake": "#dc2626",
    }
    rng = np.random.default_rng(7)

    from matplotlib import pyplot as plt

    ncols = 2
    nrows = int(math.ceil(len(METRICS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 14), constrained_layout=True)
    axes_list = np.atleast_1d(axes).flatten().tolist()

    for ax, (metric_key, metric_label) in zip(axes_list, METRICS):
        data = []
        for scenario_name in scenario_names:
            values = [
                float(run_metrics[metric_key])
                for run_metrics in grouped[scenario_name]
                if metric_key in run_metrics
            ]
            data.append(values)

        positions = list(range(len(scenario_names)))
        box = ax.boxplot(
            data,
            positions=positions,
            widths=0.52,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111827", "linewidth": 1.8},
            whiskerprops={"color": "#64748b"},
            capprops={"color": "#64748b"},
        )
        for patch, scenario_name in zip(box["boxes"], scenario_names):
            patch.set_facecolor(colors.get(scenario_name, "#94a3b8"))
            patch.set_alpha(0.35)
            patch.set_edgecolor(colors.get(scenario_name, "#475569"))
            patch.set_linewidth(1.5)

        for x, scenario_name, values in zip(positions, scenario_names, data):
            if not values:
                continue
            jitter = rng.uniform(-0.11, 0.11, size=len(values))
            ax.scatter(
                np.asarray([x] * len(values), dtype=float) + jitter,
                values,
                s=32,
                color=colors.get(scenario_name, "#475569"),
                alpha=0.85,
                edgecolors="white",
                linewidths=0.6,
                zorder=3,
            )
            mean_value = float(np.mean(values))
            ax.scatter(
                [x],
                [mean_value],
                marker="D",
                s=52,
                color="#111827",
                edgecolors="white",
                linewidths=0.7,
                zorder=4,
            )

        xtick_labels = [f"{name}\n(runs={len(data[idx])})" for idx, name in enumerate(scenario_names)]
        ax.set_xticks(positions)
        ax.set_xticklabels(xtick_labels)
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes_list[len(METRICS):]:
        ax.axis("off")

    fig.suptitle(args.title.strip() or "CARLA Run-Level Performance", fontsize=15, fontweight="bold")

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=220, bbox_inches="tight")
    print(f"Saved figure to {args.output_png}")


if __name__ == "__main__":
    main()
