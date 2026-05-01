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
    ("mean_velocity_deviation_mps", "Mean Velocity Deviation (m/s)"),
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


def _pretty_scope(scope: str) -> str:
    text = str(scope or "").strip().lower()
    if "cross" in text:
        return "Cross-subject"
    if "per" in text:
        return "Per-subject"
    return str(scope or "").replace("_", " ").strip().title()


def _group_label(run_row: dict[str, str], scenario_name: str) -> str:
    scope = str(run_row.get("model_scope") or "").strip()
    if not scope:
        return scenario_name
    return f"{_pretty_scope(scope)} | {scenario_name}"


def _group_sort_key(label: str) -> tuple[int, int, str]:
    if " | " in label:
        scope_text, scenario_text = label.split(" | ", 1)
        scope_key = 0 if scope_text == "Cross-subject" else 1 if scope_text == "Per-subject" else 99
        scenario_key = 0 if scenario_text == "Highway overtake" else 1 if scenario_text == "Lane keep" else 99
        return scope_key, scenario_key, label
    scenario_key = 0 if label == "Highway overtake" else 1 if label == "Lane keep" else 99
    return 0, scenario_key, label


def _scenario_color(label: str) -> str:
    if "Highway overtake" in label:
        return "#dc2626"
    return "#1d4ed8"


def _tick_label(label: str, run_count: int) -> str:
    if " | " not in label:
        return f"{label}\n(runs={run_count})"
    scope_text, scenario_text = label.split(" | ", 1)
    return f"{scope_text}\n{scenario_text}\n(runs={run_count})"


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
            label = _group_label(run_row, scenario_name)
            grouped[label].append(metric_row)

    if not grouped:
        raise ValueError("No drive metrics were found in the staged current_metrics run index.")

    scenario_names = sorted(grouped, key=_group_sort_key)
    rng = np.random.default_rng(7)

    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    panel_count = len(METRICS) + 1  # reserve one panel for the legend
    ncols = 3
    nrows = int(math.ceil(panel_count / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 4.35 * nrows), constrained_layout=True)
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
            patch.set_facecolor(_scenario_color(scenario_name))
            patch.set_alpha(0.35)
            patch.set_edgecolor(_scenario_color(scenario_name))
            patch.set_linewidth(1.5)

        for x, scenario_name, values in zip(positions, scenario_names, data):
            if not values:
                continue
            jitter = rng.uniform(-0.11, 0.11, size=len(values))
            ax.scatter(
                np.asarray([x] * len(values), dtype=float) + jitter,
                values,
                s=32,
                color=_scenario_color(scenario_name),
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

        xtick_labels = [_tick_label(name, len(data[idx])) for idx, name in enumerate(scenario_names)]
        ax.set_xticks(positions)
        ax.set_xticklabels(xtick_labels)
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    extra_axes = axes_list[len(METRICS):]
    if extra_axes:
        legend_ax = extra_axes[0]
        legend_ax.axis("off")
        legend_handles = [
            Patch(
                facecolor=_scenario_color("Lane keep"),
                edgecolor=_scenario_color("Lane keep"),
                alpha=0.35,
                label="Lane keep run distribution",
            ),
            Patch(
                facecolor=_scenario_color("Highway overtake"),
                edgecolor=_scenario_color("Highway overtake"),
                alpha=0.35,
                label="Highway overtake run distribution",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=6.5,
                markerfacecolor="#475569",
                markeredgecolor="white",
                markeredgewidth=0.6,
                label="Individual run value",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="None",
                markersize=7.0,
                markerfacecolor="#111827",
                markeredgecolor="white",
                markeredgewidth=0.7,
                label="Mean across runs",
            ),
            Line2D(
                [0],
                [0],
                color="#111827",
                linewidth=1.8,
                label="Median within run distribution",
            ),
        ]
        legend_ax.legend(
            handles=legend_handles,
            loc="center",
            frameon=False,
            ncol=1,
            handlelength=1.8,
            borderaxespad=0.0,
            labelspacing=1.0,
        )
        for ax in extra_axes[1:]:
            ax.axis("off")

    fig.suptitle(args.title.strip() or "CARLA Run-Level Performance", fontsize=15, fontweight="bold")

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=220, bbox_inches="tight")
    print(f"Saved figure to {args.output_png}")


if __name__ == "__main__":
    main()
