from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval_metrics.config import CURRENT_METRICS_ROOT, FIGURES_ROOT
from eval_metrics.plot_scripts.model_plot_utils import CORE_METRICS, load_current_model_entries


def _to_percent(value) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value) * 100.0
    except (TypeError, ValueError):
        return None


def _mean_ci95(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, 0.0
    sd = float(arr.std(ddof=1))
    ci95 = 1.96 * sd / math.sqrt(float(arr.size))
    return mean, ci95


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot a research-style offline metric comparison with fold dots and confidence intervals."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=CURRENT_METRICS_ROOT / "model_metrics.csv",
        help="Current harvested model metrics CSV.",
    )
    parser.add_argument(
        "--metric",
        default="balanced_accuracy",
        choices=CORE_METRICS,
        help="Offline metric to visualize.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=FIGURES_ROOT / "model_accuracy_forest.png",
        help="Path to save the output PNG.",
    )
    parser.add_argument(
        "--title",
        default="Offline Balanced Accuracy By Model",
        help="Optional custom title.",
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    entries = load_current_model_entries(Path(args.input_csv))

    metric_labels = {
        "balanced_accuracy": "Balanced accuracy",
        "macro_precision": "Macro precision",
        "macro_recall": "Macro recall",
        "macro_f1": "Macro F1",
        "worst_class_recall": "Worst-class recall",
    }
    arm_colors = {
        "left": "#f5b60a",
        "right": "#e0001b",
    }

    rows = []
    all_values: list[float] = []
    for entry in entries:
        final_value = _to_percent(entry["metrics"].get(args.metric))
        if final_value is None:
            continue
        fold_values = [
            value
            for value in (_to_percent(fold.get(args.metric)) for fold in entry["fold_metrics"])
            if value is not None
        ]
        mean_value, ci95 = _mean_ci95(fold_values)
        rows.append(
            {
                "label": (
                    f"{entry['display_label']} (bundle only)"
                    if not fold_values
                    else str(entry["display_label"])
                ),
                "final_value": final_value,
                "fold_values": fold_values,
                "mean_value": mean_value,
                "ci95": ci95,
                "color": arm_colors.get(str(entry["arm"]), "#475569"),
            }
        )
        all_values.append(final_value)
        all_values.extend(fold_values)

    if not rows:
        raise ValueError("No model rows contained the requested metric.")

    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(13.4, 6.8), constrained_layout=True)
    rng = np.random.default_rng(11)
    y_positions = np.arange(len(rows), dtype=float)[::-1]

    for idx, row in enumerate(rows):
        y = y_positions[idx]
        ax.axhline(y, color="#e5e7eb", linewidth=0.8, zorder=0)
        if row["fold_values"]:
            jitter = rng.uniform(-0.10, 0.10, size=len(row["fold_values"]))
            ax.scatter(
                row["fold_values"],
                np.full(len(row["fold_values"]), y, dtype=float) + jitter,
                s=28,
                color=row["color"],
                alpha=0.35,
                edgecolors="none",
                zorder=2,
            )
            if row["mean_value"] is not None and row["ci95"] is not None:
                lo = row["mean_value"] - row["ci95"]
                hi = row["mean_value"] + row["ci95"]
                ax.hlines(y, lo, hi, color="#111827", linewidth=2.2, zorder=3)
                ax.vlines([lo, hi], y - 0.08, y + 0.08, color="#111827", linewidth=1.2, zorder=3)

        ax.scatter(
            [row["final_value"]],
            [y],
            marker="D",
            s=88,
            color=row["color"],
            edgecolors="white",
            linewidths=0.9,
            zorder=4,
        )
        ax.text(
            row["final_value"] + 0.55,
            y,
            f"{row['final_value']:.1f}%",
            va="center",
            ha="left",
            fontsize=9,
            color="#111827",
        )

    metric_label = metric_labels.get(args.metric, args.metric.replace("_", " ").title())
    title = args.title.strip() or f"Offline {metric_label}"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(f"{metric_label} (%)")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([row["label"] for row in rows])
    ax.grid(axis="x", alpha=0.22)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    min_value = min(all_values) if all_values else 0.0
    max_value = max(all_values) if all_values else 100.0
    lower_bound = 75.0 if min_value >= 75.0 else max(0.0, 5.0 * math.floor((min_value - 2.0) / 5.0))
    upper_bound = max(100.0, 5.0 * math.ceil((max_value + 2.0) / 5.0))
    ax.set_xlim(lower_bound, upper_bound)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor="#334155",
            markeredgecolor="white",
            markersize=8,
            label=f"Saved {metric_label.lower()}",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#64748b",
            alpha=0.35,
            markersize=6,
            label=f"Per-fold {metric_label.lower()}",
        ),
        Line2D(
            [0],
            [0],
            color="#111827",
            linewidth=2.2,
            label=f"Mean per-fold {metric_label.lower()} +/- 95% confidence interval",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
    )

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=220, bbox_inches="tight")
    print(f"Saved figure to {args.output_png}")


if __name__ == "__main__":
    main()
