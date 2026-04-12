from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from matplotlib import patches

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval_metrics.config import CURRENT_METRICS_ROOT, FIGURES_ROOT
from eval_metrics.plot_scripts.model_plot_utils import CORE_METRICS, load_current_model_entries


METRIC_LABELS = {
    "balanced_accuracy": "Balanced\naccuracy",
    "macro_precision": "Macro\nprecision",
    "macro_recall": "Macro\nrecall",
    "macro_f1": "Macro\nF1",
    "worst_class_recall": "Worst-class\nrecall",
}


def _to_percent(value) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value) * 100.0
    except (TypeError, ValueError):
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot a research-style heatmap of offline model metrics."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=CURRENT_METRICS_ROOT / "model_metrics.csv",
        help="Current harvested model metrics CSV.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=FIGURES_ROOT / "model_metric_heatmap.png",
        help="Path to save the output PNG.",
    )
    parser.add_argument(
        "--title",
        default="Offline Model Metric Comparison",
        help="Optional custom title.",
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    entries = load_current_model_entries(Path(args.input_csv))

    row_labels = []
    matrix_rows = []
    for entry in entries:
        row = []
        for metric_key in CORE_METRICS:
            value = _to_percent(entry["metrics"].get(metric_key))
            row.append(np.nan if value is None else float(value))
        if all(np.isnan(value) for value in row):
            continue
        row_labels.append(str(entry["display_label"]))
        matrix_rows.append(row)

    if not matrix_rows:
        raise ValueError("No offline metric rows were available for the heatmap.")

    matrix = np.asarray(matrix_rows, dtype=float)
    vmin = float(np.nanmin(matrix))
    vmax = float(np.nanmax(matrix))
    display_vmin = 75.0 if vmin >= 75.0 else max(0.0, 5.0 * math.floor((vmin - 2.0) / 5.0))
    display_vmax = max(100.0, 5.0 * math.ceil((vmax + 1.0) / 5.0))

    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(10.8, 6.4), constrained_layout=True)
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=display_vmin, vmax=display_vmax, aspect="auto")

    ax.set_xticks(range(len(CORE_METRICS)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels([METRIC_LABELS[key] for key in CORE_METRICS])
    ax.set_yticklabels(row_labels)
    ax.set_title(args.title.strip() or "Offline Model Metric Comparison", fontsize=14, fontweight="bold")

    best_by_metric = np.nanmax(matrix, axis=0)
    midpoint = (display_vmin + display_vmax) / 2.0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if np.isnan(value):
                text = "NA"
                color = "#111827"
            else:
                text = f"{value:.1f}%"
                color = "white" if value >= midpoint else "#111827"
            is_best = (not np.isnan(value)) and np.isclose(value, best_by_metric[col_idx])
            ax.text(
                col_idx,
                row_idx,
                text,
                ha="center",
                va="center",
                fontsize=9,
                color=color,
                fontweight="bold" if is_best else "normal",
            )
            if is_best:
                ax.add_patch(
                    patches.Rectangle(
                        (col_idx - 0.5, row_idx - 0.5),
                        1.0,
                        1.0,
                        fill=False,
                        edgecolor="#0f172a",
                        linewidth=1.3,
                    )
                )

    ax.set_xticks(np.arange(-0.5, len(CORE_METRICS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    colorbar = fig.colorbar(image, ax=ax, shrink=0.9)
    colorbar.set_label("Percent")

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=220, bbox_inches="tight")
    print(f"Saved figure to {args.output_png}")


if __name__ == "__main__":
    main()
