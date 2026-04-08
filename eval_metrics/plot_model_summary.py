import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from matplotlib.patches import Rectangle


METRICS = [
    ("balanced_accuracy", "Balanced Accuracy"),
    ("macro_precision", "Macro Precision"),
    ("macro_recall", "Macro Recall"),
    ("macro_f1", "Macro F1"),
    ("worst_class_recall", "Worst-Class Recall"),
]


def _load_rows(path: Path):
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


def _row_matches_tokens(row, include_tokens, exclude_tokens):
    haystack = " ".join(
        str(row.get(key) or "").strip().lower()
        for key in ("filename", "path", "bundle_scope", "arm", "subject", "gesture_bucket")
    )
    if include_tokens and not all(token in haystack for token in include_tokens):
        return False
    if exclude_tokens and any(token in haystack for token in exclude_tokens):
        return False
    return True


def _scope_order(value):
    text = str(value or "").strip().lower()
    if text == "cross_subject":
        return 0
    if text == "per_subject":
        return 1
    return 2


def _arm_order(value):
    text = str(value or "").strip().lower()
    if text == "left":
        return 0
    if text == "right":
        return 1
    return 2


def _latest_rows(rows):
    latest = {}
    for row in rows:
        key = (
            str(row.get("bundle_scope") or "").strip().lower(),
            str(row.get("arm") or "").strip().lower(),
            str(row.get("subject") or "").strip().lower(),
            str(row.get("gesture_bucket") or "").strip().lower(),
        )
        created_at = str(row.get("created_at") or "").strip()
        current = latest.get(key)
        if current is None or created_at > str(current.get("created_at") or "").strip():
            latest[key] = row
    return list(latest.values())


def _mean_sd(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None, None
    mean = float(sum(vals) / len(vals))
    if len(vals) == 1:
        return mean, 0.0
    variance = float(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
    return mean, math.sqrt(variance)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot a multi-metric offline model summary from harvested bundle rows."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("eval_metrics") / "out" / "current_metrics" / "model_metrics.csv",
        help="CSV from harvest_model_metrics.py",
    )
    parser.add_argument(
        "--gesture-bucket",
        default="4_gesture",
        help="Optional gesture bucket filter such as 4_gesture.",
    )
    parser.add_argument(
        "--include-token",
        action="append",
        default=[],
        help="Optional token that must appear in the filename/path metadata. Repeatable.",
    )
    parser.add_argument(
        "--exclude-token",
        action="append",
        default=[],
        help="Optional token to exclude from the filename/path metadata. Repeatable.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("eval_metrics") / "out" / "current_metrics" / "model_summary.png",
        help="Path to save the output PNG.",
    )
    parser.add_argument(
        "--title",
        default="Offline Model Summary",
        help="Optional custom plot title.",
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Keep only the latest row per arm/scope/subject/gesture bucket before aggregation.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    rows = _load_rows(args.input_csv)
    if not rows:
        raise ValueError(f"No rows found in {args.input_csv}")

    include_tokens = [str(token).strip().lower() for token in args.include_token if str(token).strip()]
    exclude_tokens = [str(token).strip().lower() for token in args.exclude_token if str(token).strip()]
    target_bucket = str(args.gesture_bucket or "").strip().lower()

    filtered = []
    for row in rows:
        if target_bucket and str(row.get("gesture_bucket") or "").strip().lower() != target_bucket:
            continue
        if not _row_matches_tokens(row, include_tokens, exclude_tokens):
            continue
        filtered.append(row)

    if not filtered:
        raise ValueError("No model rows matched the requested filters.")

    if args.latest_only:
        filtered = _latest_rows(filtered)

    grouped = defaultdict(list)
    for row in filtered:
        key = (
            str(row.get("bundle_scope") or "").strip().lower(),
            str(row.get("arm") or "").strip().lower(),
        )
        grouped[key].append(row)

    if not grouped:
        raise ValueError("No grouped model rows were produced.")

    scope_keys = sorted({key[0] for key in grouped}, key=_scope_order)
    arm_keys = sorted({key[1] for key in grouped}, key=_arm_order)
    scope_labels = {
        "cross_subject": "Cross-subject",
        "per_subject": "Per-subject",
    }
    arm_labels = {
        "left": "Left arm",
        "right": "Right arm",
    }
    arm_colors = {
        "left": "#f5b60a",
        "right": "#e0001b",
    }

    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    fig.suptitle(args.title.strip() or "Offline Model Summary", fontsize=14, fontweight="bold")
    axes_flat = list(axes.flatten())
    x = list(range(len(scope_keys)))
    bar_width = 0.34

    for axis_index, (metric_key, metric_label) in enumerate(METRICS):
        ax = axes_flat[axis_index]
        all_values = []
        legend_handles = []
        for arm_index, arm in enumerate(arm_keys):
            offset = (arm_index - (len(arm_keys) - 1) / 2.0) * bar_width
            arm_positions = []
            arm_means = []
            arm_sds = []
            for scope_index, scope in enumerate(scope_keys):
                values = [_to_float(row.get(metric_key)) for row in grouped.get((scope, arm), [])]
                mean, sd = _mean_sd(values)
                if mean is None:
                    continue
                arm_positions.append(float(scope_index) + offset)
                arm_means.append(mean * 100.0)
                arm_sds.append((sd * 100.0) if sd is not None else 0.0)
            bars = ax.bar(
                arm_positions,
                arm_means,
                yerr=arm_sds,
                capsize=4,
                color=arm_colors.get(arm, "#666666"),
                width=bar_width,
                label=arm_labels.get(arm, arm.title()),
            )
            legend_handles.append(
                Rectangle((0, 0), 1, 1, color=arm_colors.get(arm, "#666666"), label=arm_labels.get(arm, arm.title()))
            )
            all_values.extend(arm_means)
            for bar, value in zip(bars, arm_means):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    value + 0.8,
                    f"{value:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    color="#333333",
                )

        ax.set_title(metric_label)
        ax.set_ylabel("Percent")
        ax.set_xticks(x)
        ax.set_xticklabels([scope_labels.get(scope, scope.replace("_", " ").title()) for scope in scope_keys])
        min_value = min(all_values) if all_values else 0.0
        lower_bound = 75.0 if min_value >= 75.0 else max(0.0, 5.0 * math.floor(min_value / 5.0))
        ax.set_ylim(lower_bound, max(100.0, max(all_values) + 6.0 if all_values else 100.0))
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if axis_index == 0:
            ax.legend(handles=legend_handles, loc="upper left", frameon=False)

    axes_flat[-1].axis("off")

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=180, bbox_inches="tight")
    print(f"Saved figure to {args.output_png}")


if __name__ == "__main__":
    main()
