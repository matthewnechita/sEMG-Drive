import argparse
import csv
from pathlib import Path


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


def _label_for_row(row):
    scenario = str(row.get("drive_scenario_name") or "").strip()
    task = str(row.get("task") or "").strip()
    scope = str(row.get("model_scope") or "").strip()
    arm = str(row.get("arm") or "").strip().title()
    parts = [part for part in (scenario, task, scope, arm) if part]
    if parts:
        return "\n".join(parts)
    return str(row.get("row_id") or "").strip() or "row"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot aggregate end-to-end latency summaries from evaluation_aggregate_table.csv."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("eval_metrics") / "out" / "tables" / "evaluation_aggregate_table.csv",
        help="Aggregate table CSV from build_eval_tables.py",
    )
    parser.add_argument(
        "--deliverable-bucket",
        default="capstone_report",
        help="Optional deliverable bucket filter, such as capstone_report or research_paper.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("eval_metrics") / "out" / "tables" / "latency_summary_bars.png",
        help="Path to save the output PNG.",
    )
    parser.add_argument(
        "--title",
        default="End-to-End Latency Summary",
        help="Optional custom plot title.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    rows = _load_rows(args.input_csv)
    if not rows:
        raise ValueError(f"No rows found in {args.input_csv}")

    deliverable_bucket = str(args.deliverable_bucket or "").strip().lower()
    filtered = []
    for row in rows:
        if deliverable_bucket and str(row.get("deliverable_bucket") or "").strip().lower() != deliverable_bucket:
            continue
        if _to_float(row.get("lat_e2e_mean_ms_mean")) is None and _to_float(row.get("lat_e2e_p95_ms_mean")) is None:
            continue
        filtered.append(row)

    if not filtered:
        raise ValueError("No aggregate latency rows matched the requested filters.")

    labels = [_label_for_row(row) for row in filtered]
    mean_latency = [_to_float(row.get("lat_e2e_mean_ms_mean")) for row in filtered]
    mean_latency_sd = [_to_float(row.get("lat_e2e_mean_ms_sd")) or 0.0 for row in filtered]
    p95_latency = [_to_float(row.get("lat_e2e_p95_ms_mean")) for row in filtered]
    p95_latency_sd = [_to_float(row.get("lat_e2e_p95_ms_sd")) or 0.0 for row in filtered]

    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.suptitle(args.title.strip() or "End-to-End Latency Summary", fontsize=14, fontweight="bold")
    x = list(range(len(filtered)))
    color = "#0f766e"
    plots = [
        (axes[0], mean_latency, mean_latency_sd, "Mean End-to-End Latency (ms)"),
        (axes[1], p95_latency, p95_latency_sd, "P95 End-to-End Latency (ms)"),
    ]

    for ax, values, errors, ylabel in plots:
        safe_values = [0.0 if value is None else float(value) for value in values]
        bars = ax.bar(x, safe_values, yerr=errors, capsize=4, color=color, width=0.65)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        upper = max(safe_values) if safe_values else 0.0
        ax.set_ylim(0.0, upper + max(5.0, upper * 0.18))
        for bar, value in zip(bars, safe_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max(0.5, bar.get_height() * 0.03),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#333333",
            )

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=180, bbox_inches="tight")
    print(f"Saved figure to {args.output_png}")


if __name__ == "__main__":
    main()
