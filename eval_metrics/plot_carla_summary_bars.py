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
        description="Plot the final aggregate CARLA metrics from evaluation_aggregate_table.csv."
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
        default=Path("eval_metrics") / "out" / "tables" / "carla_summary_bars.png",
        help="Path to save the output PNG.",
    )
    parser.add_argument(
        "--title",
        default="CARLA Performance Summary",
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
        # Skip rows that never populated CARLA metrics so the plot stays aligned
        # with simulator-backed evaluation entries only.
        if _to_float(row.get("drive_lane_error_rmse_m_mean")) is None and _to_float(row.get("drive_completion_time_s_mean")) is None:
            continue
        filtered.append(row)

    if not filtered:
        raise ValueError("No aggregate CARLA rows matched the requested filters.")

    labels = [_label_for_row(row) for row in filtered]
    success_rate = [(_to_float(row.get("drive_scenario_success_rate")) or 0.0) * 100.0 for row in filtered]
    completion = [_to_float(row.get("drive_completion_time_s_mean")) for row in filtered]
    completion_sd = [_to_float(row.get("drive_completion_time_s_sd")) or 0.0 for row in filtered]
    mean_velocity = [_to_float(row.get("drive_mean_velocity_mps_mean")) for row in filtered]
    mean_velocity_sd = [_to_float(row.get("drive_mean_velocity_mps_sd")) or 0.0 for row in filtered]
    lane_offset = [_to_float(row.get("drive_lane_offset_mean_m_mean")) for row in filtered]
    lane_offset_sd = [_to_float(row.get("drive_lane_offset_mean_m_sd")) or 0.0 for row in filtered]
    steering_angle = [_to_float(row.get("drive_steering_angle_mean_rad_mean")) for row in filtered]
    steering_angle_sd = [_to_float(row.get("drive_steering_angle_mean_rad_sd")) or 0.0 for row in filtered]
    steering_entropy = [_to_float(row.get("drive_steering_entropy_mean")) for row in filtered]
    steering_entropy_sd = [_to_float(row.get("drive_steering_entropy_sd")) or 0.0 for row in filtered]
    lane_rmse = [_to_float(row.get("drive_lane_error_rmse_m_mean")) for row in filtered]
    lane_rmse_sd = [_to_float(row.get("drive_lane_error_rmse_m_sd")) or 0.0 for row in filtered]
    invasions = [_to_float(row.get("drive_lane_invasions_mean")) for row in filtered]
    invasions_sd = [_to_float(row.get("drive_lane_invasions_sd")) or 0.0 for row in filtered]

    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(4, 2, figsize=(15, 14), constrained_layout=True)
    fig.suptitle(args.title.strip() or "CARLA Performance Summary", fontsize=14, fontweight="bold")
    x = list(range(len(filtered)))
    color = "#1f4b99"

    plots = [
        (axes[0][0], success_rate, None, "Scenario Success Rate (%)"),
        (axes[0][1], completion, completion_sd, "Completion Time (s)"),
        (axes[1][0], mean_velocity, mean_velocity_sd, "Mean Velocity (m/s)"),
        (axes[1][1], lane_offset, lane_offset_sd, "Lane Offset Mean (m)"),
        (axes[2][0], steering_angle, steering_angle_sd, "Steering Angle Mean (rad)"),
        (axes[2][1], steering_entropy, steering_entropy_sd, "Steering Entropy"),
        (axes[3][0], lane_rmse, lane_rmse_sd, "Lane Error RMSE (m)"),
        (axes[3][1], invasions, invasions_sd, "Lane Invasions"),
    ]

    for ax, values, errors, ylabel in plots:
        safe_values = [0.0 if value is None else float(value) for value in values]
        if errors is None:
            bars = ax.bar(x, safe_values, color=color, width=0.65)
        else:
            bars = ax.bar(x, safe_values, yerr=errors, capsize=4, color=color, width=0.65)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        upper = max(safe_values) if safe_values else 0.0
        if ylabel.endswith("(%)"):
            ax.set_ylim(0.0, max(100.0, upper + 10.0))
        else:
            ax.set_ylim(0.0, upper + max(0.1, upper * 0.18))
        for bar, value in zip(bars, safe_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max(0.02, bar.get_height() * 0.03),
                f"{value:.2f}" if ylabel != "Scenario Success Rate (%)" else f"{value:.1f}",
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
