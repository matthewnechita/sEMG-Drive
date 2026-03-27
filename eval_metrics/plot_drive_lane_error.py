import argparse
from pathlib import Path

import numpy as np

from analyze_drive_metrics import _load_rows, _to_bool, _to_float, summarize_rows


def _time_axis_s(rows):
    sim_vals = [_to_float(row.get("simulation_time")) for row in rows]
    if any(value is not None for value in sim_vals):
        base = next((value for value in sim_vals if value is not None), 0.0)
        return [None if value is None else float(value - base) for value in sim_vals], "Simulation time (s)"

    wall_vals = [_to_float(row.get("wall_time_s")) for row in rows]
    if any(value is not None for value in wall_vals):
        base = next((value for value in wall_vals if value is not None), 0.0)
        return [None if value is None else float(value - base) for value in wall_vals], "Wall time (s)"

    return list(range(len(rows))), "Sample index"


def _rolling_mean(values, window):
    window = max(1, int(window))
    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="same")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot CARLA lane error over time from a carla_drive CSV log."
    )
    parser.add_argument("--log", type=Path, required=True, help="CARLA run CSV log")
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("eval_metrics") / "out" / "lane_error_plot.png",
        help="Path to save the lane error plot PNG.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=25,
        help="Rolling mean window in samples for the smoothed overlay.",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Optional custom plot title.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    rows = _load_rows(args.log)
    if not rows:
        raise ValueError(f"No rows found in {args.log}")

    time_axis, time_label = _time_axis_s(rows)
    lane_vals = [_to_float(row.get("lane_error_m")) for row in rows]

    paired = [
        (float(t), float(lane))
        for t, lane in zip(time_axis, lane_vals)
        if t is not None and lane is not None
    ]
    if not paired:
        raise ValueError(f"No usable lane_error_m values found in {args.log}")

    times = np.asarray([item[0] for item in paired], dtype=float)
    lane_error = np.asarray([item[1] for item in paired], dtype=float)
    smooth = _rolling_mean(lane_error, args.rolling_window)

    invasion_times = [
        float(t)
        for row, t in zip(rows, time_axis)
        if t is not None and _to_bool(row.get("lane_invasion_event"))
    ]

    summary = summarize_rows(rows)
    scenario_name = summary.get("scenario_name") or args.log.stem
    title = args.title.strip() or f"CARLA lane error: {scenario_name}"

    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.plot(times, lane_error, color="#2b6cb0", linewidth=1.0, alpha=0.55, label="Lane error")
    if args.rolling_window > 1:
        ax.plot(
            times,
            smooth,
            color="#c05621",
            linewidth=2.0,
            label=f"Rolling mean ({int(args.rolling_window)} samples)",
        )

    mean_lane_error = summary.get("lane_error_mean_m")
    if mean_lane_error is not None:
        ax.axhline(
            float(mean_lane_error),
            color="#2f855a",
            linestyle="--",
            linewidth=1.4,
            label=f"Mean {float(mean_lane_error):.3f} m",
        )

    if invasion_times:
        ymax = float(np.nanmax(lane_error)) if lane_error.size else 0.0
        marker_y = ymax * 1.02 if ymax > 0.0 else 0.02
        ax.scatter(
            invasion_times,
            np.full(len(invasion_times), marker_y, dtype=float),
            color="#c53030",
            marker="x",
            s=32,
            label=f"Lane invasions ({len(invasion_times)})",
            zorder=3,
        )

    summary_lines = [
        f"Rows: {int(summary.get('rows') or 0)}",
        f"Mean: {float(mean_lane_error):.3f} m" if mean_lane_error is not None else "Mean: n/a",
        (
            f"RMSE: {float(summary['lane_error_rmse_m']):.3f} m"
            if summary.get("lane_error_rmse_m") is not None
            else "RMSE: n/a"
        ),
        (
            f"Invasions: {int(summary['lane_invasions'])}"
            if summary.get("lane_invasions") is not None
            else "Invasions: n/a"
        ),
        (
            f"Completion: {float(summary['completion_time_s']):.1f} s"
            if summary.get("completion_time_s") is not None
            else "Completion: n/a"
        ),
    ]
    if summary.get("scenario_status"):
        summary_lines.append(f"Status: {summary['scenario_status']}")
    if summary.get("scenario_success") is not None:
        summary_lines.append(f"Success: {bool(summary['scenario_success'])}")

    ax.text(
        0.01,
        0.99,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.88, "edgecolor": "#d1d5db"},
    )

    ax.set_title(title)
    ax.set_xlabel(time_label)
    ax.set_ylabel("Lane error (m)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=180)
    print(f"Saved figure to {args.output_png}")


if __name__ == "__main__":
    main()
