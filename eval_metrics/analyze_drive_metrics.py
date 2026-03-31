import argparse
import csv
import json
from pathlib import Path

import numpy as np


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


def _to_bool(value):
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y"}


def _last_nonempty(rows, column):
    for row in reversed(rows):
        value = row.get(column)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _numeric_series(rows, column):
    vals = [_to_float(row.get(column)) for row in rows]
    vals = [v for v in vals if v is not None]
    return vals


def _mean_or_none(values):
    if not values:
        return None
    return float(sum(values) / len(values))


def _summary_lane_offset(rows):
    vals = _numeric_series(rows, "lane_error_m")
    if not vals:
        return {"mean_m": None, "rmse_m": None}
    arr = np.asarray(vals, dtype=float)
    return {
        "mean_m": float(np.mean(arr)),
        "rmse_m": float(np.sqrt(np.mean(np.square(arr)))),
    }


def _mean_velocity_mps(rows):
    return _mean_or_none(_numeric_series(rows, "speed_mps"))


def _mean_velocity_deviation_mps(rows):
    vals = _numeric_series(rows, "velocity_deviation_mps")
    if vals:
        return _mean_or_none(vals)

    paired = []
    for row in rows:
        speed = _to_float(row.get("speed_mps"))
        speed_limit = _to_float(row.get("speed_limit_mps"))
        if speed is None or speed_limit is None:
            continue
        paired.append(abs(speed - speed_limit))
    return _mean_or_none(paired)


def _mean_abs_steering_angle_rad(rows):
    vals = _numeric_series(rows, "steering_angle_rad")
    if not vals:
        return None
    return float(np.mean(np.abs(np.asarray(vals, dtype=float))))


def _steering_entropy(rows, bins=9):
    vals = _numeric_series(rows, "steering_angle_rad")
    if len(vals) < 3:
        return None

    arr = np.asarray(vals, dtype=float)
    predicted = (2.0 * arr[1:-1]) - arr[:-2]
    errors = arr[2:] - predicted
    if errors.size == 0:
        return None
    if np.allclose(errors, errors[0]):
        return 0.0

    lo = float(np.min(errors))
    hi = float(np.max(errors))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0

    hist, _ = np.histogram(errors, bins=max(2, int(bins)), range=(lo, hi))
    total = int(np.sum(hist))
    if total <= 0:
        return None

    probs = hist.astype(float) / float(total)
    probs = probs[probs > 0.0]
    if probs.size == 0:
        return None

    entropy = float(-np.sum(probs * np.log2(probs)))
    max_entropy = float(np.log2(max(2, int(bins))))
    if max_entropy <= 0.0:
        return None
    return float(entropy / max_entropy)


def _completion_time_s(rows):
    # Prefer scenario time if it exists, then control timestamps, then generic
    # wall time so older logs still produce one comparable duration field.
    sim_vals = [_to_float(row.get("simulation_time")) for row in rows]
    sim_vals = [v for v in sim_vals if v is not None]
    if sim_vals:
        return float(max(sim_vals) - min(sim_vals))
    control_vals = [_to_float(row.get("control_apply_ts")) for row in rows]
    control_vals = [v for v in control_vals if v is not None]
    if control_vals:
        return float(max(control_vals) - min(control_vals))
    wall_vals = [_to_float(row.get("wall_time_s")) for row in rows]
    wall_vals = [v for v in wall_vals if v is not None]
    if wall_vals:
        return float(max(wall_vals) - min(wall_vals))
    return None


def _scenario_completion_time_s(rows):
    vals = [_to_float(row.get("scenario_completion_time_s")) for row in rows]
    vals = [v for v in vals if v is not None]
    if vals:
        return float(vals[-1])
    vals = [_to_float(row.get("scenario_elapsed_s")) for row in rows if _to_bool(row.get("scenario_finished"))]
    vals = [v for v in vals if v is not None]
    if vals:
        return float(vals[-1])
    return None


def _scenario_success(rows):
    for row in reversed(rows):
        value = row.get("scenario_success")
        if value in (None, ""):
            continue
        return bool(_to_bool(value))
    return None


def _event_count(rows, *candidate_columns):
    for column in candidate_columns:
        if any(column in row for row in rows):
            return int(sum(1 for row in rows if _to_bool(row.get(column))))
    return None


def _command_success_rate(rows):
    if not rows:
        return None
    if not any("command_correct" in row for row in rows):
        return None
    vals = [row for row in rows if row.get("command_correct", "") != ""]
    if not vals:
        return None
    correct = sum(1 for row in vals if _to_bool(row.get("command_correct")))
    return float(correct / len(vals))


def summarize_rows(rows):
    lane = _summary_lane_offset(rows)
    scenario_name = _last_nonempty(rows, "scenario_name")
    scenario_completion_s = _scenario_completion_time_s(rows) if scenario_name else None
    summary = {
        "rows": len(rows),
        "scenario_name": scenario_name,
        "scenario_kind": _last_nonempty(rows, "scenario_kind"),
        "scenario_status": _last_nonempty(rows, "scenario_status"),
        "scenario_success": _scenario_success(rows),
        "mean_velocity_mps": _mean_velocity_mps(rows),
        "lane_offset_mean_m": lane["mean_m"],
        "steering_angle_mean_rad": _mean_abs_steering_angle_rad(rows),
        "mean_velocity_deviation_mps": _mean_velocity_deviation_mps(rows),
        "steering_entropy": _steering_entropy(rows),
        "lane_error_rmse_m": lane["rmse_m"],
        "lane_invasions": _event_count(rows, "lane_invasion_event", "lane_invasion"),
        "completion_time_s": scenario_completion_s if scenario_completion_s is not None else _completion_time_s(rows),
    }
    if str(scenario_name or "").strip().lower() == "highway_overtake":
        # Only the overtake scenario emits a command-level success flag today.
        command_success_rate = _command_success_rate(rows)
        if command_success_rate is not None:
            summary["command_success_rate"] = command_success_rate
    return summary


def build_parser():
    parser = argparse.ArgumentParser(
        description="Summarize CARLA drive logs into reportable metrics."
    )
    parser.add_argument("--log", type=Path, required=True, help="CARLA run CSV log")
    parser.add_argument(
        "--segment-column",
        default="segment_type",
        help="Optional column used to split straight/curved totals.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("eval_metrics") / "out" / "drive_metrics_summary.json",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    rows = _load_rows(Path(args.log))
    if not rows:
        raise ValueError(f"No rows found in {args.log}")

    payload = {"full_route": summarize_rows(rows)}
    if any(args.segment_column in row and str(row.get(args.segment_column, "")).strip() for row in rows):
        groups = {}
        for row in rows:
            key = str(row.get(args.segment_column, "")).strip()
            if not key:
                continue
            groups.setdefault(key, []).append(row)
        payload["segments"] = {key: summarize_rows(group_rows) for key, group_rows in sorted(groups.items())}

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"Saved JSON to {args.output_json}")


if __name__ == "__main__":
    main()
