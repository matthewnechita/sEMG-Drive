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


def _summary_lane_error(rows):
    vals = [_to_float(row.get("lane_error_m")) for row in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"mean_m": None, "rmse_m": None}
    arr = np.asarray(vals, dtype=float)
    return {
        "mean_m": float(arr.mean()),
        "rmse_m": float(np.sqrt(np.mean(np.square(arr)))),
    }


def _completion_time_s(rows):
    sim_vals = [_to_float(row.get("simulation_time")) for row in rows]
    sim_vals = [v for v in sim_vals if v is not None]
    if sim_vals:
        return float(max(sim_vals) - min(sim_vals))
    wall_vals = [_to_float(row.get("wall_time_s")) for row in rows]
    wall_vals = [v for v in wall_vals if v is not None]
    if wall_vals:
        return float(max(wall_vals) - min(wall_vals))
    return None


def _steering_smoothness(rows):
    steer = [_to_float(row.get("steer")) for row in rows]
    steer = [v for v in steer if v is not None]
    if len(steer) < 2:
        return None
    diffs = np.diff(np.asarray(steer, dtype=float))
    return float(np.mean(np.abs(diffs)))


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
    lane = _summary_lane_error(rows)
    return {
        "rows": len(rows),
        "lane_error_mean_m": lane["mean_m"],
        "lane_error_rmse_m": lane["rmse_m"],
        "lane_invasions": _event_count(rows, "lane_invasion_event", "lane_invasion"),
        "collisions": _event_count(rows, "collision_event", "collision"),
        "completion_time_s": _completion_time_s(rows),
        "steering_smoothness": _steering_smoothness(rows),
        "command_success_rate": _command_success_rate(rows),
    }


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
