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


def _summary_lane_error(rows):
    vals = [_to_float(row.get("lane_error_m")) for row in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"rmse_m": None}
    arr = np.asarray(vals, dtype=float)
    return {
        "rmse_m": float(np.sqrt(np.mean(np.square(arr)))),
    }


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
    lane = _summary_lane_error(rows)
    scenario_name = _last_nonempty(rows, "scenario_name")
    scenario_completion_s = _scenario_completion_time_s(rows) if scenario_name else None
    summary = {
        "rows": len(rows),
        "scenario_name": scenario_name,
        "scenario_kind": _last_nonempty(rows, "scenario_kind"),
        "scenario_status": _last_nonempty(rows, "scenario_status"),
        "scenario_success": _scenario_success(rows),
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
