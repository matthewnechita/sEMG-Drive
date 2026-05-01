from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval_metrics.config import CURRENT_METRICS_ROOT, EVAL_ROOT


LATENCY_P95_MAX_MS = 200.0
SCORE_MAX = 5

# These are descriptive engineering criteria for stable simulator control under
# the two maintained scenarios. They are not roadway safety standards.
SCENARIO_THRESHOLDS = {
    "highway_overtake": {
        "lane_offset_mean_m_max": 0.60,
        "lane_error_rmse_m_max": 0.90,
        "steering_entropy_max": 0.25,
    },
    "lane_keep_5min": {
        "lane_offset_mean_m_max": 0.75,
        "lane_error_rmse_m_max": 1.00,
        "steering_entropy_max": 0.30,
    },
}

RUN_COLUMNS = [
    "row_id",
    "model_scope",
    "scenario_name",
    "scenario_label",
    "subject",
    "stamp",
    "latency_p95_ms",
    "lane_offset_mean_m",
    "lane_error_rmse_m",
    "steering_entropy",
    "mean_velocity_deviation_mps",
    "lane_invasions",
    "completion_time_s",
    "lane_invasions_per_min",
    "criterion_completed_run",
    "criterion_latency_p95",
    "criterion_lane_offset",
    "criterion_lane_rmse",
    "criterion_steering_entropy",
    "control_stability_score",
    "control_stability_score_max",
]

SUMMARY_COLUMNS = [
    "Model",
    "Scenario",
    "Runs",
    "Control stability score (0-5)",
    "Score SD",
    "Runs meeting all criteria (%)",
    "Runs meeting >=4 criteria (%)",
    "Mean lane invasions",
    "Mean lane invasions per min",
]


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path_str: str) -> dict:
    path = Path(str(path_str or "").strip())
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _to_float(value) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _mean_sd(values) -> tuple[float | None, float | None]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None, None
    if len(vals) == 1:
        return float(vals[0]), 0.0
    mean = float(sum(vals) / len(vals))
    variance = float(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
    return mean, math.sqrt(variance)


def _round_or_blank(value, digits: int):
    number = _to_float(value)
    if number is None:
        return ""
    return round(number, digits)


def _pretty_scope(text: str) -> str:
    value = str(text or "").strip().lower()
    if "cross" in value:
        return "Cross-subject"
    if "per" in value:
        return "Per-subject"
    return str(text or "").replace("_", " ").strip().title()


def _pretty_scenario(text: str) -> str:
    value = str(text or "").strip().lower()
    if value == "highway_overtake":
        return "Highway overtake"
    if value == "lane_keep_5min":
        return "Lane keep"
    return str(text or "").replace("_", " ").strip().title()


def _criterion_int(flag: bool) -> int:
    return 1 if bool(flag) else 0


def _build_run_rows(selection_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in selection_rows:
        scenario_name = str(row.get("scenario_name", "")).strip()
        thresholds = SCENARIO_THRESHOLDS.get(scenario_name)
        if thresholds is None:
            raise ValueError(f"No control-stability thresholds configured for scenario {scenario_name!r}")

        latency_summary = _read_json(str(row.get("latency_json", "")))
        drive_summary = _read_json(str(row.get("drive_json", ""))).get("full_route", {})

        latency_p95_ms = _to_float(
            latency_summary.get("end_to_end_latency_ms", {}).get("p95_ms")
        )
        lane_offset_mean_m = _to_float(drive_summary.get("lane_offset_mean_m"))
        lane_error_rmse_m = _to_float(drive_summary.get("lane_error_rmse_m"))
        steering_entropy = _to_float(drive_summary.get("steering_entropy"))
        mean_velocity_deviation_mps = _to_float(drive_summary.get("mean_velocity_deviation_mps"))
        lane_invasions = _to_float(drive_summary.get("lane_invasions"))
        completion_time_s = _to_float(drive_summary.get("completion_time_s"))

        lane_invasions_per_min = None
        if lane_invasions is not None and completion_time_s is not None and completion_time_s > 0.0:
            lane_invasions_per_min = float(lane_invasions / (completion_time_s / 60.0))

        # The curated research set excludes incomplete starts and duplicate
        # attempts, so all retained runs count as completed scenario trials.
        criterion_completed_run = True
        criterion_latency_p95 = latency_p95_ms is not None and latency_p95_ms <= LATENCY_P95_MAX_MS
        criterion_lane_offset = (
            lane_offset_mean_m is not None
            and lane_offset_mean_m <= float(thresholds["lane_offset_mean_m_max"])
        )
        criterion_lane_rmse = (
            lane_error_rmse_m is not None
            and lane_error_rmse_m <= float(thresholds["lane_error_rmse_m_max"])
        )
        criterion_steering_entropy = (
            steering_entropy is not None
            and steering_entropy <= float(thresholds["steering_entropy_max"])
        )

        control_stability_score = sum(
            [
                _criterion_int(criterion_completed_run),
                _criterion_int(criterion_latency_p95),
                _criterion_int(criterion_lane_offset),
                _criterion_int(criterion_lane_rmse),
                _criterion_int(criterion_steering_entropy),
            ]
        )

        out.append(
            {
                "row_id": row.get("row_id", ""),
                "model_scope": row.get("model_scope", ""),
                "scenario_name": scenario_name,
                "scenario_label": row.get("scenario_label", ""),
                "subject": row.get("subject", ""),
                "stamp": row.get("stamp", ""),
                "latency_p95_ms": latency_p95_ms,
                "lane_offset_mean_m": lane_offset_mean_m,
                "lane_error_rmse_m": lane_error_rmse_m,
                "steering_entropy": steering_entropy,
                "mean_velocity_deviation_mps": mean_velocity_deviation_mps,
                "lane_invasions": lane_invasions,
                "completion_time_s": completion_time_s,
                "lane_invasions_per_min": lane_invasions_per_min,
                "criterion_completed_run": criterion_completed_run,
                "criterion_latency_p95": criterion_latency_p95,
                "criterion_lane_offset": criterion_lane_offset,
                "criterion_lane_rmse": criterion_lane_rmse,
                "criterion_steering_entropy": criterion_steering_entropy,
                "control_stability_score": control_stability_score,
                "control_stability_score_max": SCORE_MAX,
            }
        )
    return out


def _build_summary_rows(run_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in run_rows:
        key = (str(row.get("model_scope", "")), str(row.get("scenario_name", "")))
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, object]] = []
    for key in sorted(
        grouped,
        key=lambda item: (
            0 if "cross" in item[0].lower() else 1,
            0 if item[1] == "highway_overtake" else 1,
        ),
    ):
        rows = grouped[key]
        score_mean, score_sd = _mean_sd(row.get("control_stability_score") for row in rows)
        pass_all = sum(1 for row in rows if _to_float(row.get("control_stability_score")) == SCORE_MAX)
        pass_ge4 = sum(
            1 for row in rows if (_to_float(row.get("control_stability_score")) or 0.0) >= 4.0
        )
        row_count = len(rows)
        out.append(
            {
                "Model": _pretty_scope(key[0]),
                "Scenario": _pretty_scenario(key[1]),
                "Runs": row_count,
                "Control stability score (0-5)": _round_or_blank(score_mean, 2),
                "Score SD": _round_or_blank(score_sd, 2),
                "Runs meeting all criteria (%)": _round_or_blank((100.0 * pass_all / row_count), 1),
                "Runs meeting >=4 criteria (%)": _round_or_blank((100.0 * pass_ge4 / row_count), 1),
                "Mean lane invasions": _round_or_blank(_mean(row.get("lane_invasions") for row in rows), 1),
                "Mean lane invasions per min": _round_or_blank(
                    _mean(row.get("lane_invasions_per_min") for row in rows), 2
                ),
            }
        )
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build run-level and summary control-stability tables for the curated research runtime set."
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=CURRENT_METRICS_ROOT / "research_runtime_selection.csv",
        help="Curated research runtime selection CSV.",
    )
    parser.add_argument(
        "--output-runs",
        type=Path,
        default=CURRENT_METRICS_ROOT / "research_control_stability_runs.csv",
        help="Run-level control-stability output CSV.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=CURRENT_METRICS_ROOT / "research_control_stability_summary.csv",
        help="Scenario/model-scope summary CSV.",
    )
    parser.add_argument(
        "--output-final",
        type=Path,
        default=EVAL_ROOT / "final_research_control_stability_table.csv",
        help="Top-level paper-facing control-stability CSV.",
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    selection_rows = _load_csv_rows(Path(args.selection))
    run_rows = _build_run_rows(selection_rows)
    summary_rows = _build_summary_rows(run_rows)

    _write_csv(Path(args.output_runs), run_rows, RUN_COLUMNS)
    _write_csv(Path(args.output_summary), summary_rows, SUMMARY_COLUMNS)
    _write_csv(Path(args.output_final), summary_rows, SUMMARY_COLUMNS)

    print(f"Saved run-level control-stability rows to {args.output_runs}")
    print(f"Saved summary control-stability rows to {args.output_summary}")
    print(f"Saved final research control-stability table to {args.output_final}")


if __name__ == "__main__":
    main()
