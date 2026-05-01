from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval_metrics.config import CURRENT_METRICS_ROOT, FIGURES_ROOT


SCENARIO_LABELS = {
    "lane_keep_eval": "Lane keep",
    "highway_overtake_eval": "Highway overtake",
}

METRIC_COLORS = {
    "lane_error": "#2563eb",
    "lane_invasion": "#dc2626",
    "speed": "#f97316",
    "speed_limit": "#6b7280",
    "steering": "#e11d48",
}


def _to_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_run_index(path: Path) -> list[dict[str, str]]:
    return _load_csv_rows(path)


def _load_drive_summary(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    full_route = payload.get("full_route")
    if not isinstance(full_route, dict):
        raise ValueError(f"Missing full_route summary in {path}")
    return full_route


def _pretty_scope(scope: str) -> str:
    text = str(scope or "").strip().lower()
    if "cross" in text:
        return "Cross-subject"
    if "per" in text:
        return "Per-subject"
    return str(scope or "").replace("_", " ").strip().title()


def _normalize_scope(scope: str) -> str:
    text = str(scope or "").strip().lower()
    if "cross" in text:
        return "cross_subject"
    if "per" in text:
        return "per_subject"
    return text.replace(" ", "_")


def _candidate_runs(run_index_rows: list[dict[str, str]], run_dir: str, model_scope: str | None = None) -> list[dict[str, object]]:
    candidates = []
    for row in run_index_rows:
        if str(row.get("run_dir") or "").strip() != run_dir:
            continue
        if model_scope:
            if _normalize_scope(row.get("model_scope", "")) != _normalize_scope(model_scope):
                continue
        drive_json = Path(str(row.get("drive_json") or "").strip())
        if not drive_json.exists():
            continue
        summary = _load_drive_summary(drive_json)
        status = str(summary.get("scenario_status") or "").strip().lower()
        mean_velocity = _to_float(summary.get("mean_velocity_mps"))
        if status == "waiting_start":
            continue
        if mean_velocity is not None and mean_velocity < 0.5:
            continue
        candidates.append(
            {
                "row": row,
                "summary": summary,
                "status": status,
                "latency_ok": _to_bool(row.get("latency_ok")),
                "stamp": str(row.get("stamp") or "").strip(),
                "model_scope": str(row.get("model_scope") or "").strip(),
            }
        )
    return candidates


def _metric_median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=float)))


def _metric_mad(values: list[float], median: float) -> float:
    mad = float(np.median(np.abs(np.asarray(values, dtype=float) - median)))
    return mad if mad > 1e-6 else 1.0


def _rank_map(values: list[float]) -> dict[float, int]:
    ordered = sorted(set(float(value) for value in values))
    return {value: idx for idx, value in enumerate(ordered)}


def _representative_run(candidates: list[dict[str, object]]) -> dict[str, object]:
    if not candidates:
        raise ValueError("No candidate runs were available for representative selection.")

    preferred = [
        item
        for item in candidates
        if item["status"] == "success" or item["summary"].get("scenario_success") is True
    ]
    if preferred:
        feature_keys = ["completion_time_s", "lane_error_rmse_m", "lane_invasions"]
        feature_ranks = {}
        for key in feature_keys:
            values = [
                _to_float(item["summary"].get(key))
                for item in preferred
                if _to_float(item["summary"].get(key)) is not None
            ]
            if values:
                feature_ranks[key] = _rank_map(values)

        def success_key(item: dict[str, object]) -> tuple[float, int, str]:
            total_rank = 0.0
            for key in feature_keys:
                value = _to_float(item["summary"].get(key))
                if value is None or key not in feature_ranks:
                    continue
                total_rank += feature_ranks[key][float(value)]
            latency_penalty = 0 if bool(item["latency_ok"]) else 1
            return total_rank, latency_penalty, str(item["stamp"])

        return sorted(preferred, key=success_key)[0]

    pool = [item for item in candidates if bool(item["latency_ok"])]
    if not pool:
        pool = candidates

    feature_keys = ["lane_error_rmse_m", "lane_invasions", "completion_time_s"]
    medians = {}
    scales = {}
    for key in feature_keys:
        values = [
            _to_float(item["summary"].get(key))
            for item in pool
            if _to_float(item["summary"].get(key)) is not None
        ]
        if not values:
            continue
        medians[key] = _metric_median(values)
        scales[key] = _metric_mad(values, medians[key])

    def score(item: dict[str, object]) -> tuple[float, int, str]:
        total = 0.0
        for key in feature_keys:
            value = _to_float(item["summary"].get(key))
            if value is None or key not in medians:
                continue
            total += abs(value - medians[key]) / scales[key]
        latency_penalty = 0 if bool(item["latency_ok"]) else 1
        return total, latency_penalty, str(item["stamp"])

    return sorted(pool, key=score)[0]


def _time_series(rows: list[dict[str, str]]) -> np.ndarray:
    preferred_columns = ["scenario_elapsed_s", "simulation_time", "control_apply_ts", "timestamp"]
    for column in preferred_columns:
        values = np.asarray([_to_float(row.get(column)) for row in rows], dtype=float)
        valid = np.isfinite(values)
        if int(np.sum(valid)) < 3:
            continue
        series = values[valid]
        if float(np.max(series) - np.min(series)) <= 0.0:
            continue
        if np.any(np.diff(series) < -1e-6):
            continue
        return values - float(series[0])
    return np.arange(len(rows), dtype=float)


def _numeric_array(rows: list[dict[str, str]], column: str) -> np.ndarray:
    return np.asarray([_to_float(row.get(column)) for row in rows], dtype=float)


def _bool_array(rows: list[dict[str, str]], column: str) -> np.ndarray:
    return np.asarray([_to_bool(row.get(column)) for row in rows], dtype=bool)


def _plot_limits(values_by_run: list[np.ndarray], *, centered: bool = False, floor_zero: bool = False) -> tuple[float, float]:
    finite_values = []
    for values in values_by_run:
        arr = np.asarray(values, dtype=float)
        finite_values.extend(arr[np.isfinite(arr)].tolist())
    if not finite_values:
        return (-1.0, 1.0) if centered else (0.0, 1.0)

    finite_arr = np.asarray(finite_values, dtype=float)
    lo = float(np.min(finite_arr))
    hi = float(np.max(finite_arr))
    if centered:
        bound = max(abs(lo), abs(hi))
        bound = max(bound * 1.08, 1e-3)
        return -bound, bound
    if floor_zero:
        hi = max(hi * 1.08, 1.0)
        return 0.0, hi
    pad = max((hi - lo) * 0.08, 1e-3)
    return lo - pad, hi + pad


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot representative lane-keep and highway-overtake drive traces from staged CARLA logs."
    )
    parser.add_argument(
        "--run-index",
        type=Path,
        default=CURRENT_METRICS_ROOT / "carla_run_index.csv",
        help="Run index CSV produced by gather_current_metrics.py.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=FIGURES_ROOT / "drive_trace_representative.png",
        help="Path to save the output PNG.",
    )
    parser.add_argument(
        "--title",
        default="Representative Drive Traces",
        help="Optional custom title.",
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    run_index_rows = _load_run_index(Path(args.run_index))

    scenario_order = ["lane_keep_eval", "highway_overtake_eval"]
    available_scopes = [
        scope for scope in ["Cross-subject", "Per-subject"]
        if any(_normalize_scope(row.get("model_scope", "")) == _normalize_scope(scope) for row in run_index_rows)
    ]
    if not available_scopes:
        available_scopes = [""]

    selected_runs = []
    for model_scope in available_scopes:
        for run_dir in scenario_order:
            candidates = _candidate_runs(run_index_rows, run_dir, model_scope if model_scope else None)
            if not candidates:
                continue
            selected_runs.append((model_scope, run_dir, _representative_run(candidates)))

    if not selected_runs:
        raise ValueError("No representative drive runs could be selected.")

    plotted_runs = []
    for model_scope, run_dir, selected in selected_runs:
        carla_log = Path(str(selected["row"].get("carla_log") or "").strip())
        rows = _load_csv_rows(carla_log)
        time_s = _time_series(rows)
        lane_error = _numeric_array(rows, "lane_error_m")
        speed = _numeric_array(rows, "speed_mps")
        speed_limit = _numeric_array(rows, "speed_limit_mps")
        steering = _numeric_array(rows, "steering_angle_rad")
        invasions = _bool_array(rows, "lane_invasion_event")
        plotted_runs.append(
            {
                "run_dir": run_dir,
                "model_scope": model_scope,
                "stamp": str(selected["stamp"]),
                "status": str(selected["status"]),
                "summary": selected["summary"],
                "time_s": time_s,
                "lane_error": lane_error,
                "speed": speed,
                "speed_limit": speed_limit,
                "steering": steering,
                "invasions": invasions,
            }
        )

    lane_ylim = _plot_limits([item["lane_error"] for item in plotted_runs], centered=True)
    speed_ylim = _plot_limits(
        [np.concatenate([item["speed"], item["speed_limit"]]) for item in plotted_runs],
        floor_zero=True,
    )
    steer_ylim = _plot_limits([item["steering"] for item in plotted_runs], centered=True)

    from matplotlib import pyplot as plt

    ncols = len(plotted_runs)
    fig, axes = plt.subplots(
        3,
        ncols,
        figsize=(6.4 * ncols, 8.3),
        sharex="col",
        constrained_layout=True,
    )
    axes_array = np.asarray(axes, dtype=object)
    if axes_array.ndim == 1:
        axes_array = axes_array.reshape(3, 1)

    for col_idx, item in enumerate(plotted_runs):
        time_s = np.asarray(item["time_s"], dtype=float)
        lane_error = np.asarray(item["lane_error"], dtype=float)
        speed = np.asarray(item["speed"], dtype=float)
        speed_limit = np.asarray(item["speed_limit"], dtype=float)
        steering = np.asarray(item["steering"], dtype=float)
        invasion_times = time_s[np.asarray(item["invasions"], dtype=bool) & np.isfinite(time_s)]

        lane_ax = axes_array[0, col_idx]
        lane_ax.plot(time_s, lane_error, color=METRIC_COLORS["lane_error"], linewidth=1.4, label="Lane error")
        lane_ax.axhline(0.0, color="#94a3b8", linewidth=0.9, linestyle="--")
        for idx, event_time in enumerate(invasion_times):
            lane_ax.axvline(
                float(event_time),
                color=METRIC_COLORS["lane_invasion"],
                linewidth=0.7,
                alpha=0.35,
                label="Lane invasion event" if idx == 0 else None,
            )
        lane_ax.set_ylim(*lane_ylim)
        lane_ax.set_ylabel("Lane error (m)")
        lane_ax.grid(alpha=0.18)
        lane_ax.spines["top"].set_visible(False)
        lane_ax.spines["right"].set_visible(False)

        completion = _to_float(item["summary"].get("completion_time_s"))
        completion_text = f"{completion:.1f} s" if completion is not None else "n/a"
        title_bits = []
        if str(item.get("model_scope") or "").strip():
            title_bits.append(_pretty_scope(str(item["model_scope"])))
        title_bits.append(SCENARIO_LABELS.get(item["run_dir"], item["run_dir"]))
        title_bits.append(f"Example run | completion {completion_text}")
        lane_ax.set_title("\n".join(title_bits), fontsize=11.5, fontweight="bold")

        speed_ax = axes_array[1, col_idx]
        speed_ax.plot(time_s, speed, color=METRIC_COLORS["speed"], linewidth=1.4, label="Vehicle speed")
        if np.any(np.isfinite(speed_limit)):
            speed_ax.plot(
                time_s,
                speed_limit,
                color=METRIC_COLORS["speed_limit"],
                linewidth=1.2,
                linestyle="--",
                label="Speed limit",
            )
        speed_ax.set_ylim(*speed_ylim)
        speed_ax.set_ylabel("Speed (m/s)")
        speed_ax.grid(alpha=0.18)
        speed_ax.spines["top"].set_visible(False)
        speed_ax.spines["right"].set_visible(False)

        steer_ax = axes_array[2, col_idx]
        steer_ax.plot(time_s, steering, color=METRIC_COLORS["steering"], linewidth=1.35, label="Steering angle")
        steer_ax.axhline(0.0, color="#94a3b8", linewidth=0.9, linestyle="--")
        steer_ax.set_ylim(*steer_ylim)
        steer_ax.set_ylabel("Steering angle (rad)")
        steer_ax.set_xlabel("Time (s)")
        steer_ax.grid(alpha=0.18)
        steer_ax.spines["top"].set_visible(False)
        steer_ax.spines["right"].set_visible(False)

        if col_idx == ncols - 1:
            lane_ax.legend(loc="upper right", frameon=False)
            speed_ax.legend(loc="upper right", frameon=False)
            steer_ax.legend(loc="upper right", frameon=False)

    fig.suptitle(
        args.title.strip() or "Representative Drive Traces",
        fontsize=15,
        fontweight="bold",
    )

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=220, bbox_inches="tight")
    print(f"Saved figure to {args.output_png}")


if __name__ == "__main__":
    main()
