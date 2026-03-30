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


def _index_rows(rows, key):
    out = {}
    for row in rows:
        value = row.get(key)
        if value is None or value == "":
            continue
        out[str(value)] = row
    return out


def _percentile(values, q):
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _summary(values):
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {
            "count": 0,
            "mean_ms": None,
            "p95_ms": None,
        }
    arr = np.asarray(vals, dtype=float)
    return {
        "count": int(arr.size),
        "mean_ms": float(arr.mean()),
        "p95_ms": float(np.percentile(arr, 95)),
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Join realtime and CARLA logs and compute latency summaries."
    )
    parser.add_argument("--realtime-log", type=Path, required=True)
    parser.add_argument("--carla-log", type=Path, required=True)
    parser.add_argument(
        "--join-key",
        default="prediction_seq",
        help="Shared key column between the two logs.",
    )
    parser.add_argument(
        "--label-column",
        default="pred_label",
        help="Prediction label column in the realtime log.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("eval_metrics") / "out" / "latency_summary.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("eval_metrics") / "out" / "latency_joined.csv",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    realtime_rows = _load_rows(Path(args.realtime_log))
    carla_rows = _load_rows(Path(args.carla_log))
    realtime_index = _index_rows(realtime_rows, args.join_key)
    carla_index = _index_rows(carla_rows, args.join_key)

    # The join key is a per-prediction identifier emitted by both processes, so
    # we do not depend on wall-clock synchronization between the two logs.
    shared_keys = sorted(set(realtime_index) & set(carla_index))
    if not shared_keys:
        raise ValueError(
            f"No shared rows found using join key '{args.join_key}'. "
            "Add a shared prediction identifier to both logs."
        )

    joined = []
    end_to_end_latency = []

    for key in shared_keys:
        rt_row = realtime_index[key]
        carla_row = carla_index[key]
        window_end_ts = _to_float(rt_row.get("window_end_ts"))
        control_apply_ts = _to_float(carla_row.get("control_apply_ts"))
        label = str(rt_row.get(args.label_column, "")).strip()
        # Window end to control apply is the maintained end-to-end latency path
        # used in the current CARLA reporting workflow.
        e2e_ms = None if window_end_ts is None or control_apply_ts is None else (control_apply_ts - window_end_ts) * 1000.0

        if e2e_ms is not None:
            end_to_end_latency.append(e2e_ms)

        joined.append(
            {
                args.join_key: key,
                "pred_label": label,
                "end_to_end_latency_ms": e2e_ms,
            }
        )

    payload = {
        "join_key": args.join_key,
        "rows_joined": len(joined),
        "end_to_end_latency_ms": _summary(end_to_end_latency),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fieldnames = sorted({key for row in joined for key in row.keys()})
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(joined)

    print(json.dumps(payload, indent=2))
    print(f"Saved JSON to {args.output_json}")
    print(f"Saved CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
