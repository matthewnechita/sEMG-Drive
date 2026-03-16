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
            "median_ms": None,
            "p90_ms": None,
            "p95_ms": None,
            "max_ms": None,
        }
    arr = np.asarray(vals, dtype=float)
    return {
        "count": int(arr.size),
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(arr.max()),
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

    shared_keys = sorted(set(realtime_index) & set(carla_index))
    if not shared_keys:
        raise ValueError(
            f"No shared rows found using join key '{args.join_key}'. "
            "Add a shared prediction identifier to both logs."
        )

    joined = []
    classifier_latency = []
    publish_latency = []
    control_latency = []
    end_to_end_latency = []
    per_label = {}

    for key in shared_keys:
        rt_row = realtime_index[key]
        carla_row = carla_index[key]
        window_end_ts = _to_float(rt_row.get("window_end_ts"))
        prediction_ts = _to_float(rt_row.get("prediction_ts"))
        publish_ts = _to_float(rt_row.get("publish_ts"))
        control_apply_ts = _to_float(carla_row.get("control_apply_ts"))
        label = str(rt_row.get(args.label_column, "")).strip()

        cls_ms = None if window_end_ts is None or prediction_ts is None else (prediction_ts - window_end_ts) * 1000.0
        pub_ms = None if prediction_ts is None or publish_ts is None else (publish_ts - prediction_ts) * 1000.0
        ctl_ms = None if publish_ts is None or control_apply_ts is None else (control_apply_ts - publish_ts) * 1000.0
        e2e_ms = None if window_end_ts is None or control_apply_ts is None else (control_apply_ts - window_end_ts) * 1000.0

        if cls_ms is not None:
            classifier_latency.append(cls_ms)
        if pub_ms is not None:
            publish_latency.append(pub_ms)
        if ctl_ms is not None:
            control_latency.append(ctl_ms)
        if e2e_ms is not None:
            end_to_end_latency.append(e2e_ms)

        if label:
            bucket = per_label.setdefault(label, [])
            if e2e_ms is not None:
                bucket.append(e2e_ms)

        joined.append(
            {
                args.join_key: key,
                "pred_label": label,
                "classifier_latency_ms": cls_ms,
                "publish_latency_ms": pub_ms,
                "control_latency_ms": ctl_ms,
                "end_to_end_latency_ms": e2e_ms,
            }
        )

    payload = {
        "join_key": args.join_key,
        "rows_joined": len(joined),
        "classifier_latency_ms": _summary(classifier_latency),
        "publish_latency_ms": _summary(publish_latency),
        "control_latency_ms": _summary(control_latency),
        "end_to_end_latency_ms": _summary(end_to_end_latency),
        "per_label_end_to_end_ms": {
            label: _summary(values) for label, values in sorted(per_label.items())
        },
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
