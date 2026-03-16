import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _to_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_rows(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _prompted_rows(rows):
    return [row for row in rows if str(row.get("prompt_label", "")).strip()]


def _segment_rows(rows):
    segments = []
    current = []
    current_label = None
    for row in rows:
        label = str(row.get("prompt_label", "")).strip()
        if not label:
            continue
        if current and label != current_label:
            segments.append(current)
            current = []
        current.append(row)
        current_label = label
    if current:
        segments.append(current)
    return segments


def _elapsed(row):
    return _to_float(row.get("elapsed_s"))


def _prediction(row):
    return str(row.get("pred_label", "")).strip()


def _confidence(row):
    return _to_float(row.get("pred_conf"), default=0.0)


def _prompt_probability(row, label):
    return _to_float(row.get(f"fused_{label}"), default=None)


def _time_to_first_correct(segment):
    prompt = str(segment[0].get("prompt_label", "")).strip()
    start_t = _elapsed(segment[0])
    if start_t is None:
        return None
    for row in segment:
        if _prediction(row) == prompt:
            row_t = _elapsed(row)
            if row_t is not None:
                return float(row_t - start_t)
    return None


def _time_to_stable_correct(segment, stable_consecutive):
    prompt = str(segment[0].get("prompt_label", "")).strip()
    start_t = _elapsed(segment[0])
    if start_t is None:
        return None
    run = 0
    for row in segment:
        if _prediction(row) == prompt:
            run += 1
            if run >= stable_consecutive:
                row_t = _elapsed(row)
                if row_t is not None:
                    return float(row_t - start_t)
                return None
        else:
            run = 0
    return None


def _label_flip_counts(segment):
    preds = [_prediction(row) for row in segment]
    if len(preds) < 2:
        return 0, 0
    flips = 0
    transitions = 0
    for prev, cur in zip(preds[:-1], preds[1:]):
        transitions += 1
        if cur != prev:
            flips += 1
    return flips, transitions


def _segment_duration(segment):
    if not segment:
        return None
    start_t = _elapsed(segment[0])
    end_t = _elapsed(segment[-1])
    if start_t is None or end_t is None:
        return None
    return float(max(0.0, end_t - start_t))


def _carryover_stale_stats(segment, previous_segment):
    if previous_segment is None or not segment:
        return 0, 0.0
    prompt = str(segment[0].get("prompt_label", "")).strip()
    prev_pred = _prediction(previous_segment[-1])
    if not prev_pred or prev_pred == prompt:
        return 0, 0.0

    stale_rows = 0
    first_t = _elapsed(segment[0])
    stale_end_t = None
    for idx, row in enumerate(segment):
        if _prediction(row) == prev_pred and _prediction(row) != prompt:
            stale_rows += 1
            next_idx = idx + 1
            if next_idx < len(segment):
                stale_end_t = _elapsed(segment[next_idx])
            else:
                stale_end_t = _elapsed(row)
        else:
            break

    if stale_rows == 0 or first_t is None or stale_end_t is None:
        return 0, 0.0
    return stale_rows, float(max(0.0, stale_end_t - first_t))


def _summary(values):
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
        }
    arr = np.asarray(vals, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
    }


def _build_label_template():
    return {
        "rows": 0,
        "segments": 0,
        "correct_rows": 0,
        "pred_conf_sum": 0.0,
        "prompt_prob_sum": 0.0,
        "prompt_prob_count": 0,
        "time_to_first_correct_s": [],
        "time_to_stable_prediction_s": [],
        "label_flip_rates_per_s": [],
        "label_flip_fractions": [],
        "carryover_stale_rates": [],
        "carryover_stale_duration_s": [],
    }


def summarize_run(path: Path, stable_consecutive: int):
    rows = _load_rows(path)
    prompted = _prompted_rows(rows)
    segments = _segment_rows(prompted)

    total_flips = 0
    total_transitions = 0
    total_stale_rows = 0
    total_rows = 0
    total_duration = 0.0
    first_correct_values = []
    stable_values = []
    per_label = defaultdict(_build_label_template)
    segment_rows = []

    previous_segment = None
    for seg_idx, segment in enumerate(segments):
        prompt = str(segment[0].get("prompt_label", "")).strip()
        seg_rows = len(segment)
        seg_duration = _segment_duration(segment)
        ttfc = _time_to_first_correct(segment)
        tts = _time_to_stable_correct(segment, stable_consecutive)
        flips, transitions = _label_flip_counts(segment)
        stale_rows, stale_duration = _carryover_stale_stats(segment, previous_segment)

        flip_rate = None
        if seg_duration is not None and seg_duration > 0.0:
            flip_rate = float(flips / seg_duration)
        flip_fraction = float(flips / transitions) if transitions > 0 else 0.0
        stale_rate = float(stale_rows / seg_rows) if seg_rows > 0 else 0.0

        label_bucket = per_label[prompt]
        label_bucket["rows"] += seg_rows
        label_bucket["segments"] += 1
        label_bucket["correct_rows"] += sum(1 for row in segment if _prediction(row) == prompt)
        label_bucket["pred_conf_sum"] += sum(_confidence(row) for row in segment)
        for row in segment:
            prompt_prob = _prompt_probability(row, prompt)
            if prompt_prob is not None:
                label_bucket["prompt_prob_sum"] += prompt_prob
                label_bucket["prompt_prob_count"] += 1
        if ttfc is not None:
            label_bucket["time_to_first_correct_s"].append(ttfc)
            first_correct_values.append(ttfc)
        if tts is not None:
            label_bucket["time_to_stable_prediction_s"].append(tts)
            stable_values.append(tts)
        if flip_rate is not None:
            label_bucket["label_flip_rates_per_s"].append(flip_rate)
        label_bucket["label_flip_fractions"].append(flip_fraction)
        label_bucket["carryover_stale_rates"].append(stale_rate)
        label_bucket["carryover_stale_duration_s"].append(stale_duration)

        total_flips += flips
        total_transitions += transitions
        total_stale_rows += stale_rows
        total_rows += seg_rows
        if seg_duration is not None:
            total_duration += seg_duration

        segment_rows.append(
            {
                "segment_index": seg_idx,
                "prompt_label": prompt,
                "rows": seg_rows,
                "duration_s": seg_duration,
                "time_to_first_correct_s": ttfc,
                "time_to_stable_prediction_s": tts,
                "label_flips": flips,
                "label_flip_fraction": flip_fraction,
                "label_flip_rate_per_s": flip_rate,
                "carryover_stale_rows": stale_rows,
                "carryover_stale_rate": stale_rate,
                "carryover_stale_duration_s": stale_duration,
            }
        )
        previous_segment = segment

    overall_accuracy = float(
        sum(1 for row in prompted if _prediction(row) == str(row.get("prompt_label", "")).strip()) / len(prompted)
    ) if prompted else 0.0

    per_label_out = {}
    recalls = []
    for label, bucket in sorted(per_label.items()):
        rows_count = int(bucket["rows"])
        correct_rows = int(bucket["correct_rows"])
        recall = float(correct_rows / rows_count) if rows_count > 0 else 0.0
        recalls.append(recall)
        avg_pred_conf = float(bucket["pred_conf_sum"] / rows_count) if rows_count > 0 else 0.0
        avg_prompt_prob = (
            float(bucket["prompt_prob_sum"] / bucket["prompt_prob_count"])
            if bucket["prompt_prob_count"] > 0
            else None
        )
        per_label_out[label] = {
            "rows": rows_count,
            "segments": int(bucket["segments"]),
            "recall": recall,
            "avg_pred_conf": avg_pred_conf,
            "avg_prompt_prob": avg_prompt_prob,
            "time_to_first_correct_s": _summary(bucket["time_to_first_correct_s"]),
            "time_to_stable_prediction_s": _summary(bucket["time_to_stable_prediction_s"]),
            "label_flip_rate_per_s": _summary(bucket["label_flip_rates_per_s"]),
            "label_flip_fraction": _summary(bucket["label_flip_fractions"]),
            "carryover_stale_rate": _summary(bucket["carryover_stale_rates"]),
            "carryover_stale_duration_s": _summary(bucket["carryover_stale_duration_s"]),
        }

    return {
        "path": str(path),
        "rows_total": len(rows),
        "rows_prompted": len(prompted),
        "segments_prompted": len(segments),
        "stable_consecutive": int(stable_consecutive),
        "overall_accuracy": overall_accuracy,
        "balanced_accuracy": float(sum(recalls) / len(recalls)) if recalls else 0.0,
        "time_to_first_correct_s": _summary(first_correct_values),
        "time_to_stable_prediction_s": _summary(stable_values),
        "label_flip_fraction": _summary(
            [float(total_flips / total_transitions)] if total_transitions > 0 else []
        ),
        "label_flip_rate_per_s": _summary(
            [float(total_flips / total_duration)] if total_duration > 0 else []
        ),
        "carryover_stale_rate": _summary(
            [float(total_stale_rows / total_rows)] if total_rows > 0 else []
        ),
        "per_label": per_label_out,
        "segments": segment_rows,
    }


def _print_summary(summary):
    print(f"file: {summary['path']}")
    print(
        f"prompted_rows={summary['rows_prompted']} "
        f"segments={summary['segments_prompted']} "
        f"acc={summary['overall_accuracy']:.3f} "
        f"bal_acc={summary['balanced_accuracy']:.3f}"
    )
    ttfc = summary["time_to_first_correct_s"]
    tts = summary["time_to_stable_prediction_s"]
    flip_rate = summary["label_flip_rate_per_s"]
    stale_rate = summary["carryover_stale_rate"]
    print(
        "time_to_first_correct_s: "
        f"mean={ttfc['mean']} median={ttfc['median']} p90={ttfc['p90']}"
    )
    print(
        "time_to_stable_prediction_s: "
        f"mean={tts['mean']} median={tts['median']} p90={tts['p90']}"
    )
    print(
        "label_flip_rate_per_s: "
        f"mean={flip_rate['mean']} | carryover_stale_rate={stale_rate['mean']}"
    )
    print("\nPer-label:")
    for label, row in summary["per_label"].items():
        print(
            f"  {label:<12} "
            f"rows={row['rows']:<4} "
            f"seg={row['segments']:<3} "
            f"recall={row['recall']:.3f} "
            f"avg_conf={row['avg_pred_conf']:.3f} "
            f"ttfc_mean={row['time_to_first_correct_s']['mean']}"
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compute realtime behavior metrics from prompted realtime CSV runs."
    )
    parser.add_argument("--input", required=True, help="Realtime confidence-analysis CSV path.")
    parser.add_argument(
        "--stable-consecutive",
        type=int,
        default=2,
        help="Number of consecutive correct logged rows required for stable prediction.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional output JSON path.",
    )
    parser.add_argument(
        "--output-segments-csv",
        default="",
        help="Optional output CSV path for per-segment metrics.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.stable_consecutive < 1:
        raise ValueError("--stable-consecutive must be >= 1")

    summary = summarize_run(Path(args.input), stable_consecutive=args.stable_consecutive)
    _print_summary(summary)

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report to {out}")

    if args.output_segments_csv:
        out = Path(args.output_segments_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in summary["segments"] for key in row.keys()}) if summary["segments"] else []
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
                writer.writerows(summary["segments"])
        print(f"Saved segment CSV to {out}")


if __name__ == "__main__":
    main()
