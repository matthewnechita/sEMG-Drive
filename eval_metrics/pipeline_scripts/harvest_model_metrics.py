import argparse
import csv
import json
import pickle
import re
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval_metrics.config import ACTIVE_OFFLINE_MODEL_NAMES, CURRENT_METRICS_ROOT, MODELS_ROOT


CORE_METRIC_KEYS = [
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "worst_class_recall",
]


def _mean(values):
    values = [float(value) for value in values]
    if not values:
        return None
    return float(sum(values) / len(values))


def _extract_loso_metric_means(metadata):
    if not isinstance(metadata, dict):
        return {}
    zero_shot_loso = metadata.get("zero_shot_loso")
    if not isinstance(zero_shot_loso, dict):
        evaluation = metadata.get("evaluation")
        if isinstance(evaluation, dict):
            zero_shot_loso = evaluation.get("zero_shot_loso")
    if not isinstance(zero_shot_loso, dict):
        return {}

    folds = zero_shot_loso.get("folds")
    if not isinstance(folds, list):
        return {}

    out = {}
    valid_folds = [fold for fold in folds if isinstance(fold, dict) and isinstance(fold.get("metrics"), dict)]
    if valid_folds:
        out["loso_subject_count"] = len(valid_folds)
    for key in CORE_METRIC_KEYS:
        values = [
            fold["metrics"].get(key)
            for fold in valid_folds
            if fold["metrics"].get(key) not in (None, "")
        ]
        mean_value = _mean(values)
        if mean_value is not None:
            out[f"loso_{key}"] = mean_value
    return out


def _latest_rows(rows):
    latest = {}
    for row in rows:
        key = (
            str(row.get("bundle_scope") or "").strip().lower(),
            str(row.get("arm") or "").strip().lower(),
            str(row.get("subject") or row.get("target_subject") or "").strip().lower(),
            str(row.get("gesture_bucket") or "").strip().lower(),
        )
        created_at = str(row.get("created_at") or "").strip()
        current = latest.get(key)
        if current is None or created_at > str(current.get("created_at") or "").strip():
            latest[key] = row
    return list(latest.values())


def _normalize_label_map(raw_map):
    if not isinstance(raw_map, dict):
        return {}
    out = {}
    for key, value in raw_map.items():
        try:
            idx = int(key)
        except (TypeError, ValueError):
            continue
        out[idx] = str(value)
    return out


def _bundle_type_from_stream(stream_value):
    stream = str(stream_value or "").strip().lower()
    if "cross" in stream:
        return "cross_subject"
    if "per_subject" in stream or "single" in stream:
        return "per_subject"
    return "unknown"


def _gesture_labels(index_to_label, metadata):
    labels = [label for _, label in sorted(index_to_label.items())]
    if labels:
        return labels
    # Older bundles may only persist the label list in metadata.
    meta_labels = metadata.get("labels")
    if isinstance(meta_labels, (list, tuple)):
        return [str(label) for label in meta_labels]
    return []


def _gesture_bucket(labels, path: Path):
    name = path.stem.lower()
    match = re.search(r"(\d+)_gesture", name)
    if match:
        return f"{int(match.group(1))}_gesture"
    count = len(labels)
    if count == 3:
        return "3_gesture"
    if count == 4:
        return "4_gesture"
    if count == 5:
        return "5_gesture"
    if count == 6:
        return "6_gesture"
    return f"{count}_gesture" if count > 0 else "unknown"


def _arm_from_path(path: Path):
    parts = {part.lower() for part in path.parts}
    if "left" in parts:
        return "left"
    if "right" in parts:
        return "right"
    return ""


def _load_pt_bundle(path: Path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"Unexpected bundle type in {path}")
    return obj


def _load_pkl_bundle(path: Path):
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Unexpected bundle type in {path}")
    return obj


def load_bundle(path: Path):
    if path.suffix.lower() == ".pt":
        return _load_pt_bundle(path)
    if path.suffix.lower() == ".pkl":
        return _load_pkl_bundle(path)
    raise ValueError(f"Unsupported bundle suffix: {path.suffix}")


def summarize_bundle(path: Path):
    obj = load_bundle(path)
    metadata = obj.get("metadata", {}) if isinstance(obj, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    index_to_label = _normalize_label_map(obj.get("index_to_label") or metadata.get("index_to_label"))
    labels = _gesture_labels(index_to_label, metadata)
    metrics = metadata.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    stream = metadata.get("stream")

    # Keep the harvested row close to the report table schema so later scripts
    # can merge offline bundle metadata with latency and CARLA summaries.
    row = {
        "path": str(path),
        "filename": path.name,
        "suffix": path.suffix.lower(),
        "bundle_scope": _bundle_type_from_stream(stream),
        "stream": stream,
        "arm": _arm_from_path(path),
        "subject": metadata.get("subject") or metadata.get("target_subject") or "",
        "target_subject": metadata.get("target_subject") or "",
        "gesture_bucket": _gesture_bucket(labels, path),
        "gesture_count": len(labels),
        "labels": "|".join(labels),
        "included_gestures": "|".join(str(x) for x in (metadata.get("included_gestures") or [])),
        "channel_count": metadata.get("channel_count"),
        "window_size_samples": metadata.get("window_size_samples"),
        "window_step_samples": metadata.get("window_step_samples"),
        "created_at": metadata.get("created_at", ""),
    }
    for key in CORE_METRIC_KEYS:
        row[key] = metrics.get(key)
    row.update(_extract_loso_metric_means(metadata))
    return row


def _print_summary(rows):
    print(f"Found {len(rows)} bundle(s).")
    for row in rows:
        print(
            f"{row['filename']}: "
            f"scope={row['bundle_scope']} "
            f"gestures={row['gesture_bucket']} "
            f"subject={row['subject'] or '-'} "
            f"bal_acc={row.get('balanced_accuracy')}"
        )


def _write_csv(path: Path, rows):
    fieldnames = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Harvest stored offline metrics from saved model bundles."
    )
    parser.add_argument("--models-root", type=Path, default=MODELS_ROOT)
    parser.add_argument(
        "--model-name",
        action="append",
        default=[],
        help=(
            "Optional exact bundle filename or stem to keep. "
            "Repeatable. Examples: v6_4_gestures_2.pt or v6_4_gestures_2"
        ),
    )
    parser.add_argument(
        "--all-versions",
        action="store_true",
        help="Include historical bundle versions instead of keeping only the latest bundle per logical model slot.",
    )
    parser.add_argument(
        "--suffixes",
        nargs="+",
        default=[".pt", ".pkl"],
        help="Bundle suffixes to scan.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=CURRENT_METRICS_ROOT / "model_metrics.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=CURRENT_METRICS_ROOT / "model_metrics.json",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    root = Path(args.models_root)
    suffixes = {str(s).lower() for s in args.suffixes}
    requested_names = {str(name).strip().lower() for name in args.model_name if str(name).strip()}
    if not requested_names:
        requested_names = {
            str(name).strip().lower()
            for name in ACTIVE_OFFLINE_MODEL_NAMES
            if str(name).strip()
        }
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if requested_names:
            name = path.name.lower()
            stem = path.stem.lower()
            if name not in requested_names and stem not in requested_names:
                continue
        files.append(path)
    if not files:
        if requested_names:
            requested_text = ", ".join(sorted(requested_names))
            raise FileNotFoundError(
                f"No model bundles found under {root} matching requested model name(s): {requested_text}"
            )
        raise FileNotFoundError(f"No model bundles found under {root}")

    rows = [summarize_bundle(path) for path in files]
    if not args.all_versions:
        rows = _latest_rows(rows)
    _print_summary(rows)

    _write_csv(Path(args.output_csv), rows)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"Saved CSV to {args.output_csv}")
    print(f"Saved JSON to {args.output_json}")


if __name__ == "__main__":
    main()
