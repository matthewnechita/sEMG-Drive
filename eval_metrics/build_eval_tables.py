import argparse
import csv
import json
import re
from pathlib import Path


TEMPLATE_COLUMNS = [
    "row_id",
    "deliverable",
    "task",
    "model_scope",
    "condition",
    "arm",
    "subject",
    "status",
    "comparison_group",
    "model_path",
    "model_filename",
    "realtime_behavior_json",
    "latency_json",
    "drive_metrics_json",
    "notes",
]


TEMPLATE_ROWS = [
    {
        "row_id": "CR_3_PER",
        "deliverable": "Capstone final report",
        "task": "3-gesture",
        "model_scope": "Per-subject",
        "condition": "current_cnn_pipeline",
        "arm": "",
        "subject": "",
        "status": "Pending",
        "comparison_group": "",
        "model_path": "",
        "model_filename": "",
        "realtime_behavior_json": "",
        "latency_json": "",
        "drive_metrics_json": "",
        "notes": "",
    },
    {
        "row_id": "CR_3_CROSS",
        "deliverable": "Capstone final report",
        "task": "3-gesture",
        "model_scope": "Cross-subject",
        "condition": "current_cnn_pipeline",
        "arm": "",
        "subject": "",
        "status": "Pending",
        "comparison_group": "",
        "model_path": "",
        "model_filename": "",
        "realtime_behavior_json": "",
        "latency_json": "",
        "drive_metrics_json": "",
        "notes": "",
    },
    {
        "row_id": "CR_5_PER",
        "deliverable": "Capstone final report",
        "task": "5-gesture",
        "model_scope": "Per-subject",
        "condition": "current_cnn_pipeline",
        "arm": "",
        "subject": "",
        "status": "Pending",
        "comparison_group": "",
        "model_path": "",
        "model_filename": "",
        "realtime_behavior_json": "",
        "latency_json": "",
        "drive_metrics_json": "",
        "notes": "",
    },
    {
        "row_id": "CR_5_CROSS",
        "deliverable": "Capstone final report",
        "task": "5-gesture",
        "model_scope": "Cross-subject",
        "condition": "current_cnn_pipeline",
        "arm": "",
        "subject": "",
        "status": "Pending",
        "comparison_group": "",
        "model_path": "",
        "model_filename": "",
        "realtime_behavior_json": "",
        "latency_json": "",
        "drive_metrics_json": "",
        "notes": "",
    },
    {
        "row_id": "RP_3_PER",
        "deliverable": "Research paper",
        "task": "3-gesture",
        "model_scope": "Per-subject",
        "condition": "current_cnn_pipeline",
        "arm": "",
        "subject": "",
        "status": "Pending",
        "comparison_group": "Prior-paper comparison, no eye-tracker metrics",
        "model_path": "",
        "model_filename": "",
        "realtime_behavior_json": "",
        "latency_json": "",
        "drive_metrics_json": "",
        "notes": "No eye-tracker metrics",
    },
    {
        "row_id": "RP_3_CROSS",
        "deliverable": "Research paper",
        "task": "3-gesture",
        "model_scope": "Cross-subject",
        "condition": "current_cnn_pipeline",
        "arm": "",
        "subject": "",
        "status": "Pending",
        "comparison_group": "Prior-paper comparison, no eye-tracker metrics",
        "model_path": "",
        "model_filename": "",
        "realtime_behavior_json": "",
        "latency_json": "",
        "drive_metrics_json": "",
        "notes": "No eye-tracker metrics",
    },
    {
        "row_id": "RP_5_PER",
        "deliverable": "Research paper",
        "task": "5-gesture",
        "model_scope": "Per-subject",
        "condition": "current_cnn_pipeline",
        "arm": "",
        "subject": "",
        "status": "Pending",
        "comparison_group": "Prior-paper comparison, no eye-tracker metrics",
        "model_path": "",
        "model_filename": "",
        "realtime_behavior_json": "",
        "latency_json": "",
        "drive_metrics_json": "",
        "notes": "No eye-tracker metrics",
    },
    {
        "row_id": "RP_5_CROSS",
        "deliverable": "Research paper",
        "task": "5-gesture",
        "model_scope": "Cross-subject",
        "condition": "current_cnn_pipeline",
        "arm": "",
        "subject": "",
        "status": "Pending",
        "comparison_group": "Prior-paper comparison, no eye-tracker metrics",
        "model_path": "",
        "model_filename": "",
        "realtime_behavior_json": "",
        "latency_json": "",
        "drive_metrics_json": "",
        "notes": "No eye-tracker metrics",
    },
]


BASE_COLUMNS = [
    "table_row_id",
    "row_id",
    "deliverable",
    "deliverable_bucket",
    "task",
    "model_scope",
    "condition",
    "subject",
    "arm",
    "status",
    "comparison_group",
    "notes",
]


OFFLINE_COLUMNS = [
    "source_model_filename",
    "source_model_path",
    "offline_bundle_scope",
    "offline_gesture_bucket",
    "offline_labels",
    "offline_test_accuracy",
    "offline_balanced_accuracy",
    "offline_macro_f1",
    "offline_macro_precision",
    "offline_macro_recall",
    "offline_weighted_f1",
    "offline_weighted_precision",
    "offline_weighted_recall",
    "offline_worst_class_recall",
    "offline_worst_class_recall_label",
    "offline_max_precision_recall_gap",
    "offline_max_precision_recall_gap_label",
    "offline_channel_count",
    "offline_window_size_samples",
    "offline_window_step_samples",
    "offline_created_at",
]


SYSTEM_COLUMNS = [
    "realtime_behavior_json",
    "latency_json",
    "drive_metrics_json",
    "rt_rows_prompted",
    "rt_segments_prompted",
    "rt_overall_accuracy",
    "rt_balanced_accuracy",
    "rt_ttfc_mean_s",
    "rt_ttfc_median_s",
    "rt_ttfc_p90_s",
    "rt_tts_mean_s",
    "rt_tts_median_s",
    "rt_tts_p90_s",
    "rt_flip_rate_mean_per_s",
    "rt_flip_fraction_mean",
    "rt_stale_rate_mean",
    "lat_rows_joined",
    "lat_classifier_mean_ms",
    "lat_classifier_p95_ms",
    "lat_publish_mean_ms",
    "lat_control_mean_ms",
    "lat_control_p95_ms",
    "lat_e2e_mean_ms",
    "lat_e2e_median_ms",
    "lat_e2e_p95_ms",
    "lat_e2e_max_ms",
    "drive_rows",
    "drive_lane_error_mean_m",
    "drive_lane_error_rmse_m",
    "drive_lane_invasions",
    "drive_collisions",
    "drive_completion_time_s",
    "drive_steering_smoothness",
    "drive_command_success_rate",
]


def _slugify(text):
    value = str(text or "").strip().lower()
    value = value.replace("&", "and")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def _normalize_task(text):
    slug = _slugify(text)
    if slug.startswith("3_gesture"):
        return "3_gesture"
    if slug.startswith("5_gesture"):
        return "5_gesture"
    if slug.startswith("6_gesture"):
        return "6_gesture"
    return slug


def _normalize_scope(text):
    slug = _slugify(text)
    if "cross" in slug:
        return "cross_subject"
    if "per" in slug or "single" in slug:
        return "per_subject"
    return slug


def _deliverable_bucket(text):
    slug = _slugify(text)
    if "paper" in slug:
        return "research_paper"
    if "capstone" in slug or "report" in slug or "roadmap" in slug:
        return "capstone_report"
    return slug or "unknown"


def _load_csv_rows(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_markdown(path: Path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("_No rows_\n", encoding="utf-8")
        return
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        vals = []
        for column in columns:
            value = row.get(column, "")
            text = "" if value is None else str(value)
            vals.append(text.replace("\n", " ").replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json(path_str):
    path_str = str(path_str or "").strip()
    if not path_str:
        return {}
    path = Path(path_str)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _nested_get(obj, *keys):
    cur = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _flatten_realtime(summary):
    if not isinstance(summary, dict) or not summary:
        return {}
    return {
        "rt_rows_prompted": summary.get("rows_prompted"),
        "rt_segments_prompted": summary.get("segments_prompted"),
        "rt_overall_accuracy": summary.get("overall_accuracy"),
        "rt_balanced_accuracy": summary.get("balanced_accuracy"),
        "rt_ttfc_mean_s": _nested_get(summary, "time_to_first_correct_s", "mean"),
        "rt_ttfc_median_s": _nested_get(summary, "time_to_first_correct_s", "median"),
        "rt_ttfc_p90_s": _nested_get(summary, "time_to_first_correct_s", "p90"),
        "rt_tts_mean_s": _nested_get(summary, "time_to_stable_prediction_s", "mean"),
        "rt_tts_median_s": _nested_get(summary, "time_to_stable_prediction_s", "median"),
        "rt_tts_p90_s": _nested_get(summary, "time_to_stable_prediction_s", "p90"),
        "rt_flip_rate_mean_per_s": _nested_get(summary, "label_flip_rate_per_s", "mean"),
        "rt_flip_fraction_mean": _nested_get(summary, "label_flip_fraction", "mean"),
        "rt_stale_rate_mean": _nested_get(summary, "carryover_stale_rate", "mean"),
    }


def _flatten_latency(summary):
    if not isinstance(summary, dict) or not summary:
        return {}
    return {
        "lat_rows_joined": summary.get("rows_joined"),
        "lat_classifier_mean_ms": _nested_get(summary, "classifier_latency_ms", "mean_ms"),
        "lat_classifier_p95_ms": _nested_get(summary, "classifier_latency_ms", "p95_ms"),
        "lat_publish_mean_ms": _nested_get(summary, "publish_latency_ms", "mean_ms"),
        "lat_control_mean_ms": _nested_get(summary, "control_latency_ms", "mean_ms"),
        "lat_control_p95_ms": _nested_get(summary, "control_latency_ms", "p95_ms"),
        "lat_e2e_mean_ms": _nested_get(summary, "end_to_end_latency_ms", "mean_ms"),
        "lat_e2e_median_ms": _nested_get(summary, "end_to_end_latency_ms", "median_ms"),
        "lat_e2e_p95_ms": _nested_get(summary, "end_to_end_latency_ms", "p95_ms"),
        "lat_e2e_max_ms": _nested_get(summary, "end_to_end_latency_ms", "max_ms"),
    }


def _flatten_drive(summary):
    if not isinstance(summary, dict) or not summary:
        return {}
    full = summary.get("full_route", {})
    out = {
        "drive_rows": full.get("rows"),
        "drive_lane_error_mean_m": full.get("lane_error_mean_m"),
        "drive_lane_error_rmse_m": full.get("lane_error_rmse_m"),
        "drive_lane_invasions": full.get("lane_invasions"),
        "drive_collisions": full.get("collisions"),
        "drive_completion_time_s": full.get("completion_time_s"),
        "drive_steering_smoothness": full.get("steering_smoothness"),
        "drive_command_success_rate": full.get("command_success_rate"),
    }
    segments = summary.get("segments", {})
    if isinstance(segments, dict):
        for segment_name, segment_metrics in sorted(segments.items()):
            segment_slug = _slugify(segment_name)
            if not isinstance(segment_metrics, dict) or not segment_slug:
                continue
            for key, value in sorted(segment_metrics.items()):
                out[f"drive_{segment_slug}_{_slugify(key)}"] = value
    return out


def _offline_from_model_row(row):
    if not row:
        return {}
    return {
        "source_model_filename": row.get("filename", ""),
        "source_model_path": row.get("path", ""),
        "offline_bundle_scope": row.get("bundle_scope", ""),
        "offline_gesture_bucket": row.get("gesture_bucket", ""),
        "offline_labels": row.get("labels", ""),
        "offline_test_accuracy": row.get("test_accuracy", ""),
        "offline_balanced_accuracy": row.get("balanced_accuracy", ""),
        "offline_macro_f1": row.get("macro_f1", ""),
        "offline_macro_precision": row.get("macro_precision", ""),
        "offline_macro_recall": row.get("macro_recall", ""),
        "offline_weighted_f1": row.get("weighted_f1", ""),
        "offline_weighted_precision": row.get("weighted_precision", ""),
        "offline_weighted_recall": row.get("weighted_recall", ""),
        "offline_worst_class_recall": row.get("worst_class_recall", ""),
        "offline_worst_class_recall_label": row.get("worst_class_recall_label", ""),
        "offline_max_precision_recall_gap": row.get("max_precision_recall_gap", ""),
        "offline_max_precision_recall_gap_label": row.get("max_precision_recall_gap_label", ""),
        "offline_channel_count": row.get("channel_count", ""),
        "offline_window_size_samples": row.get("window_size_samples", ""),
        "offline_window_step_samples": row.get("window_step_samples", ""),
        "offline_created_at": row.get("created_at", ""),
    }


def _normalize_path_text(path_text):
    return str(path_text or "").replace("/", "\\").strip().lower()


def _match_model_rows(manifest_row, model_rows):
    model_path = str(manifest_row.get("model_path", "")).strip()
    model_filename = str(manifest_row.get("model_filename", "")).strip()
    task = _normalize_task(manifest_row.get("task", ""))
    scope = _normalize_scope(manifest_row.get("model_scope", ""))
    arm = str(manifest_row.get("arm", "")).strip().lower()
    subject = str(manifest_row.get("subject", "")).strip().lower()

    if model_path:
        target = _normalize_path_text(model_path)
        return [row for row in model_rows if _normalize_path_text(row.get("path", "")) == target]

    if model_filename:
        target = model_filename.strip().lower()
        return [row for row in model_rows if str(row.get("filename", "")).strip().lower() == target]

    matches = []
    for row in model_rows:
        if task and _normalize_task(row.get("gesture_bucket", "")) != task:
            continue
        if scope and _normalize_scope(row.get("bundle_scope", "")) != scope:
            continue
        if arm and str(row.get("arm", "")).strip().lower() != arm:
            continue
        if subject:
            row_subject = str(row.get("subject", "") or row.get("target_subject", "")).strip().lower()
            if row_subject != subject:
                continue
        matches.append(row)
    return matches


def _build_output_rows(manifest_rows, model_rows):
    out = []
    for manifest_index, manifest_row in enumerate(manifest_rows, start=1):
        matches = _match_model_rows(manifest_row, model_rows)
        if not matches:
            matches = [None]
        for match_index, model_row in enumerate(matches, start=1):
            row = {
                key: manifest_row.get(key, "")
                for key in TEMPLATE_COLUMNS
            }
            if model_row is not None:
                if not row.get("subject"):
                    row["subject"] = model_row.get("subject", "") or model_row.get("target_subject", "")
                if not row.get("arm"):
                    row["arm"] = model_row.get("arm", "")
                if not row.get("task"):
                    row["task"] = model_row.get("gesture_bucket", "")
                if not row.get("model_scope"):
                    row["model_scope"] = model_row.get("bundle_scope", "")
            row["deliverable_bucket"] = _deliverable_bucket(row.get("deliverable", ""))
            suffix_parts = []
            if row.get("subject"):
                suffix_parts.append(str(row["subject"]))
            if row.get("arm"):
                suffix_parts.append(str(row["arm"]))
            if model_row is not None and not suffix_parts:
                suffix_parts.append(str(model_row.get("filename", f"model_{match_index}")))
            row["table_row_id"] = str(row.get("row_id", f"ROW_{manifest_index}"))
            if suffix_parts:
                row["table_row_id"] += "__" + "__".join(_slugify(part) for part in suffix_parts if str(part).strip())
            row.update(_offline_from_model_row(model_row))
            row.update(_flatten_realtime(_read_json(row.get("realtime_behavior_json", ""))))
            row.update(_flatten_latency(_read_json(row.get("latency_json", ""))))
            row.update(_flatten_drive(_read_json(row.get("drive_metrics_json", ""))))
            out.append(row)
    return out


def _dynamic_system_columns(rows):
    dynamic = []
    for key in sorted({key for row in rows for key in row.keys()}):
        if key.startswith("drive_") and key not in SYSTEM_COLUMNS:
            dynamic.append(key)
    return dynamic


def _selected_columns(rows, kind):
    base = list(BASE_COLUMNS)
    if kind == "offline":
        return base + OFFLINE_COLUMNS
    if kind == "system":
        return base + SYSTEM_COLUMNS + _dynamic_system_columns(rows)
    if kind == "full":
        columns = base + OFFLINE_COLUMNS + SYSTEM_COLUMNS + _dynamic_system_columns(rows)
        remaining = [key for key in sorted({key for row in rows for key in row.keys()}) if key not in columns]
        return columns + remaining
    raise ValueError(f"Unknown table kind: {kind}")


def _filter_deliverable(rows, deliverable_bucket):
    return [row for row in rows if row.get("deliverable_bucket") == deliverable_bucket]


def _write_deliverable_tables(output_dir: Path, deliverable_bucket: str, rows):
    if not rows:
        return
    for kind in ("offline", "system", "full"):
        columns = _selected_columns(rows, kind)
        stem = f"{deliverable_bucket}_{kind}_table"
        _write_csv(output_dir / f"{stem}.csv", rows, columns)
        _write_markdown(output_dir / f"{stem}.md", rows, columns)


def _write_template_manifest(path: Path):
    _write_csv(path, TEMPLATE_ROWS, TEMPLATE_COLUMNS)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Assemble capstone/report tables from model, realtime, latency, and drive summaries."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("eval_metrics") / "table_manifest.csv",
        help="CSV manifest describing which rows belong in the paper/report tables.",
    )
    parser.add_argument(
        "--model-metrics",
        type=Path,
        default=Path("eval_metrics") / "out" / "model_metrics.csv",
        help="Offline metrics CSV from harvest_model_metrics.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_metrics") / "out" / "tables",
    )
    parser.add_argument(
        "--write-template-manifest",
        type=Path,
        default=None,
        help="Optional path to write a starter manifest CSV, then exit.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.write_template_manifest is not None:
        _write_template_manifest(Path(args.write_template_manifest))
        print(f"Saved template manifest to {args.write_template_manifest}")
        return

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. "
            f"Create one from the template with --write-template-manifest {manifest_path}"
        )

    manifest_rows = _load_csv_rows(manifest_path)
    model_rows = _load_csv_rows(Path(args.model_metrics))
    assembled_rows = _build_output_rows(manifest_rows, model_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    master_columns = _selected_columns(assembled_rows, "full")
    _write_csv(output_dir / "evaluation_master_table.csv", assembled_rows, master_columns)
    _write_json(output_dir / "evaluation_master_table.json", assembled_rows)

    _write_deliverable_tables(output_dir, "capstone_report", _filter_deliverable(assembled_rows, "capstone_report"))
    _write_deliverable_tables(output_dir, "research_paper", _filter_deliverable(assembled_rows, "research_paper"))

    print(f"Loaded {len(manifest_rows)} manifest row(s)")
    print(f"Built {len(assembled_rows)} assembled table row(s)")
    print(f"Saved tables under {output_dir}")


if __name__ == "__main__":
    main()
