import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval_metrics.config import (
    CURRENT_METRICS_ROOT,
    FINAL_CAPSTONE_TABLE_CSV,
    FINAL_RESEARCH_TABLE_CSV,
    TABLES_ROOT,
)


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
    "latency_json",
    "drive_metrics_json",
    "notes",
]


# The template is intentionally small and hand-editable so the report workflow
# can be curated row-by-row instead of inferred entirely from filenames.
TEMPLATE_ROWS = [
    {
        "row_id": "CR_4_PER",
        "deliverable": "Capstone final report",
        "task": "4-gesture",
        "model_scope": "Per-subject",
        "condition": "current_cnn_pipeline",
        "arm": "",
        "subject": "",
        "status": "Pending",
        "comparison_group": "",
        "model_path": "",
        "model_filename": "",
        "latency_json": "",
        "drive_metrics_json": "",
        "notes": "",
    },
    {
        "row_id": "CR_4_CROSS",
        "deliverable": "Capstone final report",
        "task": "4-gesture",
        "model_scope": "Cross-subject",
        "condition": "current_cnn_pipeline",
        "arm": "",
        "subject": "",
        "status": "Pending",
        "comparison_group": "",
        "model_path": "",
        "model_filename": "",
        "latency_json": "",
        "drive_metrics_json": "",
        "notes": "",
    },
    {
        "row_id": "RP_4_PER",
        "deliverable": "Research paper",
        "task": "4-gesture",
        "model_scope": "Per-subject",
        "condition": "current_cnn_pipeline",
        "arm": "",
        "subject": "",
        "status": "Pending",
        "comparison_group": "Prior-paper comparison, no eye-tracker metrics",
        "model_path": "",
        "model_filename": "",
        "latency_json": "",
        "drive_metrics_json": "",
        "notes": "No eye-tracker metrics",
    },
    {
        "row_id": "RP_4_CROSS",
        "deliverable": "Research paper",
        "task": "4-gesture",
        "model_scope": "Cross-subject",
        "condition": "current_cnn_pipeline",
        "arm": "",
        "subject": "",
        "status": "Pending",
        "comparison_group": "Prior-paper comparison, no eye-tracker metrics",
        "model_path": "",
        "model_filename": "",
        "latency_json": "",
        "drive_metrics_json": "",
        "notes": "No eye-tracker metrics",
    },
]


PARTICIPANT_COLUMNS = [
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
    "source_model_filename",
    "source_model_path",
    "offline_bundle_scope",
    "offline_gesture_bucket",
    "offline_labels",
    "offline_balanced_accuracy",
    "offline_macro_precision",
    "offline_macro_recall",
    "offline_macro_f1",
    "offline_worst_class_recall",
    "offline_loso_subject_count",
    "offline_loso_balanced_accuracy",
    "offline_loso_macro_precision",
    "offline_loso_macro_recall",
    "offline_loso_macro_f1",
    "offline_loso_worst_class_recall",
    "offline_channel_count",
    "offline_window_size_samples",
    "offline_window_step_samples",
    "offline_created_at",
    "latency_json",
    "lat_rows_joined",
    "lat_e2e_mean_ms",
    "lat_e2e_p95_ms",
    "drive_metrics_json",
    "drive_rows",
    "drive_scenario_name",
    "drive_scenario_kind",
    "drive_scenario_success",
    "drive_completion_time_s",
    "drive_mean_velocity_mps",
    "drive_lane_offset_mean_m",
    "drive_steering_angle_mean_rad",
    "drive_mean_velocity_deviation_mps",
    "drive_steering_entropy",
    "drive_lane_error_rmse_m",
    "drive_lane_invasions",
    "drive_command_success_rate",
]


AGGREGATE_COLUMNS = [
    "aggregate_row_id",
    "row_id",
    "deliverable",
    "deliverable_bucket",
    "task",
    "model_scope",
    "condition",
    "arm",
    "drive_scenario_name",
    "drive_scenario_kind",
    "status",
    "comparison_group",
    "notes",
    "participant_count",
    "source_run_count",
    "offline_balanced_accuracy_mean",
    "offline_balanced_accuracy_sd",
    "offline_macro_precision_mean",
    "offline_macro_precision_sd",
    "offline_macro_recall_mean",
    "offline_macro_recall_sd",
    "offline_macro_f1_mean",
    "offline_macro_f1_sd",
    "offline_worst_class_recall_mean",
    "offline_worst_class_recall_sd",
    "offline_loso_subject_count_mean",
    "offline_loso_subject_count_sd",
    "offline_loso_balanced_accuracy_mean",
    "offline_loso_balanced_accuracy_sd",
    "offline_loso_macro_precision_mean",
    "offline_loso_macro_precision_sd",
    "offline_loso_macro_recall_mean",
    "offline_loso_macro_recall_sd",
    "offline_loso_macro_f1_mean",
    "offline_loso_macro_f1_sd",
    "offline_loso_worst_class_recall_mean",
    "offline_loso_worst_class_recall_sd",
    "lat_e2e_mean_ms_mean",
    "lat_e2e_mean_ms_sd",
    "lat_e2e_p95_ms_mean",
    "lat_e2e_p95_ms_sd",
    "drive_scenario_success_rate",
    "drive_completion_time_s_mean",
    "drive_completion_time_s_sd",
    "drive_mean_velocity_mps_mean",
    "drive_mean_velocity_mps_sd",
    "drive_lane_offset_mean_m_mean",
    "drive_lane_offset_mean_m_sd",
    "drive_steering_angle_mean_rad_mean",
    "drive_steering_angle_mean_rad_sd",
    "drive_mean_velocity_deviation_mps_mean",
    "drive_mean_velocity_deviation_mps_sd",
    "drive_steering_entropy_mean",
    "drive_steering_entropy_sd",
    "drive_lane_error_rmse_m_mean",
    "drive_lane_error_rmse_m_sd",
    "drive_lane_invasions_mean",
    "drive_lane_invasions_sd",
    "drive_command_success_rate_mean",
    "drive_command_success_rate_sd",
]

PUBLICATION_OFFLINE_COLUMNS = [
    "Model",
    "Arm",
    "Bundled balanced accuracy (%)",
    "Bundled macro precision (%)",
    "Bundled macro recall (%)",
    "Bundled macro F1 (%)",
    "Bundled worst-class recall (%)",
    "LOSO held-out subjects",
    "LOSO balanced accuracy (%)",
    "LOSO macro precision (%)",
    "LOSO macro recall (%)",
    "LOSO macro F1 (%)",
    "LOSO worst-class recall (%)",
]

PUBLICATION_RUNTIME_COLUMNS = [
    "Model",
    "Scenario",
    "Runs",
    "Latency mean (ms)",
    "Latency p95 (ms)",
    "Scenario success rate (%)",
    "Completion time (s)",
    "Mean velocity (m/s)",
    "Mean velocity deviation (m/s)",
    "Lane offset mean (m)",
    "Steering angle mean (rad)",
    "Steering entropy",
    "Lane error RMSE (m)",
    "Lane invasions",
]


NUMERIC_PARTICIPANT_METRICS = [
    "offline_balanced_accuracy",
    "offline_macro_precision",
    "offline_macro_recall",
    "offline_macro_f1",
    "offline_worst_class_recall",
    "offline_loso_subject_count",
    "offline_loso_balanced_accuracy",
    "offline_loso_macro_precision",
    "offline_loso_macro_recall",
    "offline_loso_macro_f1",
    "offline_loso_worst_class_recall",
    "lat_e2e_mean_ms",
    "lat_e2e_p95_ms",
    "drive_completion_time_s",
    "drive_mean_velocity_mps",
    "drive_lane_offset_mean_m",
    "drive_steering_angle_mean_rad",
    "drive_mean_velocity_deviation_mps",
    "drive_steering_entropy",
    "drive_lane_error_rmse_m",
    "drive_lane_invasions",
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
    if slug.startswith("4_gesture"):
        return "4_gesture"
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


def _safe_write_csv(path: Path, rows, columns):
    try:
        _write_csv(path, rows, columns)
        return True
    except PermissionError:
        print(f"Warning: could not write {path} because it is open in another program.")
        return False


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


def _normalize_path_text(path_text):
    return str(path_text or "").replace("/", "\\").strip().lower()


def _to_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value):
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def _mean(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _mean_sd(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None, None
    if len(vals) == 1:
        return float(vals[0]), 0.0
    mean = float(sum(vals) / len(vals))
    variance = float(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
    return mean, math.sqrt(variance)


def _round_or_blank(value, digits=1):
    number = _to_float(value)
    if number is None:
        return ""
    return round(number, digits)


def _percent_or_blank(value, digits=1):
    number = _to_float(value)
    if number is None:
        return ""
    return round(number * 100.0, digits)


def _pretty_scope(text):
    slug = _normalize_scope(text)
    if slug == "cross_subject":
        return "Cross-subject"
    if slug == "per_subject":
        return "Per-subject"
    return str(text or "").replace("_", " ").strip().title()


def _pretty_arm(text):
    arm = str(text or "").strip().lower()
    if arm == "left":
        return "Left arm"
    if arm == "right":
        return "Right arm"
    return str(text or "").replace("_", " ").strip().title()


def _pretty_scenario(text):
    value = str(text or "").strip().lower()
    if value == "lane_keep_5min":
        return "Lane keep"
    if value == "highway_overtake":
        return "Highway overtake"
    return str(text or "").replace("_", " ").strip().title()


def _flatten_latency(summary):
    if not isinstance(summary, dict) or not summary:
        return {}
    e2e = summary.get("end_to_end_latency_ms", {})
    return {
        "lat_rows_joined": summary.get("rows_joined"),
        "lat_e2e_mean_ms": e2e.get("mean_ms") if isinstance(e2e, dict) else None,
        "lat_e2e_p95_ms": e2e.get("p95_ms") if isinstance(e2e, dict) else None,
    }


def _flatten_drive(summary):
    if not isinstance(summary, dict) or not summary:
        return {}
    full = summary.get("full_route", {})
    if not isinstance(full, dict):
        full = {}
    out = {
        "drive_rows": full.get("rows"),
        "drive_scenario_name": full.get("scenario_name"),
        "drive_scenario_kind": full.get("scenario_kind"),
        "drive_scenario_success": full.get("scenario_success"),
        "drive_completion_time_s": full.get("completion_time_s"),
        "drive_mean_velocity_mps": full.get("mean_velocity_mps"),
        "drive_lane_offset_mean_m": full.get("lane_offset_mean_m"),
        "drive_steering_angle_mean_rad": full.get("steering_angle_mean_rad"),
        "drive_mean_velocity_deviation_mps": full.get("mean_velocity_deviation_mps"),
        "drive_steering_entropy": full.get("steering_entropy"),
        "drive_lane_error_rmse_m": full.get("lane_error_rmse_m"),
        "drive_lane_invasions": full.get("lane_invasions"),
    }
    if "command_success_rate" in full:
        out["drive_command_success_rate"] = full.get("command_success_rate")
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
        "offline_balanced_accuracy": row.get("balanced_accuracy", ""),
        "offline_macro_precision": row.get("macro_precision", ""),
        "offline_macro_recall": row.get("macro_recall", ""),
        "offline_macro_f1": row.get("macro_f1", ""),
        "offline_worst_class_recall": row.get("worst_class_recall", ""),
        "offline_loso_subject_count": row.get("loso_subject_count", ""),
        "offline_loso_balanced_accuracy": row.get("loso_balanced_accuracy", ""),
        "offline_loso_macro_precision": row.get("loso_macro_precision", ""),
        "offline_loso_macro_recall": row.get("loso_macro_recall", ""),
        "offline_loso_macro_f1": row.get("loso_macro_f1", ""),
        "offline_loso_worst_class_recall": row.get("loso_worst_class_recall", ""),
        "offline_channel_count": row.get("channel_count", ""),
        "offline_window_size_samples": row.get("window_size_samples", ""),
        "offline_window_step_samples": row.get("window_step_samples", ""),
        "offline_created_at": row.get("created_at", ""),
    }


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
        target = model_filename.lower()
        return [row for row in model_rows if str(row.get("filename", "")).strip().lower() == target]

    # Fall back to semantic matching so a manifest can stay stable even when the
    # exact bundle filename changes after a retrain.
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


def _build_participant_rows(manifest_rows, model_rows):
    out = []
    for manifest_index, manifest_row in enumerate(manifest_rows, start=1):
        matches = _match_model_rows(manifest_row, model_rows)
        if not matches:
            matches = [None]
        for match_index, model_row in enumerate(matches, start=1):
            row = {key: manifest_row.get(key, "") for key in TEMPLATE_COLUMNS}
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
                # Keep participant row ids deterministic so CSV, JSON, and Markdown
                # exports all refer to the same logical row.
                row["table_row_id"] += "__" + "__".join(_slugify(part) for part in suffix_parts if str(part).strip())

            row.update(_offline_from_model_row(model_row))
            row.update(_flatten_latency(_read_json(row.get("latency_json", ""))))
            row.update(_flatten_drive(_read_json(row.get("drive_metrics_json", ""))))
            out.append(row)
    return out


def _aggregate_group_key(row):
    return (
        str(row.get("row_id", "")),
        str(row.get("deliverable", "")),
        str(row.get("deliverable_bucket", "")),
        str(row.get("task", "")),
        str(row.get("model_scope", "")),
        str(row.get("condition", "")),
        str(row.get("arm", "")),
        str(row.get("drive_scenario_name", "")),
        str(row.get("drive_scenario_kind", "")),
        str(row.get("status", "")),
        str(row.get("comparison_group", "")),
        str(row.get("notes", "")),
    )


def _participant_key(row):
    subject = str(row.get("subject", "")).strip()
    if subject:
        return subject.lower()
    return str(row.get("table_row_id", "")).strip().lower()


def _aggregate_participant_bucket(rows):
    out = {}
    # Average repeated runs per participant first so one subject with extra logs
    # does not dominate the final aggregate table.
    for metric in NUMERIC_PARTICIPANT_METRICS:
        out[metric] = _mean(_to_float(row.get(metric)) for row in rows)
    success_values = [_to_bool(row.get("drive_scenario_success")) for row in rows]
    success_values = [1.0 if value else 0.0 for value in success_values if value is not None]
    out["drive_scenario_success_rate"] = _mean(success_values)
    return out


def _build_aggregate_rows(participant_rows):
    grouped = defaultdict(list)
    for row in participant_rows:
        grouped[_aggregate_group_key(row)].append(row)

    aggregate_rows = []
    for key, rows in sorted(grouped.items()):
        participant_groups = defaultdict(list)
        for row in rows:
            participant_groups[_participant_key(row)].append(row)

        participant_summaries = [
            _aggregate_participant_bucket(group_rows)
            for _, group_rows in sorted(participant_groups.items())
        ]

        template_row = rows[0]
        aggregate_row = {
            "aggregate_row_id": "__".join(
                part for part in [
                    str(template_row.get("row_id", "")).strip(),
                    _slugify(template_row.get("arm", "")),
                    _slugify(template_row.get("condition", "")),
                ]
                if part
            ) or str(template_row.get("row_id", "")).strip(),
            "row_id": template_row.get("row_id", ""),
            "deliverable": template_row.get("deliverable", ""),
            "deliverable_bucket": template_row.get("deliverable_bucket", ""),
            "task": template_row.get("task", ""),
            "model_scope": template_row.get("model_scope", ""),
            "condition": template_row.get("condition", ""),
            "arm": template_row.get("arm", ""),
            "drive_scenario_name": template_row.get("drive_scenario_name", ""),
            "drive_scenario_kind": template_row.get("drive_scenario_kind", ""),
            "status": template_row.get("status", ""),
            "comparison_group": template_row.get("comparison_group", ""),
            "notes": template_row.get("notes", ""),
            "participant_count": len(participant_summaries),
            "source_run_count": len(rows),
        }

        for metric in [
            "offline_balanced_accuracy",
            "offline_macro_precision",
            "offline_macro_recall",
            "offline_macro_f1",
            "offline_worst_class_recall",
            "offline_loso_subject_count",
            "offline_loso_balanced_accuracy",
            "offline_loso_macro_precision",
            "offline_loso_macro_recall",
            "offline_loso_macro_f1",
            "offline_loso_worst_class_recall",
            "lat_e2e_mean_ms",
            "lat_e2e_p95_ms",
            "drive_completion_time_s",
            "drive_mean_velocity_mps",
            "drive_lane_offset_mean_m",
            "drive_steering_angle_mean_rad",
            "drive_mean_velocity_deviation_mps",
            "drive_steering_entropy",
            "drive_lane_error_rmse_m",
            "drive_lane_invasions",
            "drive_command_success_rate",
        ]:
            # Aggregate rows summarize participant-level means, not raw trial rows.
            mean, sd = _mean_sd(summary.get(metric) for summary in participant_summaries)
            aggregate_row[f"{metric}_mean"] = mean
            aggregate_row[f"{metric}_sd"] = sd

        aggregate_row["drive_scenario_success_rate"] = _mean(
            summary.get("drive_scenario_success_rate") for summary in participant_summaries
        )
        aggregate_rows.append(aggregate_row)

    return aggregate_rows


def _filter_deliverable(rows, deliverable_bucket):
    return [row for row in rows if row.get("deliverable_bucket") == deliverable_bucket]


def _write_deliverable_tables(output_dir: Path, deliverable_bucket: str, participant_rows, aggregate_rows):
    if participant_rows:
        _write_csv(
            output_dir / f"{deliverable_bucket}_participant_table.csv",
            participant_rows,
            PARTICIPANT_COLUMNS,
        )
        _write_markdown(
            output_dir / f"{deliverable_bucket}_participant_table.md",
            participant_rows,
            PARTICIPANT_COLUMNS,
        )
    if aggregate_rows:
        _write_csv(
            output_dir / f"{deliverable_bucket}_aggregate_table.csv",
            aggregate_rows,
            AGGREGATE_COLUMNS,
        )
        _write_markdown(
            output_dir / f"{deliverable_bucket}_aggregate_table.md",
            aggregate_rows,
            AGGREGATE_COLUMNS,
        )


def _write_top_level_final_tables(capstone_rows, research_rows):
    if capstone_rows:
        _safe_write_csv(FINAL_CAPSTONE_TABLE_CSV, capstone_rows, AGGREGATE_COLUMNS)
    if research_rows:
        _safe_write_csv(FINAL_RESEARCH_TABLE_CSV, research_rows, AGGREGATE_COLUMNS)


def _publication_offline_rows(aggregate_rows, deliverable_bucket):
    picked = {}
    for row in aggregate_rows:
        if str(row.get("deliverable_bucket", "")).strip() != deliverable_bucket:
            continue
        key = (
            str(row.get("model_scope", "")).strip().lower(),
            str(row.get("arm", "")).strip().lower(),
        )
        if key not in picked:
            picked[key] = row

    ordered = sorted(
        picked.values(),
        key=lambda row: (
            0 if _normalize_scope(row.get("model_scope", "")) == "cross_subject" else 1,
            0 if str(row.get("arm", "")).strip().lower() == "left" else 1,
        ),
    )
    out = []
    for row in ordered:
        out.append(
            {
                "Model": _pretty_scope(row.get("model_scope", "")),
                "Arm": _pretty_arm(row.get("arm", "")),
                "Bundled balanced accuracy (%)": _percent_or_blank(row.get("offline_balanced_accuracy_mean")),
                "Bundled macro precision (%)": _percent_or_blank(row.get("offline_macro_precision_mean")),
                "Bundled macro recall (%)": _percent_or_blank(row.get("offline_macro_recall_mean")),
                "Bundled macro F1 (%)": _percent_or_blank(row.get("offline_macro_f1_mean")),
                "Bundled worst-class recall (%)": _percent_or_blank(row.get("offline_worst_class_recall_mean")),
                "LOSO held-out subjects": _round_or_blank(row.get("offline_loso_subject_count_mean"), 0),
                "LOSO balanced accuracy (%)": _percent_or_blank(row.get("offline_loso_balanced_accuracy_mean")),
                "LOSO macro precision (%)": _percent_or_blank(row.get("offline_loso_macro_precision_mean")),
                "LOSO macro recall (%)": _percent_or_blank(row.get("offline_loso_macro_recall_mean")),
                "LOSO macro F1 (%)": _percent_or_blank(row.get("offline_loso_macro_f1_mean")),
                "LOSO worst-class recall (%)": _percent_or_blank(row.get("offline_loso_worst_class_recall_mean")),
            }
        )
    return out


def _publication_runtime_rows(aggregate_rows, deliverable_bucket):
    grouped = defaultdict(list)
    for row in aggregate_rows:
        if str(row.get("deliverable_bucket", "")).strip() != deliverable_bucket:
            continue
        scenario_name = str(row.get("drive_scenario_name", "")).strip()
        if not scenario_name:
            continue
        key = (
            str(row.get("model_scope", "")).strip(),
            scenario_name,
        )
        grouped[key].append(row)

    scenario_order = {
        "highway_overtake": 0,
        "lane_keep_5min": 1,
    }
    scope_order = {
        "cross_subject": 0,
        "per_subject": 1,
    }
    out = []
    for (model_scope, scenario_name), rows in sorted(
        grouped.items(),
        key=lambda item: (
            scope_order.get(_normalize_scope(item[0][0]), 99),
            scenario_order.get(str(item[0][1]).strip().lower(), 99),
        ),
    ):
        out.append(
            {
                "Model": _pretty_scope(model_scope),
                "Scenario": _pretty_scenario(scenario_name),
                "Runs": _round_or_blank(_mean(_to_float(row.get("source_run_count")) for row in rows), 0),
                "Latency mean (ms)": _round_or_blank(_mean(_to_float(row.get("lat_e2e_mean_ms_mean")) for row in rows), 1),
                "Latency p95 (ms)": _round_or_blank(_mean(_to_float(row.get("lat_e2e_p95_ms_mean")) for row in rows), 1),
                "Scenario success rate (%)": _percent_or_blank(_mean(_to_float(row.get("drive_scenario_success_rate")) for row in rows)),
                "Completion time (s)": _round_or_blank(_mean(_to_float(row.get("drive_completion_time_s_mean")) for row in rows), 1),
                "Mean velocity (m/s)": _round_or_blank(_mean(_to_float(row.get("drive_mean_velocity_mps_mean")) for row in rows), 2),
                "Mean velocity deviation (m/s)": _round_or_blank(_mean(_to_float(row.get("drive_mean_velocity_deviation_mps_mean")) for row in rows), 2),
                "Lane offset mean (m)": _round_or_blank(_mean(_to_float(row.get("drive_lane_offset_mean_m_mean")) for row in rows), 3),
                "Steering angle mean (rad)": _round_or_blank(_mean(_to_float(row.get("drive_steering_angle_mean_rad_mean")) for row in rows), 3),
                "Steering entropy": _round_or_blank(_mean(_to_float(row.get("drive_steering_entropy_mean")) for row in rows), 3),
                "Lane error RMSE (m)": _round_or_blank(_mean(_to_float(row.get("drive_lane_error_rmse_m_mean")) for row in rows), 3),
                "Lane invasions": _round_or_blank(_mean(_to_float(row.get("drive_lane_invasions_mean")) for row in rows), 1),
            }
        )
    return out


def _write_publication_tables(capstone_rows, research_rows):
    root = FINAL_CAPSTONE_TABLE_CSV.parent
    outputs = [
        (
            root / "final_capstone_offline_table.csv",
            _publication_offline_rows(capstone_rows, "capstone_report"),
            PUBLICATION_OFFLINE_COLUMNS,
        ),
        (
            root / "final_capstone_runtime_table.csv",
            _publication_runtime_rows(capstone_rows, "capstone_report"),
            PUBLICATION_RUNTIME_COLUMNS,
        ),
        (
            root / "final_research_offline_table.csv",
            _publication_offline_rows(research_rows, "research_paper"),
            PUBLICATION_OFFLINE_COLUMNS,
        ),
        (
            root / "final_research_runtime_table.csv",
            _publication_runtime_rows(research_rows, "research_paper"),
            PUBLICATION_RUNTIME_COLUMNS,
        ),
    ]
    for path, rows, columns in outputs:
        _safe_write_csv(path, rows, columns)


def _write_template_manifest(path: Path):
    _write_csv(path, TEMPLATE_ROWS, TEMPLATE_COLUMNS)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Assemble participant and aggregate tables from offline, latency, and CARLA summaries."
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
        default=CURRENT_METRICS_ROOT / "model_metrics.csv",
        help="Offline metrics CSV from harvest_model_metrics.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TABLES_ROOT,
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
    participant_rows = _build_participant_rows(manifest_rows, model_rows)
    aggregate_rows = _build_aggregate_rows(participant_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(output_dir / "evaluation_participant_table.csv", participant_rows, PARTICIPANT_COLUMNS)
    _write_json(output_dir / "evaluation_participant_table.json", participant_rows)
    _write_csv(output_dir / "evaluation_aggregate_table.csv", aggregate_rows, AGGREGATE_COLUMNS)
    _write_json(output_dir / "evaluation_aggregate_table.json", aggregate_rows)

    _write_deliverable_tables(
        output_dir,
        "capstone_report",
        _filter_deliverable(participant_rows, "capstone_report"),
        _filter_deliverable(aggregate_rows, "capstone_report"),
    )
    _write_deliverable_tables(
        output_dir,
        "research_paper",
        _filter_deliverable(participant_rows, "research_paper"),
        _filter_deliverable(aggregate_rows, "research_paper"),
    )
    _write_top_level_final_tables(
        _filter_deliverable(aggregate_rows, "capstone_report"),
        _filter_deliverable(aggregate_rows, "research_paper"),
    )
    _write_publication_tables(
        _filter_deliverable(aggregate_rows, "capstone_report"),
        _filter_deliverable(aggregate_rows, "research_paper"),
    )

    print(f"Loaded {len(manifest_rows)} manifest row(s)")
    print(f"Built {len(participant_rows)} participant row(s)")
    print(f"Built {len(aggregate_rows)} aggregate row(s)")
    print(f"Saved tables under {output_dir}")
    print(f"Saved top-level final capstone table to {FINAL_CAPSTONE_TABLE_CSV}")
    print(f"Saved top-level final research table to {FINAL_RESEARCH_TABLE_CSV}")
    print(f"Saved publication-ready capstone tables to {FINAL_CAPSTONE_TABLE_CSV.parent}")
    print(f"Saved publication-ready research tables to {FINAL_RESEARCH_TABLE_CSV.parent}")


if __name__ == "__main__":
    main()
