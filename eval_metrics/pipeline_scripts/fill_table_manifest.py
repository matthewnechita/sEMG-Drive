from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval_metrics.config import ACTIVE_TABLE_MODEL_NAME


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


def _slugify(text: str) -> str:
    value = str(text or "").strip().lower()
    value = value.replace("&", "and")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def _normalize_scope(text: str) -> str:
    slug = _slugify(text)
    if "cross" in slug:
        return "cross_subject"
    if "per" in slug or "single" in slug:
        return "per_subject"
    return slug


def _normalize_task(text: str) -> str:
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


def _report_task_name(task: str) -> str:
    task = _normalize_task(task)
    if task.endswith("_gesture"):
        return task.replace("_", "-")
    return task.replace("_", " ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Auto-fill eval_metrics/table_manifest.csv from staged current_metrics outputs."
    )
    parser.add_argument(
        "--template-manifest",
        type=Path,
        default=Path("eval_metrics") / "out" / "current_metrics" / "table_manifest_template.csv",
        help="Starter manifest CSV to expand.",
    )
    parser.add_argument(
        "--run-index",
        type=Path,
        default=Path("eval_metrics") / "out" / "current_metrics" / "carla_run_index.csv",
        help="Run index CSV from gather_current_metrics.py",
    )
    parser.add_argument(
        "--model-metrics",
        type=Path,
        default=Path("eval_metrics") / "out" / "current_metrics" / "model_metrics.csv",
        help="Harvested model metrics CSV.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("eval_metrics") / "table_manifest.csv",
        help="Filled manifest output path.",
    )
    parser.add_argument(
        "--model-name",
        action="append",
        default=[],
        help=(
            "Exact model filename or stem to include. Repeatable. "
            "Example: v6_4_gestures_2.pt"
        ),
    )
    parser.add_argument(
        "--model-scope",
        default="",
        help="Optional model scope override, such as Cross-subject or Per-subject.",
    )
    parser.add_argument(
        "--task",
        default="",
        help="Optional task override, such as 4-gesture.",
    )
    parser.add_argument(
        "--status",
        default="Complete",
        help="Status text for generated rows.",
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    template_rows = _load_csv_rows(Path(args.template_manifest))
    run_rows = _load_csv_rows(Path(args.run_index))
    model_rows = _load_csv_rows(Path(args.model_metrics))
    fieldnames = list(template_rows[0].keys()) if template_rows else [
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

    requested_names = {str(name).strip().lower() for name in args.model_name if str(name).strip()}
    if not requested_names and str(ACTIVE_TABLE_MODEL_NAME).strip():
        requested_names = {str(ACTIVE_TABLE_MODEL_NAME).strip().lower()}
    filtered_models = []
    for row in model_rows:
        if requested_names:
            name = str(row.get("filename") or "").strip().lower()
            stem = Path(name).stem.lower() if name else ""
            if name not in requested_names and stem not in requested_names:
                continue
        filtered_models.append(row)

    if not filtered_models:
        raise ValueError("No harvested model rows matched the requested --model-name filter.")

    inferred_scopes = sorted({_normalize_scope(str(row.get("bundle_scope", ""))) for row in filtered_models if str(row.get("bundle_scope", "")).strip()})
    inferred_tasks = sorted({_normalize_task(str(row.get("gesture_bucket", ""))) for row in filtered_models if str(row.get("gesture_bucket", "")).strip()})

    scope = _normalize_scope(str(args.model_scope or ""))
    if not scope:
        if len(inferred_scopes) != 1:
            raise ValueError(
                f"Could not infer one model scope from harvested rows: {inferred_scopes}. "
                "Pass --model-scope explicitly."
            )
        scope = inferred_scopes[0]

    task = _normalize_task(str(args.task or ""))
    if not task:
        if len(inferred_tasks) != 1:
            raise ValueError(
                f"Could not infer one task from harvested rows: {inferred_tasks}. "
                "Pass --task explicitly."
            )
        task = inferred_tasks[0]

    selected_templates = []
    untouched_templates = []
    for row in template_rows:
        row_scope = _normalize_scope(str(row.get("model_scope", "")))
        row_task = _normalize_task(str(row.get("task", "")))
        if row_scope == scope and row_task == task:
            selected_templates.append(row)
        else:
            untouched_templates.append(row)

    if not selected_templates:
        raise ValueError(
            f"No template rows matched scope={scope!r} and task={task!r} in {args.template_manifest}"
        )

    if not run_rows:
        raise ValueError(f"No run rows found in {args.run_index}")

    chosen_model_name = ""
    unique_names = sorted({str(row.get("filename") or "").strip() for row in filtered_models if str(row.get("filename") or "").strip()})
    if len(unique_names) == 1:
        chosen_model_name = unique_names[0]
    elif len(requested_names) == 1:
        chosen_model_name = next(iter(requested_names))

    generated_rows: list[dict[str, object]] = []
    for template_row in selected_templates:
        for run_row in run_rows:
            row = dict(template_row)
            row["task"] = _report_task_name(task)
            row["model_scope"] = "Cross-subject" if scope == "cross_subject" else "Per-subject"
            row["status"] = str(args.status).strip() or str(template_row.get("status", "")).strip()
            row["model_filename"] = chosen_model_name
            row["latency_json"] = str(run_row.get("latency_json") or "").strip()
            row["drive_metrics_json"] = str(run_row.get("drive_json") or "").strip()
            generated_rows.append(row)

    output_rows = untouched_templates + generated_rows
    _write_csv(Path(args.output_manifest), output_rows, fieldnames)

    valid_latency = sum(1 for row in run_rows if str(row.get("latency_ok") or "").strip().lower() == "true")
    print(f"Matched {len(filtered_models)} harvested model row(s)")
    print(f"Selected template row(s): {len(selected_templates)}")
    print(f"Expanded run row(s): {len(run_rows)}")
    print(f"Runs with latency summaries: {valid_latency}")
    print(f"Saved filled manifest to {args.output_manifest}")


if __name__ == "__main__":
    main()
