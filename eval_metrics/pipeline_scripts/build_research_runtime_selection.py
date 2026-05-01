from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval_metrics.config import ACTIVE_CROSS_SUBJECT_MODEL_NAME, ACTIVE_PER_SUBJECT_MODEL_NAME, CURRENT_METRICS_ROOT


SELECTIONS = {
    "Cross-subject": {
        "highway_overtake_eval": [
            ("P01", "20260410_141410"),
            ("P02", "20260410_180902"),
            ("P03", "20260410_190110"),
            ("P04", "20260410_194307"),
            ("P05", "20260410_200738"),
            ("P06", "20260411_123507"),
            ("P07", "20260411_155300"),
            ("P08", "20260411_162920"),
            ("P09", "20260411_165600"),
            ("P10", "20260411_171845"),
        ],
        "lane_keep_eval": [
            ("P01", "20260410_135108"),
            ("P02", "20260410_175647"),
            ("P03", "20260410_185156"),
            ("P04", "20260410_193624"),
            ("P05", "20260410_195835"),
            ("P06", "20260411_120306"),
            ("P07", "20260411_122415"),
            ("P08", "20260411_154114"),
            ("P09", "20260411_162114"),
            ("P10", "20260411_165048"),
        ],
    },
    "Per-subject": {
        "highway_overtake_eval": [
            ("P10", "20260411_172040"),
            ("P10", "20260411_182433"),
        ],
        "lane_keep_eval": [
            ("P10", "20260411_170945"),
            ("P10", "20260411_171442"),
            ("P10", "20260411_181418"),
        ],
    },
}


EXCLUSION_REASONS = {
    ("highway_overtake_eval", "20260410_140651"): "Excluded from research selection: waiting_start log and no latency join.",
    ("highway_overtake_eval", "20260410_141632"): "Excluded from research selection: duplicate local attempt block; earlier representative retained.",
    ("highway_overtake_eval", "20260410_185851"): "Excluded from research selection: duplicate local attempt block; later representative retained.",
    ("highway_overtake_eval", "20260410_194512"): "Excluded from research selection: duplicate local attempt block; earlier representative retained.",
    ("lane_keep_eval", "20260410_175419"): "Excluded from research selection: waiting_start log and no latency join.",
    ("lane_keep_eval", "20260410_193411"): "Excluded from research selection: waiting_start duplicate; later representative retained.",
}


MANIFEST_COLUMNS = [
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


SELECTION_COLUMNS = [
    "row_id",
    "deliverable",
    "task",
    "model_scope",
    "condition",
    "subject",
    "scenario_name",
    "scenario_label",
    "stamp",
    "run_dir",
    "carla_log",
    "realtime_log",
    "latency_ok",
    "latency_json",
    "latency_csv",
    "drive_json",
    "model_filename",
    "selection_reason",
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


def _pretty_scenario(run_dir: str) -> str:
    if str(run_dir).strip().lower() == "highway_overtake_eval":
        return "Highway overtake"
    if str(run_dir).strip().lower() == "lane_keep_eval":
        return "Lane keep"
    return str(run_dir).replace("_eval", "").replace("_", " ").strip().title()


def _selection_reason(model_scope: str, subject: str, run_dir: str, stamp: str) -> str:
    if model_scope == "Per-subject":
        return "Retained repeated-session participant rerun for per-subject runtime analysis."
    if str(subject).strip().lower() == "p10":
        return "Retained repeated-session participant run in the cross-subject set before separating the remaining per-subject reruns."
    if (run_dir, stamp) in EXCLUSION_REASONS:
        return "Retained representative run from an earlier participant block."
    return "Retained cross-subject participant run."


def _manifest_row(selection_row: dict[str, object]) -> dict[str, object]:
    model_scope = str(selection_row["model_scope"])
    row_id = "RP_4_CROSS" if model_scope == "Cross-subject" else "RP_4_PER"
    model_filename = ACTIVE_CROSS_SUBJECT_MODEL_NAME if model_scope == "Cross-subject" else ACTIVE_PER_SUBJECT_MODEL_NAME
    notes = "Research-paper curated runtime selection"
    if model_scope == "Per-subject":
        notes = "Repeated-participant per-subject runtime set"
    return {
        "row_id": row_id,
        "deliverable": "Research paper",
        "task": "4-gesture",
        "model_scope": model_scope,
        "condition": "current_cnn_pipeline",
        "arm": "",
        "subject": selection_row["subject"],
        "status": "Retained",
        "comparison_group": "Prior-paper comparison, no eye-tracker metrics",
        "model_path": "",
        "model_filename": model_filename,
        "latency_json": selection_row["latency_json"],
        "drive_metrics_json": selection_row["drive_json"],
        "notes": notes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the curated runtime selection and manifest used by the research paper."
    )
    parser.add_argument(
        "--run-index",
        type=Path,
        default=CURRENT_METRICS_ROOT / "carla_run_index.csv",
        help="Staged run index from gather_current_metrics.py",
    )
    parser.add_argument(
        "--output-selection",
        type=Path,
        default=CURRENT_METRICS_ROOT / "research_runtime_selection.csv",
        help="Output CSV describing the curated research runtime run set.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=CURRENT_METRICS_ROOT / "research_table_manifest.csv",
        help="Output manifest CSV for build_eval_tables.py",
    )
    parser.add_argument(
        "--output-audit",
        type=Path,
        default=CURRENT_METRICS_ROOT / "research_runtime_audit.json",
        help="Output JSON audit summary.",
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    run_rows = _load_csv_rows(Path(args.run_index))
    rows_by_key = {
        (str(row.get("run_dir") or "").strip(), str(row.get("stamp") or "").strip()): row
        for row in run_rows
    }

    selection_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    included_keys: set[tuple[str, str]] = set()
    missing: list[dict[str, str]] = []

    for model_scope, scenario_map in SELECTIONS.items():
        for run_dir, subject_stamps in scenario_map.items():
            for subject, stamp in subject_stamps:
                key = (run_dir, stamp)
                run_row = rows_by_key.get(key)
                if run_row is None:
                    missing.append({"run_dir": run_dir, "stamp": stamp, "model_scope": model_scope, "subject": subject})
                    continue
                included_keys.add(key)
                selection_row = {
                    "row_id": "RP_4_CROSS" if model_scope == "Cross-subject" else "RP_4_PER",
                    "deliverable": "Research paper",
                    "task": "4-gesture",
                    "model_scope": model_scope,
                    "condition": "current_cnn_pipeline",
                    "subject": subject,
                    "scenario_name": "highway_overtake" if run_dir == "highway_overtake_eval" else "lane_keep_5min",
                    "scenario_label": _pretty_scenario(run_dir),
                    "stamp": stamp,
                    "run_dir": run_dir,
                    "carla_log": run_row.get("carla_log", ""),
                    "realtime_log": run_row.get("realtime_log", ""),
                    "latency_ok": run_row.get("latency_ok", ""),
                    "latency_json": run_row.get("latency_json", ""),
                    "latency_csv": run_row.get("latency_csv", ""),
                    "drive_json": run_row.get("drive_json", ""),
                    "model_filename": ACTIVE_CROSS_SUBJECT_MODEL_NAME if model_scope == "Cross-subject" else ACTIVE_PER_SUBJECT_MODEL_NAME,
                    "selection_reason": _selection_reason(model_scope, subject, run_dir, stamp),
                }
                selection_rows.append(selection_row)
                manifest_rows.append(_manifest_row(selection_row))

    if missing:
        raise ValueError(f"Missing staged run rows for curated research selection: {missing}")

    excluded_rows = []
    for key, run_row in sorted(rows_by_key.items()):
        if key in included_keys:
            continue
        excluded_rows.append(
            {
                "run_dir": key[0],
                "stamp": key[1],
                "reason": EXCLUSION_REASONS.get(key, "Excluded from curated research-paper runtime set."),
                "latency_ok": run_row.get("latency_ok", ""),
                "drive_json": run_row.get("drive_json", ""),
            }
        )

    selection_rows.sort(key=lambda row: (row["model_scope"], row["scenario_name"], row["stamp"]))
    manifest_rows.sort(key=lambda row: (row["model_scope"], row["latency_json"]))

    _write_csv(Path(args.output_selection), selection_rows, SELECTION_COLUMNS)
    _write_csv(Path(args.output_manifest), manifest_rows, MANIFEST_COLUMNS)

    audit = {
        "source_run_index": str(args.run_index),
        "selection_counts": {
            model_scope: {
                run_dir: len(stamps)
                for run_dir, stamps in scenario_map.items()
            }
            for model_scope, scenario_map in SELECTIONS.items()
        },
        "included_rows": selection_rows,
        "excluded_rows": excluded_rows,
    }
    Path(args.output_audit).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_audit).write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(f"Saved curated runtime selection to {args.output_selection}")
    print(f"Saved research manifest to {args.output_manifest}")
    print(f"Saved audit summary to {args.output_audit}")


if __name__ == "__main__":
    main()
