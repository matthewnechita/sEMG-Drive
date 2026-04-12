from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval_metrics.pipeline_scripts.harvest_model_metrics import load_bundle


CORE_METRICS = [
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "worst_class_recall",
]


def _load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _scope_order(value: str) -> int:
    text = str(value or "").strip().lower()
    if text == "cross_subject":
        return 0
    if text == "per_subject":
        return 1
    return 2


def _arm_order(value: str) -> int:
    text = str(value or "").strip().lower()
    if text == "left":
        return 0
    if text == "right":
        return 1
    return 2


def _latest_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    latest: dict[tuple[str, str, str, str], dict[str, str]] = {}
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


def _arm_label(value: str) -> str:
    text = str(value or "").strip().lower()
    if text == "left":
        return "Left arm"
    if text == "right":
        return "Right arm"
    return text.title() or "Arm"


def _scope_label(value: str) -> str:
    text = str(value or "").strip().lower()
    if text == "cross_subject":
        return "Cross-subject"
    if text == "per_subject":
        return "Per-subject"
    return text.replace("_", " ").title() or "Model"


def _display_label(scope: str, arm: str, subject: str) -> str:
    scope_text = _scope_label(scope)
    arm_text = _arm_label(arm)
    subject_text = str(subject or "").strip()
    if scope == "per_subject" and subject_text:
        return f"{scope_text} ({subject_text}) | {arm_text}"
    return f"{scope_text} | {arm_text}"


def _extract_fold_metrics(metadata: dict[str, object]) -> tuple[list[dict[str, float]], str]:
    cross_validation = metadata.get("cross_validation")
    if isinstance(cross_validation, dict):
        folds = cross_validation.get("folds")
        if isinstance(folds, list):
            metrics = [
                fold_metrics
                for fold in folds
                if isinstance(fold, dict)
                and isinstance((fold_metrics := fold.get("metrics")), dict)
            ]
            if metrics:
                return metrics, "Grouped CV folds"

    zero_shot_loso = metadata.get("zero_shot_loso")
    if not isinstance(zero_shot_loso, dict):
        evaluation = metadata.get("evaluation")
        if isinstance(evaluation, dict):
            zero_shot_loso = evaluation.get("zero_shot_loso")
    if isinstance(zero_shot_loso, dict):
        folds = zero_shot_loso.get("folds")
        if isinstance(folds, list):
            metrics = [
                fold_metrics
                for fold in folds
                if isinstance(fold, dict)
                and isinstance((fold_metrics := fold.get("metrics")), dict)
            ]
            if metrics:
                return metrics, "LOSO folds"

    return [], ""


def load_current_model_entries(input_csv: Path) -> list[dict[str, object]]:
    rows = _latest_rows(_load_rows(input_csv))
    rows.sort(
        key=lambda row: (
            _scope_order(str(row.get("bundle_scope") or "")),
            _arm_order(str(row.get("arm") or "")),
            str(row.get("subject") or row.get("target_subject") or "").strip().lower(),
        )
    )

    entries: list[dict[str, object]] = []
    for row in rows:
        bundle_path = Path(str(row.get("path") or "").strip())
        if not bundle_path.exists():
            continue

        obj = load_bundle(bundle_path)
        metadata = obj.get("metadata", {}) if isinstance(obj, dict) else {}
        if not isinstance(metadata, dict):
            metadata = {}
        metrics = metadata.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}

        scope = str(row.get("bundle_scope") or "").strip().lower()
        arm = str(row.get("arm") or "").strip().lower()
        subject = str(row.get("subject") or row.get("target_subject") or "").strip()
        fold_metrics, fold_source = _extract_fold_metrics(metadata)

        entries.append(
            {
                "path": bundle_path,
                "scope": scope,
                "arm": arm,
                "subject": subject,
                "display_label": _display_label(scope, arm, subject),
                "metrics": metrics,
                "fold_metrics": fold_metrics,
                "fold_source": fold_source,
            }
        )

    if not entries:
        raise ValueError(f"No current model entries could be loaded from {input_csv}")
    return entries
