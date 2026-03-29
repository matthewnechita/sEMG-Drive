from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True).astype(float)
    out = np.zeros_like(matrix, dtype=float)
    np.divide(matrix, row_sums, out=out, where=row_sums != 0)
    return out


def normalize_cols(matrix: np.ndarray) -> np.ndarray:
    col_sums = matrix.sum(axis=0, keepdims=True).astype(float)
    out = np.zeros_like(matrix, dtype=float)
    np.divide(matrix, col_sums, out=out, where=col_sums != 0)
    return out


def print_confusion_matrix(
    title: str,
    matrix: np.ndarray,
    label_names: list[str],
    *,
    as_percent: bool,
) -> None:
    row_w = max(8, max(len(name) for name in label_names))
    cell_w = max(8, max(len(name) for name in label_names))
    header = " " * (row_w + 3) + " ".join(f"{name:>{cell_w}}" for name in label_names)
    print(f"\n{title}")
    print(header)
    for index, name in enumerate(label_names):
        if as_percent:
            row = " ".join(f"{(100.0 * float(value)):>{cell_w}.1f}" for value in matrix[index])
        else:
            row = " ".join(f"{int(value):>{cell_w}d}" for value in matrix[index])
        print(f"{name:>{row_w}} | {row}")


def compute_eval_artifacts(y_true, y_pred, index_to_label) -> dict[str, Any]:
    label_indices = list(range(len(index_to_label)))
    label_names = [index_to_label[index] for index in label_indices]

    report_text = classification_report(
        y_true,
        y_pred,
        labels=label_indices,
        target_names=label_names,
        zero_division=0,
    )
    report_dict_raw = classification_report(
        y_true,
        y_pred,
        labels=label_indices,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    if not isinstance(report_dict_raw, dict):
        raise TypeError("classification_report(output_dict=True) did not return a dict.")
    report_dict: dict[str, Any] = {str(key): value for key, value in report_dict_raw.items()}

    cm_counts = confusion_matrix(y_true, y_pred, labels=label_indices)
    cm_row_norm = normalize_rows(cm_counts)
    cm_col_norm = normalize_cols(cm_counts)

    per_class = []
    for name in label_names:
        stats_obj = report_dict.get(name, {})
        stats = stats_obj if isinstance(stats_obj, dict) else {}
        precision = float(stats.get("precision", 0.0))
        recall = float(stats.get("recall", 0.0))
        f1 = float(stats.get("f1-score", 0.0))
        per_class.append(
            {
                "label": name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "pr_gap": abs(precision - recall),
            }
        )

    worst_recall = min(per_class, key=lambda item: item["recall"]) if per_class else None
    confusion_to_neutral_rate = {}
    neutral_prediction_fp_rate = None
    if "neutral" in label_names:
        neutral_idx = label_names.index("neutral")
        for index, name in enumerate(label_names):
            if index == neutral_idx:
                continue
            row_total = int(cm_counts[index].sum())
            rate = float(cm_counts[index, neutral_idx] / row_total) if row_total > 0 else 0.0
            confusion_to_neutral_rate[name] = rate

        pred_neutral_total = int(cm_counts[:, neutral_idx].sum())
        neutral_tp = int(cm_counts[neutral_idx, neutral_idx])
        neutral_fp = pred_neutral_total - neutral_tp
        neutral_prediction_fp_rate = (
            float(neutral_fp / pred_neutral_total) if pred_neutral_total > 0 else 0.0
        )

    macro_avg_obj = report_dict.get("macro avg", {})
    macro_avg = macro_avg_obj if isinstance(macro_avg_obj, dict) else {}
    weighted_avg_obj = report_dict.get("weighted avg", {})
    weighted_avg = weighted_avg_obj if isinstance(weighted_avg_obj, dict) else {}

    return {
        "classification_report_text": report_text,
        "classification_report_dict": report_dict,
        "confusion_matrix_counts": cm_counts,
        "confusion_matrix_row_norm": cm_row_norm,
        "confusion_matrix_col_norm": cm_col_norm,
        "per_class": per_class,
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_avg.get("precision", 0.0)),
        "macro_recall": float(macro_avg.get("recall", 0.0)),
        "macro_f1": float(macro_avg.get("f1-score", 0.0)),
        "weighted_precision": float(weighted_avg.get("precision", 0.0)),
        "weighted_recall": float(weighted_avg.get("recall", 0.0)),
        "weighted_f1": float(weighted_avg.get("f1-score", 0.0)),
        "worst_class_recall_label": worst_recall["label"] if worst_recall else None,
        "worst_class_recall": worst_recall["recall"] if worst_recall else None,
        "max_pr_gap_label": max(per_class, key=lambda item: item["pr_gap"])["label"] if per_class else None,
        "max_pr_gap": max((item["pr_gap"] for item in per_class), default=None),
        "confusion_to_neutral_rate": confusion_to_neutral_rate,
        "neutral_prediction_fp_rate": neutral_prediction_fp_rate,
    }


def serialize_eval_metrics(eval_artifacts: dict[str, Any]) -> dict[str, Any]:
    return {
        "test_accuracy": float(eval_artifacts["test_accuracy"]),
        "balanced_accuracy": float(eval_artifacts["balanced_accuracy"]),
        "macro_precision": float(eval_artifacts["macro_precision"]),
        "macro_recall": float(eval_artifacts["macro_recall"]),
        "macro_f1": float(eval_artifacts["macro_f1"]),
        "weighted_precision": float(eval_artifacts["weighted_precision"]),
        "weighted_recall": float(eval_artifacts["weighted_recall"]),
        "weighted_f1": float(eval_artifacts["weighted_f1"]),
        "worst_class_recall_label": eval_artifacts["worst_class_recall_label"],
        "worst_class_recall": eval_artifacts["worst_class_recall"],
        "max_precision_recall_gap_label": eval_artifacts["max_pr_gap_label"],
        "max_precision_recall_gap": eval_artifacts["max_pr_gap"],
        "confusion_to_neutral_rate": eval_artifacts["confusion_to_neutral_rate"],
        "neutral_prediction_fp_rate": eval_artifacts["neutral_prediction_fp_rate"],
        "confusion_matrix_counts": eval_artifacts["confusion_matrix_counts"].tolist(),
        "confusion_matrix_row_norm": eval_artifacts["confusion_matrix_row_norm"].tolist(),
        "confusion_matrix_col_norm": eval_artifacts["confusion_matrix_col_norm"].tolist(),
    }


def print_eval_summary(
    title: str,
    accuracy_label: str,
    eval_artifacts: dict[str, Any],
    label_names: list[str],
) -> None:
    accuracy = float(eval_artifacts["test_accuracy"])
    print(f"\n{title}: {accuracy_label} {accuracy:.3f}")
    print("\nReport:\n", eval_artifacts["classification_report_text"])
    print_confusion_matrix(
        "Confusion matrix (counts, rows=true, cols=pred):",
        eval_artifacts["confusion_matrix_counts"],
        label_names,
        as_percent=False,
    )
    print_confusion_matrix(
        "Confusion matrix (row-normalized %, rows=true, cols=pred):",
        eval_artifacts["confusion_matrix_row_norm"],
        label_names,
        as_percent=True,
    )
    print_confusion_matrix(
        "Confusion matrix (col-normalized %, rows=true, cols=pred):",
        eval_artifacts["confusion_matrix_col_norm"],
        label_names,
        as_percent=True,
    )

    print("\nCore metrics:")
    print(f"  balanced_accuracy: {eval_artifacts['balanced_accuracy']:.3f}")
    print(
        f"  macro P/R/F1: "
        f"{eval_artifacts['macro_precision']:.3f} / "
        f"{eval_artifacts['macro_recall']:.3f} / "
        f"{eval_artifacts['macro_f1']:.3f}"
    )
    print(
        f"  weighted P/R/F1: "
        f"{eval_artifacts['weighted_precision']:.3f} / "
        f"{eval_artifacts['weighted_recall']:.3f} / "
        f"{eval_artifacts['weighted_f1']:.3f}"
    )
    if eval_artifacts["worst_class_recall_label"] is not None:
        print(
            f"  worst_class_recall: "
            f"{eval_artifacts['worst_class_recall_label']} = "
            f"{eval_artifacts['worst_class_recall']:.3f}"
        )
    if eval_artifacts["max_pr_gap_label"] is not None:
        print(
            f"  max_precision_recall_gap: "
            f"{eval_artifacts['max_pr_gap_label']} = "
            f"{eval_artifacts['max_pr_gap']:.3f}"
        )
    if eval_artifacts["confusion_to_neutral_rate"]:
        print("  confusion_to_neutral_rate:")
        for label, rate in sorted(eval_artifacts["confusion_to_neutral_rate"].items()):
            print(f"    {label} -> neutral: {rate:.3f}")
    if eval_artifacts["neutral_prediction_fp_rate"] is not None:
        print(
            f"  neutral_prediction_fp_rate: "
            f"{eval_artifacts['neutral_prediction_fp_rate']:.3f}"
        )


def scalar_summary(values) -> dict[str, float] | None:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
