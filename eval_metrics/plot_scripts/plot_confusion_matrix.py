from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval_metrics.config import ACTIVE_OFFLINE_MODEL_NAMES, CURRENT_METRICS_ROOT, FIGURES_ROOT, MODELS_ROOT
from eval_metrics.pipeline_scripts.harvest_model_metrics import load_bundle


def _normalize_label_map(raw_map) -> dict[int, str]:
    if not isinstance(raw_map, dict):
        return {}
    out: dict[int, str] = {}
    for key, value in raw_map.items():
        try:
            idx = int(key)
        except (TypeError, ValueError):
            continue
        out[idx] = str(value)
    return out


def _arm_from_path(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "left" in parts:
        return "left"
    if "right" in parts:
        return "right"
    return ""


def _scope_label(stream: str) -> str:
    text = str(stream or "").strip().lower()
    if "cross" in text:
        return "Cross-subject"
    if "per_subject" in text or "single" in text:
        return "Per-subject"
    return "Model"


def _arm_label(arm: str) -> str:
    text = str(arm or "").strip().lower()
    if text == "left":
        return "Left arm"
    if text == "right":
        return "Right arm"
    return text.title() or "Arm"


def _find_bundle_paths(models_root: Path, bundle_paths: list[str], requested_names: list[str]) -> list[Path]:
    if bundle_paths:
        paths = [Path(path) for path in bundle_paths]
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(path)
        return paths

    requested = {str(name).strip().lower() for name in requested_names if str(name).strip()}
    paths = []
    for path in sorted(models_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".pt", ".pkl"}:
            continue
        if requested:
            name = path.name.lower()
            stem = path.stem.lower()
            if name not in requested and stem not in requested:
                continue
        paths.append(path)
    if not paths:
        if requested:
            raise FileNotFoundError(
                f"No bundles found under {models_root} matching requested name(s): {', '.join(sorted(requested))}"
            )
        raise FileNotFoundError(f"No bundles found under {models_root}")
    return paths


def _paths_from_model_metrics(path: Path) -> list[Path]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    latest_rows = {}
    for row in rows:
        key = (
            str(row.get("bundle_scope") or "").strip().lower(),
            str(row.get("arm") or "").strip().lower(),
            str(row.get("subject") or row.get("target_subject") or "").strip().lower(),
            str(row.get("gesture_bucket") or "").strip().lower(),
        )
        created_at = str(row.get("created_at") or "").strip()
        current = latest_rows.get(key)
        if current is None or created_at > str(current.get("created_at") or "").strip():
            latest_rows[key] = row

    paths = []
    for row in latest_rows.values():
        bundle_path = Path(str(row.get("path") or "").strip())
        if bundle_path.exists():
            paths.append(bundle_path)
    deduped = []
    seen = set()
    for path in paths:
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _bundle_entry(path: Path, normalize: str) -> dict[str, object]:
    obj = load_bundle(path)
    metadata = obj.get("metadata", {}) if isinstance(obj, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    metrics = metadata.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    index_to_label = _normalize_label_map(obj.get("index_to_label") or metadata.get("index_to_label"))
    label_names = [label for _, label in sorted(index_to_label.items())]
    if not label_names:
        raw_labels = metadata.get("labels")
        if isinstance(raw_labels, (list, tuple)):
            label_names = [str(label) for label in raw_labels]
    if not label_names:
        raise ValueError(f"Could not resolve labels for {path}")

    matrix_key = {
        "row": "confusion_matrix_row_norm",
        "col": "confusion_matrix_col_norm",
        "counts": "confusion_matrix_counts",
    }[normalize]
    matrix = np.asarray(metrics.get(matrix_key), dtype=float)
    counts = np.asarray(metrics.get("confusion_matrix_counts"), dtype=float)

    expected_shape = (len(label_names), len(label_names))
    if matrix.shape != expected_shape:
        raise ValueError(
            f"{path} has confusion matrix shape {matrix.shape}, expected {expected_shape} from label count."
        )
    if counts.shape != expected_shape:
        counts = np.full(expected_shape, np.nan, dtype=float)

    arm = _arm_from_path(path)
    scope = _scope_label(metadata.get("stream"))
    subject = str(metadata.get("subject") or metadata.get("target_subject") or "").strip()
    title_parts = [scope]
    if arm:
        title_parts.append(_arm_label(arm))
    if subject:
        title_parts.append(subject)

    return {
        "path": path,
        "matrix": matrix,
        "counts": counts,
        "labels": label_names,
        "title": " | ".join(title_parts),
        "balanced_accuracy": float(metrics.get("balanced_accuracy")) if metrics.get("balanced_accuracy") is not None else None,
        "macro_f1": float(metrics.get("macro_f1")) if metrics.get("macro_f1") is not None else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot report-ready confusion matrix heatmaps from saved model bundles."
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=MODELS_ROOT,
        help="Root to scan when --bundle is not provided.",
    )
    parser.add_argument(
        "--bundle",
        action="append",
        default=[],
        help="Optional explicit bundle path. Repeatable.",
    )
    parser.add_argument(
        "--model-name",
        action="append",
        default=[],
        help="Optional exact bundle filename or stem to match under --models-root. Repeatable.",
    )
    parser.add_argument(
        "--model-metrics",
        type=Path,
        default=CURRENT_METRICS_ROOT / "model_metrics.csv",
        help=(
            "Harvested model metrics CSV. If no --bundle or --model-name is passed, "
            "bundle paths are read from this file first."
        ),
    )
    parser.add_argument(
        "--normalize",
        choices=["row", "col", "counts"],
        default="row",
        help="Which confusion matrix variant to visualize.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=FIGURES_ROOT / "confusion_matrix_row_norm.png",
        help="Path to save the PNG figure.",
    )
    parser.add_argument(
        "--title",
        default="Confusion Matrix",
        help="Optional figure title.",
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    requested_names = list(args.model_name)
    if not requested_names:
        requested_names = [str(name).strip() for name in ACTIVE_OFFLINE_MODEL_NAMES if str(name).strip()]
    if args.bundle or requested_names:
        paths = _find_bundle_paths(Path(args.models_root), list(args.bundle), requested_names)
    else:
        paths = _paths_from_model_metrics(Path(args.model_metrics))
        if not paths:
            paths = _find_bundle_paths(Path(args.models_root), list(args.bundle), list(args.model_name))
    entries = [_bundle_entry(path, args.normalize) for path in paths]
    entries.sort(key=lambda item: str(item["title"]))

    n = len(entries)
    ncols = 2 if n > 1 else 1
    nrows = int(math.ceil(n / ncols))

    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.2 * ncols, 5.6 * nrows),
        constrained_layout=True,
    )
    axes_list = np.atleast_1d(axes).flatten().tolist()

    if args.normalize == "counts":
        vmax = max(float(np.nanmax(entry["matrix"])) for entry in entries)
        colorbar_label = "Count"
    else:
        vmax = 100.0
        colorbar_label = "Percent"

    image = None
    for ax, entry in zip(axes_list, entries):
        matrix = np.asarray(entry["matrix"], dtype=float)
        counts = np.asarray(entry["counts"], dtype=float)
        if args.normalize == "counts":
            display = matrix
        else:
            display = matrix * 100.0

        image = ax.imshow(display, cmap="Blues", vmin=0.0, vmax=vmax)
        labels = list(entry["labels"])
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

        subtitle_parts = []
        balanced_accuracy = entry.get("balanced_accuracy")
        macro_f1 = entry.get("macro_f1")
        if isinstance(balanced_accuracy, float):
            subtitle_parts.append(f"Bal acc {balanced_accuracy * 100.0:.1f}%")
        if isinstance(macro_f1, float):
            subtitle_parts.append(f"Macro F1 {macro_f1 * 100.0:.1f}%")
        subtitle = " | ".join(subtitle_parts)
        ax.set_title(f"{entry['title']}\n{subtitle}" if subtitle else str(entry["title"]))

        threshold = vmax * 0.55
        for row_idx in range(display.shape[0]):
            for col_idx in range(display.shape[1]):
                value = display[row_idx, col_idx]
                if args.normalize == "counts":
                    text = f"{int(round(value))}"
                else:
                    count_text = ""
                    if not np.isnan(counts[row_idx, col_idx]):
                        count_text = f"\n({int(round(counts[row_idx, col_idx]))})"
                    text = f"{value:.1f}%{count_text}"
                ax.text(
                    col_idx,
                    row_idx,
                    text,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if value >= threshold else "#1f2937",
                    fontweight="bold" if row_idx == col_idx else "normal",
                )

    for ax in axes_list[len(entries):]:
        ax.axis("off")

    fig.suptitle(args.title.strip() or "Confusion Matrix", fontsize=15, fontweight="bold")
    if image is not None:
        fig.colorbar(image, ax=axes_list[:len(entries)], shrink=0.86, label=colorbar_label)

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=220, bbox_inches="tight")
    print(f"Saved figure to {args.output_png}")


if __name__ == "__main__":
    main()
