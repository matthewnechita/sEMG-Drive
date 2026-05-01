from __future__ import annotations

import argparse
import inspect
import math
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings(
    "ignore",
    message="The PCA initialization in TSNE will change",
    category=FutureWarning,
)

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from emg.gesture_model_cnn import GestureModelBundle, load_gesture_bundle
from emg.strict_layout import resolve_strict_indices_from_metadata
from eval_metrics.config import CURRENT_METRICS_ROOT, FIGURES_ROOT
from eval_metrics.plot_scripts.model_plot_utils import load_current_model_entries


MAX_POINTS_PER_GESTURE = 180
EMBEDDING_BATCH_SIZE = 1024
RANDOM_STATE = 42
DEFAULT_MVC_MIN_RATIO = 1.5
STRICT_RESAMPLED_ROOT = Path("data_resampled_strict")

GESTURE_COLORS = {
    "neutral": "#64748b",
    "left_turn": "#2563eb",
    "right_turn": "#f97316",
    "horn": "#dc2626",
}


def _gesture_color(label: str) -> str:
    return GESTURE_COLORS.get(str(label), "#0f172a")


def _subject_from_path(path: Path) -> str:
    return Path(path).parent.parent.name


def _strict_arm_root(root: Path, arm: str) -> Path:
    arm_name = str(arm).strip().lower()
    if arm_name not in {"left", "right"}:
        raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
    return Path(root) / f"{arm_name} arm"


def _strict_filtered_dir(root: Path, arm: str, subject: str) -> Path:
    subject_name = str(subject).strip()
    if not subject_name:
        raise ValueError("subject must not be empty.")
    return _strict_arm_root(root, arm) / subject_name / "filtered"


def _clean_label(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except Exception:
            return None
    if isinstance(value, np.str_):
        value = str(value)
    if isinstance(value, str):
        value = value.strip()
    if value == "":
        return None
    return str(value)


def _majority_label_with_confidence(segment: np.ndarray) -> tuple[str | None, float]:
    if segment.size == 0:
        return None, 0.0
    cleaned = [
        label for label in (_clean_label(item) for item in np.asarray(segment, dtype=object).reshape(-1))
        if label is not None
    ]
    if not cleaned:
        return None, 0.0
    values, counts = np.unique(np.asarray(cleaned, dtype=object), return_counts=True)
    idx = int(np.argmax(counts))
    total = int(np.sum(counts))
    return str(values[idx]), (float(counts[idx] / total) if total > 0 else 0.0)


def _compute_calibration(
    neutral_emg: np.ndarray,
    mvc_emg: np.ndarray,
    percentile: float,
    mvc_min_ratio: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    neutral = np.asarray(neutral_emg, dtype=float)
    mvc = np.asarray(mvc_emg, dtype=float)
    if neutral.size == 0 or mvc.size == 0:
        return None, None
    neutral_rms = np.sqrt(np.mean(neutral**2, axis=0))
    mvc_rms = np.sqrt(np.mean(mvc**2, axis=0))
    ratio = np.ones_like(mvc_rms, dtype=float)
    np.divide(mvc_rms, neutral_rms, out=ratio, where=neutral_rms >= 1e-9)
    if float(np.median(ratio)) < float(mvc_min_ratio):
        return None, None
    neutral_mean = np.mean(neutral, axis=0)
    mvc_scale = np.percentile(mvc, float(percentile), axis=0)
    mvc_scale = np.where(mvc_scale < 1e-6, 1.0, mvc_scale)
    return neutral_mean, mvc_scale


def _window_emg(emg: np.ndarray, window_size: int, window_step: int) -> np.ndarray:
    sample_count, channel_count = emg.shape
    if sample_count < window_size:
        return np.empty((0, channel_count, window_size), dtype=np.float32)
    n_windows = 1 + (sample_count - window_size) // window_step
    shape = (n_windows, window_size, channel_count)
    strides = (
        emg.strides[0] * int(window_step),
        emg.strides[0],
        emg.strides[1],
    )
    windows = np.lib.stride_tricks.as_strided(emg, shape=shape, strides=strides)
    return np.transpose(windows, (0, 2, 1)).astype(np.float32, copy=True)


def _load_windows_from_file(
    path: Path,
    *,
    arm: str,
    window_size: int,
    window_step: int,
    use_calibration: bool,
    mvc_percentile: float,
    mvc_min_ratio: float,
    use_min_label_confidence: bool,
    min_label_confidence: float,
    included_gestures: set[str] | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    data = np.load(path, allow_pickle=True)
    if "emg" not in data.files or "y" not in data.files:
        return None

    emg = np.asarray(data["emg"], dtype=float)
    strict_layout = resolve_strict_indices_from_metadata(data.get("metadata"), arm=arm)
    if emg.shape[1] != strict_layout.ordered_indices.size:
        raise ValueError(
            f"{path.name}: strict layout resolved {strict_layout.ordered_indices.size} channels for {arm}, "
            f"but file has {emg.shape[1]}."
        )
    emg = emg[:, strict_layout.ordered_indices]

    if use_calibration:
        calib_neutral = data.get("calib_neutral_emg")
        calib_mvc = data.get("calib_mvc_emg")
        if calib_neutral is not None and calib_mvc is not None:
            calib_neutral = np.asarray(calib_neutral, dtype=float)[:, strict_layout.ordered_indices]
            calib_mvc = np.asarray(calib_mvc, dtype=float)[:, strict_layout.ordered_indices]
            neutral_mean, mvc_scale = _compute_calibration(
                calib_neutral,
                calib_mvc,
                mvc_percentile,
                mvc_min_ratio,
            )
            if neutral_mean is not None and mvc_scale is not None:
                emg = (emg - neutral_mean) / mvc_scale

    windows = _window_emg(emg, int(window_size), int(window_step))
    if windows.size == 0:
        return None

    raw_labels = np.asarray(data["y"], dtype=object)
    starts = np.arange(int(windows.shape[0])) * int(window_step)
    ends = starts + int(window_size)

    window_labels: list[str | None] = []
    for start, end in zip(starts, ends):
        label, confidence = _majority_label_with_confidence(raw_labels[start:end])
        if label == "neutral_buffer":
            label = None
        if (
            use_min_label_confidence
            and label is not None
            and float(confidence) < float(min_label_confidence)
        ):
            label = None
        if label is not None and included_gestures is not None and label not in included_gestures:
            label = None
        window_labels.append(label)

    label_array = np.asarray(window_labels, dtype=object)
    keep = label_array != None  # noqa: E711
    if not np.any(keep):
        return None
    return windows[keep], label_array[keep]


def _resolve_source_files(
    *,
    scope: str,
    arm: str,
    subject: str,
    metadata: dict,
) -> tuple[list[Path], str]:
    arm_root = _strict_arm_root(STRICT_RESAMPLED_ROOT, arm)
    if scope == "per_subject" and subject:
        search_root = _strict_filtered_dir(STRICT_RESAMPLED_ROOT, arm, subject)
    else:
        search_root = arm_root

    all_files = sorted(search_root.rglob("*_filtered.npz"))
    by_name = {path.name.lower(): path for path in all_files}

    preferred_names = []
    source_label = "source files"
    test_files = metadata.get("test_files")
    train_files = metadata.get("train_files")
    if scope == "cross_subject" and isinstance(test_files, list) and test_files:
        preferred_names = [str(name).strip() for name in test_files if str(name).strip()]
        source_label = "held-out files"
    elif isinstance(train_files, list) and train_files:
        preferred_names = [str(name).strip() for name in train_files if str(name).strip()]
        source_label = "maintained files"

    selected: list[Path] = []
    if preferred_names:
        seen = set()
        for name in preferred_names:
            path = by_name.get(name.lower())
            if path is None:
                continue
            key = str(path).lower()
            if key in seen:
                continue
            seen.add(key)
            selected.append(path)
        if selected:
            return selected, source_label

    excluded_subjects = {
        str(value).strip().lower()
        for value in (metadata.get("excluded_subjects") or [])
        if str(value).strip()
    }
    fallback = [
        path
        for path in all_files
        if _subject_from_path(path).strip().lower() not in excluded_subjects
    ]
    if not fallback:
        raise FileNotFoundError(f"No filtered source files found under {search_root}")
    if scope == "cross_subject":
        return fallback, "pooled files"
    return fallback, "subject files"


def _load_windows_for_bundle(
    *,
    scope: str,
    arm: str,
    subject: str,
    bundle: GestureModelBundle,
) -> tuple[np.ndarray, np.ndarray, str]:
    metadata = bundle.metadata if isinstance(bundle.metadata, dict) else {}
    files, source_label = _resolve_source_files(
        scope=scope,
        arm=arm,
        subject=subject,
        metadata=metadata,
    )

    included_gestures_raw = metadata.get("included_gestures")
    included_gestures = None
    if isinstance(included_gestures_raw, (list, tuple, set)):
        included_gestures = {str(value) for value in included_gestures_raw}

    confidence_cfg = metadata.get("label_confidence_filter")
    if not isinstance(confidence_cfg, dict):
        confidence_cfg = {}

    windows_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    for path in files:
        loaded = _load_windows_from_file(
            path,
            arm=arm,
            window_size=int(metadata.get("window_size_samples", 200)),
            window_step=int(metadata.get("window_step_samples", 100)),
            use_calibration=bool(metadata.get("calibration_used", False)),
            mvc_percentile=float(metadata.get("calibration_mvc_percentile", 95.0)),
            mvc_min_ratio=DEFAULT_MVC_MIN_RATIO,
            use_min_label_confidence=bool(confidence_cfg.get("enabled", False)),
            min_label_confidence=float(confidence_cfg.get("min_label_confidence", 0.0)),
            included_gestures=included_gestures,
        )
        if loaded is None:
            continue
        windows, labels = loaded
        windows_list.append(windows)
        labels_list.append(labels)

    if not windows_list:
        raise ValueError(f"No windowed EMG data could be loaded for {scope} {arm} {subject}")

    return np.vstack(windows_list), np.concatenate(labels_list), source_label


def _sample_balanced_windows(
    X: np.ndarray,
    y: np.ndarray,
    *,
    label_order: list[str],
    max_points_per_gesture: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(random_state))
    selected_indices: list[np.ndarray] = []
    for label in label_order:
        label_idx = np.flatnonzero(y == label)
        if label_idx.size == 0:
            continue
        if label_idx.size > max_points_per_gesture:
            label_idx = rng.choice(label_idx, size=max_points_per_gesture, replace=False)
        selected_indices.append(np.sort(label_idx))
    if not selected_indices:
        raise ValueError("No label windows were available after balanced sampling.")
    keep = np.concatenate(selected_indices)
    return X[keep], y[keep]


def _extract_embeddings(bundle: GestureModelBundle, X: np.ndarray) -> np.ndarray:
    model = bundle.model
    model.eval()
    device = next(model.parameters()).device
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, int(X.shape[0]), EMBEDDING_BATCH_SIZE):
            stop = start + EMBEDDING_BATCH_SIZE
            xb = torch.from_numpy(bundle.standardize(X[start:stop])).to(device)
            embedding = model.extract_embedding(xb, l2_normalize=True)
            chunks.append(embedding.cpu().numpy())
    if not chunks:
        raise ValueError("No embeddings were produced.")
    return np.vstack(chunks)


def _project_embedding_2d(embedding: np.ndarray) -> np.ndarray:
    if embedding.shape[0] < 3:
        return PCA(n_components=2).fit_transform(embedding)

    working = embedding
    if embedding.shape[1] > 30 and embedding.shape[0] > 30:
        working = PCA(
            n_components=min(30, int(embedding.shape[1]), int(embedding.shape[0] - 1))
        ).fit_transform(embedding)

    if working.shape[0] < 10:
        return PCA(n_components=2).fit_transform(working)

    perplexity = min(30.0, max(5.0, (float(working.shape[0]) - 1.0) / 3.0))
    tsne_kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate=200.0,
        random_state=RANDOM_STATE,
    )
    if "max_iter" in inspect.signature(TSNE.__init__).parameters:
        tsne_kwargs["max_iter"] = 1000
    else:
        tsne_kwargs["n_iter"] = 1000
    tsne = TSNE(
        **tsne_kwargs,
    )
    return tsne.fit_transform(working)


def _class_centroids(projection: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    centroids: dict[str, np.ndarray] = {}
    for raw_label in np.unique(np.asarray(labels, dtype=object)):
        label = str(raw_label)
        mask = np.asarray(labels == raw_label, dtype=bool)
        if not np.any(mask):
            continue
        centroids[label] = np.mean(np.asarray(projection[mask], dtype=float), axis=0)
    return centroids


def _orient_projection_for_readability(projection: np.ndarray, labels: np.ndarray) -> np.ndarray:
    points = np.asarray(projection, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        return points

    centered = points - np.mean(points, axis=0, keepdims=True)
    best = centered
    best_score = -float("inf")

    for swap_axes in (False, True):
        base = centered[:, [1, 0]] if swap_axes else centered.copy()
        for flip_x in (-1.0, 1.0):
            for flip_y in (-1.0, 1.0):
                candidate = base.copy()
                candidate[:, 0] *= flip_x
                candidate[:, 1] *= flip_y
                centroids = _class_centroids(candidate, labels)

                score = 0.0
                left = centroids.get("left_turn")
                right = centroids.get("right_turn")
                horn = centroids.get("horn")
                neutral = centroids.get("neutral")

                if left is not None:
                    score += float(-left[0]) - 0.15 * float(abs(left[1]))
                if right is not None:
                    score += float(right[0]) - 0.15 * float(abs(right[1]))
                if horn is not None:
                    score += float(horn[1]) - 0.15 * float(abs(horn[0]))
                if neutral is not None:
                    score += float(-neutral[1]) - 0.15 * float(abs(neutral[0]))

                if left is not None and right is not None:
                    score += 0.5 * float(right[0] - left[0])
                if horn is not None and neutral is not None:
                    score += 0.5 * float(horn[1] - neutral[1])

                if score > best_score:
                    best_score = score
                    best = candidate

    return best


def _panel_title(entry: dict[str, object], n_points: int, balance_accuracy: float | None) -> str:
    line1 = str(entry["display_label"])
    line2 = f"sampled windows={n_points}"
    if balance_accuracy is None:
        return f"{line1}\n{line2}"
    return f"{line1}\n{line2}\nbalanced accuracy {balance_accuracy * 100.0:.1f}%"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot gesture-cluster embeddings from the current maintained model set."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=CURRENT_METRICS_ROOT / "model_metrics.csv",
        help="Current harvested model metrics CSV.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=FIGURES_ROOT / "gesture_embedding_clusters.png",
        help="Path to save the output PNG.",
    )
    parser.add_argument(
        "--title",
        default="Gesture Embedding Clusters (t-SNE)",
        help="Optional figure title.",
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    entries = load_current_model_entries(Path(args.input_csv))

    panels = []
    for entry in entries:
        bundle = load_gesture_bundle(Path(entry["path"]), device="cpu")
        X_all, y_all, source_label = _load_windows_for_bundle(
            scope=str(entry["scope"]),
            arm=str(entry["arm"]),
            subject=str(entry["subject"]),
            bundle=bundle,
        )
        label_order = [
            str(bundle.index_to_label[idx])
            for idx in sorted(bundle.index_to_label.keys())
            if np.any(y_all == str(bundle.index_to_label[idx]))
        ]
        X_plot, y_plot = _sample_balanced_windows(
            X_all,
            y_all,
            label_order=label_order,
            max_points_per_gesture=MAX_POINTS_PER_GESTURE,
            random_state=RANDOM_STATE,
        )
        embedding = _extract_embeddings(bundle, X_plot)
        projection = _project_embedding_2d(embedding)
        projection = _orient_projection_for_readability(projection, y_plot)
        panels.append(
            {
                "entry": entry,
                "projection": projection,
                "labels": y_plot,
                "label_order": label_order,
                "source_label": source_label,
                "n_points": int(X_plot.shape[0]),
                "balanced_accuracy": entry["metrics"].get("balanced_accuracy"),
            }
        )

    if not panels:
        raise ValueError("No model embedding panels could be generated.")

    ncols = 2 if len(panels) > 1 else 1
    nrows = int(math.ceil(len(panels) / ncols))

    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.8 * ncols, 5.4 * nrows),
        constrained_layout=True,
    )
    axes_list = np.atleast_1d(axes).flatten().tolist()

    legend_handles = {}
    for ax, panel in zip(axes_list, panels):
        projection = np.asarray(panel["projection"], dtype=float)
        labels = np.asarray(panel["labels"], dtype=object)
        label_order = list(panel["label_order"])
        for label in label_order:
            mask = labels == label
            if not np.any(mask):
                continue
            scatter = ax.scatter(
                projection[mask, 0],
                projection[mask, 1],
                s=15,
                alpha=0.68,
                color=_gesture_color(label),
                edgecolors="none",
                label=label.replace("_", " ").title(),
            )
            legend_handles[label] = scatter

        ax.set_title(
            _panel_title(
                panel["entry"],
                int(panel["n_points"]),
                float(panel["balanced_accuracy"]) if panel["balanced_accuracy"] is not None else None,
            ),
            fontsize=11.5,
            fontweight="bold",
        )
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(alpha=0.16)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes_list[len(panels):]:
        ax.axis("off")

    fig.suptitle(args.title.strip() or "Gesture Embedding Clusters (t-SNE)", fontsize=15, fontweight="bold")
    if legend_handles:
        ordered_labels = [
            label for label in ("neutral", "left_turn", "right_turn", "horn") if label in legend_handles
        ]
        fig.legend(
            [legend_handles[label] for label in ordered_labels],
            [label.replace("_", " ").title() for label in ordered_labels],
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            ncol=1,
            frameon=False,
        )

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=220, bbox_inches="tight")
    print(f"Saved figure to {args.output_png}")


if __name__ == "__main__":
    main()
