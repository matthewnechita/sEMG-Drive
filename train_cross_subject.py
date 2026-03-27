"""Cross-subject EMG gesture classifier training.

Supports the existing `cnn_v2` path and the metric-learning `metric_tcn`
variant. Zero-shot LOSO remains the baseline cross-subject metric, while the
metric-TCN path can also run a prototype-based calibrated LOSO pass.

Usage:
    python train_cross_subject.py

To run realtime inference with the cross-subject model:
    python realtime_gesture_cnn.py --model models/strict/cross_subject/right/gesture_cnn_v3_3_gestures.pt

DQ note: subject05 is excluded by default (EXCLUDED_SUBJECTS below).
All 4 sessions show MVC/neutral ratio <= 1.8x — the MVC calibration failed,
meaning mvc-normalised signal is near noise level for this subject.
Recollect calibration for subject05 before re-including them.
"""
import datetime as dt
import copy
import time
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from libemg.utils import get_windows
from emg.model_family import (
    ModelFamilyConfig,
    build_architecture_metadata,
    build_family_metadata,
    build_model,
    build_training_objectives,
    compute_training_step,
    prepare_train_eval_inputs,
    standardize_windows,
    supports_prototype_calibration,
    validate_model_family,
)
from emg.prototype_classifier import PrototypeClassifier
from emg.strict_layout import (
    resolve_strict_indices_from_metadata,
    strict_channel_count_for_arm,
    strict_layout_bundle_metadata,
)
from project_paths import STRICT_MODELS_ROOT, STRICT_RESAMPLED_ROOT, strict_arm_root


class SupportsExtractEmbedding(Protocol):
    def extract_embedding(
        self,
        x: torch.Tensor,
        l2_normalize: bool = False,
    ) -> torch.Tensor: ...


# ======== Config ========
ARM        = "left"            # ← set to "right" or "left" before running, lowercase l
DATA_ROOT  = strict_arm_root(STRICT_RESAMPLED_ROOT, ARM)
MODEL_OUT  = STRICT_MODELS_ROOT / "cross_subject" / ARM / "tcn_4_gestures.pt"
PATTERN    = "*_filtered.npz"

WINDOW_SIZE = 200
WINDOW_STEP = 100

USE_CALIBRATION          = True
MVC_PERCENTILE           = 95.0
USE_MIN_LABEL_CONFIDENCE = True
MIN_LABEL_CONFIDENCE     = 0.85

TEST_SIZE    = 0.2
RANDOM_STATE = 42
BATCH_SIZE   = 512

# Cross-subject hyperparameters.
# Keep these shared across model families unless a specific experiment needs
# a model-family-specific override.
EPOCHS  = 40
LR      = 1e-4
DROPOUT = 0.25
LABEL_SMOOTHING = 0.05

# Subject-balanced sampling is always on for cross-subject training.

# Augmentation — wide amplitude range matches inter-subject EMG amplitude
# variance (~5-10x between people due to electrode placement, muscle mass).
USE_AUGMENTATION = True
AMP_RANGE = (0.5, 2.0)   # wide inter-subject amplitude variance range
AUG_PROB  = 0.5
CHANNEL_LAYOUT_MODE = "strict"  # "strict" or "none"

# Subjects with failed MVC calibration or other DQ issues.
# subject05: all 4 sessions have mvc_ratio <= 1.8x (MVC barely > neutral).
# This corrupts the MVC normalisation; recollect before re-including.
EXCLUDED_SUBJECTS: list[str] = []

# Optional gesture filtering (code-only; no CLI flags).
# Set INCLUDED_GESTURES to a subset to train only those labels.
# Example:
# INCLUDED_GESTURES = {"neutral", "left_turn", "right_turn"} example
INCLUDED_GESTURES: set[str] | None = {"neutral", "left_turn", "right_turn", "horn"} # set to = None to include all gestures

# LOSO evaluation must be run before deploying the cross-subject model.
# This measures true cross-subject accuracy (model vs subjects it never trained on).
# Minimum recommended LOSO accuracy before deployment: 65%.
LOSO_EVAL = False
CALIBRATED_LOSO_EVAL = False
TRAIN_FINAL_MODEL = True
MODEL_FAMILY = "metric_tcn"  # "cnn_v2" or "metric_tcn"
METRIC_TCN_CHANNELS = (64, 64, 128, 128)
METRIC_TCN_KERNEL_SIZE = 5
METRIC_TCN_EMBEDDING_DIM = 128
SUPCON_WEIGHT = 0.20
SUPCON_TEMPERATURE = 0.10
PROTOTYPE_TEMPERATURE = 0.20
PROTOTYPE_L2_NORMALIZE = True

# ========================

MODEL_FAMILY = validate_model_family(MODEL_FAMILY)
FAMILY_CFG = ModelFamilyConfig(
    model_family=MODEL_FAMILY,
    metric_tcn_channels=METRIC_TCN_CHANNELS,
    metric_tcn_kernel_size=METRIC_TCN_KERNEL_SIZE,
    metric_tcn_embedding_dim=METRIC_TCN_EMBEDDING_DIM,
    supcon_weight=SUPCON_WEIGHT,
    supcon_temperature=SUPCON_TEMPERATURE,
)


# ── Label utilities ──────────────────────────────────────────────────────────

def majority_label_with_confidence(segment):
    if segment.size == 0:
        return None, 0.0
    flat = segment.reshape(-1)
    if flat.dtype == object:
        cleaned = []
        for x in flat:
            if x is None:
                continue
            if isinstance(x, bytes):
                try:
                    x = x.decode("utf-8")
                except Exception:
                    continue
            if isinstance(x, np.str_):
                x = str(x)
            cleaned.append(x)
        flat = np.array(cleaned, dtype=object)
        if flat.size == 0:
            return None, 0.0
    if flat.dtype.kind in "fc":
        flat = flat[~np.isnan(flat)]
    if flat.size == 0:
        return None, 0.0
    values, counts = np.unique(flat, return_counts=True)
    if counts.size == 0:
        return None, 0.0
    idx = counts.argmax()
    total = counts.sum()
    confidence = float(counts[idx] / total) if total > 0 else 0.0
    return values[idx], confidence


def _normalize_rows(cm: np.ndarray) -> np.ndarray:
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    out = np.zeros_like(cm, dtype=float)
    np.divide(cm, row_sums, out=out, where=row_sums != 0)
    return out


def _normalize_cols(cm: np.ndarray) -> np.ndarray:
    col_sums = cm.sum(axis=0, keepdims=True).astype(float)
    out = np.zeros_like(cm, dtype=float)
    np.divide(cm, col_sums, out=out, where=col_sums != 0)
    return out


def _print_confusion_matrix(title: str, matrix: np.ndarray, label_names: list[str], as_percent: bool):
    row_w = max(8, max(len(name) for name in label_names))
    cell_w = max(8, max(len(name) for name in label_names))
    header = " " * (row_w + 3) + " ".join(f"{name:>{cell_w}}" for name in label_names)
    print(f"\n{title}")
    print(header)
    for i, name in enumerate(label_names):
        if as_percent:
            row = " ".join(f"{(100.0 * float(v)):>{cell_w}.1f}" for v in matrix[i])
        else:
            row = " ".join(f"{int(v):>{cell_w}d}" for v in matrix[i])
        print(f"{name:>{row_w}} | {row}")


def _compute_eval_artifacts(y_true, y_pred, index_to_label):
    label_indices = list(range(len(index_to_label)))
    label_names = [index_to_label[i] for i in label_indices]

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
    report_dict: dict[str, Any] = {str(k): v for k, v in report_dict_raw.items()}

    cm_counts = confusion_matrix(y_true, y_pred, labels=label_indices)
    cm_row_norm = _normalize_rows(cm_counts)
    cm_col_norm = _normalize_cols(cm_counts)

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
        for i, name in enumerate(label_names):
            if i == neutral_idx:
                continue
            row_total = int(cm_counts[i].sum())
            rate = float(cm_counts[i, neutral_idx] / row_total) if row_total > 0 else 0.0
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


def _serialize_eval_metrics(eval_artifacts: dict[str, Any]) -> dict[str, Any]:
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


# ── Calibration ──────────────────────────────────────────────────────────────

MVC_QUALITY_MIN_RATIO = 1.5  # skip MVC normalization if median ratio falls below this

def compute_calibration(neutral_emg, mvc_emg, percentile):
    neutral = np.asarray(neutral_emg, dtype=float)
    mvc     = np.asarray(mvc_emg, dtype=float)
    if neutral.size == 0 or mvc.size == 0:
        return None, None

    neutral_rms = np.sqrt(np.mean(neutral ** 2, axis=0))
    mvc_rms     = np.sqrt(np.mean(mvc ** 2, axis=0))
    ratio       = np.where(neutral_rms < 1e-9, 1.0, mvc_rms / neutral_rms)
    median_ratio = float(np.median(ratio))

    if median_ratio < MVC_QUALITY_MIN_RATIO:
        print(
            f"  [calib] SKIP: median MVC/neutral ratio={median_ratio:.2f}x "
            f"(< {MVC_QUALITY_MIN_RATIO}x threshold). "
            "MVC calibration failed — normalization not applied for this session."
        )
        return None, None

    neutral_mean = np.mean(neutral, axis=0)
    mvc_scale    = np.percentile(mvc, percentile, axis=0)
    mvc_scale    = np.where(mvc_scale < 1e-6, 1.0, mvc_scale)
    return neutral_mean, mvc_scale


def validate_calibration_data(files):
    missing = []
    for fp in files:
        try:
            data = np.load(fp, allow_pickle=True)
            if data.get("calib_neutral_emg") is None or data.get("calib_mvc_emg") is None:
                missing.append(fp)
        except Exception:
            missing.append(fp)
    if missing:
        print(
            f"WARNING: {len(missing)} file(s) lack calibration data. "
            "Calibration normalisation will be skipped for these sessions."
        )
        for fp in missing[:5]:
            print(f"  {fp}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more.")


# ── Data loading ─────────────────────────────────────────────────────────────

def subject_from_path(path: Path) -> str:
    return path.parent.parent.name


def _keep_gesture_label(label) -> bool:
    if label is None:
        return False
    label = str(label)
    if INCLUDED_GESTURES is not None and label not in INCLUDED_GESTURES:
        return False
    return True


def load_windows_from_file(path):
    data = np.load(path, allow_pickle=True)
    if "emg" not in data.files or "y" not in data.files:
        return None
    emg = np.asarray(data["emg"], dtype=float)
    metadata = data.get("metadata")
    strict_layout = None
    if CHANNEL_LAYOUT_MODE == "strict":
        strict_layout = resolve_strict_indices_from_metadata(metadata, arm=ARM)
        if emg.shape[1] != strict_layout.ordered_indices.size:
            raise ValueError(
                f"{path.name}: strict layout resolved {strict_layout.ordered_indices.size} channels "
                f"for {ARM}, but file has {emg.shape[1]}."
            )
        emg = emg[:, strict_layout.ordered_indices]

    if USE_CALIBRATION:
        calib_neutral = data.get("calib_neutral_emg")
        calib_mvc     = data.get("calib_mvc_emg")
        if calib_neutral is not None and calib_mvc is not None:
            calib_neutral = np.asarray(calib_neutral, dtype=float)
            calib_mvc = np.asarray(calib_mvc, dtype=float)
            if strict_layout is not None:
                calib_neutral = calib_neutral[:, strict_layout.ordered_indices]
                calib_mvc = calib_mvc[:, strict_layout.ordered_indices]
            neutral_mean, mvc_scale = compute_calibration(
                calib_neutral, calib_mvc, MVC_PERCENTILE
            )
            if neutral_mean is not None and mvc_scale is not None:
                emg = (emg - neutral_mean) / mvc_scale

    windows = get_windows(emg, WINDOW_SIZE, WINDOW_STEP)
    labels  = np.asarray(data["y"], dtype=object)

    n_windows = windows.shape[0]
    starts    = np.arange(n_windows) * WINDOW_STEP
    ends      = starts + WINDOW_SIZE

    window_labels = []
    for s, e in zip(starts, ends):
        lbl, confidence = majority_label_with_confidence(labels[s:e])
        if lbl == "neutral_buffer":
            lbl = None
        if USE_MIN_LABEL_CONFIDENCE and lbl is not None and confidence < MIN_LABEL_CONFIDENCE:
            lbl = None
        if lbl is not None and not _keep_gesture_label(lbl):
            lbl = None
        window_labels.append(lbl)

    window_labels = np.asarray(window_labels, dtype=object)
    keep          = window_labels != None  # noqa: E711
    windows       = windows[keep]
    window_labels = window_labels[keep]

    if windows.size == 0:
        return None
    return windows.astype(np.float32), window_labels, strict_layout


def load_dataset():
    files = sorted(DATA_ROOT.rglob(PATTERN))
    if not files:
        raise FileNotFoundError(f"No filtered files found under {DATA_ROOT}")

    if INCLUDED_GESTURES is not None:
        print(f"Including gestures only: {sorted(INCLUDED_GESTURES)}")

    if EXCLUDED_SUBJECTS:
        files = [f for f in files if subject_from_path(f) not in EXCLUDED_SUBJECTS]
        print(f"Excluded subjects (DQ policy): {EXCLUDED_SUBJECTS}")

    if USE_CALIBRATION:
        validate_calibration_data(files)

    X_list, y_list, groups_list, subjects_list, channel_counts = [], [], [], [], []
    layout_sources: dict[str, int] = {}
    strict_pair_order = None
    strict_slot_order = None
    strict_channel_counts = None
    for fp in files:
        result = load_windows_from_file(fp)
        if result is None:
            continue
        windows, labels, strict_layout = result
        X_list.append(windows)
        y_list.append(labels)
        groups_list.append(np.array([str(fp)] * len(labels), dtype=object))
        subjects_list.append(np.array([subject_from_path(fp)] * len(labels), dtype=object))
        channel_counts.append(int(windows.shape[1]))
        if CHANNEL_LAYOUT_MODE == "strict":
            if strict_layout is None:
                raise ValueError(f"{fp.name}: strict layout resolution failed unexpectedly.")
            layout_sources["emg_channel_labels"] = layout_sources.get("emg_channel_labels", 0) + 1
            if strict_pair_order is None:
                strict_pair_order = tuple(strict_layout.pair_numbers)
                strict_slot_order = tuple(strict_layout.slot_names)
                strict_channel_counts = tuple(strict_layout.channel_counts)
            elif (
                tuple(strict_layout.pair_numbers) != strict_pair_order
                or tuple(strict_layout.slot_names) != strict_slot_order
                or tuple(strict_layout.channel_counts) != strict_channel_counts
            ):
                raise ValueError(
                    f"Inconsistent strict slot mapping while loading {fp.name}: "
                    f"pairs={tuple(strict_layout.pair_numbers)}"
                )

    if not X_list:
        raise ValueError("No labeled windows found in filtered files.")

    unique_counts = sorted(set(channel_counts))
    if len(unique_counts) != 1:
        raise ValueError(f"Channel count mismatch across files: {unique_counts}")

    return (
        np.vstack(X_list),
        np.concatenate(y_list),
        np.concatenate(groups_list),
        np.concatenate(subjects_list),
        unique_counts[0],
        layout_sources,
    )


# ── Normalisation ─────────────────────────────────────────────────────────────
def _prepare_test_data(X, _mean, _std):
    return standardize_windows(X, _mean, _std, model_family=MODEL_FAMILY)


# ── Augmentation (GPU-native) ─────────────────────────────────────────────────
# All ops run on the GPU tensor after .to(device), eliminating CPU↔GPU round-trips.
# Temporal stretch uses a single F.interpolate over the whole batch (one factor per
# batch) instead of per-sample loops, which is ~100× faster on GPU.

import torch.nn.functional as _F

def augment_emg_gpu(xb: torch.Tensor, p: float) -> torch.Tensor:
    """In-place-safe GPU augmentation. xb: (B, C, T) float32 on device."""
    B, C, T = xb.shape
    dev = xb.device
    xb = xb.clone()

    # 1. Amplitude scaling — per-sample random scale
    mask = torch.rand(B, device=dev) < p
    if mask.any():
        n = int(mask.sum())
        factors = torch.empty(n, 1, 1, device=dev).uniform_(AMP_RANGE[0], AMP_RANGE[1])
        xb[mask] = xb[mask] * factors

    # 2. Additive Gaussian noise — SNR-calibrated per sample
    mask = torch.rand(B, device=dev) < p
    if mask.any():
        n = int(mask.sum())
        snr_db     = torch.empty(n, device=dev).uniform_(10.0, 30.0)
        snr_linear = 10.0 ** (snr_db / 10.0)
        sig_power  = xb[mask].var(dim=(1, 2)).clamp(min=1e-8)        # (n,)
        noise_std  = (sig_power / snr_linear).sqrt().view(n, 1, 1)   # (n,1,1)
        xb[mask]   = xb[mask] + torch.randn(n, C, T, device=dev) * noise_std

    # 3. Temporal shift — vectorised gather (no Python loop)
    mask = torch.rand(B, device=dev) < p
    if mask.any():
        n      = int(mask.sum())
        shifts = torch.randint(-20, 21, (n,), device=dev)
        idx    = (torch.arange(T, device=dev).unsqueeze(0) - shifts.unsqueeze(1)) % T
        idx    = idx.unsqueeze(1).expand(-1, C, -1)                  # (n, C, T)
        xb[mask] = torch.gather(xb[mask], 2, idx)

    # 4. Channel dropout — vectorised scatter
    mask = torch.rand(B, device=dev) < p
    if mask.any():
        n       = int(mask.sum())
        drop_ch = torch.randint(0, C, (n,), device=dev)
        ch_mask = torch.zeros(n, C, device=dev, dtype=torch.bool)
        ch_mask.scatter_(1, drop_ch.unsqueeze(1), True)
        xb[mask] = xb[mask].masked_fill(ch_mask.unsqueeze(2), 0.0)

    # 5. Temporal stretch — single batched interpolation (one shared factor per batch)
    if torch.rand(1, device=dev).item() < p:
        factor  = torch.empty(1, device=dev).uniform_(0.85, 1.15).item()
        new_len = max(1, int(T * factor))
        stretched = _F.interpolate(xb, size=new_len, mode="linear", align_corners=False)
        if new_len >= T:
            xb = stretched[:, :, :T]
        else:
            xb = _F.pad(stretched, (0, T - new_len), mode="replicate")

    return xb


# ── Subject-balanced sampling ─────────────────────────────────────────────────

def make_subject_sample_weights(subjects: np.ndarray) -> np.ndarray:
    unique, counts = np.unique(subjects, return_counts=True)
    weight_map = {s: 1.0 / c for s, c in zip(unique, counts)}
    return np.array([weight_map[s] for s in subjects], dtype=np.float32)


# ── Model ─────────────────────────────────────────────────────────────────────

def _build_model(in_channels: int, num_classes: int, device) -> nn.Module:
    return build_model(
        MODEL_FAMILY,
        in_channels=in_channels,
        num_classes=num_classes,
        dropout=DROPOUT,
        device=device,
        family_cfg=FAMILY_CFG,
    )


def _embed_windows(model: nn.Module, X: np.ndarray, device, *, l2_normalize: bool) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(X.astype(np.float32))), batch_size=BATCH_SIZE, shuffle=False)
    chunks = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            extract_embedding = getattr(model, "extract_embedding", None)
            if callable(extract_embedding):
                emb_tensor = cast(SupportsExtractEmbedding, model).extract_embedding(
                    xb,
                    l2_normalize=l2_normalize,
                )
            else:
                logits = model(xb)
                emb_tensor = logits
                if l2_normalize:
                    emb_tensor = torch.nn.functional.normalize(emb_tensor, p=2, dim=1, eps=1e-8)
            chunks.append(emb_tensor.detach().cpu().numpy())
    if not chunks:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


# ── Training ──────────────────────────────────────────────────────────────────

def train_eval_split(
    X_train,
    y_train,
    X_eval,
    y_eval,
    channels,
    num_classes,
    epochs,
    device,
    subjects_train,
    *,
    use_eval_for_model_selection=True,
    use_eval_for_scheduler=True,
):
    X_train_t, X_eval_t, mean, std = prepare_train_eval_inputs(
        MODEL_FAMILY, X_train, X_eval
    )

    train_ds = TensorDataset(torch.from_numpy(X_train_t), torch.from_numpy(y_train))
    eval_ds  = TensorDataset(torch.from_numpy(X_eval_t),  torch.from_numpy(y_eval))

    if subjects_train is None:
        raise ValueError("subjects_train is required for cross-subject balanced sampling.")
    weights = make_subject_sample_weights(subjects_train)
    weights_seq: list[float] = [float(w) for w in weights.tolist()]
    sampler = WeightedRandomSampler(
        weights_seq, num_samples=len(weights_seq), replacement=True
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler, drop_last=False, pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, pin_memory=True,
    )

    model = _build_model(int(channels[0]), num_classes, device)

    objectives = build_training_objectives(
        MODEL_FAMILY,
        label_smoothing=LABEL_SMOOTHING,
        family_cfg=FAMILY_CFG,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # patience=5: cross-subject dataset is large; loss moves slowly
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, threshold=1e-4, min_lr=1e-6,
    )

    best_state = None
    best_metric = -1.0 if use_eval_for_model_selection else float("inf")
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_correct = train_total = 0
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            if USE_AUGMENTATION:
                xb = augment_emg_gpu(xb, p=AUG_PROB)
            optimizer.zero_grad()
            logits, loss = compute_training_step(
                MODEL_FAMILY,
                model,
                xb,
                yb,
                objectives=objectives,
                family_cfg=FAMILY_CFG,
            )
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * xb.size(0)
            preds          = torch.argmax(logits, dim=1)
            train_correct += (preds == yb).sum().item()
            train_total   += xb.size(0)

        model.eval()
        eval_correct = eval_total = eval_loss_sum = 0
        with torch.no_grad():
            for xb, yb in eval_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits, l = compute_training_step(
                    MODEL_FAMILY,
                    model,
                    xb,
                    yb,
                    objectives=objectives,
                    family_cfg=FAMILY_CFG,
                )
                eval_loss_sum  += l.item() * xb.size(0)
                eval_correct   += (torch.argmax(logits, 1) == yb).sum().item()
                eval_total     += xb.size(0)

        train_acc = train_correct / max(train_total, 1)
        eval_acc  = eval_correct  / max(eval_total, 1)
        avg_loss  = train_loss    / max(train_total, 1)
        avg_eval  = eval_loss_sum / max(eval_total, 1)
        print(
            f"Epoch {epoch:02d} | loss {avg_loss:.4f} | eval_loss {avg_eval:.4f} "
            f"| train {train_acc:.3f} | eval {eval_acc:.3f}"
        )
        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler_metric = avg_eval if use_eval_for_scheduler else avg_loss
        scheduler.step(scheduler_metric)
        if optimizer.param_groups[0]["lr"] < prev_lr:
            print(f"LR reduced: {prev_lr:.2e} -> {optimizer.param_groups[0]['lr']:.2e}")
        if epoch == 1:
            est = (time.time() - epoch_start) * epochs
            print(f"Estimated remaining: ~{max(0, est - (time.time() - start_time)):.0f}s")
        if use_eval_for_model_selection:
            improved = eval_acc > best_metric
            best_metric = max(best_metric, eval_acc)
        else:
            improved = avg_loss < best_metric
            best_metric = min(best_metric, avg_loss)
        if improved:
            # state_dict() holds tensor references; deepcopy is required
            # to preserve the true best checkpoint instead of the final epoch.
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, mean, std, best_metric


# ── LOSO evaluation ───────────────────────────────────────────────────────────

def loso_evaluate(X, y_idx, subjects, channel_count, num_classes, device, index_to_label):
    """Leave-one-subject-out: true measure of cross-subject generalisation.

    Each fold trains on N-1 subjects and tests on the held-out subject.
    This is the correct evaluation metric for cross-subject models.
    In-distribution test accuracy (final model) is NOT comparable to this.
    """
    unique_subjects = sorted(np.unique(subjects))
    if len(unique_subjects) < 2:
        print("LOSO requires at least 2 subjects — skipping.")
        return {"enabled": False, "reason": "requires_at_least_two_subjects"}

    print(f"\nLOSO evaluation over {len(unique_subjects)} subjects:")
    print("(Measures true cross-subject accuracy; lower than in-distribution is expected)")
    channels  = [int(channel_count), 32, 64, 128]
    loso_accs = []
    fold_results = []
    for held_out in unique_subjects:
        train_mask = subjects != held_out
        test_mask  = subjects == held_out
        print(f"\n  Held-out: {held_out}  (train {train_mask.sum()}, test {test_mask.sum()})")
        model, mean, std, _ = train_eval_split(
            X[train_mask], y_idx[train_mask],
            X[test_mask],  y_idx[test_mask],
            channels, num_classes, EPOCHS, device,
            subjects_train=subjects[train_mask],
            use_eval_for_model_selection=False,
            use_eval_for_scheduler=False,
        )
        model.eval()
        X_test   = _prepare_test_data(X[test_mask], mean, std)
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_idx[test_mask])),
            batch_size=BATCH_SIZE, shuffle=False,
        )
        all_preds = []
        with torch.no_grad():
            for xb, _ in test_loader:
                all_preds.append(torch.argmax(model(xb.to(device)), dim=1).cpu().numpy())
        y_pred = np.concatenate(all_preds)
        eval_artifacts = _compute_eval_artifacts(y_idx[test_mask], y_pred, index_to_label)
        acc = float(eval_artifacts["test_accuracy"])
        loso_accs.append(acc)
        fold_results.append(
            {
                "held_out_subject": str(held_out),
                "test_windows": int(test_mask.sum()),
                "metrics": _serialize_eval_metrics(eval_artifacts),
            }
        )
        print(f"    Accuracy: {acc:.3f}")
        print(classification_report(
            y_idx[test_mask], y_pred,
            labels=list(range(len(index_to_label))),
            target_names=[index_to_label[i] for i in range(len(index_to_label))],
            zero_division=0,
        ))

    loso_accs = np.asarray(loso_accs, dtype=float)
    print(
        f"\nLOSO summary: mean={loso_accs.mean():.3f}  "
        f"std={loso_accs.std():.3f}  "
        f"min={loso_accs.min():.3f}  "
        f"max={loso_accs.max():.3f}"
    )
    if loso_accs.mean() < 0.65:
        print(
            "WARNING: Mean LOSO accuracy < 65%. The model may not generalise reliably "
            "to new subjects. Collect more diverse data before deploying."
        )
    return {
        "enabled": True,
        "model_family": MODEL_FAMILY,
        "folds": fold_results,
        "summary": {
            "mean_accuracy": float(loso_accs.mean()),
            "std_accuracy": float(loso_accs.std()),
            "min_accuracy": float(loso_accs.min()),
            "max_accuracy": float(loso_accs.max()),
        },
    }


def _predict_prototype_labels(
    model: nn.Module,
    X_support: np.ndarray,
    y_support: np.ndarray,
    X_query: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    device,
    num_classes: int,
) -> np.ndarray:
    X_support_t = _prepare_test_data(X_support, mean, std)
    X_query_t = _prepare_test_data(X_query, mean, std)
    support_embeddings = _embed_windows(
        model,
        X_support_t,
        device,
        l2_normalize=PROTOTYPE_L2_NORMALIZE,
    )
    classifier = PrototypeClassifier.fit(
        support_embeddings,
        y_support,
        temperature=PROTOTYPE_TEMPERATURE,
        l2_normalize=PROTOTYPE_L2_NORMALIZE,
    )
    query_embeddings = _embed_windows(
        model,
        X_query_t,
        device,
        l2_normalize=PROTOTYPE_L2_NORMALIZE,
    )
    if query_embeddings.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    probs = np.stack(
        [
            classifier.predict_proba(emb, num_classes=int(num_classes))
            for emb in query_embeddings
        ],
        axis=0,
    )
    return np.argmax(probs, axis=1).astype(np.int64, copy=False)


def calibrated_loso_evaluate(
    X,
    y_idx,
    groups,
    subjects,
    channel_count,
    num_classes,
    device,
    index_to_label,
):
    if not supports_prototype_calibration(MODEL_FAMILY):
        print("\nCalibrated LOSO skipped: current model family does not prefer prototype calibration.")
        return {"enabled": False, "reason": "prototype_calibration_not_supported"}

    unique_subjects = sorted(np.unique(subjects))
    if len(unique_subjects) < 2:
        print("\nCalibrated LOSO skipped: requires at least 2 subjects.")
        return {"enabled": False, "reason": "requires_at_least_two_subjects"}

    print(f"\nCalibrated LOSO evaluation over {len(unique_subjects)} subjects:")
    print("(Held-out session file = support set; remaining held-out sessions = query set)")

    channels = [int(channel_count), 32, 64, 128]
    required_labels = set(range(int(num_classes)))
    all_true = []
    all_pred = []
    subject_results = []
    fold_results = []

    for held_out in unique_subjects:
        train_mask = subjects != held_out
        subject_mask = subjects == held_out
        subject_groups = sorted({str(g) for g in groups[subject_mask]})
        if len(subject_groups) < 2:
            print(f"\n  Held-out: {held_out} skipped (needs at least 2 session files for support/query split).")
            continue

        print(f"\n  Held-out: {held_out}  (train {train_mask.sum()}, subject windows {subject_mask.sum()})")
        model, mean, std, _ = train_eval_split(
            X[train_mask], y_idx[train_mask],
            X[subject_mask], y_idx[subject_mask],
            channels, num_classes, EPOCHS, device,
            subjects_train=subjects[train_mask],
            use_eval_for_model_selection=False,
            use_eval_for_scheduler=False,
        )

        subject_true = []
        subject_pred = []
        subject_fold_results = []
        for support_group in subject_groups:
            support_mask = subject_mask & (groups == support_group)
            query_mask = subject_mask & (groups != support_group)
            if not np.any(query_mask):
                continue

            support_labels = set(int(v) for v in np.unique(y_idx[support_mask]).tolist())
            missing_labels = sorted(required_labels - support_labels)
            if missing_labels:
                missing_names = [index_to_label[idx] for idx in missing_labels]
                print(
                    f"    Support {Path(str(support_group)).name} skipped: "
                    f"missing labels {missing_names}"
                )
                continue

            y_pred = _predict_prototype_labels(
                model,
                X[support_mask],
                y_idx[support_mask],
                X[query_mask],
                mean=mean,
                std=std,
                device=device,
                num_classes=num_classes,
            )
            if y_pred.size == 0:
                continue

            y_true = y_idx[query_mask]
            eval_artifacts = _compute_eval_artifacts(y_true, y_pred, index_to_label)
            fold_payload = {
                "held_out_subject": str(held_out),
                "support_file": Path(str(support_group)).name,
                "support_windows": int(support_mask.sum()),
                "query_windows": int(query_mask.sum()),
                "metrics": _serialize_eval_metrics(eval_artifacts),
            }
            fold_results.append(fold_payload)
            subject_fold_results.append(fold_payload)
            subject_true.append(y_true)
            subject_pred.append(y_pred)
            all_true.append(y_true)
            all_pred.append(y_pred)
            print(
                f"    Support {Path(str(support_group)).name}: "
                f"accuracy={eval_artifacts['test_accuracy']:.3f} "
                f"macro_f1={eval_artifacts['macro_f1']:.3f}"
            )

        if subject_true:
            y_true_subject = np.concatenate(subject_true)
            y_pred_subject = np.concatenate(subject_pred)
            subject_eval = _compute_eval_artifacts(y_true_subject, y_pred_subject, index_to_label)
            subject_results.append(
                {
                    "held_out_subject": str(held_out),
                    "support_unit": "session_file",
                    "fold_count": int(len(subject_fold_results)),
                    "metrics": _serialize_eval_metrics(subject_eval),
                    "folds": subject_fold_results,
                }
            )
            print(
                f"    Subject summary: accuracy={subject_eval['test_accuracy']:.3f} "
                f"macro_f1={subject_eval['macro_f1']:.3f}"
            )
        else:
            print(f"    No valid support/query splits for {held_out}.")

    if not all_true:
        print("\nCalibrated LOSO produced no valid support/query folds.")
        return {"enabled": False, "reason": "no_valid_support_query_splits"}

    overall_true = np.concatenate(all_true)
    overall_pred = np.concatenate(all_pred)
    overall_eval = _compute_eval_artifacts(overall_true, overall_pred, index_to_label)
    print(
        f"\nCalibrated LOSO summary: accuracy={overall_eval['test_accuracy']:.3f} "
        f"balanced_accuracy={overall_eval['balanced_accuracy']:.3f} "
        f"macro_f1={overall_eval['macro_f1']:.3f}"
    )
    return {
        "enabled": True,
        "model_family": MODEL_FAMILY,
        "support_unit": "session_file",
        "subjects": subject_results,
        "folds": fold_results,
        "summary": _serialize_eval_metrics(overall_eval),
    }


# ── Save bundle ───────────────────────────────────────────────────────────────

def _build_bundle(
    *,
    model: nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    label_to_index: dict,
    index_to_label: dict,
    labels: list[str],
    channel_count: int,
    split_mode: str,
    train_files: list[str],
    test_files: list[str],
    eval_artifacts: dict,
    extra_metadata: dict | None = None,
):
    test_accuracy = float(eval_artifacts["test_accuracy"])
    metadata = {
        "created_at": dt.datetime.now().isoformat(),
        "stream": "cross_subject",
        "subject": None,
        "excluded_subjects": list(EXCLUDED_SUBJECTS),
        "included_gestures": sorted(INCLUDED_GESTURES) if INCLUDED_GESTURES is not None else None,
        "window_size_samples": WINDOW_SIZE,
        "window_step_samples": WINDOW_STEP,
        "channel_count": int(channel_count),
        "labels": labels,
        "split_mode": split_mode,
        "test_size": float(TEST_SIZE),
        "calibration_used": bool(USE_CALIBRATION),
        "calibration_mvc_percentile": float(MVC_PERCENTILE),
        **build_family_metadata(MODEL_FAMILY, family_cfg=FAMILY_CFG),
        "label_confidence_filter": {
            "enabled": bool(USE_MIN_LABEL_CONFIDENCE),
            "min_label_confidence": float(MIN_LABEL_CONFIDENCE),
        },
        "training": {
            "epochs": int(EPOCHS),
            "batch_size": int(BATCH_SIZE),
            "lr": float(LR),
            "amp_range": list(AMP_RANGE),
            "use_augmentation": bool(USE_AUGMENTATION),
            "use_balanced_sampling": True,
            "label_smoothing": float(LABEL_SMOOTHING),
            "use_mixup": False,
        },
        "metrics": {
            "test_accuracy": test_accuracy,
            "balanced_accuracy": eval_artifacts["balanced_accuracy"],
            "macro_precision": eval_artifacts["macro_precision"],
            "macro_recall": eval_artifacts["macro_recall"],
            "macro_f1": eval_artifacts["macro_f1"],
            "weighted_precision": eval_artifacts["weighted_precision"],
            "weighted_recall": eval_artifacts["weighted_recall"],
            "weighted_f1": eval_artifacts["weighted_f1"],
            "worst_class_recall_label": eval_artifacts["worst_class_recall_label"],
            "worst_class_recall": eval_artifacts["worst_class_recall"],
            "max_precision_recall_gap_label": eval_artifacts["max_pr_gap_label"],
            "max_precision_recall_gap": eval_artifacts["max_pr_gap"],
            "confusion_to_neutral_rate": eval_artifacts["confusion_to_neutral_rate"],
            "neutral_prediction_fp_rate": eval_artifacts["neutral_prediction_fp_rate"],
            "confusion_matrix_counts": eval_artifacts["confusion_matrix_counts"].tolist(),
            "confusion_matrix_row_norm": eval_artifacts["confusion_matrix_row_norm"].tolist(),
            "confusion_matrix_col_norm": eval_artifacts["confusion_matrix_col_norm"].tolist(),
        },
        "train_files": [Path(f).name for f in train_files],
        "test_files": [Path(f).name for f in test_files],
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return {
        "model_state": model.state_dict(),
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
        "label_to_index": label_to_index,
        "index_to_label": index_to_label,
        "architecture": build_architecture_metadata(
            MODEL_FAMILY,
            int(channel_count),
            dropout=DROPOUT,
            family_cfg=FAMILY_CFG,
        ),
        "metadata": metadata,
    }

def _train_and_save(
    X,
    y_idx,
    groups,
    subjects,
    channels,
    num_classes,
    device,
    labels,
    label_to_index,
    index_to_label,
    channel_count,
    model_out,
    layout_sources,
    evaluation_metadata=None,
):
    unique_groups = np.unique(groups)
    print(f"{X.shape[0]} windows across {len(unique_groups)} session file(s).")

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y_idx, groups))
    split_mode = "group-file"

    train_files = sorted({str(g) for g in groups[train_idx]})
    test_files  = sorted({str(g) for g in groups[test_idx]})
    print(f"Train ({len(train_files)} files): {[Path(f).name for f in train_files]}")
    print(f"Test  ({len(test_files)} files):  {[Path(f).name for f in test_files]}")
    print(f"\nTraining {MODEL_FAMILY} for {EPOCHS} epochs on {len(train_idx)} windows.")

    model, mean, std, _ = train_eval_split(
        X[train_idx], y_idx[train_idx],
        X[test_idx],  y_idx[test_idx],
        channels, num_classes, EPOCHS, device,
        subjects_train=subjects[train_idx],
    )

    model.eval()
    X_test = _prepare_test_data(X[test_idx], mean, std)
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_idx[test_idx])),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    all_preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            all_preds.append(torch.argmax(model(xb.to(device)), dim=1).cpu().numpy())
    y_pred = np.concatenate(all_preds)
    y_test = y_idx[test_idx]

    eval_artifacts = _compute_eval_artifacts(y_test, y_pred, index_to_label)
    test_accuracy = float(eval_artifacts["test_accuracy"])
    print(f"\nIn-distribution test accuracy: {test_accuracy:.3f}")
    print("(Note: this is NOT the cross-subject accuracy. See LOSO results above.)")
    print("\nReport:\n", eval_artifacts["classification_report_text"])
    _print_confusion_matrix(
        "Confusion matrix (counts, rows=true, cols=pred):",
        eval_artifacts["confusion_matrix_counts"],
        [index_to_label[i] for i in range(len(labels))],
        as_percent=False,
    )
    _print_confusion_matrix(
        "Confusion matrix (row-normalized %, rows=true, cols=pred):",
        eval_artifacts["confusion_matrix_row_norm"],
        [index_to_label[i] for i in range(len(labels))],
        as_percent=True,
    )
    _print_confusion_matrix(
        "Confusion matrix (col-normalized %, rows=true, cols=pred):",
        eval_artifacts["confusion_matrix_col_norm"],
        [index_to_label[i] for i in range(len(labels))],
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

    test_subjects = subjects[test_idx]
    unique_test   = np.unique(test_subjects)
    if len(unique_test) > 1:
        print("Per-subject breakdown:")
        for subj in sorted(unique_test):
            mask = test_subjects == subj
            print(f"  {subj}: {accuracy_score(y_test[mask], y_pred[mask]):.3f}  ({mask.sum()} windows)")

    extra_metadata = None
    if CHANNEL_LAYOUT_MODE == "strict":
        extra_metadata = {
            "channel_layout": {
                **strict_layout_bundle_metadata(ARM),
                "type_canonicalization_enabled": False,
                "canonical_block_order": [],
                "permutation_groups": [],
                "permutation_augmentation_enabled": False,
                "layout_inference_sources": dict(layout_sources),
                "kind_counts": {},
            }
        }
    if evaluation_metadata:
        if extra_metadata is None:
            extra_metadata = {}
        extra_metadata["evaluation"] = evaluation_metadata

    bundle = _build_bundle(
        model=model,
        mean=mean,
        std=std,
        label_to_index=label_to_index,
        index_to_label=index_to_label,
        labels=labels,
        channel_count=channel_count,
        split_mode=split_mode,
        train_files=train_files,
        test_files=test_files,
        eval_artifacts=eval_artifacts,
        extra_metadata=extra_metadata,
    )

    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, model_out)
    print(f"Saved to {model_out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if ARM not in ("right", "left"):
        raise ValueError(f"ARM must be 'right' or 'left', got {ARM!r}")
    if CHANNEL_LAYOUT_MODE not in ("strict", "none"):
        raise ValueError(
            f"CHANNEL_LAYOUT_MODE must be 'strict' or 'none', got {CHANNEL_LAYOUT_MODE!r}"
        )
    validate_model_family(MODEL_FAMILY)
    if CALIBRATED_LOSO_EVAL and not LOSO_EVAL:
        raise ValueError("CALIBRATED_LOSO_EVAL requires LOSO_EVAL = True.")
    confirm = input(f"Training {ARM} arm cross-subject model — continue? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    X, y, groups, subjects, channel_count, layout_sources = load_dataset()
    unique_subjects = sorted(np.unique(subjects))
    print(
        f"Loaded {X.shape[0]} windows, {channel_count} channels, "
        f"{len(np.unique(y))} classes, {len(unique_subjects)} subject(s): "
        f"{unique_subjects}"
    )
    print(f"Model family: {MODEL_FAMILY}")
    if CHANNEL_LAYOUT_MODE == "strict":
        expected_channels = strict_channel_count_for_arm(ARM)
        if channel_count != expected_channels:
            raise ValueError(
                f"Strict layout for {ARM} expects {expected_channels} channels, got {channel_count}."
            )
        print(
            f"Channel layout mode: STRICT ({ARM}) | pairs={strict_layout_bundle_metadata(ARM)['pair_order']}"
        )

    labels         = sorted({str(lbl) for lbl in np.unique(y)})
    if INCLUDED_GESTURES is not None:
        missing = sorted(set(INCLUDED_GESTURES) - set(labels))
        if missing:
            print(f"WARNING: requested INCLUDED_GESTURES not found in dataset: {missing}")
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    y_idx          = np.array([label_to_index[str(lbl)] for lbl in y], dtype=np.int64)

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels    = [int(channel_count), 32, 64, 128]
    num_classes = len(labels)

    # LOSO runs first so you can assess cross-subject accuracy before committing
    # to the final model weights.
    evaluation_metadata = {}
    if LOSO_EVAL:
        evaluation_metadata["zero_shot_loso"] = loso_evaluate(
            X, y_idx, subjects, channel_count, num_classes, device, index_to_label
        )
        if CALIBRATED_LOSO_EVAL:
            evaluation_metadata["calibrated_loso"] = calibrated_loso_evaluate(
                X,
                y_idx,
                groups,
                subjects,
                channel_count,
                num_classes,
                device,
                index_to_label,
            )

    if not TRAIN_FINAL_MODEL:
        print("\nTRAIN_FINAL_MODEL = False; skipping pooled final fit.")
        return

    print(f"\n{'=' * 55}")
    print("Final cross-subject model (all subjects pooled)")
    _train_and_save(
        X, y_idx, groups, subjects,
        channels, num_classes, device,
        labels, label_to_index, index_to_label, channel_count,
        model_out=MODEL_OUT,
        layout_sources=layout_sources,
        evaluation_metadata=evaluation_metadata or None,
    )

    print(f"\n{'=' * 55}")
    print(f"Cross-subject model saved to {MODEL_OUT}")
    print(f"Run realtime inference:")
    print(f"  python realtime_gesture_cnn.py --model {MODEL_OUT}")


if __name__ == "__main__":
    main()
