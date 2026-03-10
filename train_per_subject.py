"""Single-subject EMG gesture classifier training (cross-subject-aligned).

Uses GestureCNNv2 (InstanceNorm input + energy bypass) with the same core
training/evaluation style as train_cross_subject.py, but trains exactly one
model for TARGET_SUBJECT.

Usage:
    python train_per_subject.py

To run realtime inference with the trained model:
    python realtime_gesture_cnn.py --model models/per_subject/right/Matthew_cnn.pt
"""
import copy
import datetime as dt
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as _F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from libemg.utils import get_windows
from emg.gesture_model_cnn import GestureCNNv2


# ======== Config ========
ARM = "right"  # set to "right" or "left" before running
TARGET_SUBJECT = "Matthew"

DATA_ROOT = Path("data_resampled") / f"{ARM} arm"
PATTERN = "*_filtered.npz"
MODEL_OUT = Path("models/per_subject") / ARM / f"{TARGET_SUBJECT}_3_gesture.pt"

WINDOW_SIZE = 200
WINDOW_STEP = 100

USE_CALIBRATION = True
MVC_PERCENTILE = 95.0
USE_MIN_LABEL_CONFIDENCE = True
MIN_LABEL_CONFIDENCE = 0.75

TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 512

EPOCHS = 65
LR = 1e-4
DROPOUT = 0.25
LABEL_SMOOTHING = 0.05

USE_AUGMENTATION = True
AMP_RANGE = (0.7, 1.4)
AUG_PROB = 0.4

EXCLUDED_SUBJECTS: list[str] = []  # Keep empty for Matthew-only runs unless you need to blacklist a subject.
INCLUDED_GESTURES: set[str] | None = {"neutral", "left_turn", "right_turn"}  # Example subset: {"neutral", "left_turn", "right_turn"}.
# ========================


# -- Label utilities -----------------------------------------------------------

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
        target_names=label_names,
        zero_division=0,
    )
    report_dict_raw = classification_report(
        y_true,
        y_pred,
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


# -- Calibration ---------------------------------------------------------------

MVC_QUALITY_MIN_RATIO = 1.5


def compute_calibration(neutral_emg, mvc_emg, percentile):
    neutral = np.asarray(neutral_emg, dtype=float)
    mvc = np.asarray(mvc_emg, dtype=float)
    if neutral.size == 0 or mvc.size == 0:
        return None, None

    neutral_rms = np.sqrt(np.mean(neutral ** 2, axis=0))
    mvc_rms = np.sqrt(np.mean(mvc ** 2, axis=0))
    ratio = np.where(neutral_rms < 1e-9, 1.0, mvc_rms / neutral_rms)
    median_ratio = float(np.median(ratio))

    if median_ratio < MVC_QUALITY_MIN_RATIO:
        print(
            f"  [calib] SKIP: median MVC/neutral ratio={median_ratio:.2f}x "
            f"(< {MVC_QUALITY_MIN_RATIO}x threshold). "
            "MVC calibration failed - normalization not applied for this session."
        )
        return None, None

    neutral_mean = np.mean(neutral, axis=0)
    mvc_scale = np.percentile(mvc, percentile, axis=0)
    mvc_scale = np.where(mvc_scale < 1e-6, 1.0, mvc_scale)
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
            f"WARNING: {len(missing)} file(s) lack calibration data "
            "(calib_neutral_emg / calib_mvc_emg). "
            "Calibration normalisation will be skipped for these sessions."
        )
        for fp in missing[:5]:
            print(f"  {fp}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more.")


# -- Data loading --------------------------------------------------------------

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

    if USE_CALIBRATION:
        calib_neutral = data.get("calib_neutral_emg")
        calib_mvc = data.get("calib_mvc_emg")
        if calib_neutral is not None and calib_mvc is not None:
            neutral_mean, mvc_scale = compute_calibration(
                calib_neutral, calib_mvc, MVC_PERCENTILE
            )
            if neutral_mean is not None and mvc_scale is not None:
                emg = (emg - neutral_mean) / mvc_scale

    windows = get_windows(emg, WINDOW_SIZE, WINDOW_STEP)
    labels = np.asarray(data["y"], dtype=object)

    n_windows = windows.shape[0]
    starts = np.arange(n_windows) * WINDOW_STEP
    ends = starts + WINDOW_SIZE

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
    keep = window_labels != None  # noqa: E711
    windows = windows[keep]
    window_labels = window_labels[keep]

    if windows.size == 0:
        return None
    return windows.astype(np.float32), window_labels


def load_dataset(target_subject: str):
    files = sorted(DATA_ROOT.rglob(PATTERN))
    if not files:
        raise FileNotFoundError(f"No filtered files found under {DATA_ROOT}")

    files = [f for f in files if subject_from_path(f) == target_subject]
    if not files:
        raise FileNotFoundError(
            f"No filtered files found for TARGET_SUBJECT={target_subject!r} under {DATA_ROOT}"
        )

    if EXCLUDED_SUBJECTS and target_subject in EXCLUDED_SUBJECTS:
        raise ValueError(
            f"TARGET_SUBJECT={target_subject!r} is listed in EXCLUDED_SUBJECTS."
        )

    if INCLUDED_GESTURES is not None:
        print(f"Including gestures only: {sorted(INCLUDED_GESTURES)}")

    if USE_CALIBRATION:
        validate_calibration_data(files)

    X_list, y_list, groups_list, subjects_list, channel_counts = [], [], [], [], []
    for fp in files:
        result = load_windows_from_file(fp)
        if result is None:
            continue
        windows, labels = result
        X_list.append(windows)
        y_list.append(labels)
        groups_list.append(np.array([str(fp)] * len(labels), dtype=object))
        subjects_list.append(np.array([subject_from_path(fp)] * len(labels), dtype=object))
        channel_counts.append(int(windows.shape[1]))

    if not X_list:
        raise ValueError(f"No labeled windows found for TARGET_SUBJECT={target_subject!r}.")

    unique_counts = sorted(set(channel_counts))
    if len(unique_counts) != 1:
        raise ValueError(f"Channel count mismatch across files: {unique_counts}")

    return (
        np.vstack(X_list),
        np.concatenate(y_list),
        np.concatenate(groups_list),
        np.concatenate(subjects_list),
        unique_counts[0],
    )


# -- Normalisation -------------------------------------------------------------

def _prepare_test_data(X, _mean, _std):
    # GestureCNNv2 handles normalisation internally via InstanceNorm1d.
    return X.astype(np.float32)


# -- Augmentation --------------------------------------------------------------

def augment_emg_gpu(xb: torch.Tensor, p: float) -> torch.Tensor:
    """In-place-safe GPU augmentation. xb: (B, C, T) float32 on device."""
    bsz, channels, t_len = xb.shape
    dev = xb.device
    xb = xb.clone()

    mask = torch.rand(bsz, device=dev) < p
    if mask.any():
        n = int(mask.sum())
        factors = torch.empty(n, 1, 1, device=dev).uniform_(AMP_RANGE[0], AMP_RANGE[1])
        xb[mask] = xb[mask] * factors

    mask = torch.rand(bsz, device=dev) < p
    if mask.any():
        n = int(mask.sum())
        snr_db = torch.empty(n, device=dev).uniform_(10.0, 30.0)
        snr_linear = 10.0 ** (snr_db / 10.0)
        sig_power = xb[mask].var(dim=(1, 2)).clamp(min=1e-8)
        noise_std = (sig_power / snr_linear).sqrt().view(n, 1, 1)
        xb[mask] = xb[mask] + torch.randn(n, channels, t_len, device=dev) * noise_std

    mask = torch.rand(bsz, device=dev) < p
    if mask.any():
        n = int(mask.sum())
        shifts = torch.randint(-20, 21, (n,), device=dev)
        idx = (torch.arange(t_len, device=dev).unsqueeze(0) - shifts.unsqueeze(1)) % t_len
        idx = idx.unsqueeze(1).expand(-1, channels, -1)
        xb[mask] = torch.gather(xb[mask], 2, idx)

    mask = torch.rand(bsz, device=dev) < p
    if mask.any():
        n = int(mask.sum())
        drop_ch = torch.randint(0, channels, (n,), device=dev)
        ch_mask = torch.zeros(n, channels, device=dev, dtype=torch.bool)
        ch_mask.scatter_(1, drop_ch.unsqueeze(1), True)
        xb[mask] = xb[mask].masked_fill(ch_mask.unsqueeze(2), 0.0)

    if torch.rand(1, device=dev).item() < p:
        factor = torch.empty(1, device=dev).uniform_(0.8, 1.2).item()
        new_len = max(1, int(t_len * factor))
        stretched = _F.interpolate(xb, size=new_len, mode="linear", align_corners=False)
        if new_len >= t_len:
            xb = stretched[:, :, :t_len]
        else:
            xb = _F.pad(stretched, (0, t_len - new_len), mode="replicate")

    return xb


# -- Training ------------------------------------------------------------------

def _build_model(in_channels: int, num_classes: int, device) -> nn.Module:
    return GestureCNNv2(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout=DROPOUT,
    ).to(device)


def train_eval_split(
    X_train, y_train, X_eval, y_eval,
    channel_count, num_classes, epochs, device,
):
    mean = X_train.mean(axis=(0, 2))
    std = X_train.std(axis=(0, 2))
    std = np.where(std < 1e-6, 1.0, std)

    X_train_t = X_train.astype(np.float32)
    X_eval_t = X_eval.astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(X_train_t), torch.from_numpy(y_train))
    eval_ds = TensorDataset(torch.from_numpy(X_eval_t), torch.from_numpy(y_eval))

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, pin_memory=True
    )

    eval_loader = DataLoader(
        eval_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, pin_memory=True
    )

    model = _build_model(int(channel_count), num_classes, device)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, threshold=1e-4, min_lr=1e-6
    )

    best_state = None
    best_acc = -1.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if USE_AUGMENTATION:
                xb = augment_emg_gpu(xb, p=AUG_PROB)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == yb).sum().item()
            train_total += xb.size(0)

        model.eval()
        eval_correct = 0
        eval_total = 0
        eval_loss = 0.0
        with torch.no_grad():
            for xb, yb in eval_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                eval_loss += loss.item() * xb.size(0)
                preds = torch.argmax(logits, dim=1)
                eval_correct += (preds == yb).sum().item()
                eval_total += xb.size(0)

        train_acc = train_correct / max(train_total, 1)
        eval_acc = eval_correct / max(eval_total, 1)
        avg_loss = train_loss / max(train_total, 1)
        avg_eval_loss = eval_loss / max(eval_total, 1)
        print(
            f"Epoch {epoch:02d} | loss {avg_loss:.4f} | "
            f"eval_loss {avg_eval_loss:.4f} | train {train_acc:.3f} | "
            f"eval {eval_acc:.3f}"
        )

        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(avg_eval_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < prev_lr:
            print(f"LR reduced: {prev_lr:.2e} -> {new_lr:.2e}")

        if epoch == 1:
            epoch_time = time.time() - epoch_start
            est_total = epoch_time * epochs
            est_remaining = max(0.0, est_total - (time.time() - start_time))
            print(f"Estimated remaining time (this phase): ~{est_remaining:.1f}s")

        if eval_acc > best_acc:
            best_acc = eval_acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, mean, std, best_acc


# -- Train/eval/save -----------------------------------------------------------

def _train_and_save(
    X, y_idx, groups, subjects, channel_count, num_classes, device,
    labels, label_to_index, index_to_label, model_out, subject_tag,
):
    unique_groups = np.unique(groups)
    print(f"{X.shape[0]} windows across {len(unique_groups)} session file(s).")

    if unique_groups.size >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        train_idx, test_idx = next(splitter.split(X, y_idx, groups))
        split_mode = "group-file"
    else:
        indices = np.arange(X.shape[0])
        train_idx, test_idx = train_test_split(
            indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_idx
        )
        split_mode = "stratified-random"

    train_files = sorted({str(g) for g in groups[train_idx]})
    test_files = sorted({str(g) for g in groups[test_idx]})
    print(f"Train ({len(train_files)} files): {[Path(f).name for f in train_files]}")
    print(f"Test  ({len(test_files)} files):  {[Path(f).name for f in test_files]}")
    print(f"\nTraining GestureCNNv2 for {EPOCHS} epochs on {len(train_idx)} windows.")

    model, mean, std, _ = train_eval_split(
        X[train_idx], y_idx[train_idx],
        X[test_idx], y_idx[test_idx],
        channel_count, num_classes, EPOCHS, device,
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

    bundle = {
        "model_state": model.state_dict(),
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
        "label_to_index": label_to_index,
        "index_to_label": index_to_label,
        "architecture": {
            "type": "GestureCNNv2",
            "in_channels": int(channel_count),
            "dropout": float(DROPOUT),
        },
        "metadata": {
            "created_at": dt.datetime.now().isoformat(),
            "stream": "per_subject_single",
            "subject": subject_tag,
            "target_subject": subject_tag,
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
            "use_instance_norm_input": True,
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
                "use_balanced_sampling": False,
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
        },
    }

    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, model_out)
    print(f"Saved to {model_out}")


# -- Main ----------------------------------------------------------------------

def main():
    if ARM not in ("right", "left"):
        raise ValueError(f"ARM must be 'right' or 'left', got {ARM!r}")

    confirm = input(
        f"Training {ARM} arm single-subject model for '{TARGET_SUBJECT}' - continue? [y/N] "
    ).strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    X, y, groups, subjects, channel_count = load_dataset(target_subject=TARGET_SUBJECT)
    unique_subjects = sorted(np.unique(subjects))
    print(
        f"Loaded {X.shape[0]} windows, {channel_count} channels, "
        f"{len(np.unique(y))} classes, {len(unique_subjects)} subject(s): "
        f"{unique_subjects}"
    )

    labels = sorted({str(lbl) for lbl in np.unique(y)})
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    y_idx = np.array([label_to_index[str(lbl)] for lbl in y], dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(labels)

    print(f"\n{'=' * 55}")
    print(f"Single-subject model: {TARGET_SUBJECT}")
    _train_and_save(
        X, y_idx, groups, subjects, channel_count, num_classes, device,
        labels, label_to_index, index_to_label, MODEL_OUT, TARGET_SUBJECT,
    )

    print(f"\n{'=' * 55}")
    print("Single-subject training complete.")
    print(f"  python realtime_gesture_cnn.py --model {MODEL_OUT}")


if __name__ == "__main__":
    main()
