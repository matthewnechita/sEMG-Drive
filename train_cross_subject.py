"""Cross-subject EMG gesture classifier training — Stream 2.

Uses GestureCNNv2 (503K params, InstanceNorm at input, energy bypass scalar).
Trains one model on ALL subjects pooled; this model works on new participants
without any retraining. LOSO evaluation is mandatory before deployment.

Usage:
    python train_cross_subject.py

To run realtime inference with the cross-subject model:
    python realtime_gesture_cnn.py --model models/cross_subject/right/gesture_cnn_v3_m2_excl05_label_smoothing005.pt

DQ note: subject05 is excluded by default (EXCLUDED_SUBJECTS below).
All 4 sessions show MVC/neutral ratio <= 1.8x — the MVC calibration failed,
meaning mvc-normalised signal is near noise level for this subject.
Recollect calibration for subject05 before re-including them.
"""
import datetime as dt
import copy
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from libemg.utils import get_windows
from emg.gesture_model_cnn import GestureCNNv2


# ======== Config ========
ARM        = "left"            # ← set to "right" or "left" before running, lowercase l
DATA_ROOT  = Path("data_resampled") / f"{ARM} arm"
MODEL_OUT  = Path("models/cross_subject") / ARM / "gesture_cnn_v3_left_arm_new.pt"
PATTERN    = "*_filtered.npz"

WINDOW_SIZE = 200
WINDOW_STEP = 100

USE_CALIBRATION          = True
MVC_PERCENTILE           = 95.0
USE_MIN_LABEL_CONFIDENCE = True
MIN_LABEL_CONFIDENCE     = 0.75

TEST_SIZE    = 0.2
RANDOM_STATE = 42
BATCH_SIZE   = 512

# Cross-subject hyperparameters.
# GestureCNNv2 at 503K params on ~122K training windows = ~4 params/sample.
# More epochs than per-subject: larger model + larger dataset benefits from
# longer training. Convergence argument, not wall-clock.
EPOCHS  = 80
LR      = 1e-4
DROPOUT = 0.25
LABEL_SMOOTHING = 0.05

USE_CLASS_WEIGHTS    = True
USE_BALANCED_SAMPLING = True   # up-weights under-represented subjects

# Augmentation — wide amplitude range matches inter-subject EMG amplitude
# variance (~5-10x between people due to electrode placement, muscle mass).
USE_AUGMENTATION = True
AMP_RANGE = (0.5, 2.0)   # wide inter-subject amplitude variance range
AUG_PROB  = 0.5

# Subjects with failed MVC calibration or other DQ issues.
# subject05: all 4 sessions have mvc_ratio <= 1.8x (MVC barely > neutral).
# This corrupts the MVC normalisation; recollect before re-including.
EXCLUDED_SUBJECTS: list[str] = []

# Optional gesture filtering (code-only; no CLI flags).
# Set INCLUDED_GESTURES to a subset to train only those labels.
# Example:
# INCLUDED_GESTURES = {"neutral", "left_turn", "right_turn"} example
INCLUDED_GESTURES: set[str] | None = None # set to = None to include all gestures

# LOSO evaluation must be run before deploying the cross-subject model.
# This measures true cross-subject accuracy (model vs subjects it never trained on).
# Minimum recommended LOSO accuracy before deployment: 65%.
LOSO_EVAL = False

# Two-stage training mode:
#   Stage A: neutral vs active (binary)
#   Stage B: active gesture classification only (neutral excluded)
# Set to False to keep legacy single-stage training.
USE_TWO_STAGE = False
TWO_STAGE_NEUTRAL_LABEL = "neutral"
# ========================


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
        target_names=label_names,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    cm_counts = confusion_matrix(y_true, y_pred, labels=label_indices)
    cm_row_norm = _normalize_rows(cm_counts)
    cm_col_norm = _normalize_cols(cm_counts)

    per_class = []
    for name in label_names:
        stats = report_dict.get(name, {})
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

    return {
        "classification_report_text": report_text,
        "classification_report_dict": report_dict,
        "confusion_matrix_counts": cm_counts,
        "confusion_matrix_row_norm": cm_row_norm,
        "confusion_matrix_col_norm": cm_col_norm,
        "per_class": per_class,
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(report_dict["macro avg"]["precision"]),
        "macro_recall": float(report_dict["macro avg"]["recall"]),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "weighted_precision": float(report_dict["weighted avg"]["precision"]),
        "weighted_recall": float(report_dict["weighted avg"]["recall"]),
        "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
        "worst_class_recall_label": worst_recall["label"] if worst_recall else None,
        "worst_class_recall": worst_recall["recall"] if worst_recall else None,
        "max_pr_gap_label": max(per_class, key=lambda item: item["pr_gap"])["label"] if per_class else None,
        "max_pr_gap": max((item["pr_gap"] for item in per_class), default=None),
        "confusion_to_neutral_rate": confusion_to_neutral_rate,
        "neutral_prediction_fp_rate": neutral_prediction_fp_rate,
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

    if USE_CALIBRATION:
        calib_neutral = data.get("calib_neutral_emg")
        calib_mvc     = data.get("calib_mvc_emg")
        if calib_neutral is not None and calib_mvc is not None:
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
    return windows.astype(np.float32), window_labels


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
    )


# ── Normalisation ─────────────────────────────────────────────────────────────
# GestureCNNv2 handles normalisation internally via InstanceNorm1d.
# We pass raw float32; no external z-score applied.

def _prepare_test_data(X, _mean, _std):
    return X.astype(np.float32)


def _predict_indices(model: nn.Module, X: np.ndarray, device) -> np.ndarray:
    X_t = _prepare_test_data(X, None, None)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_t), torch.zeros(len(X_t), dtype=torch.long)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    preds = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb.to(device))
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds) if preds else np.empty((0,), dtype=np.int64)


def _derive_two_stage_paths(model_out: Path) -> tuple[Path, Path]:
    stem = model_out.stem
    stage_a = model_out.with_name(f"{stem}_stage_a_neutral_active.pt")
    stage_b = model_out.with_name(f"{stem}_stage_b_active_gestures.pt")
    return stage_a, stage_b


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
    """Always GestureCNNv2 for cross-subject training."""
    return GestureCNNv2(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout=DROPOUT,
    ).to(device)


# ── Training ──────────────────────────────────────────────────────────────────

def train_eval_split(X_train, y_train, X_eval, y_eval,
                     channels, num_classes, epochs, device, subjects_train=None):
    # No z-score: GestureCNNv2 normalises internally via InstanceNorm.
    # Compute mean/std for bundle compatibility (inference path checks metadata).
    mean = X_train.mean(axis=(0, 2))
    std  = X_train.std(axis=(0, 2))
    std  = np.where(std < 1e-6, 1.0, std)

    X_train_t = X_train.astype(np.float32)
    X_eval_t  = X_eval.astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(X_train_t), torch.from_numpy(y_train))
    eval_ds  = TensorDataset(torch.from_numpy(X_eval_t),  torch.from_numpy(y_eval))

    if USE_BALANCED_SAMPLING and subjects_train is not None:
        weights = make_subject_sample_weights(subjects_train)
        sampler = WeightedRandomSampler(
            torch.from_numpy(weights), num_samples=len(weights), replacement=True
        )
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, sampler=sampler, drop_last=False, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, pin_memory=True,
        )
    eval_loader = DataLoader(
        eval_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, pin_memory=True,
    )

    model = _build_model(int(channels[0]), num_classes, device)

    class_weight = None
    if USE_CLASS_WEIGHTS:
        class_counts = np.bincount(y_train, minlength=num_classes)
        w = class_counts.sum() / np.maximum(class_counts, 1)
        class_weight = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=float(LABEL_SMOOTHING))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # patience=5: cross-subject dataset is large; loss moves slowly
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, threshold=1e-4, min_lr=1e-6,
    )

    best_state, best_acc = None, -1.0
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
            logits = model(xb)
            loss   = criterion(logits, yb)
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
                logits = model(xb)
                l      = criterion(logits, yb)
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
        scheduler.step(avg_eval)
        if optimizer.param_groups[0]["lr"] < prev_lr:
            print(f"LR reduced: {prev_lr:.2e} -> {optimizer.param_groups[0]['lr']:.2e}")
        if epoch == 1:
            est = (time.time() - epoch_start) * epochs
            print(f"Estimated remaining: ~{max(0, est - (time.time() - start_time)):.0f}s")
        if eval_acc > best_acc:
            best_acc   = eval_acc
            # state_dict() holds tensor references; deepcopy is required
            # to preserve the true best checkpoint instead of the final epoch.
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, mean, std, best_acc


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
        return

    print(f"\nLOSO evaluation over {len(unique_subjects)} subjects:")
    print("(Measures true cross-subject accuracy; lower than in-distribution is expected)")
    channels  = [int(channel_count), 32, 64, 128]
    loso_accs = []
    for held_out in unique_subjects:
        train_mask = subjects != held_out
        test_mask  = subjects == held_out
        print(f"\n  Held-out: {held_out}  (train {train_mask.sum()}, test {test_mask.sum()})")
        model, mean, std, _ = train_eval_split(
            X[train_mask], y_idx[train_mask],
            X[test_mask],  y_idx[test_mask],
            channels, num_classes, EPOCHS, device,
            subjects_train=subjects[train_mask],
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
        acc = float(accuracy_score(y_idx[test_mask], y_pred))
        loso_accs.append(acc)
        print(f"    Accuracy: {acc:.3f}")
        print(classification_report(
            y_idx[test_mask], y_pred,
            target_names=[index_to_label[i] for i in range(len(index_to_label))],
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
            "use_balanced_sampling": bool(USE_BALANCED_SAMPLING),
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
        "architecture": {
            "type": "GestureCNNv2",
            "in_channels": int(channel_count),
            "dropout": float(DROPOUT),
        },
        "metadata": metadata,
    }

def _train_and_save(X, y_idx, groups, subjects, channels, num_classes, device,
                    labels, label_to_index, index_to_label, channel_count, model_out):
    unique_groups = np.unique(groups)
    print(f"{X.shape[0]} windows across {len(unique_groups)} session file(s).")

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y_idx, groups))
    split_mode = "group-file"

    train_files = sorted({str(g) for g in groups[train_idx]})
    test_files  = sorted({str(g) for g in groups[test_idx]})
    print(f"Train ({len(train_files)} files): {[Path(f).name for f in train_files]}")
    print(f"Test  ({len(test_files)} files):  {[Path(f).name for f in test_files]}")
    print(f"\nTraining GestureCNNv2 for {EPOCHS} epochs on {len(train_idx)} windows.")

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
                "use_balanced_sampling": bool(USE_BALANCED_SAMPLING),
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
            "test_files":  [Path(f).name for f in test_files],
        },
    }

    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, model_out)
    print(f"Saved to {model_out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def _train_two_stage_and_save(
    X,
    y_labels,
    groups,
    subjects,
    channel_count,
    device,
    model_out_stage_a: Path,
    model_out_stage_b: Path,
):
    y_labels = np.asarray([str(lbl) for lbl in y_labels], dtype=object)
    neutral_label = str(TWO_STAGE_NEUTRAL_LABEL)
    if neutral_label not in set(y_labels.tolist()):
        raise ValueError(
            f"TWO_STAGE_NEUTRAL_LABEL={neutral_label!r} not found in dataset labels: "
            f"{sorted(set(y_labels.tolist()))}"
        )

    unique_groups = np.unique(groups)
    print(f"{X.shape[0]} windows across {len(unique_groups)} session file(s).")

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, np.zeros(len(X), dtype=np.int64), groups))
    split_mode = "group-file"

    train_files = sorted({str(g) for g in groups[train_idx]})
    test_files = sorted({str(g) for g in groups[test_idx]})
    print(f"Train ({len(train_files)} files): {[Path(f).name for f in train_files]}")
    print(f"Test  ({len(test_files)} files):  {[Path(f).name for f in test_files]}")

    channels = [int(channel_count), 32, 64, 128]

    print(f"\nTraining Stage A (neutral vs active) for {EPOCHS} epochs.")
    y_stage_a = np.where(y_labels == neutral_label, 0, 1).astype(np.int64)
    label_to_index_a = {neutral_label: 0, "active": 1}
    index_to_label_a = {0: neutral_label, 1: "active"}
    labels_a = [neutral_label, "active"]

    model_a, mean_a, std_a, _ = train_eval_split(
        X[train_idx], y_stage_a[train_idx],
        X[test_idx], y_stage_a[test_idx],
        channels, 2, EPOCHS, device,
        subjects_train=subjects[train_idx],
    )
    y_pred_a = _predict_indices(model_a, X[test_idx], device)
    eval_a = _compute_eval_artifacts(y_stage_a[test_idx], y_pred_a, index_to_label_a)
    cm_a = eval_a["confusion_matrix_counts"]
    neutral_row = float(cm_a[0].sum()) if cm_a.shape[0] > 0 else 0.0
    active_row = float(cm_a[1].sum()) if cm_a.shape[0] > 1 else 0.0
    stage_a_false_activation_rate = float(cm_a[0, 1] / neutral_row) if neutral_row > 0 else 0.0
    stage_a_missed_activation_rate = float(cm_a[1, 0] / active_row) if active_row > 0 else 0.0
    print(
        f"Stage A test accuracy: {eval_a['test_accuracy']:.3f} | "
        f"false_activation_rate: {stage_a_false_activation_rate:.3f} | "
        f"missed_activation_rate: {stage_a_missed_activation_rate:.3f}"
    )
    print("\nStage A report:\n", eval_a["classification_report_text"])

    extra_a = {
        "two_stage": {
            "enabled": True,
            "role": "stage_a_neutral_vs_active",
            "neutral_label": neutral_label,
            "active_label": "active",
            "recommended_enter_threshold": 0.60,
            "recommended_exit_threshold": 0.45,
            "false_activation_rate": stage_a_false_activation_rate,
            "missed_activation_rate": stage_a_missed_activation_rate,
        }
    }
    bundle_a = _build_bundle(
        model=model_a,
        mean=mean_a,
        std=std_a,
        label_to_index=label_to_index_a,
        index_to_label=index_to_label_a,
        labels=labels_a,
        channel_count=channel_count,
        split_mode=split_mode,
        train_files=train_files,
        test_files=test_files,
        eval_artifacts=eval_a,
        extra_metadata=extra_a,
    )

    print(f"\nTraining Stage B (active gestures only) for {EPOCHS} epochs.")
    active_mask = y_labels != neutral_label
    labels_b = sorted({lbl for lbl in y_labels[active_mask]})
    if len(labels_b) < 2:
        raise ValueError(
            "Stage B requires at least 2 active gesture classes after filtering. "
            f"Found: {labels_b}"
        )
    label_to_index_b = {label: idx for idx, label in enumerate(labels_b)}
    index_to_label_b = {idx: label for label, idx in label_to_index_b.items()}

    y_stage_b = np.full(len(y_labels), -1, dtype=np.int64)
    for i, lbl in enumerate(y_labels):
        if lbl != neutral_label:
            y_stage_b[i] = label_to_index_b[lbl]

    train_mask = np.zeros(len(y_labels), dtype=bool)
    test_mask = np.zeros(len(y_labels), dtype=bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    train_active_mask = train_mask & active_mask
    test_active_mask = test_mask & active_mask
    if int(train_active_mask.sum()) == 0:
        raise ValueError(
            "Stage B has zero active training windows after split/filtering. "
            "Adjust TEST_SIZE, INCLUDED_GESTURES, or dataset composition."
        )
    if int(test_active_mask.sum()) == 0:
        raise ValueError(
            "Stage B has zero active test windows after split/filtering. "
            "Adjust TEST_SIZE, INCLUDED_GESTURES, or dataset composition."
        )
    print(
        f"Stage B windows: train={int(train_active_mask.sum())} "
        f"test={int(test_active_mask.sum())} classes={labels_b}"
    )

    model_b, mean_b, std_b, _ = train_eval_split(
        X[train_active_mask], y_stage_b[train_active_mask],
        X[test_active_mask], y_stage_b[test_active_mask],
        channels, len(labels_b), EPOCHS, device,
        subjects_train=subjects[train_active_mask],
    )
    y_pred_b = _predict_indices(model_b, X[test_active_mask], device)
    eval_b = _compute_eval_artifacts(y_stage_b[test_active_mask], y_pred_b, index_to_label_b)
    print(f"Stage B active-only test accuracy: {eval_b['test_accuracy']:.3f}")
    print("\nStage B report:\n", eval_b["classification_report_text"])

    extra_b = {
        "two_stage": {
            "enabled": True,
            "role": "stage_b_active_gestures",
            "neutral_label": neutral_label,
            "active_labels": labels_b,
        }
    }
    bundle_b = _build_bundle(
        model=model_b,
        mean=mean_b,
        std=std_b,
        label_to_index=label_to_index_b,
        index_to_label=index_to_label_b,
        labels=labels_b,
        channel_count=channel_count,
        split_mode=split_mode,
        train_files=train_files,
        test_files=test_files,
        eval_artifacts=eval_b,
        extra_metadata=extra_b,
    )

    y_true_labels = y_labels[test_idx]
    pred_labels = np.full(test_idx.shape[0], neutral_label, dtype=object)
    pred_active_mask = y_pred_a == 1
    if np.any(pred_active_mask):
        y_pred_b_for_active = _predict_indices(model_b, X[test_idx][pred_active_mask], device)
        pred_labels[pred_active_mask] = np.array(
            [index_to_label_b[int(i)] for i in y_pred_b_for_active], dtype=object
        )

    final_labels = [neutral_label] + labels_b
    final_label_to_index = {label: idx for idx, label in enumerate(final_labels)}
    final_index_to_label = {idx: label for label, idx in final_label_to_index.items()}
    y_true_final = np.array([final_label_to_index[str(lbl)] for lbl in y_true_labels], dtype=np.int64)
    y_pred_final = np.array([final_label_to_index[str(lbl)] for lbl in pred_labels], dtype=np.int64)
    eval_final = _compute_eval_artifacts(y_true_final, y_pred_final, final_index_to_label)
    print(f"\nTwo-stage end-to-end test accuracy: {eval_final['test_accuracy']:.3f}")
    print("Two-stage end-to-end report:\n", eval_final["classification_report_text"])
    if eval_final["confusion_to_neutral_rate"]:
        print("Two-stage confusion_to_neutral_rate:")
        for label, rate in sorted(eval_final["confusion_to_neutral_rate"].items()):
            print(f"  {label} -> neutral: {rate:.3f}")
    if eval_final["neutral_prediction_fp_rate"] is not None:
        print(f"Two-stage neutral_prediction_fp_rate: {eval_final['neutral_prediction_fp_rate']:.3f}")

    model_out_stage_a.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle_a, model_out_stage_a)
    torch.save(bundle_b, model_out_stage_b)
    print(f"\nSaved Stage A bundle: {model_out_stage_a}")
    print(f"Saved Stage B bundle: {model_out_stage_b}")


def main():
    if ARM not in ("right", "left"):
        raise ValueError(f"ARM must be 'right' or 'left', got {ARM!r}")
    confirm = input(f"Training {ARM} arm cross-subject model — continue? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    X, y, groups, subjects, channel_count = load_dataset()
    unique_subjects = sorted(np.unique(subjects))
    print(
        f"Loaded {X.shape[0]} windows, {channel_count} channels, "
        f"{len(np.unique(y))} classes, {len(unique_subjects)} subject(s): "
        f"{unique_subjects}"
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
    if LOSO_EVAL:
        loso_evaluate(X, y_idx, subjects, channel_count, num_classes, device, index_to_label)

    print(f"\n{'=' * 55}")
    if USE_TWO_STAGE:
        stage_a_out, stage_b_out = _derive_two_stage_paths(MODEL_OUT)
        print("Final cross-subject model (two-stage: A neutral/active + B active gestures)")
        _train_two_stage_and_save(
            X, y, groups, subjects,
            channel_count, device,
            model_out_stage_a=stage_a_out,
            model_out_stage_b=stage_b_out,
        )
        print(f"\n{'=' * 55}")
        print("Two-stage cross-subject bundles saved.")
        print(f"  Stage A: {stage_a_out}")
        print(f"  Stage B: {stage_b_out}")
        print("Run realtime inference:")
        print(
            "  python realtime_gesture_cnn.py "
            f"--two-stage --model-stage-a {stage_a_out} --model-stage-b {stage_b_out}"
        )
    else:
        print("Final cross-subject model (all subjects pooled)")
        _train_and_save(
            X, y_idx, groups, subjects,
            channels, num_classes, device,
            labels, label_to_index, index_to_label, channel_count,
            model_out=MODEL_OUT,
        )

        print(f"\n{'=' * 55}")
        print(f"Cross-subject model saved to {MODEL_OUT}")
        print(f"Run realtime inference:")
        print(f"  python realtime_gesture_cnn.py --model {MODEL_OUT}")


if __name__ == "__main__":
    main()
