# DEPRECATED — use train_per_subject.py or train_cross_subject.py instead.
#
# This script has been superseded by the two-stream architecture:
#   Per-subject models (GestureCNN, 120K params):
#       python train_per_subject.py
#       models saved to models/per_subject/
#
#   Cross-subject model (GestureCNNv2, 503K params, works on new participants):
#       python train_cross_subject.py
#       model saved to models/cross_subject/gesture_cnn_v2.pt
#
# This file is kept for reference during transition and will be removed.

import datetime as dt
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from libemg.utils import get_windows

from gesture_model_cnn import GestureCNN, GestureCNNv2


# ======== Config (edit as needed) ========
DATA_ROOT = Path("data")
PATTERN = "*_filtered.npz"
MODEL_OUT = Path("models") / "gesture_cnn_cross_subject.pt"  # only used when PER_SUBJECT_MODELS = False

WINDOW_SIZE = 200
WINDOW_STEP = 100

USE_CALIBRATION = True
MVC_PERCENTILE = 95.0
USE_MIN_LABEL_CONFIDENCE = True
MIN_LABEL_CONFIDENCE = 0.8

TEST_SIZE = 0.2
RANDOM_STATE = 42

BATCH_SIZE = 256
EPOCHS = 70
LR = 1e-4
DROPOUT = 0.3
KERNEL_SIZE = 11

USE_CLASS_WEIGHTS = True

# V2 architecture and training improvements
#
# MODE SELECTION — these two settings must be used together:
#   Cross-subject model (end goal): PER_SUBJECT_MODELS=False, USE_INSTANCE_NORM=True
#     → GestureCNNv2 (503K params) + InstanceNorm to strip inter-subject amplitude
#     → Trains on ALL subjects pooled; single model works on unseen subjects
#   Per-subject model (debugging/baseline): PER_SUBJECT_MODELS=True, USE_INSTANCE_NORM=False
#     → GestureCNN (120K params), no InstanceNorm, amplitude stays discriminative
#     → Requires sufficient sessions (≥6) per subject for 503K-param model
USE_INSTANCE_NORM = True      # Use GestureCNNv2 with built-in InstanceNorm
USE_AUGMENTATION = False       # EMG augmentation during training
AUG_PROB = 0.5                # Probability of applying each augmentation
USE_BALANCED_SAMPLING = True  # Weight sampler to balance subjects
USE_MIXUP = False             # Mixup augmentation (low impact, off by default)
MIXUP_ALPHA = 0.2

# Train one model per subject, saved as models/{subject}_gesture_cnn.pt.
# Set False to train a single cross-subject model that generalises to new subjects.
PER_SUBJECT_MODELS = False

# Leave-one-subject-out cross-validation (only meaningful when
# PER_SUBJECT_MODELS = False; requires at least 2 subjects).
# Run this to measure true cross-subject accuracy before deploying.
LOSO_EVAL = True

CV_ENABLED = False
CV_FOLDS = 6
CV_EPOCHS = 90
# ========================================


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


# ── Calibration ──────────────────────────────────────────────────────────────

def compute_calibration(neutral_emg, mvc_emg, percentile):
    neutral = np.asarray(neutral_emg, dtype=float)
    mvc = np.asarray(mvc_emg, dtype=float)
    if neutral.size == 0 or mvc.size == 0:
        return None, None
    neutral_mean = np.mean(neutral, axis=0)
    mvc_scale = np.percentile(mvc, percentile, axis=0)
    mvc_scale = np.where(mvc_scale < 1e-6, 1.0, mvc_scale)
    return neutral_mean, mvc_scale


def validate_calibration_data(files):
    """Warn about session files that are missing calibration arrays."""
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
            f"(calib_neutral_emg / calib_mvc_emg). "
            f"Calibration normalisation will be skipped for these sessions."
        )
        for fp in missing[:5]:
            print(f"  {fp}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more.")


# ── Data loading ─────────────────────────────────────────────────────────────

def subject_from_path(path: Path) -> str:
    """Extract subject ID from a filtered file path.

    Expected layout: data/{subject}/filtered/{name}_filtered.npz
    The subject directory is two levels above the file.
    """
    return path.parent.parent.name


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
        if (
            USE_MIN_LABEL_CONFIDENCE
            and lbl is not None
            and confidence < MIN_LABEL_CONFIDENCE
        ):
            lbl = None
        window_labels.append(lbl)

    window_labels = np.asarray(window_labels, dtype=object)
    keep = window_labels != None  # noqa: E711
    windows = windows[keep]
    window_labels = window_labels[keep]

    if windows.size == 0:
        return None
    return windows.astype(np.float32), window_labels


def load_dataset():
    files = sorted(DATA_ROOT.rglob(PATTERN))
    if not files:
        raise FileNotFoundError(f"No filtered files found under {DATA_ROOT}")

    if USE_CALIBRATION:
        validate_calibration_data(files)

    X_list = []
    y_list = []
    groups_list = []
    subjects_list = []
    channel_counts = []
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

    channel_counts = sorted(set(channel_counts))
    if len(channel_counts) != 1:
        raise ValueError(f"Channel count mismatch: {channel_counts}")

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    groups = np.concatenate(groups_list)
    subjects = np.concatenate(subjects_list)
    return X, y, groups, subjects, channel_counts[0]


# ── Normalisation ─────────────────────────────────────────────────────────────

def standardize_per_channel(X, mean, std):
    mean = mean.reshape(1, -1, 1)
    std = std.reshape(1, -1, 1)
    return (X - mean) / std


def _prepare_test_data(X, mean, std):
    """Return standardised test data, skipping z-score when InstanceNorm is used."""
    if USE_INSTANCE_NORM:
        return X.astype(np.float32)
    return standardize_per_channel(X, mean, std).astype(np.float32)


# ── Augmentation (plain functions, no classes) ────────────────────────────────

def _aug_amplitude(xb: np.ndarray, p: float) -> np.ndarray:
    """Scale each sample in the batch by an independent random factor."""
    mask = np.random.random(xb.shape[0]) < p
    if not mask.any():
        return xb
    factors = np.random.uniform(0.5, 2.0, size=(mask.sum(), 1, 1)).astype(np.float32)
    result = xb.copy()
    result[mask] *= factors
    return result


def _aug_noise(xb: np.ndarray, p: float) -> np.ndarray:
    """Add Gaussian noise at a random SNR between 10 and 30 dB."""
    mask = np.random.random(xb.shape[0]) < p
    if not mask.any():
        return xb
    result = xb.copy()
    snr_db = np.random.uniform(10.0, 30.0, size=mask.sum())
    snr_linear = 10.0 ** (snr_db / 10.0)
    signal_power = result[mask].var(axis=(1, 2), keepdims=True) + 1e-8
    noise_std = np.sqrt(signal_power / snr_linear[:, None, None]).astype(np.float32)
    result[mask] += np.random.randn(*result[mask].shape).astype(np.float32) * noise_std
    return result


def _aug_temporal_shift(xb: np.ndarray, p: float) -> np.ndarray:
    """Circularly roll each sample by a random offset up to +-20 samples."""
    mask = np.random.random(xb.shape[0]) < p
    if not mask.any():
        return xb
    shifts = np.random.randint(-20, 21, size=mask.sum())
    result = xb.copy()
    for i, shift in zip(np.where(mask)[0], shifts):
        result[i] = np.roll(result[i], shift, axis=-1)
    return result


def _aug_channel_dropout(xb: np.ndarray, p: float) -> np.ndarray:
    """Zero out one random channel per sample."""
    mask = np.random.random(xb.shape[0]) < p
    if not mask.any():
        return xb
    result = xb.copy()
    n_channels = xb.shape[1]
    drop_channels = np.random.randint(0, n_channels, size=mask.sum())
    for i, ch in zip(np.where(mask)[0], drop_channels):
        result[i, ch] = 0.0
    return result


def _aug_temporal_stretch(xb: np.ndarray, p: float) -> np.ndarray:
    """Stretch time axis by +-20% then crop/pad back to original length."""
    import torch.nn.functional as F

    mask = np.random.random(xb.shape[0]) < p
    if not mask.any():
        return xb
    result = xb.copy()
    T = xb.shape[2]
    factors = np.random.uniform(0.8, 1.2, size=mask.sum())
    for i, factor in zip(np.where(mask)[0], factors):
        new_len = max(1, int(T * factor))
        # F.interpolate handles all channels at once — no per-channel Python loop
        sub = torch.from_numpy(result[i : i + 1])  # (1, C, T)
        stretched = F.interpolate(sub, size=new_len, mode="linear", align_corners=False)
        if new_len >= T:
            result[i] = stretched[0, :, :T].numpy()
        else:
            result[i] = F.pad(stretched[0], (0, T - new_len), mode="replicate").numpy()
    return result


def augment_emg(xb: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Apply all EMG augmentations sequentially with probability p each."""
    xb = _aug_amplitude(xb, p)
    xb = _aug_noise(xb, p)
    xb = _aug_temporal_shift(xb, p)
    xb = _aug_channel_dropout(xb, p)
    xb = _aug_temporal_stretch(xb, p)
    return xb


# ── Subject-balanced sampling ─────────────────────────────────────────────────

def make_subject_sample_weights(subjects: np.ndarray) -> np.ndarray:
    """Return per-sample weights that up-weight underrepresented subjects."""
    unique, counts = np.unique(subjects, return_counts=True)
    weight_map = {s: 1.0 / c for s, c in zip(unique, counts)}
    return np.array([weight_map[s] for s in subjects], dtype=np.float32)


# ── Mixup ─────────────────────────────────────────────────────────────────────

def mixup_data(xb, yb, alpha: float = 0.2):
    """Return mixed inputs and both label tensors with mixing coefficient."""
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(xb.size(0), device=xb.device)
    mixed_x = lam * xb + (1.0 - lam) * xb[idx]
    return mixed_x, yb, yb[idx], lam


# ── Training ──────────────────────────────────────────────────────────────────

def _build_model(in_channels: int, num_classes: int, device) -> nn.Module:
    if USE_INSTANCE_NORM:
        return GestureCNNv2(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=DROPOUT,
        ).to(device)
    channels = [in_channels, 32, 64, 128]
    return GestureCNN(
        channels=channels,
        num_classes=num_classes,
        dropout=DROPOUT,
        kernel_size=KERNEL_SIZE,
    ).to(device)


def train_eval_split(
    X_train, y_train, X_eval, y_eval,
    channels, num_classes, epochs, device,
    subjects_train=None,
):
    mean = X_train.mean(axis=(0, 2))
    std = X_train.std(axis=(0, 2))
    std = np.where(std < 1e-6, 1.0, std)

    if USE_INSTANCE_NORM:
        X_train_t = X_train.astype(np.float32)
        X_eval_t = X_eval.astype(np.float32)
    else:
        X_train_t = standardize_per_channel(X_train, mean, std).astype(np.float32)
        X_eval_t = standardize_per_channel(X_eval, mean, std).astype(np.float32)

    train_ds = TensorDataset(
        torch.from_numpy(X_train_t), torch.from_numpy(y_train)
    )
    eval_ds = TensorDataset(torch.from_numpy(X_eval_t), torch.from_numpy(y_eval))

    if USE_BALANCED_SAMPLING and subjects_train is not None:
        weights = make_subject_sample_weights(subjects_train)
        sampler = WeightedRandomSampler(
            torch.from_numpy(weights), num_samples=len(weights), replacement=True
        )
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, sampler=sampler, drop_last=False,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
            pin_memory=True,
        )
    eval_loader = DataLoader(
        eval_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
        pin_memory=True,
    )

    model = _build_model(int(channels[0]), num_classes, device)

    class_weight = None
    if USE_CLASS_WEIGHTS:
        class_counts = np.bincount(y_train, minlength=num_classes)
        weights_cls = class_counts.sum() / np.maximum(class_counts, 1)
        weights_cls = weights_cls / weights_cls.mean()
        class_weight = torch.tensor(weights_cls, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, threshold=1e-4, min_lr=1e-6,
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
            # Augment on CPU before transferring to GPU — avoids GPU→CPU sync
            if USE_AUGMENTATION:
                xb = torch.from_numpy(augment_emg(xb.numpy(), p=AUG_PROB))
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            if USE_MIXUP:
                xb_mixed, yb_a, yb_b, lam = mixup_data(xb, yb, MIXUP_ALPHA)
                logits = model(xb_mixed)
                loss = lam * criterion(logits, yb_a) + (1.0 - lam) * criterion(logits, yb_b)
            else:
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
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, mean, std, best_acc


# ── LOSO evaluation ───────────────────────────────────────────────────────────

def loso_evaluate(X, y_idx, subjects, channel_count, num_classes, device, index_to_label):
    """Leave-one-subject-out cross-validation for cross-subject generalisation."""
    unique_subjects = sorted(np.unique(subjects))
    if len(unique_subjects) < 2:
        print("LOSO requires at least 2 subjects — skipping.")
        return

    print(f"\nLOSO evaluation over {len(unique_subjects)} subjects:")
    loso_accs = []
    channels = [int(channel_count), 32, 64, 128]
    for held_out in unique_subjects:
        train_mask = subjects != held_out
        test_mask = subjects == held_out
        print(
            f"  Held-out: {held_out}  "
            f"(train {train_mask.sum()}, test {test_mask.sum()})"
        )
        model, mean, std, _ = train_eval_split(
            X[train_mask], y_idx[train_mask],
            X[test_mask], y_idx[test_mask],
            channels, num_classes, EPOCHS, device,
            subjects_train=subjects[train_mask],
        )
        model.eval()
        X_test = _prepare_test_data(X[test_mask], mean, std)
        test_ds = TensorDataset(
            torch.from_numpy(X_test), torch.from_numpy(y_idx[test_mask])
        )
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        all_preds = []
        with torch.no_grad():
            for xb, _ in test_loader:
                logits = model(xb.to(device))
                all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        y_pred = np.concatenate(all_preds)
        acc = float(accuracy_score(y_idx[test_mask], y_pred))
        loso_accs.append(acc)
        print(f"    Accuracy: {acc:.3f}")

    loso_accs = np.asarray(loso_accs, dtype=float)
    print(
        f"\nLOSO summary: mean={loso_accs.mean():.3f}  "
        f"std={loso_accs.std():.3f}  "
        f"min={loso_accs.min():.3f}  "
        f"max={loso_accs.max():.3f}"
    )


# ── Save bundle ───────────────────────────────────────────────────────────────

def _train_and_save(
    X, y_idx, groups, subjects,
    channels, num_classes, device,
    labels, label_to_index, index_to_label, channel_count,
    model_out, subject_tag,
):
    """Split, train, evaluate, and save a model bundle for the given data slice."""
    unique_groups = np.unique(groups)
    print(f"{X.shape[0]} windows across {len(unique_groups)} session file(s).")

    if unique_groups.size >= 2:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        train_idx, test_idx = next(splitter.split(X, y_idx, groups))
        split_mode = "group-file"
    else:
        indices = np.arange(X.shape[0])
        train_idx, test_idx = train_test_split(
            indices,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_idx,
        )
        split_mode = "stratified-random"

    train_files = sorted({str(g) for g in groups[train_idx]})
    test_files = sorted({str(g) for g in groups[test_idx]})
    print(f"Train ({len(train_files)} files): {[Path(f).name for f in train_files]}")
    print(f"Test  ({len(test_files)} files):  {[Path(f).name for f in test_files]}")

    arch_label = "GestureCNNv2" if USE_INSTANCE_NORM else "GestureCNN"
    print(
        f"\nTraining {arch_label} for {EPOCHS} epochs (batch {BATCH_SIZE}) "
        f"on {len(train_idx)} windows; testing on {len(test_idx)} windows."
    )
    model, mean, std, _ = train_eval_split(
        X[train_idx],
        y_idx[train_idx],
        X[test_idx],
        y_idx[test_idx],
        channels,
        num_classes,
        EPOCHS,
        device,
        subjects_train=subjects[train_idx],
    )

    # Final test evaluation
    model.eval()
    X_test = _prepare_test_data(X[test_idx], mean, std)
    test_ds = TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_idx[test_idx])
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    all_preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            logits = model(xb.to(device))
            all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    y_pred = np.concatenate(all_preds)
    y_test = y_idx[test_idx]

    test_accuracy = float(accuracy_score(y_test, y_pred))
    report = classification_report(
        y_test, y_pred, target_names=[index_to_label[i] for i in range(len(labels))]
    )
    print(f"\nFinal test accuracy: {test_accuracy:.3f}")
    print("\nReport:\n", report)

    test_subjects = subjects[test_idx]
    unique_test_subjects = np.unique(test_subjects)
    if len(unique_test_subjects) > 1:
        print("Per-subject test accuracy:")
        for subj in sorted(unique_test_subjects):
            mask = test_subjects == subj
            subj_acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"  {subj}: {subj_acc:.3f}  ({mask.sum()} windows)")

    if USE_INSTANCE_NORM:
        arch_dict = {
            "type": "GestureCNNv2",
            "in_channels": int(channel_count),
            "dropout": float(DROPOUT),
        }
    else:
        arch_dict = {
            "type": "GestureCNN",
            "channels": channels,
            "dropout": float(DROPOUT),
            "kernel_size": int(KERNEL_SIZE),
        }

    meta = {
        "created_at": dt.datetime.now().isoformat(),
        "model_type": "cnn",
        "arch_type": arch_label,
        "subject": subject_tag,
        "window_size_samples": WINDOW_SIZE,
        "window_step_samples": WINDOW_STEP,
        "channel_count": int(channel_count),
        "labels": labels,
        "split_mode": split_mode,
        "test_size": float(TEST_SIZE),
        "calibration_used": bool(USE_CALIBRATION),
        "calibration_mvc_percentile": float(MVC_PERCENTILE),
        "label_confidence_filter": {
            "enabled": bool(USE_MIN_LABEL_CONFIDENCE),
            "min_label_confidence": float(MIN_LABEL_CONFIDENCE),
        },
        "training": {
            "epochs": int(EPOCHS),
            "batch_size": int(BATCH_SIZE),
            "lr": float(LR),
            "class_weights": bool(USE_CLASS_WEIGHTS),
            "use_instance_norm": bool(USE_INSTANCE_NORM),
            "use_augmentation": bool(USE_AUGMENTATION),
            "use_balanced_sampling": bool(USE_BALANCED_SAMPLING),
            "use_mixup": bool(USE_MIXUP),
        },
        "metrics": {
            "test_accuracy": test_accuracy,
        },
        "train_files": [Path(f).name for f in train_files],
        "test_files": [Path(f).name for f in test_files],
        "use_instance_norm_input": bool(USE_INSTANCE_NORM),
    }

    bundle = {
        "model_state": model.state_dict(),
        "normalization": {"mean": mean, "std": std},
        "label_to_index": label_to_index,
        "index_to_label": index_to_label,
        "metadata": meta,
        "architecture": arch_dict,
    }

    model_out.parent.mkdir(parents=True, exist_ok=True)
    if model_out.exists():
        print(f"Warning: overwriting existing model at {model_out}")
    torch.save(bundle, model_out)
    print(f"Saved model bundle to {model_out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    # Config sanity check — catch the common mismatch before wasting GPU time
    if USE_INSTANCE_NORM and PER_SUBJECT_MODELS:
        print(
            "WARNING: USE_INSTANCE_NORM=True with PER_SUBJECT_MODELS=True is a mismatch.\n"
            "  GestureCNNv2 (503K params) is overparameterised for single-subject data.\n"
            "  InstanceNorm strips within-subject amplitude info that helps per-subject.\n"
            "  For per-subject models: set USE_INSTANCE_NORM=False (uses GestureCNN, 120K).\n"
            "  For cross-subject model: set PER_SUBJECT_MODELS=False (uses all subjects).\n"
            "  Continuing anyway, but accuracy will likely be poor for low-session subjects."
        )

    X, y, groups, subjects, channel_count = load_dataset()
    unique_subjects = sorted(np.unique(subjects))
    print(
        f"Loaded {X.shape[0]} windows, {channel_count} channels, "
        f"{len(np.unique(y))} classes, {len(unique_subjects)} subject(s): "
        f"{unique_subjects}"
    )

    # All per-subject models share the same global label contract so the
    # realtime script can load any bundle and get consistent class indices.
    labels = sorted({str(lbl) for lbl in np.unique(y)})
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    y_idx = np.array([label_to_index[str(lbl)] for lbl in y], dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels = [int(channel_count), 32, 64, 128]
    num_classes = len(labels)

    if LOSO_EVAL and not PER_SUBJECT_MODELS:
        loso_evaluate(X, y_idx, subjects, channel_count, num_classes, device, index_to_label)

    if PER_SUBJECT_MODELS:
        for subject in unique_subjects:
            print(f"\n{'=' * 55}")
            print(f"Training per-subject model: {subject}")
            mask = subjects == subject
            _train_and_save(
                X[mask], y_idx[mask], groups[mask], subjects[mask],
                channels, num_classes, device,
                labels, label_to_index, index_to_label, channel_count,
                model_out=Path("models") / f"{subject}_gesture_cnn.pt",
                subject_tag=subject,
            )
        print(f"\n{'=' * 55}")
        print("Per-subject training complete. To run realtime inference:")
        for subject in unique_subjects:
            print(f"  python realtime_gesture_cnn.py --model models/{subject}_gesture_cnn.pt")

    else:
        cv_scores = None
        unique_file_groups = np.unique(groups)
        if CV_ENABLED and unique_file_groups.size >= 2:
            from sklearn.model_selection import GroupKFold

            splits = min(CV_FOLDS, int(unique_file_groups.size))
            if splits >= 2:
                cv = GroupKFold(n_splits=splits)
                cv_scores = []
                for fold, (train_idx, val_idx) in enumerate(
                    cv.split(X, y_idx, groups), start=1
                ):
                    print(f"\nCV fold {fold}/{splits}")
                    _, _, _, fold_acc = train_eval_split(
                        X[train_idx],
                        y_idx[train_idx],
                        X[val_idx],
                        y_idx[val_idx],
                        channels,
                        num_classes,
                        CV_EPOCHS,
                        device,
                    )
                    cv_scores.append(fold_acc)
                cv_scores = np.asarray(cv_scores, dtype=float)
                print(
                    f"\nCV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}"
                )

        print("\nFinal global train/test split")
        _train_and_save(
            X, y_idx, groups, subjects,
            channels, num_classes, device,
            labels, label_to_index, index_to_label, channel_count,
            model_out=MODEL_OUT,
            subject_tag=None,
        )


if __name__ == "__main__":
    main()
