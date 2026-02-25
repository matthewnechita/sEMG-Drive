"""Per-subject EMG gesture classifier training — Stream 1.

Uses GestureCNN (120K params, z-score normalisation, no InstanceNorm).
Trains one model per subject; models are saved to models/per_subject/.

Usage:
    python train_per_subject.py

To run realtime inference with a per-subject model:
    python realtime_gesture_cnn.py --model models/per_subject/Matthew_cnn.pt
"""
import datetime as dt
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from libemg.utils import get_windows
from emg.gesture_model_cnn import GestureCNN


# ======== Config ========
DATA_ROOT  = Path("data")
PATTERN    = "*_filtered.npz"
MODEL_DIR  = Path("models/per_subject")

WINDOW_SIZE = 200
WINDOW_STEP = 100

USE_CALIBRATION        = True
MVC_PERCENTILE         = 95.0
USE_MIN_LABEL_CONFIDENCE = True
MIN_LABEL_CONFIDENCE   = 0.8

TEST_SIZE     = 0.2
RANDOM_STATE  = 42
BATCH_SIZE    = 512

# Per-subject hyperparameters
# GestureCNN at 120K params fits well with 11K–23K per-subject training windows.
EPOCHS   = 80      # convergence-appropriate for smaller model + per-subject data volume
LR       = 3e-4    # slightly higher LR than cross-subject; smaller model, smaller dataset
DROPOUT  = 0.2     # less regularisation; GestureCNN is already compact
KERNEL_SIZE = 7    # original GestureCNN kernel

USE_CLASS_WEIGHTS = True

# Augmentation — narrow amplitude range matches within-subject EMG variance.
# Between-subject amplitude can vary 5–10×; within-subject varies ~30%.
USE_AUGMENTATION = True
AMP_RANGE = (0.7, 1.4)   # within-subject amplitude variance range
AUG_PROB  = 0.4           # slightly lower probability than cross-subject

# Subjects to skip entirely (e.g. data quality issues).
# Per-subject training handles each subject independently so exclusion here
# simply means no model is produced for that subject.
EXCLUDED_SUBJECTS: list[str] = []
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

    if EXCLUDED_SUBJECTS:
        files = [f for f in files if subject_from_path(f) not in EXCLUDED_SUBJECTS]
        print(f"Excluded subjects: {EXCLUDED_SUBJECTS}")

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

def standardize_per_channel(X, mean, std):
    return (X - mean.reshape(1, -1, 1)) / std.reshape(1, -1, 1)


def _prepare_test_data(X, mean, std):
    return standardize_per_channel(X, mean, std).astype(np.float32)


# ── Augmentation (GPU-native) ─────────────────────────────────────────────────
# AMP_RANGE = (0.7, 1.4) is narrowed from the cross-subject (0.5, 2.0) because
# within-subject EMG amplitude is much more consistent than between-subject.

import torch.nn.functional as _F

def augment_emg_gpu(xb: torch.Tensor, p: float) -> torch.Tensor:
    """In-place-safe GPU augmentation. xb: (B, C, T) float32 on device."""
    B, C, T = xb.shape
    dev = xb.device
    xb = xb.clone()

    # 1. Amplitude scaling
    mask = torch.rand(B, device=dev) < p
    if mask.any():
        n = int(mask.sum())
        factors = torch.empty(n, 1, 1, device=dev).uniform_(AMP_RANGE[0], AMP_RANGE[1])
        xb[mask] = xb[mask] * factors

    # 2. Additive Gaussian noise (SNR-calibrated)
    mask = torch.rand(B, device=dev) < p
    if mask.any():
        n = int(mask.sum())
        snr_db     = torch.empty(n, device=dev).uniform_(10.0, 30.0)
        snr_linear = 10.0 ** (snr_db / 10.0)
        sig_power  = xb[mask].var(dim=(1, 2)).clamp(min=1e-8)
        noise_std  = (sig_power / snr_linear).sqrt().view(n, 1, 1)
        xb[mask]   = xb[mask] + torch.randn(n, C, T, device=dev) * noise_std

    # 3. Temporal shift (vectorised gather)
    mask = torch.rand(B, device=dev) < p
    if mask.any():
        n      = int(mask.sum())
        shifts = torch.randint(-20, 21, (n,), device=dev)
        idx    = (torch.arange(T, device=dev).unsqueeze(0) - shifts.unsqueeze(1)) % T
        idx    = idx.unsqueeze(1).expand(-1, C, -1)
        xb[mask] = torch.gather(xb[mask], 2, idx)

    # 4. Channel dropout (vectorised scatter)
    mask = torch.rand(B, device=dev) < p
    if mask.any():
        n       = int(mask.sum())
        drop_ch = torch.randint(0, C, (n,), device=dev)
        ch_mask = torch.zeros(n, C, device=dev, dtype=torch.bool)
        ch_mask.scatter_(1, drop_ch.unsqueeze(1), True)
        xb[mask] = xb[mask].masked_fill(ch_mask.unsqueeze(2), 0.0)

    # 5. Temporal stretch (single batched interpolation — one factor per batch)
    if torch.rand(1, device=dev).item() < p:
        factor  = torch.empty(1, device=dev).uniform_(0.85, 1.15).item()
        new_len = max(1, int(T * factor))
        stretched = _F.interpolate(xb, size=new_len, mode="linear", align_corners=False)
        if new_len >= T:
            xb = stretched[:, :, :T]
        else:
            xb = _F.pad(stretched, (0, T - new_len), mode="replicate")

    return xb


# ── Model ─────────────────────────────────────────────────────────────────────

def _build_model(in_channels: int, num_classes: int, device) -> nn.Module:
    """Always GestureCNN for per-subject training."""
    return GestureCNN(
        channels=[in_channels, 32, 64, 128],
        num_classes=num_classes,
        dropout=DROPOUT,
        kernel_size=KERNEL_SIZE,
    ).to(device)


# ── Training ──────────────────────────────────────────────────────────────────

def train_eval_split(X_train, y_train, X_eval, y_eval, channels, num_classes, epochs, device):
    mean = X_train.mean(axis=(0, 2))
    std  = X_train.std(axis=(0, 2))
    std  = np.where(std < 1e-6, 1.0, std)

    X_train_t = standardize_per_channel(X_train, mean, std).astype(np.float32)
    X_eval_t  = standardize_per_channel(X_eval,  mean, std).astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_t), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=False, pin_memory=True,
    )
    eval_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_eval_t), torch.from_numpy(y_eval)),
        batch_size=BATCH_SIZE, shuffle=False, drop_last=False, pin_memory=True,
    )

    model = _build_model(int(channels[0]), num_classes, device)

    class_weight = None
    if USE_CLASS_WEIGHTS:
        class_counts = np.bincount(y_train, minlength=num_classes)
        w = class_counts.sum() / np.maximum(class_counts, 1)
        class_weight = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # patience=8 is generous relative to per-subject data volume (fewer batches per epoch)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, threshold=1e-4, min_lr=1e-6,
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
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, mean, std, best_acc


# ── Save bundle ───────────────────────────────────────────────────────────────

def _train_and_save(X, y_idx, groups, subjects, channels, num_classes, device,
                    labels, label_to_index, index_to_label, channel_count,
                    model_out, subject_tag):
    unique_groups = np.unique(groups)
    print(f"{X.shape[0]} windows across {len(unique_groups)} session file(s).")

    if unique_groups.size >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        train_idx, test_idx = next(splitter.split(X, y_idx, groups))
        split_mode = "group-file"
    else:
        indices = np.arange(X.shape[0])
        train_idx, test_idx = train_test_split(
            indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_idx,
        )
        split_mode = "stratified-random"

    train_files = sorted({str(g) for g in groups[train_idx]})
    test_files  = sorted({str(g) for g in groups[test_idx]})
    print(f"Train ({len(train_files)} files): {[Path(f).name for f in train_files]}")
    print(f"Test  ({len(test_files)} files):  {[Path(f).name for f in test_files]}")
    print(f"\nTraining GestureCNN for {EPOCHS} epochs on {len(train_idx)} windows.")

    model, mean, std, _ = train_eval_split(
        X[train_idx], y_idx[train_idx],
        X[test_idx],  y_idx[test_idx],
        channels, num_classes, EPOCHS, device,
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

    test_accuracy = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred,
                                   target_names=[index_to_label[i] for i in range(len(labels))])
    print(f"\nTest accuracy: {test_accuracy:.3f}")
    print("\nReport:\n", report)

    bundle = {
        "model_state": model.state_dict(),
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
        "label_to_index": label_to_index,
        "index_to_label": index_to_label,
        "architecture": {
            "type": "GestureCNN",
            "channels": channels,
            "dropout": float(DROPOUT),
            "kernel_size": int(KERNEL_SIZE),
        },
        "metadata": {
            "created_at": dt.datetime.now().isoformat(),
            "stream": "per_subject",
            "subject": subject_tag,
            "window_size_samples": WINDOW_SIZE,
            "window_step_samples": WINDOW_STEP,
            "channel_count": int(channel_count),
            "labels": labels,
            "split_mode": split_mode,
            "test_size": float(TEST_SIZE),
            "calibration_used": bool(USE_CALIBRATION),
            "calibration_mvc_percentile": float(MVC_PERCENTILE),
            "use_instance_norm_input": False,
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
                "use_mixup": False,
            },
            "metrics": {"test_accuracy": test_accuracy},
            "train_files": [Path(f).name for f in train_files],
            "test_files":  [Path(f).name for f in test_files],
        },
    }

    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, model_out)
    print(f"Saved to {model_out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
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
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    y_idx          = np.array([label_to_index[str(lbl)] for lbl in y], dtype=np.int64)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels = [int(channel_count), 32, 64, 128]
    num_classes = len(labels)

    for subject in unique_subjects:
        print(f"\n{'=' * 55}")
        print(f"Per-subject model: {subject}")
        mask = subjects == subject
        _train_and_save(
            X[mask], y_idx[mask], groups[mask], subjects[mask],
            channels, num_classes, device,
            labels, label_to_index, index_to_label, channel_count,
            model_out=MODEL_DIR / f"{subject}_cnn.pt",
            subject_tag=subject,
        )

    print(f"\n{'=' * 55}")
    print("Per-subject training complete.")
    for subject in unique_subjects:
        print(f"  python realtime_gesture_cnn.py --model models/per_subject/{subject}_cnn.pt")


if __name__ == "__main__":
    main()
