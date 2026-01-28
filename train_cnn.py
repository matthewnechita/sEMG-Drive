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

from gesture_model_cnn import GestureCNN


# ======== Config (edit as needed) ========
DATA_ROOT = Path("data")
PATTERN = "*_filtered.npz"
MODEL_OUT = Path("models") / "gesture_cnn.pt"

WINDOW_SIZE = 200
WINDOW_STEP = 100

USE_CALIBRATION = True
MVC_PERCENTILE = 95.0

TEST_SIZE = 0.2
RANDOM_STATE = 42

BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-3
DROPOUT = 0.2
KERNEL_SIZE = 7

USE_CLASS_WEIGHTS = True

CV_ENABLED = True
CV_FOLDS = 3
CV_EPOCHS = 5
# ========================================


def majority_label(segment):
    if segment.size == 0:
        return None
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
            return None
    if flat.dtype.kind in "fc":
        flat = flat[~np.isnan(flat)]
    if flat.size == 0:
        return None
    values, counts = np.unique(flat, return_counts=True)
    return values[counts.argmax()]


def compute_calibration(neutral_emg, mvc_emg, percentile):
    neutral = np.asarray(neutral_emg, dtype=float)
    mvc = np.asarray(mvc_emg, dtype=float)
    if neutral.size == 0 or mvc.size == 0:
        return None, None
    neutral_mean = np.mean(neutral, axis=0)
    mvc_scale = np.percentile(mvc, percentile, axis=0)
    mvc_scale = np.where(mvc_scale < 1e-6, 1.0, mvc_scale)
    return neutral_mean, mvc_scale


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
        lbl = majority_label(labels[s:e])
        if lbl == "neutral_buffer":
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

    X_list = []
    y_list = []
    groups_list = []
    channel_counts = []
    for fp in files:
        result = load_windows_from_file(fp)
        if result is None:
            continue
        windows, labels = result
        X_list.append(windows)
        y_list.append(labels)
        groups_list.append(np.array([str(fp)] * len(labels), dtype=object))
        channel_counts.append(int(windows.shape[1]))

    if not X_list:
        raise ValueError("No labeled windows found in filtered files.")

    channel_counts = sorted(set(channel_counts))
    if len(channel_counts) != 1:
        raise ValueError(f"Channel count mismatch: {channel_counts}")

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    groups = np.concatenate(groups_list)
    return X, y, groups, channel_counts[0]


def standardize_per_channel(X, mean, std):
    mean = mean.reshape(1, -1, 1)
    std = std.reshape(1, -1, 1)
    return (X - mean) / std


def train_eval_split(
    X_train, y_train, X_eval, y_eval, channels, num_classes, epochs, device
):
    mean = X_train.mean(axis=(0, 2))
    std = X_train.std(axis=(0, 2))
    std = np.where(std < 1e-6, 1.0, std)

    X_train = standardize_per_channel(X_train, mean, std).astype(np.float32)
    X_eval = standardize_per_channel(X_eval, mean, std).astype(np.float32)

    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    eval_ds = TensorDataset(torch.from_numpy(X_eval), torch.from_numpy(y_eval))

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )

    model = GestureCNN(
        channels=channels,
        num_classes=num_classes,
        dropout=DROPOUT,
        kernel_size=KERNEL_SIZE,
    ).to(device)

    class_weight = None
    if USE_CLASS_WEIGHTS:
        class_counts = np.bincount(y_train, minlength=num_classes)
        weights = class_counts.sum() / np.maximum(class_counts, 1)
        weights = weights / weights.mean()
        class_weight = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

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
            xb = xb.to(device)
            yb = yb.to(device)
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
        with torch.no_grad():
            for xb, yb in eval_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1)
                eval_correct += (preds == yb).sum().item()
                eval_total += xb.size(0)

        train_acc = train_correct / max(train_total, 1)
        eval_acc = eval_correct / max(eval_total, 1)
        avg_loss = train_loss / max(train_total, 1)
        print(
            f"Epoch {epoch:02d} | loss {avg_loss:.4f} | train {train_acc:.3f} | eval {eval_acc:.3f}"
        )
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


def main():
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    X, y, groups, channel_count = load_dataset()
    print(
        f"Loaded {X.shape[0]} windows, {channel_count} channels, "
        f"{len(np.unique(y))} classes."
    )

    labels = sorted({str(lbl) for lbl in np.unique(y)})
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    y_idx = np.array([label_to_index[str(lbl)] for lbl in y], dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels = [int(channel_count), 32, 64, 128]
    num_classes = len(labels)

    cv_scores = None
    unique_groups = np.unique(groups)
    if CV_ENABLED and unique_groups.size >= 2:
        from sklearn.model_selection import GroupKFold

        splits = min(CV_FOLDS, int(unique_groups.size))
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

    print("\nFinal train/test split")
    print(
        f"Training CNN for {EPOCHS} epochs (batch {BATCH_SIZE}) "
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
    )

    model.eval()
    X_test = standardize_per_channel(
        X[test_idx], mean, std
    ).astype(np.float32)
    test_ds = TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_idx[test_idx])
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )
    all_preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.append(preds)
    y_pred = np.concatenate(all_preds)
    y_test = y_idx[test_idx]
    test_accuracy = float(accuracy_score(y_test, y_pred))
    report = classification_report(
        y_test, y_pred, target_names=[index_to_label[i] for i in range(len(labels))]
    )
    print(f"Final test accuracy: {test_accuracy:.3f}")
    print("\nReport:\n", report)

    meta = {
        "created_at": dt.datetime.now().isoformat(),
        "model_type": "cnn",
        "window_size_samples": WINDOW_SIZE,
        "window_step_samples": WINDOW_STEP,
        "channel_count": int(channel_count),
        "labels": labels,
        "split_mode": split_mode,
        "test_size": float(TEST_SIZE),
        "calibration_used": bool(USE_CALIBRATION),
        "calibration_mvc_percentile": float(MVC_PERCENTILE),
        "training": {
            "epochs": int(EPOCHS),
            "batch_size": int(BATCH_SIZE),
            "lr": float(LR),
            "class_weights": bool(USE_CLASS_WEIGHTS),
        },
        "metrics": {
            "test_accuracy": test_accuracy,
            "cv_accuracy_mean": float(cv_scores.mean()) if cv_scores is not None else None,
            "cv_accuracy_std": float(cv_scores.std()) if cv_scores is not None else None,
        },
    }

    bundle = {
        "model_state": model.state_dict(),
        "normalization": {"mean": mean, "std": std},
        "label_to_index": label_to_index,
        "index_to_label": index_to_label,
        "metadata": meta,
        "architecture": {
            "type": "GestureCNN",
            "channels": channels,
            "dropout": float(DROPOUT),
            "kernel_size": int(KERNEL_SIZE),
        },
    }

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_OUT.exists():
        print(f"Warning: overwriting existing model at {MODEL_OUT}")
    torch.save(bundle, MODEL_OUT)
    print(f"Saved model bundle to {MODEL_OUT}")


if __name__ == "__main__":
    main()
