import copy
import datetime as dt
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as _F
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from emg.cnn_training import (
    build_architecture_metadata,
    build_model_metadata,
    build_model,
    build_training_objective,
    compute_training_step,
    prepare_train_eval_inputs,
    prepare_train_inputs,
    standardize_windows,
)
from emg.eval_utils import (
    compute_eval_artifacts,
    print_eval_summary,
    scalar_summary,
    serialize_eval_metrics,
)
from emg.strict_layout import strict_channel_count_for_arm, strict_layout_bundle_metadata
from emg.training_data import (
    DEFAULT_MVC_MIN_RATIO,
    load_strict_windows_from_file,
    print_missing_calibration_warning,
    subject_from_path,
    validate_calibration_data,
)
from project_paths import STRICT_MODELS_ROOT, STRICT_RESAMPLED_ROOT, strict_arm_root


# ======== Config ========
ARM = "right"  # set to "right" or "left" before running
TARGET_SUBJECT = "Matthew"

DATA_ROOT = strict_arm_root(STRICT_RESAMPLED_ROOT, ARM)
PATTERN = "*_filtered.npz"
MODEL_OUT = STRICT_MODELS_ROOT / "per_subject" / ARM / f"{TARGET_SUBJECT}v6_4_gestures.pt"

WINDOW_SIZE = 200
WINDOW_STEP = 100

USE_CALIBRATION = True
MVC_PERCENTILE = 95.0
USE_MIN_LABEL_CONFIDENCE = True
MIN_LABEL_CONFIDENCE = 0.85

RANDOM_STATE = 42
BATCH_SIZE = 512
# Session-grouped k-fold CV for unbiased per-subject metrics.
PER_SUBJECT_CV_FOLDS = 2

EPOCHS = 40
LR = 1e-4
DROPOUT = 0.25
LABEL_SMOOTHING = 0.05

USE_AUGMENTATION = True
AMP_RANGE = (0.7, 1.4)
AUG_PROB = 0.4

EXCLUDED_SUBJECTS: list[str] = []  # Keep empty for Matthew-only runs unless you need to blacklist a subject.
INCLUDED_GESTURES: set[str] | None = {"neutral", "left_turn", "right_turn", "horn"}  # Example subset: {"neutral", "left_turn", "right_turn"}.
# ========================

MVC_QUALITY_MIN_RATIO = DEFAULT_MVC_MIN_RATIO


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
        print_missing_calibration_warning(validate_calibration_data(files))

    X_list, y_list, groups_list, subjects_list, channel_counts = [], [], [], [], []
    layout_sources: dict[str, int] = {}
    strict_pair_order = None
    strict_slot_order = None
    strict_channel_counts = None
    for fp in files:
        windowed = load_strict_windows_from_file(
            fp,
            arm=ARM,
            window_size=WINDOW_SIZE,
            window_step=WINDOW_STEP,
            use_calibration=USE_CALIBRATION,
            mvc_percentile=MVC_PERCENTILE,
            mvc_min_ratio=MVC_QUALITY_MIN_RATIO,
            use_min_label_confidence=USE_MIN_LABEL_CONFIDENCE,
            min_label_confidence=MIN_LABEL_CONFIDENCE,
            included_gestures=INCLUDED_GESTURES,
            verbose_calibration_skip=True,
        )
        if windowed is None:
            continue
        windows = windowed.windows
        labels = windowed.labels
        strict_layout = windowed.strict_layout
        X_list.append(windows)
        y_list.append(labels)
        groups_list.append(np.array([str(fp)] * len(labels), dtype=object))
        subjects_list.append(np.array([subject_from_path(fp)] * len(labels), dtype=object))
        channel_counts.append(int(windows.shape[1]))
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
        layout_sources,
    )


# -- Normalisation -------------------------------------------------------------

def _prepare_test_data(X, _mean, _std):
    return standardize_windows(X, _mean, _std)


# -- Augmentation --------------------------------------------------------------

def augment_emg_gpu(
    xb: torch.Tensor,
    p: float,
) -> torch.Tensor:
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
    return build_model(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout=DROPOUT,
        device=device,
    )


def train_eval_split(
    X_train, y_train, X_eval, y_eval,
    channel_count, num_classes, epochs, device,
):
    X_train_t, X_eval_t, mean, std = prepare_train_eval_inputs(X_train, X_eval)

    train_ds = TensorDataset(torch.from_numpy(X_train_t), torch.from_numpy(y_train))
    eval_ds = TensorDataset(torch.from_numpy(X_eval_t), torch.from_numpy(y_eval))

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, pin_memory=True
    )

    eval_loader = DataLoader(
        eval_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, pin_memory=True
    )

    model = _build_model(int(channel_count), num_classes, device)

    objective = build_training_objective(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, threshold=1e-4, min_lr=1e-6
    )

    best_state = None
    best_acc = -1.0
    best_epoch = 1
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
            logits, loss = compute_training_step(
                model,
                xb,
                yb,
                objective=objective,
            )
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
                logits, loss = compute_training_step(
                    model,
                    xb,
                    yb,
                    objective=objective,
                )
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
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, mean, std, best_acc, best_epoch


def train_full_dataset(
    X_train,
    y_train,
    channel_count,
    num_classes,
    epochs,
    device,
):
    X_train_t, mean, std = prepare_train_inputs(X_train)
    train_ds = TensorDataset(torch.from_numpy(X_train_t), torch.from_numpy(y_train))
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, pin_memory=True
    )

    model = _build_model(int(channel_count), num_classes, device)
    objective = build_training_objective(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, threshold=1e-4, min_lr=1e-6
    )

    best_state = None
    best_loss = float("inf")
    best_epoch = 1
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
            logits, loss = compute_training_step(
                model,
                xb,
                yb,
                objective=objective,
            )
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == yb).sum().item()
            train_total += xb.size(0)

        train_acc = train_correct / max(train_total, 1)
        avg_loss = train_loss / max(train_total, 1)
        print(f"Final fit epoch {epoch:02d} | loss {avg_loss:.4f} | train {train_acc:.3f}")

        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(avg_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < prev_lr:
            print(f"LR reduced: {prev_lr:.2e} -> {new_lr:.2e}")

        if epoch == 1:
            epoch_time = time.time() - epoch_start
            est_total = epoch_time * epochs
            est_remaining = max(0.0, est_total - (time.time() - start_time))
            print(f"Estimated remaining time (final fit): ~{est_remaining:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, mean, std, float(best_loss), int(best_epoch)


def _predict_labels(model, X, device, mean, std):
    model.eval()
    X_test = _prepare_test_data(X, mean, std)
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    all_preds = []
    with torch.no_grad():
        for (xb,) in test_loader:
            all_preds.append(torch.argmax(model(xb.to(device)), dim=1).cpu().numpy())
    if not all_preds:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(all_preds).astype(np.int64, copy=False)


def _run_grouped_cross_validation(
    X,
    y_idx,
    groups,
    channel_count,
    num_classes,
    device,
    labels,
    index_to_label,
):
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        raise ValueError(
            "Per-subject grouped cross-validation requires at least 2 session files."
        )

    class_group_counts = [
        np.unique(groups[y_idx == class_idx]).size for class_idx in np.unique(y_idx)
    ]
    max_supported_splits = min(int(unique_groups.size), *[int(v) for v in class_group_counts])
    n_splits = min(int(PER_SUBJECT_CV_FOLDS), max_supported_splits)
    if n_splits < 2:
        raise ValueError(
            "Per-subject grouped cross-validation requires each class to appear in at least "
            "2 distinct session files."
        )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    oof_pred = np.full(y_idx.shape, -1, dtype=np.int64)
    fold_results = []

    print(f"\n{'=' * 55}")
    print(
        f"Per-subject grouped cross-validation: {n_splits} folds over "
        f"{len(unique_groups)} session file(s)."
    )

    for fold_idx, (train_idx, test_idx) in enumerate(
        splitter.split(X, y_idx, groups),
        start=1,
    ):
        train_files = sorted({str(g) for g in groups[train_idx]})
        test_files = sorted({str(g) for g in groups[test_idx]})
        print(f"\nCV fold {fold_idx}/{n_splits}")
        print(f"Train ({len(train_files)} files): {[Path(f).name for f in train_files]}")
        print(f"Test  ({len(test_files)} files):  {[Path(f).name for f in test_files]}")
        print(f"Training GestureCNNv2 for {EPOCHS} epochs on {len(train_idx)} windows.")

        model, mean, std, _, best_epoch = train_eval_split(
            X[train_idx],
            y_idx[train_idx],
            X[test_idx],
            y_idx[test_idx],
            channel_count,
            num_classes,
            EPOCHS,
            device,
        )

        y_pred = _predict_labels(model, X[test_idx], device, mean, std)
        oof_pred[test_idx] = y_pred
        fold_eval = compute_eval_artifacts(y_idx[test_idx], y_pred, index_to_label)
        fold_metrics = serialize_eval_metrics(fold_eval)
        fold_results.append(
            {
                "fold_index": int(fold_idx),
                "train_windows": int(train_idx.size),
                "test_windows": int(test_idx.size),
                "best_epoch": int(best_epoch),
                "train_files": [Path(f).name for f in train_files],
                "test_files": [Path(f).name for f in test_files],
                "metrics": fold_metrics,
            }
        )
        print(
            f"Fold {fold_idx} metrics: accuracy {fold_metrics['test_accuracy']:.3f} | "
            f"balanced {fold_metrics['balanced_accuracy']:.3f} | "
            f"macro_f1 {fold_metrics['macro_f1']:.3f} | best_epoch {best_epoch}"
        )

    if np.any(oof_pred < 0):
        missing = int(np.sum(oof_pred < 0))
        raise RuntimeError(f"Cross-validation left {missing} window(s) without predictions.")

    eval_artifacts = compute_eval_artifacts(y_idx, oof_pred, index_to_label)
    accuracy_values = [fold["metrics"]["test_accuracy"] for fold in fold_results]
    balanced_values = [fold["metrics"]["balanced_accuracy"] for fold in fold_results]
    macro_f1_values = [fold["metrics"]["macro_f1"] for fold in fold_results]
    best_epoch_values = [fold["best_epoch"] for fold in fold_results]
    selected_final_fit_epochs = max(1, int(np.rint(np.median(best_epoch_values))))

    print("\nFold metric summary:")
    for metric_name, values in (
        ("accuracy", accuracy_values),
        ("balanced_accuracy", balanced_values),
        ("macro_f1", macro_f1_values),
        ("best_epoch", best_epoch_values),
    ):
        summary = scalar_summary(values)
        if summary is None:
            continue
        print(
            f"  {metric_name}: mean {summary['mean']:.3f} | std {summary['std']:.3f} | "
            f"min {summary['min']:.3f} | max {summary['max']:.3f}"
        )

    print_eval_summary(
        "Cross-validated out-of-fold metrics",
        "cross-validated accuracy",
        eval_artifacts,
        [index_to_label[i] for i in range(len(labels))],
    )

    return {
        "split_mode": "stratified-group-kfold",
        "n_splits": int(n_splits),
        "group_unit": "session_file",
        "eval_artifacts": eval_artifacts,
        "folds": fold_results,
        "fold_metric_summary": {
            "test_accuracy": scalar_summary(accuracy_values),
            "balanced_accuracy": scalar_summary(balanced_values),
            "macro_f1": scalar_summary(macro_f1_values),
            "best_epoch": scalar_summary(best_epoch_values),
        },
        "selected_final_fit_epochs": int(selected_final_fit_epochs),
    }


# -- Train/eval/save -----------------------------------------------------------

def _train_and_save(
    X, y_idx, groups, subjects, channel_count, num_classes, device,
    labels, label_to_index, index_to_label, model_out, subject_tag,
    layout_sources,
):
    unique_groups = np.unique(groups)
    print(f"{X.shape[0]} windows across {len(unique_groups)} session file(s).")
    cv_results = _run_grouped_cross_validation(
        X,
        y_idx,
        groups,
        channel_count,
        num_classes,
        device,
        labels,
        index_to_label,
    )
    selected_final_fit_epochs = cv_results["selected_final_fit_epochs"]
    print(f"\n{'=' * 55}")
    print(
        f"Final per-subject fit on all windows for {selected_final_fit_epochs} epoch(s) "
        f"(median best epoch from grouped CV)."
    )
    model, mean, std, final_fit_best_loss, final_fit_best_epoch = train_full_dataset(
        X,
        y_idx,
        channel_count,
        num_classes,
        selected_final_fit_epochs,
        device,
    )

    eval_artifacts = cv_results["eval_artifacts"]
    metrics_payload = serialize_eval_metrics(eval_artifacts)
    train_files = [Path(str(f)).name for f in sorted(unique_groups)]
    test_files = []

    bundle = {
        "model_state": model.state_dict(),
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
        "label_to_index": label_to_index,
        "index_to_label": index_to_label,
        "architecture": {
            **build_architecture_metadata(
                int(channel_count),
                dropout=DROPOUT,
            ),
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
            "split_mode": cv_results["split_mode"],
            "test_size": None,
            "calibration_used": bool(USE_CALIBRATION),
            "calibration_mvc_percentile": float(MVC_PERCENTILE),
            **build_model_metadata(),
            "label_confidence_filter": {
                "enabled": bool(USE_MIN_LABEL_CONFIDENCE),
                "min_label_confidence": float(MIN_LABEL_CONFIDENCE),
            },
            "channel_layout": {
                **strict_layout_bundle_metadata(ARM),
                "layout_inference_sources": dict(layout_sources),
            },
            "training": {
                "cv_max_epochs_per_fold": int(EPOCHS),
                "final_fit_epochs": int(selected_final_fit_epochs),
                "final_fit_epoch_selection": "median_best_epoch_from_grouped_cv",
                "batch_size": int(BATCH_SIZE),
                "lr": float(LR),
                "amp_range": list(AMP_RANGE),
                "use_augmentation": bool(USE_AUGMENTATION),
                "use_balanced_sampling": False,
                "label_smoothing": float(LABEL_SMOOTHING),
                "use_mixup": False,
            },
            "metrics": metrics_payload,
            "cross_validation": {
                "enabled": True,
                "type": "StratifiedGroupKFold",
                "group_unit": cv_results["group_unit"],
                "n_splits": int(cv_results["n_splits"]),
                "fold_metric_summary": cv_results["fold_metric_summary"],
                "selected_final_fit_epochs": int(selected_final_fit_epochs),
                "folds": cv_results["folds"],
            },
            "final_fit": {
                "fit_on_all_windows": True,
                "train_windows": int(X.shape[0]),
                "train_files": train_files,
                "selected_epochs_from_cv": int(selected_final_fit_epochs),
                "best_train_loss": float(final_fit_best_loss),
                "best_train_loss_epoch": int(final_fit_best_epoch),
            },
            "train_files": train_files,
            "test_files": test_files,
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

    X, y, groups, subjects, channel_count, layout_sources = load_dataset(target_subject=TARGET_SUBJECT)
    unique_subjects = sorted(np.unique(subjects))
    print(
        f"Loaded {X.shape[0]} windows, {channel_count} channels, "
        f"{len(np.unique(y))} classes, {len(unique_subjects)} subject(s): "
        f"{unique_subjects}"
    )
    expected_channels = strict_channel_count_for_arm(ARM)
    if channel_count != expected_channels:
        raise ValueError(
            f"Strict layout for {ARM} expects {expected_channels} channels, got {channel_count}."
        )
    print("Model architecture: GestureCNNv2")
    print(
        f"Channel layout mode: STRICT ({ARM}) | pairs={strict_layout_bundle_metadata(ARM)['pair_order']}"
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
        layout_sources,
    )

    print(f"\n{'=' * 55}")
    print("Single-subject training complete.")
    print(f"  python realtime_gesture_cnn.py --model {MODEL_OUT}")


if __name__ == "__main__":
    main()
