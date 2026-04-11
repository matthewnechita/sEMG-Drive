import datetime as dt
import copy
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from emg.cnn_training import (
    build_architecture_metadata,
    build_model_metadata,
    build_model,
    build_training_objective,
    compute_training_step,
    prepare_train_eval_inputs,
    standardize_windows,
)
from emg.eval_utils import (
    compute_eval_artifacts,
    print_eval_summary,
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


# -- Config --------------------------------------------------------------------
ARM = "left"  # Set to "right" or "left" before running.
DATA_ROOT  = strict_arm_root(STRICT_RESAMPLED_ROOT, ARM)
MODEL_OUT  = STRICT_MODELS_ROOT / "cross_subject" / ARM / "v6_4_gestures_2.pt"
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
EPOCHS  = 50
LR      = 1e-4
DROPOUT = 0.25
LABEL_SMOOTHING = 0.05

# Augmentation uses a wider amplitude range because cross-subject EMG varies
# more across participants than within one participant.
USE_AUGMENTATION = True
AMP_RANGE = (0.5, 2.0)   # wide inter-subject amplitude variance range
AUG_PROB  = 0.5

# Keep empty unless a participant should be excluded for known data-quality issues.
EXCLUDED_SUBJECTS: list[str] = []

# Optional gesture filtering (code-only; no CLI flags).
# Set INCLUDED_GESTURES to a subset to train only those labels.
# Example:
# INCLUDED_GESTURES = {"neutral", "left_turn", "right_turn"}
INCLUDED_GESTURES: set[str] | None = {"neutral", "left_turn", "right_turn", "horn"}  # Set to None to include all gestures.

# LOSO evaluation must be run before deploying the cross-subject model.
# This measures true cross-subject accuracy on held-out participants.
# Minimum recommended LOSO accuracy before deployment: 65%.
LOSO_EVAL = False
TRAIN_FINAL_MODEL = True

# -- Label utilities ---------------------------------------------------------
MVC_QUALITY_MIN_RATIO = DEFAULT_MVC_MIN_RATIO

# -- Data loading ------------------------------------------------------------
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
        # Every file in a cross-subject training run must resolve to the same strict slot order.
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


# -- Normalisation -----------------------------------------------------------
def _prepare_test_data(X, _mean, _std):
    return standardize_windows(X, _mean, _std)


# -- Augmentation (GPU-native) -----------------------------------------------
# All ops run on the GPU tensor after .to(device), eliminating CPU-GPU round-trips.
# Temporal stretch uses a single F.interpolate over the whole batch (one factor per
# batch) instead of per-sample loops, which is ~100x faster on GPU.

import torch.nn.functional as _F

def augment_emg_gpu(xb: torch.Tensor, p: float) -> torch.Tensor:
    """In-place-safe GPU augmentation. xb: (B, C, T) float32 on device."""
    B, C, T = xb.shape
    dev = xb.device
    xb = xb.clone()

    # 1. Amplitude scaling - per-sample random scale
    mask = torch.rand(B, device=dev) < p
    if mask.any():
        n = int(mask.sum())
        factors = torch.empty(n, 1, 1, device=dev).uniform_(AMP_RANGE[0], AMP_RANGE[1])
        xb[mask] = xb[mask] * factors

    # 2. Additive Gaussian noise - SNR-calibrated per sample
    mask = torch.rand(B, device=dev) < p
    if mask.any():
        n = int(mask.sum())
        snr_db     = torch.empty(n, device=dev).uniform_(10.0, 30.0)
        snr_linear = 10.0 ** (snr_db / 10.0)
        sig_power  = xb[mask].var(dim=(1, 2)).clamp(min=1e-8)        # (n,)
        noise_std  = (sig_power / snr_linear).sqrt().view(n, 1, 1)   # (n,1,1)
        xb[mask]   = xb[mask] + torch.randn(n, C, T, device=dev) * noise_std

    # 3. Temporal shift - vectorised gather (no Python loop)
    mask = torch.rand(B, device=dev) < p
    if mask.any():
        n      = int(mask.sum())
        shifts = torch.randint(-20, 21, (n,), device=dev)
        idx    = (torch.arange(T, device=dev).unsqueeze(0) - shifts.unsqueeze(1)) % T
        idx    = idx.unsqueeze(1).expand(-1, C, -1)                  # (n, C, T)
        xb[mask] = torch.gather(xb[mask], 2, idx)

    # 4. Channel dropout - vectorised scatter
    mask = torch.rand(B, device=dev) < p
    if mask.any():
        n       = int(mask.sum())
        drop_ch = torch.randint(0, C, (n,), device=dev)
        ch_mask = torch.zeros(n, C, device=dev, dtype=torch.bool)
        ch_mask.scatter_(1, drop_ch.unsqueeze(1), True)
        xb[mask] = xb[mask].masked_fill(ch_mask.unsqueeze(2), 0.0)

    # 5. Temporal stretch - single batched interpolation (one shared factor per batch)
    if torch.rand(1, device=dev).item() < p:
        factor  = torch.empty(1, device=dev).uniform_(0.85, 1.15).item()
        new_len = max(1, int(T * factor))
        stretched = _F.interpolate(xb, size=new_len, mode="linear", align_corners=False)
        if new_len >= T:
            xb = stretched[:, :, :T]
        else:
            xb = _F.pad(stretched, (0, T - new_len), mode="replicate")

    return xb


# -- Subject-balanced sampling -----------------------------------------------
def make_subject_sample_weights(subjects: np.ndarray) -> np.ndarray:
    unique, counts = np.unique(subjects, return_counts=True)
    weight_map = {s: 1.0 / c for s, c in zip(unique, counts)}
    return np.array([weight_map[s] for s in subjects], dtype=np.float32)


# -- Model -------------------------------------------------------------------
def _build_model(in_channels: int, num_classes: int, device) -> nn.Module:
    return build_model(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout=DROPOUT,
        device=device,
    )


# -- Training ----------------------------------------------------------------
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
    X_train_t, X_eval_t, mean, std = prepare_train_eval_inputs(X_train, X_eval)

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

    objective = build_training_objective(label_smoothing=LABEL_SMOOTHING)
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
                model,
                xb,
                yb,
                objective=objective,
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
                    model,
                    xb,
                    yb,
                    objective=objective,
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


# -- LOSO evaluation ---------------------------------------------------------
def loso_evaluate(X, y_idx, subjects, channel_count, num_classes, device, index_to_label):
    """Leave-one-subject-out: true measure of cross-subject generalisation.

    Each fold trains on N-1 subjects and tests on the held-out subject.
    This is the correct evaluation metric for cross-subject models.
    In-distribution test accuracy (final model) is NOT comparable to this.
    """
    unique_subjects = sorted(np.unique(subjects))
    if len(unique_subjects) < 2:
        print("LOSO requires at least 2 subjects - skipping.")
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
        eval_artifacts = compute_eval_artifacts(y_idx[test_mask], y_pred, index_to_label)
        acc = float(eval_artifacts["test_accuracy"])
        loso_accs.append(acc)
        fold_results.append(
            {
                "held_out_subject": str(held_out),
                "test_windows": int(test_mask.sum()),
                "metrics": serialize_eval_metrics(eval_artifacts),
            }
        )
        print(f"    Accuracy: {acc:.3f}")
        print(eval_artifacts["classification_report_text"])

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
        "folds": fold_results,
        "summary": {
            "mean_accuracy": float(loso_accs.mean()),
            "std_accuracy": float(loso_accs.std()),
            "min_accuracy": float(loso_accs.min()),
            "max_accuracy": float(loso_accs.max()),
        },
    }


# -- Save bundle -------------------------------------------------------------
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
        **build_model_metadata(),
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
            int(channel_count),
            dropout=DROPOUT,
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

    eval_artifacts = compute_eval_artifacts(y_test, y_pred, index_to_label)
    print("(Note: this is NOT the cross-subject accuracy. See LOSO results above.)")
    print_eval_summary(
        "In-distribution test summary",
        "accuracy",
        eval_artifacts,
        [index_to_label[i] for i in range(len(labels))],
    )

    test_subjects = subjects[test_idx]
    unique_test   = np.unique(test_subjects)
    if len(unique_test) > 1:
        print("Per-subject breakdown:")
        for subj in sorted(unique_test):
            mask = test_subjects == subj
            print(f"  {subj}: {accuracy_score(y_test[mask], y_pred[mask]):.3f}  ({mask.sum()} windows)")

    extra_metadata = {
        "channel_layout": {
            **strict_layout_bundle_metadata(ARM),
            "layout_inference_sources": dict(layout_sources),
        }
    }
    if evaluation_metadata:
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


# -- Main --------------------------------------------------------------------
def main():
    if ARM not in ("right", "left"):
        raise ValueError(f"ARM must be 'right' or 'left', got {ARM!r}")
    confirm = input(f"Training {ARM} arm cross-subject model - continue? [y/N] ").strip().lower()
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
    expected_channels = strict_channel_count_for_arm(ARM)
    if channel_count != expected_channels:
        raise ValueError(
            f"Strict layout for {ARM} expects {expected_channels} channels, got {channel_count}."
        )
    print("Model architecture: GestureCNNv2")
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

    # LOSO runs before the pooled final fit so deployment decisions can use zero-shot accuracy.
    evaluation_metadata = {}
    if LOSO_EVAL:
        evaluation_metadata["zero_shot_loso"] = loso_evaluate(
            X, y_idx, subjects, channel_count, num_classes, device, index_to_label
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
