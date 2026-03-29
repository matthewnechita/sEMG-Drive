from __future__ import annotations

import numpy as np
import torch
from torch import nn

from emg.gesture_model_cnn import GestureCNNv2


def compute_normalization_stats(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(X_train.mean(axis=(0, 2)), dtype=np.float32)
    std = np.asarray(X_train.std(axis=(0, 2)), dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32, copy=False)
    return mean, std


def standardize_windows(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    del mean, std
    return np.asarray(X, dtype=np.float32)


def prepare_train_eval_inputs(
    X_train: np.ndarray,
    X_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean, std = compute_normalization_stats(X_train)
    X_train_t = standardize_windows(X_train, mean, std)
    X_eval_t = standardize_windows(X_eval, mean, std)
    return X_train_t, X_eval_t, mean, std


def prepare_train_inputs(
    X_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, std = compute_normalization_stats(X_train)
    X_train_t = standardize_windows(X_train, mean, std)
    return X_train_t, mean, std


def build_model(
    in_channels: int,
    num_classes: int,
    *,
    dropout: float,
    device,
) -> nn.Module:
    model = GestureCNNv2(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        dropout=float(dropout),
    )
    return model.to(device)


def build_training_objective(*, label_smoothing: float) -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))


def compute_training_step(
    model: nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    *,
    objective: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = model(xb)
    total_loss = objective(logits, yb)
    return logits, total_loss


def build_architecture_metadata(
    in_channels: int,
    *,
    dropout: float,
) -> dict:
    return {
        "type": "GestureCNNv2",
        "in_channels": int(in_channels),
        "dropout": float(dropout),
    }


def build_model_metadata() -> dict:
    return {
        "use_instance_norm_input": True,
    }
