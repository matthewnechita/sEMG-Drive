from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
import torch
from torch import nn

from emg.gesture_model_cnn import GestureCNNv2
from emg.gesture_model_metric_tcn import MetricTCN
from emg.metric_losses import SupervisedContrastiveLoss


SUPPORTED_MODEL_FAMILIES = ("cnn_v2", "metric_tcn")
DEFAULT_METRIC_TCN_CHANNELS = (64, 64, 128, 128)
DEFAULT_METRIC_TCN_KERNEL_SIZE = 5
DEFAULT_METRIC_TCN_EMBEDDING_DIM = 128
DEFAULT_SUPCON_WEIGHT = 0.20
DEFAULT_SUPCON_TEMPERATURE = 0.10


@dataclass(frozen=True)
class ModelFamilyConfig:
    model_family: str
    metric_tcn_channels: tuple[int, ...] = DEFAULT_METRIC_TCN_CHANNELS
    metric_tcn_kernel_size: int = DEFAULT_METRIC_TCN_KERNEL_SIZE
    metric_tcn_embedding_dim: int = DEFAULT_METRIC_TCN_EMBEDDING_DIM
    supcon_weight: float = DEFAULT_SUPCON_WEIGHT
    supcon_temperature: float = DEFAULT_SUPCON_TEMPERATURE


@dataclass
class TrainingObjectives:
    cross_entropy: nn.Module
    supervised_contrastive: nn.Module | None


class SupportsForwardWithEmbedding(Protocol):
    def forward_with_embedding(
        self,
        x: torch.Tensor,
        *,
        l2_normalize_embedding: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


def validate_model_family(model_family: str) -> str:
    value = str(model_family).strip().lower()
    if value not in SUPPORTED_MODEL_FAMILIES:
        raise ValueError(
            f"MODEL_FAMILY must be one of {SUPPORTED_MODEL_FAMILIES}, got {model_family!r}"
        )
    return value


def uses_instance_norm_input(model_family: str) -> bool:
    return validate_model_family(model_family) == "cnn_v2"


def supports_prototype_calibration(model_family: str) -> bool:
    return validate_model_family(model_family) == "metric_tcn"


def build_model(
    model_family: str,
    in_channels: int,
    num_classes: int,
    *,
    dropout: float,
    device,
    family_cfg: ModelFamilyConfig,
) -> nn.Module:
    model_family = validate_model_family(model_family)
    if model_family == "metric_tcn":
        model = MetricTCN(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            channels=tuple(int(v) for v in family_cfg.metric_tcn_channels),
            kernel_size=int(family_cfg.metric_tcn_kernel_size),
            embedding_dim=int(family_cfg.metric_tcn_embedding_dim),
            dropout=float(dropout),
        )
    else:
        model = GestureCNNv2(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            dropout=float(dropout),
        )
    return model.to(device)


def build_training_objectives(
    model_family: str,
    *,
    label_smoothing: float,
    family_cfg: ModelFamilyConfig,
) -> TrainingObjectives:
    model_family = validate_model_family(model_family)
    ce = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))
    supcon = None
    if model_family == "metric_tcn":
        supcon = SupervisedContrastiveLoss(temperature=float(family_cfg.supcon_temperature))
    return TrainingObjectives(cross_entropy=ce, supervised_contrastive=supcon)


def compute_normalization_stats(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(X_train.mean(axis=(0, 2)), dtype=np.float32)
    std = np.asarray(X_train.std(axis=(0, 2)), dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32, copy=False)
    return mean, std


def standardize_windows(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    model_family: str,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if uses_instance_norm_input(model_family):
        return X.astype(np.float32, copy=False)
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, -1, 1)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, -1, 1)
    return ((X - mean_arr) / std_arr).astype(np.float32, copy=False)


def prepare_train_eval_inputs(
    model_family: str,
    X_train: np.ndarray,
    X_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean, std = compute_normalization_stats(X_train)
    X_train_t = standardize_windows(X_train, mean, std, model_family=model_family)
    X_eval_t = standardize_windows(X_eval, mean, std, model_family=model_family)
    return X_train_t, X_eval_t, mean, std


def prepare_train_inputs(
    model_family: str,
    X_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, std = compute_normalization_stats(X_train)
    X_train_t = standardize_windows(X_train, mean, std, model_family=model_family)
    return X_train_t, mean, std


def compute_training_step(
    model_family: str,
    model: nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    *,
    objectives: TrainingObjectives,
    family_cfg: ModelFamilyConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    model_family = validate_model_family(model_family)
    if model_family == "metric_tcn":
        forward_with_embedding = getattr(model, "forward_with_embedding", None)
        if not callable(forward_with_embedding):
            raise AttributeError("metric_tcn model must define forward_with_embedding(...).")
        metric_model = cast(SupportsForwardWithEmbedding, model)
        logits, embeddings = metric_model.forward_with_embedding(
            xb,
            l2_normalize_embedding=False,
        )
        ce_loss = objectives.cross_entropy(logits, yb)
        if objectives.supervised_contrastive is None:
            raise RuntimeError("metric_tcn requires a supervised contrastive objective.")
        supcon_loss = objectives.supervised_contrastive(embeddings, yb)
        total_loss = ce_loss + (float(family_cfg.supcon_weight) * supcon_loss)
        return logits, total_loss

    logits = model(xb)
    total_loss = objectives.cross_entropy(logits, yb)
    return logits, total_loss


def build_architecture_metadata(
    model_family: str,
    in_channels: int,
    *,
    dropout: float,
    family_cfg: ModelFamilyConfig,
) -> dict:
    model_family = validate_model_family(model_family)
    if model_family == "metric_tcn":
        return {
            "type": "MetricTCN",
            "in_channels": int(in_channels),
            "channels": [int(v) for v in family_cfg.metric_tcn_channels],
            "kernel_size": int(family_cfg.metric_tcn_kernel_size),
            "embedding_dim": int(family_cfg.metric_tcn_embedding_dim),
            "dropout": float(dropout),
        }
    return {
        "type": "GestureCNNv2",
        "in_channels": int(in_channels),
        "dropout": float(dropout),
    }


def build_family_metadata(
    model_family: str,
    *,
    family_cfg: ModelFamilyConfig,
) -> dict:
    model_family = validate_model_family(model_family)
    metadata = {
        "model_family": model_family,
        "supports_prototype_calibration": bool(supports_prototype_calibration(model_family)),
        "decoder_preference": "prototype" if supports_prototype_calibration(model_family) else "softmax",
        "use_instance_norm_input": bool(uses_instance_norm_input(model_family)),
    }
    if model_family == "metric_tcn":
        metadata["metric_learning"] = {
            "loss": "cross_entropy_plus_supervised_contrastive",
            "embedding_dim": int(family_cfg.metric_tcn_embedding_dim),
            "supcon_weight": float(family_cfg.supcon_weight),
            "supcon_temperature": float(family_cfg.supcon_temperature),
        }
    return metadata
