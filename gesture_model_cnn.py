from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


class GestureCNN(nn.Module):
    def __init__(self, channels, num_classes, dropout=0.2, kernel_size=7):
        super().__init__()
        blocks = []
        padding = kernel_size // 2
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            blocks.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, bias=False))
            blocks.append(nn.BatchNorm1d(out_ch))
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(nn.MaxPool1d(kernel_size=2, stride=2))
            if dropout > 0:
                blocks.append(nn.Dropout(dropout))
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x)


@dataclass
class CnnBundle:
    model: nn.Module
    mean: np.ndarray
    std: np.ndarray
    label_to_index: dict[str, int]
    index_to_label: dict[int, str]
    metadata: dict

    @property
    def channel_count(self) -> int:
        return int(self.mean.shape[0])

    def standardize(self, X: np.ndarray) -> np.ndarray:
        mean = self.mean.reshape(1, -1, 1)
        std = self.std.reshape(1, -1, 1)
        return (X - mean) / std

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            xb = torch.from_numpy(X.astype(np.float32)).to(device)
            logits = self.model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return np.array([self.index_to_label[int(i)] for i in idx], dtype=object)


def _resolve_architecture(bundle: dict, in_channels: int, num_classes: int):
    arch = bundle.get("architecture") or {}
    if arch.get("type") != "GestureCNN":
        arch = {}
    channels = arch.get("channels")
    if not channels:
        channels = [in_channels, 32, 64, 128]
    else:
        channels = list(channels)
        if channels[0] != in_channels:
            channels[0] = in_channels
    dropout = float(arch.get("dropout", 0.2))
    kernel_size = int(arch.get("kernel_size", 7))
    return channels, dropout, kernel_size


def _torch_load_bundle(path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_cnn_bundle(path: str | Path, device: str = "cpu") -> CnnBundle:
    path = Path(path)
    bundle = _torch_load_bundle(path, device)
    if not isinstance(bundle, dict):
        raise ValueError(f"{path} is not a valid CNN bundle.")

    state = bundle.get("model_state") or bundle.get("model")
    if state is None:
        raise ValueError(f"{path} missing model_state.")

    normalization = bundle.get("normalization") or {}
    mean = np.asarray(normalization.get("mean"), dtype=np.float32)
    std = np.asarray(normalization.get("std"), dtype=np.float32)
    if mean.ndim != 1 or std.ndim != 1:
        raise ValueError(f"{path} missing mean/std normalization arrays.")

    label_to_index = bundle.get("label_to_index") or {}
    index_to_label = bundle.get("index_to_label")
    if not index_to_label:
        index_to_label = {int(v): str(k) for k, v in label_to_index.items()}

    metadata = bundle.get("metadata") or {}
    channels, dropout, kernel_size = _resolve_architecture(
        bundle, int(mean.shape[0]), len(index_to_label)
    )
    model = GestureCNN(
        channels=channels,
        num_classes=len(index_to_label),
        dropout=dropout,
        kernel_size=kernel_size,
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return CnnBundle(
        model=model,
        mean=mean,
        std=std,
        label_to_index=label_to_index,
        index_to_label=index_to_label,
        metadata=metadata,
    )
