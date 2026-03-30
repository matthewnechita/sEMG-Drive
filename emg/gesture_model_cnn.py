from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class ResBlock1d(nn.Module):
    def __init__(self, channels, kernel_size=11, dropout=0.2):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.relu(self.block(x) + x))


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weights = self.pool(x).squeeze(-1)
        weights = self.fc(weights).unsqueeze(-1)
        return x * weights


class GestureCNNv2(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0.3):
        super().__init__()
        self.input_norm = nn.InstanceNorm1d(
            in_channels, affine=False, track_running_stats=False
        )
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=11, padding=5, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            ResBlock1d(32, kernel_size=11, dropout=dropout),
            ChannelAttention(32),
            nn.MaxPool1d(2, 2),
        )
        self.stage2_proj = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
        )
        self.stage2 = nn.Sequential(
            ResBlock1d(64, kernel_size=11, dropout=dropout),
            ChannelAttention(64),
            nn.MaxPool1d(2, 2),
        )
        self.stage3_proj = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
        )
        self.stage3 = nn.Sequential(
            ResBlock1d(128, kernel_size=11, dropout=dropout),
            ChannelAttention(128),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.head = nn.Linear(129, num_classes)

    def forward(self, x):
        combined = self.extract_embedding(x, l2_normalize=False)
        return self.head(combined)

    def extract_embedding(self, x, l2_normalize=False):
        # Concatenate a single global energy feature so the head can use coarse
        # activation magnitude alongside the learned temporal embedding.
        energy = x.pow(2).mean(dim=(1, 2)).unsqueeze(1)
        x = self.input_norm(x)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(self.stage2_proj(x))
        feats = self.stage3(self.stage3_proj(x))
        combined = torch.cat([feats, energy], dim=1)
        if l2_normalize:
            combined = F.normalize(combined, p=2, dim=1, eps=1e-8)
        return combined


@dataclass
class GestureModelBundle:
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
        # Runtime bundles keep mean/std fields for compatibility, but the active
        # CNN path applies InstanceNorm inside the model instead.
        return np.asarray(X, dtype=np.float32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            xb = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
            logits = self.model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return np.array([self.index_to_label[int(i)] for i in idx], dtype=object)


def _resolve_architecture(bundle: dict, in_channels: int, num_classes: int):
    arch = bundle.get("architecture") or {}
    metadata = bundle.get("metadata") or {}
    arch_type = str(arch.get("type") or "").strip()
    if not arch_type:
        # Older bundles infer the active architecture from metadata only.
        if metadata.get("use_instance_norm_input", False):
            arch_type = "GestureCNNv2"
    if arch_type != "GestureCNNv2":
        raise ValueError(f"Unsupported architecture type {arch_type!r} in gesture bundle.")
    dropout = float(arch.get("dropout", 0.3))
    return GestureCNNv2(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout=dropout,
    )


def _torch_load_bundle(path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_gesture_bundle(path: str | Path, device: str = "cpu") -> GestureModelBundle:
    path = Path(path)
    bundle = _torch_load_bundle(path, device)
    if not isinstance(bundle, dict):
        raise ValueError(f"{path} is not a valid gesture model bundle.")

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
    model = _resolve_architecture(bundle, int(mean.shape[0]), len(index_to_label))
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return GestureModelBundle(
        model=model,
        mean=mean,
        std=std,
        label_to_index=label_to_index,
        index_to_label=index_to_label,
        metadata=metadata,
    )
