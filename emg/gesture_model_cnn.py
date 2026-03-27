from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from emg.gesture_model_metric_tcn import MetricTCN


# ── Original architecture (kept for backward compatibility) ──────────────────

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

    def extract_embedding(self, x, l2_normalize=False):
        x = self.features(x)
        x = self.head[0](x)  # AdaptiveAvgPool1d(1)
        x = self.head[1](x)  # Flatten
        if l2_normalize:
            x = F.normalize(x, p=2, dim=1, eps=1e-8)
        return x


# ── V2 building blocks ────────────────────────────────────────────────────────

class ResBlock1d(nn.Module):
    """Residual block with two Conv1d layers, BatchNorm, and optional Dropout."""

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
    """Squeeze-and-Excitation channel attention."""

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
        w = self.pool(x).squeeze(-1)
        w = self.fc(w).unsqueeze(-1)
        return x * w


class GestureCNNv2(nn.Module):
    """Residual CNN with per-window InstanceNorm input normalisation and an
    energy bypass scalar.

    InstanceNorm at the input removes inter-subject amplitude differences on a
    per-window basis.  A raw energy scalar computed *before* normalisation is
    concatenated into the classification head so the model can still
    distinguish near-zero (neutral) windows from active gesture windows even
    after normalisation has equalised their apparent scale.
    """

    def __init__(self, in_channels, num_classes, dropout=0.3):
        super().__init__()
        self.input_norm = nn.InstanceNorm1d(
            in_channels, affine=False, track_running_stats=False
        )
        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=11, padding=5, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        # Stage 1 – 32 channels
        self.stage1 = nn.Sequential(
            ResBlock1d(32, kernel_size=11, dropout=dropout),
            ChannelAttention(32),
            nn.MaxPool1d(2, 2),
        )
        # Stage 2 – 64 channels (projection conv)
        self.stage2_proj = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
        )
        self.stage2 = nn.Sequential(
            ResBlock1d(64, kernel_size=11, dropout=dropout),
            ChannelAttention(64),
            nn.MaxPool1d(2, 2),
        )
        # Stage 3 – 128 channels (projection conv)
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
        # Head: 128 features + 1 energy scalar
        self.head = nn.Linear(129, num_classes)

    def forward(self, x):
        combined = self.extract_embedding(x, l2_normalize=False)
        return self.head(combined)

    def extract_embedding(self, x, l2_normalize=False):
        # Capture raw energy BEFORE InstanceNorm (preserves neutral detection)
        energy = x.pow(2).mean(dim=(1, 2)).unsqueeze(1)  # (B, 1)
        x = self.input_norm(x)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(self.stage2_proj(x))
        feats = self.stage3(self.stage3_proj(x))  # (B, 128)
        combined = torch.cat([feats, energy], dim=1)  # (B, 129)
        if l2_normalize:
            combined = F.normalize(combined, p=2, dim=1, eps=1e-8)
        return combined


# ── Architecture registry ────────────────────────────────────────────────────

ARCHITECTURE_REGISTRY = {
    "GestureCNN": GestureCNN,
    "GestureCNNv2": GestureCNNv2,
    "MetricTCN": MetricTCN,
}


# ── Bundle ───────────────────────────────────────────────────────────────────

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
        # GestureCNNv2 handles normalisation internally via InstanceNorm1d;
        # skip external z-score so the energy bypass scalar is not distorted.
        if self.metadata.get("use_instance_norm_input", False):
            return X
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

    def embed(self, X: np.ndarray, l2_normalize: bool = False) -> np.ndarray:
        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            xb = torch.from_numpy(X.astype(np.float32)).to(device)
            extract_embedding = getattr(self.model, "extract_embedding", None)
            if callable(extract_embedding):
                emb_out = extract_embedding(xb, l2_normalize=l2_normalize)
                if not isinstance(emb_out, torch.Tensor):
                    raise TypeError("Model extract_embedding returned a non-tensor object.")
                emb = emb_out
            else:
                logits = self.model(xb)
                if not isinstance(logits, torch.Tensor):
                    raise TypeError("Model forward pass returned a non-tensor object.")
                emb = logits
                if l2_normalize:
                    emb = F.normalize(emb, p=2, dim=1, eps=1e-8)
            return emb.cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return np.array([self.index_to_label[int(i)] for i in idx], dtype=object)


# ── Model loading ────────────────────────────────────────────────────────────

def _resolve_architecture(bundle: dict, in_channels: int, num_classes: int):
    arch = bundle.get("architecture") or {}
    metadata = bundle.get("metadata") or {}
    arch_type = arch.get("type")
    if not arch_type:
        family = str(metadata.get("model_family", "")).strip().lower()
        if family == "cnn_v2":
            arch_type = "GestureCNNv2"
        elif family == "metric_tcn":
            arch_type = "MetricTCN"
        else:
            arch_type = "GestureCNN"

    if arch_type == "GestureCNNv2":
        dropout = float(arch.get("dropout", 0.3))
        return GestureCNNv2(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=dropout,
        )

    if arch_type == "MetricTCN":
        dropout = float(arch.get("dropout", 0.25))
        channels = tuple(int(v) for v in arch.get("channels", [64, 64, 128, 128]))
        kernel_size = int(arch.get("kernel_size", 5))
        embedding_dim = int(arch.get("embedding_dim", 128))
        return MetricTCN(
            in_channels=in_channels,
            num_classes=num_classes,
            channels=channels,
            kernel_size=kernel_size,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

    # Default: GestureCNN (or any unrecognised arch falls back to it)
    channels = arch.get("channels")
    if not channels:
        channels = [in_channels, 32, 64, 128]
    else:
        channels = list(channels)
        if channels[0] != in_channels:
            channels[0] = in_channels
    dropout = float(arch.get("dropout", 0.2))
    kernel_size = int(arch.get("kernel_size", 7))
    return GestureCNN(
        channels=channels,
        num_classes=num_classes,
        dropout=dropout,
        kernel_size=kernel_size,
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


def load_cnn_bundle(path: str | Path, device: str = "cpu") -> GestureModelBundle:
    return load_gesture_bundle(path, device=device)


# ── Quick fine-tune (standalone function, not a method) ──────────────────────

def quick_finetune(
    bundle: GestureModelBundle,
    calib_windows: np.ndarray,
    calib_labels: np.ndarray,
    device: str = "cpu",
    lr: float = 1e-4,
    epochs: int = 20,
) -> GestureModelBundle:
    """Fine-tune only the classification head on a small calibration set.

    Freezes all parameters except the final linear layer so that ~70 s of
    calibration data from a new subject is sufficient to adapt the model
    without overfitting the feature extractor.

    Parameters
    ----------
    bundle:
        Existing loaded CnnBundle (will be mutated in-place and returned).
    calib_windows:
        Shape (N, C, T) float32 array of windowed EMG data.
    calib_labels:
        Shape (N,) int64 array of class indices.
    device:
        Torch device string.
    lr:
        Learning rate for the head.
    epochs:
        Number of training epochs over calibration data.

    Returns
    -------
    The same CnnBundle with the head fine-tuned.
    """
    from torch.utils.data import DataLoader, TensorDataset

    # Apply the same normalisation the bundle uses at inference time
    calib_windows = bundle.standardize(calib_windows)

    model = bundle.model.to(device)

    # Freeze everything except the head
    for param in model.parameters():
        param.requires_grad = False
    head_module = getattr(model, "head", None)
    if not isinstance(head_module, nn.Module):
        raise AttributeError("quick_finetune expects bundle.model.head to be an nn.Module.")
    for param in head_module.parameters():
        param.requires_grad = True

    xb = torch.from_numpy(calib_windows.astype(np.float32))
    yb = torch.from_numpy(calib_labels.astype(np.int64))
    loader = DataLoader(TensorDataset(xb, yb), batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()

    # Restore eval mode and unfreeze for normal inference
    for param in model.parameters():
        param.requires_grad = True
    model.eval()

    return bundle


CnnBundle = GestureModelBundle
