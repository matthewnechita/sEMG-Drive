from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.left_padding = int((kernel_size - 1) * dilation)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.left_padding > 0:
            x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class TemporalResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 5,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = CausalConv1d(
            out_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.relu(self.bn1(out))
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out + residual)
        return self.dropout(out)


class MetricTCN(nn.Module):
    """Dilated temporal encoder with an embedding head and classifier head."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        channels: tuple[int, ...] = (64, 64, 128, 128),
        kernel_size: int = 5,
        embedding_dim: int = 128,
        dropout: float = 0.25,
    ):
        super().__init__()
        if not channels:
            raise ValueError("MetricTCN requires at least one hidden channel stage.")

        blocks = []
        prev_channels = int(in_channels)
        for idx, out_channels in enumerate(channels):
            blocks.append(
                TemporalResidualBlock(
                    prev_channels,
                    int(out_channels),
                    kernel_size=int(kernel_size),
                    dilation=2 ** idx,
                    dropout=float(dropout),
                )
            )
            prev_channels = int(out_channels)

        self.encoder = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embedding_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.embedding = nn.Linear(prev_channels, int(embedding_dim))
        self.head = nn.Linear(int(embedding_dim), int(num_classes))

    def _forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.pool(x).squeeze(-1)
        x = self.embedding_dropout(x)
        return self.embedding(x)

    def extract_embedding(self, x: torch.Tensor, l2_normalize: bool = False) -> torch.Tensor:
        emb = self._forward_embedding(x)
        if l2_normalize:
            emb = F.normalize(emb, p=2, dim=1, eps=1e-8)
        return emb

    def forward_with_embedding(
        self,
        x: torch.Tensor,
        *,
        l2_normalize_embedding: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.extract_embedding(x, l2_normalize=l2_normalize_embedding)
        logits = self.head(emb)
        return logits, emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_embedding(x, l2_normalize_embedding=False)
        return logits
