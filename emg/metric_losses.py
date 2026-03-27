from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """Single-view supervised contrastive loss over a batch of embeddings."""

    def __init__(self, temperature: float = 0.10, eps: float = 1e-8):
        super().__init__()
        self.temperature = float(max(temperature, 1e-6))
        self.eps = float(max(eps, 1e-12))

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got shape {tuple(embeddings.shape)}")

        labels = labels.reshape(-1)
        if embeddings.shape[0] != labels.shape[0]:
            raise ValueError(
                f"embeddings/labels length mismatch: {embeddings.shape[0]} vs {labels.shape[0]}"
            )
        if embeddings.shape[0] < 2:
            return embeddings.new_zeros(())

        emb = F.normalize(embeddings, p=2, dim=1, eps=self.eps)
        logits = torch.matmul(emb, emb.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        labels = labels.contiguous().view(-1, 1)
        positive_mask = torch.eq(labels, labels.T)
        eye_mask = torch.eye(positive_mask.shape[0], device=positive_mask.device, dtype=torch.bool)
        positive_mask = positive_mask & (~eye_mask)
        logits_mask = ~eye_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(self.eps))

        positive_counts = positive_mask.sum(dim=1)
        valid = positive_counts > 0
        if not torch.any(valid):
            return embeddings.new_zeros(())

        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_counts.clamp_min(1)
        return -mean_log_prob_pos[valid].mean()
