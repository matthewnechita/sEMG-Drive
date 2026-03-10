from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _l2_normalize(vec: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    denom = np.linalg.norm(vec, axis=axis, keepdims=True)
    denom = np.maximum(denom, eps)
    return vec / denom


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.size == 0:
        return logits
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    denom = np.sum(exps)
    if denom <= 0:
        return np.full_like(exps, 1.0 / max(exps.size, 1))
    return exps / denom


@dataclass
class PrototypeClassifier:
    """Nearest-centroid classifier over embedding vectors."""

    class_indices: np.ndarray
    centroids: np.ndarray
    counts: np.ndarray
    temperature: float = 0.20
    l2_normalize: bool = True

    @classmethod
    def fit(
        cls,
        embeddings: np.ndarray,
        labels: np.ndarray,
        temperature: float = 0.20,
        l2_normalize: bool = True,
    ) -> "PrototypeClassifier":
        emb = np.asarray(embeddings, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64).reshape(-1)
        if emb.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got shape {emb.shape}")
        if emb.shape[0] != y.shape[0]:
            raise ValueError(
                f"embeddings/labels length mismatch: {emb.shape[0]} vs {y.shape[0]}"
            )
        if emb.shape[0] == 0:
            raise ValueError("Cannot fit PrototypeClassifier with zero samples.")

        class_indices = np.array(sorted(set(int(v) for v in y.tolist())), dtype=np.int64)
        centroids = []
        counts = []
        for idx in class_indices:
            mask = y == idx
            if not np.any(mask):
                continue
            c = emb[mask].mean(axis=0)
            centroids.append(c.astype(np.float32, copy=False))
            counts.append(int(mask.sum()))

        if not centroids:
            raise ValueError("No class centroids could be computed.")

        centroids_arr = np.vstack(centroids).astype(np.float32, copy=False)
        counts_arr = np.asarray(counts, dtype=np.int64)
        if l2_normalize:
            centroids_arr = _l2_normalize(centroids_arr, axis=1).astype(np.float32, copy=False)

        return cls(
            class_indices=class_indices,
            centroids=centroids_arr,
            counts=counts_arr,
            temperature=float(max(temperature, 1e-6)),
            l2_normalize=bool(l2_normalize),
        )

    def _cosine_scores(self, embedding: np.ndarray) -> np.ndarray:
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if emb.shape[0] != self.centroids.shape[1]:
            raise ValueError(
                f"Embedding dim mismatch: got {emb.shape[0]}, expected {self.centroids.shape[1]}"
            )
        if self.l2_normalize:
            emb = _l2_normalize(emb, axis=0).reshape(-1)
            return self.centroids @ emb

        emb_norm = float(np.linalg.norm(emb))
        cent_norm = np.linalg.norm(self.centroids, axis=1)
        denom = np.maximum(emb_norm * cent_norm, 1e-8)
        return (self.centroids @ emb) / denom

    def predict_proba(self, embedding: np.ndarray, num_classes: int) -> np.ndarray:
        if num_classes <= 0:
            raise ValueError("num_classes must be > 0")
        scores = self._cosine_scores(embedding)
        class_probs = _softmax(scores / self.temperature).astype(np.float32, copy=False)
        probs = np.zeros(int(num_classes), dtype=np.float32)
        probs[self.class_indices] = class_probs
        return probs

    def update(self, class_idx: int, embedding: np.ndarray, alpha: float = 0.05) -> bool:
        """EMA-update one centroid with a new embedding.

        Returns True if an update occurred.
        """
        idx = int(class_idx)
        loc = np.where(self.class_indices == idx)[0]
        if loc.size == 0:
            return False

        row = int(loc[0])
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if emb.shape[0] != self.centroids.shape[1]:
            return False
        if self.l2_normalize:
            emb = _l2_normalize(emb, axis=0).reshape(-1).astype(np.float32, copy=False)

        a = float(np.clip(alpha, 0.0, 1.0))
        self.centroids[row] = (1.0 - a) * self.centroids[row] + (a * emb)
        if self.l2_normalize:
            self.centroids[row] = _l2_normalize(self.centroids[row], axis=0).reshape(-1)
        self.counts[row] = int(self.counts[row] + 1)
        return True
