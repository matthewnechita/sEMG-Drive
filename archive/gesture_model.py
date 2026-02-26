from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pickle


@dataclass
class ModelBundle:
    model: object
    metadata: dict

    @property
    def feature_order(self) -> list[str]:
        order = self.metadata.get("feature_order") or self.metadata.get("feature_list")
        if order is None:
            return []
        return [str(x) for x in order]

    @property
    def channel_count(self) -> int | None:
        count = self.metadata.get("channel_count")
        if count is None:
            return None
        return int(count)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("Model does not support predict_proba().")
        return self.model.predict_proba(X)


def load_model_bundle(path: str | Path) -> ModelBundle:
    path = Path(path)
    with path.open("rb") as f:
        bundle = pickle.load(f)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"{path} is not a valid model bundle.")
    metadata = bundle.get("metadata") or {}
    return ModelBundle(model=bundle["model"], metadata=metadata)


def stack_feature_dict(
    feature_dict: dict,
    feature_order: Iterable[str],
) -> np.ndarray:
    feature_order = list(feature_order)
    arrays = []
    for key in feature_order:
        if key not in feature_dict:
            raise KeyError(f"Missing feature '{key}' in feature_dict")
        arr = np.asarray(feature_dict[key])
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        arrays.append(arr)
    stacked = np.stack(arrays, axis=-1)
    if stacked.ndim != 3:
        raise ValueError(f"Expected stacked features to be 3D, got shape {stacked.shape}")
    return stacked


def flatten_feature_dict(
    feature_dict: dict,
    feature_order: Iterable[str],
    channel_count: int | None = None,
) -> np.ndarray:
    stacked = stack_feature_dict(feature_dict, feature_order)
    if channel_count is not None and stacked.shape[1] != channel_count:
        raise ValueError(
            f"Channel count mismatch (expected {channel_count}, got {stacked.shape[1]})"
        )
    return stacked.reshape(stacked.shape[0], -1)
