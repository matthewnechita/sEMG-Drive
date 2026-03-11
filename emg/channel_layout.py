from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


CANONICAL_BLOCK_ORDER = ("galileo", "single", "maize", "unknown")


@dataclass
class ChannelLayout:
    ordered_indices: np.ndarray
    permutation_groups: list[list[int]]
    kind_counts: dict[str, int]
    source: str


def _coerce_metadata_dict(metadata: Any) -> dict[str, Any]:
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, np.ndarray) and metadata.shape == () and metadata.dtype == object:
        try:
            value = metadata.item()
        except Exception:
            return {}
        return value if isinstance(value, dict) else {}
    return {}


def _classify_kind_from_label(label: Any) -> str:
    text = str(label).strip().lower()
    if "galileo" in text:
        return "galileo"
    if "maize" in text:
        return "maize"
    if "avanti" in text or "mini" in text:
        return "single"
    return "unknown"


def _classify_kind_from_fs(value: Any) -> str:
    try:
        fs_hz = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if not np.isfinite(fs_hz):
        return "unknown"
    if 2190.0 <= fs_hz <= 2260.0:
        return "galileo"
    if 1970.0 <= fs_hz <= 2030.0:
        return "maize"
    if 2120.0 <= fs_hz <= 2175.0:
        return "single"
    return "unknown"


def _estimate_fs_per_channel(timestamps: np.ndarray) -> list[float]:
    ts = np.asarray(timestamps, dtype=float)
    if ts.ndim != 2:
        return []
    fs_values = []
    for ch in range(ts.shape[1]):
        diffs = np.diff(ts[:, ch])
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size == 0:
            fs_values.append(float("nan"))
        else:
            fs_values.append(float(1.0 / np.median(diffs)))
    return fs_values


def _kind_sequence_from_inputs(
    metadata: Any,
    channel_count: int | None = None,
    timestamps: np.ndarray | None = None,
    channel_labels: list[str] | tuple[str, ...] | None = None,
) -> tuple[list[str], str] | tuple[None, None]:
    if channel_labels is not None:
        labels = [str(v) for v in channel_labels]
        if channel_count is None or len(labels) == int(channel_count):
            return [_classify_kind_from_label(v) for v in labels], "channel_labels"

    md = _coerce_metadata_dict(metadata)
    for key in ("emg_channel_labels", "emg_channel_names", "channel_labels"):
        labels = md.get(key)
        if labels is None:
            continue
        labels = [str(v) for v in labels]
        if channel_count is not None and len(labels) != int(channel_count):
            continue
        return [_classify_kind_from_label(v) for v in labels], key

    resampling = md.get("resampling")
    if isinstance(resampling, dict):
        fs_values = resampling.get("source_fs_hz_per_channel")
        if fs_values is not None:
            fs_values = list(fs_values)
            if channel_count is None or len(fs_values) == int(channel_count):
                return [_classify_kind_from_fs(v) for v in fs_values], "source_fs_hz_per_channel"

    if timestamps is not None:
        fs_values = _estimate_fs_per_channel(np.asarray(timestamps, dtype=float))
        if fs_values and (channel_count is None or len(fs_values) == int(channel_count)):
            return [_classify_kind_from_fs(v) for v in fs_values], "timestamps"

    return None, None


def infer_channel_layout(
    metadata: Any = None,
    channel_count: int | None = None,
    timestamps: np.ndarray | None = None,
    channel_labels: list[str] | tuple[str, ...] | None = None,
) -> ChannelLayout | None:
    kinds, source = _kind_sequence_from_inputs(
        metadata=metadata,
        channel_count=channel_count,
        timestamps=timestamps,
        channel_labels=channel_labels,
    )
    if kinds is None or not kinds:
        return None

    ordered_indices = []
    permutation_groups = []
    kind_counts = {kind: 0 for kind in CANONICAL_BLOCK_ORDER}

    for kind in CANONICAL_BLOCK_ORDER:
        idxs = [idx for idx, value in enumerate(kinds) if value == kind]
        kind_counts[kind] = len(idxs)
        if not idxs:
            continue
        start = len(ordered_indices)
        ordered_indices.extend(idxs)
        if kind == "single" and len(idxs) > 1:
            permutation_groups.append(list(range(start, start + len(idxs))))

    if len(ordered_indices) != len(kinds):
        return None

    default_order = list(range(len(kinds)))
    if ordered_indices == default_order and not permutation_groups:
        return ChannelLayout(
            ordered_indices=np.asarray(default_order, dtype=int),
            permutation_groups=[],
            kind_counts={k: v for k, v in kind_counts.items() if v},
            source=source,
        )

    return ChannelLayout(
        ordered_indices=np.asarray(ordered_indices, dtype=int),
        permutation_groups=permutation_groups,
        kind_counts={k: v for k, v in kind_counts.items() if v},
        source=source,
    )


def reorder_by_layout(matrix: np.ndarray, layout: ChannelLayout | None) -> np.ndarray:
    if layout is None:
        return matrix
    arr = np.asarray(matrix)
    if arr.ndim != 2:
        return arr
    if arr.shape[1] != layout.ordered_indices.size:
        return arr
    return arr[:, layout.ordered_indices]
