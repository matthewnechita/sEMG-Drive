from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from libemg.utils import get_windows

from emg.strict_layout import StrictLayoutResolution, resolve_strict_indices_from_metadata


DEFAULT_MVC_MIN_RATIO = 1.5


@dataclass(frozen=True)
class WindowedStrictFile:
    windows: np.ndarray
    labels: np.ndarray
    strict_layout: StrictLayoutResolution


def clean_label(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except Exception:
            return None
    if isinstance(value, np.str_):
        value = str(value)
    if isinstance(value, str):
        value = value.strip()
    if value == "":
        return None
    return str(value)


def majority_label_with_confidence(segment: np.ndarray) -> tuple[str | None, float]:
    if segment.size == 0:
        return None, 0.0
    flat = np.asarray(segment, dtype=object).reshape(-1)
    cleaned = [label for label in (clean_label(item) for item in flat) if label is not None]
    if not cleaned:
        return None, 0.0
    values, counts = np.unique(np.asarray(cleaned, dtype=object), return_counts=True)
    if counts.size == 0:
        return None, 0.0
    idx = int(np.argmax(counts))
    total = int(np.sum(counts))
    confidence = float(counts[idx] / total) if total > 0 else 0.0
    return str(values[idx]), confidence


def compute_calibration(
    neutral_emg,
    mvc_emg,
    percentile: float,
    mvc_min_ratio: float = DEFAULT_MVC_MIN_RATIO,
    *,
    verbose: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    neutral = np.asarray(neutral_emg, dtype=float)
    mvc = np.asarray(mvc_emg, dtype=float)
    if neutral.size == 0 or mvc.size == 0:
        return None, None

    neutral_rms = np.sqrt(np.mean(neutral ** 2, axis=0))
    mvc_rms = np.sqrt(np.mean(mvc ** 2, axis=0))
    ratio = np.ones_like(mvc_rms, dtype=float)
    np.divide(mvc_rms, neutral_rms, out=ratio, where=neutral_rms >= 1e-9)
    median_ratio = float(np.median(ratio))
    threshold = float(mvc_min_ratio)

    if median_ratio < threshold:
        if verbose:
            print(
                f"  [calib] SKIP: median MVC/neutral ratio={median_ratio:.2f}x "
                f"(< {threshold:.2f}x threshold). "
                "MVC calibration failed - normalization not applied for this session."
            )
        return None, None

    neutral_mean = np.mean(neutral, axis=0)
    mvc_scale = np.percentile(mvc, float(percentile), axis=0)
    mvc_scale = np.where(mvc_scale < 1e-6, 1.0, mvc_scale)
    return neutral_mean, mvc_scale


def validate_calibration_data(files: Iterable[Path]) -> list[Path]:
    missing: list[Path] = []
    for path in files:
        try:
            data = np.load(path, allow_pickle=True)
            if data.get("calib_neutral_emg") is None or data.get("calib_mvc_emg") is None:
                missing.append(Path(path))
        except Exception:
            missing.append(Path(path))
    return missing


def print_missing_calibration_warning(missing_files: Iterable[Path]) -> None:
    missing = [Path(path) for path in missing_files]
    if not missing:
        return
    print(
        f"WARNING: {len(missing)} file(s) lack calibration data "
        "(calib_neutral_emg / calib_mvc_emg). "
        "Calibration normalization will be skipped for these sessions."
    )
    for path in missing[:5]:
        print(f"  {path}")
    if len(missing) > 5:
        print(f"  ... and {len(missing) - 5} more.")


def subject_from_path(path: Path) -> str:
    return Path(path).parent.parent.name


def load_strict_windows_from_file(
    path: Path,
    *,
    arm: str,
    window_size: int,
    window_step: int,
    use_calibration: bool,
    mvc_percentile: float,
    mvc_min_ratio: float,
    use_min_label_confidence: bool,
    min_label_confidence: float,
    included_gestures: set[str] | None,
    verbose_calibration_skip: bool = False,
) -> WindowedStrictFile | None:
    data = np.load(path, allow_pickle=True)
    if "emg" not in data.files or "y" not in data.files:
        return None

    emg = np.asarray(data["emg"], dtype=float)
    metadata = data.get("metadata")
    strict_layout = resolve_strict_indices_from_metadata(metadata, arm=arm)
    if emg.shape[1] != strict_layout.ordered_indices.size:
        raise ValueError(
            f"{Path(path).name}: strict layout resolved {strict_layout.ordered_indices.size} "
            f"channels for {arm}, but file has {emg.shape[1]}."
        )
    emg = emg[:, strict_layout.ordered_indices]

    if use_calibration:
        calib_neutral = data.get("calib_neutral_emg")
        calib_mvc = data.get("calib_mvc_emg")
        if calib_neutral is not None and calib_mvc is not None:
            calib_neutral = np.asarray(calib_neutral, dtype=float)[:, strict_layout.ordered_indices]
            calib_mvc = np.asarray(calib_mvc, dtype=float)[:, strict_layout.ordered_indices]
            neutral_mean, mvc_scale = compute_calibration(
                calib_neutral,
                calib_mvc,
                mvc_percentile,
                mvc_min_ratio,
                verbose=verbose_calibration_skip,
            )
            if neutral_mean is not None and mvc_scale is not None:
                emg = (emg - neutral_mean) / mvc_scale

    windows = get_windows(emg, int(window_size), int(window_step))
    labels = np.asarray(data["y"], dtype=object)

    n_windows = int(windows.shape[0])
    starts = np.arange(n_windows) * int(window_step)
    ends = starts + int(window_size)

    window_labels: list[str | None] = []
    for start, end in zip(starts, ends):
        label, confidence = majority_label_with_confidence(labels[start:end])
        if label == "neutral_buffer":
            label = None
        if (
            use_min_label_confidence
            and label is not None
            and float(confidence) < float(min_label_confidence)
        ):
            label = None
        if label is not None and included_gestures is not None and label not in included_gestures:
            label = None
        window_labels.append(label)

    label_array = np.asarray(window_labels, dtype=object)
    keep = label_array != None  # noqa: E711
    windows = windows[keep]
    label_array = label_array[keep]

    if windows.size == 0:
        return None

    return WindowedStrictFile(
        windows=windows.astype(np.float32),
        labels=label_array,
        strict_layout=strict_layout,
    )
