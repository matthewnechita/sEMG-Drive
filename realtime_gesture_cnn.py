import argparse
import csv
import json
import re
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import os
import threading

# Base directory of this script (used to build absolute model paths)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import torch
from scipy.signal import butter, sosfilt

from AeroPy.TrignoBase import TrignoBase
from AeroPy.DataManager import DataKernel

from libemg import filtering as libemg_filter
from emg.channel_layout import infer_channel_layout
from emg.gesture_model_cnn import load_cnn_bundle, quick_finetune
from emg.prototype_classifier import PrototypeClassifier
from emg.runtime_tuning import get_runtime_tuning_preset
from emg.strict_layout import (
    parse_pair_number as parse_strict_pair_number,
    resolve_strict_channel_indices,
    strict_channel_count_for_arm,
    strict_pair_numbers_for_arm,
)

RUNTIME_TUNING_PRESET = get_runtime_tuning_preset()
RUNTIME_TUNING_PRESET_NAME = RUNTIME_TUNING_PRESET.name
REALTIME_TUNING = RUNTIME_TUNING_PRESET.realtime


# ======== Config (edit as needed) ========
WINDOW_SIZE = 200
WINDOW_STEP = 100
REALTIME_FILTER_MODE = "scipy_stateful"  # "scipy_stateful" or "libemg_rolling"
FILTER_WARMUP = 200  # only used when REALTIME_FILTER_MODE == "libemg_rolling"

REALTIME_RESAMPLE = True
# Keep this aligned with your training data preprocessing.
# Set to None to use model metadata when available.
REALTIME_TARGET_FS_HZ = 2000.0

SMOOTHING = REALTIME_TUNING.smoothing
MIN_CONFIDENCE = REALTIME_TUNING.min_confidence
DUAL_ARM_AGREE_THRESHOLD = REALTIME_TUNING.dual_arm_agree_threshold
DUAL_ARM_SINGLE_THRESHOLD = REALTIME_TUNING.resolved_dual_arm_single_threshold
LOW_CONFIDENCE_LABEL = "neutral"

OUTPUT_HYSTERESIS = REALTIME_TUNING.output_hysteresis
HYSTERESIS_ACTIVE_ENTER_THRESHOLD = REALTIME_TUNING.hysteresis_active_enter_threshold
HYSTERESIS_ACTIVE_EXIT_THRESHOLD = REALTIME_TUNING.hysteresis_active_exit_threshold
HYSTERESIS_ACTIVE_SWITCH_THRESHOLD = REALTIME_TUNING.hysteresis_active_switch_threshold
HYSTERESIS_NEUTRAL_ENTER_THRESHOLD = REALTIME_TUNING.hysteresis_neutral_enter_threshold
HYSTERESIS_ENTER_CONFIRM_FRAMES = REALTIME_TUNING.hysteresis_enter_confirm_frames
HYSTERESIS_SWITCH_CONFIRM_FRAMES = REALTIME_TUNING.hysteresis_switch_confirm_frames
HYSTERESIS_NEUTRAL_CONFIRM_FRAMES = REALTIME_TUNING.hysteresis_neutral_confirm_frames


@dataclass(frozen=True)
class ArmGestureState:
    label: str = LOW_CONFIDENCE_LABEL
    confidence: float = 0.0


@dataclass(frozen=True)
class ArmPredictionTrace:
    raw_top_label: str = LOW_CONFIDENCE_LABEL
    raw_top_confidence: float = 0.0
    second_label: str = ""
    second_confidence: float = 0.0
    margin: float = 0.0
    gate_label: str = LOW_CONFIDENCE_LABEL
    gate_confidence: float = 0.0
    gate_reason: str = ""
    hysteresis_label: str = LOW_CONFIDENCE_LABEL
    hysteresis_confidence: float = 0.0


@dataclass(frozen=True)
class PredictionRanking:
    top_idx: int = -1
    top_label: str = LOW_CONFIDENCE_LABEL
    top_confidence: float = 0.0
    second_idx: int = -1
    second_label: str = ""
    second_confidence: float = 0.0
    margin: float = 0.0


@dataclass(frozen=True)
class PublishedGesture:
    arm: str = "right"
    label: str = LOW_CONFIDENCE_LABEL
    confidence: float = 0.0


@dataclass(frozen=True)
class PublishedGestureOutput:
    mode: str = "single"
    gestures: tuple[PublishedGesture, ...] = field(default_factory=tuple)
    prediction_seq: int = 0
    window_end_ts: float = 0.0
    prediction_ts: float = 0.0
    publish_ts: float = 0.0


@dataclass(frozen=True)
class DualGestureState:
    mode: str = "right"
    right: ArmGestureState = field(default_factory=ArmGestureState)
    left: ArmGestureState = field(default_factory=ArmGestureState)
    combined: ArmGestureState = field(default_factory=ArmGestureState)
    right_trace: ArmPredictionTrace = field(default_factory=ArmPredictionTrace)
    left_trace: ArmPredictionTrace = field(default_factory=ArmPredictionTrace)
    published: PublishedGestureOutput = field(default_factory=PublishedGestureOutput)
    prediction_seq: int = 0
    window_end_ts: float = 0.0
    prediction_ts: float = 0.0
    timestamp: float = 0.0


LATEST_LOCK = threading.Lock()
LATEST_STATE = DualGestureState()
LATEST_GESTURE = LOW_CONFIDENCE_LABEL
LATEST_TIMESTAMP = 0.0
LATEST_PREDICTION_SEQ = 0
PUBLISH_FILE_PATH = None


class PredictionCSVLogger:
    def __init__(self, path: str):
        self.path = str(path)
        out_path = Path(self.path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = out_path.open("w", newline="", encoding="utf-8")
        self._fieldnames = [
            "runtime_preset",
            "prediction_seq",
            "window_end_ts",
            "prediction_ts",
            "publish_ts",
            "mode",
            "published_mode",
            "pred_label",
            "pred_conf",
            "right_raw_top_label",
            "right_raw_top_conf",
            "right_second_label",
            "right_second_conf",
            "right_margin",
            "right_gate_label",
            "right_gate_conf",
            "right_gate_reason",
            "right_hysteresis_label",
            "right_hysteresis_conf",
            "right_label",
            "right_conf",
            "left_raw_top_label",
            "left_raw_top_conf",
            "left_second_label",
            "left_second_conf",
            "left_margin",
            "left_gate_label",
            "left_gate_conf",
            "left_gate_reason",
            "left_hysteresis_label",
            "left_hysteresis_conf",
            "left_label",
            "left_conf",
            "published_labels",
        ]
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
        self._writer.writeheader()

    def write_state(self, state: DualGestureState) -> None:
        published = state.published
        row = {
            "runtime_preset": RUNTIME_TUNING_PRESET_NAME,
            "prediction_seq": int(getattr(published, "prediction_seq", 0)),
            "window_end_ts": float(getattr(published, "window_end_ts", 0.0)),
            "prediction_ts": float(getattr(published, "prediction_ts", 0.0)),
            "publish_ts": float(getattr(published, "publish_ts", state.timestamp)),
            "mode": str(state.mode),
            "published_mode": str(getattr(published, "mode", "")),
            "pred_label": str(state.combined.label),
            "pred_conf": float(state.combined.confidence),
            "right_raw_top_label": str(state.right_trace.raw_top_label),
            "right_raw_top_conf": float(state.right_trace.raw_top_confidence),
            "right_second_label": str(state.right_trace.second_label),
            "right_second_conf": float(state.right_trace.second_confidence),
            "right_margin": float(state.right_trace.margin),
            "right_gate_label": str(state.right_trace.gate_label),
            "right_gate_conf": float(state.right_trace.gate_confidence),
            "right_gate_reason": str(state.right_trace.gate_reason),
            "right_hysteresis_label": str(state.right_trace.hysteresis_label),
            "right_hysteresis_conf": float(state.right_trace.hysteresis_confidence),
            "right_label": str(state.right.label),
            "right_conf": float(state.right.confidence),
            "left_raw_top_label": str(state.left_trace.raw_top_label),
            "left_raw_top_conf": float(state.left_trace.raw_top_confidence),
            "left_second_label": str(state.left_trace.second_label),
            "left_second_conf": float(state.left_trace.second_confidence),
            "left_margin": float(state.left_trace.margin),
            "left_gate_label": str(state.left_trace.gate_label),
            "left_gate_conf": float(state.left_trace.gate_confidence),
            "left_gate_reason": str(state.left_trace.gate_reason),
            "left_hysteresis_label": str(state.left_trace.hysteresis_label),
            "left_hysteresis_conf": float(state.left_trace.hysteresis_confidence),
            "left_label": str(state.left.label),
            "left_conf": float(state.left.confidence),
            "published_labels": "|".join(
                f"{gesture.arm}:{gesture.label}"
                for gesture in getattr(published, "gestures", ())
            ),
        }
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()

CALIBRATE = True
CALIB_NEUTRAL_S = 5.0
CALIB_MVC_S = 5.0
CALIB_MVC_PREP_S = 2.0    # countdown pause before MVC window
MVC_PERCENTILE = 95.0
# Keep this aligned with train_cross_subject.py (MVC_QUALITY_MIN_RATIO).
MVC_MIN_RATIO = 1.5        # allow normalization unless calibration is clearly weak

# ======== Dual-arm config ========
# Strict layout uses fixed pair identities; pairing/scan order may vary.
RIGHT_ARM_CHANNELS = strict_channel_count_for_arm("right")
RIGHT_ARM_PAIR_NUMBERS = set(strict_pair_numbers_for_arm("right"))
LEFT_ARM_PAIR_NUMBERS = set(strict_pair_numbers_for_arm("left"))
# Canonical within-arm pair order used to build model input columns.
RIGHT_ARM_PAIR_ORDER = list(strict_pair_numbers_for_arm("right"))
LEFT_ARM_PAIR_ORDER = list(strict_pair_numbers_for_arm("left"))
AUTO_DUAL_ARM_CHANNEL_MAPPING = True
# Left arm uses 5 sensors → 16 channels (one fewer sensor than right arm)
# =================================

# ======== Per-gesture calibration ========
# Collect a short sample of each gesture before inference to fine-tune
# the model's classification head for this specific subject.
# Requires CALIBRATE = True (needs filter_obj and MVC normalization).
GESTURE_CALIB        = False  # adapt Stage B head to current user/session
GESTURE_CALIB_S      = 5.0   # seconds of EMG to collect per gesture
GESTURE_CALIB_PREP_S = 2.0   # countdown pause before each gesture

# Prototype classifier (phase 1): build per-gesture embedding centroids from
# gesture calibration windows, then classify by cosine-distance-derived scores.
# Keep this on for dual-arm prototype testing.
USE_PROTOTYPE_CLASSIFIER = False
PROTOTYPE_L2_NORMALIZE = True
PROTOTYPE_TEMPERATURE = 0.20
PROTOTYPE_REJECT_MIN_CONFIDENCE = REALTIME_TUNING.prototype_reject_min_confidence
PROTOTYPE_REJECT_MIN_MARGIN = REALTIME_TUNING.prototype_reject_min_margin

# Optional runtime gesture filtering (code-only; no CLI flags).
# Example (3-class mode):
# INCLUDED_GESTURES = {"neutral", "left_turn", "right_turn"}
INCLUDED_GESTURES: set[str] | None = {"neutral", "left_turn", "right_turn"}

GESTURE_LABELS = ["neutral", "left_turn", "right_turn", "signal_left", "signal_right", "horn"]

GESTURE_INSTRUCTIONS = {
    "neutral":      "relax completely (rest position)",
    "left_turn":    "perform LEFT TURN gesture",
    "right_turn":   "perform RIGHT TURN gesture",
    "signal_left":  "perform SIGNAL LEFT gesture",
    "signal_right": "perform SIGNAL RIGHT gesture",
    "horn":         "perform HORN gesture",
}
# ==========================================

# ======== Inference mode ========
# Set MODE to control which arm(s) run inference:
#   "right" - right arm only  (pair right arm sensors first in Delsys)
#   "left"  - left arm only   (pair left arm sensors first in Delsys)
#   "dual"  - both arms inferred separately (legacy accessor still exposes one combined label)
MODE        = "dual"
# Default to Matthew strict per-subject bundles. Override with CLI flags as needed.
MODEL_RIGHT = os.path.join(
    BASE_DIR,
    "models",
    "strict",
    "per_subject",
    "right",
    "Matthew_3_gesture_15.pt",
)

MODEL_LEFT = os.path.join(
    BASE_DIR,
    "models",
    "strict",
    "per_subject",
    "left",
    "Matthew_3_gesture_15.pt",
)
# ================================

class _StreamingHandler:
    def __init__(self):
        self.streamYTData = True
        self.pauseFlag = False
        self.DataHandler: Any = None
        self.EMGplot: Any = None

    def threadManager(self, start_trigger: bool, stop_trigger: bool) -> None:
        return


def define_filters(fs):
    """
    libEMG filtering stack (must match emg/filtering.py exactly):
    - Notch @ 60 Hz  (bandwidth 3) — power line fundamental
    - Notch @ 120 Hz (bandwidth 3) — 2nd power line harmonic
    - Bandpass 20-450 Hz (order 6)
    """
    fi = libemg_filter.Filter(fs)
    fi.install_filters({"name": "notch",    "cutoff": 60,        "bandwidth": 3})
    fi.install_filters({"name": "notch",    "cutoff": 120,       "bandwidth": 3})
    fi.install_filters({"name": "bandpass", "cutoff": [20, 450], "order": 6})
    return fi


def apply_filters(fi, emg):
    filtered_data = fi.filter(emg)
    return np.array(filtered_data, dtype=float)


def _define_realtime_stateful_filters(fs):
    """
    Causal SOS stack for realtime use.

    This approximates the active offline filter settings:
    - notch 60 Hz  (bandwidth 3 Hz)
    - notch 120 Hz (bandwidth 3 Hz)
    - bandpass 20-450 Hz, order 6
    """
    sos_n60 = butter(2, [58.5, 61.5], btype="bandstop", fs=fs, output="sos")
    sos_n120 = butter(2, [118.5, 121.5], btype="bandstop", fs=fs, output="sos")
    sos_bp = butter(6, [20.0, 450.0], btype="bandpass", fs=fs, output="sos")
    return (sos_n60, sos_n120, sos_bp)


def _make_realtime_filter_state(filters, num_channels):
    sos_n60, sos_n120, sos_bp = filters
    channels = int(num_channels)
    return [
        np.zeros((sos_n60.shape[0], 2, channels), dtype=float),
        np.zeros((sos_n120.shape[0], 2, channels), dtype=float),
        np.zeros((sos_bp.shape[0], 2, channels), dtype=float),
    ]


def _apply_filters_stateful(filters, samples, state):
    samples_arr = np.asarray(samples, dtype=float)
    if samples_arr.size == 0:
        return samples_arr.reshape(0, 0), state

    sos_n60, sos_n120, sos_bp = filters
    zi_n60, zi_n120, zi_bp = state
    out, zi_n60 = sosfilt(sos_n60, samples_arr, axis=0, zi=zi_n60)
    out, zi_n120 = sosfilt(sos_n120, out, axis=0, zi=zi_n120)
    out, zi_bp = sosfilt(sos_bp, out, axis=0, zi=zi_bp)
    return np.asarray(out, dtype=float), [zi_n60, zi_n120, zi_bp]


def _make_windows(emg, window_size=WINDOW_SIZE, step=WINDOW_STEP):
    """Slide a window over emg (N_samples, C) → ndarray (N_windows, C, window_size)."""
    n = len(emg)
    starts = list(range(0, n - window_size + 1, step))
    if not starts:
        return np.empty((0, emg.shape[1], window_size), dtype=np.float32)
    return np.stack([emg[s:s + window_size].T for s in starts]).astype(np.float32)


def _pair_time_value(sample):
    if hasattr(sample, "Item1"):
        return float(sample.Item1), float(sample.Item2)
    return float(sample[0]), float(sample[1])


def _estimate_fs(times):
    times = np.asarray(times, dtype=float)
    if times.size < 3:
        return None
    diffs = np.diff(times)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None
    return 1.0 / float(np.median(diffs))


def _parse_yt_frame(out):
    """Unpack a GetYTData() frame into per-channel times and values.

    Preserves channel positions (empty channels become empty lists).
    """
    channel_times = []
    channel_values = []
    for channel in out:
        if not channel:
            channel_times.append([])
            channel_values.append([])
            continue
        chan_array = np.asarray(channel[0], dtype=object)
        if chan_array.size == 0:
            channel_times.append([])
            channel_values.append([])
            continue
        t_vals, v_vals = zip(*(_pair_time_value(s) for s in chan_array))
        channel_times.append(list(t_vals))
        channel_values.append(list(v_vals))
    return channel_times, channel_values


def _resolve_target_fs_hz_from_bundle(bundle):
    metadata = bundle.metadata if isinstance(bundle.metadata, dict) else {}
    resampling_meta = metadata.get("resampling")
    candidates = [
        metadata.get("target_fs_hz"),
        metadata.get("sampling_rate_hz"),
        metadata.get("fs"),
    ]
    if isinstance(resampling_meta, dict):
        candidates.append(resampling_meta.get("target_fs_hz"))

    for value in candidates:
        if value is None:
            continue
        try:
            fs_hz = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fs_hz) and fs_hz > 0:
            return fs_hz
    return None


def _bundle_channel_layout_meta(bundle):
    metadata = bundle.metadata if isinstance(bundle.metadata, dict) else {}
    layout_meta = metadata.get("channel_layout")
    return layout_meta if isinstance(layout_meta, dict) else {}


def _bundle_layout_mode(bundle):
    value = _bundle_channel_layout_meta(bundle).get("layout_mode")
    return str(value).strip().lower() if value is not None else ""


def _bundle_uses_strict_layout(bundle):
    return _bundle_layout_mode(bundle) == "strict"


def _bundle_uses_type_layout(bundle):
    return bool(_bundle_channel_layout_meta(bundle).get("type_canonicalization_enabled"))


def _bundle_expected_kind_counts(bundle):
    counts = _bundle_channel_layout_meta(bundle).get("kind_counts")
    if not isinstance(counts, dict):
        return {}
    out = {}
    for key, value in counts.items():
        try:
            out[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return out


def _ordered_indices_from_strict_layout(channel_labels, bundle, arm_label):
    if not _bundle_uses_strict_layout(bundle):
        return None

    layout_meta = _bundle_channel_layout_meta(bundle)
    expected_arm = str(layout_meta.get("arm") or arm_label).strip().lower()
    try:
        resolved = resolve_strict_channel_indices(channel_labels, arm=expected_arm)
    except Exception as exc:
        raise RuntimeError(f"[gesture:{arm_label}] strict layout validation failed: {exc}") from exc

    if resolved.ordered_indices.size != int(bundle.channel_count):
        raise RuntimeError(
            f"[gesture:{arm_label}] strict layout resolved {resolved.ordered_indices.size} channels, "
            f"but model expects {bundle.channel_count}."
        )

    expected_pairs = layout_meta.get("pair_order")
    if isinstance(expected_pairs, (list, tuple)):
        expected_pairs = [int(value) for value in expected_pairs]
        if list(resolved.pair_numbers) != expected_pairs:
            raise RuntimeError(
                f"[gesture:{arm_label}] strict pair order mismatch: "
                f"live={list(resolved.pair_numbers)} bundle={expected_pairs}"
            )

    print(f"[gesture] {arm_label}-arm strict slots: {list(resolved.slot_names)}")
    print(f"[gesture] {arm_label}-arm strict pairs: {list(resolved.pair_numbers)}")
    print(
        f"[gesture] {arm_label}-arm strict indices ({len(resolved.ordered_indices)}): "
        f"{resolved.ordered_indices.tolist()}"
    )
    return resolved.ordered_indices


def _ordered_indices_from_type_layout(
    channel_labels,
    base_indices,
    target_channels,
    bundle,
    arm_label,
):
    base_idx = np.asarray(base_indices, dtype=int)
    if base_idx.size != int(target_channels):
        print(
            f"[gesture:{arm_label}] type canonicalization skipped: "
            f"candidate subset has {base_idx.size} channels, expected {target_channels}."
        )
        return None

    subset_labels = [str(channel_labels[int(i)]) for i in base_idx]
    layout = infer_channel_layout(channel_labels=subset_labels, channel_count=base_idx.size)
    if layout is None:
        print(f"[gesture:{arm_label}] type canonicalization unavailable from live channel labels.")
        return None

    ordered = base_idx[layout.ordered_indices]
    expected_kind_counts = _bundle_expected_kind_counts(bundle)
    if expected_kind_counts and layout.kind_counts != expected_kind_counts:
        print(
            f"[gesture:{arm_label}] WARNING: live channel kind counts "
            f"{layout.kind_counts} do not match bundle metadata {expected_kind_counts}."
        )
    print(
        f"[gesture] {arm_label}-arm type layout via {layout.source}: "
        f"{layout.kind_counts}"
    )
    print(f"[gesture] {arm_label}-arm canonical indices ({len(ordered)}): {ordered.tolist()}")
    return ordered


def _canonicalize_indices_by_type(channel_labels, indices, bundle, arm_label):
    if not _bundle_uses_type_layout(bundle):
        return np.asarray(indices, dtype=int)
    ordered = _ordered_indices_from_type_layout(
        channel_labels,
        indices,
        bundle.channel_count,
        bundle,
        arm_label,
    )
    if ordered is None:
        return np.asarray(indices, dtype=int)
    return ordered


def _reorder_calibration_vectors(neutral_mean, mvc_scale, original_indices, ordered_indices):
    if neutral_mean is None or mvc_scale is None:
        return neutral_mean, mvc_scale
    original = np.asarray(original_indices, dtype=int)
    ordered = np.asarray(ordered_indices, dtype=int)
    if original.size != ordered.size:
        return neutral_mean, mvc_scale
    lookup = {int(idx): pos for pos, idx in enumerate(original.tolist())}
    try:
        perm = [lookup[int(idx)] for idx in ordered.tolist()]
    except KeyError:
        return neutral_mean, mvc_scale
    return neutral_mean[perm], mvc_scale[perm]


class _RealtimeTimestampResampler:
    """Timestamp-align all channels to a common fixed-rate grid."""

    def __init__(self, channel_count, target_fs_hz, max_buffer_s=2.0):
        if target_fs_hz <= 0:
            raise ValueError("target_fs_hz must be > 0.")
        self.channel_count = int(channel_count)
        self.target_fs_hz = float(target_fs_hz)
        self.step_s = 1.0 / self.target_fs_hz
        self.max_buffer_s = float(max_buffer_s)
        self._next_t = None
        self._time_buf = [deque() for _ in range(self.channel_count)]
        self._value_buf = [deque() for _ in range(self.channel_count)]

    def _append_channel_samples(self, ch_idx, times, values):
        if times is None or values is None:
            return
        t_arr = np.asarray(times, dtype=float).reshape(-1)
        v_arr = np.asarray(values, dtype=float).reshape(-1)
        n = min(t_arr.size, v_arr.size)
        if n <= 0:
            return
        t_arr = t_arr[:n]
        v_arr = v_arr[:n]

        mask = np.isfinite(t_arr) & np.isfinite(v_arr)
        if not np.any(mask):
            return
        t_arr = t_arr[mask]
        v_arr = v_arr[mask]
        if t_arr.size == 0:
            return

        order = np.argsort(t_arr, kind="mergesort")
        t_arr = t_arr[order]
        v_arr = v_arr[order]

        t_buf = self._time_buf[ch_idx]
        v_buf = self._value_buf[ch_idx]
        last_t = t_buf[-1] if t_buf else None

        for t_val, v_val in zip(t_arr, v_arr):
            if last_t is not None and t_val <= last_t:
                continue
            t_buf.append(float(t_val))
            v_buf.append(float(v_val))
            last_t = float(t_val)

    def push(self, channel_times, channel_values):
        if channel_times is None or channel_values is None:
            return np.empty((0, self.channel_count), dtype=float), np.empty((0,), dtype=float)

        n_seen = min(len(channel_times), len(channel_values), self.channel_count)
        for ch_idx in range(n_seen):
            self._append_channel_samples(ch_idx, channel_times[ch_idx], channel_values[ch_idx])

        if any(len(tb) < 2 for tb in self._time_buf):
            return np.empty((0, self.channel_count), dtype=float), np.empty((0,), dtype=float)

        overlap_start = max(tb[0] for tb in self._time_buf)
        overlap_end = min(tb[-1] for tb in self._time_buf)
        if self._next_t is None:
            self._next_t = float(overlap_start)

        if overlap_end < self._next_t:
            return np.empty((0, self.channel_count), dtype=float), np.empty((0,), dtype=float)

        n_out = int(np.floor((overlap_end - self._next_t) * self.target_fs_hz)) + 1
        if n_out <= 0:
            return np.empty((0, self.channel_count), dtype=float), np.empty((0,), dtype=float)

        t_grid = self._next_t + (np.arange(n_out, dtype=float) / self.target_fs_hz)
        out = np.empty((n_out, self.channel_count), dtype=float)
        for ch_idx in range(self.channel_count):
            t_vec = np.asarray(self._time_buf[ch_idx], dtype=float)
            v_vec = np.asarray(self._value_buf[ch_idx], dtype=float)
            out[:, ch_idx] = np.interp(t_grid, t_vec, v_vec)

        self._next_t = float(t_grid[-1] + self.step_s)

        # Keep one point before next_t for interpolation continuity.
        for ch_idx in range(self.channel_count):
            t_buf = self._time_buf[ch_idx]
            v_buf = self._value_buf[ch_idx]
            while len(t_buf) >= 2 and t_buf[1] <= self._next_t:
                t_buf.popleft()
                v_buf.popleft()

            # Optional memory cap for long runs.
            cutoff = self._next_t - self.max_buffer_s
            while len(t_buf) >= 2 and t_buf[1] < cutoff:
                t_buf.popleft()
                v_buf.popleft()

        return out, t_grid

def control_hook(gesture: str) -> None:
    set_latest_gesture(gesture)
    return


def _published_output_to_dict(output: PublishedGestureOutput) -> dict[str, Any]:
    return {
        "mode": str(output.mode),
        "prediction_seq": int(output.prediction_seq),
        "window_end_ts": float(output.window_end_ts),
        "prediction_ts": float(output.prediction_ts),
        "publish_ts": float(output.publish_ts),
        "gestures": [
            {
                "arm": str(gesture.arm),
                "label": str(gesture.label),
                "confidence": float(gesture.confidence),
            }
            for gesture in output.gestures
        ],
    }


def _prediction_trace_to_dict(trace: ArmPredictionTrace) -> dict[str, Any]:
    return {
        "raw_top_label": str(trace.raw_top_label),
        "raw_top_confidence": float(trace.raw_top_confidence),
        "second_label": str(trace.second_label),
        "second_confidence": float(trace.second_confidence),
        "margin": float(trace.margin),
        "gate_label": str(trace.gate_label),
        "gate_confidence": float(trace.gate_confidence),
        "gate_reason": str(trace.gate_reason),
        "hysteresis_label": str(trace.hysteresis_label),
        "hysteresis_confidence": float(trace.hysteresis_confidence),
    }


def _write_latest_state_file(state: DualGestureState) -> None:
    if not PUBLISH_FILE_PATH:
        return

    payload = {
        "timestamp": float(state.timestamp),
        "prediction_seq": int(state.prediction_seq),
        "window_end_ts": float(state.window_end_ts),
        "prediction_ts": float(state.prediction_ts),
        "mode": str(state.mode),
        "right": {
            "label": str(state.right.label),
            "confidence": float(state.right.confidence),
            "trace": _prediction_trace_to_dict(state.right_trace),
        },
        "left": {
            "label": str(state.left.label),
            "confidence": float(state.left.confidence),
            "trace": _prediction_trace_to_dict(state.left_trace),
        },
        "combined": {
            "label": str(state.combined.label),
            "confidence": float(state.combined.confidence),
        },
        "published": _published_output_to_dict(state.published),
    }

    tmp_path = f"{PUBLISH_FILE_PATH}.tmp"
    try:
        parent = os.path.dirname(PUBLISH_FILE_PATH)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.replace(tmp_path, PUBLISH_FILE_PATH)
    except OSError:
        pass


def set_latest_dual_state(
    *,
    mode: str,
    right_label: str,
    right_confidence: float,
    left_label: str = LOW_CONFIDENCE_LABEL,
    left_confidence: float = 0.0,
    right_trace: ArmPredictionTrace | None = None,
    left_trace: ArmPredictionTrace | None = None,
    combined_label: str | None = None,
    combined_confidence: float | None = None,
    published: PublishedGestureOutput | None = None,
    window_end_ts: float = 0.0,
    prediction_ts: float = 0.0,
) -> DualGestureState:
    """Publish the latest per-arm gesture state.

    `combined_*` is retained only for callers that still consume the legacy
    single-label interface via `get_latest_gesture()`.
    """
    global LATEST_STATE, LATEST_GESTURE, LATEST_TIMESTAMP, LATEST_PREDICTION_SEQ

    mode_name = str(mode)
    right_state = ArmGestureState(str(right_label), float(right_confidence))
    left_state = ArmGestureState(str(left_label), float(left_confidence))
    if right_trace is None:
        right_trace = ArmPredictionTrace(
            gate_label=right_state.label,
            gate_confidence=right_state.confidence,
            hysteresis_label=right_state.label,
            hysteresis_confidence=right_state.confidence,
        )
    if left_trace is None:
        left_trace = ArmPredictionTrace(
            gate_label=left_state.label,
            gate_confidence=left_state.confidence,
            hysteresis_label=left_state.label,
            hysteresis_confidence=left_state.confidence,
        )
    if combined_label is None:
        combined_label = right_state.label if mode_name != "dual" else LOW_CONFIDENCE_LABEL
    if combined_confidence is None:
        combined_confidence = right_state.confidence if mode_name != "dual" else 0.0
    combined_state = ArmGestureState(str(combined_label), float(combined_confidence))
    LATEST_PREDICTION_SEQ += 1
    prediction_seq = int(LATEST_PREDICTION_SEQ)
    timestamp = time.time()
    if published is None:
        active_arm = mode_name if mode_name in {"right", "left"} else "right"
        published = PublishedGestureOutput(
            mode="single",
            gestures=(PublishedGesture(active_arm, combined_state.label, combined_state.confidence),),
            prediction_seq=prediction_seq,
            window_end_ts=float(window_end_ts),
            prediction_ts=float(prediction_ts),
            publish_ts=float(timestamp),
        )
    else:
        published = replace(
            published,
            prediction_seq=prediction_seq,
            window_end_ts=float(window_end_ts),
            prediction_ts=float(prediction_ts),
            publish_ts=float(timestamp),
        )
    state = DualGestureState(
        mode=mode_name,
        right=right_state,
        left=left_state,
        combined=combined_state,
        right_trace=right_trace,
        left_trace=left_trace,
        published=published,
        prediction_seq=prediction_seq,
        window_end_ts=float(window_end_ts),
        prediction_ts=float(prediction_ts),
        timestamp=timestamp,
    )
    with LATEST_LOCK:
        LATEST_STATE = state
        LATEST_GESTURE = combined_state.label
        LATEST_TIMESTAMP = timestamp
    _write_latest_state_file(state)
    return state


def set_latest_gesture(label: str) -> None:
    set_latest_dual_state(
        mode="legacy",
        right_label=str(label),
        right_confidence=0.0,
        combined_label=str(label),
        combined_confidence=0.0,
    )


def get_latest_dual_state() -> tuple[DualGestureState, float]:
    with LATEST_LOCK:
        state = LATEST_STATE
        ts = LATEST_TIMESTAMP if LATEST_TIMESTAMP else state.timestamp
        age = (time.time() - ts) if ts else float('inf')
        return state, age


def get_latest_published_gestures() -> tuple[PublishedGestureOutput, float]:
    state, age = get_latest_dual_state()
    return state.published, age


def get_latest_gesture(default: str = LOW_CONFIDENCE_LABEL):
    with LATEST_LOCK:
        label = LATEST_GESTURE if LATEST_GESTURE is not None else default
        ts = LATEST_TIMESTAMP
        age = (time.time() - ts) if ts else float('inf')
        return label, age


def resolve_published_gesture_output(
    mode: str,
    right: ArmGestureState,
    left: ArmGestureState,
    combined: ArmGestureState,
) -> PublishedGestureOutput:
    mode_name = str(mode)
    if mode_name != "dual":
        active_arm = mode_name if mode_name in {"right", "left"} else "right"
        return PublishedGestureOutput(
            mode="single",
            gestures=(PublishedGesture(active_arm, combined.label, combined.confidence),),
        )

    if right.label == left.label:
        return PublishedGestureOutput(
            mode="single",
            gestures=(PublishedGesture("dual", combined.label, combined.confidence),),
        )

    return PublishedGestureOutput(
        mode="split",
        gestures=(
            PublishedGesture("right", right.label, right.confidence),
            PublishedGesture("left", left.label, left.confidence),
        ),
    )


def format_published_gesture_output(output: PublishedGestureOutput) -> str:
    gestures = tuple(output.gestures)
    if not gestures:
        return f"Gesture: {LOW_CONFIDENCE_LABEL} (conf 0.00)"

    if output.mode == "split":
        parts = [
            f"{gesture.arm}={gesture.label} ({gesture.confidence:.2f})"
            for gesture in gestures
        ]
        return "Gesture: " + " | ".join(parts)

    gesture = gestures[0]
    return f"Gesture: {gesture.label} (conf {gesture.confidence:.2f})"


def published_gesture_signature(output: PublishedGestureOutput):
    gestures = tuple(
        (str(gesture.arm), str(gesture.label))
        for gesture in output.gestures
    )
    return (str(output.mode), gestures)


def fuse_predictions(label_r, conf_r, label_l, conf_l):
    """Combine right and left arm predictions into one fused label.

    If both arms agree on the same gesture, the prediction is reinforced and
    a lower confidence threshold is used to emit it. If they disagree, each
    arm is evaluated independently against the single-arm threshold.

    Returns: (fused_label, fused_confidence)
    """
    label_r = str(label_r)
    label_l = str(label_l)
    neutral = str(LOW_CONFIDENCE_LABEL)

    if label_r == label_l:
        # Both arms agree — strengthen the prediction
        combined_conf = max(conf_r, conf_l)
        if combined_conf >= DUAL_ARM_AGREE_THRESHOLD:
            return label_r, combined_conf

    # Arms disagree (or agree but below threshold):
    # Prefer non-neutral labels over neutral fallback when confidence is sufficient.
    r_ok = conf_r >= DUAL_ARM_SINGLE_THRESHOLD
    l_ok = conf_l >= DUAL_ARM_SINGLE_THRESHOLD

    r_active = r_ok and label_r != neutral
    l_active = l_ok and label_l != neutral

    if r_active and l_active:
        if conf_r >= conf_l:
            return label_r, conf_r
        return label_l, conf_l
    if r_active:
        return label_r, conf_r
    if l_active:
        return label_l, conf_l

    # If neither arm has a confident active label, fall back to confident neutral.
    if r_ok and label_r == neutral:
        return label_r, conf_r
    if l_ok and label_l == neutral:
        return label_l, conf_l

    return LOW_CONFIDENCE_LABEL, 0.0


def _canonical_label_map(index_to_label):
    """Normalise bundle label maps to {int_index: str_label}."""
    canonical = {}
    for idx, label in index_to_label.items():
        canonical[int(idx)] = str(label)
    return canonical


def _resolve_allowed_labels(index_to_label, arm_name):
    """Resolve the active label subset for one arm."""
    model_labels = {str(label) for label in index_to_label.values()}

    if INCLUDED_GESTURES is None:
        allowed_labels = set(model_labels)
    else:
        requested = {str(label) for label in INCLUDED_GESTURES}
        missing = sorted(requested - model_labels)
        if missing:
            print(f"[gesture:{arm_name}] INCLUDED_GESTURES missing from model and ignored: {missing}")
        allowed_labels = requested & model_labels

    if not allowed_labels:
        raise ValueError(f"No labels left for {arm_name} arm after applying INCLUDED_GESTURES.")

    allowed_indices = {idx for idx, label in index_to_label.items() if label in allowed_labels}
    print(f"[gesture:{arm_name}] active labels: {sorted(allowed_labels)}")
    return allowed_labels, allowed_indices


def _restrict_probs(probs, allowed_indices):
    """Zero out disallowed classes, then renormalize over allowed classes."""
    if len(allowed_indices) >= len(probs):
        return probs
    filtered = np.array(probs, dtype=float, copy=True)
    mask = np.zeros_like(filtered, dtype=bool)
    idx = np.array(sorted(allowed_indices), dtype=int)
    mask[idx] = True
    filtered[~mask] = 0.0
    total = float(filtered.sum())
    if total > 0.0:
        filtered /= total
    return filtered


def _rank_prediction_probs(probs, index_to_label):
    probs = np.asarray(probs, dtype=float).reshape(-1)
    if probs.size == 0:
        return PredictionRanking()

    order = np.argsort(probs)[::-1]
    top_idx = int(order[0])
    top_label = index_to_label.get(top_idx, LOW_CONFIDENCE_LABEL)
    top_conf = float(probs[top_idx])

    if probs.size > 1:
        second_idx = int(order[1])
        second_label = index_to_label.get(second_idx, "")
        second_conf = float(probs[second_idx])
    else:
        second_idx = -1
        second_label = ""
        second_conf = 0.0

    return PredictionRanking(
        top_idx=top_idx,
        top_label=str(top_label),
        top_confidence=top_conf,
        second_idx=second_idx,
        second_label=str(second_label),
        second_confidence=second_conf,
        margin=float(top_conf - second_conf),
    )


def _decode_prediction(probs, index_to_label):
    """Decode probs with a single confidence gate."""
    ranking = _rank_prediction_probs(probs, index_to_label)
    pred_idx = int(ranking.top_idx)
    pred_label = str(ranking.top_label)
    pred_conf = float(ranking.top_confidence)
    gate_reason = ""

    if pred_conf < float(MIN_CONFIDENCE):
        pred_label = LOW_CONFIDENCE_LABEL
        gate_reason = "min_confidence"

    return ranking, pred_label, pred_conf, pred_idx, gate_reason


class _OutputHysteresis:
    """Latch output labels to suppress brief confidence dips and one-frame flips."""

    def __init__(self):
        self.current_label = str(LOW_CONFIDENCE_LABEL)
        self.current_conf = 0.0
        self.candidate_label = None
        self.candidate_conf = 0.0
        self.candidate_count = 0

    def _reset_candidate(self):
        self.candidate_label = None
        self.candidate_conf = 0.0
        self.candidate_count = 0

    def _observe_candidate(self, label, conf):
        if self.candidate_label == label:
            self.candidate_count += 1
            self.candidate_conf = max(float(conf), self.candidate_conf)
            return
        self.candidate_label = str(label)
        self.candidate_conf = float(conf)
        self.candidate_count = 1

    def _switch_to(self, label, conf):
        self.current_label = str(label)
        self.current_conf = float(conf)
        self._reset_candidate()
        return self.current_label, self.current_conf

    def update(self, label, conf):
        label = str(label)
        conf = float(conf)
        neutral = str(LOW_CONFIDENCE_LABEL)
        if not OUTPUT_HYSTERESIS:
            return self._switch_to(label, conf)

        active_enter = max(float(MIN_CONFIDENCE), float(HYSTERESIS_ACTIVE_ENTER_THRESHOLD))
        active_switch = max(float(MIN_CONFIDENCE), float(HYSTERESIS_ACTIVE_SWITCH_THRESHOLD))
        neutral_enter = max(float(MIN_CONFIDENCE), float(HYSTERESIS_NEUTRAL_ENTER_THRESHOLD))

        if self.current_label == neutral:
            if label != neutral and conf >= active_enter:
                self._observe_candidate(label, conf)
                if self.candidate_count >= int(max(1, HYSTERESIS_ENTER_CONFIRM_FRAMES)):
                    return self._switch_to(label, self.candidate_conf)
            else:
                self._reset_candidate()
                self.current_conf = conf if label == neutral else 0.0
            return self.current_label, self.current_conf

        # Currently holding an active gesture.
        if label == self.current_label:
            self._reset_candidate()
            if conf >= float(HYSTERESIS_ACTIVE_EXIT_THRESHOLD):
                self.current_conf = conf
            return self.current_label, self.current_conf

        if label == neutral:
            if conf >= neutral_enter:
                self._observe_candidate(label, conf)
                if self.candidate_count >= int(max(1, HYSTERESIS_NEUTRAL_CONFIRM_FRAMES)):
                    return self._switch_to(neutral, self.candidate_conf)
            else:
                self._reset_candidate()
            return self.current_label, self.current_conf

        # Competing active label: require sustained high-confidence evidence.
        if conf >= active_switch:
            self._observe_candidate(label, conf)
            if self.candidate_count >= int(max(1, HYSTERESIS_SWITCH_CONFIRM_FRAMES)):
                return self._switch_to(label, self.candidate_conf)
        else:
            self._reset_candidate()
        return self.current_label, self.current_conf


def _apply_softmax_reject_gate(pred_label, pred_conf, pred_idx, ranking):
    """Reject ambiguous softmax decisions to neutral."""
    if not REALTIME_TUNING.softmax_reject_enabled:
        return pred_label, pred_conf, pred_idx, ""

    if (
        float(ranking.top_confidence) < float(REALTIME_TUNING.softmax_reject_min_confidence)
        or float(ranking.margin) < float(REALTIME_TUNING.softmax_reject_min_margin)
    ):
        return LOW_CONFIDENCE_LABEL, float(ranking.top_confidence), pred_idx, "softmax_margin"
    return pred_label, pred_conf, pred_idx, ""


def _apply_prototype_reject_gate(pred_label, pred_conf, pred_idx, ranking):
    """Reject ambiguous prototype decisions to neutral."""
    if (
        float(ranking.top_confidence) < float(PROTOTYPE_REJECT_MIN_CONFIDENCE)
        or float(ranking.margin) < float(PROTOTYPE_REJECT_MIN_MARGIN)
    ):
        return LOW_CONFIDENCE_LABEL, float(ranking.top_confidence), pred_idx, "prototype_margin"
    return pred_label, pred_conf, pred_idx, ""


def _collect_samples(handler, duration_s, stream_channels, model_channels, poll_sleep):
    samples = []
    times = []
    end_time = time.time() + duration_s

    while time.time() < end_time:
        out = handler.DataHandler.GetYTData()
        if out is None:
            time.sleep(poll_sleep)
            continue

        channel_times, channel_values = _parse_yt_frame(out)

        if len(channel_values) < stream_channels:
            continue

        sample_count = min(len(c) for c in channel_values)
        if sample_count == 0:
            continue

        for idx in range(sample_count):
            sample = [channel_values[ch][idx] for ch in range(stream_channels)]
            if model_channels is not None:
                if len(sample) < model_channels:
                    sample.extend([0.0] * (model_channels - len(sample)))
                elif len(sample) > model_channels:
                    sample = sample[:model_channels]
            samples.append(sample)
            if channel_times:
                times.append(channel_times[0][idx])

    if not samples:
        target_channels = model_channels if model_channels is not None else stream_channels
        return np.empty((0, target_channels), dtype=float), np.asarray(times, dtype=float)
    return np.asarray(samples, dtype=float), np.asarray(times, dtype=float)


def _align_calibration_vectors(neutral_mean, mvc_scale, target_channels, arm_label):
    """Trim/pad calibration vectors to match current filtered channel count."""
    if neutral_mean is not None:
        neutral_mean = np.asarray(neutral_mean, dtype=float).reshape(-1)
    if mvc_scale is not None:
        mvc_scale = np.asarray(mvc_scale, dtype=float).reshape(-1)

    if neutral_mean is not None and neutral_mean.size != target_channels:
        old = neutral_mean.size
        if old > target_channels:
            neutral_mean = neutral_mean[:target_channels]
        else:
            neutral_mean = np.pad(neutral_mean, (0, target_channels - old), mode="constant", constant_values=0.0)
        print(f"[calib:{arm_label}] adjusted neutral_mean channels {old} -> {target_channels}")

    if mvc_scale is not None and mvc_scale.size != target_channels:
        old = mvc_scale.size
        if old > target_channels:
            mvc_scale = mvc_scale[:target_channels]
        else:
            mvc_scale = np.pad(mvc_scale, (0, target_channels - old), mode="constant", constant_values=1.0)
        print(f"[calib:{arm_label}] adjusted mvc_scale channels {old} -> {target_channels}")

    return neutral_mean, mvc_scale


def _slice_channels(batch_arr, start, target_channels):
    """Return a fixed-width channel slice using zero-pad for missing channels."""
    out = batch_arr[:, start:start + target_channels]
    if out.shape[1] < target_channels:
        out = np.pad(
            out,
            ((0, 0), (0, target_channels - out.shape[1])),
            mode="constant",
            constant_values=0.0,
        )
    elif out.shape[1] > target_channels:
        out = out[:, :target_channels]
    return out


def _slice_channels_by_indices(batch_arr, indices, target_channels):
    """Select arbitrary channel indices and keep output width fixed."""
    arr = np.asarray(batch_arr, dtype=float)
    idx = np.asarray(indices, dtype=int).reshape(-1)
    idx = idx[(idx >= 0) & (idx < arr.shape[1])]
    out = arr[:, idx] if idx.size else np.empty((arr.shape[0], 0), dtype=float)
    if out.shape[1] < target_channels:
        out = np.pad(
            out,
            ((0, 0), (0, target_channels - out.shape[1])),
            mode="constant",
            constant_values=0.0,
        )
    elif out.shape[1] > target_channels:
        out = out[:, :target_channels]
    return out


def _parse_pair_number(channel_label):
    return parse_strict_pair_number(channel_label)


def _build_pair_groups(channel_labels):
    """Return ordered pair groups: key -> list[channel_idx]."""
    groups = {}
    for idx, label in enumerate(channel_labels):
        pair = _parse_pair_number(label)
        key = f"pair:{pair}" if pair is not None else f"ch:{idx}"
        groups.setdefault(key, []).append(idx)
    return groups


def _pair_order_for_arm(explicit_order, explicit_pairs):
    if explicit_order:
        return [int(p) for p in explicit_order]
    if explicit_pairs:
        return sorted(int(p) for p in explicit_pairs)
    return []


def _ordered_indices_from_pairs(channel_labels, pair_order, target_channels, arm_label):
    """Return channel indices ordered by the requested pair-number sequence."""
    if not pair_order:
        return None

    groups = _build_pair_groups(channel_labels)
    pair_to_indices = {}
    for idxs in groups.values():
        idx_arr = np.asarray(idxs, dtype=int)
        pair = _parse_pair_number(channel_labels[int(idx_arr[0])])
        if pair is None:
            continue
        pair_to_indices[int(pair)] = idx_arr

    missing = [int(pair) for pair in pair_order if int(pair) not in pair_to_indices]
    if missing:
        print(f"[gesture:{arm_label}] missing pair(s) for ordered mapping: {missing}")
        return None

    ordered = []
    for pair in pair_order:
        ordered.extend(pair_to_indices[int(pair)].tolist())

    ordered_idx = np.asarray(ordered, dtype=int)
    if ordered_idx.size != target_channels:
        print(
            f"[gesture:{arm_label}] ordered pair mapping size mismatch: "
            f"{ordered_idx.size}/{target_channels}"
        )
        return None
    return ordered_idx


def _derive_dual_indices_from_pairs(
    channel_labels,
    right_channels,
    left_channels,
    right_ratio_full,
    left_ratio_full,
):
    """Infer per-arm channel indices using pair-level calibration asymmetry."""
    groups = _build_pair_groups(channel_labels)
    diff = np.asarray(right_ratio_full, dtype=float) - np.asarray(left_ratio_full, dtype=float)
    items = []
    for key, idxs in groups.items():
        idx_arr = np.asarray(idxs, dtype=int)
        score = float(np.sum(diff[idx_arr]))
        items.append((key, idx_arr, idx_arr.size, score))

    # If explicit pair-number sets are provided, use them.
    if RIGHT_ARM_PAIR_NUMBERS or LEFT_ARM_PAIR_NUMBERS:
        right_order = _pair_order_for_arm(RIGHT_ARM_PAIR_ORDER, RIGHT_ARM_PAIR_NUMBERS)
        left_order = _pair_order_for_arm(LEFT_ARM_PAIR_ORDER, LEFT_ARM_PAIR_NUMBERS)
        right_idx = _ordered_indices_from_pairs(
            channel_labels, right_order, right_channels, "right"
        )
        left_idx = _ordered_indices_from_pairs(
            channel_labels, left_order, left_channels, "left"
        )
        if right_idx is not None and left_idx is not None:
            return right_idx, left_idx
        print(
            "[gesture] explicit RIGHT/LEFT pair mapping count mismatch; "
            "falling back to auto mapping."
        )

    # Auto mapping: choose pair groups for right arm that maximize (right-left) score
    # with exact right channel count, then assign remaining channels to left.
    n = len(items)
    best_mask = None
    best_score = None
    for mask in range(1 << n):
        count = 0
        score = 0.0
        for i in range(n):
            if (mask >> i) & 1:
                count += items[i][2]
                score += items[i][3]
        if count != right_channels:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_mask = mask

    if best_mask is None:
        raise RuntimeError(
            "Unable to infer dual-arm channel mapping from calibration data "
            f"(need exact {right_channels}/{left_channels} channel split)."
        )

    right_idx = []
    left_idx = []
    for i, (_, idx_arr, _, _) in enumerate(items):
        if (best_mask >> i) & 1:
            right_idx.extend(idx_arr.tolist())
        else:
            left_idx.extend(idx_arr.tolist())
    right_idx = np.asarray(sorted(right_idx), dtype=int)
    left_idx = np.asarray(sorted(left_idx), dtype=int)
    if right_idx.size != right_channels or left_idx.size != left_channels:
        raise RuntimeError(
            f"Inferred mapping size mismatch: right={right_idx.size}/{right_channels}, "
            f"left={left_idx.size}/{left_channels}."
        )
    return right_idx, left_idx


def main(argv=None):
    if MODE not in ("right", "left", "dual"):
        raise ValueError(f"MODE must be 'right', 'left', or 'dual', got {MODE!r}")

    # Derive defaults from config; CLI args can still override.
    # In "left" mode, the left model runs as single-arm (left sensors paired first).
    _config_right = MODEL_LEFT if MODE == "left" else MODEL_RIGHT
    _config_left = MODEL_LEFT if MODE == "dual" else None

    parser = argparse.ArgumentParser(description="Real-time CNN gesture inference.")
    parser.add_argument("--model",       default=_config_right,
                        help="Right arm model (overrides MODEL_RIGHT / MODEL_LEFT config).")
    parser.add_argument("--model-right", default=None,
                        help="Right arm model alias; overrides --model if both given.")
    parser.add_argument("--model-left",  default=_config_left,
                        help="Left arm model. Set automatically from MODE=dual; overrides config.")
    parser.add_argument(
        "--prediction-log",
        default="",
        help="Optional CSV path for per-prediction timing/state logging.",
    )
    parser.add_argument(
        "--publish-file",
        default=None,
        help="Optional JSON path for publishing the latest gesture state.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device (default: auto).",
    )
    args = parser.parse_args(argv)

    # --model-right overrides --model so either flag works
    model_right_path = args.model_right if args.model_right else args.model
    model_left_path = args.model_left   # None -> single-arm mode
    dual_arm = model_left_path is not None
    prediction_logger = PredictionCSVLogger(args.prediction_log) if args.prediction_log else None
    global PUBLISH_FILE_PATH
    PUBLISH_FILE_PATH = os.path.abspath(args.publish_file) if args.publish_file else None

    print("[gesture] cwd:", os.getcwd())

    if args.device == "auto":
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
    print('[gesture] device:', device)
    print(
        f"[gesture] runtime preset: {RUNTIME_TUNING_PRESET_NAME} "
        f"(smoothing={SMOOTHING}, min_conf={MIN_CONFIDENCE:.2f}, "
        f"hysteresis={OUTPUT_HYSTERESIS}, softmax_reject={REALTIME_TUNING.softmax_reject_enabled})"
    )

    print("[gesture] right arm model:", model_right_path)
    bundle_right = load_cnn_bundle(model_right_path, device=device)
    use_strict_layout_right = _bundle_uses_strict_layout(bundle_right)

    model_channels = bundle_right.channel_count   # kept for single-arm compat
    index_to_label_right = _canonical_label_map(bundle_right.index_to_label)
    allowed_labels_right, allowed_indices_right = _resolve_allowed_labels(
        index_to_label_right, "right"
    )

    # Left arm bundle (dual-arm mode only)
    bundle_left   = None
    left_channels = 0
    index_to_label_left = {}
    allowed_labels_left = set()
    allowed_indices_left = set()
    use_strict_layout_left = False
    if dual_arm:
        print("[gesture] left arm model:", model_left_path)
        bundle_left   = load_cnn_bundle(model_left_path, device=device)
        left_channels = bundle_left.channel_count
        use_strict_layout_left = _bundle_uses_strict_layout(bundle_left)
        index_to_label_left = _canonical_label_map(bundle_left.index_to_label)
        allowed_labels_left, allowed_indices_left = _resolve_allowed_labels(
            index_to_label_left, "left"
        )
        print(f"[gesture] dual-arm mode | right={RIGHT_ARM_CHANNELS}ch left={left_channels}ch")
    else:
        print("[gesture] single-arm mode")
    if dual_arm and use_strict_layout_right != use_strict_layout_left:
        raise RuntimeError(
            "Dual-arm strict layout mismatch: right/left bundles must both use strict mode or both avoid it."
        )

    use_prototype_classifier = bool(USE_PROTOTYPE_CLASSIFIER)
    if use_prototype_classifier and (not CALIBRATE or not GESTURE_CALIB):
        print(
            "[gesture] prototype classifier disabled: requires CALIBRATE=True and GESTURE_CALIB=True."
        )
        use_prototype_classifier = False
    if use_prototype_classifier:
        print(
            "[gesture] prototype classifier: ON "
            "(calibration centroids + cosine scoring + reject gate)"
        )
    else:
        print("[gesture] prototype classifier: OFF (softmax decoder)")

    prototype_right = None
    prototype_left = None

    bundle_fs_right = _resolve_target_fs_hz_from_bundle(bundle_right)
    bundle_fs_left = _resolve_target_fs_hz_from_bundle(bundle_left) if dual_arm else None
    bundle_target_fs = bundle_fs_right
    if bundle_target_fs is None:
        bundle_target_fs = bundle_fs_left
    if dual_arm and bundle_fs_right is not None and bundle_fs_left is not None:
        if abs(bundle_fs_right - bundle_fs_left) > 1e-6:
            raise RuntimeError(
                f"Right/left bundle target fs mismatch: {bundle_fs_right:.3f} vs "
                f"{bundle_fs_left:.3f} Hz. Re-export bundles with matching preprocessing."
            )

    runtime_target_fs: float | None = (
        float(REALTIME_TARGET_FS_HZ)
        if REALTIME_TARGET_FS_HZ is not None
        else (float(bundle_target_fs) if bundle_target_fs is not None else None)
    )
    use_timestamp_resampling = bool(
        REALTIME_RESAMPLE and runtime_target_fs is not None and runtime_target_fs > 0
    )
    if REALTIME_RESAMPLE and runtime_target_fs is None:
        print(
            "[gesture] realtime resampling requested but no target fs available "
            "(set REALTIME_TARGET_FS_HZ or store target_fs_hz in bundle metadata)."
        )
    elif (
        use_timestamp_resampling
        and runtime_target_fs is not None
        and bundle_target_fs is not None
    ):
        if abs(runtime_target_fs - bundle_target_fs) > 1e-6:
            print(
                "[gesture] WARNING: realtime target fs differs from bundle metadata "
                f"({runtime_target_fs:.3f} vs {bundle_target_fs:.3f} Hz)."
            )

    handler = _StreamingHandler()
    base = TrignoBase(handler)
    handler.DataHandler = DataKernel(base)

    base.Connect_Callback()
    sensors = base.Scan_Callback()
    if not sensors:
        raise RuntimeError("No Trigno sensors found during scan.")

    base.start_trigger = False
    base.stop_trigger = False
    configured = base.ConfigureCollectionOutput()
    if not configured:
        raise RuntimeError("Failed to configure Trigno pipeline.")

    stream_channels = len(base.channel_guids)
    channel_labels = list(getattr(base, "emgChannelNames", []))
    if len(channel_labels) != stream_channels:
        channel_labels = [f"ch{idx}" for idx in range(stream_channels)]

    # For single-arm left mode: auto-detect channel offset so the system works
    # whether only left arm sensors are connected (left_arm_start=0) or all
    # sensors are connected from a prior dual-arm session (left_arm_start=17).
    left_arm_start = 0
    single_arm_channel_indices = None
    right_channel_indices = np.arange(min(RIGHT_ARM_CHANNELS, stream_channels), dtype=int)
    left_channel_indices = np.arange(
        RIGHT_ARM_CHANNELS,
        min(RIGHT_ARM_CHANNELS + left_channels, stream_channels),
        dtype=int,
    )
    strict_dual_layout = dual_arm and use_strict_layout_right and use_strict_layout_left
    if not dual_arm:
        pair_order_fallback = False
        if use_strict_layout_right:
            single_arm_channel_indices = _ordered_indices_from_strict_layout(
                channel_labels,
                bundle_right,
                MODE,
            )
        elif _bundle_uses_type_layout(bundle_right):
            if stream_channels == model_channels:
                single_arm_channel_indices = _ordered_indices_from_type_layout(
                    channel_labels,
                    np.arange(stream_channels, dtype=int),
                    model_channels,
                    bundle_right,
                    MODE,
                )
            else:
                print(
                    f"[gesture:{MODE}] type canonicalization deferred: "
                    f"stream has {stream_channels} channels, model expects {model_channels}."
                )
        if single_arm_channel_indices is None:
            pair_order_fallback = True
            if MODE == "right":
                right_order = _pair_order_for_arm(RIGHT_ARM_PAIR_ORDER, RIGHT_ARM_PAIR_NUMBERS)
                single_arm_channel_indices = _ordered_indices_from_pairs(
                    channel_labels, right_order, model_channels, "right"
                )
            elif MODE == "left":
                left_order = _pair_order_for_arm(LEFT_ARM_PAIR_ORDER, LEFT_ARM_PAIR_NUMBERS)
                single_arm_channel_indices = _ordered_indices_from_pairs(
                    channel_labels, left_order, model_channels, "left"
                )
        if pair_order_fallback and single_arm_channel_indices is not None:
            print(
                f"[gesture] {MODE}-arm ordered indices ({len(single_arm_channel_indices)}): "
                f"{single_arm_channel_indices.tolist()}"
            )
            ordered_pairs = []
            for i in single_arm_channel_indices:
                pair = _parse_pair_number(channel_labels[int(i)])
                if pair is not None:
                    ordered_pairs.append(int(pair))
            print(f"[gesture] {MODE}-arm ordered pairs: {sorted(set(ordered_pairs))}")
    elif strict_dual_layout:
        assert bundle_left is not None
        right_channel_indices = _ordered_indices_from_strict_layout(
            channel_labels,
            bundle_right,
            "right",
        )
        left_channel_indices = _ordered_indices_from_strict_layout(
            channel_labels,
            bundle_left,
            "left",
        )
    if not dual_arm and MODE == "left" and single_arm_channel_indices is None:
        if stream_channels >= RIGHT_ARM_CHANNELS + model_channels:
            left_arm_start = RIGHT_ARM_CHANNELS
            print(
                f"[gesture] all sensors connected; left-arm inference using "
                f"channels {left_arm_start}–{left_arm_start + model_channels - 1}"
            )
        else:
            print(f"[gesture] left-arm sensors only; using channels 0–{model_channels - 1}")

    if dual_arm:
        if strict_dual_layout:
            print("[gesture] dual-arm strict layout active; pairing/scan order is ignored after pair resolution.")
        else:
            expected_total = RIGHT_ARM_CHANNELS + left_channels
            if stream_channels < expected_total:
                raise RuntimeError(
                    f"Dual-arm requires {expected_total} stream channels "
                    f"({RIGHT_ARM_CHANNELS} right + {left_channels} left) "
                    f"but stream only has {stream_channels}. "
                    "Pair right arm sensors first, then left arm sensors, before scanning."
                )
            if stream_channels > expected_total:
                print(
                    f"[gesture] stream has {stream_channels} channels; "
                    f"dual models use {expected_total}. Extra channels may be ignored."
                )
    else:
        required = left_arm_start + model_channels
        if stream_channels < required:
            raise RuntimeError(
                f"Channel count mismatch: need at least {required} stream channels "
                f"(model expects {model_channels} at offset {left_arm_start}), "
                f"but stream has {stream_channels}. Fix sensor pairing/mode before inference."
            )
        if single_arm_channel_indices is None and stream_channels != required:
            print(
                f"Stream has {stream_channels} channels; using slice "
                f"[{left_arm_start}:{left_arm_start + model_channels}] for model input."
            )
    base.TrigBase.Start(handler.streamYTData)

    fs: float | None
    if use_timestamp_resampling:
        assert runtime_target_fs is not None
        fs = float(runtime_target_fs)
    else:
        fs = None
    if REALTIME_FILTER_MODE not in {"scipy_stateful", "libemg_rolling"}:
        raise ValueError(
            f"Unsupported REALTIME_FILTER_MODE={REALTIME_FILTER_MODE!r}. "
            "Use 'scipy_stateful' or 'libemg_rolling'."
        )
    use_stateful_realtime_filter = REALTIME_FILTER_MODE == "scipy_stateful"
    filter_obj = define_filters(fs) if fs is not None else None
    realtime_filter_spec = (
        _define_realtime_stateful_filters(fs)
        if fs is not None and use_stateful_realtime_filter
        else None
    )
    resampler = _RealtimeTimestampResampler(stream_channels, fs) if fs is not None else None
    if use_timestamp_resampling:
        print(f"[gesture] realtime timestamp resampling enabled at {fs:.2f} Hz")
    else:
        print("[gesture] realtime timestamp resampling disabled (legacy index alignment).")
    if use_stateful_realtime_filter:
        print("[gesture] realtime filter mode: scipy_stateful")
    else:
        print(f"[gesture] realtime filter mode: libemg_rolling (warmup={FILTER_WARMUP})")

    poll_sleep = 0.001

    # Per-arm calibration values (single-arm uses only the _right variants)
    neutral_mean       = None   # single-arm compat alias → points to neutral_mean_right
    mvc_scale          = None   # single-arm compat alias → points to mvc_scale_right
    neutral_mean_right = None
    mvc_scale_right    = None
    neutral_mean_left  = None
    mvc_scale_left     = None

    def _ensure_filter_backends():
        nonlocal filter_obj, realtime_filter_spec
        if fs is None:
            return
        if filter_obj is None:
            filter_obj = define_filters(fs)
        if use_stateful_realtime_filter and realtime_filter_spec is None:
            realtime_filter_spec = _define_realtime_stateful_filters(fs)

    def _do_calibration(arm_label, arm_channels, collect_channels, channel_start=0):
        """Run one neutral+MVC calibration sequence for a single arm.
        arm_label:      display name, e.g. 'right' or 'left'
        arm_channels:   slice of the full stream belonging to this arm (int count)
        collect_channels: total channels to collect from stream (None = all)
        Returns: (neutral_mean, mvc_scale) arrays of shape (arm_channels,), or (None, None).
        """
        calib_done = False
        while not calib_done:
            print(f"\nCalibration [{arm_label}]: RELAX {arm_label} arm completely "
                  f"for {CALIB_NEUTRAL_S:.1f}s.")
            n_samples, n_times = _collect_samples(
                handler, CALIB_NEUTRAL_S, stream_channels, collect_channels, poll_sleep
            )
            nonlocal fs, filter_obj
            if fs is None and n_times.size:
                fs = _estimate_fs(n_times)
                if fs is not None:
                    _ensure_filter_backends()
                    print(f"Estimated fs: {fs:.2f} Hz")

            if filter_obj is None:
                print("Warning: calibration skipped (no filter available).")
                return None, None

            if CALIB_MVC_PREP_S > 0:
                print(f"Prepare: SQUEEZE {arm_label.upper()} ARM as hard as possible "
                      f"in {CALIB_MVC_PREP_S:.0f}s...")
                time.sleep(CALIB_MVC_PREP_S)

            print(f"Calibration [{arm_label}]: SQUEEZE {arm_label.upper()} ARM "
                  f"as hard as possible for {CALIB_MVC_S:.1f}s.")
            m_samples, m_times = _collect_samples(
                handler, CALIB_MVC_S, stream_channels, collect_channels, poll_sleep
            )
            if fs is None and m_times.size:
                fs = _estimate_fs(m_times)
                if fs is not None:
                    _ensure_filter_backends()
                    print(f"Estimated fs: {fs:.2f} Hz")

            if not (m_samples.size and n_samples.size):
                calib_done = True
                return None, None

            # Slice to this arm's channels if collecting full stream
            if collect_channels is None:
                n_arr = _slice_channels(n_samples, channel_start, arm_channels)
                m_arr = _slice_channels(m_samples, channel_start, arm_channels)
            else:
                n_arr = _slice_channels(n_samples, 0, arm_channels)
                m_arr = _slice_channels(m_samples, 0, arm_channels)

            neutral_f = apply_filters(filter_obj, n_arr)
            mvc_f     = apply_filters(filter_obj, m_arr)

            neutral_rms = np.sqrt(np.mean(neutral_f ** 2, axis=0))
            mvc_rms     = np.sqrt(np.mean(mvc_f     ** 2, axis=0))
            ratio = np.ones_like(mvc_rms, dtype=float)
            np.divide(mvc_rms, neutral_rms, out=ratio, where=neutral_rms >= 1e-9)
            median_ratio = float(np.median(ratio))
            n_weak       = int(np.sum(ratio < MVC_MIN_RATIO))
            print(
                f"MVC quality [{arm_label}]: {median_ratio:.1f}x median "
                f"({n_weak}/{len(ratio)} channels below {MVC_MIN_RATIO:.0f}x)"
            )

            if median_ratio < MVC_MIN_RATIO:
                print(
                    f"\n*** WARNING: Weak {arm_label} calibration ({median_ratio:.1f}x) ***\n"
                    f"  Squeeze your {arm_label} arm/wrist muscles simultaneously with full force.\n"
                )
                retry = input("Retry calibration? [y/N]: ").strip().lower()
                if retry == "y":
                    continue
                print(
                    f"[{arm_label}] Calibration quality below threshold; "
                    "MVC normalization disabled for this arm to match training policy."
                )
                return None, None

            n_mean  = np.mean(neutral_f, axis=0)
            m_scale = np.percentile(mvc_f, MVC_PERCENTILE, axis=0)
            m_scale = np.where(m_scale < 1e-6, 1.0, m_scale)
            print(f"[{arm_label}] Calibration complete. (quality: {median_ratio:.1f}x)")
            return n_mean, m_scale

        return None, None  # unreachable but keeps linter happy

    def _ratio_from_filtered(neutral_f, mvc_f):
        neutral_rms = np.sqrt(np.mean(neutral_f ** 2, axis=0))
        mvc_rms = np.sqrt(np.mean(mvc_f ** 2, axis=0))
        ratio = np.ones_like(mvc_rms, dtype=float)
        np.divide(mvc_rms, neutral_rms, out=ratio, where=neutral_rms >= 1e-9)
        return ratio

    def _collect_calibration_full(arm_label):
        print(f"\nCalibration [{arm_label}]: RELAX {arm_label} arm completely "
              f"for {CALIB_NEUTRAL_S:.1f}s.")
        n_samples, n_times = _collect_samples(
            handler, CALIB_NEUTRAL_S, stream_channels, None, poll_sleep
        )
        nonlocal fs, filter_obj
        if fs is None and n_times.size:
            fs = _estimate_fs(n_times)
            if fs is not None:
                _ensure_filter_backends()
                print(f"Estimated fs: {fs:.2f} Hz")
        if filter_obj is None:
            return None, None

        if CALIB_MVC_PREP_S > 0:
            print(f"Prepare: SQUEEZE {arm_label.upper()} ARM as hard as possible "
                  f"in {CALIB_MVC_PREP_S:.0f}s...")
            time.sleep(CALIB_MVC_PREP_S)
        print(f"Calibration [{arm_label}]: SQUEEZE {arm_label.upper()} ARM "
              f"as hard as possible for {CALIB_MVC_S:.1f}s.")
        m_samples, m_times = _collect_samples(
            handler, CALIB_MVC_S, stream_channels, None, poll_sleep
        )
        if fs is None and m_times.size:
            fs = _estimate_fs(m_times)
            if fs is not None:
                _ensure_filter_backends()
                print(f"Estimated fs: {fs:.2f} Hz")
        if not (n_samples.size and m_samples.size):
            return None, None
        return apply_filters(filter_obj, n_samples), apply_filters(filter_obj, m_samples)

    def _finalize_calibration_from_indices(arm_label, neutral_f, mvc_f, indices):
        target = len(indices)
        n_arr = _slice_channels_by_indices(neutral_f, indices, target)
        m_arr = _slice_channels_by_indices(mvc_f, indices, target)
        ratio = _ratio_from_filtered(n_arr, m_arr)
        median_ratio = float(np.median(ratio))
        n_weak = int(np.sum(ratio < MVC_MIN_RATIO))
        print(
            f"MVC quality [{arm_label}]: {median_ratio:.1f}x median "
            f"({n_weak}/{len(ratio)} channels below {MVC_MIN_RATIO:.0f}x)"
        )
        if median_ratio < MVC_MIN_RATIO:
            print(
                f"[{arm_label}] Calibration quality below threshold; "
                "MVC normalization disabled for this arm."
            )
            return None, None
        n_mean = np.mean(n_arr, axis=0)
        m_scale = np.percentile(m_arr, MVC_PERCENTILE, axis=0)
        m_scale = np.where(m_scale < 1e-6, 1.0, m_scale)
        print(f"[{arm_label}] Calibration complete. (quality: {median_ratio:.1f}x)")
        return n_mean, m_scale

    def _print_dual_channel_mapping(right_idx, left_idx):
        print(f"[gesture] dual mapping right indices ({len(right_idx)}): {right_idx.tolist()}")
        print(f"[gesture] dual mapping left indices  ({len(left_idx)}): {left_idx.tolist()}")
        if channel_labels:
            right_pairs_vals = []
            for i in right_idx:
                pair = _parse_pair_number(channel_labels[int(i)])
                if pair is not None:
                    right_pairs_vals.append(int(pair))
            right_pairs = sorted(set(right_pairs_vals))

            left_pairs_vals = []
            for i in left_idx:
                pair = _parse_pair_number(channel_labels[int(i)])
                if pair is not None:
                    left_pairs_vals.append(int(pair))
            left_pairs = sorted(set(left_pairs_vals))
            print(f"[gesture] dual mapping right pairs: {right_pairs}")
            print(f"[gesture] dual mapping left pairs:  {left_pairs}")

    if CALIBRATE:
        if dual_arm:
            if strict_dual_layout:
                right_neutral_f, right_mvc_f = _collect_calibration_full("right")
                left_neutral_f, left_mvc_f = _collect_calibration_full("left")
                if (
                    right_neutral_f is None or right_mvc_f is None
                    or left_neutral_f is None or left_mvc_f is None
                ):
                    print("[gesture] dual-arm calibration unavailable; MVC normalization disabled.")
                    neutral_mean_right = mvc_scale_right = None
                    neutral_mean_left = mvc_scale_left = None
                else:
                    _print_dual_channel_mapping(right_channel_indices, left_channel_indices)
                    neutral_mean_right, mvc_scale_right = _finalize_calibration_from_indices(
                        "right", right_neutral_f, right_mvc_f, right_channel_indices
                    )
                    neutral_mean_left, mvc_scale_left = _finalize_calibration_from_indices(
                        "left", left_neutral_f, left_mvc_f, left_channel_indices
                    )
            elif AUTO_DUAL_ARM_CHANNEL_MAPPING:
                right_neutral_f, right_mvc_f = _collect_calibration_full("right")
                left_neutral_f, left_mvc_f = _collect_calibration_full("left")
                if (
                    right_neutral_f is None or right_mvc_f is None
                    or left_neutral_f is None or left_mvc_f is None
                ):
                    print("[gesture] dual-arm calibration unavailable; MVC normalization disabled.")
                    neutral_mean_right = mvc_scale_right = None
                    neutral_mean_left = mvc_scale_left = None
                else:
                    right_ratio_full = _ratio_from_filtered(right_neutral_f, right_mvc_f)
                    left_ratio_full = _ratio_from_filtered(left_neutral_f, left_mvc_f)
                    right_channel_indices, left_channel_indices = _derive_dual_indices_from_pairs(
                        channel_labels,
                        RIGHT_ARM_CHANNELS,
                        left_channels,
                        right_ratio_full,
                        left_ratio_full,
                    )
                    right_channel_indices = _canonicalize_indices_by_type(
                        channel_labels, right_channel_indices, bundle_right, "right"
                    )
                    left_channel_indices = _canonicalize_indices_by_type(
                        channel_labels, left_channel_indices, bundle_left, "left"
                    )
                    _print_dual_channel_mapping(right_channel_indices, left_channel_indices)
                    neutral_mean_right, mvc_scale_right = _finalize_calibration_from_indices(
                        "right", right_neutral_f, right_mvc_f, right_channel_indices
                    )
                    neutral_mean_left, mvc_scale_left = _finalize_calibration_from_indices(
                        "left", left_neutral_f, left_mvc_f, left_channel_indices
                    )
            else:
                # Legacy dual-arm contiguous split (requires right channels first).
                original_right_indices = right_channel_indices.copy()
                original_left_indices = left_channel_indices.copy()
                neutral_mean_right, mvc_scale_right = _do_calibration(
                    "right", RIGHT_ARM_CHANNELS, collect_channels=None, channel_start=0
                )
                neutral_mean_left, mvc_scale_left = _do_calibration(
                    "left", left_channels, collect_channels=None, channel_start=RIGHT_ARM_CHANNELS
                )
                right_channel_indices = _canonicalize_indices_by_type(
                    channel_labels, right_channel_indices, bundle_right, "right"
                )
                left_channel_indices = _canonicalize_indices_by_type(
                    channel_labels, left_channel_indices, bundle_left, "left"
                )
                neutral_mean_right, mvc_scale_right = _reorder_calibration_vectors(
                    neutral_mean_right,
                    mvc_scale_right,
                    original_right_indices,
                    right_channel_indices,
                )
                neutral_mean_left, mvc_scale_left = _reorder_calibration_vectors(
                    neutral_mean_left,
                    mvc_scale_left,
                    original_left_indices,
                    left_channel_indices,
                )
                _print_dual_channel_mapping(right_channel_indices, left_channel_indices)
        else:
            if single_arm_channel_indices is not None:
                neutral_f, mvc_f = _collect_calibration_full(MODE)
                if neutral_f is None or mvc_f is None:
                    neutral_mean_right = mvc_scale_right = None
                else:
                    neutral_mean_right, mvc_scale_right = _finalize_calibration_from_indices(
                        MODE, neutral_f, mvc_f, single_arm_channel_indices
                    )
            else:
                # Single-arm: if left_arm_start > 0 (all sensors connected in left mode),
                # collect all channels then slice; otherwise collect only model_channels.
                calib_collect = None if left_arm_start > 0 else model_channels
                neutral_mean_right, mvc_scale_right = _do_calibration(
                    MODE, model_channels, collect_channels=calib_collect, channel_start=left_arm_start
                )

        # Single-arm compat aliases
        neutral_mean = neutral_mean_right
        mvc_scale    = mvc_scale_right

    if dual_arm and GESTURE_CALIB and not use_prototype_classifier:
        print("[gesture] dual-arm mode: skipping per-gesture fine-tuning calibration.")
    if (
        CALIBRATE
        and GESTURE_CALIB
        and filter_obj is not None
        and ((not dual_arm) or use_prototype_classifier)
    ):
        print("\n=== Per-gesture calibration ===")
        print(f"Perform each gesture for {GESTURE_CALIB_S:.0f}s when prompted.\n")

        label_to_idx_r = {label: idx for idx, label in index_to_label_right.items()}
        label_to_idx_l = {label: idx for idx, label in index_to_label_left.items()} if dual_arm else {}

        wins_r, labs_r = [], []
        wins_l, labs_l = [], []

        gestures_for_calibration = []
        for gesture in GESTURE_LABELS:
            if gesture in allowed_labels_right:
                gestures_for_calibration.append(gesture)
                continue
            if dual_arm and gesture in allowed_labels_left:
                gestures_for_calibration.append(gesture)

        if not gestures_for_calibration:
            print("No gesture labels available for calibration after filter settings.")
        for gesture in gestures_for_calibration:
            if gesture not in label_to_idx_r and (not dual_arm or gesture not in label_to_idx_l):
                continue
            instr = GESTURE_INSTRUCTIONS.get(gesture, gesture)
            if GESTURE_CALIB_PREP_S > 0:
                print(f"Next: {instr}  —  get ready ({GESTURE_CALIB_PREP_S:.0f}s)...")
                time.sleep(GESTURE_CALIB_PREP_S)
            print(f"GO: {instr}")

            gesture_collect_channels = None
            if not dual_arm and left_arm_start == 0 and single_arm_channel_indices is None:
                # Single-arm right mode: keep calibration collection aligned with model input width.
                gesture_collect_channels = model_channels
            samples, _ = _collect_samples(
                handler, GESTURE_CALIB_S, stream_channels, gesture_collect_channels, poll_sleep
            )
            if samples.size == 0:
                print(f"  Warning: no samples for '{gesture}', skipping.")
                continue

            # Right arm
            if dual_arm:
                r_raw = _slice_channels_by_indices(samples, right_channel_indices, RIGHT_ARM_CHANNELS)
            elif single_arm_channel_indices is not None:
                r_raw = _slice_channels_by_indices(samples, single_arm_channel_indices, model_channels)
            else:
                r_raw = _slice_channels(samples, left_arm_start, model_channels)
            r_filt = apply_filters(filter_obj, r_raw)
            if neutral_mean_right is not None and mvc_scale_right is not None:
                neutral_mean_right, mvc_scale_right = _align_calibration_vectors(
                    neutral_mean_right, mvc_scale_right, r_filt.shape[1], "right"
                )
                r_filt = (r_filt - neutral_mean_right) / mvc_scale_right
            w_r = _make_windows(r_filt)
            if len(w_r) and gesture in label_to_idx_r and gesture in allowed_labels_right:
                wins_r.append(w_r)
                labs_r.extend([label_to_idx_r[gesture]] * len(w_r))
                print(f"  Right: {len(w_r)} windows collected")

            # Left arm (dual-arm only)
            if dual_arm and gesture in label_to_idx_l and gesture in allowed_labels_left:
                l_raw = _slice_channels_by_indices(samples, left_channel_indices, left_channels)
                l_filt = apply_filters(filter_obj, l_raw)
                if neutral_mean_left is not None and mvc_scale_left is not None:
                    neutral_mean_left, mvc_scale_left = _align_calibration_vectors(
                        neutral_mean_left, mvc_scale_left, l_filt.shape[1], "left"
                    )
                    l_filt = (l_filt - neutral_mean_left) / mvc_scale_left
                w_l = _make_windows(l_filt)
                if len(w_l):
                    wins_l.append(w_l)
                    labs_l.extend([label_to_idx_l[gesture]] * len(w_l))
                    print(f"  Left:  {len(w_l)} windows collected")

        if wins_r:
            X_r = np.concatenate(wins_r, axis=0)
            y_r = np.array(labs_r, dtype=np.int64)
            if use_prototype_classifier:
                print(f"\nBuilding right-arm prototypes ({len(y_r)} windows)...")
                X_r_std = bundle_right.standardize(X_r)
                E_r = bundle_right.embed(X_r_std, l2_normalize=PROTOTYPE_L2_NORMALIZE)
                prototype_right = PrototypeClassifier.fit(
                    E_r,
                    y_r,
                    temperature=PROTOTYPE_TEMPERATURE,
                    l2_normalize=PROTOTYPE_L2_NORMALIZE,
                )
                print(f"  Right-arm prototypes ready: {len(prototype_right.class_indices)} classes.")
            else:
                print(f"\nFine-tuning right arm model ({len(y_r)} windows)...")
                quick_finetune(bundle_right, X_r, y_r, device=device)
                print("  Right arm model updated.")

        if dual_arm and wins_l:
            assert bundle_left is not None
            X_l = np.concatenate(wins_l, axis=0)
            y_l = np.array(labs_l, dtype=np.int64)
            if use_prototype_classifier:
                print(f"Building left-arm prototypes ({len(y_l)} windows)...")
                X_l_std = bundle_left.standardize(X_l)
                E_l = bundle_left.embed(X_l_std, l2_normalize=PROTOTYPE_L2_NORMALIZE)
                prototype_left = PrototypeClassifier.fit(
                    E_l,
                    y_l,
                    temperature=PROTOTYPE_TEMPERATURE,
                    l2_normalize=PROTOTYPE_L2_NORMALIZE,
                )
                print(f"  Left-arm prototypes ready: {len(prototype_left.class_indices)} classes.")
            else:
                print(f"Fine-tuning left arm model ({len(y_l)} windows)...")
                quick_finetune(bundle_left, X_l, y_l, device=device)
                print("  Left arm model updated.")

        if use_prototype_classifier and prototype_right is None:
            print(
                "[gesture] WARNING: prototype calibration produced no right-arm centroids; "
                "falling back to softmax decoder."
            )
            use_prototype_classifier = False

        print("\n=== Gesture calibration complete. Starting inference... ===\n")

    if use_prototype_classifier and prototype_right is None:
        print(
            "[gesture] WARNING: prototype classifier enabled but no centroids were built; "
            "using softmax decoder."
        )
        use_prototype_classifier = False

    pending_right = 0
    pred_history_right = deque(maxlen=max(1, SMOOTHING))
    pending_left = 0
    pred_history_left = deque(maxlen=max(1, SMOOTHING))
    if use_stateful_realtime_filter:
        filtered_buffer_right = deque(maxlen=WINDOW_SIZE)
        filtered_buffer_left = deque(maxlen=WINDOW_SIZE)
        filter_state_right = None
        filter_state_left = None
    else:
        buffer_len = WINDOW_SIZE + FILTER_WARMUP

        # Right arm (always present)
        raw_buffer_right = deque(maxlen=buffer_len)
        warmup_right = FILTER_WARMUP == 0

        # Left arm (dual-arm mode only)
        raw_buffer_left = deque(maxlen=buffer_len)
        warmup_left = FILTER_WARMUP == 0

    last_output  = None
    last_msg_len = 0
    output_hysteresis_right = _OutputHysteresis()
    output_hysteresis_left = _OutputHysteresis()
    current_right_label = LOW_CONFIDENCE_LABEL
    current_right_conf = 0.0
    current_left_label = LOW_CONFIDENCE_LABEL
    current_left_conf = 0.0
    right_trace_state = ArmPredictionTrace()
    left_trace_state = ArmPredictionTrace()
    # === LATENCY MEASURE START ===
    # Comment out this whole block (and other LATENCY blocks below) after measuring.
    # Measures end-to-end latency from "data available in this process" to prediction time.
    # latency_enabled = True
    # latency_print_every = 20
    # latency_max_preds = 200
    # latency_ms = []
    # proc_ms = []
    # preds = 0
    # time_buffer = deque(maxlen=WINDOW_SIZE + FILTER_WARMUP)
    # # === LATENCY MEASURE END ===

    try:
        while True:
            out = handler.DataHandler.GetYTData()
            if out is None:
                time.sleep(poll_sleep)
                continue

            channel_times, channel_values = _parse_yt_frame(out)

            if len(channel_values) < stream_channels:
                continue

            # Capture when this batch of samples became available to this process.
            batch_wall_time = time.time()

            if use_timestamp_resampling:
                assert resampler is not None
                batch_arr, _ = resampler.push(channel_times, channel_values)
                if batch_arr.size == 0:
                    continue
            else:
                sample_count = min(len(c) for c in channel_values)
                if sample_count == 0:
                    continue

                if fs is None and channel_times:
                    for times in channel_times:
                        if not times:
                            continue
                        fs = _estimate_fs(times)
                        if fs is not None:
                            _ensure_filter_backends()
                            print(f"Estimated fs: {fs:.2f} Hz")
                            break

                # --- legacy collect path: index-aligned rows ---
                sample_batch = []
                for idx in range(sample_count):
                    sample = [channel_values[ch][idx] for ch in range(stream_channels)]
                    sample_batch.append(sample)

                if not sample_batch:
                    continue

                batch_arr = np.asarray(sample_batch, dtype=float)  # (N, stream_channels)

            # --- split channels by arm ---
            if dual_arm:
                batch_right = _slice_channels_by_indices(
                    batch_arr, right_channel_indices, RIGHT_ARM_CHANNELS
                )
                batch_left = _slice_channels_by_indices(
                    batch_arr, left_channel_indices, left_channels
                )
            elif single_arm_channel_indices is not None:
                batch_right = _slice_channels_by_indices(
                    batch_arr, single_arm_channel_indices, model_channels
                )
            else:
                batch_right = _slice_channels(batch_arr, left_arm_start, model_channels)

            if use_stateful_realtime_filter:
                if realtime_filter_spec is None:
                    continue
                if filter_state_right is None:
                    filter_state_right = _make_realtime_filter_state(
                        realtime_filter_spec,
                        batch_right.shape[1],
                    )
                filtered_batch_right, filter_state_right = _apply_filters_stateful(
                    realtime_filter_spec,
                    batch_right,
                    filter_state_right,
                )
                for sample in filtered_batch_right:
                    filtered_buffer_right.append(sample)
                pending_right += int(filtered_batch_right.shape[0])

                if dual_arm:
                    if filter_state_left is None:
                        filter_state_left = _make_realtime_filter_state(
                            realtime_filter_spec,
                            batch_left.shape[1],
                        )
                    filtered_batch_left, filter_state_left = _apply_filters_stateful(
                        realtime_filter_spec,
                        batch_left,
                        filter_state_left,
                    )
                    for sample in filtered_batch_left:
                        filtered_buffer_left.append(sample)
                    pending_left += int(filtered_batch_left.shape[0])
            else:
                # Rolling full-buffer libEMG refiltering. Higher latency, but
                # kept as a comparison path.
                for sample in batch_right:
                    raw_buffer_right.append(sample)
                    pending_right += 1
                if not warmup_right and len(raw_buffer_right) >= buffer_len:
                    warmup_right = True
                    pending_right = WINDOW_STEP

                if dual_arm:
                    for sample in batch_left:
                        raw_buffer_left.append(sample)
                        pending_left += 1
                    if not warmup_left and len(raw_buffer_left) >= buffer_len:
                        warmup_left = True
                        pending_left = WINDOW_STEP

                if filter_obj is None:
                    pending_right = min(pending_right, WINDOW_STEP)
                    if dual_arm:
                        pending_left = min(pending_left, WINDOW_STEP)
                    continue

            # --- inference: right arm ---
            inference_ran = False
            label_right = LOW_CONFIDENCE_LABEL
            conf_right  = 0.0
            if use_stateful_realtime_filter:
                right_ready = pending_right >= WINDOW_STEP and len(filtered_buffer_right) == WINDOW_SIZE
            else:
                right_ready = warmup_right and pending_right >= WINDOW_STEP and len(raw_buffer_right) >= WINDOW_SIZE

            if right_ready:
                if use_stateful_realtime_filter:
                    filtered = np.asarray(filtered_buffer_right, dtype=float)
                else:
                    raw = np.asarray(raw_buffer_right, dtype=float)
                    filtered_full = apply_filters(filter_obj, raw)
                    filtered = filtered_full[-WINDOW_SIZE:]
                if neutral_mean_right is not None and mvc_scale_right is not None:
                    neutral_mean_right, mvc_scale_right = _align_calibration_vectors(
                        neutral_mean_right, mvc_scale_right, filtered.shape[1], "right"
                    )
                    filtered = (filtered - neutral_mean_right) / mvc_scale_right

                window_raw = filtered.T[np.newaxis, :, :].astype(np.float32)
                window = bundle_right.standardize(window_raw)
                if use_prototype_classifier and prototype_right is not None:
                    emb_right = bundle_right.embed(
                        window, l2_normalize=PROTOTYPE_L2_NORMALIZE
                    )[0]
                    probs = prototype_right.predict_proba(
                        emb_right, num_classes=len(index_to_label_right)
                    )
                else:
                    probs = bundle_right.predict_proba(window)[0]
                probs  = _restrict_probs(probs, allowed_indices_right)

                pred_history_right.append(probs)
                if SMOOTHING > 1:
                    probs = np.mean(np.stack(pred_history_right, axis=0), axis=0)

                ranking_right, label_right, conf_right, pred_idx_right, gate_reason_right = _decode_prediction(
                    probs, index_to_label_right
                )
                if label_right != LOW_CONFIDENCE_LABEL:
                    if use_prototype_classifier and prototype_right is not None:
                        label_right, conf_right, pred_idx_right, reject_reason = _apply_prototype_reject_gate(
                            label_right, conf_right, pred_idx_right, ranking_right
                        )
                    else:
                        label_right, conf_right, pred_idx_right, reject_reason = _apply_softmax_reject_gate(
                            label_right, conf_right, pred_idx_right, ranking_right
                        )
                    if reject_reason:
                        gate_reason_right = reject_reason
                current_right_label, current_right_conf = output_hysteresis_right.update(
                    label_right, conf_right
                )
                right_trace_state = ArmPredictionTrace(
                    raw_top_label=ranking_right.top_label,
                    raw_top_confidence=ranking_right.top_confidence,
                    second_label=ranking_right.second_label,
                    second_confidence=ranking_right.second_confidence,
                    margin=ranking_right.margin,
                    gate_label=label_right,
                    gate_confidence=conf_right,
                    gate_reason=gate_reason_right,
                    hysteresis_label=current_right_label,
                    hysteresis_confidence=current_right_conf,
                )

                inference_ran = True
                # We intentionally keep only the newest inference point and
                # drop backlog to avoid re-scoring the same latest window.
                pending_right = 0

            # --- inference: left arm (dual-arm mode only) ---
            label_left = LOW_CONFIDENCE_LABEL
            conf_left  = 0.0
            if dual_arm:
                assert bundle_left is not None
                if use_stateful_realtime_filter:
                    left_ready = pending_left >= WINDOW_STEP and len(filtered_buffer_left) == WINDOW_SIZE
                else:
                    left_ready = warmup_left and pending_left >= WINDOW_STEP and len(raw_buffer_left) >= WINDOW_SIZE

                if left_ready:
                    if use_stateful_realtime_filter:
                        filtered = np.asarray(filtered_buffer_left, dtype=float)
                    else:
                        raw = np.asarray(raw_buffer_left, dtype=float)
                        filtered_full = apply_filters(filter_obj, raw)
                        filtered = filtered_full[-WINDOW_SIZE:]
                    if neutral_mean_left is not None and mvc_scale_left is not None:
                        neutral_mean_left, mvc_scale_left = _align_calibration_vectors(
                            neutral_mean_left, mvc_scale_left, filtered.shape[1], "left"
                        )
                        filtered = (filtered - neutral_mean_left) / mvc_scale_left

                    window = filtered.T[np.newaxis, :, :].astype(np.float32)
                    window = bundle_left.standardize(window)
                    if use_prototype_classifier and prototype_left is not None:
                        emb_left = bundle_left.embed(
                            window, l2_normalize=PROTOTYPE_L2_NORMALIZE
                        )[0]
                        probs = prototype_left.predict_proba(
                            emb_left, num_classes=len(index_to_label_left)
                        )
                    else:
                        probs = bundle_left.predict_proba(window)[0]
                    probs  = _restrict_probs(probs, allowed_indices_left)

                    pred_history_left.append(probs)
                    if SMOOTHING > 1:
                        probs = np.mean(np.stack(pred_history_left, axis=0), axis=0)

                    ranking_left, label_left, conf_left, pred_idx_left, gate_reason_left = _decode_prediction(
                        probs, index_to_label_left
                    )
                    if label_left != LOW_CONFIDENCE_LABEL:
                        if use_prototype_classifier and prototype_left is not None:
                            label_left, conf_left, pred_idx_left, reject_reason = _apply_prototype_reject_gate(
                                label_left, conf_left, pred_idx_left, ranking_left
                            )
                        else:
                            label_left, conf_left, pred_idx_left, reject_reason = _apply_softmax_reject_gate(
                                label_left, conf_left, pred_idx_left, ranking_left
                            )
                        if reject_reason:
                            gate_reason_left = reject_reason
                    current_left_label, current_left_conf = output_hysteresis_left.update(
                        label_left, conf_left
                    )
                    left_trace_state = ArmPredictionTrace(
                        raw_top_label=ranking_left.top_label,
                        raw_top_confidence=ranking_left.top_confidence,
                        second_label=ranking_left.second_label,
                        second_confidence=ranking_left.second_confidence,
                        margin=ranking_left.margin,
                        gate_label=label_left,
                        gate_confidence=conf_left,
                        gate_reason=gate_reason_left,
                        hysteresis_label=current_left_label,
                        hysteresis_confidence=current_left_conf,
                    )

                    inference_ran = True
                    # Same backlog policy as right arm: latest window only.
                    pending_left = 0

            # --- fusion and output ---
            if inference_ran:
                if dual_arm:
                    label, confidence = fuse_predictions(
                        current_right_label,
                        current_right_conf,
                        current_left_label,
                        current_left_conf,
                    )
                else:
                    label, confidence = current_right_label, current_right_conf

                combined_state = ArmGestureState(label, confidence)
                prediction_wall_time = time.time()
                published_output = resolve_published_gesture_output(
                    "dual" if dual_arm else MODE,
                    ArmGestureState(current_right_label, current_right_conf),
                    ArmGestureState(
                        current_left_label if dual_arm else LOW_CONFIDENCE_LABEL,
                        current_left_conf if dual_arm else 0.0,
                    ),
                    combined_state,
                )

                state = set_latest_dual_state(
                    mode="dual" if dual_arm else MODE,
                    right_label=current_right_label,
                    right_confidence=current_right_conf,
                    left_label=current_left_label if dual_arm else LOW_CONFIDENCE_LABEL,
                    left_confidence=current_left_conf if dual_arm else 0.0,
                    right_trace=right_trace_state,
                    left_trace=left_trace_state if dual_arm else ArmPredictionTrace(),
                    combined_label=label,
                    combined_confidence=confidence,
                    published=published_output,
                    window_end_ts=batch_wall_time,
                    prediction_ts=prediction_wall_time,
                )
                if prediction_logger is not None:
                    prediction_logger.write_state(state)

                output_signature = published_gesture_signature(published_output)
                is_neutral_only = all(
                    gesture.label == LOW_CONFIDENCE_LABEL
                    for gesture in published_output.gestures
                )
                if not published_output.gestures:
                    is_neutral_only = True
                if (not is_neutral_only) or (last_output is not None):
                    if output_signature != last_output:
                        msg = format_published_gesture_output(published_output)
                        if len(msg) < last_msg_len:
                            msg = msg.ljust(last_msg_len)
                        else:
                            last_msg_len = len(msg)
                        print(msg, end="\r", flush=True)
                        last_output = output_signature
    except KeyboardInterrupt:
        print("\nStopping stream...")
    finally:
        base.Stop_Callback()
        if prediction_logger is not None:
            prediction_logger.close()
        # === LATENCY MEASURE START ===
        # if latency_enabled and latency_ms:
        #     lat = np.asarray(latency_ms, dtype=float)
        #     proc = np.asarray(proc_ms, dtype=float) if proc_ms else None
        #     print("\nLatency summary (ms):")
        #     print(
        #         f"count={lat.size} "
        #         f"avg={lat.mean():.1f} "
        #         f"median={np.median(lat):.1f} "
        #         f"p90={np.percentile(lat, 90):.1f} "
        #         f"p95={np.percentile(lat, 95):.1f} "
        #         f"min={lat.min():.1f} "
        #         f"max={lat.max():.1f}"
        #     )
        #     if proc is not None and proc.size:
        #         print(
        #             "Processing-only summary (ms): "
        #             f"avg={proc.mean():.1f} "
        #             f"median={np.median(proc):.1f} "
        #             f"p90={np.percentile(proc, 90):.1f} "
        #             f"p95={np.percentile(proc, 95):.1f} "
        #             f"min={proc.min():.1f} "
        #             f"max={proc.max():.1f}"
        #         )
        # === LATENCY MEASURE END ===


if __name__ == "__main__":
    main()
