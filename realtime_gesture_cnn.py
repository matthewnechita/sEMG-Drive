import argparse
import csv
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
from emg.gesture_model_cnn import load_gesture_bundle
from emg.runtime_tuning import REALTIME_TUNING, RUNTIME_TUNING_NAME
from emg.strict_layout import (
    resolve_strict_channel_indices,
    strict_channel_count_for_arm,
)

# -- Config -------------------------------------------------------------------
WINDOW_SIZE = 200
WINDOW_STEP = 100
# Keep this aligned with your training data preprocessing.
# Set to None to require the bundle to provide target_fs_hz metadata.
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
class PredictionRanking:
    top_idx: int = -1
    top_label: str = LOW_CONFIDENCE_LABEL
    top_confidence: float = 0.0
    margin: float = 0.0


@dataclass(frozen=True)
class PublishedGesture:
    arm: str = "dual"
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
    mode: str = "dual"
    right: ArmGestureState = field(default_factory=ArmGestureState)
    left: ArmGestureState = field(default_factory=ArmGestureState)
    combined: ArmGestureState = field(default_factory=ArmGestureState)
    published: PublishedGestureOutput = field(default_factory=PublishedGestureOutput)
    prediction_seq: int = 0
    window_end_ts: float = 0.0
    prediction_ts: float = 0.0
    timestamp: float = 0.0


LATEST_LOCK = threading.Lock()
LATEST_STATE = DualGestureState()
LATEST_TIMESTAMP = 0.0
LATEST_PREDICTION_SEQ = 0


class PredictionCSVLogger:
    def __init__(self, path: str):
        self.path = str(path)
        out_path = Path(self.path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = out_path.open("w", newline="", encoding="utf-8")
        # One row per published prediction keeps the latency join keyed to what
        # CARLA actually consumed rather than every intermediate buffer update.
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
            "right_label",
            "right_conf",
            "left_label",
            "left_conf",
            "published_labels",
        ]
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
        self._writer.writeheader()

    def write_state(self, state: DualGestureState) -> None:
        published = state.published
        row = {
            "runtime_preset": RUNTIME_TUNING_NAME,
            "prediction_seq": int(getattr(published, "prediction_seq", 0)),
            "window_end_ts": float(getattr(published, "window_end_ts", 0.0)),
            "prediction_ts": float(getattr(published, "prediction_ts", 0.0)),
            "publish_ts": float(getattr(published, "publish_ts", state.timestamp)),
            "mode": str(state.mode),
            "published_mode": str(getattr(published, "mode", "")),
            "pred_label": str(state.combined.label),
            "pred_conf": float(state.combined.confidence),
            "right_label": str(state.right.label),
            "right_conf": float(state.right.confidence),
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

# -- Dual-arm config ----------------------------------------------------------
# Strict layout uses fixed pair identities; pairing/scan order may vary.
RIGHT_ARM_CHANNELS = strict_channel_count_for_arm("right")
# Left arm uses 5 sensors -> 16 channels (one fewer sensor than right arm)

# Optional runtime gesture filtering (code-only; no CLI flags).
# Example (3-class mode):
# INCLUDED_GESTURES = {"neutral", "left_turn", "right_turn"}
INCLUDED_GESTURES: set[str] | None = {"neutral", "left_turn", "right_turn", "horn"}

GESTURE_LABELS = ["neutral", "left_turn", "right_turn", "signal_left", "signal_right", "horn"]

GESTURE_INSTRUCTIONS = {
    "neutral":      "relax completely (rest position)",
    "left_turn":    "perform LEFT TURN gesture",
    "right_turn":   "perform RIGHT TURN gesture",
    "signal_left":  "perform SIGNAL LEFT gesture",
    "signal_right": "perform SIGNAL RIGHT gesture",
    "horn":         "perform HORN gesture",
}
# -- Inference bundles --------------------------------------------------------
# Dual-arm strict realtime uses both checked-in per-subject bundles by default.
MODEL_RIGHT = os.path.join(
    BASE_DIR,
    "models",
    "strict",
    "per_subject",
    "right",
    "Matthewv6_4_gestures.pt",
)

MODEL_LEFT = os.path.join(
    BASE_DIR,
    "models",
    "strict",
    "per_subject",
    "left",
    "Matthewv6_4_gestures.pt",
)
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
    - Notch @ 60 Hz  (bandwidth 3) - power line fundamental
    - Notch @ 120 Hz (bandwidth 3) - 2nd power line harmonic
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


def _pair_time_value(sample):
    if hasattr(sample, "Item1"):
        return float(sample.Item1), float(sample.Item2)
    return float(sample[0]), float(sample[1])


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

        # Only interpolate over the time span that every channel can currently support.
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
def set_latest_dual_state(
    *,
    right_label: str,
    right_confidence: float,
    left_label: str = LOW_CONFIDENCE_LABEL,
    left_confidence: float = 0.0,
    combined_label: str | None = None,
    combined_confidence: float | None = None,
    published: PublishedGestureOutput | None = None,
    window_end_ts: float = 0.0,
    prediction_ts: float = 0.0,
) -> DualGestureState:
    global LATEST_STATE, LATEST_TIMESTAMP, LATEST_PREDICTION_SEQ

    right_state = ArmGestureState(str(right_label), float(right_confidence))
    left_state = ArmGestureState(str(left_label), float(left_confidence))
    if combined_label is None:
        combined_label = LOW_CONFIDENCE_LABEL
    if combined_confidence is None:
        combined_confidence = 0.0
    combined_state = ArmGestureState(str(combined_label), float(combined_confidence))
    LATEST_PREDICTION_SEQ += 1
    prediction_seq = int(LATEST_PREDICTION_SEQ)
    timestamp = time.time()
    if published is None:
        published = PublishedGestureOutput(
            mode="single",
            gestures=(PublishedGesture("dual", combined_state.label, combined_state.confidence),),
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
        mode="dual",
        right=right_state,
        left=left_state,
        combined=combined_state,
        published=published,
        prediction_seq=prediction_seq,
        window_end_ts=float(window_end_ts),
        prediction_ts=float(prediction_ts),
        timestamp=timestamp,
    )
    with LATEST_LOCK:
        LATEST_STATE = state
        LATEST_TIMESTAMP = timestamp
    return state


def get_latest_dual_state() -> tuple[DualGestureState, float]:
    with LATEST_LOCK:
        state = LATEST_STATE
        ts = LATEST_TIMESTAMP if LATEST_TIMESTAMP else state.timestamp
        age = (time.time() - ts) if ts else float('inf')
        return state, age


def get_latest_published_gestures() -> tuple[PublishedGestureOutput, float]:
    state, age = get_latest_dual_state()
    return state.published, age


def resolve_published_gesture_output(
    right: ArmGestureState,
    left: ArmGestureState,
    combined: ArmGestureState,
) -> PublishedGestureOutput:
    # Publish one fused label when both arms agree; otherwise expose the split
    # arm view so downstream logging can still see the disagreement.
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


def fuse_predictions(label_r, conf_r, label_l, conf_l):
    """Combine right and left arm predictions into one fused label.

    If both arms agree on the same gesture, the prediction is reinforced and
    a lower confidence threshold is used to emit it. If they disagree, each
    arm is evaluated independently against the per-arm confidence threshold.

    Returns: (fused_label, fused_confidence)
    """
    label_r = str(label_r)
    label_l = str(label_l)
    neutral = str(LOW_CONFIDENCE_LABEL)

    if label_r == label_l:
        # Both arms agree - strengthen the prediction.
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
    second_conf = float(probs[int(order[1])]) if probs.size > 1 else 0.0

    return PredictionRanking(
        top_idx=top_idx,
        top_label=str(top_label),
        top_confidence=top_conf,
        margin=float(top_conf - second_conf),
    )


def _decode_prediction(probs, index_to_label):
    """Decode probs with a single confidence gate."""
    ranking = _rank_prediction_probs(probs, index_to_label)
    pred_label = str(ranking.top_label)
    pred_conf = float(ranking.top_confidence)

    if pred_conf < float(MIN_CONFIDENCE):
        pred_label = LOW_CONFIDENCE_LABEL

    return ranking, pred_label, pred_conf


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


def _apply_softmax_reject_gate(pred_label, pred_conf, ranking):
    """Reject ambiguous softmax decisions to neutral."""
    if not REALTIME_TUNING.softmax_reject_enabled:
        return pred_label, pred_conf

    if (
        float(ranking.top_confidence) < float(REALTIME_TUNING.softmax_reject_min_confidence)
        or float(ranking.margin) < float(REALTIME_TUNING.softmax_reject_min_margin)
    ):
        return LOW_CONFIDENCE_LABEL, float(ranking.top_confidence)
    return pred_label, pred_conf


def _collect_samples(handler, duration_s, stream_channels, poll_sleep):
    samples = []
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
            samples.append(sample)

    if not samples:
        return np.empty((0, stream_channels), dtype=float)
    return np.asarray(samples, dtype=float)


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


def _slice_channels_by_indices(batch_arr, indices, target_channels):
    """Select arbitrary channel indices and keep output width fixed."""
    arr = np.asarray(batch_arr, dtype=float)
    idx = np.asarray(indices, dtype=int).reshape(-1)
    idx = idx[(idx >= 0) & (idx < arr.shape[1])]
    out = arr[:, idx] if idx.size else np.empty((arr.shape[0], 0), dtype=float)
    # Padding keeps the per-arm tensor width stable even if the live stream is
    # temporarily missing channels, which lets the strict model fail closed downstream.
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


def main(argv=None):
    parser = argparse.ArgumentParser(description="Real-time dual-arm CNN gesture inference.")
    parser.add_argument(
        "--model-right",
        default=MODEL_RIGHT,
        help="Right-arm model bundle.",
    )
    parser.add_argument(
        "--model-left",
        default=MODEL_LEFT,
        help="Left-arm model bundle.",
    )
    parser.add_argument(
        "--prediction-log",
        default="",
        help="Optional CSV path for per-prediction timing/state logging.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device (default: auto).",
    )
    args = parser.parse_args(argv)

    model_right_path = args.model_right
    model_left_path = args.model_left
    prediction_logger = PredictionCSVLogger(args.prediction_log) if args.prediction_log else None

    print("[gesture] cwd:", os.getcwd())

    if args.device == "auto":
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
    print('[gesture] device:', device)
    print(
        f"[gesture] runtime tuning: {RUNTIME_TUNING_NAME} "
        f"(smoothing={SMOOTHING}, min_conf={MIN_CONFIDENCE:.2f}, "
        f"hysteresis={OUTPUT_HYSTERESIS}, softmax_reject={REALTIME_TUNING.softmax_reject_enabled})"
    )

    print("[gesture] right arm model:", model_right_path)
    bundle_right = load_gesture_bundle(model_right_path, device=device)
    use_strict_layout_right = _bundle_uses_strict_layout(bundle_right)

    index_to_label_right = _canonical_label_map(bundle_right.index_to_label)
    _, allowed_indices_right = _resolve_allowed_labels(
        index_to_label_right, "right"
    )

    print("[gesture] left arm model:", model_left_path)
    bundle_left = load_gesture_bundle(model_left_path, device=device)
    left_channels = bundle_left.channel_count
    use_strict_layout_left = _bundle_uses_strict_layout(bundle_left)
    index_to_label_left = _canonical_label_map(bundle_left.index_to_label)
    _, allowed_indices_left = _resolve_allowed_labels(
        index_to_label_left, "left"
    )
    print(f"[gesture] dual-arm mode | right={RIGHT_ARM_CHANNELS}ch left={left_channels}ch")
    if not use_strict_layout_right:
        raise RuntimeError(
            "Strict realtime requires a strict-layout right-arm bundle. "
            "Re-export the active model with strict channel metadata."
        )
    if not use_strict_layout_left:
        raise RuntimeError(
            "Strict dual-arm realtime requires a strict-layout left-arm bundle. "
            "Re-export the active model with strict channel metadata."
        )

    bundle_fs_right = _resolve_target_fs_hz_from_bundle(bundle_right)
    bundle_fs_left = _resolve_target_fs_hz_from_bundle(bundle_left)
    bundle_target_fs = bundle_fs_right
    if bundle_target_fs is None:
        bundle_target_fs = bundle_fs_left
    if bundle_fs_right is not None and bundle_fs_left is not None:
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
    if runtime_target_fs is None or runtime_target_fs <= 0:
        raise RuntimeError(
            "Strict realtime requires a positive target fs. "
            "Set REALTIME_TARGET_FS_HZ or store target_fs_hz in bundle metadata."
        )
    if bundle_target_fs is not None and abs(runtime_target_fs - bundle_target_fs) > 1e-6:
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
        raise RuntimeError(
            "Strict realtime requires Delsys EMG channel labels for every live stream channel. "
            f"Received {len(channel_labels)} labels for {stream_channels} channels."
        )

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
    print("[gesture] dual-arm strict layout active; pairing/scan order is ignored after pair resolution.")
    base.TrigBase.Start(handler.streamYTData)

    fs = float(runtime_target_fs)
    filter_obj = define_filters(fs)
    realtime_filter_spec = _define_realtime_stateful_filters(fs)
    resampler = _RealtimeTimestampResampler(stream_channels, fs)
    print(f"[gesture] realtime timestamp resampling enabled at {fs:.2f} Hz")
    print("[gesture] realtime filter mode: scipy_stateful")

    poll_sleep = 0.001

    # Per-arm calibration values
    neutral_mean_right = None
    mvc_scale_right    = None
    neutral_mean_left  = None
    mvc_scale_left     = None

    def _ratio_from_filtered(neutral_f, mvc_f):
        neutral_rms = np.sqrt(np.mean(neutral_f ** 2, axis=0))
        mvc_rms = np.sqrt(np.mean(mvc_f ** 2, axis=0))
        ratio = np.ones_like(mvc_rms, dtype=float)
        np.divide(mvc_rms, neutral_rms, out=ratio, where=neutral_rms >= 1e-9)
        return ratio

    def _collect_calibration_full(arm_label):
        print(f"\nCalibration [{arm_label}]: RELAX {arm_label} arm completely "
              f"for {CALIB_NEUTRAL_S:.1f}s.")
        n_samples = _collect_samples(handler, CALIB_NEUTRAL_S, stream_channels, poll_sleep)

        if CALIB_MVC_PREP_S > 0:
            print(f"Prepare: SQUEEZE {arm_label.upper()} ARM as hard as possible "
                  f"in {CALIB_MVC_PREP_S:.0f}s...")
            time.sleep(CALIB_MVC_PREP_S)
        print(f"Calibration [{arm_label}]: SQUEEZE {arm_label.upper()} ARM "
              f"as hard as possible for {CALIB_MVC_S:.1f}s.")
        m_samples = _collect_samples(handler, CALIB_MVC_S, stream_channels, poll_sleep)
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

    if CALIBRATE:
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
    pending_right = 0
    pred_history_right = deque(maxlen=max(1, SMOOTHING))
    pending_left = 0
    pred_history_left = deque(maxlen=max(1, SMOOTHING))
    filtered_buffer_right = deque(maxlen=WINDOW_SIZE)
    filtered_buffer_left = deque(maxlen=WINDOW_SIZE)
    filter_state_right = None
    filter_state_left = None

    last_output  = None
    last_msg_len = 0
    output_hysteresis_right = _OutputHysteresis()
    output_hysteresis_left = _OutputHysteresis()
    current_right_label = LOW_CONFIDENCE_LABEL
    current_right_conf = 0.0
    current_left_label = LOW_CONFIDENCE_LABEL
    current_left_conf = 0.0

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

            batch_arr, _ = resampler.push(channel_times, channel_values)
            if batch_arr.size == 0:
                continue

            # --- split channels by arm ---
            batch_right = _slice_channels_by_indices(
                batch_arr, right_channel_indices, RIGHT_ARM_CHANNELS
            )
            batch_left = _slice_channels_by_indices(
                batch_arr, left_channel_indices, left_channels
            )

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

            # --- inference: right arm ---
            inference_ran = False
            label_right = LOW_CONFIDENCE_LABEL
            conf_right  = 0.0
            right_ready = pending_right >= WINDOW_STEP and len(filtered_buffer_right) == WINDOW_SIZE

            if right_ready:
                filtered = np.asarray(filtered_buffer_right, dtype=float)
                if neutral_mean_right is not None and mvc_scale_right is not None:
                    neutral_mean_right, mvc_scale_right = _align_calibration_vectors(
                        neutral_mean_right, mvc_scale_right, filtered.shape[1], "right"
                    )
                    filtered = (filtered - neutral_mean_right) / mvc_scale_right

                window_raw = filtered.T[np.newaxis, :, :].astype(np.float32)
                window = bundle_right.standardize(window_raw)
                probs = bundle_right.predict_proba(window)[0]
                probs  = _restrict_probs(probs, allowed_indices_right)

                pred_history_right.append(probs)
                if SMOOTHING > 1:
                    probs = np.mean(np.stack(pred_history_right, axis=0), axis=0)

                ranking_right, label_right, conf_right = _decode_prediction(
                    probs, index_to_label_right
                )
                if label_right != LOW_CONFIDENCE_LABEL:
                    label_right, conf_right = _apply_softmax_reject_gate(
                        label_right, conf_right, ranking_right
                    )
                current_right_label, current_right_conf = output_hysteresis_right.update(
                    label_right, conf_right
                )

                inference_ran = True
                # We intentionally keep only the newest inference point and
                # drop backlog to avoid re-scoring the same latest window.
                pending_right = 0

            # --- inference: left arm ---
            label_left = LOW_CONFIDENCE_LABEL
            conf_left  = 0.0
            left_ready = pending_left >= WINDOW_STEP and len(filtered_buffer_left) == WINDOW_SIZE

            if left_ready:
                filtered = np.asarray(filtered_buffer_left, dtype=float)
                if neutral_mean_left is not None and mvc_scale_left is not None:
                    neutral_mean_left, mvc_scale_left = _align_calibration_vectors(
                        neutral_mean_left, mvc_scale_left, filtered.shape[1], "left"
                    )
                    filtered = (filtered - neutral_mean_left) / mvc_scale_left

                window = filtered.T[np.newaxis, :, :].astype(np.float32)
                window = bundle_left.standardize(window)
                probs = bundle_left.predict_proba(window)[0]
                probs  = _restrict_probs(probs, allowed_indices_left)

                pred_history_left.append(probs)
                if SMOOTHING > 1:
                    probs = np.mean(np.stack(pred_history_left, axis=0), axis=0)

                ranking_left, label_left, conf_left = _decode_prediction(
                    probs, index_to_label_left
                )
                if label_left != LOW_CONFIDENCE_LABEL:
                    label_left, conf_left = _apply_softmax_reject_gate(
                        label_left, conf_left, ranking_left
                    )
                current_left_label, current_left_conf = output_hysteresis_left.update(
                    label_left, conf_left
                )

                inference_ran = True
                # Same backlog policy as right arm: latest window only.
                pending_left = 0

            # --- fusion and output ---
            if inference_ran:
                label, confidence = fuse_predictions(
                    current_right_label,
                    current_right_conf,
                    current_left_label,
                    current_left_conf,
                )

                combined_state = ArmGestureState(label, confidence)
                prediction_wall_time = time.time()
                published_output = resolve_published_gesture_output(
                    ArmGestureState(current_right_label, current_right_conf),
                    ArmGestureState(current_left_label, current_left_conf),
                    combined_state,
                )

                state = set_latest_dual_state(
                    right_label=current_right_label,
                    right_confidence=current_right_conf,
                    left_label=current_left_label,
                    left_confidence=current_left_conf,
                    combined_label=label,
                    combined_confidence=confidence,
                    published=published_output,
                    window_end_ts=batch_wall_time,
                    prediction_ts=prediction_wall_time,
                )
                if prediction_logger is not None:
                    prediction_logger.write_state(state)

                is_neutral_only = all(
                    gesture.label == LOW_CONFIDENCE_LABEL
                    for gesture in published_output.gestures
                )
                if not published_output.gestures:
                    is_neutral_only = True
                if (not is_neutral_only) or (last_output is not None):
                    msg = format_published_gesture_output(published_output)
                    if msg != last_output:
                        if len(msg) < last_msg_len:
                            msg = msg.ljust(last_msg_len)
                        else:
                            last_msg_len = len(msg)
                        print(msg, end="\r", flush=True)
                        last_output = format_published_gesture_output(published_output)
    except KeyboardInterrupt:
        print("\nStopping stream...")
    finally:
        base.Stop_Callback()
        if prediction_logger is not None:
            prediction_logger.close()


if __name__ == "__main__":
    main()
