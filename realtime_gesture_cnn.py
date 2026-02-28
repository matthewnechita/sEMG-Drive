import argparse
import time
from collections import deque

import os
import threading

import numpy as np
import torch

from AeroPy.TrignoBase import TrignoBase
from AeroPy.DataManager import DataKernel

# Old libEMG filtering (re-enabled). Commenting out the SciPy SOS filtering.
# from emg.filtering import define_filters, apply_filters, make_filter_state, apply_filters_stateful
from libemg import filtering as libemg_filter
from emg.gesture_model_cnn import load_cnn_bundle, quick_finetune


# ======== Config (edit as needed) ========
WINDOW_SIZE = 200
WINDOW_STEP = 100
FILTER_WARMUP = 200  # extra samples to stabilize old libEMG filtering on rolling buffers

SMOOTHING = 11  # number of windows to average (~550ms at 2kHz/step-100)
MIN_CONFIDENCE = 0.65  # for 6 classes, 40% was too low to gate uncertainty
LOW_CONFIDENCE_LABEL = "neutral"

LATEST_LOCK = threading.Lock()
LATEST_GESTURE = LOW_CONFIDENCE_LABEL
LATEST_TIMESTAMP = 0.0

CALIBRATE = True
CALIB_NEUTRAL_S = 5.0
CALIB_MVC_S = 5.0
CALIB_MVC_PREP_S = 2.0    # countdown pause before MVC window
MVC_PERCENTILE = 95.0
MVC_MIN_RATIO = 2.0        # minimum acceptable median MVC/neutral ratio

# ======== Dual-arm config ========
# Right arm channels come first in the Delsys stream (pair right arm sensors first).
# Left arm channels follow immediately after; inferred from bundle_left.channel_count at load time.
RIGHT_ARM_CHANNELS = 17   # fixed: 6 right arm sensors → 17 channels
# Left arm uses 5 sensors → 16 channels (one fewer sensor than right arm)
# Fusion thresholds:
#   AGREE_THRESHOLD  — minimum confidence to emit a gesture when BOTH arms agree
#                      (lower than single-arm threshold because agreement strengthens prediction)
#   SINGLE_THRESHOLD — minimum confidence to emit when only one arm is available/confident
DUAL_ARM_AGREE_THRESHOLD  = 0.55
DUAL_ARM_SINGLE_THRESHOLD = 0.65   # matches MIN_CONFIDENCE above
# =================================

# ======== Per-gesture calibration ========
# Collect a short sample of each gesture before inference to fine-tune
# the model's classification head for this specific subject.
# Requires CALIBRATE = True (needs filter_obj and MVC normalization).
GESTURE_CALIB        = True
GESTURE_CALIB_S      = 3.0   # seconds of EMG to collect per gesture
GESTURE_CALIB_PREP_S = 2.0   # countdown pause before each gesture

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
#   "right" — right arm only  (pair right arm sensors first in Delsys)
#   "left"  — left arm only   (pair left arm sensors first in Delsys)
#   "dual"  — both arms fused (pair right first, then left in Delsys)
MODE        = "right"
MODEL_RIGHT = "models/cross_subject/right/gesture_cnn_v2.pt"
MODEL_LEFT  = "models/cross_subject/left/gesture_cnn_v2.pt"
# ================================


class _StreamingHandler:
    def __init__(self):
        self.streamYTData = True
        self.pauseFlag = False
        self.DataHandler = None
        self.EMGplot = None

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
    """Unpack a GetYTData() frame into parallel lists of per-channel times and values."""
    channel_times = []
    channel_values = []
    for channel in out:
        if not channel:
            continue
        chan_array = np.asarray(channel[0], dtype=object)
        if chan_array.size == 0:
            continue
        t_vals, v_vals = zip(*(_pair_time_value(s) for s in chan_array))
        channel_times.append(list(t_vals))
        channel_values.append(list(v_vals))
    return channel_times, channel_values


def control_hook(gesture: str) -> None:
    set_latest_gesture(gesture)
    return

def set_latest_gesture(label: str) -> None:
    global LATEST_GESTURE, LATEST_TIMESTAMP
    with LATEST_LOCK:
        LATEST_GESTURE = str(label)
        LATEST_TIMESTAMP = time.time()

def get_latest_gesture(default: str = LOW_CONFIDENCE_LABEL):
    with LATEST_LOCK:
        label = LATEST_GESTURE if LATEST_GESTURE is not None else default
        ts = LATEST_TIMESTAMP
        age = (time.time() - ts) if ts else float('inf')
        return label, age

def fuse_predictions(label_r, conf_r, label_l, conf_l):
    """Combine right and left arm predictions into one fused label.

    If both arms agree on the same gesture, the prediction is reinforced and
    a lower confidence threshold is used to emit it. If they disagree, each
    arm is evaluated independently against the single-arm threshold.

    Returns: (fused_label, fused_confidence)
    """
    if label_r == label_l:
        # Both arms agree — strengthen the prediction
        combined_conf = max(conf_r, conf_l)
        if combined_conf >= DUAL_ARM_AGREE_THRESHOLD:
            return label_r, combined_conf
    # Arms disagree (or agree but below threshold) — use whichever clears single threshold
    # Right arm takes priority in a tie.
    if conf_r >= DUAL_ARM_SINGLE_THRESHOLD:
        return label_r, conf_r
    if conf_l >= DUAL_ARM_SINGLE_THRESHOLD:
        return label_l, conf_l
    return LOW_CONFIDENCE_LABEL, 0.0


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


def main(argv=None):
    if MODE not in ("right", "left", "dual"):
        raise ValueError(f"MODE must be 'right', 'left', or 'dual', got {MODE!r}")

    # Derive defaults from config; CLI args can still override.
    # In "left" mode, the left model runs as single-arm (left sensors paired first).
    _config_right = MODEL_LEFT if MODE == "left" else MODEL_RIGHT
    _config_left  = MODEL_LEFT if MODE == "dual"  else None

    parser = argparse.ArgumentParser(description="Real-time CNN gesture inference.")
    parser.add_argument("--model",       default=_config_right,
                        help="Right arm model (overrides MODEL_RIGHT / MODEL_LEFT config).")
    parser.add_argument("--model-right", default=None,
                        help="Right arm model alias; overrides --model if both given.")
    parser.add_argument("--model-left",  default=_config_left,
                        help="Left arm model. Set automatically from MODE=dual; overrides config.")

    args = parser.parse_args(argv)

    # --model-right overrides --model so either flag works
    model_right_path = args.model_right if args.model_right else args.model
    model_left_path  = args.model_left   # None → single-arm mode

    print("[gesture] cwd:", os.getcwd())

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('[gesture] device:', device)

    # Right arm bundle (always loaded)
    print("[gesture] right arm model:", model_right_path)
    bundle_right  = load_cnn_bundle(model_right_path, device=device)
    model_channels = bundle_right.channel_count   # kept for single-arm compat

    # Left arm bundle (dual-arm mode only)
    dual_arm = model_left_path is not None
    bundle_left   = None
    left_channels = 0
    if dual_arm:
        print("[gesture] left arm model:", model_left_path)
        bundle_left   = load_cnn_bundle(model_left_path, device=device)
        left_channels = bundle_left.channel_count
        print(f"[gesture] dual-arm mode | right={RIGHT_ARM_CHANNELS}ch left={left_channels}ch")
    else:
        print("[gesture] single-arm mode")

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

    # For single-arm left mode: auto-detect channel offset so the system works
    # whether only left arm sensors are connected (left_arm_start=0) or all
    # sensors are connected from a prior dual-arm session (left_arm_start=17).
    left_arm_start = 0
    if not dual_arm and MODE == "left":
        if stream_channels >= RIGHT_ARM_CHANNELS + model_channels:
            left_arm_start = RIGHT_ARM_CHANNELS
            print(
                f"[gesture] all sensors connected; left-arm inference using "
                f"channels {left_arm_start}–{left_arm_start + model_channels - 1}"
            )
        else:
            print(f"[gesture] left-arm sensors only; using channels 0–{model_channels - 1}")

    if dual_arm:
        expected_total = RIGHT_ARM_CHANNELS + left_channels
        if stream_channels < expected_total:
            raise RuntimeError(
                f"Dual-arm requires {expected_total} stream channels "
                f"({RIGHT_ARM_CHANNELS} right + {left_channels} left) "
                f"but stream only has {stream_channels}. "
                "Pair right arm sensors first, then left arm sensors, before scanning."
            )
    else:
        required = left_arm_start + model_channels
        if stream_channels < required:
            print(
                f"Channel count mismatch: need {required} channels "
                f"(model expects {model_channels} at offset {left_arm_start}), "
                f"stream has {stream_channels}; padding/trimming."
            )

    base.TrigBase.Start(handler.streamYTData)

    filter_obj = None
    fs = None
    poll_sleep = 0.001

    # Per-arm calibration values (single-arm uses only the _right variants)
    neutral_mean       = None   # single-arm compat alias → points to neutral_mean_right
    mvc_scale          = None   # single-arm compat alias → points to mvc_scale_right
    neutral_mean_right = None
    mvc_scale_right    = None
    neutral_mean_left  = None
    mvc_scale_left     = None

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
                if fs is not None and filter_obj is None:
                    filter_obj = define_filters(fs)
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
                if fs is not None and filter_obj is None:
                    filter_obj = define_filters(fs)
                    print(f"Estimated fs: {fs:.2f} Hz")

            if not (m_samples.size and n_samples.size):
                calib_done = True
                return None, None

            # Slice to this arm's channels if collecting full stream
            if collect_channels is None:
                n_arr = n_samples[:, channel_start:channel_start + arm_channels]
                m_arr = m_samples[:, channel_start:channel_start + arm_channels]
            else:
                n_arr = n_samples
                m_arr = m_samples

            neutral_f = apply_filters(filter_obj, n_arr)
            mvc_f     = apply_filters(filter_obj, m_arr)

            neutral_rms = np.sqrt(np.mean(neutral_f ** 2, axis=0))
            mvc_rms     = np.sqrt(np.mean(mvc_f     ** 2, axis=0))
            ratio       = np.where(neutral_rms < 1e-9, 1.0, mvc_rms / neutral_rms)
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

    if CALIBRATE:
        if dual_arm:
            # Dual-arm: calibrate each arm separately so MVC squeeze is arm-specific
            neutral_mean_right, mvc_scale_right = _do_calibration(
                "right", RIGHT_ARM_CHANNELS, collect_channels=None, channel_start=0
            )
            neutral_mean_left, mvc_scale_left = _do_calibration(
                "left", left_channels, collect_channels=None, channel_start=RIGHT_ARM_CHANNELS
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

    if CALIBRATE and GESTURE_CALIB and filter_obj is not None:
        print("\n=== Per-gesture calibration ===")
        print(f"Perform each gesture for {GESTURE_CALIB_S:.0f}s when prompted.\n")

        label_to_idx_r = {v: k for k, v in bundle_right.index_to_label.items()}
        label_to_idx_l = {v: k for k, v in bundle_left.index_to_label.items()} if dual_arm else {}

        wins_r, labs_r = [], []
        wins_l, labs_l = [], []

        for gesture in GESTURE_LABELS:
            if gesture not in label_to_idx_r:
                continue
            instr = GESTURE_INSTRUCTIONS.get(gesture, gesture)
            if GESTURE_CALIB_PREP_S > 0:
                print(f"Next: {instr}  —  get ready ({GESTURE_CALIB_PREP_S:.0f}s)...")
                time.sleep(GESTURE_CALIB_PREP_S)
            print(f"GO: {instr}")

            samples, _ = _collect_samples(
                handler, GESTURE_CALIB_S, stream_channels, None, poll_sleep
            )
            if samples.size == 0:
                print(f"  Warning: no samples for '{gesture}', skipping.")
                continue

            # Right arm
            r_raw = samples[:, :RIGHT_ARM_CHANNELS] if dual_arm else samples[:, left_arm_start:left_arm_start + model_channels]
            r_filt = apply_filters(filter_obj, r_raw)
            if neutral_mean_right is not None and mvc_scale_right is not None:
                r_filt = (r_filt - neutral_mean_right) / mvc_scale_right
            w_r = _make_windows(r_filt)
            if len(w_r):
                wins_r.append(w_r)
                labs_r.extend([label_to_idx_r[gesture]] * len(w_r))
                print(f"  Right: {len(w_r)} windows collected")

            # Left arm (dual-arm only)
            if dual_arm and gesture in label_to_idx_l:
                l_raw = samples[:, RIGHT_ARM_CHANNELS:RIGHT_ARM_CHANNELS + left_channels]
                l_filt = apply_filters(filter_obj, l_raw)
                if neutral_mean_left is not None and mvc_scale_left is not None:
                    l_filt = (l_filt - neutral_mean_left) / mvc_scale_left
                w_l = _make_windows(l_filt)
                if len(w_l):
                    wins_l.append(w_l)
                    labs_l.extend([label_to_idx_l[gesture]] * len(w_l))
                    print(f"  Left:  {len(w_l)} windows collected")

        if wins_r:
            X_r = np.concatenate(wins_r, axis=0)
            y_r = np.array(labs_r, dtype=np.int64)
            print(f"\nFine-tuning right arm model ({len(y_r)} windows)...")
            quick_finetune(bundle_right, X_r, y_r, device=device)
            print("  Right arm model updated.")

        if dual_arm and wins_l:
            X_l = np.concatenate(wins_l, axis=0)
            y_l = np.array(labs_l, dtype=np.int64)
            print(f"Fine-tuning left arm model ({len(y_l)} windows)...")
            quick_finetune(bundle_left, X_l, y_l, device=device)
            print("  Left arm model updated.")

        print("\n=== Gesture calibration complete. Starting inference... ===\n")

    buffer_len = WINDOW_SIZE + FILTER_WARMUP

    # Right arm (always present)
    raw_buffer_right   = deque(maxlen=buffer_len)
    warmup_right       = FILTER_WARMUP == 0
    pending_right      = 0
    pred_history_right = deque(maxlen=max(1, SMOOTHING))

    # Left arm (dual-arm mode only)
    raw_buffer_left    = deque(maxlen=buffer_len)
    warmup_left        = FILTER_WARMUP == 0
    pending_left       = 0
    pred_history_left  = deque(maxlen=max(1, SMOOTHING))

    last_output  = None
    last_msg_len = 0
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

            sample_count = min(len(c) for c in channel_values)
            if sample_count == 0:
                continue

            if fs is None and channel_times:
                fs = _estimate_fs(channel_times[0])
                if fs is not None and filter_obj is None:
                    filter_obj = define_filters(fs)
                    print(f"Estimated fs: {fs:.2f} Hz")

            # Capture when this batch of samples became available to this process.
            batch_wall_time = time.time()

            # --- collect raw batch (all stream channels) ---
            sample_batch = []
            for idx in range(sample_count):
                sample = [channel_values[ch][idx] for ch in range(stream_channels)]
                # sample_buffer.append(sample)  # old: append to raw rolling buffer
                sample_batch.append(sample)

            if not sample_batch:
                continue

            batch_arr = np.asarray(sample_batch, dtype=float)  # (N, stream_channels)

            # --- split channels by arm ---
            if dual_arm:
                batch_right = batch_arr[:, :RIGHT_ARM_CHANNELS]
                batch_left = batch_arr[:, RIGHT_ARM_CHANNELS:RIGHT_ARM_CHANNELS + left_channels]
            else:
                batch_right = batch_arr[:, left_arm_start:left_arm_start + model_channels]

            # --- old approach: raw rolling buffers + libEMG filtering ---
            for s in batch_right:
                raw_buffer_right.append(s)
                pending_right += 1
            if not warmup_right and len(raw_buffer_right) >= buffer_len:
                warmup_right = True
                pending_right = WINDOW_STEP

            if dual_arm:
                for s in batch_left:
                    raw_buffer_left.append(s)
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
            if warmup_right and pending_right >= WINDOW_STEP and len(raw_buffer_right) >= WINDOW_SIZE:
                raw = np.asarray(raw_buffer_right, dtype=float)
                filtered_full = apply_filters(filter_obj, raw)
                filtered = filtered_full[-WINDOW_SIZE:]
                if neutral_mean_right is not None and mvc_scale_right is not None:
                    filtered = (filtered - neutral_mean_right) / mvc_scale_right

                window = filtered.T[np.newaxis, :, :].astype(np.float32)
                window = bundle_right.standardize(window)
                probs  = bundle_right.predict_proba(window)[0]

                pred_history_right.append(probs)
                if SMOOTHING > 1:
                    probs = np.mean(np.stack(pred_history_right, axis=0), axis=0)

                conf_right  = float(np.max(probs))
                pred_idx    = int(np.argmax(probs))
                label_right = bundle_right.index_to_label[pred_idx]
                if conf_right < MIN_CONFIDENCE:
                    label_right = LOW_CONFIDENCE_LABEL

                inference_ran = True
                # We intentionally keep only the newest inference point and
                # drop backlog to avoid re-scoring the same latest window.
                pending_right = 0

            # --- inference: left arm (dual-arm mode only) ---
            label_left = LOW_CONFIDENCE_LABEL
            conf_left  = 0.0
            if dual_arm:
                if warmup_left and pending_left >= WINDOW_STEP and len(raw_buffer_left) >= WINDOW_SIZE:
                    raw = np.asarray(raw_buffer_left, dtype=float)
                    filtered_full = apply_filters(filter_obj, raw)
                    filtered = filtered_full[-WINDOW_SIZE:]
                    if neutral_mean_left is not None and mvc_scale_left is not None:
                        filtered = (filtered - neutral_mean_left) / mvc_scale_left

                    window = filtered.T[np.newaxis, :, :].astype(np.float32)
                    window = bundle_left.standardize(window)
                    probs  = bundle_left.predict_proba(window)[0]

                    pred_history_left.append(probs)
                    if SMOOTHING > 1:
                        probs = np.mean(np.stack(pred_history_left, axis=0), axis=0)

                    conf_left  = float(np.max(probs))
                    pred_idx   = int(np.argmax(probs))
                    label_left = bundle_left.index_to_label[pred_idx]
                    if conf_left < MIN_CONFIDENCE:
                        label_left = LOW_CONFIDENCE_LABEL

                    inference_ran = True
                    # Same backlog policy as right arm: latest window only.
                    pending_left = 0

            # --- fusion and output ---
            if inference_ran:
                if dual_arm:
                    label, confidence = fuse_predictions(
                        label_right, conf_right, label_left, conf_left
                    )
                else:
                    label, confidence = label_right, conf_right

                if label != LOW_CONFIDENCE_LABEL or last_output != LOW_CONFIDENCE_LABEL:
                    control_hook(label)

    #             # === LATENCY MEASURE START ===
    #             if latency_enabled:
    #                 t1 = time.time()
    #                 proc_ms.append((t1 - t0) * 1000.0)
    #                 latest_ts = time_buffer[-1] if time_buffer else None
    #                 if latest_ts is not None:
    #                     latency_ms.append((t1 - latest_ts) * 1000.0)
    #                 preds += 1
    #                 if latency_print_every > 0 and preds % latency_print_every == 0:
    #                     if latency_ms:
    #                         print(
    #                             f"latency_ms={latency_ms[-1]:.1f} "
    #                             f"proc_ms={proc_ms[-1]:.1f}",
    #                             end="\r",
    #                             flush=True,
    #                         )
    #                 if latency_max_preds > 0 and preds >= latency_max_preds:
    #                     raise KeyboardInterrupt
    #             # === LATENCY MEASURE END ===

                    if label != last_output:
                        msg = f"Gesture: {label} (conf {confidence:.2f})"
                        if len(msg) < last_msg_len:
                            msg = msg.ljust(last_msg_len)
                        else:
                            last_msg_len = len(msg)
                        print(msg, end="\r", flush=True)
                        last_output = label
    except KeyboardInterrupt:
        print("\nStopping stream...")
    finally:
        base.Stop_Callback()
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
