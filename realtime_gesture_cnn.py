import argparse
import time
from collections import deque

import os
import threading

import numpy as np
import torch

from AeroPy.TrignoBase import TrignoBase
from AeroPy.DataManager import DataKernel

# from emg.filtering import define_filters, apply_filters
from emg.filtering import define_filters, apply_filters, make_filter_state, apply_filters_stateful
from emg.gesture_model_cnn import load_cnn_bundle


# ======== Config (edit as needed) ========
WINDOW_SIZE = 200
WINDOW_STEP = 100
# FILTER_WARMUP = 200  # commented out — replaced by stateful sosfilt (no warmup needed)

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
# ========================================


class _StreamingHandler:
    def __init__(self):
        self.streamYTData = True
        self.pauseFlag = False
        self.DataHandler = None
        self.EMGplot = None

    def threadManager(self, start_trigger: bool, stop_trigger: bool) -> None:
        return


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
    parser = argparse.ArgumentParser(description="Real-time CNN gesture inference.")
    
    DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "cross_subject", "gesture_cnn_v2.pt")
    parser.add_argument("--model", default=DEFAULT_MODEL)

    args = parser.parse_args(argv)
    print("[gesture] using model:", args.model)
    print("[gesture] cwd:", os.getcwd())

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('[gesture] device:', device)
    bundle = load_cnn_bundle(args.model, device=device)
    model_channels = bundle.channel_count

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
    if stream_channels != model_channels:
        print(
            "Channel count mismatch "
            f"(model expects {model_channels}, stream has {stream_channels}); "
            "padding/trimming stream to match model."
        )

    base.TrigBase.Start(handler.streamYTData)

    filter_obj = None
    fs = None
    poll_sleep = 0.001

    neutral_mean = None
    mvc_scale = None
    if CALIBRATE:
        calib_done = False
        while not calib_done:
            print(f"\nCalibration: RELAX completely — arm still for {CALIB_NEUTRAL_S:.1f}s.")
            neutral_samples, neutral_times = _collect_samples(
                handler, CALIB_NEUTRAL_S, stream_channels, model_channels, poll_sleep
            )
            if fs is None and neutral_times.size:
                fs = _estimate_fs(neutral_times)
                if fs is not None and filter_obj is None:
                    filter_obj = define_filters(fs)
                    print(f"Estimated fs: {fs:.2f} Hz")

            if filter_obj is None:
                print("Warning: calibration skipped (no filter available).")
                break

            if CALIB_MVC_PREP_S > 0:
                print(f"Prepare: SQUEEZE AS HARD AS POSSIBLE in {CALIB_MVC_PREP_S:.0f}s...")
                time.sleep(CALIB_MVC_PREP_S)

            print(f"Calibration: SQUEEZE AS HARD AS POSSIBLE — all muscles — for {CALIB_MVC_S:.1f}s.")
            mvc_samples, mvc_times = _collect_samples(
                handler, CALIB_MVC_S, stream_channels, model_channels, poll_sleep
            )
            if fs is None and mvc_times.size:
                fs = _estimate_fs(mvc_times)
                if fs is not None and filter_obj is None:
                    filter_obj = define_filters(fs)
                    print(f"Estimated fs: {fs:.2f} Hz")

            if mvc_samples.size and neutral_samples.size:
                neutral_f = apply_filters(filter_obj, neutral_samples)
                mvc_f = apply_filters(filter_obj, mvc_samples)

                # Quality check before committing calibration
                neutral_rms = np.sqrt(np.mean(neutral_f ** 2, axis=0))
                mvc_rms = np.sqrt(np.mean(mvc_f ** 2, axis=0))
                ratio = np.where(neutral_rms < 1e-9, 1.0, mvc_rms / neutral_rms)
                median_ratio = float(np.median(ratio))
                n_weak = int(np.sum(ratio < MVC_MIN_RATIO))
                print(
                    f"MVC quality: {median_ratio:.1f}x median "
                    f"({n_weak}/{len(ratio)} channels below {MVC_MIN_RATIO:.0f}x)"
                )

                if median_ratio < MVC_MIN_RATIO:
                    print(
                        f"\n*** WARNING: Weak calibration ({median_ratio:.1f}x) ***\n"
                        "  Your MVC was too close to neutral rest.\n"
                        "  Squeeze ALL arm/wrist muscles simultaneously with full force.\n"
                    )
                    retry = input("Retry calibration? [y/N]: ").strip().lower()
                    if retry == "y":
                        continue

                neutral_mean = np.mean(neutral_f, axis=0)
                mvc_scale = np.percentile(mvc_f, MVC_PERCENTILE, axis=0)
                mvc_scale = np.where(mvc_scale < 1e-6, 1.0, mvc_scale)
                if median_ratio < MVC_MIN_RATIO:
                    print("Calibration accepted with quality warning. Accuracy may be reduced.")
                else:
                    print(f"Calibration complete. (quality: {median_ratio:.1f}x)")
            calib_done = True

    # sample_buffer = deque(maxlen=WINDOW_SIZE + FILTER_WARMUP)  # old: raw buffer + filtfilt on rolling window
    filtered_buffer = deque(maxlen=WINDOW_SIZE)  # new: stores already-filtered samples
    filter_state = None  # new: persistent sosfilt state — initialized on first batch
    pending_samples = 0
    pred_history = deque(maxlen=max(1, SMOOTHING))
    last_output = None
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

            # --- collect raw batch ---
            sample_batch = []
            for idx in range(sample_count):
                sample = [channel_values[ch][idx] for ch in range(stream_channels)]
                if len(sample) < model_channels:
                    sample.extend([0.0] * (model_channels - len(sample)))
                elif len(sample) > model_channels:
                    sample = sample[:model_channels]
                # sample_buffer.append(sample)  # old: append to raw rolling buffer
                sample_batch.append(sample)

            # --- stateful causal filtering (new) ---
            # Filters the whole batch at once, carrying state across calls.
            # Eliminates the ~16-25% train/test mismatch from filtfilt on short buffers.
            if sample_batch and filter_obj is not None:
                if filter_state is None:
                    filter_state = make_filter_state(filter_obj, model_channels)
                batch_arr = np.asarray(sample_batch, dtype=float)
                filtered_batch, filter_state = apply_filters_stateful(filter_obj, batch_arr, filter_state)
                for s in filtered_batch:
                    filtered_buffer.append(s)
                    pending_samples += 1
            elif sample_batch:
                pending_samples += len(sample_batch)  # filter not ready yet, just count

            while pending_samples >= WINDOW_STEP and len(filtered_buffer) == WINDOW_SIZE:
                if filter_obj is None:
                    pending_samples -= WINDOW_STEP
                    continue

                # old approach (filtfilt on rolling raw buffer):
                # raw = np.asarray(sample_buffer, dtype=float)
                # filtered_full = apply_filters(filter_obj, raw)
                # filtered = filtered_full[-WINDOW_SIZE:]

                filtered = np.asarray(filtered_buffer, dtype=float)
                if neutral_mean is not None and mvc_scale is not None:
                    filtered = (filtered - neutral_mean) / mvc_scale

                window = filtered.T[np.newaxis, :, :].astype(np.float32)
                window = bundle.standardize(window)
                probs = bundle.predict_proba(window)[0]

                pred_history.append(probs)
                if SMOOTHING > 1:
                    probs = np.mean(np.stack(pred_history, axis=0), axis=0)

                confidence = float(np.max(probs))
                pred_idx = int(np.argmax(probs))
                label = bundle.index_to_label[pred_idx]

                if confidence < MIN_CONFIDENCE:
                    label = LOW_CONFIDENCE_LABEL

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

                pending_samples -= WINDOW_STEP
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
