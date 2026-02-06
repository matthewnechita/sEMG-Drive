import argparse
import time
from collections import deque

import numpy as np

from AeroPy.TrignoBase import TrignoBase
from AeroPy.DataManager import DataKernel

from filtering import define_filters, apply_filters
from gesture_model_cnn import load_cnn_bundle


# ======== Config (edit as needed) ========
WINDOW_SIZE = 200
WINDOW_STEP = 100
FILTER_WARMUP = 200  # extra samples for filter warmup

SMOOTHING = 5  # number of windows to average
MIN_CONFIDENCE = 0.4
LOW_CONFIDENCE_LABEL = "neutral"

CALIBRATE = True
CALIB_NEUTRAL_S = 3.0
CALIB_MVC_S = 3.0
MVC_PERCENTILE = 95.0
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


def control_hook(gesture: str) -> None:
    # TODO: replace with control system integration
    return


def _collect_samples(handler, duration_s, stream_channels, model_channels, poll_sleep):
    samples = []
    times = []
    end_time = time.time() + duration_s

    while time.time() < end_time:
        out = handler.DataHandler.GetYTData()
        if out is None:
            time.sleep(poll_sleep)
            continue

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
    parser.add_argument("--model", default="models/gesture_cnn.pt")
    args = parser.parse_args(argv)

    bundle = load_cnn_bundle(args.model)
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
        print(f"Calibration: neutral rest for {CALIB_NEUTRAL_S:.1f}s.")
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
        else:
            print(f"Calibration: max contraction for {CALIB_MVC_S:.1f}s.")
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
                neutral_mean = np.mean(neutral_f, axis=0)
                mvc_scale = np.percentile(mvc_f, MVC_PERCENTILE, axis=0)
                mvc_scale = np.where(mvc_scale < 1e-6, 1.0, mvc_scale)
                print("Calibration complete.")

    sample_buffer = deque(maxlen=WINDOW_SIZE + FILTER_WARMUP)
    pending_samples = 0
    pred_history = deque(maxlen=max(1, SMOOTHING))
    last_output = None
    last_msg_len = 0
    # === LATENCY MEASURE START ===
    # Comment out this whole block (and other LATENCY blocks below) after measuring.
    # Measures end-to-end latency from "data available in this process" to prediction time.
    latency_enabled = True
    latency_print_every = 20
    latency_max_preds = 200
    latency_ms = []
    proc_ms = []
    preds = 0
    time_buffer = deque(maxlen=WINDOW_SIZE + FILTER_WARMUP)
    # === LATENCY MEASURE END ===

    try:
        while True:
            out = handler.DataHandler.GetYTData()
            if out is None:
                time.sleep(poll_sleep)
                continue

            channel_times = []
            channel_values = []
            for channel in out:
                if not channel:
                    continue
                chan_array = np.asarray(channel[0], dtype=object)
                if chan_array.size == 0:
                    continue
                times, values = zip(*(_pair_time_value(s) for s in chan_array))
                channel_times.append(list(times))
                channel_values.append(list(values))

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

            for idx in range(sample_count):
                sample = [channel_values[ch][idx] for ch in range(stream_channels)]
                if len(sample) < model_channels:
                    sample.extend([0.0] * (model_channels - len(sample)))
                elif len(sample) > model_channels:
                    sample = sample[:model_channels]
                sample_buffer.append(sample)
                pending_samples += 1
                # === LATENCY MEASURE START ===
                if latency_enabled:
                    time_buffer.append(batch_wall_time)
                # === LATENCY MEASURE END ===

                while pending_samples >= WINDOW_STEP and len(sample_buffer) >= WINDOW_SIZE:
                    if filter_obj is None:
                        pending_samples -= WINDOW_STEP
                        continue

                    # === LATENCY MEASURE START ===
                    if latency_enabled:
                        t0 = time.time()
                    # === LATENCY MEASURE END ===

                    raw = np.asarray(sample_buffer, dtype=float)
                    filtered_full = apply_filters(filter_obj, raw)
                    filtered = filtered_full[-WINDOW_SIZE:]
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

                    # === LATENCY MEASURE START ===
                    if latency_enabled:
                        t1 = time.time()
                        proc_ms.append((t1 - t0) * 1000.0)
                        latest_ts = time_buffer[-1] if time_buffer else None
                        if latest_ts is not None:
                            latency_ms.append((t1 - latest_ts) * 1000.0)
                        preds += 1
                        if latency_print_every > 0 and preds % latency_print_every == 0:
                            if latency_ms:
                                print(
                                    f"latency_ms={latency_ms[-1]:.1f} "
                                    f"proc_ms={proc_ms[-1]:.1f}",
                                    end="\r",
                                    flush=True,
                                )
                        if latency_max_preds > 0 and preds >= latency_max_preds:
                            raise KeyboardInterrupt
                    # === LATENCY MEASURE END ===

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
        if latency_enabled and latency_ms:
            lat = np.asarray(latency_ms, dtype=float)
            proc = np.asarray(proc_ms, dtype=float) if proc_ms else None
            print("\nLatency summary (ms):")
            print(
                f"count={lat.size} "
                f"avg={lat.mean():.1f} "
                f"median={np.median(lat):.1f} "
                f"p90={np.percentile(lat, 90):.1f} "
                f"p95={np.percentile(lat, 95):.1f} "
                f"min={lat.min():.1f} "
                f"max={lat.max():.1f}"
            )
            if proc is not None and proc.size:
                print(
                    "Processing-only summary (ms): "
                    f"avg={proc.mean():.1f} "
                    f"median={np.median(proc):.1f} "
                    f"p90={np.percentile(proc, 90):.1f} "
                    f"p95={np.percentile(proc, 95):.1f} "
                    f"min={proc.min():.1f} "
                    f"max={proc.max():.1f}"
                )
        # === LATENCY MEASURE END ===


if __name__ == "__main__":
    main()
