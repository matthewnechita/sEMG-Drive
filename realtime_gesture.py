import argparse
import time
from collections import Counter, deque

import numpy as np

from AeroPy.TrignoBase import TrignoBase
from AeroPy.DataManager import DataKernel
from libemg.feature_extractor import FeatureExtractor

from filtering import define_filters, apply_filters
from gesture_model import load_model_bundle, flatten_feature_dict


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


def _majority_vote(values):
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def control_hook(gesture: str) -> None:
    # Placeholder for CARLA/ROS control integration.
    return


def _collect_samples(
    handler,
    duration_s: float,
    stream_channels: int,
    model_channels: int | None,
    poll_sleep: float,
):
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


def build_parser():
    parser = argparse.ArgumentParser(description="Real-time gesture inference from EMG stream.")
    parser.add_argument("--model", default="models/gesture_classifier.pkl")
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--window-step", type=int, default=None)
    parser.add_argument("--smoothing", type=int, default=5)
    parser.add_argument("--min-confidence", type=float, default=0.7)
    parser.add_argument("--low-confidence-label", default="neutral")
    parser.add_argument("--fs", type=float, default=None)
    parser.add_argument("--poll-sleep", type=float, default=0.001)
    parser.add_argument("--show-confidence", action="store_true")
    parser.add_argument(
        "--calibrate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run neutral/MVC calibration at startup.",
    )
    parser.add_argument("--calibration-neutral", type=float, default=5.0)
    parser.add_argument("--calibration-mvc", type=float, default=5.0)
    parser.add_argument("--mvc-percentile", type=float, default=95.0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    bundle = load_model_bundle(args.model)
    meta = bundle.metadata
    if args.min_confidence > 0.0 and not hasattr(bundle.model, "predict_proba"):
        print("Warning: min-confidence set but model lacks predict_proba; retrain with --svm-probability.")

    feature_order = bundle.feature_order
    if not feature_order:
        raise ValueError("Model metadata missing feature order.")

    window_size = args.window_size or int(meta.get("window_size_samples", 200))
    window_step = args.window_step or int(meta.get("window_step_samples", 100))
    expected_channels = bundle.channel_count

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
    model_channels = expected_channels if expected_channels is not None else stream_channels
    if stream_channels != model_channels:
        print(
            "Channel count mismatch "
            f"(model expects {model_channels}, stream has {stream_channels}); "
            "padding/trimming stream to match model."
        )
        # TODO: RETRAIN DATA ON FINAL 17-CHANNEL SENSOR SETUP SO MODEL EXPECTS 17 CHANNELS.

    base.TrigBase.Start(handler.streamYTData)

    fe = FeatureExtractor()
    filter_obj = None
    fs = None
    if args.fs is not None:
        fs = float(args.fs)
    else:
        meta_fs = meta.get("fs_hz") or meta.get("fs")
        if meta_fs is not None:
            fs = float(meta_fs)
    if fs is not None:
        filter_obj = define_filters(fs)
        print(f"Using fs: {fs:.2f} Hz")

    neutral_mean = None
    mvc_scale = None
    if args.calibrate:
        print(f"Calibration: neutral rest for {args.calibration_neutral:.1f}s.")
        neutral_samples, neutral_times = _collect_samples(
            handler,
            args.calibration_neutral,
            stream_channels,
            model_channels,
            args.poll_sleep,
        )
        if fs is None and neutral_times.size:
            fs = _estimate_fs(neutral_times)
            if fs is not None and filter_obj is None:
                filter_obj = define_filters(fs)
                print(f"Estimated fs: {fs:.2f} Hz")

        if filter_obj is None:
            print("Warning: calibration skipped (no filter available).")
        else:
            print(f"Calibration: max contraction for {args.calibration_mvc:.1f}s.")
            mvc_samples, mvc_times = _collect_samples(
                handler,
                args.calibration_mvc,
                stream_channels,
                model_channels,
                args.poll_sleep,
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
                mvc_scale = np.percentile(mvc_f, args.mvc_percentile, axis=0)
                eps = 1e-6
                mvc_scale = np.where(mvc_scale < eps, 1.0, mvc_scale)
                print("Calibration complete.")
            else:
                print("Warning: calibration skipped (no samples collected).")

    sample_buffer = deque(maxlen=window_size)
    pending_samples = 0
    pred_history = deque(maxlen=max(1, args.smoothing))
    last_output = None
    last_msg_len = 0

    try:
        while True:
            out = handler.DataHandler.GetYTData()
            if out is None:
                time.sleep(args.poll_sleep)
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
                if fs is not None:
                    filter_obj = define_filters(fs)
                    print(f"Estimated fs: {fs:.2f} Hz")

            for idx in range(sample_count):
                sample = [channel_values[ch][idx] for ch in range(stream_channels)]
                if model_channels is not None:
                    if len(sample) < model_channels:
                        sample.extend([0.0] * (model_channels - len(sample)))
                    elif len(sample) > model_channels:
                        sample = sample[:model_channels]
                sample_buffer.append(sample)
                pending_samples += 1

                while pending_samples >= window_step and len(sample_buffer) == window_size:
                    if filter_obj is None:
                        pending_samples -= window_step
                        continue

                    window = np.asarray(sample_buffer, dtype=float)
                    filtered = apply_filters(filter_obj, window)
                    if neutral_mean is not None and mvc_scale is not None:
                        filtered = (filtered - neutral_mean) / mvc_scale
                    # Match libemg.get_windows output: (n_windows, channels, window_size).
                    windows = filtered.T[np.newaxis, :, :]
                    feature_dict = fe.extract_features(feature_order, windows)
                    X = flatten_feature_dict(
                        feature_dict,
                        feature_order,
                        channel_count=model_channels,
                    )
                    pred = bundle.predict(X)[0]

                    confidence = None
                    use_proba = args.min_confidence > 0.0 or args.show_confidence
                    if use_proba:
                        try:
                            proba = bundle.predict_proba(X)[0]
                            confidence = float(np.max(proba))
                        except AttributeError:
                            confidence = None

                    pred_history.append(pred)
                    label = _majority_vote(pred_history) if args.smoothing > 1 else pred

                    if confidence is not None and confidence < args.min_confidence:
                        label = args.low_confidence_label

                    control_hook(label)

                    if label != last_output or args.show_confidence:
                        if confidence is not None and args.show_confidence:
                            msg = f"Gesture: {label} (conf {confidence:.2f})"
                        else:
                            msg = f"Gesture: {label}"
                        if len(msg) < last_msg_len:
                            msg = msg.ljust(last_msg_len)
                        else:
                            last_msg_len = len(msg)
                        print(msg, end="\r", flush=True)
                        last_output = label

                    pending_samples -= window_step
    except KeyboardInterrupt:
        print("\nStopping stream...")
    finally:
        base.Stop_Callback()


if __name__ == "__main__":
    main()
