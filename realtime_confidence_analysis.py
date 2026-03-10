import argparse
import csv
import datetime as dt
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from AeroPy.TrignoBase import TrignoBase
from AeroPy.DataManager import DataKernel

import realtime_gesture_cnn as rt
from emg.gesture_model_cnn import load_cnn_bundle


def _build_label_to_idx(index_to_label):
    return {str(label): int(idx) for idx, label in index_to_label.items()}


def _safe_prob_map(probs, index_to_label):
    probs = np.asarray(probs, dtype=float).reshape(-1)
    out = {}
    for idx, label in index_to_label.items():
        i = int(idx)
        if 0 <= i < probs.size:
            out[str(label)] = float(probs[i])
    return out


def _derive_prompt_sequence(available_labels, user_sequence):
    if user_sequence:
        requested = [s.strip() for s in user_sequence.split(",") if s.strip()]
        seq = [s for s in requested if s in available_labels]
        if seq:
            return seq
    priority = ["neutral", "left_turn", "right_turn", "signal_left", "signal_right", "horn"]
    seq = [g for g in priority if g in available_labels]
    return seq if seq else sorted(available_labels)


def _pair_mapped_indices(channel_labels, right_channels, left_channels):
    right_pairs = {int(p) for p in (rt.RIGHT_ARM_PAIR_NUMBERS or set())}
    left_pairs = {int(p) for p in (rt.LEFT_ARM_PAIR_NUMBERS or set())}
    if not right_pairs and not left_pairs:
        return None, None

    right_idx = []
    left_idx = []
    for idx, label in enumerate(channel_labels):
        pair = rt._parse_pair_number(label)
        if pair in right_pairs:
            right_idx.append(idx)
        elif pair in left_pairs:
            left_idx.append(idx)

    right_idx = np.asarray(sorted(right_idx), dtype=int)
    left_idx = np.asarray(sorted(left_idx), dtype=int)
    if right_idx.size == right_channels and left_idx.size == left_channels:
        return right_idx, left_idx
    return None, None


def _calibration_vectors_from_filtered(arm_label, neutral_f, mvc_f):
    neutral_f = np.asarray(neutral_f, dtype=float)
    mvc_f = np.asarray(mvc_f, dtype=float)
    neutral_rms = np.sqrt(np.mean(neutral_f ** 2, axis=0))
    mvc_rms = np.sqrt(np.mean(mvc_f ** 2, axis=0))
    ratio = np.where(neutral_rms < 1e-9, 1.0, mvc_rms / neutral_rms)
    median_ratio = float(np.median(ratio))
    weak = int(np.sum(ratio < 2.0))
    print(
        f"MVC quality [{arm_label}]: {median_ratio:.1f}x median "
        f"({weak}/{ratio.size} channels below 2x)"
    )
    if median_ratio < float(rt.MVC_MIN_RATIO):
        print(
            f"[{arm_label}] Calibration skipped for normalization "
            f"(median ratio < {rt.MVC_MIN_RATIO})."
        )
        return None, None

    neutral_mean = np.mean(neutral_f, axis=0)
    mvc_scale = np.percentile(mvc_f, float(rt.MVC_PERCENTILE), axis=0)
    mvc_scale = np.where(mvc_scale < 1e-6, 1.0, mvc_scale)
    return neutral_mean, mvc_scale


def _collect_calibration_pair(
    handler,
    stream_channels,
    poll_sleep,
    arm_label,
    arm_indices,
    arm_channels,
    filter_obj,
):
    print(f"\nCalibration [{arm_label}]: RELAX for {rt.CALIB_NEUTRAL_S:.1f}s.")
    n_samples, _ = rt._collect_samples(
        handler, rt.CALIB_NEUTRAL_S, stream_channels, None, poll_sleep
    )
    if rt.CALIB_MVC_PREP_S > 0:
        print(f"Prepare: SQUEEZE {arm_label.upper()} arm in {rt.CALIB_MVC_PREP_S:.0f}s...")
        time.sleep(rt.CALIB_MVC_PREP_S)
    print(f"Calibration [{arm_label}]: SQUEEZE for {rt.CALIB_MVC_S:.1f}s.")
    m_samples, _ = rt._collect_samples(
        handler, rt.CALIB_MVC_S, stream_channels, None, poll_sleep
    )
    if n_samples.size == 0 or m_samples.size == 0:
        return None, None

    n_arr = rt._slice_channels_by_indices(n_samples, arm_indices, arm_channels)
    m_arr = rt._slice_channels_by_indices(m_samples, arm_indices, arm_channels)
    n_f = rt.apply_filters(filter_obj, n_arr)
    m_f = rt.apply_filters(filter_obj, m_arr)
    return _calibration_vectors_from_filtered(arm_label, n_f, m_f)


def _infer_one_arm(
    raw_buffer,
    filter_obj,
    bundle,
    index_to_label,
    allowed_indices,
    neutral_mean,
    mvc_scale,
    pred_history,
):
    raw = np.asarray(raw_buffer, dtype=float)
    filtered_full = rt.apply_filters(filter_obj, raw)
    filtered = filtered_full[-rt.WINDOW_SIZE:]
    if neutral_mean is not None and mvc_scale is not None:
        neutral_mean, mvc_scale = rt._align_calibration_vectors(
            neutral_mean, mvc_scale, filtered.shape[1], "arm"
        )
        filtered = (filtered - neutral_mean) / mvc_scale

    window = filtered.T[np.newaxis, :, :].astype(np.float32)
    window = bundle.standardize(window)
    probs = bundle.predict_proba(window)[0]
    probs = rt._restrict_probs(probs, allowed_indices)

    pred_history.append(probs)
    if rt.SMOOTHING > 1:
        probs = np.mean(np.stack(pred_history, axis=0), axis=0)

    label, conf, _ = rt._decode_prediction(probs, index_to_label)
    return str(label), float(conf), np.asarray(probs, dtype=float)


def _now_stamp():
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Realtime confidence analysis with prompted gestures and CSV logging."
    )
    parser.add_argument("--mode", choices=["right", "left", "dual"], default="dual")
    parser.add_argument("--model-right", default=rt.MODEL_RIGHT)
    parser.add_argument("--model-left", default=rt.MODEL_LEFT)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--segment-s", type=float, default=5.0)
    parser.add_argument("--log-interval-s", type=float, default=1.0)
    parser.add_argument("--sequence", default="", help="Comma-separated prompt labels.")
    parser.add_argument("--no-calibration", action="store_true", default=False)
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)

    if args.duration_s <= 0:
        raise ValueError("--duration-s must be > 0")
    if args.segment_s <= 0:
        raise ValueError("--segment-s must be > 0")
    if args.log_interval_s <= 0:
        raise ValueError("--log-interval-s must be > 0")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[analysis] device:", device)

    mode = str(args.mode)
    dual_arm = mode == "dual"
    if mode == "left":
        model_right_path = str(args.model_left)
    else:
        model_right_path = str(args.model_right)
    model_left_path: str | None = str(args.model_left) if dual_arm else None

    print("[analysis] right model:", model_right_path)
    bundle_right = load_cnn_bundle(model_right_path, device=device)
    index_to_label_right = rt._canonical_label_map(bundle_right.index_to_label)
    allowed_labels_right, allowed_indices_right = rt._resolve_allowed_labels(
        index_to_label_right, "right"
    )
    model_channels = bundle_right.channel_count

    bundle_left = None
    left_channels = 0
    index_to_label_left = {}
    allowed_labels_left = set()
    allowed_indices_left = set()
    if dual_arm:
        if model_left_path is None:
            raise ValueError("Dual-arm mode requires --model-left.")
        print("[analysis] left model:", model_left_path)
        bundle_left = load_cnn_bundle(model_left_path, device=device)
        left_channels = bundle_left.channel_count
        index_to_label_left = rt._canonical_label_map(bundle_left.index_to_label)
        allowed_labels_left, allowed_indices_left = rt._resolve_allowed_labels(
            index_to_label_left, "left"
        )

    all_labels = set(allowed_labels_right)
    if dual_arm:
        all_labels |= set(allowed_labels_left)
    prompt_sequence = _derive_prompt_sequence(all_labels, args.sequence)
    print("[analysis] prompt sequence:", prompt_sequence)

    handler = rt._StreamingHandler()
    base = TrignoBase(handler)
    handler.DataHandler = DataKernel(base)

    base.Connect_Callback()
    sensors = base.Scan_Callback()
    if not sensors:
        raise RuntimeError("No Trigno sensors found during scan.")

    base.start_trigger = False
    base.stop_trigger = False
    if not base.ConfigureCollectionOutput():
        raise RuntimeError("Failed to configure Trigno pipeline.")

    stream_channels = len(base.channel_guids)
    channel_labels = list(getattr(base, "emgChannelNames", []))
    if len(channel_labels) != stream_channels:
        channel_labels = [f"ch{idx}" for idx in range(stream_channels)]

    # Channel mapping
    left_arm_start = 0
    right_channel_indices = np.arange(min(rt.RIGHT_ARM_CHANNELS, stream_channels), dtype=int)
    left_channel_indices = np.arange(
        rt.RIGHT_ARM_CHANNELS,
        min(rt.RIGHT_ARM_CHANNELS + left_channels, stream_channels),
        dtype=int,
    )
    if dual_arm:
        mapped_r, mapped_l = _pair_mapped_indices(
            channel_labels, rt.RIGHT_ARM_CHANNELS, left_channels
        )
        if mapped_r is not None and mapped_l is not None:
            right_channel_indices = mapped_r
            left_channel_indices = mapped_l
            print(f"[analysis] fixed pair mapping right: {right_channel_indices.tolist()}")
            print(f"[analysis] fixed pair mapping left:  {left_channel_indices.tolist()}")
        else:
            print("[analysis] using contiguous dual mapping fallback.")
    else:
        if mode == "left" and stream_channels >= rt.RIGHT_ARM_CHANNELS + model_channels:
            left_arm_start = rt.RIGHT_ARM_CHANNELS
            print(
                f"[analysis] left mode with all sensors connected, offset={left_arm_start}"
            )

    # Sampling + filter setup
    runtime_target_fs: float | None = (
        float(rt.REALTIME_TARGET_FS_HZ)
        if rt.REALTIME_TARGET_FS_HZ is not None
        else None
    )
    use_timestamp_resampling = bool(
        rt.REALTIME_RESAMPLE and runtime_target_fs is not None and runtime_target_fs > 0
    )
    base.TrigBase.Start(handler.streamYTData)
    fs: float | None
    if use_timestamp_resampling:
        assert runtime_target_fs is not None
        fs = float(runtime_target_fs)
    else:
        fs = None
    filter_obj = rt.define_filters(fs) if fs is not None else None
    resampler = rt._RealtimeTimestampResampler(stream_channels, fs) if fs is not None else None
    if use_timestamp_resampling:
        print(f"[analysis] timestamp resampling enabled @ {fs:.2f} Hz")
    else:
        print("[analysis] timestamp resampling disabled.")

    poll_sleep = 0.001

    # Optional calibration
    neutral_mean_right = None
    mvc_scale_right = None
    neutral_mean_left = None
    mvc_scale_left = None
    if (not args.no_calibration) and rt.CALIBRATE:
        if filter_obj is None:
            # Fallback filter init from first collected timestamps
            tmp_samples, tmp_times = rt._collect_samples(
                handler, 1.5, stream_channels, None, poll_sleep
            )
            if tmp_times.size:
                fs_est = rt._estimate_fs(tmp_times)
                if fs_est is not None:
                    fs = float(fs_est)
                    filter_obj = rt.define_filters(fs)
                    print(f"[analysis] estimated fs: {fs:.2f} Hz")
            if filter_obj is None and tmp_samples.size:
                filter_obj = rt.define_filters(2000.0)
                print("[analysis] fallback filter fs=2000.0 Hz")

        if dual_arm:
            neutral_mean_right, mvc_scale_right = _collect_calibration_pair(
                handler,
                stream_channels,
                poll_sleep,
                "right",
                right_channel_indices,
                rt.RIGHT_ARM_CHANNELS,
                filter_obj,
            )
            neutral_mean_left, mvc_scale_left = _collect_calibration_pair(
                handler,
                stream_channels,
                poll_sleep,
                "left",
                left_channel_indices,
                left_channels,
                filter_obj,
            )
        else:
            arm_indices = np.arange(left_arm_start, left_arm_start + model_channels, dtype=int)
            neutral_mean_right, mvc_scale_right = _collect_calibration_pair(
                handler,
                stream_channels,
                poll_sleep,
                mode,
                arm_indices,
                model_channels,
                filter_obj,
            )

    if filter_obj is None:
        filter_obj = rt.define_filters(2000.0)
        print("[analysis] fallback filter initialized at 2000.0 Hz.")

    # Realtime buffers
    buffer_len = rt.WINDOW_SIZE + rt.FILTER_WARMUP
    raw_buffer_right = deque(maxlen=buffer_len)
    raw_buffer_left = deque(maxlen=buffer_len)
    pending_right = 0
    pending_left = 0
    warmup_right = rt.FILTER_WARMUP == 0
    warmup_left = rt.FILTER_WARMUP == 0

    pred_history_right = deque(maxlen=max(1, int(rt.SMOOTHING)))
    pred_history_left = deque(maxlen=max(1, int(rt.SMOOTHING)))

    latest = {
        "right_label": rt.LOW_CONFIDENCE_LABEL,
        "right_conf": 0.0,
        "right_probs": np.zeros(len(index_to_label_right), dtype=float),
        "left_label": rt.LOW_CONFIDENCE_LABEL,
        "left_conf": 0.0,
        "left_probs": np.zeros(len(index_to_label_left), dtype=float),
        "fused_label": rt.LOW_CONFIDENCE_LABEL,
        "fused_conf": 0.0,
        "fused_probs": {},
    }

    rows = []
    start_t = time.time()
    next_log_t = start_t
    current_prompt = None
    segment_index = -1

    try:
        while True:
            now = time.time()
            elapsed = now - start_t
            if elapsed >= args.duration_s:
                break

            idx = int(elapsed // args.segment_s) % max(len(prompt_sequence), 1)
            if idx != segment_index and prompt_sequence:
                segment_index = idx
                current_prompt = prompt_sequence[idx]
                print(f"\n[analysis] NOW PERFORM: {current_prompt}")

            out = handler.DataHandler.GetYTData()
            if out is None:
                time.sleep(poll_sleep)
                continue

            channel_times, channel_values = rt._parse_yt_frame(out)
            if len(channel_values) < stream_channels:
                continue

            if use_timestamp_resampling:
                assert resampler is not None
                batch_arr, _ = resampler.push(channel_times, channel_values)
                if batch_arr.size == 0:
                    continue
            else:
                sample_count = min(len(c) for c in channel_values)
                if sample_count == 0:
                    continue
                sample_batch = []
                for i in range(sample_count):
                    sample_batch.append([channel_values[ch][i] for ch in range(stream_channels)])
                batch_arr = np.asarray(sample_batch, dtype=float)

            if dual_arm:
                batch_right = rt._slice_channels_by_indices(
                    batch_arr, right_channel_indices, rt.RIGHT_ARM_CHANNELS
                )
                batch_left = rt._slice_channels_by_indices(
                    batch_arr, left_channel_indices, left_channels
                )
            else:
                batch_right = rt._slice_channels(batch_arr, left_arm_start, model_channels)
                batch_left = np.empty((0, 0), dtype=float)

            for s in batch_right:
                raw_buffer_right.append(s)
                pending_right += 1
            if (not warmup_right) and len(raw_buffer_right) >= buffer_len:
                warmup_right = True
                pending_right = rt.WINDOW_STEP

            if dual_arm:
                for s in batch_left:
                    raw_buffer_left.append(s)
                    pending_left += 1
                if (not warmup_left) and len(raw_buffer_left) >= buffer_len:
                    warmup_left = True
                    pending_left = rt.WINDOW_STEP

            if warmup_right and pending_right >= rt.WINDOW_STEP and len(raw_buffer_right) >= rt.WINDOW_SIZE:
                r_label, r_conf, r_probs = _infer_one_arm(
                    raw_buffer_right,
                    filter_obj,
                    bundle_right,
                    index_to_label_right,
                    allowed_indices_right,
                    neutral_mean_right,
                    mvc_scale_right,
                    pred_history_right,
                )
                latest["right_label"] = r_label
                latest["right_conf"] = r_conf
                latest["right_probs"] = r_probs
                pending_right = 0

            if dual_arm and warmup_left and pending_left >= rt.WINDOW_STEP and len(raw_buffer_left) >= rt.WINDOW_SIZE:
                assert bundle_left is not None
                l_label, l_conf, l_probs = _infer_one_arm(
                    raw_buffer_left,
                    filter_obj,
                    bundle_left,
                    index_to_label_left,
                    allowed_indices_left,
                    neutral_mean_left,
                    mvc_scale_left,
                    pred_history_left,
                )
                latest["left_label"] = l_label
                latest["left_conf"] = l_conf
                latest["left_probs"] = l_probs
                pending_left = 0

            if dual_arm:
                fused_label, fused_conf = rt.fuse_predictions(
                    latest["right_label"],
                    latest["right_conf"],
                    latest["left_label"],
                    latest["left_conf"],
                )
                latest["fused_label"] = fused_label
                latest["fused_conf"] = fused_conf
                r_map = _safe_prob_map(latest["right_probs"], index_to_label_right)
                l_map = _safe_prob_map(latest["left_probs"], index_to_label_left)
                fused_map = {}
                for lbl in sorted(all_labels):
                    rv = float(r_map.get(lbl, 0.0))
                    lv = float(l_map.get(lbl, 0.0))
                    fused_map[lbl] = 0.5 * (rv + lv)
                s = sum(fused_map.values())
                if s > 0:
                    fused_map = {k: v / s for k, v in fused_map.items()}
                latest["fused_probs"] = fused_map
            else:
                latest["fused_label"] = latest["right_label"]
                latest["fused_conf"] = latest["right_conf"]
                latest["fused_probs"] = _safe_prob_map(latest["right_probs"], index_to_label_right)

            if now >= next_log_t:
                next_log_t += args.log_interval_s
                row = {
                    "timestamp": dt.datetime.now().isoformat(),
                    "elapsed_s": round(elapsed, 3),
                    "prompt_label": current_prompt if current_prompt is not None else "",
                    "mode": mode,
                    "pred_label": latest["fused_label"],
                    "pred_conf": float(latest["fused_conf"]),
                    "right_label": latest["right_label"],
                    "right_conf": float(latest["right_conf"]),
                    "left_label": latest["left_label"] if dual_arm else "",
                    "left_conf": float(latest["left_conf"]) if dual_arm else 0.0,
                }
                for lbl in sorted(all_labels):
                    if dual_arm:
                        r_map = _safe_prob_map(latest["right_probs"], index_to_label_right)
                        l_map = _safe_prob_map(latest["left_probs"], index_to_label_left)
                        row[f"right_{lbl}"] = float(r_map.get(lbl, 0.0))
                        row[f"left_{lbl}"] = float(l_map.get(lbl, 0.0))
                    row[f"fused_{lbl}"] = float(latest["fused_probs"].get(lbl, 0.0))

                rows.append(row)

                fused_text = " ".join(
                    [f"{lbl}:{row[f'fused_{lbl}']:.2f}" for lbl in sorted(all_labels)]
                )
                print(
                    f"[{elapsed:6.1f}s] prompt={row['prompt_label']:<12} "
                    f"pred={row['pred_label']:<12} conf={row['pred_conf']:.2f} | {fused_text}"
                )

    except KeyboardInterrupt:
        print("\n[analysis] Interrupted by user.")
    finally:
        base.Stop_Callback()

    out_path = args.output.strip()
    if not out_path:
        out_dir = Path("tmp")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / f"realtime_confidence_analysis_{_now_stamp()}.csv")
    out_path = str(Path(out_path))

    if not rows:
        print("[analysis] No rows collected; nothing saved.")
        return

    fieldnames = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[analysis] Saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
