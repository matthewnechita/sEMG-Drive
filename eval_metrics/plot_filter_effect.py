import argparse
from pathlib import Path

import numpy as np


def _load_arrays(raw_path: Path, filtered_path: Path):
    raw = np.load(raw_path, allow_pickle=True)
    filtered = np.load(filtered_path, allow_pickle=True)

    if "X" not in raw.files:
        raise KeyError(f"{raw_path} is missing raw array 'X'")
    if "emg" not in filtered.files:
        raise KeyError(f"{filtered_path} is missing filtered array 'emg'")

    raw_emg = np.asarray(raw["X"], dtype=float)
    filtered_emg = np.asarray(filtered["emg"], dtype=float)
    if raw_emg.shape != filtered_emg.shape:
        raise ValueError(
            f"Raw/filtered shape mismatch: {raw_emg.shape} vs {filtered_emg.shape}"
        )
    return raw, filtered, raw_emg, filtered_emg


def _time_axis_ms(raw_npz, filtered_npz, sample_count: int, channel: int):
    if "timestamps" in filtered_npz.files:
        ts = np.asarray(filtered_npz["timestamps"], dtype=float)
        if ts.ndim == 2 and ts.shape[1] > channel:
            base = ts[:, channel]
            base = base - base[0]
            return base * 1000.0
        if ts.ndim == 1:
            base = ts - ts[0]
            return base * 1000.0
    if "timestamps" in raw_npz.files:
        ts = np.asarray(raw_npz["timestamps"], dtype=float)
        if ts.ndim == 2 and ts.shape[1] > channel:
            base = ts[:, channel]
            base = base - base[0]
            return base * 1000.0
        if ts.ndim == 1:
            base = ts - ts[0]
            return base * 1000.0
    fs = float(np.asarray(filtered_npz["fs"]).squeeze()) if "fs" in filtered_npz.files else 2000.0
    return np.arange(sample_count, dtype=float) / fs * 1000.0


def _estimate_lag_ms(raw_signal, filtered_signal, fs_hz: float):
    raw_centered = raw_signal - np.mean(raw_signal)
    filtered_centered = filtered_signal - np.mean(filtered_signal)
    corr = np.correlate(filtered_centered, raw_centered, mode="full")
    lags = np.arange(-len(raw_signal) + 1, len(raw_signal))
    lag_samples = int(lags[int(np.argmax(corr))])
    return float(lag_samples) / float(fs_hz) * 1000.0


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot raw vs filtered EMG and PSD comparison for one session."
    )
    parser.add_argument("--raw", type=Path, required=True, help="Path to *_raw.npz")
    parser.add_argument("--filtered", type=Path, required=True, help="Path to *_filtered.npz")
    parser.add_argument("--channel", type=int, default=0, help="0-based channel index")
    parser.add_argument("--start-ms", type=float, default=0.0, help="Plot start time in ms")
    parser.add_argument("--duration-ms", type=float, default=300.0, help="Plot duration in ms")
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("eval_metrics") / "out" / "filter_effect.png",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    raw_npz, filtered_npz, raw_emg, filtered_emg = _load_arrays(args.raw, args.filtered)

    if args.channel < 0 or args.channel >= raw_emg.shape[1]:
        raise ValueError(f"--channel out of range for {raw_emg.shape[1]} channels")

    time_ms = _time_axis_ms(raw_npz, filtered_npz, raw_emg.shape[0], args.channel)
    fs_hz = float(np.asarray(filtered_npz["fs"]).squeeze()) if "fs" in filtered_npz.files else 2000.0

    start_ms = float(args.start_ms)
    end_ms = start_ms + float(args.duration_ms)
    mask = (time_ms >= start_ms) & (time_ms <= end_ms)
    if not np.any(mask):
        raise ValueError("Requested time window does not overlap the signal")

    raw_signal = raw_emg[:, args.channel]
    filtered_signal = filtered_emg[:, args.channel]
    lag_ms = _estimate_lag_ms(raw_signal, filtered_signal, fs_hz)

    from matplotlib import pyplot as plt
    from scipy.signal import welch

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

    axes[0].plot(time_ms[mask], raw_signal[mask], label="Raw", linewidth=1.0)
    axes[0].plot(time_ms[mask], filtered_signal[mask], label="Filtered", linewidth=1.0)
    axes[0].set_title(
        f"Raw vs filtered EMG, ch {args.channel} "
        f"(window {start_ms:.0f}-{end_ms:.0f} ms, lag≈{lag_ms:.3f} ms)"
    )
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()

    freq_raw, psd_raw = welch(raw_signal, fs=fs_hz, nperseg=min(4096, raw_signal.size))
    freq_filt, psd_filt = welch(filtered_signal, fs=fs_hz, nperseg=min(4096, filtered_signal.size))
    axes[1].semilogy(freq_raw, psd_raw, label="Raw PSD", linewidth=1.0)
    axes[1].semilogy(freq_filt, psd_filt, label="Filtered PSD", linewidth=1.0)
    axes[1].set_title("Power spectral density")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("PSD")
    axes[1].legend()

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=180)
    print(f"Saved figure to {args.output_png}")


if __name__ == "__main__":
    main()
