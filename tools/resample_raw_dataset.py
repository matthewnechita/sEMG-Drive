"""
Resample raw EMG dataset to a single common sampling rate using per-channel timestamps.

Why this exists:
- Mixed Trigno sensor types can produce different per-channel sampling rates.
- Current pipelines assume aligned channels by row index; this introduces time skew
  when rates differ.
- This script builds one shared time grid and interpolates each channel onto it.

Input format (from data collection GUI):
- .npz with keys: X, timestamps, y, events, metadata, and optional calibration arrays.

Output:
- Writes new .npz files with the same relative layout under OUTPUT_ROOT.
- Keeps original files untouched by default.

Usage:
- Edit config constants below.
- Run: python tools/resample_raw_dataset.py

  Order should be:

  1. python tools/resample_raw_dataset.py
  2. python emg/filtering.py
  3. python tools/recalibrate.py --data-root <your_resampledbtw _root> (dry run)
  4. If you see failed sessions, run:
     python tools/recalibrate.py --data-root <your_resampled_root> --apply
  5. Retrain.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np


# ======== Config (edit in code) ========
DATA_ROOT = Path("data")
OUTPUT_ROOT = Path("data_resampled")
RAW_PATTERN = "*_raw.npz"
TARGET_FS_HZ = 2000.0
OVERWRITE = False
MAX_FILES = None  # set to int for a quick smoke run, e.g. 2
# =======================================


def _estimate_fs_per_channel(timestamps: np.ndarray) -> list[float]:
    fs_values: list[float] = []
    for ch in range(timestamps.shape[1]):
        t = np.asarray(timestamps[:, ch], dtype=float)
        diffs = np.diff(t)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size == 0:
            fs_values.append(float("nan"))
        else:
            fs_values.append(float(1.0 / np.median(diffs)))
    return fs_values


def _to_python_object(value):
    if isinstance(value, np.ndarray) and value.shape == () and value.dtype == object:
        return value.item()
    return value


def _monotonic_unique_time_series(t: np.ndarray, x: np.ndarray):
    t = np.asarray(t, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)
    if t.size != x.size:
        n = min(t.size, x.size)
        t = t[:n]
        x = x[:n]

    mask = np.isfinite(t) & np.isfinite(x)
    t = t[mask]
    x = x[mask]
    if t.size < 2:
        return t, x

    order = np.argsort(t, kind="mergesort")
    t = t[order]
    x = x[order]

    unique_t, unique_idx = np.unique(t, return_index=True)
    x = x[unique_idx]
    return unique_t, x


def _build_common_grid(timestamps: np.ndarray, target_fs_hz: float):
    starts = []
    ends = []
    for ch in range(timestamps.shape[1]):
        t = np.asarray(timestamps[:, ch], dtype=float).reshape(-1)
        t = t[np.isfinite(t)]
        if t.size < 2:
            continue
        starts.append(float(np.min(t)))
        ends.append(float(np.max(t)))

    if not starts or not ends:
        raise ValueError("No valid timestamp ranges found.")

    t_start = max(starts)
    t_end = min(ends)
    if not np.isfinite(t_start) or not np.isfinite(t_end) or t_end <= t_start:
        raise ValueError(
            f"Invalid overlap window across channels (start={t_start}, end={t_end})."
        )

    n_samples = int(np.floor((t_end - t_start) * target_fs_hz)) + 1
    if n_samples < 2:
        raise ValueError(
            f"Overlap window too short for target fs ({target_fs_hz} Hz): {n_samples} sample(s)."
        )

    t_grid = t_start + (np.arange(n_samples, dtype=float) / float(target_fs_hz))
    return t_grid, t_start, t_end


def _resample_matrix(x: np.ndarray, timestamps: np.ndarray, t_grid: np.ndarray):
    x = np.asarray(x, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    if x.ndim != 2 or timestamps.ndim != 2:
        raise ValueError("Expected 2D arrays for X and timestamps.")
    if x.shape != timestamps.shape:
        raise ValueError(f"Shape mismatch X{tuple(x.shape)} vs timestamps{tuple(timestamps.shape)}")

    n_out = t_grid.size
    n_ch = x.shape[1]
    out = np.empty((n_out, n_ch), dtype=np.float64)

    for ch in range(n_ch):
        t_ch, x_ch = _monotonic_unique_time_series(timestamps[:, ch], x[:, ch])
        if t_ch.size < 2:
            raise ValueError(f"Channel {ch} has insufficient valid timestamps for interpolation.")
        out[:, ch] = np.interp(t_grid, t_ch, x_ch)

    ts_out = np.tile(t_grid.reshape(-1, 1), (1, n_ch))
    return out, ts_out


def _resample_labels_nearest(y: np.ndarray, ref_times: np.ndarray, t_grid: np.ndarray):
    y = np.asarray(y, dtype=object).reshape(-1)
    ref_times = np.asarray(ref_times, dtype=float).reshape(-1)
    n = min(y.size, ref_times.size)
    y = y[:n]
    ref_times = ref_times[:n]

    mask = np.isfinite(ref_times)
    if not np.any(mask):
        return np.asarray([None] * t_grid.size, dtype=object)

    ref_times = ref_times[mask]
    y = y[mask]

    order = np.argsort(ref_times, kind="mergesort")
    ref_times = ref_times[order]
    y = y[order]

    unique_t, unique_idx = np.unique(ref_times, return_index=True)
    ref_times = unique_t
    y = y[unique_idx]

    if ref_times.size == 0:
        return np.asarray([None] * t_grid.size, dtype=object)
    if ref_times.size == 1:
        return np.asarray([y[0]] * t_grid.size, dtype=object)

    idx_right = np.searchsorted(ref_times, t_grid, side="left")
    idx_left = np.clip(idx_right - 1, 0, ref_times.size - 1)
    idx_right = np.clip(idx_right, 0, ref_times.size - 1)

    t_left = ref_times[idx_left]
    t_right = ref_times[idx_right]
    choose_right = np.abs(t_right - t_grid) < np.abs(t_left - t_grid)
    idx = np.where(choose_right, idx_right, idx_left)
    return y[idx].astype(object, copy=False)


def _resample_optional_segment(data_dict: dict, x_key: str, ts_key: str, target_fs_hz: float):
    if x_key not in data_dict or ts_key not in data_dict:
        return
    x_seg = np.asarray(data_dict[x_key], dtype=float)
    ts_seg = np.asarray(data_dict[ts_key], dtype=float)
    if x_seg.size == 0 or ts_seg.size == 0:
        return
    if x_seg.ndim != 2 or ts_seg.ndim != 2 or x_seg.shape != ts_seg.shape:
        return
    t_grid, _, _ = _build_common_grid(ts_seg, target_fs_hz)
    x_out, ts_out = _resample_matrix(x_seg, ts_seg, t_grid)
    data_dict[x_key] = x_out
    data_dict[ts_key] = ts_out


def _resample_file(src: Path, dst: Path, target_fs_hz: float) -> tuple[int, int]:
    with np.load(src, allow_pickle=True) as data:
        payload = {k: data[k] for k in data.files}

    if "X" not in payload or "timestamps" not in payload or "y" not in payload:
        raise ValueError("Missing required keys (X, timestamps, y).")

    x_in = np.asarray(payload["X"], dtype=float)
    ts_in = np.asarray(payload["timestamps"], dtype=float)
    y_in = np.asarray(payload["y"], dtype=object)

    if x_in.ndim != 2 or ts_in.ndim != 2:
        raise ValueError("X and timestamps must be 2D.")
    if x_in.shape != ts_in.shape:
        raise ValueError(f"Shape mismatch X{tuple(x_in.shape)} vs timestamps{tuple(ts_in.shape)}")

    t_grid, t_start, t_end = _build_common_grid(ts_in, target_fs_hz)
    x_out, ts_out = _resample_matrix(x_in, ts_in, t_grid)

    # Labels are tied to sample times from acquisition; remap labels to new grid
    # using nearest-neighbor transfer from channel-0 timestamps.
    y_out = _resample_labels_nearest(y_in, ts_in[:, 0], t_grid)

    payload["X"] = x_out
    payload["timestamps"] = ts_out
    payload["y"] = y_out
    payload["fs"] = np.asarray(float(target_fs_hz))

    _resample_optional_segment(payload, "calib_neutral_X", "calib_neutral_timestamps", target_fs_hz)
    _resample_optional_segment(payload, "calib_mvc_X", "calib_mvc_timestamps", target_fs_hz)

    metadata = _to_python_object(payload.get("metadata", {}))
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {"original_metadata": metadata}

    fs_per_ch = _estimate_fs_per_channel(ts_in)
    metadata["resampling"] = {
        "enabled": True,
        "method": "per_channel_linear_interp",
        "target_fs_hz": float(target_fs_hz),
        "source_fs_hz_per_channel": fs_per_ch,
        "overlap_window_start_s": float(t_start),
        "overlap_window_end_s": float(t_end),
        "samples_in": int(x_in.shape[0]),
        "samples_out": int(x_out.shape[0]),
        "created_at": dt.datetime.now().isoformat(),
    }
    payload["metadata"] = metadata

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(str(dst) + ".tmp.npz")
    np.savez_compressed(tmp_path, **payload)
    tmp_path.replace(dst)
    return int(x_in.shape[0]), int(x_out.shape[0])


def main():
    if TARGET_FS_HZ <= 0:
        raise ValueError("TARGET_FS_HZ must be > 0.")

    files = sorted(DATA_ROOT.rglob(RAW_PATTERN))
    if MAX_FILES is not None:
        files = files[: int(MAX_FILES)]
    if not files:
        raise FileNotFoundError(f"No files matching {RAW_PATTERN!r} under {DATA_ROOT}")

    print(f"Input root:  {DATA_ROOT}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Target fs:   {TARGET_FS_HZ:.3f} Hz")
    print(f"Files:       {len(files)}")

    ok = 0
    skipped = 0
    failed = 0
    total_in = 0
    total_out = 0

    for src in files:
        rel = src.relative_to(DATA_ROOT)
        dst = OUTPUT_ROOT / rel

        if dst.exists() and not OVERWRITE:
            print(f"[skip] {rel} (already exists)")
            skipped += 1
            continue

        try:
            n_in, n_out = _resample_file(src, dst, TARGET_FS_HZ)
            total_in += n_in
            total_out += n_out
            ok += 1
            ratio = (n_out / n_in) if n_in else float("nan")
            print(f"[ok]   {rel} | {n_in} -> {n_out} samples ({ratio:.3f}x)")
        except Exception as exc:
            failed += 1
            print(f"[fail] {rel} | {exc}")

    print("\nSummary:")
    print(f"  success: {ok}")
    print(f"  skipped: {skipped}")
    print(f"  failed:  {failed}")
    if ok > 0:
        print(f"  samples: {total_in} -> {total_out} ({(total_out / total_in):.3f}x)")


if __name__ == "__main__":
    main()
