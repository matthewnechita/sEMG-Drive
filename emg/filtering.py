import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfilt, sosfiltfilt


# LOAD IN THE RAW EMG DATA
def load_emg_data(file_path):
    """
    Load the raw EMG data from the .npz file
    Format of the .npz file {"emg": array, "fs": sampling_rate} or {"X": array, "timestamps": array}
    """
    data = np.load(file_path, allow_pickle=True)
    extras = {}

    if "emg" in data:
        emg = data["emg"]
        fs = data["fs"]
        # carry any additional keys through to the filtered file
        extras = {k: data[k] for k in data.files if k not in ["emg", "fs"]}
    else:
        emg = data["X"]
        timestamps = data.get("timestamps")
        y = data.get("y")
        events = data.get("events")
        metadata = data.get("metadata")
        extras = {
            "timestamps": timestamps,
            "y": y,
            "events": events,
            "metadata": metadata,
        }
        for key in data.files:
            if key.startswith("calib_"):
                extras[key] = data.get(key)
        # derive sampling rate from timestamp spacing if not present
        if "fs" in data:
            fs = data["fs"]
        elif isinstance(metadata, dict) and "fs" in metadata:
            fs = metadata["fs"]
        elif timestamps is not None:
            step = np.median(np.diff(timestamps[:, 0]))
            fs = 1.0 / step if step else None
        else:
            fs = None

    return emg, fs, extras


# DEFINE THE FILTERS
def define_filters(fs):
    """
    Returns a tuple of SOS filter arrays (sos_n60, sos_n120, sos_bp):
    - Notch @ 60 Hz  (±1.5 Hz, power line fundamental)
    - Notch @ 120 Hz (±1.5 Hz, 2nd power line harmonic)
    - Bandpass 25–450 Hz, order 6

    IMPORTANT: after changing any parameter here you must:
      1. Delete all data/**/*_filtered.npz files
      2. python emg/filtering.py          (re-filter raw data + calibration)
      3. python tools/recalibrate.py --apply
      4. Retrain both models
    """
    sos_n60  = butter(2, [58.5,  61.5],  btype='bandstop', fs=fs, output='sos')
    sos_n120 = butter(2, [118.5, 121.5], btype='bandstop', fs=fs, output='sos')
    sos_bp   = butter(6, [25.0,  450.0], btype='bandpass', fs=fs, output='sos')
    return (sos_n60, sos_n120, sos_bp)


# APPLY THE FILTERS TO THE EMG DATA (offline / bulk use)
def apply_filters(filters, emg):
    """
    Zero-phase offline filtering for full session recordings. emg: (N, C).
    Uses sosfiltfilt so boundary effects are negligible on long recordings.
    Do NOT use this in the realtime loop — use apply_filters_stateful instead.
    """
    sos_n60, sos_n120, sos_bp = filters
    out = sosfiltfilt(sos_n60,  emg, axis=0)
    out = sosfiltfilt(sos_n120, out, axis=0)
    out = sosfiltfilt(sos_bp,   out, axis=0)
    # rectified = np.abs(out)  # rectification
    return np.array(out, dtype=float)


def make_filter_state(filters, num_channels):
    """
    Create zero initial conditions for stateful realtime filtering.
    Call once after define_filters(), before the inference loop starts.
    """
    sos_n60, sos_n120, sos_bp = filters
    return [
        np.zeros((sos_n60.shape[0],  2, num_channels)),
        np.zeros((sos_n120.shape[0], 2, num_channels)),
        np.zeros((sos_bp.shape[0],   2, num_channels)),
    ]


def apply_filters_stateful(filters, samples, state):
    """
    Causal stateful filtering for the realtime inference loop.
    samples: (N, C) array of new raw samples.
    state:   list of zi arrays from make_filter_state() or a previous call.
    Returns: (filtered_samples, new_state)
      filtered_samples: (N, C) — feed directly into the prediction buffer.
      new_state: updated zi arrays to pass into the next call.
    Eliminates the train/test filter mismatch caused by applying filtfilt
    to short rolling buffers in realtime vs. full sessions offline.
    """
    sos_n60, sos_n120, sos_bp = filters
    zi_n60, zi_n120, zi_bp = state
    out, zi_n60  = sosfilt(sos_n60,  samples, axis=0, zi=zi_n60)
    out, zi_n120 = sosfilt(sos_n120, out,     axis=0, zi=zi_n120)
    out, zi_bp   = sosfilt(sos_bp,   out,     axis=0, zi=zi_bp)
    return np.array(out, dtype=float), [zi_n60, zi_n120, zi_bp]


# SAVE THE FILTERED DATA INTO A .npz FILE
def save_filtered_data(output_path, filtered, fs, extras):
    """
    This method saves the filtered data into a .npz file for later use
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, emg=filtered, fs=fs, **extras)


def destination_for_filtered(raw_path: Path) -> Path:
    """
    Place filtered files in ../filtered/ when raw files are in ../raw/.
    Fallback: same directory as the raw file.
    """
    if raw_path.parent.name == "raw":
        base_dir = raw_path.parent.parent
        return base_dir / "filtered" / (raw_path.stem.replace("_raw", "") + "_filtered.npz")
    return raw_path.with_name(raw_path.stem.replace("_raw", "") + "_filtered.npz")


if __name__ == "__main__":
    root = Path("data")

    for fp in root.rglob("*_raw.npz"):
        out_path = destination_for_filtered(fp)
        if out_path.exists():
            print(f"Skipping {fp} (filtered exists at {out_path})")
            continue
        emg, fs, extras = load_emg_data(fp)
        if fs is None:
            print(f"Skipping {fp} (no sampling rate found)")
            continue
        fi = define_filters(fs)
        filtered_emg = apply_filters(fi, emg)

        calib_neutral = extras.get("calib_neutral_X")
        if calib_neutral is not None:
            calib_neutral = np.asarray(calib_neutral, dtype=float)
            if calib_neutral.size:
                extras["calib_neutral_emg"] = apply_filters(fi, calib_neutral)
            extras.pop("calib_neutral_X", None)

        calib_mvc = extras.get("calib_mvc_X")
        if calib_mvc is not None:
            calib_mvc = np.asarray(calib_mvc, dtype=float)
            if calib_mvc.size:
                extras["calib_mvc_emg"] = apply_filters(fi, calib_mvc)
            extras.pop("calib_mvc_X", None)

        save_filtered_data(out_path, filtered_emg, fs, extras)
        print(f"Wrote {out_path}")
