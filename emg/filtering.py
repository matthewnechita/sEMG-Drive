import numpy as np
from pathlib import Path
import sys
from libemg import filtering as libemg_filter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_paths import STRICT_RESAMPLED_ROOT


def _coerce_scalar_fs(value):
    """
    Convert fs payloads (Python scalar, 0-d array, size-1 array) to float.
    Returns None when conversion is not possible.
    """
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if arr.size != 1:
        return None
    fs = float(arr[0])
    if not np.isfinite(fs):
        return None
    return fs


# -- Loading -----------------------------------------------------------------
def load_emg_data(file_path):
    """
    Load the raw EMG data from the .npz file
    Format of the .npz file {"emg": array, "fs": sampling_rate} or {"X": array, "timestamps": array}
    """
    data = np.load(file_path, allow_pickle=True)
    extras = {}

    if "emg" in data:
        emg = data["emg"]
        fs = _coerce_scalar_fs(data.get("fs"))
        # Carry metadata and auxiliary arrays through untouched.
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
        # Resampled raw files may not carry a scalar fs field yet.
        if "fs" in data:
            fs = _coerce_scalar_fs(data.get("fs"))
        elif isinstance(metadata, dict) and "fs" in metadata:
            fs = _coerce_scalar_fs(metadata.get("fs"))
        elif timestamps is not None:
            step = np.median(np.diff(timestamps[:, 0]))
            fs = 1.0 / step if step else None
        else:
            fs = None

    return emg, _coerce_scalar_fs(fs), extras


# -- Filter definition -------------------------------------------------------
def define_filters(fs):
    """
    Create the filter object and install:
    - Notch filter @ 60 Hz  (power line fundamental)
    - Notch filter @ 120 Hz (2nd power line harmonic)
    - Bandpass filter 20-450 Hz, order 6

    IMPORTANT: after changing any parameter here you must:
      1. Delete all data_resampled_strict/**/*_filtered.npz files
      2. python emg/filtering.py          (re-filter raw data + calibration)
      3. Retrain the affected models
    """
    fi = libemg_filter.Filter(fs)
    fi.install_filters({"name": "notch",    "cutoff": 60,        "bandwidth": 3})
    fi.install_filters({"name": "notch",    "cutoff": 120,       "bandwidth": 3})
    fi.install_filters({"name": "bandpass", "cutoff": [20, 450], "order": 6})
    return fi


# -- Filtering ---------------------------------------------------------------
def apply_filters(fi, emg):
    """
    This method applies the defined filters onto the raw EMG data
    """
    filtered_data = fi.filter(emg)
    return np.array(filtered_data, dtype=float)


# -- Saving ------------------------------------------------------------------
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
    root = STRICT_RESAMPLED_ROOT

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
                # Filter calibration segments with the same chain as the training
                # stream so MVC normalization uses comparable signal statistics.
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
