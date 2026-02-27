import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfilt, sosfiltfilt
from libemg import filtering as libemg_filter


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


# DEFINE THE FILTERS USING libEMG
def define_filters(fs):
    """
    Create the filter object and install:
    - Notch filter @ 60 Hz
    - Bandpass filter @ 20-450 Hz
    """

    fi = libemg_filter.Filter(fs)

    notch = {"name": "notch", "cutoff": 60, "bandwidth": 3}
    bandpass = {"name": "bandpass", "cutoff": [20, 450], "order": 4}

    fi.install_filters(notch)
    fi.install_filters(bandpass)

    return fi


# APPLY THE FILTERS TO THE EMG DATA
def apply_filters(fi, emg):
    """
    This method applies the defined filters onto the raw EMG data
    """
    filtered_data = fi.filter(emg)
    # rectified = np.abs(filtered_data)  # rectification
    return np.array(filtered_data)


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
