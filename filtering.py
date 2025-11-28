import numpy as np
from pathlib import Path
from libemg import filtering

# LOAD IN THE RAW EMG DATA
def load_emg_data(file_path):
    '''
    Load the raw EMG data from the .npz file
    Format of the .npz file {"emg": array, "fs": sampling_rate} or {"X": array, "timestamps": array}
    '''
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
    '''
    Create the filter object and install:
    - Notch filter @ 60 Hz
    - Bandpass filter @ 20-450 Hz
    '''

    fi = filtering.Filter(fs)

    notch = {"name": "notch", "cutoff": 60, "bandwidth": 3}
    bandpass = {"name": "bandpass", "cutoff": [20, 450], "order": 4}

    fi.install_filters(notch)
    fi.install_filters(bandpass)

    return fi

# APPLY THE FILTERS TO THE EMG DATA
def apply_filters(fi, emg):
    '''
    This method applies the defined filters onto the raw EMG data
    '''
    filtered_data = fi.filter(emg)
    
    return np.array(filtered_data)

# SAVE THE FILTERED DATA INTO A .npz FILE
def save_filtered_data(output_path, filtered, fs, extras):
    '''
    This method saves the filtered data into a .npz file for later use
    '''
    np.savez(output_path, emg = filtered, fs = fs, **extras)

if __name__ == "__main__":
    root = Path("data")

    for fp in root.rglob("*_raw.npz"):
        emg, fs, extras = load_emg_data(fp)
        if fs is None:
            print(f"Skipping {fp} (no sampling rate found)")
            continue
        fi = define_filters(fs)
        filtered_emg = apply_filters(fi, emg)

        out_name = fp.stem.replace("_raw", "") + "_filtered.npz"
        out_path = fp.with_name(out_name)
        save_filtered_data(out_path, filtered_emg, fs, extras)
        print(f"Wrote {out_path}")
