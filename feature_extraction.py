import numpy as np
from pathlib import Path
from libemg.feature_extractor import FeatureExtractor
from libemg.utils import get_windows

# Extract a predefined feature group
# features_2 = fe.extract_feature_group('HTD', windows) # feature group defined by libEMG
# from research, right now we are using MAV, RMS, WL, ZC, and SSC. This seems to be pretty
# lightweight and common based on research papers, I think using RMS may change our filtering process a bit
fe = FeatureExtractor()
root = Path("data")
feature_list = ['MAV', 'RMS', 'WL', 'ZC', 'SSC']  # list we choose
WINDOW_SIZE = 200
WINDOW_STEP = 100


def majority_label(segment):
    """Return majority label and confidence (fraction) for a 1D label segment."""
    if segment.size == 0:
        return None, 0.0
    flat = segment.reshape(-1)
    # drop NaNs if numeric
    if flat.dtype.kind in "fc":
        flat = flat[~np.isnan(flat)]
    if flat.size == 0:
        return None, 0.0
    values, counts = np.unique(flat, return_counts=True)
    idx = counts.argmax()
    return values[idx], counts[idx] / counts.sum()


def destination_for_features(filtered_path: Path) -> Path:
    """
    Place features in ../features/ when filtered files are in ../filtered/.
    Fallback: same directory as the filtered file.
    """
    if filtered_path.parent.name == "filtered":
        base_dir = filtered_path.parent.parent
        return base_dir / "features" / (filtered_path.stem.replace("_filtered", "") + "_features.npz")
    return filtered_path.with_name(filtered_path.stem.replace("_filtered", "") + "_features.npz")


for fp in root.rglob("*_filtered.npz"):
    out_path = destination_for_features(fp)
    if out_path.exists():
        print(f"Skipping {fp} (features exists at {out_path})")
        continue
    data = np.load(fp, allow_pickle=True)
    emg = data["emg"]
    fs = float(np.asarray(data["fs"]).squeeze())
    windows = get_windows(emg, WINDOW_SIZE, WINDOW_STEP)  # 200 samples, 100 step, 50% overlap
    n_windows = windows.shape[0]
    starts = np.arange(n_windows) * WINDOW_STEP
    ends = starts + WINDOW_SIZE

    # Align window labels to gestures if available
    y = data.get("y")
    window_labels = None
    window_label_confidence = None
    if y is not None:
        y_arr = np.asarray(y)
        labels = []
        confs = []
        for s, e in zip(starts, ends):
            seg = y_arr[s:e]
            lbl, conf = majority_label(seg)
            labels.append(lbl)
            confs.append(conf)
        window_labels = np.array(labels, dtype=object)
        window_label_confidence = np.array(confs, dtype=float)

    features_1 = fe.extract_features(feature_list, windows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        features=features_1,
        feature_list=feature_list,
        fs=fs,
        window_size_samples=WINDOW_SIZE,
        window_step_samples=WINDOW_STEP,
        window_start_samples=starts,
        window_end_samples=ends,
        window_labels=window_labels,
        window_label_confidence=window_label_confidence,
        source_file=str(fp),
    )
    print(f"Wrote {out_path}")
