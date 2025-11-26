import numpy as np
from libemg.feature_extractor import FeatureExtractor
from libemg.utils import get_windows
from pathlib import Path
# Extract a predefined feature group
# features_2 = fe.extract_feature_group('HTD', windows) # feature group defined by libEMG
# from research, right now we are using MAV, RMS, WL, ZC, and possibly SSC. This seems to be pretty
# lightweight and common based on research papers, I think using RMS may change our filtering process a bit 
fe = FeatureExtractor()
root = Path("data")
feature_list = ['MAV', 'RMS' 'WL', 'ZC'] # list we choose

for fp in root.rglob("*_filtered.npz"):
    data = np.load(fp)
    emg = data["emg"]
    fs = float(np.asarray(data["fs"]).squeeze())
    windows = get_windows(emg, 200, 100) # 200 samples, 100 step, 50% overlap
    features_1 = fe.extract_features(feature_list, windows)

    out_name = fp.stem.replace("_filtered", "") + "_features.npz"
    out_path = fp.with_name(out_name)
    np.savez_compressed(
        out_path,
        features=features_1,
        feature_list=feature_list,
        fs=fs,
        window_size_samples=200,
        window_step_samples=100,
        source_file=str(fp),
    )
    print(f"Wrote {out_path}")

