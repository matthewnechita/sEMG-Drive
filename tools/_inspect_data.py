"""Summarize labeled window counts per subject and session."""
import numpy as np
from pathlib import Path

DATA_ROOT = Path("data")
WINDOW_SIZE = 200
WINDOW_STEP = 100

def count_windows(fp):
    data = np.load(fp, allow_pickle=True)
    if "emg" not in data.files or "y" not in data.files:
        return 0, 0, {}
    emg = np.asarray(data["emg"])
    n_total = max(0, (emg.shape[0] - WINDOW_SIZE) // WINDOW_STEP + 1)

    y = np.asarray(data["y"], dtype=object)
    has_calib = ("calib_neutral_emg" in data.files and "calib_mvc_emg" in data.files)

    # Count label distribution (rough — per sample not per window)
    labels, counts = np.unique(
        [str(v) for v in y if v is not None and str(v) != "neutral_buffer"],
        return_counts=True
    )
    label_dist = dict(zip(labels, counts.tolist()))
    return n_total, has_calib, label_dist

subjects = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir())
grand_total = 0
for subj in subjects:
    files = sorted((DATA_ROOT / subj).rglob("*_filtered.npz"))
    print(f"\n{subj}: {len(files)} session file(s)")
    subj_total = 0
    for fp in files:
        n_win, has_calib, ldist = count_windows(fp)
        calib_str = "calib:YES" if has_calib else "calib:NO "
        print(f"  {fp.name:<50} {calib_str}  ~{n_win:4d} windows  labels: {ldist}")
        subj_total += n_win
    print(f"  SUBTOTAL: ~{subj_total} windows")
    grand_total += subj_total

print(f"\nGRAND TOTAL: ~{grand_total} windows across all subjects")
