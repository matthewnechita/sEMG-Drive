"""Retroactive data-driven recalibration for filtered EMG sessions.

For sessions where the explicit MVC calibration failed (median MVC/neutral
ratio < threshold), this script replaces the stored calibration with
statistics derived from the session's labeled gesture windows:

  neutral_mean  = mean EMG amplitude over neutral-labeled samples
  mvc_scale     = 95th-percentile EMG amplitude over active-gesture samples

This does NOT require recollecting data. The original calibration arrays are
preserved under calib_neutral_emg_original / calib_mvc_emg_original so the
replacement is reversible.

Usage:
    python tools/recalibrate.py                  # dry run: report only
    python tools/recalibrate.py --apply          # write fixes to filtered files
    python tools/recalibrate.py --threshold 1.5  # custom quality threshold
    python tools/recalibrate.py --restore        # revert to original calibration
"""

import argparse
import sys
from pathlib import Path

import numpy as np

DATA_ROOT  = Path("data")
PATTERN    = "*_filtered.npz"
PERCENTILE = 95.0

NEUTRAL_LABELS = {"neutral", "neutral_buffer"}
ACTIVE_LABELS  = {"horn", "left_turn", "right_turn", "signal_left", "signal_right"}


def _mvc_ratio(neutral_emg, mvc_emg):
    neutral = np.asarray(neutral_emg, dtype=float)
    mvc     = np.asarray(mvc_emg, dtype=float)
    n_rms   = np.sqrt(np.mean(neutral ** 2, axis=0))
    m_rms   = np.sqrt(np.mean(mvc ** 2, axis=0))
    ratio   = np.where(n_rms < 1e-9, 1.0, m_rms / n_rms)
    return ratio


def _data_driven_calibration(emg, labels):
    """Compute neutral_mean and mvc_scale from labeled session windows."""
    emg    = np.asarray(emg, dtype=float)
    labels = np.asarray(labels, dtype=object)

    neutral_mask = np.array(
        [str(l).lower() in NEUTRAL_LABELS for l in labels], dtype=bool
    )
    active_mask  = np.array(
        [str(l).lower() in ACTIVE_LABELS  for l in labels], dtype=bool
    )

    if neutral_mask.sum() < 100:
        return None, None, "not enough neutral samples"
    if active_mask.sum() < 100:
        return None, None, "not enough active gesture samples"

    neutral_mean = np.mean(emg[neutral_mask], axis=0)
    mvc_scale    = np.percentile(emg[active_mask], PERCENTILE, axis=0)
    mvc_scale    = np.where(mvc_scale < 1e-6, 1.0, mvc_scale)
    return neutral_mean, mvc_scale, None


def process_file(fp: Path, threshold: float, apply: bool, restore: bool):
    data = np.load(fp, allow_pickle=True)

    neutral_key = "calib_neutral_emg"
    mvc_key     = "calib_mvc_emg"

    if restore:
        if "calib_neutral_emg_original" not in data.files:
            print(f"  {fp.name}: no backup found, skipping restore")
            return
        new_data = dict(data)
        new_data[neutral_key] = new_data.pop("calib_neutral_emg_original")
        new_data[mvc_key]     = new_data.pop("calib_mvc_emg_original")
        if apply:
            np.savez_compressed(fp, **new_data)
            print(f"  {fp.name}: RESTORED original calibration")
        else:
            print(f"  {fp.name}: would restore (use --apply to write)")
        return

    neutral_emg = data.get(neutral_key)
    mvc_emg     = data.get(mvc_key)

    if neutral_emg is None or mvc_emg is None:
        print(f"  {fp.name}: missing calibration keys, skipping")
        return

    ratio        = _mvc_ratio(neutral_emg, mvc_emg)
    median_ratio = float(np.median(ratio))
    n_weak       = int(np.sum(ratio < threshold))

    status = "OK" if median_ratio >= threshold else "FAIL"
    print(
        f"  {fp.name}: {status}  ratio={median_ratio:.2f}x  "
        f"weak={n_weak}/{len(ratio)}"
    )

    if median_ratio >= threshold:
        return  # calibration is good, nothing to do

    # Calibration failed — compute data-driven replacement
    if "emg" not in data.files or "y" not in data.files:
        print(f"    -> cannot recalibrate: missing emg or y arrays")
        return

    new_neutral, new_scale, err = _data_driven_calibration(data["emg"], data["y"])
    if err:
        print(f"    -> cannot recalibrate: {err}")
        return

    new_ratio        = _mvc_ratio(new_neutral[np.newaxis], new_scale[np.newaxis])
    new_median_ratio = float(np.median(new_ratio))
    print(
        f"    -> data-driven recalibration: ratio={new_median_ratio:.2f}x "
        f"(from {median_ratio:.2f}x)"
    )

    if apply:
        new_data = dict(data)
        # Preserve originals if not already backed up
        if "calib_neutral_emg_original" not in new_data:
            new_data["calib_neutral_emg_original"] = np.asarray(neutral_emg)
            new_data["calib_mvc_emg_original"]     = np.asarray(mvc_emg)
        new_data[neutral_key] = new_neutral
        new_data[mvc_key]     = new_scale[np.newaxis]  # keep (1, channels) shape for compat
        np.savez_compressed(fp, **new_data)
        print(f"    -> WRITTEN to {fp.name}")
    else:
        print(f"    -> dry run (use --apply to write)")


def main():
    parser = argparse.ArgumentParser(description="Retroactive EMG session recalibration.")
    parser.add_argument("--apply",     action="store_true", help="Write fixes to filtered files")
    parser.add_argument("--restore",   action="store_true", help="Revert to original calibration")
    parser.add_argument("--threshold", type=float, default=1.5,
                        help="MVC/neutral ratio below which calibration is replaced (default 1.5)")
    parser.add_argument("--data-root", default=str(DATA_ROOT), help="Root data directory")
    args = parser.parse_args()

    root  = Path(args.data_root)
    files = sorted(root.rglob(PATTERN))
    if not files:
        print(f"No filtered files found under {root}")
        sys.exit(1)

    print(f"Threshold: {args.threshold}x  |  apply={args.apply}  |  restore={args.restore}")
    print(f"Found {len(files)} filtered files.\n")

    for fp in files:
        process_file(fp, threshold=args.threshold, apply=args.apply, restore=args.restore)

    print()
    if not args.apply and not args.restore:
        print("Dry run complete. Re-run with --apply to write changes.")
    else:
        print("Done. Re-run training scripts to use updated calibration.")


if __name__ == "__main__":
    main()
