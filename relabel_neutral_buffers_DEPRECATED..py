import argparse
import shutil
from pathlib import Path

import numpy as np


def _to_scalar(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _label_equals(label, target):
    if label is None:
        return False
    if isinstance(label, bytes):
        try:
            label = label.decode("utf-8")
        except Exception:
            return False
    if isinstance(label, (np.str_, str)):
        return str(label) == target
    return False


def _estimate_sample_period(timestamps, fs, auto_scale_ms=True):
    warnings = []
    info = {
        "ts_period": None,
        "fs_period": None,
        "ratio": None,
        "auto_scaled_ms": False,
    }
    if timestamps is not None:
        t = np.asarray(timestamps)
        if t.ndim > 1:
            t = t[:, 0]
        t = t.astype(float)
        diffs = np.diff(t)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            ts_period = float(np.median(diffs))
        else:
            ts_period = None
    else:
        ts_period = None
    info["ts_period"] = ts_period

    fs_period = None
    if fs is not None:
        fs_val = float(np.asarray(fs).squeeze())
        if np.isfinite(fs_val) and fs_val > 0:
            fs_period = 1.0 / fs_val
    info["fs_period"] = fs_period

    if ts_period is None and fs_period is None:
        return None, warnings, info

    period = ts_period if ts_period is not None else fs_period

    if ts_period is not None and fs_period is not None:
        ratio = ts_period / fs_period if fs_period else None
        info["ratio"] = ratio
        if ratio is not None:
            if 500.0 <= ratio <= 2000.0:
                if auto_scale_ms:
                    period = ts_period / 1000.0
                    warnings.append(
                        "timestamps look like milliseconds; auto-scaled by 1/1000"
                    )
                    info["auto_scaled_ms"] = True
                else:
                    warnings.append(
                        "timestamp/fs mismatch (timestamps may be milliseconds)"
                    )
            elif ratio < 0.1 or ratio > 10.0:
                warnings.append("timestamp/fs mismatch; verify time units")

    if period is not None:
        if period > 0.05:
            warnings.append(
                f"sample period {period:.4f}s looks large for EMG; verify time units"
            )
        if period < 1e-5:
            warnings.append(
                f"sample period {period:.6f}s looks small for EMG; verify time units"
            )

    return period, warnings, info


def relabel_short_runs(
    labels,
    sample_period,
    max_duration_s,
    target_label,
    new_label,
):
    labels = np.asarray(labels, dtype=object).reshape(-1)
    updated = labels.copy()
    relabeled_runs = 0
    relabeled_samples = 0

    run_start = None
    for idx, lbl in enumerate(labels):
        if _label_equals(lbl, target_label):
            if run_start is None:
                run_start = idx
            continue
        if run_start is not None:
            run_end = idx - 1
            run_len = run_end - run_start + 1
            run_duration = run_len * sample_period
            if run_duration < max_duration_s:
                updated[run_start : run_end + 1] = new_label
                relabeled_runs += 1
                relabeled_samples += run_len
            run_start = None

    if run_start is not None:
        run_end = len(labels) - 1
        run_len = run_end - run_start + 1
        run_duration = run_len * sample_period
        if run_duration < max_duration_s:
            updated[run_start : run_end + 1] = new_label
            relabeled_runs += 1
            relabeled_samples += run_len

    return updated, relabeled_runs, relabeled_samples


def _as_dict(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, dict):
        return value
    return None


def _check_metadata_thresholds(metadata, max_duration_s):
    warnings = []
    neutral_duration = metadata.get("neutral_duration_s")
    rest_duration = metadata.get("inter_gesture_rest_s")
    if neutral_duration is not None:
        try:
            neutral_duration = float(neutral_duration)
            if neutral_duration <= max_duration_s:
                warnings.append(
                    "neutral_duration_s <= max_duration_s; true neutral may be relabeled"
                )
        except (TypeError, ValueError):
            pass
    if rest_duration is not None:
        try:
            rest_duration = float(rest_duration)
            if rest_duration >= max_duration_s:
                warnings.append(
                    "inter_gesture_rest_s >= max_duration_s; buffers may not be relabeled"
                )
        except (TypeError, ValueError):
            pass
    return warnings


def process_file(path, args):
    data = np.load(path, allow_pickle=True)
    if "y" not in data.files:
        return {"path": path, "status": "skip", "reason": "no labels"}

    labels = data["y"]
    if labels.size == 0:
        return {"path": path, "status": "skip", "reason": "empty labels"}

    timestamps = data.get("timestamps")
    fs = _to_scalar(data.get("fs"))
    sample_period, time_warnings, time_info = _estimate_sample_period(
        timestamps, fs, auto_scale_ms=not args.no_auto_scale_ms
    )
    if sample_period is None or sample_period <= 0:
        return {"path": path, "status": "skip", "reason": "no timestamps or fs"}

    warnings = list(time_warnings)
    metadata = _as_dict(data.get("metadata"))
    if metadata:
        warnings.extend(_check_metadata_thresholds(metadata, args.max_duration_s))

    updated, relabeled_runs, relabeled_samples = relabel_short_runs(
        labels,
        sample_period,
        args.max_duration_s,
        args.target_label,
        args.new_label,
    )

    if relabeled_runs == 0:
        return {
            "path": path,
            "status": "ok",
            "runs": 0,
            "samples": 0,
            "warnings": warnings,
            "time_info": time_info,
        }

    if args.apply:
        if args.in_place:
            out_path = path
            if args.backup:
                backup_path = path.with_suffix(path.suffix + ".bak")
                shutil.copy2(path, backup_path)
        else:
            stem = path.stem
            if stem.endswith("_raw"):
                stem = stem[: -len("_raw")]
            out_path = path.with_name(stem + args.output_suffix + path.suffix)
            if out_path.exists() and not args.overwrite:
                return {
                    "path": path,
                    "status": "skip",
                    "reason": f"output exists ({out_path.name})",
                }
        payload = {k: data[k] for k in data.files}
        payload["y"] = updated
        if path.name.endswith("_filtered.npz"):
            np.savez(out_path, **payload)
        else:
            np.savez_compressed(out_path, **payload)

    return {
        "path": path,
        "status": "updated" if args.apply else "dry-run",
        "runs": relabeled_runs,
        "samples": relabeled_samples,
        "warnings": warnings,
        "time_info": time_info,
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Relabel short neutral runs as neutral_buffer in existing data."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--pattern", default="*_raw.npz")
    parser.add_argument("--target-label", default="neutral")
    parser.add_argument("--new-label", default="neutral_buffer")
    parser.add_argument("--max-duration-s", type=float, default=2.3)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes to files (default is dry-run).",
    )
    parser.add_argument(
        "--output-suffix",
        default="_relabeled_raw",
        help="Suffix added to new output files when not writing in place.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input files instead of writing new ones.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Write .bak copies before overwriting in-place.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    parser.add_argument(
        "--no-auto-scale-ms",
        action="store_true",
        help="Disable auto-scaling when timestamps look like milliseconds.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file timing info (sample period, fs period, ratio).",
    )
    return parser


def main():
    args = build_parser().parse_args()
    files = sorted(args.data_root.rglob(args.pattern))
    if not files:
        print(f"No files matched {args.pattern} under {args.data_root}")
        return

    totals = {"updated": 0, "runs": 0, "samples": 0, "skipped": 0}
    for fp in files:
        result = process_file(fp, args)
        status = result["status"]
        if status == "skip":
            totals["skipped"] += 1
            print(f"Skip {fp}: {result['reason']}")
            continue
        for warn in result.get("warnings", []):
            print(f"Warning {fp}: {warn}")
        if args.verbose:
            info = result.get("time_info", {})
            ts_period = info.get("ts_period")
            fs_period = info.get("fs_period")
            ratio = info.get("ratio")
            scaled = info.get("auto_scaled_ms")
            print(
                f"Info {fp}: ts_period={ts_period} fs_period={fs_period} "
                f"ratio={ratio} auto_scaled_ms={scaled}"
            )
        totals["runs"] += result.get("runs", 0)
        totals["samples"] += result.get("samples", 0)
        if result.get("runs", 0) > 0:
            print(
                f"{status.upper()} {fp} (runs={result['runs']}, samples={result['samples']})"
            )
        if status in {"updated", "dry-run"}:
            totals["updated"] += 1

    mode = "APPLIED" if args.apply else "DRY-RUN"
    print(
        f"{mode}: files={totals['updated']} skipped={totals['skipped']} "
        f"runs={totals['runs']} samples={totals['samples']}"
    )


if __name__ == "__main__":
    main()
