import argparse
import shutil
from pathlib import Path

import numpy as np


def _label_to_str(label):
    if label is None:
        return None
    if isinstance(label, bytes):
        try:
            label = label.decode("utf-8")
        except Exception:
            return None
    if isinstance(label, (np.str_, str)):
        return str(label)
    return None


def drop_first_gesture_run(labels, neutral_labels):
    labels = np.asarray(labels, dtype=object).reshape(-1)
    updated = labels.copy()
    neutral_set = set(neutral_labels)

    start = None
    run_label = None
    for idx, raw in enumerate(labels):
        lbl = _label_to_str(raw)
        if lbl is None or lbl in neutral_set:
            continue
        start = idx
        run_label = lbl
        break

    if start is None:
        return updated, 0, None

    end = start
    for idx in range(start + 1, len(labels)):
        lbl = _label_to_str(labels[idx])
        if lbl != run_label:
            end = idx - 1
            break
    else:
        end = len(labels) - 1

    updated[start : end + 1] = None
    return updated, end - start + 1, run_label


def process_file(path, args):
    data = np.load(path, allow_pickle=True)
    if "y" not in data.files:
        return {"path": path, "status": "skip", "reason": "no labels"}

    labels = data["y"]
    if labels.size == 0:
        return {"path": path, "status": "skip", "reason": "empty labels"}

    updated, dropped, dropped_label = drop_first_gesture_run(
        labels, args.neutral_labels
    )
    if dropped == 0:
        return {"path": path, "status": "ok", "dropped": 0, "label": None}

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
        "dropped": dropped,
        "label": dropped_label,
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Drop the first non-neutral gesture run from labeled data."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--pattern", default="*_relabeled_raw.npz")
    parser.add_argument(
        "--neutral-labels",
        nargs="+",
        default=["neutral", "neutral_buffer"],
        help="Labels considered neutral/rest (not treated as gestures).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes to files (default is dry-run).",
    )
    parser.add_argument(
        "--output-suffix",
        default="_first_gesture_dropped_raw",
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
    return parser


def main():
    args = build_parser().parse_args()
    files = sorted(args.data_root.rglob(args.pattern))
    if not files:
        print(f"No files matched {args.pattern} under {args.data_root}")
        return

    totals = {"updated": 0, "dropped": 0, "skipped": 0}
    for fp in files:
        result = process_file(fp, args)
        status = result["status"]
        if status == "skip":
            totals["skipped"] += 1
            print(f"Skip {fp}: {result['reason']}")
            continue
        totals["dropped"] += result.get("dropped", 0)
        if result.get("dropped", 0) > 0:
            print(
                f"{status.upper()} {fp} (label={result.get('label')}, "
                f"dropped={result['dropped']})"
            )
        if status in {"updated", "dry-run"}:
            totals["updated"] += 1

    mode = "APPLIED" if args.apply else "DRY-RUN"
    print(
        f"{mode}: files={totals['updated']} skipped={totals['skipped']} "
        f"dropped_samples={totals['dropped']}"
    )


if __name__ == "__main__":
    main()
