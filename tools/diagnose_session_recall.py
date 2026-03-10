import argparse
import csv
import datetime as dt
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from libemg.utils import get_windows

from emg.gesture_model_cnn import load_cnn_bundle


def _clean_label(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except Exception:
            return None
    if isinstance(value, np.str_):
        value = str(value)
    if isinstance(value, str):
        value = value.strip()
    if value == "":
        return None
    return str(value)


def majority_label_with_confidence(segment):
    if segment.size == 0:
        return None, 0.0
    flat = segment.reshape(-1)
    cleaned = []
    for item in flat:
        lbl = _clean_label(item)
        if lbl is not None:
            cleaned.append(lbl)
    if not cleaned:
        return None, 0.0
    values, counts = np.unique(np.asarray(cleaned, dtype=object), return_counts=True)
    idx = int(np.argmax(counts))
    total = int(np.sum(counts))
    confidence = float(counts[idx] / total) if total > 0 else 0.0
    return str(values[idx]), confidence


def compute_calibration(neutral_emg, mvc_emg, percentile, mvc_min_ratio):
    neutral = np.asarray(neutral_emg, dtype=float)
    mvc = np.asarray(mvc_emg, dtype=float)
    if neutral.size == 0 or mvc.size == 0:
        return None, None

    neutral_rms = np.sqrt(np.mean(neutral ** 2, axis=0))
    mvc_rms = np.sqrt(np.mean(mvc ** 2, axis=0))
    ratio = np.ones_like(mvc_rms, dtype=float)
    np.divide(mvc_rms, neutral_rms, out=ratio, where=neutral_rms >= 1e-9)
    median_ratio = float(np.median(ratio))
    if median_ratio < float(mvc_min_ratio):
        return None, None

    neutral_mean = np.mean(neutral, axis=0)
    mvc_scale = np.percentile(mvc, float(percentile), axis=0)
    mvc_scale = np.where(mvc_scale < 1e-6, 1.0, mvc_scale)
    return neutral_mean, mvc_scale


def load_windows_from_file(
    path,
    window_size,
    window_step,
    min_label_confidence,
    use_min_label_confidence,
    use_calibration,
    mvc_percentile,
    mvc_min_ratio,
    included_gestures,
):
    data = np.load(path, allow_pickle=True)
    if "emg" not in data.files or "y" not in data.files:
        return None, None

    emg = np.asarray(data["emg"], dtype=float)
    if use_calibration:
        calib_neutral = data.get("calib_neutral_emg")
        calib_mvc = data.get("calib_mvc_emg")
        if calib_neutral is not None and calib_mvc is not None:
            neutral_mean, mvc_scale = compute_calibration(
                calib_neutral, calib_mvc, mvc_percentile, mvc_min_ratio
            )
            if neutral_mean is not None and mvc_scale is not None:
                emg = (emg - neutral_mean) / mvc_scale

    windows = get_windows(emg, int(window_size), int(window_step))
    labels = np.asarray(data["y"], dtype=object)

    n_windows = int(windows.shape[0])
    starts = np.arange(n_windows) * int(window_step)
    ends = starts + int(window_size)

    window_labels = []
    for s, e in zip(starts, ends):
        lbl, conf = majority_label_with_confidence(labels[s:e])
        if lbl == "neutral_buffer":
            lbl = None
        if (
            use_min_label_confidence
            and lbl is not None
            and float(conf) < float(min_label_confidence)
        ):
            lbl = None
        if lbl is not None and included_gestures is not None and lbl not in included_gestures:
            lbl = None
        window_labels.append(lbl)

    window_labels = np.asarray(window_labels, dtype=object)
    keep = window_labels != None  # noqa: E711
    windows = windows[keep]
    window_labels = window_labels[keep]

    if windows.size == 0:
        return None, None
    return windows.astype(np.float32), window_labels


def predict_proba_batched(model, X, device, batch_size):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], int(batch_size)):
            xb = torch.from_numpy(X[i : i + int(batch_size)].astype(np.float32)).to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            out.append(probs)
    if not out:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(out, axis=0)


def _safe_recall(mask, pred_idx, true_idx):
    denom = int(np.sum(mask))
    if denom <= 0:
        return float("nan"), 0
    num = int(np.sum(pred_idx[mask] == true_idx))
    return float(num / denom), denom


def evaluate_file(
    bundle,
    file_path,
    label_to_index,
    labels_order,
    target_labels,
    split_tag,
    args,
):
    X, y = load_windows_from_file(
        path=file_path,
        window_size=args.window_size,
        window_step=args.window_step,
        min_label_confidence=args.min_label_confidence,
        use_min_label_confidence=bool(args.use_min_label_confidence),
        use_calibration=bool(args.use_calibration),
        mvc_percentile=args.mvc_percentile,
        mvc_min_ratio=args.mvc_min_ratio,
        included_gestures=args.included_gestures,
    )
    if X is None or y is None or len(y) == 0:
        return None

    X = bundle.standardize(X)
    probs = predict_proba_batched(bundle.model, X, args.device, args.batch_size)
    pred_idx = np.argmax(probs, axis=1).astype(int)
    y_idx = np.asarray([int(label_to_index[str(lbl)]) for lbl in y], dtype=int)
    acc = float(np.mean(pred_idx == y_idx))

    row = {
        "file": file_path.name,
        "split": split_tag,
        "windows": int(len(y_idx)),
        "accuracy": acc,
    }

    for target in target_labels:
        t_idx = label_to_index.get(target)
        if t_idx is None:
            row[f"{target}_support"] = 0
            row[f"{target}_recall"] = float("nan")
            row[f"{target}_top_confusion"] = ""
            continue
        mask = y_idx == int(t_idx)
        rec, support = _safe_recall(mask, pred_idx, int(t_idx))
        row[f"{target}_support"] = int(support)
        row[f"{target}_recall"] = rec
        if support > 0:
            wrong = pred_idx[mask]
            wrong = wrong[wrong != int(t_idx)]
            if wrong.size:
                top_idx = int(Counter(wrong.tolist()).most_common(1)[0][0])
                row[f"{target}_top_confusion"] = str(labels_order[top_idx])
            else:
                row[f"{target}_top_confusion"] = ""
        else:
            row[f"{target}_top_confusion"] = ""
    return row


def _parse_target_labels(value):
    parts = [p.strip() for p in str(value).split(",")]
    return [p for p in parts if p]


def _parse_included_gestures(value):
    if value is None:
        return None
    txt = str(value).strip().lower()
    if txt in ("", "none", "all"):
        return None
    parts = [p.strip() for p in str(value).split(",")]
    out = {p for p in parts if p}
    return out if out else None


def _resolve_files(args, model_meta):
    root = Path(args.data_root) / f"{args.arm} arm" / args.subject / "filtered"
    files = sorted(root.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files found under {root} matching {args.pattern}")

    train_names = set(model_meta.get("train_files") or [])
    test_names = set(model_meta.get("test_files") or [])

    if args.subset == "all":
        return files, train_names, test_names
    if args.subset == "train":
        files = [f for f in files if f.name in train_names]
    elif args.subset == "test":
        files = [f for f in files if f.name in test_names]
    if not files:
        raise ValueError(f"No files left after subset={args.subset}.")
    return files, train_names, test_names


def _split_tag(name, train_names, test_names):
    if name in train_names:
        return "train"
    if name in test_names:
        return "test"
    return "other"


def _mean_ignore_nan(values):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Per-session diagnostics for a saved CNN bundle. "
            "Focuses on class recall collapse (for example right_turn/signal_right)."
        )
    )
    parser.add_argument("--model", required=True, help="Path to saved .pt bundle")
    parser.add_argument("--data-root", default="data_resampled", help="Dataset root")
    parser.add_argument("--arm", default="right", choices=["right", "left"])
    parser.add_argument("--subject", default="Matthew")
    parser.add_argument("--pattern", default="*_filtered.npz")
    parser.add_argument("--subset", default="all", choices=["all", "train", "test"])
    parser.add_argument(
        "--target-labels",
        default="right_turn,signal_right",
        help="Comma-separated target labels to highlight",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--window-step", type=int, default=None)
    parser.add_argument("--use-calibration", type=int, default=1, choices=[0, 1])
    parser.add_argument("--mvc-percentile", type=float, default=95.0)
    parser.add_argument("--mvc-min-ratio", type=float, default=1.5)
    parser.add_argument("--use-min-label-confidence", type=int, default=1, choices=[0, 1])
    parser.add_argument("--min-label-confidence", type=float, default=0.75)
    parser.add_argument(
        "--included-gestures",
        default=None,
        help="Comma-separated allowed gestures or 'all'/'none' for no filter",
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.included_gestures = _parse_included_gestures(args.included_gestures)
    target_labels = _parse_target_labels(args.target_labels)
    if not target_labels:
        raise ValueError("No valid --target-labels provided.")

    bundle = load_cnn_bundle(args.model, device=args.device)
    model_meta = bundle.metadata if isinstance(bundle.metadata, dict) else {}

    if args.window_size is None:
        args.window_size = int(model_meta.get("window_size_samples", 200))
    if args.window_step is None:
        args.window_step = int(model_meta.get("window_step_samples", 100))
    if args.included_gestures is None and model_meta.get("included_gestures") is not None:
        args.included_gestures = {str(x) for x in model_meta.get("included_gestures") or []}

    index_to_label = {int(k): str(v) for k, v in bundle.index_to_label.items()}
    labels_order = [index_to_label[i] for i in sorted(index_to_label.keys())]
    label_to_index = {label: i for i, label in enumerate(labels_order)}

    files, train_names, test_names = _resolve_files(args, model_meta)

    print(f"device: {args.device}")
    print(f"model: {args.model}")
    print(f"data root: {Path(args.data_root) / f'{args.arm} arm' / args.subject / 'filtered'}")
    print(f"files: {len(files)} (subset={args.subset})")
    print(f"window: size={args.window_size}, step={args.window_step}")
    print(f"targets: {target_labels}")
    if args.included_gestures is not None:
        print(f"included gestures: {sorted(args.included_gestures)}")

    rows = []
    for i, fp in enumerate(files, start=1):
        split_tag = _split_tag(fp.name, train_names, test_names)
        row = evaluate_file(
            bundle=bundle,
            file_path=fp,
            label_to_index=label_to_index,
            labels_order=labels_order,
            target_labels=target_labels,
            split_tag=split_tag,
            args=args,
        )
        if row is None:
            print(f"[{i}/{len(files)}] {fp.name}: skipped (no usable windows)")
            continue
        rows.append(row)
        parts = [f"[{i}/{len(files)}] {fp.name}", f"acc={row['accuracy']:.3f}"]
        for target in target_labels:
            rec = row.get(f"{target}_recall", float("nan"))
            sup = row.get(f"{target}_support", 0)
            if np.isfinite(rec):
                parts.append(f"{target}={rec:.3f} ({sup})")
            else:
                parts.append(f"{target}=nan ({sup})")
        print(" | ".join(parts))

    if not rows:
        raise RuntimeError("No evaluable files.")

    def score_key(r):
        vals = [r.get(f"{t}_recall", float("nan")) for t in target_labels]
        return _mean_ignore_nan(vals)

    rows_sorted = sorted(rows, key=score_key)
    print("\nWorst sessions by mean target recall:")
    for r in rows_sorted[: min(10, len(rows_sorted))]:
        metrics = []
        for t in target_labels:
            rec = r.get(f"{t}_recall", float("nan"))
            sup = r.get(f"{t}_support", 0)
            conf = r.get(f"{t}_top_confusion", "")
            if np.isfinite(rec):
                metrics.append(f"{t}={rec:.3f} ({sup}) conf->{conf or '-'}")
            else:
                metrics.append(f"{t}=nan ({sup})")
        print(f"  {r['file']} [{r['split']}] acc={r['accuracy']:.3f} | " + " | ".join(metrics))

    summary = {
        "files_evaluated": len(rows),
        "mean_accuracy": _mean_ignore_nan([r["accuracy"] for r in rows]),
    }
    for t in target_labels:
        summary[f"mean_{t}_recall"] = _mean_ignore_nan([r.get(f"{t}_recall", float("nan")) for r in rows])

    print("\nSummary:")
    print(f"  mean accuracy: {summary['mean_accuracy']:.3f}")
    for t in target_labels:
        val = summary[f"mean_{t}_recall"]
        if np.isfinite(val):
            print(f"  mean {t} recall: {val:.3f}")
        else:
            print(f"  mean {t} recall: nan")

    if args.output:
        out_path = Path(args.output)
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path("tmp") / f"session_recall_diagnostic_{args.arm}_{args.subject}_{stamp}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["file", "split", "windows", "accuracy"]
    for t in target_labels:
        fieldnames.extend([f"{t}_support", f"{t}_recall", f"{t}_top_confusion"])
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_sorted:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

