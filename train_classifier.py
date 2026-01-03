import argparse
import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def normalize_feature_list(raw_list) -> list[str]:
    arr = np.asarray(raw_list)
    return [str(x) for x in arr.tolist()]


def features_to_array(raw_features, feature_order):
    arr = np.asarray(raw_features)
    if arr.dtype == object and arr.shape == ():
        obj = arr.item()
        if isinstance(obj, dict):
            if feature_order is None:
                feature_order = sorted(obj.keys())
            stacked = np.stack([np.asarray(obj[k]) for k in feature_order], axis=-1)
            return stacked, feature_order
        return np.asarray(obj), feature_order
    return arr, feature_order


def pad_channels(X, target_channels):
    pad = np.zeros((X.shape[0], target_channels - X.shape[1], X.shape[2]), dtype=X.dtype)
    return np.concatenate([X, pad], axis=1)


def resolve_target_channels(channel_counts, channel_mode):
    if not channel_counts:
        raise ValueError("No channel counts available to resolve target channels")
    if channel_mode == "match":
        target = channel_counts[0]
        if any(count != target for count in channel_counts):
            raise ValueError(
                f"Channel count mismatch across files (expected {target}, got {sorted(set(channel_counts))})"
            )
        return target
    if channel_mode == "pad":
        return max(channel_counts)
    if channel_mode == "trim":
        return min(channel_counts)
    raise ValueError(f"Unsupported channel_mode: {channel_mode}")


def load_dataset(root, pattern, min_label_confidence=0.0, channel_mode="pad"):
    root = Path(root)
    files = sorted(root.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"No feature files found under {root}")

    collected = []
    channel_counts = []
    data_files = []
    feature_order = None
    window_size = None
    window_step = None

    for fp in files:
        data = np.load(fp, allow_pickle=True)
        if "features" not in data.files:
            continue

        file_feature_order = None
        if "feature_list" in data.files:
            file_feature_order = normalize_feature_list(data["feature_list"])

        X, file_feature_order = features_to_array(data["features"], file_feature_order)
        if file_feature_order is None:
            raise ValueError(f"{fp} is missing feature_list; cannot determine feature order")

        if feature_order is None:
            feature_order = list(file_feature_order)
        elif list(file_feature_order) != list(feature_order):
            raise ValueError(f"Feature order mismatch in {fp}")

        if window_size is None:
            window_size = int(data["window_size_samples"])
            window_step = int(data["window_step_samples"])
        else:
            if int(data["window_size_samples"]) != window_size:
                raise ValueError(f"window_size_samples mismatch in {fp}")
            if int(data["window_step_samples"]) != window_step:
                raise ValueError(f"window_step_samples mismatch in {fp}")

        labels = data.get("window_labels")
        if labels is None:
            labels = np.array([None] * X.shape[0], dtype=object)
        else:
            labels = np.asarray(labels, dtype=object)

        if min_label_confidence > 0.0:
            conf = data.get("window_label_confidence")
            if conf is None:
                raise ValueError(
                    f"min_label_confidence set but {fp} has no window_label_confidence"
                )
            conf = np.asarray(conf, dtype=float)
            conf_mask = conf >= min_label_confidence
            labels = labels[conf_mask]
            X = X[conf_mask]

        label_mask = labels != None  # noqa: E711
        X = X[label_mask]
        labels = labels[label_mask]

        if X.size == 0:
            continue

        collected.append((fp, X, labels))
        channel_counts.append(int(X.shape[1]))
        data_files.append(str(fp))

    if not collected:
        raise ValueError("No labeled windows found in feature files")

    target_channels = resolve_target_channels(channel_counts, channel_mode)
    X_list = []
    y_list = []
    for fp, X, labels in collected:
        if channel_mode == "match" and int(X.shape[1]) != target_channels:
            raise ValueError(
                f"Channel count mismatch in {fp} (expected {target_channels}, got {X.shape[1]})"
            )
        if channel_mode == "pad" and int(X.shape[1]) < target_channels:
            X = pad_channels(X, target_channels)
        elif channel_mode == "trim" and int(X.shape[1]) > target_channels:
            X = X[:, :target_channels, :]
        X_list.append(X)
        y_list.append(labels)

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)

    meta = {
        "feature_order": feature_order,
        "channel_count": int(target_channels),
        "channel_mode": channel_mode,
        "window_size_samples": int(window_size),
        "window_step_samples": int(window_step),
        "labels": sorted({str(x) for x in np.unique(y_all)}),
        "data_files": data_files,
    }
    return X_all, y_all, meta


def build_model(model_name, args):
    if model_name == "svm":
        clf = SVC(
            kernel="rbf",
            C=args.svm_c,
            gamma=args.svm_gamma,
            probability=args.svm_probability,
        )
        return make_pipeline(StandardScaler(), clf)
    raise ValueError(f"Unsupported model: {model_name}")


def build_parser():
    parser = argparse.ArgumentParser(description="Train and export a gesture classifier.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--pattern", default="*_features.npz")
    parser.add_argument("--model", default="svm", choices=["svm"])
    parser.add_argument("--model-out", type=Path, default=Path("models") / "gesture_classifier.pkl")
    parser.add_argument("--metrics-out", type=Path, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-splits", type=int, default=10)
    parser.add_argument("--min-label-confidence", type=float, default=0.0)
    parser.add_argument(
        "--channel-mode",
        choices=["match", "pad", "trim"],
        default="pad",
        help="How to handle channel count mismatches across feature files.",
    )
    parser.add_argument("--svm-c", type=float, default=100.0)
    parser.add_argument("--svm-gamma", default="scale")
    parser.add_argument("--svm-probability", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    X, y, meta = load_dataset(
        args.data_root,
        args.pattern,
        args.min_label_confidence,
        channel_mode=args.channel_mode,
    )
    X_flat = X.reshape(X.shape[0], -1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_flat,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    model = build_model(args.model, args)
    cv = StratifiedKFold(n_splits=args.cv_splits, shuffle=True, random_state=args.random_state)
    cv_scores = cross_validate(model, X_flat, y, cv=cv, scoring="accuracy", return_train_score=False)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "cv_accuracy_mean": float(cv_scores["test_score"].mean()),
        "cv_accuracy_std": float(cv_scores["test_score"].std()),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "train_accuracy": float(model.score(X_train, y_train)),
        "n_samples": int(X_flat.shape[0]),
        "n_classes": int(len(np.unique(y))),
    }

    print(f"CV accuracy: {metrics['cv_accuracy_mean']:.3f} +/- {metrics['cv_accuracy_std']:.3f}")
    print(f"Train accuracy: {metrics['train_accuracy']:.3f}")
    print(f"Test accuracy: {metrics['test_accuracy']:.3f}")
    print("\nReport:\n", classification_report(y_test, y_pred))

    meta.update(
        {
            "created_at": dt.datetime.now().isoformat(),
            "model_type": args.model,
            "model_params": {
                "svm_c": args.svm_c,
                "svm_gamma": args.svm_gamma,
                "svm_probability": bool(args.svm_probability),
            },
            "metrics": metrics,
        }
    )

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    with args.model_out.open("wb") as f:
        pickle.dump({"model": model, "metadata": meta}, f)

    print(f"Saved model bundle to {args.model_out}")

    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_out.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved metrics to {args.metrics_out}")


if __name__ == "__main__":
    main()
