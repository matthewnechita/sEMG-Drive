import argparse
import datetime as dt
import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from train_classifier import build_model, load_dataset


def build_parser():
    parser = argparse.ArgumentParser(
        description="Quickly compare a few SVM configs (no CV/grid search)."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--pattern", default="*_features.npz")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--channel-mode",
        choices=["match", "pad", "trim"],
        default="pad",
        help="How to handle channel count mismatches across feature files.",
    )
    parser.add_argument("--min-label-confidence", type=float, default=0.0)
    parser.add_argument(
        "--svm-c",
        nargs="+",
        type=float,
        default=[1.0, 100.0],
        help="C values to compare.",
    )
    parser.add_argument(
        "--svm-gamma",
        default="scale",
        help="Gamma value for all comparisons (number or 'scale'/'auto').",
    )
    parser.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="balanced",
        help="Class weighting for SVM.",
    )
    parser.add_argument(
        "--svm-probability",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable probability estimates.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models") / "gesture_classifier.pkl",
        help="Path to save the best model bundle.",
    )
    return parser


def resolve_split(X, y, groups, test_size, random_state):
    unique_groups = np.unique(groups)
    if unique_groups.size >= 2:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        train_idx, test_idx = next(splitter.split(X, y, groups))
        print(f"Using group split across {unique_groups.size} sessions/files.")
        return train_idx, test_idx
    train_idx, test_idx = train_test_split(
        np.arange(X.shape[0]),
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print("Warning: group split disabled (not enough sessions/files).")
    return train_idx, test_idx


def main():
    args = build_parser().parse_args()
    X, y, groups, meta = load_dataset(
        args.data_root,
        args.pattern,
        args.min_label_confidence,
        channel_mode=args.channel_mode,
    )
    X_flat = X.reshape(X.shape[0], -1)
    groups = np.asarray(groups, dtype=object)

    train_idx, test_idx = resolve_split(
        X_flat, y, groups, args.test_size, args.random_state
    )

    X_train = X_flat[train_idx]
    X_test = X_flat[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    results = []
    for c_val in args.svm_c:
        model = build_model(
            "svm",
            args,
            svm_c=float(c_val),
            svm_gamma=args.svm_gamma,
        )
        model.fit(X_train, y_train)
        test_acc = float(model.score(X_test, y_test))
        train_acc = float(model.score(X_train, y_train))
        results.append((float(c_val), test_acc, train_acc))
        print(f"C={float(c_val)} gamma={args.svm_gamma} test={test_acc:.3f} train={train_acc:.3f}")

    results_sorted = sorted(
        results, key=lambda r: (r[1], -r[2], -r[0]), reverse=True
    )
    best_c, best_test, best_train = results_sorted[0]
    print(f"Best config: C={best_c} gamma={args.svm_gamma} (test={best_test:.3f})")

    final_model = build_model(
        "svm",
        args,
        svm_c=best_c,
        svm_gamma=args.svm_gamma,
    )
    final_model.fit(X_flat, y)
    meta.update(
        {
            "created_at": dt.datetime.now().isoformat(),
            "model_type": "svm",
            "model_params": {
                "svm_c": float(best_c),
                "svm_gamma": args.svm_gamma,
                "class_weight": args.class_weight,
                "svm_probability": bool(args.svm_probability),
            },
            "metrics": {
                "test_accuracy": float(best_test),
                "train_accuracy": float(best_train),
                "n_samples": int(X_flat.shape[0]),
                "n_classes": int(len(np.unique(y))),
            },
            "evaluation_split": {
                "group_split": bool(np.unique(groups).size >= 2),
                "test_size": float(args.test_size),
                "cv_splits": 0,
            },
            "fit_all_data": True,
            "quick_compare": {
                "c_values": [float(v) for v in args.svm_c],
                "gamma": args.svm_gamma,
                "results": [
                    {"c": float(c), "test_accuracy": float(t), "train_accuracy": float(tr)}
                    for c, t, tr in results
                ],
            },
        }
    )
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    with args.model_out.open("wb") as f:
        pickle.dump({"model": final_model, "metadata": meta}, f)
    print(f"Saved best model bundle to {args.model_out}")


if __name__ == "__main__":
    main()
