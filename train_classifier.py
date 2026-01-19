import argparse
import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
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
    fs_values = []

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

        fs_value = data.get("fs")
        if fs_value is not None:
            fs_value = float(np.asarray(fs_value).squeeze())
            if np.isfinite(fs_value):
                fs_values.append(fs_value)

        collected.append((fp, X, labels))
        channel_counts.append(int(X.shape[1]))
        data_files.append(str(fp))

    if not collected:
        raise ValueError("No labeled windows found in feature files")

    target_channels = resolve_target_channels(channel_counts, channel_mode)
    X_list = []
    y_list = []
    group_list = []
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
        group_list.append(np.array([str(fp)] * len(labels), dtype=object))

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    groups_all = np.concatenate(group_list)

    fs_hz = None
    fs_hz_values = None
    if fs_values:
        fs_arr = np.asarray(fs_values, dtype=float)
        if np.allclose(fs_arr, fs_arr[0], rtol=0.0, atol=1e-3):
            fs_hz = float(fs_arr[0])
        else:
            fs_hz_values = sorted({round(float(x), 3) for x in fs_arr})

    meta = {
        "feature_order": feature_order,
        "channel_count": int(target_channels),
        "channel_mode": channel_mode,
        "window_size_samples": int(window_size),
        "window_step_samples": int(window_step),
        "labels": sorted({str(x) for x in np.unique(y_all)}),
        "data_files": data_files,
        "group_by": "feature_file",
        "group_count": int(np.unique(groups_all).size),
    }
    if fs_hz is not None:
        meta["fs_hz"] = fs_hz
    elif fs_hz_values is not None:
        meta["fs_hz_values"] = fs_hz_values
    return X_all, y_all, groups_all, meta


def build_model(model_name, args, svm_c=None, svm_gamma=None):
    if model_name == "svm":
        class_weight = None if args.class_weight == "none" else "balanced"
        gamma = args.svm_gamma if svm_gamma is None else svm_gamma
        if isinstance(gamma, str) and gamma not in {"scale", "auto"}:
            gamma = float(gamma)
        clf = SVC(
            kernel="rbf",
            C=args.svm_c if svm_c is None else svm_c,
            gamma=gamma,
            class_weight=class_weight,
            probability=args.svm_probability,
        )
        return make_pipeline(StandardScaler(), clf)
    raise ValueError(f"Unsupported model: {model_name}")


def parse_gamma_values(values):
    parsed = []
    for value in values:
        if isinstance(value, str):
            lowered = value.lower()
            if lowered in {"scale", "auto"}:
                parsed.append(lowered)
                continue
        parsed.append(float(value))
    return parsed


def resolve_stratified_cv(y, max_splits, random_state):
    _, counts = np.unique(y, return_counts=True)
    if counts.size == 0:
        return None
    min_count = int(counts.min())
    if min_count < 2:
        return None
    splits = min(int(max_splits), min_count)
    return StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)


def resolve_group_cv(groups, max_splits):
    unique = np.unique(groups)
    splits = min(int(max_splits), int(unique.size))
    if splits < 2:
        return None
    return GroupKFold(n_splits=splits)


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
        help="Enable probability estimates (required for confidence gating in realtime).",
    )
    parser.add_argument(
        "--grid-search",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run grid search over SVM C/gamma before training.",
    )
    parser.add_argument(
        "--grid-c",
        nargs="+",
        type=float,
        default=[0.1, 1.0, 10.0, 100.0, 1000.0],
        help="C values for grid search.",
    )
    parser.add_argument(
        "--grid-gamma",
        nargs="+",
        default=["scale", "auto", "1e-4", "1e-3", "1e-2", "1e-1"],
        help="Gamma values for grid search (numbers or 'scale'/'auto').",
    )
    parser.add_argument(
        "--fit-all",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refit the final model on all data after evaluation.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    X, y, groups, meta = load_dataset(
        args.data_root,
        args.pattern,
        args.min_label_confidence,
        channel_mode=args.channel_mode,
    )
    X_flat = X.reshape(X.shape[0], -1)
    groups = np.asarray(groups, dtype=object)

    indices = np.arange(X_flat.shape[0])
    _, class_counts = np.unique(y, return_counts=True)
    min_class_count = int(class_counts.min()) if class_counts.size else 0
    cv_ok = min_class_count >= 2
    if not cv_ok:
        print("Warning: CV/grid search disabled (a class has <2 samples).")

    full_group_cv = resolve_group_cv(groups, args.cv_splits)
    use_group_split = full_group_cv is not None
    if use_group_split:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=args.test_size,
            random_state=args.random_state,
        )
    train_idx, test_idx = next(splitter.split(X_flat, y, groups))
    print(f"Using group split across {np.unique(groups).size} sessions/files.")
    else:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y,
        )
        print("Warning: group split disabled (not enough sessions/files).")

    X_train = X_flat[train_idx]
    X_test = X_flat[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    groups_train = groups[train_idx] if use_group_split else None

    train_group_cv = resolve_group_cv(groups_train, args.cv_splits) if use_group_split else None
    if not cv_ok:
        train_cv = None
        train_cv_uses_groups = False
    elif train_group_cv is None:
        train_cv = resolve_stratified_cv(y_train, args.cv_splits, args.random_state)
        train_cv_uses_groups = False
    else:
        train_cv = train_group_cv
        train_cv_uses_groups = True

    if use_group_split:
        full_cv = full_group_cv if cv_ok else None
        full_cv_uses_groups = bool(full_cv)
    else:
        full_cv = resolve_stratified_cv(y, args.cv_splits, args.random_state) if cv_ok else None
        full_cv_uses_groups = False

    best_svm_c = args.svm_c
    best_svm_gamma = args.svm_gamma
    grid_results = None

    grid_search_enabled = args.grid_search
    if grid_search_enabled and train_cv is None:
        print("Warning: grid search disabled (training split lacks >=2 samples per class).")
        grid_search_enabled = False

    if grid_search_enabled:
        gamma_grid = parse_gamma_values(args.grid_gamma)
        param_grid = {
            "svc__C": args.grid_c,
            "svc__gamma": gamma_grid,
        }
        base_model = build_model(args.model, args)
        grid = GridSearchCV(
            base_model,
            param_grid=param_grid,
            cv=train_cv,
            scoring="accuracy",
            refit=True,
        )
        if train_cv_uses_groups:
            grid.fit(X_train, y_train, groups=groups_train)
        else:
            grid.fit(X_train, y_train)
        model = grid.best_estimator_
        raw_best_params = grid.best_params_
        best_params = {}
        for key, value in raw_best_params.items():
            if isinstance(value, (np.floating, np.integer)):
                best_params[key] = float(value)
            else:
                best_params[key] = value
        best_svm_c = float(best_params["svc__C"])
        best_svm_gamma = best_params["svc__gamma"]
        if isinstance(best_svm_gamma, (np.floating, np.integer)):
            best_svm_gamma = float(best_svm_gamma)
        grid_results = {
            "enabled": True,
            "best_params": best_params,
            "best_score": float(grid.best_score_),
            "param_grid": param_grid,
        }
        print(f"Grid search best params: C={best_svm_c}, gamma={best_svm_gamma}")
    else:
        model = build_model(args.model, args, svm_c=best_svm_c, svm_gamma=best_svm_gamma)
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    train_accuracy = float(model.score(X_train, y_train))

    cv_scores = None
    if full_cv is not None:
        cv_model = build_model(args.model, args, svm_c=best_svm_c, svm_gamma=best_svm_gamma)
        if full_cv_uses_groups:
            cv_scores = cross_validate(
                cv_model,
                X_flat,
                y,
                cv=full_cv,
                scoring="accuracy",
                return_train_score=False,
                groups=groups,
            )
        else:
            cv_scores = cross_validate(
                cv_model,
                X_flat,
                y,
                cv=full_cv,
                scoring="accuracy",
                return_train_score=False,
            )

    metrics = {
        "cv_accuracy_mean": float(cv_scores["test_score"].mean()) if cv_scores is not None else None,
        "cv_accuracy_std": float(cv_scores["test_score"].std()) if cv_scores is not None else None,
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "train_accuracy": train_accuracy,
        "n_samples": int(X_flat.shape[0]),
        "n_classes": int(len(np.unique(y))),
    }
    if grid_results is not None:
        metrics["grid_search_best_score"] = grid_results["best_score"]

    if metrics["cv_accuracy_mean"] is None:
        print("CV accuracy: n/a (insufficient class counts for CV)")
    else:
        print(f"CV accuracy: {metrics['cv_accuracy_mean']:.3f} +/- {metrics['cv_accuracy_std']:.3f}")
    print(f"Train accuracy: {metrics['train_accuracy']:.3f}")
    print(f"Test accuracy: {metrics['test_accuracy']:.3f}")
    print("\nReport:\n", classification_report(y_test, y_pred))

    final_model = model
    if args.fit_all:
        final_model = build_model(args.model, args, svm_c=best_svm_c, svm_gamma=best_svm_gamma)
        final_model.fit(X_flat, y)

    meta.update(
        {
            "created_at": dt.datetime.now().isoformat(),
            "model_type": args.model,
            "model_params": {
                "svm_c": best_svm_c,
                "svm_gamma": best_svm_gamma,
                "class_weight": args.class_weight,
                "svm_probability": bool(args.svm_probability),
            },
            "metrics": metrics,
            "grid_search": grid_results or {"enabled": False},
            "evaluation_split": {
                "group_split": bool(use_group_split),
                "test_size": float(args.test_size),
                "cv_splits": int(getattr(full_cv, "n_splits", 0)) if full_cv is not None else 0,
            },
            "fit_all_data": bool(args.fit_all),
        }
    )

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    with args.model_out.open("wb") as f:
        pickle.dump({"model": final_model, "metadata": meta}, f)

    print(f"Saved model bundle to {args.model_out}")

    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_out.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved metrics to {args.metrics_out}")


if __name__ == "__main__":
    main()
