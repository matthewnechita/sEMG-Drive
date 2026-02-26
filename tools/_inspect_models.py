"""Print metadata from all .pt model bundles."""
import sys
import json
from pathlib import Path
import torch
import numpy as np

models_dir = Path("models")
pt_files = sorted(models_dir.glob("*.pt"))

if not pt_files:
    print("No .pt files found in models/")
    sys.exit(0)

for pt in pt_files:
    print(f"\n{'='*60}")
    print(f"MODEL: {pt.name}")
    try:
        bundle = torch.load(pt, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  ERROR loading: {e}")
        continue

    arch = bundle.get("architecture", {})
    meta = bundle.get("metadata", {})
    norm = bundle.get("normalization", {})

    print(f"  arch type      : {arch.get('type', 'GestureCNN (legacy)')}")
    print(f"  subject        : {meta.get('subject', '?')}")
    print(f"  labels         : {meta.get('labels', '?')}")
    print(f"  channel_count  : {meta.get('channel_count', '?')}")
    print(f"  window_size    : {meta.get('window_size_samples', '?')}")
    print(f"  window_step    : {meta.get('window_step_samples', '?')}")
    print(f"  instance_norm  : {meta.get('use_instance_norm_input', False)}")
    print(f"  calibration    : {meta.get('calibration_used', '?')}")
    print(f"  split_mode     : {meta.get('split_mode', '?')}")

    metrics = meta.get("metrics", {})
    print(f"  test_accuracy  : {metrics.get('test_accuracy', '?')}")

    training = meta.get("training", {})
    print(f"  epochs         : {training.get('epochs', '?')}")
    print(f"  use_augment    : {training.get('use_augmentation', '?')}")
    print(f"  use_balanced   : {training.get('use_balanced_sampling', '?')}")
    print(f"  use_mixup      : {training.get('use_mixup', '?')}")
    print(f"  lr             : {training.get('lr', '?')}")
    print(f"  batch_size     : {training.get('batch_size', '?')}")

    train_files = meta.get("train_files", [])
    test_files = meta.get("test_files", [])
    print(f"  train_files    : {len(train_files)} -> {train_files}")
    print(f"  test_files     : {len(test_files)} -> {test_files}")

    mean = norm.get("mean")
    std = norm.get("std")
    if mean is not None:
        mean_arr = np.asarray(mean)
        std_arr = np.asarray(std)
        print(f"  norm mean range: [{mean_arr.min():.4f}, {mean_arr.max():.4f}]")
        print(f"  norm std range : [{std_arr.min():.4f}, {std_arr.max():.4f}]")

    state = bundle.get("model_state") or bundle.get("model", {})
    n_params = sum(v.numel() for v in state.values() if hasattr(v, 'numel'))
    print(f"  param count    : {n_params:,}")

print("\nDone.")
