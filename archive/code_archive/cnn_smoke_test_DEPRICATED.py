import tempfile

import numpy as np
import torch

from gesture_model_cnn import GestureCNN, load_cnn_bundle
from train_cnn import majority_label, standardize_per_channel


def main():
    # Basic majority label sanity (bytes + None)
    labels = np.array([b"left_turn", b"left_turn", None], dtype=object)
    assert majority_label(labels) == "left_turn"

    # Model forward shape
    channels = 4
    num_classes = 3
    model = GestureCNN([channels, 8, 16], num_classes, dropout=0.0, kernel_size=5)
    X = np.random.randn(2, channels, 200).astype(np.float32)
    logits = model(torch.from_numpy(X))
    assert logits.shape == (2, num_classes)

    # Standardization shape
    mean = np.zeros(channels, dtype=np.float32)
    std = np.ones(channels, dtype=np.float32)
    X_std = standardize_per_channel(X, mean, std)
    assert X_std.shape == X.shape

    # Save/load bundle + predict_proba
    bundle = {
        "model_state": model.state_dict(),
        "normalization": {"mean": mean, "std": std},
        "label_to_index": {"a": 0, "b": 1, "c": 2},
        "index_to_label": {0: "a", 1: "b", 2: "c"},
        "metadata": {},
        "architecture": {
            "type": "GestureCNN",
            "channels": [channels, 8, 16],
            "dropout": 0.0,
            "kernel_size": 5,
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/cnn_bundle.pt"
        torch.save(bundle, path)
        loaded = load_cnn_bundle(path)
        probs = loaded.predict_proba(X)
        assert probs.shape == (2, num_classes)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    print("cnn_smoke_test ok")


if __name__ == "__main__":
    main()
