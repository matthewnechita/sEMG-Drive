# CNN Improvement Plan: Cross-Subject EMG Gesture Generalization

## 1. Root Cause Analysis: Why Per-Subject Models Outperform Cross-Subject

### 1.1 The Core Problem

EMG signals are person-specific due to four factors that compound on each other:

| Factor | Description | Impact on Current Model |
|--------|-------------|------------------------|
| **Amplitude scaling** | Muscle mass, fat layer thickness, and skin impedance vary across subjects. Two people performing the same gesture produce signals with wildly different absolute magnitudes per channel. | Global z-score normalization (`mean`/`std` computed across all training windows) compresses some subjects and stretches others. The model memorizes absolute amplitude ranges instead of learning shape. |
| **Spatial pattern shift** | Electrode placement varies session-to-session and subject-to-subject. The same muscle activation maps to different channel indices. | The CNN treats channel 0 as always being the same muscle. A 1 cm placement shift can reassign activation patterns across channels, making the model's learned spatial filters invalid. |
| **Temporal dynamics** | Motor unit firing rates, conduction velocity, and fatigue profiles differ by person. | Kernel size 11 at ~2 kHz covers ~5.5 ms. The temporal features it learns are tuned to one subject's firing pattern. |
| **Baseline drift** | Resting EMG tone and electrode impedance change over time and across subjects. | MVC calibration partially addresses this, but `train_cnn.py` computes a single global `mean`/`std` for the cross-subject case, which buries per-subject baseline differences into the normalization. |

### 1.2 Critical Bug in Current Evaluation

`GroupShuffleSplit` in `_train_and_save()` splits by **file path**, not by **subject**. When `PER_SUBJECT_MODELS = False`, windows from the same subject can appear in both train and test. This inflates the reported cross-subject accuracy. The model is never truly evaluated on an unseen subject, so the "poor performance" is actually worse than the numbers suggest.

**Data inventory** (from the filtered file listing):
- Matthew: 8 sessions
- subject01: 4 sessions
- subject02: 7 sessions
- subject03: 6 sessions
- subject04: 5 sessions
- subject05: 4 sessions
- subject06: 8 sessions
- **Total: 7 subjects, 42 sessions**

---

## 2. Proposed Changes, Ranked by Expected Impact

### Change 1: Leave-One-Subject-Out (LOSO) Evaluation [CRITICAL - Do First]

**Why:** Without proper evaluation, you cannot tell whether any other change actually helps. This is the prerequisite for everything else.

**What to change in `train_cnn.py`:**

Add a `LOSO_EVAL` mode that trains on 6 subjects and tests on the 7th, rotating through all 7. This gives a true cross-subject accuracy estimate.

```python
# New config flag
LOSO_EVAL = True  # Leave-One-Subject-Out cross-validation

def loso_evaluate(X, y_idx, subjects, channels, num_classes, device,
                  labels, label_to_index, index_to_label, channel_count):
    unique_subjects = sorted(np.unique(subjects))
    loso_scores = {}

    for held_out in unique_subjects:
        print(f"\n{'='*55}")
        print(f"LOSO: holding out {held_out}")

        train_mask = subjects != held_out
        test_mask = subjects == held_out

        X_train, y_train = X[train_mask], y_idx[train_mask]
        X_test, y_test = X[test_mask], y_idx[test_mask]

        model, mean, std, _ = train_eval_split(
            X_train, y_train, X_test, y_test,
            channels, num_classes, EPOCHS, device,
        )

        # Evaluate
        model.eval()
        X_test_norm = standardize_per_channel(X_test, mean, std).astype(np.float32)
        with torch.no_grad():
            logits = model(torch.from_numpy(X_test_norm).to(device))
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        acc = accuracy_score(y_test, preds)
        loso_scores[held_out] = acc
        print(f"  {held_out}: {acc:.3f} ({test_mask.sum()} windows)")

    mean_acc = np.mean(list(loso_scores.values()))
    std_acc = np.std(list(loso_scores.values()))
    print(f"\nLOSO mean accuracy: {mean_acc:.3f} +/- {std_acc:.3f}")
    return loso_scores
```

**Expected baseline:** Likely 30-50% accuracy (compared to ~78% per-subject), establishing the true gap to close.

---

### Change 2: Per-Window Instance Normalization [HIGH IMPACT]

**Why:** This is the single most impactful architectural change for cross-subject EMG. The core insight: if you normalize each window independently (zero-mean, unit-variance per channel within that window), you remove inter-subject amplitude differences while preserving the temporal shape of the signal. The model then learns **relative activation patterns** rather than absolute magnitudes.

**What to change in `gesture_model_cnn.py`:**

Replace the external z-score normalization with instance normalization inside the model itself. This eliminates the need for saved `mean`/`std` statistics entirely (which are the primary source of subject-specificity in the current bundle).

```python
class GestureCNN(nn.Module):
    def __init__(self, channels, num_classes, dropout=0.2, kernel_size=7,
                 use_instance_norm_input=True):
        super().__init__()
        self.use_instance_norm_input = use_instance_norm_input
        if use_instance_norm_input:
            # Normalize each window independently per channel
            self.input_norm = nn.InstanceNorm1d(
                channels[0], affine=False, track_running_stats=False
            )

        blocks = []
        padding = kernel_size // 2
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            blocks.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, bias=False))
            blocks.append(nn.BatchNorm1d(out_ch))
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(nn.MaxPool1d(kernel_size=2, stride=2))
            if dropout > 0:
                blocks.append(nn.Dropout(dropout))
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], num_classes),
        )

    def forward(self, x):
        if self.use_instance_norm_input:
            x = self.input_norm(x)
        x = self.features(x)
        return self.head(x)
```

**Impact on CnnBundle:** When `use_instance_norm_input=True`, the saved `mean`/`std` in the bundle become vestigial (store zeros/ones). The `standardize()` method in `CnnBundle` can detect this via metadata and skip external normalization. This keeps backward compatibility.

**Impact on `realtime_gesture_cnn.py`:** Minimal. The realtime code currently calls `bundle.standardize(window)` before prediction. With instance norm inside the model, this call becomes a no-op (identity transform), and per-window normalization happens automatically in the forward pass. No timing impact since instance norm on a single (1, C, 200) tensor is negligible.

**Why not BatchNorm for this?** BatchNorm computes running statistics across the training set, making it subject-dependent. InstanceNorm computes statistics per-sample, making each inference self-contained.

---

### Change 3: EMG-Specific Data Augmentation [HIGH IMPACT]

**Why:** With only 7 subjects, the model has very limited exposure to inter-subject variability. Data augmentation can simulate the amplitude, timing, and spatial variations that differ across subjects, effectively multiplying your training diversity.

**Augmentations to implement (in order of importance):**

```python
import torch

class EMGAugmentor:
    """Apply during training only. All ops work on (C, T) tensors."""

    def __init__(self, p=0.5):
        self.p = p  # probability of applying each augmentation

    def channel_amplitude_scaling(self, x):
        """Simulate different muscle mass / electrode impedance per channel.
        Each channel gets a random scale factor."""
        if torch.rand(1) > self.p:
            return x
        scales = torch.empty(x.shape[0], 1).uniform_(0.5, 2.0)
        return x * scales

    def gaussian_noise(self, x, snr_db_range=(10, 30)):
        """Add Gaussian noise at a random SNR."""
        if torch.rand(1) > self.p:
            return x
        snr_db = torch.empty(1).uniform_(*snr_db_range).item()
        signal_power = (x ** 2).mean()
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(x) * noise_power.sqrt()
        return x + noise

    def temporal_shift(self, x, max_shift=20):
        """Shift the signal in time (circular) to simulate timing variability."""
        if torch.rand(1) > self.p:
            return x
        shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        return torch.roll(x, shift, dims=1)

    def channel_dropout(self, x, max_drop=1):
        """Zero out a random channel to simulate electrode failure / poor contact."""
        if torch.rand(1) > self.p:
            return x
        n_drop = torch.randint(1, max_drop + 1, (1,)).item()
        channels_to_drop = torch.randperm(x.shape[0])[:n_drop]
        x = x.clone()
        x[channels_to_drop] = 0.0
        return x

    def temporal_stretch(self, x, range=(0.9, 1.1)):
        """Slight time warping to simulate different conduction velocities."""
        if torch.rand(1) > self.p:
            return x
        factor = torch.empty(1).uniform_(*range).item()
        new_len = int(x.shape[1] * factor)
        # Use interpolate to resample, then crop/pad to original length
        x_3d = x.unsqueeze(0)  # (1, C, T)
        x_resampled = torch.nn.functional.interpolate(
            x_3d, size=new_len, mode='linear', align_corners=False
        ).squeeze(0)  # (C, new_len)
        T = x.shape[1]
        if x_resampled.shape[1] >= T:
            return x_resampled[:, :T]
        else:
            pad = torch.zeros(x.shape[0], T - x_resampled.shape[1])
            return torch.cat([x_resampled, pad], dim=1)

    def __call__(self, x):
        """Apply all augmentations in sequence."""
        x = self.channel_amplitude_scaling(x)
        x = self.gaussian_noise(x)
        x = self.temporal_shift(x)
        x = self.channel_dropout(x)
        x = self.temporal_stretch(x)
        return x
```

**Integration into `train_cnn.py`:** Replace `TensorDataset` with a custom dataset that applies augmentation on-the-fly during training:

```python
class AugmentedEMGDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, augmentor=None):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.augmentor = augmentor

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augmentor is not None:
            x = self.augmentor(x)
        return x, self.y[idx]
```

---

### Change 4: Deeper Architecture with Residual Connections [MEDIUM IMPACT]

**Why:** The current 3-block CNN (32 -> 64 -> 128) is shallow. Each MaxPool2 halves the temporal dimension, so after 3 blocks a 200-sample input is reduced to 25 time steps. Adding residual connections allows the network to be deeper without degradation, and attention helps focus on discriminative temporal regions.

```python
class ResBlock1d(nn.Module):
    def __init__(self, channels, kernel_size=7, dropout=0.2):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.relu(self.block(x) + x))


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style attention over EMG channels."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 1)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 1), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, T)
        w = self.pool(x).squeeze(-1)  # (B, C)
        w = self.fc(w).unsqueeze(-1)  # (B, C, 1)
        return x * w


class GestureCNNv2(nn.Module):
    """Enhanced architecture for cross-subject generalization."""

    def __init__(self, in_channels, num_classes, dropout=0.3, kernel_size=7):
        super().__init__()
        self.input_norm = nn.InstanceNorm1d(in_channels, affine=False,
                                            track_running_stats=False)

        # Stem: project to feature dimension
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 64 channels
        self.stage1 = nn.Sequential(
            ResBlock1d(64, kernel_size, dropout),
            ChannelAttention(64),
            nn.MaxPool1d(2),
        )

        # Stage 2: 64 -> 128
        self.stage2_proj = nn.Sequential(
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            ResBlock1d(128, kernel_size, dropout),
            ChannelAttention(128),
            nn.MaxPool1d(2),
        )

        # Stage 3: 128 -> 128 (no expansion, keep it efficient)
        self.stage3 = nn.Sequential(
            ResBlock1d(128, kernel_size, dropout),
            ChannelAttention(128),
        )

        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.input_norm(x)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2_proj(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)
```

**Parameter count estimate:** ~180K parameters (vs ~115K for current GestureCNN). Still well within real-time budget on CPU.

**Inference latency:** The current model runs in <1ms on CPU for a single window. This model roughly doubles the compute but remains <2ms, far below the ~50ms window step at 2kHz. No latency concern.

---

### Change 5: Subject-Adversarial Training [MEDIUM IMPACT]

**Why:** Even with instance normalization and augmentation, the network may still learn features that correlate with subject identity rather than gesture class. An adversarial subject classifier forces the feature extractor to discard subject-specific information.

This is a domain-adversarial neural network (DANN) approach.

```python
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class SubjectAdversary(nn.Module):
    """Predicts subject ID from features — trained adversarially."""
    def __init__(self, feature_dim, num_subjects):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_subjects),
        )

    def forward(self, features, lambda_=1.0):
        reversed_features = GradientReversalLayer.apply(features, lambda_)
        return self.classifier(reversed_features)
```

**Training loop modification:**

```python
# In the training loop, after computing gesture loss:
# Extract features before the classification head
features = model.get_features(xb)  # output of AdaptiveAvgPool+Flatten, before Linear
subject_logits = subject_adversary(features, lambda_=adv_lambda)
subject_loss = subject_criterion(subject_logits, subject_labels_batch)

total_loss = gesture_loss + adv_weight * subject_loss
total_loss.backward()
```

**Caveat with 7 subjects:** The adversary needs enough subjects to learn meaningful subject-invariant features. With 7 this is borderline; the value comes from combining it with Changes 2+3 rather than using it alone. Start with `adv_weight = 0.1` and tune.

**Training data requirement:** Each training batch needs subject ID labels. The current `train_cnn.py` already tracks `subjects` per window, so this is available.

---

### Change 6: MVC Calibration Normalization Consistency [MEDIUM IMPACT]

**Why:** The current pipeline has a subtle inconsistency. `train_cnn.py` applies MVC calibration per-file during loading (`load_windows_from_file`), then computes a global z-score (`mean`/`std`) on top of the already-calibrated data. This double normalization is redundant and harmful for cross-subject use because the global z-score re-introduces subject-specific statistics.

**Fix:** When using instance normalization (Change 2), keep MVC calibration at the file-loading stage (it helps equalize amplitudes across sessions for the same subject and across subjects), but remove the global z-score step entirely. The instance norm in the model handles the rest.

In `train_eval_split()`, when the model uses instance norm:
```python
# Instead of computing mean/std and applying standardize_per_channel:
if model_uses_instance_norm:
    mean = np.zeros(X_train.shape[1], dtype=np.float32)
    std = np.ones(X_train.shape[1], dtype=np.float32)
    # No external normalization — the model's InstanceNorm handles it
else:
    mean = X_train.mean(axis=(0, 2))
    std = X_train.std(axis=(0, 2))
    std = np.where(std < 1e-6, 1.0, std)
    X_train = standardize_per_channel(X_train, mean, std)
    X_eval = standardize_per_channel(X_eval, mean, std)
```

---

### Change 7: Mixup Across Subjects [LOW-MEDIUM IMPACT]

**Why:** Mixup interpolates between samples from different subjects, creating synthetic "in-between" examples that smooth the decision boundary and reduce overfitting to any single subject's distribution.

```python
def mixup_data(x, y, subjects, alpha=0.2):
    """Mixup that preferentially mixes across subjects."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)

    # Shuffle indices, preferring cross-subject pairs
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# In training loop:
xb_mixed, ya, yb, lam = mixup_data(xb, yb, subject_batch)
logits = model(xb_mixed)
loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
```

---

## 3. Implementation Roadmap

### Phase 1: Evaluation Infrastructure (Week 1)

1. **Add LOSO evaluation to `train_cnn.py`** (Change 1)
2. Run LOSO with the current architecture to establish a baseline cross-subject accuracy
3. Record per-subject and per-gesture confusion matrices

### Phase 2: Core Normalization + Augmentation (Week 2)

4. **Implement instance normalization** in `GestureCNN` (Change 2)
5. **Implement EMG augmentation** pipeline (Change 3)
6. **Fix the double-normalization** issue (Change 6)
7. Re-run LOSO to measure improvement

### Phase 3: Architecture Upgrade (Week 3)

8. **Implement GestureCNNv2** with residual blocks and channel attention (Change 4)
9. Run LOSO comparison: GestureCNNv2 vs GestureCNN (both with instance norm + augmentation)
10. If GestureCNNv2 wins, update `_resolve_architecture` in `gesture_model_cnn.py` to support both architectures via the `architecture.type` field in the bundle

### Phase 4: Adversarial Training (Week 3-4)

11. **Add subject-adversarial training** (Change 5)
12. **Add mixup** (Change 7)
13. Re-run LOSO; compare all configurations

### Phase 5: Deployment (Week 4)

14. Train the final cross-subject model on all 7 subjects
15. Update bundle format to include `architecture.type = "GestureCNNv2"` and `architecture.use_instance_norm_input = True`
16. Update `realtime_gesture_cnn.py` to handle the new architecture
17. Verify real-time inference latency

---

## 4. Changes Required in `realtime_gesture_cnn.py`

The changes to the realtime script are minimal:

1. **`load_cnn_bundle` in `gesture_model_cnn.py`**: Update `_resolve_architecture` to handle `type = "GestureCNNv2"` and instantiate the correct class. The bundle format remains the same dictionary structure.

2. **Normalization path**: When the bundle metadata indicates `use_instance_norm_input = True`, `CnnBundle.standardize()` should return the input unchanged (the model handles normalization internally). The realtime code does not need to change since it already calls `bundle.standardize()`.

3. **MVC calibration**: Keep as-is. The realtime calibration step (`CALIBRATE = True`) normalizes raw EMG by neutral baseline and MVC. This is compatible with instance norm since it just rescales the input before the model's own normalization.

4. **No changes to windowing, smoothing, or confidence gating.** These remain identical.

```python
# In gesture_model_cnn.py, updated CnnBundle.standardize():
def standardize(self, X: np.ndarray) -> np.ndarray:
    if self.metadata.get("use_instance_norm_input", False):
        return X  # Model handles normalization internally
    mean = self.mean.reshape(1, -1, 1)
    std = self.std.reshape(1, -1, 1)
    return (X - mean) / std
```

---

## 5. Bundle Format Compatibility

The updated bundle dictionary will look like:

```python
bundle = {
    "model_state": model.state_dict(),
    "normalization": {
        "mean": np.zeros(channel_count, dtype=np.float32),  # vestigial
        "std": np.ones(channel_count, dtype=np.float32),     # vestigial
    },
    "label_to_index": label_to_index,
    "index_to_label": index_to_label,
    "metadata": {
        # ... existing fields ...
        "use_instance_norm_input": True,
    },
    "architecture": {
        "type": "GestureCNNv2",  # or "GestureCNN" for backward compat
        "channels": [channel_count, 64, 128, 128],
        "dropout": 0.3,
        "kernel_size": 7,
        "use_instance_norm_input": True,
    },
}
```

Old bundles without `use_instance_norm_input` will default to `False`, preserving full backward compatibility with existing per-subject models.

---

## 6. Verification Questions

**Q1: With only 7 subjects, is subject-adversarial training (DANN) likely to help, or will it overfit to the 7 training subject identities rather than learning truly subject-invariant features?**

**A1:** This is a real concern. With only 7 subjects, the adversary has a trivially easy classification task (7 classes), and the gradient reversal may not produce features that generalize to an 8th unseen subject. The adversary may learn to distinguish subjects by session-specific artifacts rather than genuine inter-subject differences. **Mitigation:** Use a weak adversary (small hidden size, low learning rate) and a small `adv_weight` (0.05-0.1). More importantly, rely on instance normalization + augmentation as the primary generalization mechanism, and treat DANN as an optional refinement. During LOSO evaluation, if DANN does not improve held-out subject accuracy, drop it.

**Q2: Does instance normalization destroy discriminative information for gestures like "neutral" that have low amplitude across all channels?**

**A2:** Yes, this is a risk. Instance normalization on a near-zero signal amplifies noise to unit variance, potentially making neutral windows look like active gestures. **Mitigation:** Add a small epsilon or a learnable affine transform. Better yet, add a "signal energy" feature: compute the RMS of the raw window before normalization and concatenate it as an extra scalar input to the classifier head. This preserves the ability to distinguish "nothing is happening" from "something is happening" while still normalizing the pattern of what is happening.

```python
# In GestureCNNv2.forward():
raw_energy = x.pow(2).mean(dim=(1, 2), keepdim=False)  # (B,)
x = self.input_norm(x)
# ... conv stages ...
features = self.pool_flatten(x)  # (B, 128)
features = torch.cat([features, raw_energy.unsqueeze(1)], dim=1)  # (B, 129)
return self.classifier(features)  # Linear(129, num_classes)
```

**Q3: The current WINDOW_SIZE=200 at ~2kHz gives 100ms windows. Is this sufficient temporal context for cross-subject gesture recognition, or would longer windows help?**

**A3:** 100ms is on the short side for cross-subject work. Longer windows capture more of the gesture's temporal envelope, which is more consistent across subjects than instantaneous amplitude patterns. However, longer windows increase latency. **Recommendation:** Test WINDOW_SIZE=400 (200ms) with WINDOW_STEP=100 (unchanged). The model's AdaptiveAvgPool1d(1) already handles variable input lengths, so this is a config-only change. If 200ms windows improve LOSO accuracy without unacceptable latency, adopt them. The existing experimental bundles (200-100.pt, 250-125.pt) suggest this was already explored for per-subject models.

**Q4: Channel permutation augmentation could address electrode placement variability, but does it make sense for EMG where each channel corresponds to a specific muscle?**

**A4:** Full random channel permutation would destroy the spatial structure. However, **adjacent channel swaps** (swapping channels i and i+1 with some probability) simulate small electrode placement shifts and could help. Alternatively, if the electrode array has a known geometry (e.g., linear array), you could apply small spatial translations. For this project, the Delsys Trigno sensors are placed on specific muscles, so aggressive channel permutation is inappropriate. **Decision:** Do not use channel permutation. Rely on channel amplitude scaling (Change 3) to simulate placement-related impedance differences.

**Q5: How do you handle the model architecture change in `_resolve_architecture` while maintaining backward compatibility with existing per-subject bundles that use the original `GestureCNN`?**

**A5:** The `_resolve_architecture` function already dispatches on `arch.get("type")`. Add a second branch for `"GestureCNNv2"`:

```python
def _resolve_architecture(bundle, in_channels, num_classes):
    arch = bundle.get("architecture") or {}
    arch_type = arch.get("type", "GestureCNN")

    if arch_type == "GestureCNNv2":
        # New architecture
        return arch_type, in_channels, num_classes, arch
    elif arch_type == "GestureCNN":
        # Legacy path
        channels = arch.get("channels") or [in_channels, 32, 64, 128]
        # ... existing logic ...
        return arch_type, channels, dropout, kernel_size
```

Then in `load_cnn_bundle`, instantiate the correct class:

```python
if arch_type == "GestureCNNv2":
    model = GestureCNNv2(in_channels, num_classes, ...)
else:
    model = GestureCNN(channels, num_classes, dropout, kernel_size)
```

This ensures old `.pt` files with `type = "GestureCNN"` (or no type field) load correctly, while new cross-subject bundles use `GestureCNNv2`.

---

## 7. Revision Based on Verification Analysis

After answering the verification questions, the following changes were made to the plan:

### Revision 1: Add signal energy bypass to handle neutral gesture under instance normalization

**What changed:** Added a raw energy scalar that bypasses instance normalization and is concatenated to the feature vector before the final classifier. This is described in the answer to Q2 above.

**Why:** Instance normalization is the plan's most impactful change, but it has a specific failure mode on low-energy windows (neutral gesture). Without this fix, the neutral class would degrade significantly, defeating the purpose since neutral is the most common class during real-time use (resting state). The energy bypass costs exactly 1 extra parameter in the final linear layer (129 -> 6 instead of 128 -> 6) and adds negligible compute.

### Revision 2: Downgrade subject-adversarial training from HIGH to MEDIUM impact

**What changed:** Moved DANN (Change 5) from Phase 2 to Phase 4, and added explicit guidance to treat it as optional.

**Why:** With only 7 subjects, the adversary is likely to overfit to training subject identities. The expected ROI is lower than instance normalization or augmentation, and it adds significant training complexity (gradient reversal scheduling, hyperparameter tuning for `adv_weight` and `lambda_` schedule). Better to get the simpler changes working first and measure their impact before adding this complexity.

### Revision 3: Added recommendation to test WINDOW_SIZE=400

**What changed:** Added a note under Q3 recommending a test with 400-sample windows.

**Why:** Longer windows capture more of the gesture temporal envelope, which is a more subject-invariant feature than instantaneous amplitude. This is a zero-code-change experiment (config only) that could yield meaningful improvement. The existing experimental bundles (250-125.pt) suggest the team has already explored window size variations, so extending to 400 is a natural next step.

### Revision 4: Explicitly excluded channel permutation augmentation

**What changed:** In the augmentation list (Change 3), confirmed that channel permutation is NOT included, and explained why.

**Why:** The Delsys Trigno sensors are placed on specific muscles. Unlike dense electrode arrays, there is no spatial continuity between channels. Randomly permuting channels would create physiologically impossible signal patterns, teaching the model the wrong invariances. Channel amplitude scaling is the correct augmentation for electrode placement variability in this setup.

---

## 8. Summary of Expected Outcomes

| Change | LOSO Accuracy Improvement (Estimated) | Effort |
|--------|---------------------------------------|--------|
| LOSO Evaluation | N/A (measurement only) | Low |
| Instance Normalization | +15-25% absolute | Low |
| EMG Augmentation | +5-10% | Medium |
| GestureCNNv2 Architecture | +3-7% | Medium |
| Subject-Adversarial (DANN) | +0-5% | High |
| Normalization Consistency Fix | +2-5% | Low |
| Mixup | +1-3% | Low |

**Estimated total LOSO accuracy with all changes:** 60-75% (up from an estimated 30-50% baseline). This is a realistic target for 7 subjects with 6 gesture classes. Achieving >70% LOSO would make the cross-subject model viable for real-time use with the existing smoothing and confidence gating (SMOOTHING=11, MIN_CONFIDENCE=0.65).

**If the target is not met:** Consider collecting 2-3 more subjects. Cross-subject EMG generalization in the literature typically requires 10+ subjects to reach 80%+ accuracy with 6+ gesture classes. Alternatively, implement a **few-shot calibration** mode: collect 30 seconds of neutral + 1 repetition of each gesture from the new subject, and fine-tune only the classifier head (freeze the feature extractor). This hybrid approach gives cross-subject generalization with minimal per-subject data collection.
