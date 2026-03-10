# CNN Improvement Plan: Cross-Subject EMG Gesture Recognition

## 1. Root Cause Analysis

### Why per-subject models outperform the cross-subject model

The current cross-subject model (`gesture_cnn.pt`) fails because EMG signals vary dramatically between subjects due to four compounding factors:

**1a. Amplitude scale differences.** Different subjects have different muscle mass, skin impedance, electrode-skin contact quality, and subcutaneous fat thickness. Subject A's "horn" activation might produce amplitudes 3x larger than Subject B's across all 17 channels. The current per-channel z-score normalization (`mean`/`std` computed over training data) averages these scales together, creating a normalization that fits no one well.

**1b. Spatial pattern differences (channel weighting).** Even with identical electrode placement instructions, electrodes shift by millimeters between subjects. Because EMG is highly localized, a 5mm shift can move a channel from directly over a muscle belly to a tendon or neighboring muscle. The CNN's first convolutional layer learns fixed spatial filters that break when channel-to-muscle mapping changes.

**1c. Temporal pattern differences.** Different subjects perform the same gesture with different contraction dynamics -- speed of onset, co-contraction patterns, fatigue profiles. The CNN's temporal convolutions (kernel_size=11 at ~2kHz = 5ms receptive field per layer) learn subject-specific temporal signatures.

**1d. Evaluation methodology is misleading.** The current `GroupShuffleSplit` splits by *file path*, not by *subject*. When training a "cross-subject" model with `PER_SUBJECT_MODELS=False`, the test set contains windows from the same subjects seen during training (just different sessions). This inflates the reported accuracy. True cross-subject generalization has never been properly measured.

### Summary of the core problem

The model learns to classify `(subject_identity + gesture)` rather than `gesture` alone. The subject-specific signal characteristics dominate the learned features, making the representation brittle to new subjects.

---

## 2. Current System Inventory

| Component | Current State | Limitation |
|-----------|--------------|------------|
| Architecture | GestureCNN: Conv1d [17->32->64->128], kernel=11, dropout=0.4 | No mechanism to factor out subject variability |
| Normalization | Per-channel z-score from training set means/stds | Global stats blur subject-specific scales |
| Calibration | MVC-based `(emg - neutral_mean) / mvc_scale` in data loading | Applied inconsistently; calibration data exists per-session but normalization stats are computed *after* calibration, creating double-normalization |
| Training split | GroupShuffleSplit by file path | Does not measure cross-subject performance |
| Data | 7 subjects, 4-8 sessions each, 42 files total, 17 channels, ~2kHz | Small subject count; sessions within a subject are highly correlated |
| Augmentation | None | No robustness to amplitude/spatial variation |
| Inference | WINDOW_SIZE=200, WINDOW_STEP=100, SMOOTHING=11 | ~50ms window, ~100ms latency budget |

---

## 3. Proposed Changes (Ranked by Expected Impact)

### Change 1: Leave-One-Subject-Out (LOSO) Evaluation [CRITICAL -- do first]

**Impact: Does not improve the model but is required to measure anything meaningful.**

The current evaluation is broken for cross-subject work. Before any model changes, implement proper LOSO cross-validation.

**Implementation:**

```python
# In train_cnn.py, add a LOSO evaluation function

def loso_cross_validation(X, y_idx, subjects, channels, num_classes, device, epochs):
    """Leave-one-subject-out cross-validation."""
    unique_subjects = sorted(np.unique(subjects))
    results = {}

    for held_out in unique_subjects:
        print(f"\n--- LOSO fold: holding out {held_out} ---")
        train_mask = subjects != held_out
        test_mask = subjects == held_out

        model, mean, std, _ = train_eval_split(
            X[train_mask], y_idx[train_mask],
            X[test_mask], y_idx[test_mask],
            channels, num_classes, epochs, device,
        )

        # Evaluate
        X_test_norm = standardize_per_channel(X[test_mask], mean, std).astype(np.float32)
        with torch.no_grad():
            logits = model(torch.from_numpy(X_test_norm).to(device))
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        acc = accuracy_score(y_idx[test_mask], preds)
        results[held_out] = acc
        print(f"  {held_out}: {acc:.3f}")

    mean_acc = np.mean(list(results.values()))
    print(f"\nLOSO mean accuracy: {mean_acc:.3f}")
    return results
```

**Success metric:** Establish the LOSO baseline accuracy for the current architecture. Expected: 30-50% (near chance for 6 classes = 16.7%).

---

### Change 2: Subject-Adaptive Normalization via Calibration Alignment [HIGH IMPACT]

**Rationale:** The single highest-impact change is fixing the normalization so that every subject's data lands in a comparable feature space *before* the CNN sees it. The calibration infrastructure already exists (neutral + MVC collection) but is underutilized.

**What changes:**

Instead of using global training-set z-score stats at inference, normalize each subject's data using their own calibration session. The bundle stores no `mean`/`std` -- instead, the realtime script computes them from the calibration period.

**Implementation -- Training side:**

```python
def per_session_normalize(emg, calib_neutral, calib_mvc, mvc_percentile=95.0):
    """Normalize EMG using session-specific calibration data.

    After this, all subjects' data should have:
    - neutral ~= 0
    - MVC ~= 1
    across all channels.
    """
    neutral_mean = np.mean(calib_neutral, axis=0)  # shape: (17,)
    mvc_ref = np.percentile(calib_mvc, mvc_percentile, axis=0)  # shape: (17,)
    mvc_ref = np.where(mvc_ref < 1e-6, 1.0, mvc_ref)

    emg_norm = (emg - neutral_mean) / mvc_ref
    return emg_norm
```

During training, each session file is normalized by its *own* calibration data (which already happens via `USE_CALIBRATION=True`). The key change: **do not apply a second global z-score normalization on top**. Instead, after per-session calibration normalization, apply only a light global clipping/scaling:

```python
# After per-session MVC normalization, clip and scale to [-1, 1] range
emg_norm = np.clip(emg_norm, -5.0, 5.0) / 5.0
```

The bundle then stores no `mean`/`std` (or stores sentinel values `mean=0, std=1`), and the realtime script relies solely on the calibration period.

**Implementation -- Inference side (realtime_gesture_cnn.py):**

The existing calibration flow already computes `neutral_mean` and `mvc_scale`. The only change is removing the `bundle.standardize(window)` call and replacing it with the same clip-and-scale used in training:

```python
# Replace:
#   window = bundle.standardize(window)
# With:
window = np.clip(window, -5.0, 5.0) / 5.0
```

**Why this works:** MVC normalization maps every subject into a common physiological coordinate system: 0 = resting, 1 = maximum voluntary contraction. This removes the dominant source of inter-subject variance (amplitude scale) while preserving the gesture-discriminative signal structure.

**Compatibility with CnnBundle:** The bundle format is unchanged -- just set `mean=np.zeros(17)` and `std=np.ones(17)` so that `bundle.standardize()` becomes a no-op. Store a flag in metadata: `"normalization_mode": "mvc_calibration"`.

---

### Change 3: Channel-Wise Amplitude Augmentation [HIGH IMPACT]

**Rationale:** Even with MVC normalization, residual amplitude differences exist. Training with random per-channel scaling makes the CNN robust to these variations.

**Implementation:**

```python
class EMGAugmenter:
    def __init__(self, channel_scale_range=(0.5, 2.0), noise_std=0.05,
                 channel_dropout_prob=0.1, time_shift_max=10):
        self.channel_scale_range = channel_scale_range
        self.noise_std = noise_std
        self.channel_dropout_prob = channel_dropout_prob
        self.time_shift_max = time_shift_max

    def __call__(self, x):
        """x: shape (channels, time)"""
        C, T = x.shape

        # Per-channel random amplitude scaling
        lo, hi = self.channel_scale_range
        scales = np.random.uniform(lo, hi, size=(C, 1)).astype(np.float32)
        x = x * scales

        # Additive Gaussian noise
        if self.noise_std > 0:
            x = x + np.random.randn(*x.shape).astype(np.float32) * self.noise_std

        # Random channel dropout (simulate electrode failure / different placement)
        if self.channel_dropout_prob > 0:
            mask = np.random.rand(C, 1) > self.channel_dropout_prob
            x = x * mask.astype(np.float32)

        # Random temporal shift (simulate different reaction times)
        if self.time_shift_max > 0:
            shift = np.random.randint(-self.time_shift_max, self.time_shift_max + 1)
            x = np.roll(x, shift, axis=1)

        return x
```

**Integration:** Apply augmentation on-the-fly in a custom PyTorch Dataset that wraps the training windows:

```python
class AugmentedEMGDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, augmenter=None):
        self.X = X  # shape (N, C, T)
        self.y = y
        self.augmenter = augmenter

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        if self.augmenter is not None:
            x = self.augmenter(x)
        return torch.from_numpy(x), self.y[idx]
```

---

### Change 4: Subject-Adversarial Training (Domain-Adversarial Neural Network) [HIGH IMPACT]

**Rationale:** Force the feature extractor to learn representations that are *gesture-discriminative* but *subject-invariant*. This directly attacks the root cause: the CNN encoding subject identity into its features.

**Architecture modification:**

```
                         +---> Gesture Classifier (6 classes) [maximize accuracy]
                        /
Input -> Conv Backbone -> Feature Vector
                        \
                         +---> Subject Classifier (7 classes) [gradient reversal]
```

**Implementation:**

```python
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class GestureCNNAdversarial(nn.Module):
    def __init__(self, channels, num_classes, num_subjects, dropout=0.4, kernel_size=11):
        super().__init__()
        # Shared feature extractor (same as current GestureCNN.features)
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
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Gesture classifier head
        self.gesture_head = nn.Linear(channels[-1], num_classes)

        # Subject discriminator (with gradient reversal)
        self.subject_head = nn.Sequential(
            GradientReversal(lambda_=1.0),
            nn.Linear(channels[-1], 64),
            nn.ReLU(),
            nn.Linear(64, num_subjects),
        )

    def forward(self, x, return_subject=False):
        feat = self.features(x)
        feat = self.pool(feat).squeeze(-1)  # (B, 128)
        gesture_logits = self.gesture_head(feat)
        if return_subject:
            subject_logits = self.subject_head(feat)
            return gesture_logits, subject_logits
        return gesture_logits
```

**Training loop modification:**

```python
# Combined loss
gesture_loss = F.cross_entropy(gesture_logits, gesture_labels, weight=class_weight)
subject_loss = F.cross_entropy(subject_logits, subject_labels)
total_loss = gesture_loss + alpha * subject_loss  # alpha controls adversarial strength

# Ramp alpha from 0 to 1 over training to stabilize early learning
alpha = 2.0 / (1.0 + np.exp(-10 * epoch / total_epochs)) - 1.0
```

**Deployment:** At inference time, only the `features` + `gesture_head` path is used. The subject discriminator is discarded. The saved bundle stores only the feature extractor + gesture head weights, so it remains compatible with the existing bundle format. The `GestureCNNAdversarial.forward()` with `return_subject=False` produces identical output shape to the current `GestureCNN`.

**Bundle compatibility:** Store the adversarial model's feature extractor + gesture head as a standard `GestureCNN` by extracting those weights:

```python
# After training, create a standard GestureCNN for the bundle
deploy_model = GestureCNN(channels, num_classes, dropout, kernel_size)
deploy_model.features.load_state_dict(adv_model.features.state_dict())
# Reconstruct head to match GestureCNN's head (AdaptiveAvgPool + Flatten + Linear)
deploy_model.head[2].weight.data = adv_model.gesture_head.weight.data
deploy_model.head[2].bias.data = adv_model.gesture_head.bias.data
```

---

### Change 5: Improved Architecture -- Deeper + Residual Connections [MEDIUM IMPACT]

**Rationale:** The current 3-layer CNN (17->32->64->128) is shallow. Deeper networks with residual connections learn more abstract, transferable features. The model is small enough that adding depth is free in terms of real-time latency.

```python
class ResBlock1d(nn.Module):
    def __init__(self, channels, kernel_size=11, dropout=0.4):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class GestureCNNv2(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0.4, kernel_size=11):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.stage1 = nn.Sequential(ResBlock1d(64, kernel_size, dropout), nn.MaxPool1d(2))
        self.stage2 = nn.Sequential(
            nn.Conv1d(64, 128, 1, bias=False), nn.BatchNorm1d(128),
            ResBlock1d(128, kernel_size, dropout), nn.MaxPool1d(2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        return self.head(x)
```

**Latency check:** Input is (17, 200). After stem+pool: (64, 100). After stage1: (64, 50). After stage2: (128, 25). AdaptiveAvgPool to (128, 1). Total: ~6 conv layers, ~50K parameters. Inference on CPU at this size is <1ms per window. Well within real-time budget.

---

### Change 6: Electrode-Shift Augmentation (Spatial) [MEDIUM IMPACT]

**Rationale:** Simulates the dominant physical difference between subjects: electrode array displacement on the skin surface.

```python
def electrode_shift_augmentation(x, max_shift=2):
    """Circularly shift channels to simulate electrode placement variation.
    x: shape (channels, time)
    """
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(x, shift, axis=0)
```

This is only meaningful if channels correspond to a spatially ordered array (e.g., a Delsys Trigno grid). If the 17 channels are from an ordered sensor strip or array, circular shifting is a valid approximation of physical displacement. If channels are from arbitrary body locations, skip this augmentation.

---

### Change 7: Session-Mixing Training Strategy [MEDIUM IMPACT]

**Rationale:** Currently, each training batch is randomly sampled from all windows. This means a batch could be dominated by one subject. Instead, ensure each batch contains balanced representation from multiple subjects.

```python
class SubjectBalancedSampler(torch.utils.data.Sampler):
    """Yields batches with equal representation from each subject."""

    def __init__(self, subject_ids, batch_size):
        self.subject_ids = np.asarray(subject_ids)
        self.unique_subjects = np.unique(self.subject_ids)
        self.batch_size = batch_size
        self.per_subject = batch_size // len(self.unique_subjects)

        self.subject_indices = {
            s: np.where(self.subject_ids == s)[0]
            for s in self.unique_subjects
        }

    def __iter__(self):
        # Shuffle within each subject
        shuffled = {
            s: np.random.permutation(idx)
            for s, idx in self.subject_indices.items()
        }
        # Yield balanced mini-batches
        pointers = {s: 0 for s in self.unique_subjects}
        total = sum(len(v) for v in shuffled.values())
        yielded = 0

        while yielded < total:
            batch = []
            for s in self.unique_subjects:
                idx = shuffled[s]
                ptr = pointers[s]
                end = min(ptr + self.per_subject, len(idx))
                if ptr >= len(idx):
                    # Reshuffle and restart for this subject
                    shuffled[s] = np.random.permutation(self.subject_indices[s])
                    ptr = 0
                    end = min(self.per_subject, len(shuffled[s]))
                batch.extend(idx[ptr:end].tolist())
                pointers[s] = end
            np.random.shuffle(batch)
            yield from batch
            yielded += len(batch)

    def __len__(self):
        return sum(len(v) for v in self.subject_indices.values())
```

---

### Change 8: Prototypical Network for Few-Shot Calibration [LOWER IMPACT, HIGHER COMPLEXITY]

**Rationale:** If LOSO accuracy remains below an acceptable threshold (e.g., <70%) after Changes 1-7, a prototypical network allows the system to adapt to a new subject with just a few labeled examples per gesture (e.g., one 5-second session).

**Concept:** Instead of classifying directly, the backbone produces an embedding. During calibration, the new subject performs each gesture briefly. The system computes a "prototype" (mean embedding) for each gesture from the calibration data. At inference, classify by nearest-prototype in embedding space.

```python
class ProtoNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone  # GestureCNNv2 without the final Linear layer

    def embed(self, x):
        """x: (B, C, T) -> (B, D) embedding"""
        return self.backbone(x)  # returns (B, 128) from AdaptiveAvgPool

    def compute_prototypes(self, support_x, support_y, num_classes):
        """Compute class prototypes from support set."""
        embeddings = self.embed(support_x)  # (N, D)
        prototypes = torch.zeros(num_classes, embeddings.shape[1], device=embeddings.device)
        for c in range(num_classes):
            mask = support_y == c
            if mask.any():
                prototypes[c] = embeddings[mask].mean(dim=0)
        return prototypes

    def forward(self, x, prototypes):
        """Classify by distance to prototypes."""
        embeddings = self.embed(x)  # (B, D)
        # Negative squared Euclidean distance
        dists = -torch.cdist(embeddings, prototypes)  # (B, num_classes)
        return dists  # treat as logits
```

**Deployment scenario:** The new-subject experience becomes:
1. Strap on sensors.
2. Perform a 30-second calibration: ~5 seconds per gesture (prompted on screen).
3. System computes MVC normalization + prototype embeddings.
4. Real-time inference uses prototype-based classification.

This is more complex but provides a graceful fallback. The calibration time is comparable to the current MVC calibration that already exists.

---

## 4. Implementation Roadmap

### Phase 1: Measurement (1-2 days)
1. Implement LOSO evaluation in `train_cnn.py` (Change 1).
2. Run LOSO with the current architecture to establish baseline.
3. Record per-subject and mean accuracy.

### Phase 2: Normalization Fix (1-2 days)
4. Switch training to rely solely on per-session MVC normalization (Change 2).
5. Remove the second z-score layer.
6. Re-run LOSO to measure improvement.

### Phase 3: Augmentation + Adversarial Training (3-5 days)
7. Implement `EMGAugmenter` and `AugmentedEMGDataset` (Change 3).
8. Implement `GestureCNNAdversarial` with gradient reversal (Change 4).
9. Implement `SubjectBalancedSampler` (Change 7).
10. Re-run LOSO with all three changes combined.

### Phase 4: Architecture Upgrade (2-3 days)
11. Implement `GestureCNNv2` with residual connections (Change 5).
12. Add electrode-shift augmentation if channel layout is spatially ordered (Change 6).
13. Re-run LOSO.

### Phase 5: Prototypical Network (optional, 3-5 days)
14. Only if Phase 3+4 LOSO accuracy is below 65%.
15. Implement `ProtoNet` wrapper (Change 8).
16. Modify realtime script to support calibration-based prototype computation.

### Phase 6: Integration and Deployment
17. Update `gesture_model_cnn.py` to support new architecture variant in `_resolve_architecture`.
18. Update `realtime_gesture_cnn.py` to use MVC-only normalization.
19. Save final model as `models/gesture_cnn_cross_subject.pt` with updated metadata.

---

## 5. Deployment Scenario: New Subject Experience

### Zero-shot (after Changes 1-7):
1. New subject straps on the 17-channel Delsys Trigno sensors.
2. Realtime script starts; subject performs 3s neutral + 3s MVC calibration (already exists).
3. MVC normalization maps their signal into the universal coordinate system.
4. Cross-subject model performs inference immediately. No retraining.
5. Expected accuracy: 60-75% (vs. ~78% per-subject).

### Few-shot (after Change 8, if needed):
1. Same as above, plus an additional 30s gesture calibration (5s per gesture).
2. Prototype embeddings are computed from the calibration.
3. Expected accuracy: 70-80% (approaching per-subject performance).

---

## 6. How to Measure Success

### Primary metric: LOSO accuracy
- For each of the 7 subjects, train on 6 subjects and test on the held-out subject.
- Report: per-subject accuracy, mean accuracy, and standard deviation across folds.

### Secondary metrics:
- **Per-class F1 score** (averaged across LOSO folds): ensures no gesture is systematically misclassified.
- **Confusion matrix analysis**: identify if specific gesture pairs (e.g., `signal_left` vs `left_turn`) are confounded cross-subject.
- **Inference latency**: verify <5ms per window on CPU (currently ~1ms, should remain unchanged).

### Target thresholds:
| Metric | Baseline (estimated) | Target (Phase 3) | Stretch (Phase 5) |
|--------|---------------------|-------------------|-------------------|
| LOSO mean accuracy | ~35% | 60% | 75% |
| Worst-subject accuracy | ~20% | 45% | 60% |
| Per-class F1 (macro) | ~0.30 | 0.55 | 0.70 |

---

## 7. Verification Questions and Self-Answers

### Q1: Is the MVC calibration data actually present and reliable for all 42 session files?

**Answer:** The code in `load_windows_from_file` conditionally applies calibration only if `calib_neutral_emg` and `calib_mvc_emg` are present in the file. Inspecting the data layout, Matthew's sessions have calibration data (`calib_neutral_emg`, `calib_mvc_emg` keys confirmed). However, I did not verify this for all 42 files. If some sessions lack calibration data, the MVC normalization step in Change 2 would silently fall back to unnormalized data, mixing normalized and unnormalized windows in training.

**Mitigation:** Before training, add a validation step that logs which files have/lack calibration data. If calibration is missing, exclude those files or use a per-file z-score fallback (compute mean/std from that file's resting segments). Revised plan includes this check.

### Q2: Does the 17-channel Delsys setup have a spatially ordered electrode layout that makes circular channel shifting meaningful?

**Answer:** The Delsys Trigno system uses individual wireless sensors placed at arbitrary body locations (typically forearm). They are NOT a spatially contiguous grid. Circular shifting of channels would mix unrelated muscle groups and produce physiologically meaningless signals.

**Mitigation:** Remove electrode-shift augmentation (Change 6) from the default pipeline. Instead, implement *channel permutation* augmentation with small random swaps of adjacent sensor pairs, only if sensors are placed in a consistent spatial order. For now, rely on channel dropout (already in Change 3's augmenter) to achieve partial robustness to placement differences. Revised plan flags Change 6 as conditional.

### Q3: With only 7 subjects, is adversarial training (Change 4) at risk of overfitting the subject discriminator to training subjects rather than learning truly subject-invariant features?

**Answer:** Yes, this is a real risk. With only 7 subjects (6 in training for each LOSO fold), the subject discriminator has a trivial classification task and the gradient reversal signal may be too specific to the training subjects' identity rather than capturing general inter-subject variability.

**Mitigation:** (a) Use a small subject discriminator (already proposed: only 64 hidden units). (b) Use the session ID as the "domain" label instead of subject ID -- with 4-8 sessions per subject and 6 subjects in training, this gives ~30-40 domains, making the adversarial task harder and more generalizable. (c) Ramp the adversarial weight slowly (already proposed). Revised plan uses session-level domain labels.

### Q4: The current CnnBundle stores `mean`/`std` arrays. If Change 2 sets these to zeros/ones (no-op), will the existing `realtime_gesture_cnn.py` still work correctly without modification?

**Answer:** Yes. `CnnBundle.standardize()` computes `(X - mean) / std`. With `mean=0, std=1`, this returns X unchanged. The realtime script calls `bundle.standardize(window)` after already applying MVC normalization and the clip/scale. So the data flow is:

`raw -> filter -> MVC normalize -> clip/scale -> bundle.standardize (no-op) -> predict`

This is correct and backward-compatible. Older per-subject bundles with real `mean`/`std` values will continue to work with the existing script since they don't use MVC normalization at the bundle level.

### Q5: Is the proposed `GestureCNNv2` architecture compatible with the existing bundle save/load format?

**Answer:** Partially. The `_resolve_architecture` function in `gesture_model_cnn.py` currently hardcodes `GestureCNN` as the only recognized architecture type. Loading a `GestureCNNv2` bundle would fail because `_resolve_architecture` would fall back to default `GestureCNN` parameters and the state dict keys would not match.

**Mitigation:** Extend `_resolve_architecture` and `load_cnn_bundle` to support a `"type": "GestureCNNv2"` architecture entry. Add `GestureCNNv2` to `gesture_model_cnn.py`. The bundle format itself (dict with `model_state`, `normalization`, etc.) does not change -- only the model class instantiation logic needs updating. Revised plan includes this.

---

## 8. Revised Plan -- Changes from Initial Draft

### Revision 1: Calibration data validation (from Q1)
Added a mandatory pre-training validation step to `load_dataset()`:

```python
# After loading all files, verify calibration coverage
files_without_calib = []
for fp in files:
    data = np.load(fp, allow_pickle=True)
    has_calib = ("calib_neutral_emg" in data.files and "calib_mvc_emg" in data.files)
    if not has_calib:
        files_without_calib.append(fp)

if files_without_calib:
    print(f"WARNING: {len(files_without_calib)} files lack calibration data:")
    for f in files_without_calib:
        print(f"  {f}")
    print("These files will use per-file z-score normalization as fallback.")
```

**Why:** Without this, the MVC normalization strategy (Change 2) could silently produce garbage for uncalibrated sessions, poisoning training data.

### Revision 2: Electrode-shift augmentation marked conditional (from Q2)
Change 6 is now marked as **conditional on sensor layout verification**. Default pipeline uses channel dropout (from Change 3) instead of circular shift. Added a note that if future data collection uses an HD-EMG grid, Change 6 becomes highly relevant.

**Why:** Applying circular shift to non-contiguous sensors would add harmful noise rather than useful augmentation.

### Revision 3: Session-level adversarial domains (from Q3)
Changed the adversarial discriminator (Change 4) to predict **session ID** instead of **subject ID**:

```python
# In training data preparation:
# session_ids: unique integer per (subject, session) pair
session_to_idx = {s: i for i, s in enumerate(sorted(np.unique(session_groups)))}
session_labels = np.array([session_to_idx[g] for g in session_groups])

# In GestureCNNAdversarial:
# num_subjects -> num_sessions (~35 instead of ~7)
```

**Why:** With only 6-7 subject IDs as domain labels, the adversarial signal is too coarse and risks overfitting to specific subject identities. Session-level domains (30-40 domains) force the model to learn more general invariance across time, placement, and individual differences simultaneously.

### Revision 4: Architecture registry for bundle loading (from Q5)
Added an architecture registry to `gesture_model_cnn.py`:

```python
ARCHITECTURE_REGISTRY = {
    "GestureCNN": GestureCNN,
    "GestureCNNv2": GestureCNNv2,
}

def _resolve_architecture(bundle, in_channels, num_classes):
    arch = bundle.get("architecture") or {}
    arch_type = arch.get("type", "GestureCNN")
    model_class = ARCHITECTURE_REGISTRY.get(arch_type, GestureCNN)
    # ... resolve parameters based on model_class
```

**Why:** Without this, deploying a `GestureCNNv2` model would break `load_cnn_bundle`, making the new architecture unusable in the realtime pipeline.

### Revision 5: Added fallback normalization for uncalibrated files
For files missing MVC calibration, compute per-file z-score as fallback:

```python
if not has_calibration:
    # Fallback: per-file z-score (less ideal but consistent)
    file_mean = emg.mean(axis=0)
    file_std = emg.std(axis=0)
    file_std = np.where(file_std < 1e-6, 1.0, file_std)
    emg = (emg - file_mean) / file_std
    # Then apply same clip/scale
    emg = np.clip(emg, -5.0, 5.0) / 5.0
```

**Why:** Mixing calibrated and uncalibrated data without any normalization would create a distribution mismatch that degrades training.

---

## 9. Summary of All Changes

| # | Change | Impact | Effort | Phase |
|---|--------|--------|--------|-------|
| 1 | LOSO evaluation | Critical (measurement) | 1 day | 1 |
| 2 | MVC-only normalization | High | 1-2 days | 2 |
| 3 | Channel-wise amplitude augmentation | High | 1 day | 3 |
| 4 | Session-adversarial training (DANN) | High | 2-3 days | 3 |
| 5 | GestureCNNv2 with residual blocks | Medium | 2 days | 4 |
| 6 | Electrode-shift augmentation | Conditional | 0.5 day | 4 |
| 7 | Subject-balanced batch sampling | Medium | 1 day | 3 |
| 8 | Prototypical network (few-shot) | Fallback | 3-5 days | 5 |

**Total estimated effort:** 12-16 days for full implementation through Phase 4. Phase 5 (prototypical network) is contingent on Phase 4 results.

**Expected outcome:** A single deployable model (`gesture_cnn_cross_subject.pt`) that achieves 60-75% LOSO accuracy on the 7-subject dataset, usable by new subjects after a 6-second MVC calibration with no retraining. The bundle format remains backward-compatible with the existing `load_cnn_bundle` / `CnnBundle` contract.
