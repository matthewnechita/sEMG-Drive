# CNN Cross-Subject Generalization Improvement Plan

## 1. Root Cause Analysis: Why Per-Subject Models Outperform Cross-Subject

### 1.1 The Core Problem: Subject-Specific EMG Signatures

EMG signals are fundamentally person-specific for three reasons:

1. **Amplitude scale differences.** Each person's muscle fiber density, subcutaneous fat thickness, skin impedance, and electrode placement produce vastly different absolute signal amplitudes. Subject A's "horn" gesture might produce peak rectified EMG of 0.8 mV on channel 3, while Subject B's identical gesture produces 0.2 mV. The current per-channel z-score normalization (computed from training data) cannot compensate because the training mean/std is a mixture of all subjects' distributions, which is not representative of any single subject.

2. **Channel activation pattern differences.** Even with identical sensor placement protocols, small positional differences (millimeters matter on forearm muscles) cause the same gesture to activate different channel combinations across subjects. Channel 2 might be the dominant discriminator for Subject A but irrelevant for Subject B. A model trained on Subject A's channel patterns will misclassify Subject B's data.

3. **Temporal dynamics differences.** Muscle fiber recruitment speed, fatigue patterns, and co-contraction habits differ across individuals. The 1D convolutional kernels learn temporal features (onset slopes, oscillation frequencies) that are partially subject-specific.

### 1.2 Current Pipeline Failures (Specific to This Codebase)

| Issue | Location | Impact |
|-------|----------|--------|
| **Global z-score uses mixed-subject statistics** | `train_cnn.py` line 195-197: `mean = X_train.mean(axis=(0,2))` computes mean across ALL subjects in the training set | A per-channel mean computed from 6 subjects is wrong for all 6 of them individually. The model receives inputs normalized to a distribution it never truly sees at inference. |
| **GroupShuffleSplit splits by file, not by subject** | `train_cnn.py` line 318-321: `GroupShuffleSplit` uses `groups` (file paths) | Windows from the SAME subject appear in both train and test. Test accuracy is inflated; it does not measure cross-subject generalization at all. |
| **MVC calibration is per-session but z-score is per-training-set** | `load_windows_from_file()` applies `(emg - neutral_mean) / mvc_scale` per-file, then `train_eval_split()` applies a SECOND normalization using global train stats | Double normalization. The calibration partially removes subject-specific scale, but the subsequent z-score re-introduces a population-level bias. |
| **No data augmentation** | `train_cnn.py` has no augmentation | With only 42 session files across 7 subjects, the model memorizes subject-specific patterns instead of learning gesture-invariant features. |
| **BatchNorm locks subject-specific statistics** | `gesture_model_cnn.py` line 18: `nn.BatchNorm1d(out_ch)` | During inference, BatchNorm uses running mean/var computed during training (dominated by training subjects). For a new subject whose feature distribution differs, this shifts activations incorrectly. |

### 1.3 Quantitative Framing

With 7 subjects and the current GroupShuffleSplit (by file), test windows from Subject X's session 3 are evaluated using a model that also trained on Subject X's sessions 1, 2, 4-8. This is **within-subject evaluation disguised as held-out evaluation**. True cross-subject accuracy (leave-one-subject-out) is likely 30-50% lower than reported.

---

## 2. Proposed Changes, Ranked by Expected Impact

### Change 1: Leave-One-Subject-Out (LOSO) Evaluation [CRITICAL -- DO FIRST]

**Why:** Without proper evaluation, no improvement can be measured. Every other change depends on this.

**What:** Replace `GroupShuffleSplit(by file)` with leave-one-subject-out cross-validation: train on 6 subjects, test on the 7th, repeat 7 times.

**Implementation in `train_cnn.py`:**

```python
# New config flag
LOSO_EVAL = True  # Leave-One-Subject-Out evaluation

def loso_cross_validation(X, y_idx, subjects, channels, num_classes, device,
                          labels, label_to_index, index_to_label, channel_count):
    """Train on N-1 subjects, test on the held-out subject. Repeat for each subject."""
    unique_subjects = sorted(np.unique(subjects))
    results = {}

    for held_out in unique_subjects:
        print(f"\n{'='*55}")
        print(f"LOSO: Holding out {held_out}")
        train_mask = subjects != held_out
        test_mask = subjects == held_out

        X_train, y_train = X[train_mask], y_idx[train_mask]
        X_test, y_test = X[test_mask], y_idx[test_mask]

        print(f"  Train: {train_mask.sum()} windows from {len(np.unique(subjects[train_mask]))} subjects")
        print(f"  Test:  {test_mask.sum()} windows from {held_out}")

        model, mean, std, best_eval_acc = train_eval_split(
            X_train, y_train,
            X_test, y_test,  # NOTE: using held-out subject as eval during training
            channels, num_classes, EPOCHS, device,
        )

        # Final predictions
        model.eval()
        X_test_norm = standardize_per_channel(X_test, mean, std).astype(np.float32)
        test_ds = TensorDataset(torch.from_numpy(X_test_norm), torch.from_numpy(y_test))
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        all_preds = []
        with torch.no_grad():
            for xb, _ in test_loader:
                logits = model(xb.to(device))
                all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        y_pred = np.concatenate(all_preds)
        acc = float(accuracy_score(y_test, y_pred))
        results[held_out] = acc
        print(f"  LOSO accuracy for {held_out}: {acc:.3f}")
        print(classification_report(y_test, y_pred,
              target_names=[index_to_label[i] for i in range(len(labels))]))

    print(f"\n{'='*55}")
    print("LOSO Summary:")
    accs = list(results.values())
    for subj, acc in sorted(results.items()):
        print(f"  {subj}: {acc:.3f}")
    print(f"  Mean: {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
    return results
```

**Measurement:** Run LOSO before any other changes to establish the TRUE cross-subject baseline. This number is the one to beat.

---

### Change 2: Per-Window Instance Normalization (Replace Global Z-Score) [HIGH IMPACT]

**Why:** The global z-score normalization (`mean`/`std` from training set) is the single largest source of cross-subject mismatch. At inference on a new subject, their signal amplitudes are normalized using statistics from other people's muscles.

**What:** Replace the external z-score with **per-window, per-channel normalization** done inside the model itself, so the model always sees zero-mean, unit-variance inputs regardless of who generated them.

**Option A -- Per-window z-score at input (simplest):**

```python
def per_window_normalize(x):
    """x: (batch, channels, time) -> normalized per channel per window."""
    mean = x.mean(dim=2, keepdim=True)    # (B, C, 1)
    std = x.std(dim=2, keepdim=True) + 1e-6
    return (x - mean) / std
```

This goes into the model's `forward()` or as a preprocessing step. The bundle no longer needs `mean`/`std` arrays (or they become dummy values for backward compatibility).

**Option B -- Instance Normalization layer (learnable):**

Replace `nn.BatchNorm1d` with `nn.InstanceNorm1d(out_ch, affine=True)` throughout `GestureCNN`. InstanceNorm normalizes each sample independently (per channel, per instance), so it never accumulates population-level running statistics. This is the approach used in style transfer to strip domain-specific (here: subject-specific) information.

**Recommended: Use BOTH.** Apply per-window z-score at the input (Option A) AND replace BatchNorm with InstanceNorm in the convolutional blocks (Option B).

**Architecture change in `gesture_model_cnn.py`:**

```python
class GestureCNN(nn.Module):
    def __init__(self, channels, num_classes, dropout=0.2, kernel_size=7,
                 use_instance_norm=False):
        super().__init__()
        blocks = []
        padding = kernel_size // 2
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            blocks.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, bias=False))
            if use_instance_norm:
                blocks.append(nn.InstanceNorm1d(out_ch, affine=True))
            else:
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
        self.use_instance_norm = use_instance_norm

    def forward(self, x):
        # Per-window input normalization (subject-agnostic)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True).clamp(min=1e-6)
        x = (x - mean) / std
        x = self.features(x)
        return self.head(x)
```

**Bundle compatibility:** Store `use_instance_norm: True` in the `architecture` dict. The `mean`/`std` arrays in the bundle become no-ops (store zeros/ones). `_resolve_architecture` reads the flag and passes it to the constructor. `CnnBundle.standardize()` still works (dividing by 1.0 is a no-op) so `realtime_gesture_cnn.py` requires zero changes.

**Expected impact:** This single change likely accounts for 15-25% absolute accuracy improvement in LOSO because it eliminates the population-statistics mismatch entirely.

---

### Change 3: Data Augmentation [HIGH IMPACT]

**Why:** 7 subjects is a tiny dataset for learning subject-invariant features. Augmentation synthetically expands the effective subject pool by simulating the kinds of variation seen across subjects.

**What:** Apply the following augmentations randomly during training (not at test time):

| Augmentation | Simulates | Implementation |
|-------------|-----------|----------------|
| **Amplitude scaling** (per-channel, random factor 0.5-2.0) | Electrode placement differences, skin impedance variation | `x * scale` where `scale ~ Uniform(0.5, 2.0)` per channel |
| **Additive Gaussian noise** (SNR 15-30 dB) | Sensor noise, electromagnetic interference | `x + noise` where `noise ~ N(0, sigma)`, sigma chosen per window |
| **Channel dropout** (zero out 1 random channel with p=0.15) | Sensor detachment, dead channel | Set one channel to zero with probability 0.15 |
| **Time shift** (shift window by -10 to +10 samples) | Timing jitter in gesture onset | Roll the time axis; pad with edge values |
| **Magnitude warping** (smooth random scaling over time) | Non-stationary muscle contraction | Multiply by a smooth curve generated from cubic spline interpolation of random knots |

**Implementation as a PyTorch transform applied in the DataLoader:**

```python
class EMGAugment:
    def __init__(self, p=0.8):
        self.p = p

    def __call__(self, x):
        """x: (channels, time) tensor."""
        if torch.rand(1).item() > self.p:
            return x
        C, T = x.shape

        # 1. Amplitude scaling per channel
        if torch.rand(1).item() < 0.5:
            scale = torch.empty(C, 1).uniform_(0.5, 2.0)
            x = x * scale

        # 2. Additive noise
        if torch.rand(1).item() < 0.5:
            snr_db = torch.empty(1).uniform_(15, 30).item()
            signal_power = x.pow(2).mean()
            noise_power = signal_power / (10 ** (snr_db / 10))
            x = x + torch.randn_like(x) * noise_power.sqrt()

        # 3. Channel dropout
        if torch.rand(1).item() < 0.15:
            ch = torch.randint(0, C, (1,)).item()
            x[ch, :] = 0.0

        # 4. Time shift
        if torch.rand(1).item() < 0.3:
            shift = torch.randint(-10, 11, (1,)).item()
            x = torch.roll(x, shifts=shift, dims=1)

        return x
```

**Integration:** Create an `AugmentedEMGDataset` that wraps the TensorDataset and applies `EMGAugment` to each sample during `__getitem__`. Use this only for the training DataLoader.

```python
class AugmentedEMGDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, augment_fn=None):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.augment_fn = augment_fn

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment_fn is not None:
            x = self.augment_fn(x)
        return x, self.y[idx]
```

**Expected impact:** 5-15% LOSO improvement. Augmentation is most impactful when combined with the normalization fix (Change 2).

---

### Change 4: Subject-Adversarial Training (Domain Adversarial Neural Network) [MEDIUM-HIGH IMPACT]

**Why:** Even after per-window normalization, the learned feature representations may still encode subject identity (through subtle channel correlation patterns). Subject-adversarial training explicitly forces the feature extractor to produce representations that are INDISTINGUISHABLE across subjects, using a gradient reversal layer.

**What:** Add a subject classifier branch that tries to predict which subject produced a given feature vector. During backpropagation, REVERSE the gradient from this branch so the feature extractor learns to FOOL the subject classifier (i.e., produce subject-invariant features).

**Architecture:**

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


class GestureCNNWithDANN(nn.Module):
    def __init__(self, channels, num_classes, num_subjects, dropout=0.2,
                 kernel_size=7, use_instance_norm=True):
        super().__init__()
        # Feature extractor (shared)
        blocks = []
        padding = kernel_size // 2
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            blocks.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, bias=False))
            if use_instance_norm:
                blocks.append(nn.InstanceNorm1d(out_ch, affine=True))
            else:
                blocks.append(nn.BatchNorm1d(out_ch))
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(nn.MaxPool1d(kernel_size=2, stride=2))
            if dropout > 0:
                blocks.append(nn.Dropout(dropout))
        self.features = nn.Sequential(*blocks)

        # Gesture classifier head (main task)
        self.gesture_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], num_classes),
        )

        # Subject classifier head (adversarial -- only used during training)
        self.subject_head = nn.Sequential(
            GradientReversal(lambda_=1.0),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_subjects),
        )

    def forward(self, x, return_subject_logits=False):
        # Per-window normalization
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True).clamp(min=1e-6)
        x = (x - mean) / std

        feats = self.features(x)
        gesture_logits = self.gesture_head(feats)

        if return_subject_logits:
            subject_logits = self.subject_head(feats)
            return gesture_logits, subject_logits
        return gesture_logits
```

**Training loop modification:**

```python
# Combined loss
gesture_loss = gesture_criterion(gesture_logits, gesture_labels)
subject_loss = subject_criterion(subject_logits, subject_labels)

# Lambda schedule: ramp up adversarial weight from 0 to 1 over training
p = epoch / total_epochs
lambda_ = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0  # sigmoid schedule
for module in model.modules():
    if isinstance(module, GradientReversal):
        module.lambda_ = lambda_

total_loss = gesture_loss + 0.3 * subject_loss
total_loss.backward()
```

**At inference:** Only the gesture head is used. The subject branch is discarded. The saved bundle contains only the feature extractor + gesture head weights (same format as current `GestureCNN`).

**Bundle compatibility:** At save time, extract only the `features` and `gesture_head` state dicts and map them back to a standard `GestureCNN` structure. Alternatively, save the `GestureCNNWithDANN` but load with `strict=False` to ignore subject head weights.

**Expected impact:** 5-10% LOSO improvement on top of Changes 2-3. Most effective when the feature space still contains subject-discriminative information after normalization.

---

### Change 5: MVC Calibration Consistency Fix [MEDIUM IMPACT]

**Why:** The current pipeline applies MVC calibration in `load_windows_from_file()` (line 118-119: `emg = (emg - neutral_mean) / mvc_scale`) and then ALSO applies a global z-score in `train_eval_split()`. This double normalization is counterproductive: the calibration partially removes subject-specific amplitude, then the z-score re-introduces a population-level shift.

**What:** Two options:

**Option A (recommended with Change 2):** Remove the global z-score entirely. The per-window normalization in the model (Change 2) replaces it. Keep MVC calibration as a first-pass coarse normalization, then let per-window norm handle the rest.

**Option B (if not using Change 2):** Remove MVC calibration and rely solely on a properly computed normalization. But this is strictly worse than Option A.

**Implementation:** In `train_eval_split()`, skip the `standardize_per_channel` call when `use_instance_norm=True`. The bundle's `mean`/`std` become `np.zeros(C)` / `np.ones(C)`.

```python
if USE_INSTANCE_NORM:
    mean = np.zeros(X_train.shape[1], dtype=np.float32)
    std = np.ones(X_train.shape[1], dtype=np.float32)
    # No external normalization; the model handles it internally
else:
    mean = X_train.mean(axis=(0, 2))
    std = X_train.std(axis=(0, 2))
    std = np.where(std < 1e-6, 1.0, std)
    X_train = standardize_per_channel(X_train, mean, std).astype(np.float32)
    X_eval = standardize_per_channel(X_eval, mean, std).astype(np.float32)
```

---

### Change 6: Deeper / Wider Architecture with Residual Connections [MEDIUM IMPACT]

**Why:** The current model is 3 conv layers ([input, 32, 64, 128]) with a total of ~50K parameters. This may be too small to learn subject-invariant features. Cross-subject generalization requires the model to learn a richer representation that separates gesture information from subject information.

**What:** Add residual connections and increase to 4 blocks: [input, 64, 128, 128, 128].

```python
class ResBlock1d(nn.Module):
    def __init__(self, channels, kernel_size, dropout, use_instance_norm):
        super().__init__()
        padding = kernel_size // 2
        norm = nn.InstanceNorm1d if use_instance_norm else nn.BatchNorm1d
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False),
            norm(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False),
            norm(channels, affine=True),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))
```

Keep the existing progressive-width structure for the first 3 blocks, then add 1-2 ResBlocks at the 128-channel level before the classification head. This adds representational capacity without dramatically increasing inference time.

**Real-time constraint check:** The current model processes a (C, 200) window in <1ms on CPU. Adding a ResBlock roughly doubles the conv operations at the 128-channel level, bringing inference to ~1.5-2ms. This is well within the 50ms budget (WINDOW_STEP=100 at ~2kHz = 50ms between predictions).

---

### Change 7: Minimal Calibration Protocol for New Subjects (Transfer + Fast Adaptation) [MEDIUM IMPACT, OPTIONAL]

**Why:** Even with all the above changes, a 10-second calibration at inference time can dramatically boost accuracy for a new subject. This is NOT retraining -- it is a lightweight adaptation.

**What:** At inference startup, collect ~10 seconds of neutral rest + 10 seconds of each gesture (total ~70 seconds). Use this to:

1. Compute the new subject's per-channel neutral baseline and MVC scale (already done in `realtime_gesture_cnn.py` with CALIBRATE=True).
2. **Optionally: fine-tune ONLY the classification head** (last linear layer) on the calibration data. This takes <5 seconds on CPU and adapts the decision boundaries without changing the learned feature extractor.

```python
def quick_finetune(bundle, calib_windows, calib_labels, device='cpu', lr=1e-3, epochs=20):
    """Fine-tune only the classification head on calibration data.
    calib_windows: (N, C, T) numpy array
    calib_labels: (N,) numpy array of integer labels
    """
    model = bundle.model
    model.to(device)

    # Freeze feature extractor
    for param in model.features.parameters():
        param.requires_grad = False

    # Only train the head
    optimizer = torch.optim.Adam(model.head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X = torch.from_numpy(calib_windows.astype(np.float32)).to(device)
    y = torch.from_numpy(calib_labels).to(device)

    model.train()
    for _ in range(epochs):
        logits = model(X)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Unfreeze for future use
    for param in model.features.parameters():
        param.requires_grad = True

    model.eval()
    return model
```

**Calibration protocol (70 seconds total):**
1. "Rest your arm naturally" -- 10 seconds (neutral)
2. "Contract as hard as you can" -- 5 seconds (MVC for normalization)
3. For each of the 5 non-neutral gestures: "Perform [gesture] now" -- 10 seconds each
4. Fine-tune head on these ~700 windows (10s * 2kHz / step 100 = ~200 windows per gesture = ~1200 total)

This is presented as an optional enhancement. The goal is that the base model (Changes 1-6) works acceptably WITHOUT this step, but calibration pushes accuracy from "good" to "great."

---

## 3. Implementation Roadmap

### Phase 1: Measurement (Changes 1 + baseline LOSO)
1. Add `LOSO_EVAL = True` flag to `train_cnn.py`
2. Implement `loso_cross_validation()` function
3. Run LOSO with the CURRENT architecture and normalization to get a true baseline
4. Record per-subject accuracy and overall mean

### Phase 2: Normalization Fix (Changes 2 + 5)
1. Add `use_instance_norm` parameter to `GestureCNN`
2. Add per-window z-score in `forward()`
3. Replace `BatchNorm1d` with `InstanceNorm1d` in conv blocks
4. Update `_resolve_architecture()` to read `use_instance_norm` from bundle
5. Set bundle `mean`/`std` to zeros/ones (backward compatible)
6. Re-run LOSO. Compare to Phase 1 baseline.

### Phase 3: Data Augmentation (Change 3)
1. Implement `EMGAugment` class
2. Implement `AugmentedEMGDataset` wrapper
3. Use augmented dataset only for training loader
4. Re-run LOSO. Compare to Phase 2.

### Phase 4: Subject-Adversarial Training (Change 4)
1. Implement `GradientReversal` layer
2. Implement `GestureCNNWithDANN` (or add adversarial branch to existing model)
3. Modify training loop to compute combined loss with lambda schedule
4. At save time, strip subject head from bundle
5. Re-run LOSO. Compare to Phase 3.

### Phase 5: Architecture Refinement (Change 6)
1. Add `ResBlock1d` to `gesture_model_cnn.py`
2. Increase channel widths: [input, 64, 128, 128]
3. Add 1 ResBlock after the last conv stage
4. Re-run LOSO. Compare to Phase 4.

### Phase 6: Optional Fast Adaptation (Change 7)
1. Add `quick_finetune()` utility
2. Add calibration gesture collection to `realtime_gesture_cnn.py`
3. Measure LOSO where each held-out subject gets 1 session for calibration fine-tuning, rest for testing

---

## 4. How to Modify `train_cnn.py` for LOSO

The core change is in `main()`. When `LOSO_EVAL` is True:

```python
def main():
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    X, y, groups, subjects, channel_count = load_dataset()
    unique_subjects = sorted(np.unique(subjects))

    labels = sorted({str(lbl) for lbl in np.unique(y)})
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    y_idx = np.array([label_to_index[str(lbl)] for lbl in y], dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels = [int(channel_count), 64, 128, 128]  # wider architecture
    num_classes = len(labels)

    if LOSO_EVAL:
        loso_cross_validation(X, y_idx, subjects, channels, num_classes, device,
                              labels, label_to_index, index_to_label, channel_count)
        return

    # ... rest of existing main() for final model training ...

    # For the FINAL deployable model: train on ALL subjects
    print("\nTraining final cross-subject model on ALL data...")
    # Use 10% of windows (stratified, random) as early-stopping eval
    indices = np.arange(X.shape[0])
    train_idx, eval_idx = train_test_split(
        indices, test_size=0.1, random_state=RANDOM_STATE, stratify=y_idx
    )
    model, mean, std, _ = train_eval_split(
        X[train_idx], y_idx[train_idx],
        X[eval_idx], y_idx[eval_idx],
        channels, num_classes, EPOCHS, device,
    )
    # Save bundle...
```

The key insight: LOSO is for EVALUATION. The final deployable model trains on ALL 7 subjects (using a small held-out fraction for early stopping only). LOSO tells you how well this model will perform on subject #8.

---

## 5. Verification Questions and Answers

### Q1: With only 7 subjects, does LOSO produce reliable estimates, or is the variance too high?

**A1:** With 7 folds, variance will be high. Each fold tests on 1 subject, so a single weak subject can swing the mean significantly. This is a real limitation but LOSO is still the correct evaluation -- it directly measures what we care about (new-subject performance). Mitigation: report both mean and per-subject accuracy. If one subject is a consistent outlier, investigate whether their data quality differs (electrode placement, fewer sessions, etc.). With this dataset, subject05 (4 sessions) and subject01 (4 sessions, missing session01) have less data and may underperform.

### Q2: Does per-window normalization destroy useful amplitude information?

**A2:** Yes, partially. Per-window z-score removes absolute amplitude, which IS discriminative for gestures (e.g., "horn" might consistently produce higher amplitude than "neutral"). However, what's preserved is the RELATIVE amplitude pattern across channels and the temporal shape of the signal within the window. These relative patterns are more consistent across subjects than absolute amplitudes. Empirically, per-window normalization consistently helps cross-subject performance in EMG literature even though it sacrifices some within-subject discriminability. The tradeoff is worth it for generalization.

**Revision needed:** To partially recover amplitude information, consider normalizing per-channel but keeping the RATIO between channels. Specifically, normalize by the L2 norm of the entire window rather than per-channel z-score:

```python
# Alternative: window-level L2 normalization (preserves inter-channel ratios)
norm = x.pow(2).sum(dim=(1, 2), keepdim=True).sqrt().clamp(min=1e-6)
x = x / norm
```

This preserves which channels are active relative to each other while removing overall amplitude. We should test both approaches in LOSO.

### Q3: Can InstanceNorm + per-window z-score cause numerical instability for "silent" windows (all near-zero)?

**A3:** Yes. A window where the subject is at rest may have very small EMG values across all channels. Dividing by a near-zero std amplifies noise. The `clamp(min=1e-6)` prevents division by zero but may produce very large normalized values from pure noise. **Mitigation:** Add a "silence gate" -- if the raw window's RMS is below a threshold, classify it as "neutral" without running the model, or clip the normalized values to a reasonable range (e.g., [-5, 5]).

### Q4: How does the gradient reversal lambda schedule interact with training dynamics? Could adversarial training destabilize learning?

**A4:** Yes, adversarial training can be unstable. If lambda is too high too early, the feature extractor may collapse to a trivial representation that fools the subject classifier but loses gesture information too. The sigmoid schedule (`2/(1+exp(-10*p))-1`) starts near 0 and ramps to 1, giving the model time to learn useful features before the adversarial pressure kicks in. However, with only 7 subjects (and the adversarial branch only sees 6 during LOSO), the subject classifier may not have enough signal to provide useful gradients. **Mitigation:** Start with lambda_max=0.3 instead of 1.0. If adversarial training hurts, it may be unnecessary (Changes 2-3 might already remove enough subject information).

### Q5: Is the proposed architecture change (wider channels, ResBlock) justified given the small dataset, or will it overfit?

**A5:** With 7 subjects * ~4-8 sessions * ~6 gestures * ~200 windows per gesture = roughly 40K-80K training windows (estimating), increasing from ~50K to ~150K parameters is reasonable. The ResBlock adds capacity without doubling parameters because it reuses the same channel width. Dropout (0.4) and data augmentation (Change 3) provide regularization. However, if LOSO accuracy does not improve from Phase 4 to Phase 5, revert to the smaller architecture -- the added capacity is only useful if the model is currently underfitting cross-subject patterns.

---

## 6. Revisions Based on Verification Answers

### Revision A: Dual normalization strategy

Added per-window L2 normalization as an alternative to per-window z-score (from Q2). The implementation should test both and pick the better one during LOSO:

```python
# In GestureCNN.forward():
NORM_MODE = 'zscore'  # or 'l2' -- store in architecture dict

if self.norm_mode == 'zscore':
    mean = x.mean(dim=2, keepdim=True)
    std = x.std(dim=2, keepdim=True).clamp(min=1e-6)
    x = (x - mean) / std
elif self.norm_mode == 'l2':
    norm = x.pow(2).sum(dim=(1, 2), keepdim=True).sqrt().clamp(min=1e-6)
    x = x / norm
```

Store `norm_mode` in the architecture dict so the bundle knows which was used.

### Revision B: Silence gate for neutral detection

Added from Q3. Before running the model, check the window's energy:

```python
# In realtime_gesture_cnn.py and in evaluation code:
SILENCE_RMS_THRESHOLD = 0.01  # tune on calibration data

def is_silent(window):
    """window: (1, C, T) numpy array, pre-MVC-calibration."""
    rms = np.sqrt(np.mean(window ** 2))
    return rms < SILENCE_RMS_THRESHOLD
```

If silent, return "neutral" with confidence 1.0 without running inference. This avoids numerical instability in normalization AND reduces computation.

### Revision C: Conservative adversarial training

From Q4: cap lambda_max at 0.3 and make it a tunable hyperparameter:

```python
DANN_LAMBDA_MAX = 0.3  # in config section of train_cnn.py
```

If LOSO with DANN (Phase 4) does not improve over Phase 3, skip it entirely and proceed to Phase 5.

### Revision D: Architecture change is conditional

From Q5: only apply Change 6 (wider architecture + ResBlock) if Phase 3 results suggest the model is underfitting (training accuracy >> eval accuracy by a small margin, and eval accuracy is plateauing). If the model is already overfitting (large train-eval gap), the smaller architecture is better and more regularization is needed instead.

---

## 7. Summary of Revisions

| Revision | What Changed | Why |
|----------|-------------|-----|
| A | Added L2 normalization as alternative to per-window z-score | Per-channel z-score destroys inter-channel amplitude ratios that are gesture-discriminative. L2 norm preserves these ratios while still removing absolute scale. Both should be tested. |
| B | Added silence gate before normalization | Near-silent windows (rest/neutral) produce numerical instability when normalized. A simple RMS threshold check avoids this and speeds up inference. |
| C | Capped adversarial lambda at 0.3 | With only 7 subjects, the subject classifier has weak signal. Aggressive gradient reversal could collapse features. Conservative lambda prevents this. |
| D | Made architecture widening conditional on underfitting evidence | With small data, a bigger model may overfit. Only add capacity if LOSO shows underfitting. |

---

## 8. Expected Outcome

| Phase | Change | Expected LOSO Accuracy (cumulative) |
|-------|--------|--------------------------------------|
| 0 | Current system (true LOSO, not file-split) | ~25-40% (estimated) |
| 1 | LOSO evaluation only (measurement) | Same as Phase 0 |
| 2 | Per-window norm + InstanceNorm | ~45-60% |
| 3 | + Data augmentation | ~55-70% |
| 4 | + Subject-adversarial training | ~60-75% |
| 5 | + Architecture refinement (if needed) | ~65-78% |
| 6 | + 70-second calibration fine-tuning | ~75-88% |

These estimates assume the current per-subject accuracy of ~78% reflects a ceiling for this sensor setup and gesture set. Cross-subject with calibration should approach per-subject levels; cross-subject without calibration will likely plateau 10-15% below per-subject.

---

## 9. Files to Modify

| File | Changes |
|------|---------|
| `gesture_model_cnn.py` | Add `use_instance_norm` and `norm_mode` params to `GestureCNN`; add per-window normalization in `forward()`; add `ResBlock1d` class; update `_resolve_architecture()` to read new fields; add `GradientReversal` and optionally `GestureCNNWithDANN` |
| `train_cnn.py` | Add `LOSO_EVAL` flag and `loso_cross_validation()`; add `EMGAugment` and `AugmentedEMGDataset`; add DANN training loop with subject labels; skip external z-score when using instance norm; add final-model training on all subjects |
| `realtime_gesture_cnn.py` | Add silence gate; optionally add `quick_finetune()` for calibration adaptation; no other changes needed (bundle backward compatibility maintained) |
| `gesture_model_cnn.py` (bundle) | Architecture dict gains: `use_instance_norm`, `norm_mode`, optionally `num_res_blocks`; `normalization.mean`/`std` become dummy values when instance norm is used |
