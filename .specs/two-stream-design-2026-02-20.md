# Two-Stream EMG Model Architecture Design
**Date:** 2026-02-20
**Status:** Proposed — pending review
**Branch:** model_refactor

---

## 1. Problem Statement

The current codebase has one training script (`train_cnn.py`) controlled by boolean flags that select between fundamentally different architectures. The flags `USE_INSTANCE_NORM` and `PER_SUBJECT_MODELS` are paired opposites that must be set consistently — getting them wrong silently trains an overparameterized model on insufficient data. There is no separation between the per-subject and cross-subject workflows, their different data pipelines, or their different inference requirements.

**Goal:** Establish two independent, purpose-built model streams with clear ownership, clean repo organization, and architecture/hyperparameters optimized for each stream's actual use case.

---

## 2. DQ Analysis Results

Data quality analysis was run across all 42 sessions from 7 subjects (152,352 total windows).

### What is fine
- **Label distribution:** Perfectly balanced active gestures (14.5% each across 5 classes), neutral at 27.3%. No class imbalance issues.
- **Calibration presence:** All 42 sessions have calibration data (`calib_neutral_emg`, `calib_mvc_emg`).
- **Cross-subject consistency:** All subjects use 17 channels and the same 5 gesture labels (`horn`, `left_turn`, `right_turn`, `signal_left`, `signal_right`).
- **No missing/corrupt files:** All sessions load cleanly with valid EMG and label arrays.

### What is a false positive
- The DQ script flagged 16–17 "dead channels" per session (RMS < 1 µV threshold). This is wrong. The threshold of 1.0 is too high — filtered EMG data in this dataset has typical per-channel RMS of 0.2–0.8 (in stored units). No channels are dead.

### Real issues: MVC calibration quality

MVC calibration quality directly predicts model accuracy. When a subject does not contract maximally during the MVC calibration, the normalization scale is corrupted.

| Subject | Typical MVC/Neutral ratio | Test Accuracy | Verdict |
|---------|--------------------------|---------------|---------|
| subject02 | 3.2–8.9× | **85.8%** | Good calibration |
| subject03 | 6.2–22.2× | 50.6% | Good calib, other issues |
| Matthew | 1.6–29.8× (variable) | 76.2% | Session01 weak |
| subject06 | 1.4–10.4× | 69.0% | Many weak sessions |
| subject04 | 2.0–4.9× | 51.0% | Weak across board |
| subject01 | 2.0–12.2× | 36.0% | 15/17 channels < 2× in most sessions |
| subject05 | **0.9–1.8×** | **43.3%** | ALL sessions: MVC barely exceeds neutral |

**Finding:** subject05 has fully failed MVC calibration — the "maximum voluntary contraction" barely exceeds the neutral baseline across all 4 sessions. This means MVC normalization is producing near-noise signals for this subject. Data is technically usable but MVC-scaled normalization is unreliable.

### DQ Verdict
- **PASS overall** — no missing data, no corrupt files, no channel/label inconsistencies
- **FLAG subject05** — MVC calibration failed across all sessions; consider recollection or excluding MVC normalization for this subject
- **WARN subject01, subject04, subject06** — MVC calibration is weak in many sessions; recollect calibration if possible
- **ACTION** — improve the MVC collection protocol: instruct subjects to contract "as hard as possible" and verify the MVC signal exceeds the neutral RMS by at least 5× before accepting the session

---

## 3. Proposed Repo Structure

```
capstone-emg/
├── gesture_model_cnn.py        # Shared: model class definitions (GestureCNN, GestureCNNv2)
├── train_per_subject.py        # Stream 1: per-subject training (replaces train_cnn.py for this use)
├── train_cross_subject.py      # Stream 2: cross-subject training
├── realtime_gesture_cnn.py     # Inference: model-agnostic, loads any CnnBundle
├── _dq_analysis.py             # Tooling: data quality analysis
├── _inspect_models.py          # Tooling: model metadata inspection
├── data/
│   ├── Matthew/filtered/
│   ├── subject01/filtered/
│   └── ...
└── models/
    ├── per_subject/            # Output of train_per_subject.py
    │   ├── Matthew_cnn.pt
    │   ├── subject01_cnn.pt
    │   └── ...
    └── cross_subject/          # Output of train_cross_subject.py
        └── gesture_cnn_v2.pt
```

The old `train_cnn.py` is replaced by the two dedicated scripts. It can be kept for reference during transition.

---

## 4. Stream 1: Per-Subject Model (`train_per_subject.py`)

### Purpose
Best possible accuracy for a subject already in the system. Requires prior data collection from that specific person. Not expected to generalize to new subjects.

### Architecture: GestureCNN
- **Parameters:** ~120,297
- **Normalization:** z-score per channel (mean/std computed from training data)
- **Rationale:** 120K params / ~11K–23K training windows = 5–11 params per sample — appropriate density. InstanceNorm is NOT used because amplitude is discriminative *within* a subject (harder flex = higher amplitude; neutral = low amplitude). Stripping amplitude via InstanceNorm hurts within-subject classification.

### Config (`train_per_subject.py` constants)
```python
EPOCHS          = 100           # more epochs; GestureCNN is smaller, trains faster
LR              = 3e-4          # slightly higher LR for smaller model
DROPOUT         = 0.2           # less regularization needed (smaller model)
KERNEL_SIZE     = 7             # original GestureCNN kernel
USE_INSTANCE_NORM    = False    # MUST be False for per-subject
PER_SUBJECT_MODELS   = True     # MUST be True
USE_AUGMENTATION     = True
AUG_AMPLITUDE_RANGE  = (0.7, 1.4)   # within-subject amplitude variance is narrow
AUG_PROB             = 0.4
USE_BALANCED_SAMPLING = False   # not needed; single-subject data is already uniform
LOSO_EVAL            = False    # meaningless for per-subject
MODEL_OUT_DIR        = Path("models/per_subject")
```

### Key differences from current `train_cnn.py`
1. `AUG_AMPLITUDE_RANGE` narrowed to (0.7, 1.4) — current (0.5, 2.0) is calibrated for cross-subject variance, which is 5–10× wider than within-subject variance
2. `DROPOUT = 0.2` instead of 0.3 — smaller model needs less regularization
3. `EPOCHS = 100` instead of 70 — smaller model trains faster per epoch, more epochs benefit
4. No balanced sampling — single subject has no subject-imbalance problem
5. Output directory `models/per_subject/` instead of `models/`

### Expected outcome
Should reproduce or exceed the old GestureCNN global model's 81.4% per-subject, with proper per-subject specialization. Subjects with good MVC calibration (subject02) should reach 85%+. Subjects with poor calibration (subject05) will remain limited by data quality.

---

## 5. Stream 2: Cross-Subject Model (`train_cross_subject.py`)

### Purpose
A single model that works on any new subject without retraining. The model must handle inter-subject amplitude variation, electrode placement differences, and muscle activation pattern differences.

### Architecture: GestureCNNv2
- **Parameters:** ~503,821
- **Normalization:** InstanceNorm at input (per-window, strips amplitude scale)
- **Rationale:** Pools all 152K windows from 7 subjects → 122K training windows → ~4 params per training sample (good ratio for this model size). InstanceNorm removes inter-subject amplitude differences so the model learns shape-based features rather than subject-specific amplitude levels. Energy bypass scalar preserves neutral detection signal.

### Config (`train_cross_subject.py` constants)
```python
EPOCHS          = 80            # GestureCNNv2 is larger; 80 epochs balances training time
LR              = 1e-4
DROPOUT         = 0.3
USE_INSTANCE_NORM    = True     # MUST be True for cross-subject
PER_SUBJECT_MODELS   = False    # MUST be False
USE_AUGMENTATION     = True
AUG_AMPLITUDE_RANGE  = (0.5, 2.0)   # wide range for cross-subject variance
AUG_PROB             = 0.5
USE_BALANCED_SAMPLING = True    # balance subjects during training
LOSO_EVAL            = True     # mandatory: measures true cross-subject accuracy
MODEL_OUT            = Path("models/cross_subject/gesture_cnn_v2.pt")
```

### Key differences from current `train_cnn.py`
1. Mandatory `LOSO_EVAL = True` — the cross-subject stream must be validated against held-out subjects before deployment
2. Wide amplitude augmentation (0.5, 2.0) appropriate for cross-subject variance
3. Output directory `models/cross_subject/`
4. Config comments removed from constants (live in docstring instead)
5. Config sanity check moved to `train_per_subject.py` (where the mismatch is more likely)

### Expected outcome
With 152K windows from 7 subjects and GestureCNNv2, the cross-subject model should outperform the old 82.7% global GestureCNN because:
- More training data than any per-subject model
- InstanceNorm handles inter-subject amplitude variation that the old model had no mechanism for
- Better augmentation pipeline
- Subject-balanced sampling ensures no subject dominates training

LOSO accuracy (true cross-subject metric) is expected to be 60–75% initially with 7 subjects; this will improve as more subjects are added.

---

## 6. Shared Components (unchanged)

- `gesture_model_cnn.py` — GestureCNN, GestureCNNv2, CnnBundle, load_cnn_bundle(), quick_finetune() — no changes needed; both streams use it
- `realtime_gesture_cnn.py` — loads any CnnBundle via `load_cnn_bundle()` regardless of stream; no changes needed except updating default model path to cross_subject
- Data loading, calibration, windowing, GroupShuffleSplit logic — duplicated across both scripts (acceptable; they have different enough config that sharing would add complexity with little benefit)

---

## 7. Migration Plan

1. Create `models/per_subject/` and `models/cross_subject/` directories
2. Move existing per-subject `.pt` files to `models/per_subject/`
3. Create `train_per_subject.py` from current `train_cnn.py` with per-subject defaults
4. Create `train_cross_subject.py` from current `train_cnn.py` with cross-subject defaults
5. Keep `train_cnn.py` with a deprecation notice for 1-2 weeks, then remove
6. Update `realtime_gesture_cnn.py` default model path to `models/cross_subject/gesture_cnn_v2.pt`
7. Run `train_cross_subject.py` with LOSO to establish cross-subject baseline

---

## 8. What is NOT changed

- The MVC calibration normalization logic (already correct, just needs better data collection protocol)
- The data collection pipeline (DelsysPythonGUI.py, DataCollector/)
- The filtering pipeline (filtering.py)
- The inference logic in realtime_gesture_cnn.py
- Window size (200 samples) and step (100 samples) — already validated as optimal
- The CnnBundle format — backward compatible with existing models

---

## 9. Open Questions

1. **subject05 exclusion?** The subject has completely failed MVC calibration (mvc_ratio ≤ 1.8 across all 4 sessions). Including this data in cross-subject training may hurt the model. Recommend: exclude subject05 from cross-subject training until recalibrated.
2. **Amplitude augmentation range for per-subject**: (0.7, 1.4) is proposed based on reasoning; actual within-subject EMG amplitude variance has not been measured from this dataset. Could be validated by computing per-subject channel RMS variance across sessions.
3. **LOSO accuracy threshold**: What minimum LOSO accuracy makes the cross-subject model deployable for a new participant? Suggest 65% as minimum before using in real-time.
