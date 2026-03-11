# AGENTS.md

Project context
- Data layout: data/<subject>/{raw,filtered,features}
- Gestures: left_turn, right_turn, neutral, signal_left, signal_right, horn
  - Inter-gesture buffer is labeled `neutral_buffer` (not part of training).
- Offline SVM pipeline: data_collection.py -> filtering.py -> feature_extraction.py -> train_classifier.py
- Offline CNN pipeline: data_collection.py -> filtering.py -> train_cnn.py (windows raw filtered EMG)
- Real-time SVM pipeline: realtime_gesture.py uses gesture_model.py and the saved SVM bundle
- Real-time CNN pipeline: realtime_gesture_cnn.py uses gesture_model_cnn.py and the CNN bundle

Model contract
- Features: MAV, RMS, WL, ZC, SSC (order from feature_list)
- Windowing: 200 samples, step 100 (unless overridden by model metadata)
- Channels: must match model metadata at runtime
- Classifier: SVM + StandardScaler (swap later with same contract)

CNN contract
- Input: filtered EMG windows (channels x time), no handcrafted features
- Standardization: per-channel z-score (train-only stats saved in bundle)
- Softmax used at inference; temporal smoothing only online
- Bundle: model_state + mean/std + label map + metadata
- Per-subject bundle path: models/{subject}_gesture_cnn.pt
- Global bundle path (PER_SUBJECT_MODELS=False): models/gesture_cnn.pt
- All bundles share the same label contract (global label set across all data)

Per-subject model workflow (default, PER_SUBJECT_MODELS = True in train_cnn.py)
- EMG is highly person-specific; per-subject models consistently outperform cross-subject models.
- train_cnn.py trains one model per subject found under data/ and saves each to models/{subject}_gesture_cnn.pt.
- Training prints which session files went into train vs. test so results are reproducible.
- Each run also prints per-subject accuracy breakdown when multiple subjects exist in the test split.
- To switch back to a single cross-subject model, set PER_SUBJECT_MODELS = False in train_cnn.py.

Common commands
- Collect data (GUI):
  python DelsysPythonGUI.py  (enter subject ID + session # in the UI before running protocol)
- Filter new raw files:
  python filtering.py
- Extract features:
  python feature_extraction.py
- Train/export model:
  python train_classifier.py --data-root data --model-out models/gesture_classifier.pkl
- Train/export per-subject CNN models (default):
  python train_cnn.py
  → saves models/{subject}_gesture_cnn.pt for every subject found under data/
- Train/export single global CNN model:
  Set PER_SUBJECT_MODELS = False in train_cnn.py, then: python train_cnn.py
- Quick compare (no CV/grid search; picks best C and saves model):
  python compare_svm_quick.py --svm-c 1 100 --svm-gamma scale --model-out models/gesture_classifier_new.pkl
- Run live inference:
  python realtime_gesture.py --model models/gesture_classifier.pkl --show-confidence
- Run live CNN inference (per-subject — replace subject01 with actual subject ID):
  python realtime_gesture_cnn.py --model models/subject01_gesture_cnn.pt
- Train cross-subject CNN (current default is two-stage):
  python train_cross_subject.py
- Run live cross-subject CNN (two-stage):
  python realtime_gesture_cnn.py --two-stage --model-stage-a models/cross_subject/right/gesture_cnn_v3_m2_excl05_label_smoothing005_stage_a_neutral_active.pt --model-stage-b models/cross_subject/right/gesture_cnn_v3_m2_excl05_label_smoothing005_stage_b_active_gestures.pt
- Run live cross-subject CNN (single-stage fallback):
  python realtime_gesture_cnn.py --no-two-stage --model models/cross_subject/right/gesture_cnn_v3_m2_excl05_label_smoothing005.pt

Cross-subject CNN status (March 4, 2026)
- `train_cross_subject.py` default settings:
  - `MODEL_OUT = models/cross_subject/right/gesture_cnn_v3_m2_excl05_label_smoothing005.pt`
  - `MIN_LABEL_CONFIDENCE = 0.8`
  - `LABEL_SMOOTHING = 0.05`
  - `USE_TWO_STAGE = True` (Stage A neutral/active + Stage B active gestures)
- Two-stage model outputs are saved as:
  - `..._stage_a_neutral_active.pt`
  - `..._stage_b_active_gestures.pt`
- Realtime two-stage support is implemented in `realtime_gesture_cnn.py`:
  - Flags: `--two-stage`, `--model-stage-a`, `--model-stage-b`, `--no-two-stage`
  - Stage A gate thresholds: `TWO_STAGE_ACTIVE_ENTER_THRESHOLD=0.60`, `TWO_STAGE_ACTIVE_EXIT_THRESHOLD=0.45`
  - Safety check: Stage B must NOT contain `neutral` label.
  - Limitation: two-stage is currently single-arm only (not dual-arm).
- Realtime calibration threshold aligned with training:
  - `MVC_MIN_RATIO = 1.5` in realtime (matches training policy).
- Current realtime config in file is tuned for responsiveness tests:
  - `SMOOTHING=7`, `MIN_CONFIDENCE=0.55`.
- `INCLUDED_GESTURES` is still code-driven; keep using it for custom gesture subset experiments.

Notes
- data_collection.py now saves into a raw/ subfolder with _raw.npz suffix.
- Protocol update: after calibration and between gestures, inter-gesture rest is labeled `neutral_buffer`.
- Inter-gesture rest segments are labeled `neutral_buffer` and are filtered out during feature extraction/CNN windowing.
- train_cnn.py supports optional label-confidence filtering to drop ambiguous windows (USE_MIN_LABEL_CONFIDENCE + MIN_LABEL_CONFIDENCE).
- filtering.py and feature_extraction.py skip files when outputs already exist.
- train_cnn.py uses ReduceLROnPlateau on eval loss (manual LR-reduction log).
- realtime_gesture_cnn.py loads CNN bundles with torch.load(weights_only=False) for PyTorch 2.6+ compatibility.
- Data collection GUI plot throttling: plot updates ~30 Hz, plot queue maxlen 64, plot window 5,000 samples to reduce lag.
- EMG drift over time can cause gesture flicker; collect longer sessions and keep placement consistent.
- Realtime legacy defaults (older runs): SMOOTHING=11 (~550ms), MIN_CONFIDENCE=0.65.
- Consider per-channel normalization (rolling mean/std) in both training and realtime if drift persists.
- If realtime starts from a non-neutral pose and biases to one label, treat it as baseline/pose shift; mitigate via per-session calibration (neutral + MVC) with the same normalization in training + realtime, add an "other/rest" class, and/or rely on confidence gating to force neutral when uncertain.
- Adding a new participant's data to a cross-subject model rarely improves test accuracy because: (a) the test metric is dominated by existing subjects, (b) inter-subject EMG patterns don't transfer without a larger model. Use PER_SUBJECT_MODELS = True instead.
- Future control integration: `realtime_gesture.py` has a `control_hook` stub; plan to extend it to accept a continuous strength value (e.g., `control_hook(gesture, strength)`) for variable steering intensity once the control module is added.
- Future change: shrink SVM grid search, report best C/gamma after tuning, and consider a dedicated optimization module.
- One-off data cleanup scripts are deprecated: relabel_neutral_buffers_DEPRECATED..py, drop_first_gesture_run_DEPRECATED.py (kept for history only).
- Dual-arm plan (Feb 13, 2026):
  - Use two separate CNN models: one for right arm, one for left arm.
  - Collect data per arm separately (no simultaneous collection).
  - Keep channel count consistent within each arm's dataset/model; left can have fewer channels than right if fixed.
  - Inference: compute left/right labels + confidences, then fuse post-hoc:
    - If both arms agree with high confidence, collapse to one control label (strengthen).
    - If they disagree, emit per-arm labels or pick higher-confidence label if above threshold; otherwise neutral.
- Dev environment note:
  - Expected conda env for training/inference/testing is `capstone-emg` (has `torch` and `libemg`).
- Branch planning note (March 11, 2026):
  - See `project_notes/strict_sensor_placement_branch_plan_2026-03-11.txt`
  - Current branch preserves the salvage path (`USE_SENSOR_TYPE_CANONICALIZATION=True`, `USE_SINGLE_BLOCK_PERMUTATION_AUGMENTATION=True`).
  - Planned comparison branch is a strict fixed-position workflow with clean recollection and both toggles disabled.
