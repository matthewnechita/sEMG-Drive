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
- Bundle: model_state + mean/std + label map + metadata (models/gesture_cnn.pt)

Common commands
- Collect data:
  python data_collection.py --subject S01 --session 01 --output-dir data/iggy
- Filter new raw files:
  python filtering.py
- Extract features:
  python feature_extraction.py
- Train/export model:
  python train_classifier.py --data-root data --model-out models/gesture_classifier.pkl
- Train/export CNN:
  python train_cnn.py
- Quick compare (no CV/grid search; picks best C and saves model):
  python compare_svm_quick.py --svm-c 1 100 --svm-gamma scale --model-out models/gesture_classifier_new.pkl
- Run live inference:
  python realtime_gesture.py --model models/gesture_classifier.pkl --show-confidence
- Run live CNN inference:
  python realtime_gesture_cnn.py --model models/gesture_cnn.pt

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
- For realtime stability, use `--smoothing` and `--min-confidence` (e.g. `--smoothing 9 --min-confidence 0.7 --low-confidence-label neutral --show-confidence`).
- Consider per-channel normalization (rolling mean/std) in both training and realtime if drift persists.
- If realtime starts from a non-neutral pose and biases to one label, treat it as baseline/pose shift; mitigate via per-session calibration (neutral + MVC) with the same normalization in training + realtime, add an "other/rest" class, and/or rely on confidence gating to force neutral when uncertain.
- Future control integration: `realtime_gesture.py` has a `control_hook` stub; plan to extend it to accept a continuous strength value (e.g., `control_hook(gesture, strength)`) for variable steering intensity once the control module is added.
- Future change: shrink SVM grid search, report best C/gamma after tuning, and consider a dedicated optimization module.
- One-off data cleanup scripts are deprecated: relabel_neutral_buffers_DEPRECATED..py, drop_first_gesture_run_DEPRECATED.py (kept for history only).
