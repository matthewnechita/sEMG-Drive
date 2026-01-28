# AGENTS.md

Project context
- Data layout: data/<subject>/{raw,filtered,features}
- Gestures: left_turn, right_turn, neutral, signal_left, signal_right, horn
  - Inter-gesture buffer is labeled `neutral_buffer` (not part of training).
- Offline pipeline: data_collection.py -> filtering.py -> feature_extraction.py -> train_classifier.py
- Real-time pipeline: realtime_gesture.py uses gesture_model.py and the saved model bundle

Model contract
- Features: MAV, RMS, WL, ZC, SSC (order from feature_list)
- Windowing: 200 samples, step 100 (unless overridden by model metadata)
- Channels: must match model metadata at runtime
- Classifier: SVM + StandardScaler (swap later with same contract)

Common commands
- Collect data:
  python data_collection.py --subject S01 --session 01 --output-dir data/iggy
- Filter new raw files:
  python filtering.py
- Extract features:
  python feature_extraction.py
- Train/export model:
  python train_classifier.py --data-root data --model-out models/gesture_classifier.pkl
- Quick compare (no CV/grid search; picks best C and saves model):
  python compare_svm_quick.py --svm-c 1 100 --svm-gamma scale --model-out models/gesture_classifier_new.pkl
- Run live inference:
  python realtime_gesture.py --model models/gesture_classifier.pkl --show-confidence

Notes
- data_collection.py now saves into a raw/ subfolder with _raw.npz suffix.
- Inter-gesture rest segments are labeled `neutral_buffer` (filtered out during feature extraction).
- filtering.py and feature_extraction.py skip files when outputs already exist.
- EMG drift over time can cause gesture flicker; collect longer sessions and keep placement consistent.
- For realtime stability, use `--smoothing` and `--min-confidence` (e.g. `--smoothing 9 --min-confidence 0.7 --low-confidence-label neutral --show-confidence`).
- Consider per-channel normalization (rolling mean/std) in both training and realtime if drift persists.
- If realtime starts from a non-neutral pose and biases to one label, treat it as baseline/pose shift; mitigate via per-session calibration (neutral + MVC) with the same normalization in training + realtime, add an "other/rest" class, and/or rely on confidence gating to force neutral when uncertain.
- Future control integration: `realtime_gesture.py` has a `control_hook` stub; plan to extend it to accept a continuous strength value (e.g., `control_hook(gesture, strength)`) for variable steering intensity once the control module is added.
- Future change: shrink SVM grid search, report best C/gamma after tuning, and consider a dedicated optimization module.
