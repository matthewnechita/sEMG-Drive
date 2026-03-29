# AGENTS.md

Project context
- The active repo state is the CNN workflow with fixed sensor placement.
- The maintained pipeline is `DelsysPythonGUI.py` -> `emg/resample_raw_dataset.py` -> `emg/filtering.py` -> `train_per_subject.py` / `train_cross_subject.py` -> `realtime_gesture_cnn.py` -> `carla_integration/manual_control_emg.py`.
- The canonical technical note is `project_notes/technical_reference.md`.
- `carla_integration/manual_control_emg.py` is the canonical CARLA client entrypoint.
- Active roots:
  - `data_strict/`
  - `data_resampled_strict/`
  - `models/strict/`

Label contract
- Active gesture labels: `left_turn`, `right_turn`, `neutral`, `horn`.
- `neutral_buffer` is not a trainable gesture label.

Sensor placement contract
- Right arm pair order: `1, 2, 3, 7, 9, 11`
- Left arm pair order: `4, 5, 6, 8, 10`
- Right arm channel count: `17`
- Left arm channel count: `16`
- Sensor layout helpers live in `emg/strict_layout.py`.
- Realtime and training should fail closed when required sensor pairs are missing or mismatched.

Training and bundle notes
- Active trainers:
  - `train_per_subject.py`
  - `train_cross_subject.py`
- Active architecture: `GestureCNNv2`
- Active model outputs:
  - per-subject: `models/strict/per_subject/<arm>/<subject>v6_4_gestures.pt`
  - cross-subject: `models/strict/cross_subject/<arm>/v6_4_gestures.pt`
- Current checked-in realtime defaults point to:
  - `models/strict/per_subject/right/Matthewv6_4_gestures.pt`
  - `models/strict/per_subject/left/Matthewv6_4_gestures.pt`

Realtime notes
- `realtime_gesture_cnn.py` is the source of truth for live inference behavior.
- `emg/runtime_tuning.py` is the source of truth for runtime thresholds and smoothing.
- The active realtime workflow is fixed to the 4-gesture set:
  - `INCLUDED_GESTURES = {"neutral", "left_turn", "right_turn", "horn"}`
- Published output comes from `get_latest_published_gestures()`.
- `manual_control_emg.py` launches realtime internally for CARLA runs.

CARLA notes
- Named scenario wrappers are preferred for evaluation runs:
  - `carla_integration\lane_keep_5min.cmd`
  - `carla_integration\highway_overtake.cmd`
- Current named scenario maps:
  - `lane_keep_5min` -> `Town04_Opt`
  - `highway_overtake` -> `Town04_Opt`
- The free-roam launcher uses `Town03_Opt` with default ambient traffic of `10` vehicles and `18` pedestrians.
- `manual_control_emg.py` is now keyboard/EMG vehicle-only; wheel, walker, GNSS, and multi-camera mode support have been removed from the maintained path.
- Weather hotkeys are still kept:
  - `C` = next weather
  - `Shift+C` = previous weather
- In free roam, `Backspace` restarts into another random vehicle.

Common commands
- Collect data:
  - `python DelsysPythonGUI.py`
- Resample raw collections:
  - `python emg/resample_raw_dataset.py`
- Filter resampled data:
  - `python emg/filtering.py`
- Train a per-subject CNN:
  - edit `ARM`, `TARGET_SUBJECT`, `DATA_ROOT`, and `MODEL_OUT` in `train_per_subject.py`
  - run `python train_per_subject.py`
- Train a cross-subject CNN:
  - edit `ARM`, `DATA_ROOT`, and `MODEL_OUT` in `train_cross_subject.py`
  - run `python train_cross_subject.py`
- Run live dual-arm inference:
  - `python realtime_gesture_cnn.py --model-right <right_bundle> --model-left <left_bundle>`
- Run lane-keep:
  - `carla_integration\lane_keep_5min.cmd`
- Run overtake:
  - `carla_integration\highway_overtake.cmd`
