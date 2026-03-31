# Technical Reference

This is the canonical technical reference for the cleaned repo state.

## Scope

The repo contains one maintained EMG control stack:

1. Delsys GUI collection
2. Preprocessing for the maintained sensor layout
3. CNN training with `GestureCNNv2`
4. Live gesture inference
5. CARLA control and evaluation logging

The repo assumes one consistent sensor placement workflow. Older experimental branches, alternate model families, and per-gesture adaptation paths are not part of the active system.

The active gesture set is fixed to:

- `left_turn`
- `right_turn`
- `neutral`
- `horn`

## Data roots

- Raw collections: `data_strict/<arm> arm/<subject>/raw/*.npz`
- Resampled and filtered training root: `data_resampled_strict/<arm> arm/<subject>/{raw,filtered}`
- Saved models: `models/strict/`

Helpers for these roots live in `project_paths.py`.

## Sensor Placement Contract

Sensor layout resolution lives in `emg/strict_layout.py`.

Right arm pair order:

- `1`
- `2`
- `3`
- `7`
- `9`
- `11`

Left arm pair order:

- `4`
- `5`
- `6`
- `8`
- `10`

Expected EMG channel counts:

- Right arm: `17`
- Left arm: `16`

Training and realtime both resolve channels by fixed pair identity from `metadata.emg_channel_labels`. Missing or mismatched pairs should fail closed.

In practical terms, this means the maintained workflow assumes that the same numbered sensors stay in the same physical positions from session to session. The code is intentionally written to trust that placement instead of trying to guess around inconsistent sensor ordering at runtime.

## Collection and preprocessing

Collection entrypoint:

```powershell
python DelsysPythonGUI.py
```

Preprocessing flow:

```powershell
python emg/resample_raw_dataset.py
python emg/filtering.py
```

`emg/resample_raw_dataset.py` interpolates each channel onto a shared time grid and preserves metadata. `emg/filtering.py` applies the maintained EMG filter chain and carries calibration arrays into the filtered files.

## Training

Active training entrypoints:

- `train_per_subject.py`
- `train_cross_subject.py`

Both scripts train `GestureCNNv2` bundles and write:

- model weights
- label maps
- architecture metadata
- training metadata
- stored metrics
- normalization statistics
- sensor-layout metadata

Default output patterns:

- per-subject: `models/strict/per_subject/<arm>/<subject>v6_4_gestures.pt`
- cross-subject: `models/strict/cross_subject/<arm>/v6_4_gestures.pt`

Shared CNN-only training helpers live in `emg/cnn_training.py`.

## Realtime inference

Realtime entrypoint:

```powershell
python realtime_gesture_cnn.py
```

Key facts:

- Channel resolution is enforced at runtime from the maintained sensor layout metadata.
- The active model loader is `emg/gesture_model_cnn.py`.
- The active published output interface is `get_latest_published_gestures()`.
- Runtime thresholds and smoothing live in `emg/runtime_tuning.py`.
- Realtime still performs neutral/MVC calibration before inference.
- Per-prediction CSV logging is available through `--prediction-log`.

Current checked-in defaults in `realtime_gesture_cnn.py`:

- `INCLUDED_GESTURES = {"neutral", "left_turn", "right_turn", "horn"}`
- right bundle: `models/strict/per_subject/right/Matthewv6_4_gestures.pt`
- left bundle: `models/strict/per_subject/left/Matthewv6_4_gestures.pt`

## CARLA integration

Canonical CARLA client:

- `carla_integration/manual_control_emg.py`

Important behavior:

- launches realtime internally for CARLA runs
- reads published split-or-single gesture output
- treats dual-horn as the reverse toggle request
- can load named scenarios
- can write per-tick drive logs and forwarded realtime logs
- uses keyboard/EMG vehicle control only in the maintained repo state
- keeps one standard RGB camera view
- keeps keyboard weather controls (`C`, `Shift+C`)

Named scenarios in `carla_integration/scenario_presets.py`:

- `lane_keep_5min`
  - map: `Town04_Opt`
  - ego spawn shifted closer to the first active checkpoint
- `highway_overtake`
  - map: `Town04_Opt`
  - reactive lead vehicle behavior during overtakes

Preferred named wrappers:

- `carla_integration/lane_keep_5min.cmd`
- `carla_integration/highway_overtake.cmd`
- `carla_integration/lane_keep_5min_eval.cmd`
- `carla_integration/highway_overtake_eval.cmd`

Free-roam practice launcher:

- `carla_integration/test_run_manual_control_emg_0_9_16.bat`
  - map: `Town03_Opt`
  - defaults: `150` vehicles, `0` pedestrians
  - `Backspace` respawns a random free-roam vehicle

## Evaluation

Evaluation scripts live in `eval_metrics/`.

Common roles:

- harvest offline bundle metrics
- summarize end-to-end latency from joined realtime and CARLA logs
- summarize CARLA drive logs
- build participant-level and aggregate tables for reports

Generated outputs under `eval_metrics/out/` and `eval_metrics/logs/` are disposable artifacts, not source-of-truth inputs.

## Maintained docs

Use these as the active documentation set:

- `README.md`
- `AGENTS.md`
- `carla_integration/README.md`
- `eval_metrics/README.md`
- `project_notes/technical_reference.md`

Subsystem notes:

- `project_notes/01_data_collection.md`
- `project_notes/02_model_architecture.md`
- `project_notes/03_training_and_validation.md`
- `project_notes/04_realtime_inference.md`
- `project_notes/05_carla_control.md`
- `project_notes/06_scenario_design_and_validation.md`
- `project_notes/07_evaluation_metrics.md`
