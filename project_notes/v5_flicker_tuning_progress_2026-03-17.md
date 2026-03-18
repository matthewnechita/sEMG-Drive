# v5 Flicker Tuning Progress

Date: 2026-03-17
Branch: `v5-flicker-tuning`

## Purpose

Track the neutral-flicker tuning implementation work so lab testing can focus on preset switching and data collection rather than last-minute coding.

## Completed

### Step 1: Shared runtime tuning layer

- Added `emg/runtime_tuning.py` as the shared runtime preset registry.
- Added one active selector: `ACTIVE_RUNTIME_TUNING_PRESET`.
- Added named runtime presets:
  - `baseline`
  - `flicker_mild`
  - `flicker_strong`
- Wired `realtime_gesture_cnn.py` to read its runtime knobs from the shared preset.
- Wired `carla_integration/manual_control_emg.py` to read CARLA runtime knobs from the shared preset.
- Added runtime preset logging to realtime prediction CSVs and CARLA drive CSVs.

### Step 2: Realtime anti-flicker logic

- Added softmax ambiguity rejection for the CNN path.
- Added prediction-stage trace logging for each arm:
  - raw top label/confidence
  - second label/confidence
  - margin
  - post-gate label/confidence/reason
  - post-hysteresis label/confidence
- Added margin-enabled presets:
  - `flicker_mild_margin`
  - `flicker_strong_margin`

### Step 3: CARLA steering dwell/debounce

- Added CARLA-side steering dwell before steer is applied.
- Added separate CARLA dwell knobs:
  - `active_steer_dwell_frames`
  - `neutral_steer_dwell_frames`
- Added requested-vs-applied steering logging:
  - `steer_key`
  - `applied_steer_key`
  - `steer_dwell_pending_key`
  - `steer_dwell_pending_count`
  - `steer_dwell_required`
- Added dwell-enabled presets:
  - `flicker_mild_margin_dwell2`
  - `flicker_strong_margin_dwell2`

### Step 4: Neutral recovery collection flow

- Kept the standard collection protocol unchanged.
- Added a new GUI button in `CollectDataWindow.py`:
  - `Run Neutral Recovery Protocol`
- Added a separate `neutral_recovery` protocol path.
- Recovery protocol currently collects repeated:
  - `left_turn -> neutral`
  - `right_turn -> neutral`
- Recovery-neutral samples are labeled as `neutral`, but the immediate release slice is trimmed internally with a leading-only trim.
- Recovery sessions save to a distinct output filename suffix:
  - `*_neutral_recovery_raw.npz`
- Added protocol metadata:
  - `protocol_name`
  - `neutral_recovery.lead_trim_s`
  - `neutral_recovery.trail_trim_s`
  - `neutral_recovery.sequence`

## Files Changed

- `emg/runtime_tuning.py`
- `realtime_gesture_cnn.py`
- `carla_integration/manual_control_emg.py`
- `DataCollector/CollectDataWindow.py`

## Verification Completed

- `python -m py_compile emg/runtime_tuning.py realtime_gesture_cnn.py`
- `python -m py_compile emg/runtime_tuning.py carla_integration/manual_control_emg.py`
- `python -m py_compile DataCollector/CollectDataWindow.py`
- Direct preset resolution checks for the new runtime preset names

## Not Yet Done

- No live realtime run yet with the new preset/logging path
- No live CARLA run yet with steering dwell enabled
- No live GUI run yet for the new neutral recovery button
- No training preset implementation yet
- No retraining yet using recovery data

## Upcoming Steps

### Step 5: Training-side support

Add training presets so the retrain path is repeatable without hand-editing:

- `baseline_3g`
  - standard protocol data only
  - labels limited to `neutral`, `left_turn`, `right_turn`
- `recovery_3g`
  - standard data plus `neutral_recovery` sessions
  - same 3 labels
- `recovery_3g_high_purity`
  - standard data plus `neutral_recovery` sessions
  - same 3 labels
  - higher label-purity threshold than the baseline trainer setting

Likely files:

- `train_per_subject.py`
- `train_cross_subject.py`

### Step 6: Lab-ready switching and test matrix

- Decide which runtime preset should be the first live test candidate.
- Decide which training preset should be used for the first retrain after recovery collection.
- Create the short lab matrix:
  - current bundle + `baseline`
  - current bundle + `flicker_mild_margin`
  - current bundle + `flicker_mild_margin_dwell2`
  - retrained recovery bundle + matching runtime preset

## Current Safe Default

The active runtime preset is still `baseline`. Nothing experimental is enabled by default unless `ACTIVE_RUNTIME_TUNING_PRESET` is changed.
