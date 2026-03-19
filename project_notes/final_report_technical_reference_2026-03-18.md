# Final Report Technical Reference

Date: 2026-03-18
Branch reference: `v5-flicker-tuning`

## Purpose

This document is the technical reference for the capstone final report. It is based on the code paths that are active on the current branch and is intended to cover the pipeline end to end, from data collection through offline training, realtime inference, CARLA integration, and evaluation support.

This document is intentionally grounded in what is actually used now. It does not treat archived or deprecated code as part of the final system unless explicitly called out.

## 1. Active System Scope

The active system is an EMG gesture-classification pipeline built around Delsys Trigno hardware, a strict sensor-placement policy, CNN-based gesture classification, realtime inference, and CARLA-based control evaluation.

Active top-level entrypoints:

- `python DelsysPythonGUI.py`
- `python tools/resample_raw_dataset.py`
- `python emg/filtering.py`
- `python tools/recalibrate.py --data-root data_resampled_strict`
- `python train_per_subject.py`
- `python train_cross_subject.py`
- `python realtime_gesture_cnn.py --model ...`
- `python carla_integration/manual_control_emg.py ...`

Active storage roots:

- raw strict collections: `data_strict/`
- resampled strict data: `data_resampled_strict/`
- active model bundles: `models/strict/`

The current repo root `README.md` is still mostly the upstream Delsys demo README and is not the best description of the active CNN workflow. The active workflow is defined by the files listed above.

## 2. Software Stack

The environment file currently targets:

- Python `3.10`
- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `pythonnet==3.0.3`
- `PySide6`
- `vispy`
- `PyOpenGL`
- `libemg`
- `torch`
- `torchvision`
- `plotly`

Important platform dependency:

- Windows with .NET 6+ runtime is expected for the Delsys SDK bridge through `pythonnet` and `resources/DelsysAPI.dll`.

High-level software roles:

- `PySide6`: data collection GUI
- `pythonnet` + Delsys SDK: sensor/base communication
- `libemg`: windowing and filtering helpers
- `PyTorch`: CNN training and inference
- `pygame` + CARLA Python API: simulator-side control and rendering

## 3. Data Collection Pipeline

### 3.1 Collection entrypoint

The collection GUI entrypoint is `DelsysPythonGUI.py`, which launches the Qt application and the `LandingScreenController`, then ultimately the collection window in `DataCollector/CollectDataWindow.py`.

The collection GUI supports:

- base connection
- sensor pairing with explicit pair numbers
- scanning previously paired sensors
- sample-mode selection
- live EMG streaming/plotting
- scripted labeled collection protocols

### 3.2 Sensor pairing and scan model

The GUI exposes explicit pair-number assignment during pairing. After pairing, a scan enumerates available sensors and only EMG channels are shown in the collection sensor list. The system depends on those pair numbers for the strict-layout workflow later in training and realtime inference.

### 3.3 Active collection protocols

Two labeled protocols are currently implemented in `CollectDataWindow.py`.

#### Standard protocol

The standard protocol currently uses:

- gestures:
  - `left_turn`
  - `right_turn`
  - `neutral`
  - `signal_left`
  - `signal_right`
  - `horn`
- gesture duration: `5.0 s`
- neutral duration: `5.0 s`
- repetitions: `5`
- preparation countdown: `5.0 s`
- inter-gesture rest: `1.0 s`
- label trim: `0.5 s`
- calibration enabled: yes
- calibration neutral duration: `5.0 s`
- calibration MVC duration: `5.0 s`
- calibration MVC prep countdown: `2.0 s`
- calibration minimum ratio warning threshold: `2.0x`

#### Neutral-recovery protocol

The neutral-recovery protocol currently uses:

- gestures:
  - `left_turn`
  - `right_turn`
- gesture duration: `3.0 s`
- neutral duration: `5.0 s`
- repetitions: `6`
- preparation countdown: `5.0 s`
- inter-gesture rest: `0.0 s`
- label trim: `0.5 s`
- neutral recovery lead trim: `1.0 s`
- neutral recovery trail trim: `0.0 s`
- calibration enabled: yes
- calibration neutral duration: `5.0 s`
- calibration MVC duration: `5.0 s`
- calibration MVC prep countdown: `2.0 s`

### 3.4 Labels produced during collection

The current system uses these active gesture labels:

- `left_turn`
- `right_turn`
- `neutral`
- `signal_left`
- `signal_right`
- `horn`

The inter-gesture rest label is:

- `neutral_buffer`

`neutral_buffer` is a collection label but is not treated as a trainable gesture class. Both training scripts explicitly drop it during window labeling.

### 3.5 Raw output format

Collection output is written under strict-layout roots:

- standard protocol:
  - `data_strict/<arm> arm/<subject>/raw/<subject>_session<session>_raw.npz`
- neutral-recovery protocol:
  - `data_strict/<arm> arm/<subject>/raw/<subject>_session<session>_neutral_recovery_raw.npz`

The raw `.npz` payload contains:

- `X`: raw EMG sample matrix shaped `(samples, channels)`
- `timestamps`: timestamp matrix shaped `(samples, channels)`
- `y`: per-sample labels
- `events`: protocol event list
- `metadata`: collection metadata dict
- optional calibration arrays when calibration is enabled:
  - `calib_neutral_X`
  - `calib_neutral_timestamps`
  - `calib_mvc_X`
  - `calib_mvc_timestamps`

### 3.6 Raw metadata saved with each collection

The current metadata dict includes:

- `subject`
- `arm`
- `session`
- `protocol_name`
- `data_root`
- `layout_mode`
- `gestures`
- `gesture_duration_s`
- `neutral_duration_s`
- `repetitions`
- `channel_count`
- `created_at`
- `prep_duration_s`
- `inter_gesture_rest_s`
- `label_trim_s`
- `rest_label_trim_s`
- `ramp_style`
- `emg_channel_labels`

Optional metadata is also added for:

- neutral-recovery protocol settings
- calibration enablement and durations

The `emg_channel_labels` field is critical because both strict-layout training and strict-layout realtime inference use it to resolve channel order from pair numbers and sensor identities.

## 4. Strict Sensor Placement Workflow

### 4.1 Purpose

The current active branch uses a strict fixed-position sensor policy so that training and inference see a consistent channel ordering across sessions. This replaces older mixed-layout assumptions.

### 4.2 Strict layout identifier

The strict layout helper in `emg/strict_layout.py` defines:

- `STRICT_LAYOUT_VERSION = "strict_pair_v1"`

### 4.3 Pair-to-slot mapping

Current right-arm strict mapping:

- pair `1`: `R_Avanti_1` with `1` channel
- pair `2`: `R_Avanti_2` with `1` channel
- pair `3`: `R_Avanti_3` with `1` channel
- pair `7`: `R_Maize` with `9` channels
- pair `9`: `R_Galileo` with `4` channels
- pair `11`: `R_Mini` with `1` channel

Current left-arm strict mapping:

- pair `4`: `L_Avanti_1` with `1` channel
- pair `5`: `L_Avanti_2` with `1` channel
- pair `6`: `L_Avanti_3` with `1` channel
- pair `8`: `L_Maize` with `9` channels
- pair `10`: `L_Galileo` with `4` channels

### 4.4 Strict channel counts

- right arm: `17` channels
- left arm: `16` channels

### 4.5 Enforcement behavior

Strict resolution uses the saved `metadata.emg_channel_labels` and enforces:

- required pair numbers must be present
- each slot must contribute the expected number of channels
- observed sensor kind must match the expected slot type when inferable from labels
- resolved total channel count must match the arm’s expected strict count

If these conditions fail, the training or realtime strict path fails closed instead of silently continuing with a mismatched layout.

## 5. Preprocessing Pipeline

The recommended strict preprocessing order is:

1. `python tools/resample_raw_dataset.py`
2. `python emg/filtering.py`
3. `python tools/recalibrate.py --data-root data_resampled_strict`
4. `python tools/recalibrate.py --data-root data_resampled_strict --apply` if needed
5. retrain

### 5.1 Resampling

The resampling script exists because different Delsys sensor types can produce different per-channel sampling rates. The rest of the pipeline assumes aligned channels by sample index, so raw multi-sensor streams are first interpolated onto a shared common time grid.

Current resampling settings:

- input root: `data_strict`
- output root: `data_resampled_strict`
- raw file pattern: `*_raw.npz`
- target sampling rate: `2000.0 Hz`
- overwrite: `False` by default

Current resampling method:

- estimate per-channel sampling rates from timestamp differences
- compute the overlapping valid time interval across all channels
- build one common uniform time grid over that overlap
- linearly interpolate each channel onto that grid
- transfer labels onto the new grid using nearest-neighbor remapping from channel-0 timestamps
- resample calibration segments in the same way when present

The resampler stores resampling metadata back into `metadata["resampling"]`, including:

- target frequency
- source per-channel frequencies
- overlap window bounds
- input/output sample counts
- creation timestamp

### 5.2 Filtering

The active offline filtering stack is implemented in `emg/filtering.py` and mirrored in the realtime code.

Current filter stack:

- notch filter at `60 Hz`, bandwidth `3`
- notch filter at `120 Hz`, bandwidth `3`
- bandpass filter `20–450 Hz`, order `6`

Important operational rule:

- if the filtering stack changes, the filtered strict dataset must be regenerated and the models retrained

Filtering reads resampled raw `.npz` files and writes filtered files next to the subject root under a `filtered/` directory, for example:

- input:
  - `data_resampled_strict/right arm/<subject>/raw/..._raw.npz`
- output:
  - `data_resampled_strict/right arm/<subject>/filtered/..._filtered.npz`

The filtered file preserves:

- `emg`
- `fs`
- labels and timestamps
- metadata
- events
- calibration arrays, converted from raw `*_X` arrays into filtered `*_emg` arrays

### 5.3 Retroactive recalibration

`tools/recalibrate.py` is a recovery path for sessions where explicit MVC calibration is too weak.

The script checks the median MVC-to-neutral RMS ratio using:

- existing `calib_neutral_emg`
- existing `calib_mvc_emg`

Default threshold for replacement:

- `1.5x`

If a session fails this threshold, the script can compute replacement calibration from the filtered session itself:

- `neutral_mean`: mean EMG over labels in `{"neutral", "neutral_buffer"}`
- `mvc_scale`: `95th` percentile over labels in `{"horn", "left_turn", "right_turn", "signal_left", "signal_right"}`

Original calibration arrays are preserved under backup keys so the change is reversible.

## 6. Windowing and Label Generation

Both active training scripts use the same basic windowing configuration:

- window size: `200` samples
- step size: `100` samples

Given the resampled target frequency of `2000 Hz`, these correspond to:

- window duration: `100 ms`
- step duration: `50 ms`

Label assignment is window-wise and uses majority vote over the per-sample labels in each window.

Current label filtering rules:

- `neutral_buffer` is mapped to `None` and dropped
- windows below the minimum label-confidence threshold are dropped
- if `INCLUDED_GESTURES` is set, windows outside that set are dropped

Current minimum label-confidence thresholds:

- per-subject training: `0.85`
- cross-subject training: `0.75`

## 7. Model Architecture

### 7.1 Active architecture

The active architecture is `GestureCNNv2` in `emg/gesture_model_cnn.py`.

An older `GestureCNN` is still present for backward compatibility, but the active training scripts instantiate `GestureCNNv2`.

### 7.2 Input representation

The model consumes filtered EMG windows shaped:

- `(batch, channels, time)`

The trainers convert the window arrays into this format before passing them to the network.

### 7.3 GestureCNNv2 structure

Current `GestureCNNv2` components:

- input `InstanceNorm1d`
- stem:
  - `Conv1d(in_channels, 32, kernel_size=11, padding=5, bias=False)`
  - `BatchNorm1d(32)`
  - `ReLU`
- stage 1:
  - residual block at `32` channels
  - squeeze-and-excitation channel attention
  - `MaxPool1d(2,2)`
- stage 2:
  - `1x1` projection from `32` to `64`
  - residual block at `64` channels
  - channel attention
  - `MaxPool1d(2,2)`
- stage 3:
  - `1x1` projection from `64` to `128`
  - residual block at `128` channels
  - channel attention
  - `AdaptiveAvgPool1d(1)`
  - flatten
- classification head:
  - concatenate `128` learned features with a `1`-dimensional raw energy scalar
  - `Linear(129, num_classes)`

### 7.4 Energy bypass

The model computes raw window energy before input normalization:

- `energy = mean(x^2)` over channels and time

This energy scalar is concatenated into the head to preserve amplitude information that would otherwise be damped by per-window `InstanceNorm1d`. The explicit intent in the code is to help the model separate near-zero neutral windows from active gesture windows even after normalization.

### 7.5 External normalization behavior

The bundle still stores train-set `mean` and `std`, but `CnnBundle.standardize()` skips external z-score normalization when `metadata["use_instance_norm_input"]` is true. That is the current path for `GestureCNNv2`.

## 8. Per-Subject Training Pipeline

### 8.1 Purpose

`train_per_subject.py` trains a subject-specific model for one arm.

Current default config on this branch:

- arm: `right`
- target subject: `Matthew`
- data root: `data_resampled_strict/right arm`
- file pattern: `*_filtered.npz`
- output path: `models/strict/per_subject/right/Matthew_all_gesture_15.pt`

### 8.2 Current per-subject training settings

- window size: `200`
- window step: `100`
- calibration: enabled
- MVC percentile: `95.0`
- minimum label confidence enabled: yes
- minimum label confidence: `0.85`
- test size: `0.2`
- random seed: `42`
- batch size: `512`
- epochs: `60`
- learning rate: `1e-4`
- dropout: `0.25`
- label smoothing: `0.05`
- augmentation enabled: yes
- amplitude augmentation range: `(0.7, 1.4)`
- augmentation probability: `0.4`
- channel layout mode: `strict`
- included gestures by default: all six active gesture labels

### 8.3 Calibration use at training time

If calibration arrays are available and pass quality checks, the trainer applies:

- `emg = (emg - neutral_mean) / mvc_scale`

Training-time calibration quality threshold:

- median MVC/neutral ratio must be at least `1.5x`

If calibration quality is below that threshold, normalization is skipped for that file.

### 8.4 Data split

The trainer uses:

- group-aware file split via `GroupShuffleSplit` when there are at least two session files
- otherwise a stratified random split over windows

This keeps windows from the same file together in the train or test split when possible.

### 8.5 Per-subject augmentation

The current per-subject GPU-native augmentation pipeline includes:

- optional permutation of exchangeable single-channel groups when salvage mode is used
- amplitude scaling
- additive Gaussian noise with random SNR between `10` and `30 dB`
- temporal circular shift with integer shift in `[-20, 20]` samples
- single-channel dropout
- temporal stretch through interpolation with factor sampled from approximately `0.8` to `1.2`

### 8.6 Optimization

Current optimizer and scheduler:

- optimizer: Adam
- loss: cross-entropy with label smoothing
- scheduler: `ReduceLROnPlateau`
  - mode `min`
  - factor `0.5`
  - patience `5`
  - threshold `1e-4`
  - minimum learning rate `1e-6`

The trainer tracks the best validation accuracy and restores the best checkpoint before saving.

### 8.7 Saved per-subject bundle

Saved bundle contents:

- `model_state`
- `normalization.mean`
- `normalization.std`
- `label_to_index`
- `index_to_label`
- `architecture`
- `metadata`

Saved metadata includes:

- creation time
- stream type
- subject info
- included gestures
- window size and step
- channel count
- labels
- split mode and test size
- calibration settings
- strict-layout metadata
- training hyperparameters
- offline metrics
- train/test file names

## 9. Cross-Subject Training Pipeline

### 9.1 Purpose

`train_cross_subject.py` trains a pooled model intended to generalize to unseen subjects.

Current default config on this branch:

- arm: `right`
- data root: `data_resampled_strict/right arm`
- output path: `models/strict/cross_subject/right/gesture_cnn_v3_3_gestures.pt`
- included gestures: `None` by default, which means include all available labels
- LOSO evaluation: `False` by default

### 9.2 Current cross-subject training settings

- window size: `200`
- window step: `100`
- calibration: enabled
- MVC percentile: `95.0`
- minimum label confidence enabled: yes
- minimum label confidence: `0.75`
- test size: `0.2`
- random seed: `42`
- batch size: `512`
- epochs: `80`
- learning rate: `1e-4`
- dropout: `0.25`
- label smoothing: `0.05`
- augmentation enabled: yes
- amplitude augmentation range: `(0.5, 2.0)`
- augmentation probability: `0.5`
- channel layout mode: `strict`

### 9.3 Subject-balanced sampling

Cross-subject training always uses subject-balanced sampling through `WeightedRandomSampler`.

The per-sample weight is:

- `1 / (# training windows from that subject)`

This reduces dominance by subjects with more training windows.

### 9.4 Cross-subject augmentation

The cross-subject augmentation operations are similar to the per-subject trainer:

- amplitude scaling
- additive Gaussian noise with random SNR between `10` and `30 dB`
- temporal shift in `[-20, 20]`
- channel dropout
- temporal stretch through interpolation with factor approximately `0.85` to `1.15`

### 9.5 LOSO evaluation support

The cross-subject trainer contains a leave-one-subject-out evaluation path, intended to measure true subject-generalization performance. The code explicitly warns that the final pooled model’s in-distribution test accuracy is not the same thing as true cross-subject accuracy.

The file comments recommend LOSO evaluation before deployment and note a warning threshold of `65%` mean LOSO accuracy.

## 10. Offline Metrics Produced by Training

Both active trainers compute and print:

- test accuracy
- balanced accuracy
- macro precision
- macro recall
- macro F1
- weighted precision
- weighted recall
- weighted F1
- worst-class recall
- maximum precision-recall gap
- confusion-to-neutral rates
- neutral prediction false-positive rate
- confusion matrices:
  - counts
  - row-normalized
  - column-normalized

These metrics are also stored in the saved model bundle metadata and harvested later by the evaluation scripts.

## 11. Realtime Inference Pipeline

### 11.1 Entry point

The active realtime entrypoint is `realtime_gesture_cnn.py`.

### 11.2 Current realtime defaults on this branch

- window size: `200`
- window step: `100`
- realtime filter mode: `"scipy_stateful"`
- fallback rolling filter mode available: `"libemg_rolling"`
- filter warmup for rolling mode: `200`
- realtime resampling enabled: yes
- realtime target sampling rate: `2000.0 Hz`
- low-confidence fallback label: `neutral`
- included gestures subset: `{"neutral", "left_turn", "right_turn"}`
- active inference mode: `"dual"`
- default right-arm bundle:
  - `models/strict/per_subject/right/Matthew_3_gesture_15.pt`
- default left-arm bundle:
  - `models/strict/per_subject/left/Matthew_3_gesture_15.pt`

### 11.3 Runtime tuning preset

The active runtime preset is:

- `flicker_mild_margin`

This preset sets:

- smoothing: `3`
- minimum confidence: `0.80`
- dual-arm agree threshold: `0.55`
- output hysteresis: enabled
- neutral enter threshold: `0.80`
- hysteresis enter confirmation frames: `2`
- hysteresis switch confirmation frames: `2`
- hysteresis neutral confirmation frames: `2`
- softmax ambiguity rejection: enabled
- softmax reject minimum confidence: `0.80`
- softmax reject minimum margin: `0.10`

### 11.4 Realtime filtering and calibration

The realtime file explicitly mirrors the offline filtering stack:

- 60 Hz notch
- 120 Hz notch
- 20–450 Hz bandpass

If calibration is available in the run, realtime also applies:

- neutral mean subtraction
- MVC scaling

before window classification.

### 11.5 Label restriction and confidence gating

Realtime can restrict active output labels to a subset of the labels present in the bundle. Disallowed classes are zeroed and probabilities are renormalized over the allowed set.

Prediction decoding then applies:

- minimum-confidence gate
- optional softmax ambiguity rejection
- optional prototype rejection if prototype mode is enabled
- output hysteresis to reduce flicker

The prototype classifier path exists, but the default runtime configuration has it disabled. The active path is the softmax decoder.

### 11.6 Dual-arm fusion

The current dual-arm fusion logic is:

- if both arms agree on the same label and confidence passes the dual-arm agree threshold, publish the shared label
- if the arms disagree, evaluate each arm independently against the single-arm threshold
- prefer confident non-neutral labels over neutral fallback
- if neither arm has a confident active label, fall back to confident neutral

The realtime module now maintains full per-arm state internally and exposes published output separately from the old single-label compatibility accessor.

### 11.7 Published output semantics

Published dual-arm output uses these semantics:

- `mode="single"`:
  - used when both arms agree or only one arm is effectively published
- `mode="split"`:
  - used when the two arms publish different labels

Realtime also supports optional per-prediction CSV logging through `--prediction-log`.

## 12. CARLA Integration

### 12.1 Entry point

The active CARLA client entrypoint is:

- `carla_integration/manual_control_emg.py`

The file is a CARLA manual-control client extended to launch and consume `realtime_gesture_cnn.py` internally.

### 12.2 Current CARLA-side defaults

- client FPS limit: `30`
- low graphics mode: enabled
- no-rendering mode: disabled by default
- camera resolution in low mode: `640x360`
- camera FPS in low mode: `10`
- HUD hidden by default

### 12.3 Runtime preset coupling

CARLA uses the same runtime tuning preset name as the realtime module and additionally consumes CARLA-specific tuning values from that preset.

For the active `flicker_mild_margin` preset, CARLA tuning remains at the base values:

- gesture max age: `0.75 s`
- active steer dwell: `1` frame
- neutral steer dwell: `1` frame
- steering values:
  - left: `-0.08`
  - right: `0.08`
  - left_strong: `-0.4`
  - right_strong: `0.4`
  - neutral: `0.0`

### 12.4 Gesture-to-vehicle mapping

The current CARLA integration treats gestures as steering/signal/reverse commands only. It does not generate throttle or braking from gestures.

Dual-arm gesture resolution:

- reverse:
  - both arms must publish `horn` simultaneously
- turn signals:
  - `signal_left` present on either arm turns on left signal
  - `signal_right` present on either arm turns on right signal
  - both simultaneously cancel to off
- steering:
  - both arms `left_turn` -> `left_strong`
  - both arms `right_turn` -> `right_strong`
  - one `left_turn` anywhere -> `left`
  - one `right_turn` anywhere -> `right`
  - conflicting left/right turn labels -> `neutral`

CARLA also enforces gesture freshness: stale outputs older than the configured max age are ignored.

### 12.5 Logging

The CARLA integration supports:

- realtime prediction CSV forwarding via `--realtime-log`
- per-tick drive logging via `--carla-log`
- convenience log-directory creation via `--eval-log-dir`

The drive log stores:

- prediction sequence
- prediction and publish timestamps
- gesture freshness
- published mode
- right/left labels and confidences
- steering dwell state
- applied steer
- throttle/brake/reverse
- vehicle position
- event counts and other control-state data

### 12.6 Map loading

The CARLA client supports `--map <name>` and calls `client.load_world(...)` from the client side before vehicle and HUD setup. This is the active map-selection path on this repo state.

## 13. Evaluation and Reporting Toolchain

The evaluation utilities live under `eval_metrics/`.

### 13.1 Offline model metrics

`harvest_model_metrics.py` scans saved `.pt` or `.pkl` bundles and exports their stored metadata and core offline metrics into CSV/JSON.

### 13.2 Session-level diagnostics

`diagnose_session_recall.py` evaluates a saved bundle against filtered subject sessions and reports per-file accuracy and per-label recall/confusion patterns.

### 13.3 Realtime behavior metrics

`realtime_behavior_metrics.py` computes segment-based behavior metrics from prompted prediction logs, including:

- time to first correct prediction
- time to stable prediction
- label flip rate
- label flip fraction
- carryover stale rate
- carryover stale duration
- confidence summaries
- balanced accuracy across prompt labels

### 13.4 Latency analysis

`analyze_latency.py` joins realtime and CARLA logs on `prediction_seq` and computes:

- classifier latency
- publish latency
- control latency
- end-to-end latency

It also reports per-label end-to-end latency summaries.

### 13.5 Drive metrics

`analyze_drive_metrics.py` summarizes CARLA run logs into:

- mean lane error
- lane error RMSE
- lane invasions
- collisions
- completion time
- steering smoothness
- command success rate

### 13.6 Table builder

`build_eval_tables.py` builds merged tables for:

- capstone final report
- research paper

It supports rows for:

- 3-gesture per-subject
- 3-gesture cross-subject
- 5-gesture per-subject
- 5-gesture cross-subject

The evaluation README also documents runtime log collection and table-building workflows.

## 14. Active vs Inactive Code Paths

### 14.1 Active

The active workflow is the strict-layout CNN path:

- `DelsysPythonGUI.py`
- `DataCollector/CollectDataWindow.py`
- `tools/resample_raw_dataset.py`
- `emg/filtering.py`
- `tools/recalibrate.py`
- `train_per_subject.py`
- `train_cross_subject.py`
- `realtime_gesture_cnn.py`
- `carla_integration/manual_control_emg.py`
- `eval_metrics/*`

### 14.2 Present but not active by default

These exist but are not the active default path:

- legacy `GestureCNN`
- salvage layout path in `train_per_subject.py`
- prototype classifier path in realtime
- older `realtime_confidence_analysis.py` pair-membership logic

### 14.3 Archived/deprecated

The following are historical and should not be presented as the active system:

- old SVM / handcrafted-feature pipeline
- files in `code_archive/`
- deprecated one-off dataset cleanup scripts

## 15. Recommended Report Usage

For the final report, this document can support:

- system architecture section
- methodology section
- implementation details section
- evaluation methodology section
- limitations section

Recommended caution:

- if the actual experiments reported in the final document used a gesture subset different from the current code defaults, report the experiment-specific subset and note the branch default separately
- if a model bundle path or runtime preset changed after results were collected, the report should distinguish “branch default at time of writing” from “configuration used for the reported experiment”
