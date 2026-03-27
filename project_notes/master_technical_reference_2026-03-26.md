# Master Technical Reference

Date basis: 2026-03-26  
Code basis: current active working tree on branch `test`  
Scope: active, non-archived repo code paths only

## Purpose

This document is the current technical reference for the active EMG pipeline in this repository. It is meant to replace older branch-specific notes when you need one place that explains:

- what the active system is
- how data is collected and transformed
- how the current model families work
- how training, realtime inference, CARLA integration, and evaluation fit together

This document excludes `archive/` and other deprecated code paths except where a repo-state note is needed to explain a stale comment or missing helper.

## 1. Active System Map

The active stack is:

- `DelsysPythonGUI.py`
  - GUI entrypoint for collection.
- `DataCollector/CollectDataWindow.py`
  - strict-layout collection protocols and raw `.npz` writing.
- `emg/resample_raw_dataset.py`
  - timestamp-based resampling onto a shared fixed-rate grid.
- `emg/filtering.py`
  - offline EMG filtering and filtered dataset export.
- `train_per_subject.py`
  - one-subject, one-arm training with grouped cross-validation.
- `train_cross_subject.py`
  - pooled one-arm training with zero-shot LOSO and calibrated LOSO support.
- `realtime_gesture_cnn.py`
  - live strict-layout inference for single-arm or dual-arm use.
- `carla_integration/manual_control_emg.py`
  - CARLA client that launches realtime internally and maps EMG output to vehicle control.
- `eval_metrics/*.py`
  - offline, realtime, latency, drive, and table-building evaluation utilities.

The current active model families are:

- `cnn_v2`
- `metric_tcn`

Both are loaded through the same bundle interface.

## 2. Roots, Paths, and File Layout

`project_paths.py` defines the active strict roots:

- `STRICT_DATA_ROOT = data_strict`
- `STRICT_RESAMPLED_ROOT = data_resampled_strict`
- `STRICT_MODELS_ROOT = models/strict`

Arm-specific strict paths are always:

- `data_strict/right arm/<subject>/raw/*.npz`
- `data_strict/left arm/<subject>/raw/*.npz`
- `data_resampled_strict/right arm/<subject>/raw/*.npz`
- `data_resampled_strict/right arm/<subject>/filtered/*.npz`
- `data_resampled_strict/left arm/<subject>/raw/*.npz`
- `data_resampled_strict/left arm/<subject>/filtered/*.npz`
- `models/strict/per_subject/<arm>/*.pt`
- `models/strict/cross_subject/<arm>/*.pt`

The current code assumes left and right arm datasets stay physically separate and are trained separately.

## 3. Collection Pipeline

### 3.1 Entry point

Collection starts from:

```bash
python DelsysPythonGUI.py
```

`DelsysPythonGUI.py` constructs `LandingScreenController`, which instantiates `CollectDataWindow`.

### 3.2 Collection protocols

`DataCollector/CollectDataWindow.py` defines `TrialConfig` and several protocol builders. The active protocol families in code are:

- `standard`
- `left_turn_focus`
- `right_turn_focus`
- `horn_focus`
- `signal_left_focus`
- `signal_right_focus`
- `neutral_dynamic`
- `neutral_recovery`

The current standard protocol is:

- gestures: `["left_turn", "right_turn", "neutral", "horn"]`
- gesture duration: `5.0 s`
- neutral duration: `5.0 s`
- repetitions: `5`
- inter-gesture rest: `1.5 s`
- label trim: `0.5 s`

The collection GUI can also capture calibration segments before the task protocol:

- neutral calibration segment
- MVC calibration segment

These are stored in the raw output when calibration is enabled.

### 3.3 Raw output contract

Raw strict collection files are written as compressed `.npz` files under the strict raw root. The output payload contains:

- `X`
  - raw EMG samples shaped `(samples, channels)`
- `timestamps`
  - per-channel timestamps aligned to the raw acquisition stream shape
- `y`
  - sample-level string labels
- `events`
  - session event log
- `metadata`
  - collection/session metadata

If calibration was enabled, the file also stores:

- `calib_neutral_X`
- `calib_neutral_timestamps`
- `calib_mvc_X`
- `calib_mvc_timestamps`

### 3.4 Raw metadata

The collection metadata includes:

- subject
- arm
- session
- protocol name
- data root
- `layout_mode = "strict"`
- gesture list
- durations and repetition counts
- trim settings
- creation timestamp
- `emg_channel_labels`

`emg_channel_labels` is important: strict layout resolution depends on it.

## 4. Label Contract

The gesture vocabulary present in active code is:

- `left_turn`
- `right_turn`
- `neutral`
- `signal_left`
- `signal_right`
- `horn`

The collection code also emits:

- `neutral_buffer`

`neutral_buffer` is an inter-gesture rest label. It is intentionally not treated as a trainable gesture. During training-data window generation, windows whose majority label is `neutral_buffer` are dropped.

The current checked-in trainer and realtime defaults use the 4-gesture subset:

- `neutral`
- `left_turn`
- `right_turn`
- `horn`

## 5. Strict Sensor Placement Workflow

The strict workflow is implemented in `emg/strict_layout.py`.

### 5.1 Strict layout version

Current identifier:

- `STRICT_LAYOUT_VERSION = "strict_pair_v1"`

### 5.2 Fixed pair-to-slot contract

The strict arm layouts are hard-coded as fixed pair identities.

Right arm:

- pair `1` -> `R_Avanti_1` -> 1 channel
- pair `2` -> `R_Avanti_2` -> 1 channel
- pair `3` -> `R_Avanti_3` -> 1 channel
- pair `7` -> `R_Maize` -> 9 channels
- pair `9` -> `R_Galileo` -> 4 channels
- pair `11` -> `R_Mini` -> 1 channel

Left arm:

- pair `4` -> `L_Avanti_1` -> 1 channel
- pair `5` -> `L_Avanti_2` -> 1 channel
- pair `6` -> `L_Avanti_3` -> 1 channel
- pair `8` -> `L_Maize` -> 9 channels
- pair `10` -> `L_Galileo` -> 4 channels

### 5.3 Channel counts

Strict totals are:

- right arm: `17` channels
- left arm: `16` channels

### 5.4 Enforcement behavior

Strict resolution parses pair numbers from `metadata.emg_channel_labels`, checks that the required pairs exist, validates expected channel counts per pair, and reorders channels into canonical strict slot order.

This is fail-closed behavior. Missing, duplicated, or inconsistent strict pair data raises an error rather than silently guessing.

### 5.5 Salvage path

A non-archived but non-default salvage path still exists in `emg/channel_layout.py` and `train_per_subject.py`.

That path:

- infers channel kind from labels or per-channel sampling rate
- canonicalizes to `("galileo", "single", "maize", "unknown")`
- supports permutation augmentation across repeated single-channel sensors

It is not the preferred workflow for new data. The active default is strict layout.

## 6. Preprocessing Pipeline

### 6.1 Resampling

The active resampling script in this working tree is:

```bash
python emg/resample_raw_dataset.py
```

`emg/resample_raw_dataset.py`:

- reads raw strict `.npz` files from `data_strict`
- builds a shared overlap window across all channels
- linearly interpolates each channel onto one common time grid
- defaults to `TARGET_FS_HZ = 2000.0`
- remaps sample labels to the new grid using nearest-neighbor transfer from channel-0 timestamps
- resamples optional calibration segments too
- writes the same relative file layout under `data_resampled_strict`
- records resampling metadata back into `payload["metadata"]["resampling"]`

Output files remain raw-layout files, still using `X`, `timestamps`, and `y`, but on a common grid.

### 6.2 Filtering

The active filtering script is:

```bash
python emg/filtering.py
```

`emg/filtering.py`:

- reads resampled raw files
- preserves metadata and extra fields
- applies:
  - notch 60 Hz, bandwidth 3
  - notch 120 Hz, bandwidth 3
  - bandpass 20-450 Hz, order 6
- writes filtered files as `*_filtered.npz`
- stores filtered signal in `emg`
- filters calibration arrays too, rewriting:
  - `calib_neutral_emg`
  - `calib_mvc_emg`

The filtered file therefore becomes the training input source.

### 6.3 Recalibration note

Comments in active files still reference `tools/recalibrate.py`, but this working tree does not contain a non-archived recalibration script at that path. The only `recalibrate.py` currently present is under `archive/code_archive/`.

So the active, non-archived preprocessing chain in this repo state is:

1. collect raw strict data
2. `python emg/resample_raw_dataset.py`
3. `python emg/filtering.py`
4. train

## 7. Windowing and Label Generation

Both training scripts turn filtered continuous EMG into windows using `libemg.utils.get_windows`.

Current defaults:

- `WINDOW_SIZE = 200`
- `WINDOW_STEP = 100`

Window labels are assigned by majority vote over the sample-level label segment covered by each window.

Important rules:

- `neutral_buffer` windows are discarded
- low-confidence majority windows can be discarded
- current label-confidence filter:
  - enabled
  - minimum confidence `0.85`

If calibration arrays exist in the filtered file and calibration use is enabled, the scripts first apply:

- per-channel neutral mean subtraction
- per-channel MVC scaling using the configured percentile

Only after that do they window the EMG.

## 8. Bundle Interface and Model-Family Layer

The active shared bundle class is `GestureModelBundle` in `emg/gesture_model_cnn.py`.

Bundle contents:

- `model`
- `mean`
- `std`
- `label_to_index`
- `index_to_label`
- `metadata`

Bundle methods:

- `standardize(X)`
- `predict_proba(X)`
- `embed(X, l2_normalize=False)`

The loader is now generic:

- `load_gesture_bundle(path, device=...)`

Architecture resolution supports:

- `GestureCNN`
- `GestureCNNv2`
- `MetricTCN`

The model-family abstraction lives in `emg/model_family.py`.

Supported model families:

- `cnn_v2`
- `metric_tcn`

Family metadata added to bundles includes:

- `model_family`
- `supports_prototype_calibration`
- `decoder_preference`
- `use_instance_norm_input`

For `metric_tcn`, metadata also stores a `metric_learning` block with embedding size and loss hyperparameters.

## 9. GestureCNNv2 Architecture

`GestureCNNv2` is a 1D residual CNN for time-series EMG windows with input `InstanceNorm1d`, channel attention, and an energy bypass scalar.

### 9.1 Input

- shape: `(batch, channels, time)`
- before normalization, the model computes one raw energy scalar per sample:

  `energy = mean(x^2)` over all channels and time

- this gives shape `(B, 1)` and is saved for the classifier head

### 9.2 Input normalization

- `InstanceNorm1d(in_channels, affine=False, track_running_stats=False)`

This is per-window normalization. It removes absolute amplitude scale inside each window.

### 9.3 Stem

- `Conv1d(in_channels -> 32, kernel_size=11, padding=5, bias=False)`
- `BatchNorm1d(32)`
- `ReLU`

### 9.4 Stage 1

- `ResBlock1d(32, kernel_size=11, dropout=dropout)`
- `ChannelAttention(32)`
- `MaxPool1d(2, 2)`

`ResBlock1d` itself is:

- `Conv1d(32 -> 32, k=11, pad=5, bias=False)`
- `BatchNorm1d(32)`
- `ReLU`
- `Conv1d(32 -> 32, k=11, pad=5, bias=False)`
- `BatchNorm1d(32)`
- residual add with input
- `ReLU`
- `Dropout`

`ChannelAttention` is squeeze-and-excitation style channel gating:

- `AdaptiveAvgPool1d(1)`
- `Linear(channels -> hidden)`
- `ReLU`
- `Linear(hidden -> channels)`
- `Sigmoid`
- elementwise channel reweighting

### 9.5 Stage 2

Projection first:

- `Conv1d(32 -> 64, kernel_size=1, bias=False)`
- `BatchNorm1d(64)`

Then:

- `ResBlock1d(64, kernel_size=11, dropout=dropout)`
- `ChannelAttention(64)`
- `MaxPool1d(2, 2)`

### 9.6 Stage 3

Projection first:

- `Conv1d(64 -> 128, kernel_size=1, bias=False)`
- `BatchNorm1d(128)`

Then:

- `ResBlock1d(128, kernel_size=11, dropout=dropout)`
- `ChannelAttention(128)`
- `AdaptiveAvgPool1d(1)`
- `Flatten`

This produces learned features of shape `(B, 128)`.

### 9.7 Final head

The model concatenates:

- learned features `(B, 128)`
- raw pre-normalization energy `(B, 1)`

Result:

- combined feature `(B, 129)`

Classifier:

- `Linear(129 -> num_classes)`

### 9.8 Key design idea

Because `InstanceNorm1d` removes scale cues, neutral or low-activity windows could otherwise become harder to separate from active gestures. The energy bypass preserves one explicit raw-amplitude summary and injects it straight into the classifier head.

### 9.9 Important practical detail

For `cnn_v2`, external bundle standardization is intentionally bypassed. `GestureModelBundle.standardize(...)` returns the input unchanged when `use_instance_norm_input=True`, so the model sees raw windows and computes its own internal normalization plus energy bypass exactly as designed.

Also note:

- `extract_embedding(...)` on `GestureCNNv2` returns the combined `(128 + energy)` vector, not just the pure CNN feature vector
- so prototype-style decoding on a CNN bundle uses a 129-D representation

## 10. MetricTCN Architecture

`MetricTCN` is a temporal convolutional encoder with a learned embedding head and a classifier head. It is designed for supervised metric learning plus few-shot prototype adaptation.

### 10.1 Input

- shape: `(batch, channels, time)`
- unlike `cnn_v2`, this family expects explicit train-stat z-score standardization outside the model

### 10.2 Basic block: `CausalConv1d`

`CausalConv1d` is a regular `Conv1d` with manual left-padding:

- left padding = `(kernel_size - 1) * dilation`
- no right padding

That makes the convolution causal with respect to time.

### 10.3 Basic block: `TemporalResidualBlock`

Each block contains:

- `CausalConv1d(in_channels -> out_channels, kernel_size, dilation)`
- `BatchNorm1d`
- `ReLU`
- `Dropout`
- `CausalConv1d(out_channels -> out_channels, kernel_size, dilation)`
- `BatchNorm1d`
- residual path:
  - identity if channel count matches
  - otherwise `1x1 Conv1d + BatchNorm1d`
- residual add
- `ReLU`
- `Dropout`

### 10.4 Default network structure

Current default hyperparameters from the trainers:

- channels: `(64, 64, 128, 128)`
- kernel size: `5`
- embedding dim: `128`
- dropout: `0.25`

The encoder stack therefore is:

1. `TemporalResidualBlock(in_channels -> 64, dilation=1)`
2. `TemporalResidualBlock(64 -> 64, dilation=2)`
3. `TemporalResidualBlock(64 -> 128, dilation=4)`
4. `TemporalResidualBlock(128 -> 128, dilation=8)`

Then:

- `AdaptiveAvgPool1d(1)`
- squeeze last dimension
- embedding dropout
- `Linear(128 -> 128)` embedding layer
- `Linear(128 -> num_classes)` classifier head

### 10.5 Public outputs

`MetricTCN` exposes:

- `extract_embedding(x, l2_normalize=False)`
- `forward_with_embedding(x, l2_normalize_embedding=False)`
- `forward(x)`

So the same model can be used as:

- a direct classifier via logits
- an embedding encoder for prototype-based few-shot decoding

### 10.6 Plain-English interpretation

This is not just "replace the CNN with a TCN."

The point is:

- the TCN learns a representation of EMG windows in an embedding space
- that space is trained so same-gesture windows are closer together
- at runtime, a few short labeled calibration reps from the current user can define class prototypes in that space

That is why `metric_tcn` is the family that advertises:

- `supports_prototype_calibration = True`
- `decoder_preference = "prototype"`

## 11. Metric Learning and Few-Shot Adaptation

### 11.1 Training loss

`emg/metric_losses.py` defines `SupervisedContrastiveLoss`.

For `metric_tcn`, training uses:

`total_loss = cross_entropy(logits, y) + lambda * supervised_contrastive(embeddings, y)`

Current defaults:

- `lambda = 0.20`
- temperature `tau = 0.10`

The loss implementation:

- L2-normalizes embeddings
- builds batchwise similarity logits from embedding dot products
- treats same-label samples in the batch as positives
- excludes self-pairs
- averages log-probability over positive pairs

### 11.2 Why both CE and SupCon are used

Cross-entropy keeps the model usable as a normal classifier. Supervised contrastive loss shapes the embedding geometry so prototype decoding has a cleaner class structure at runtime.

### 11.3 Prototype classifier

`emg/prototype_classifier.py` implements `PrototypeClassifier`.

It:

- groups support embeddings by class
- averages them into class centroids
- optionally L2-normalizes centroids
- scores a query embedding by cosine similarity to each centroid
- converts those scores to probabilities with a temperature-scaled softmax

Current default prototype settings:

- temperature `0.20`
- L2 normalization enabled

### 11.4 What "few-shot" means in this repo

Few-shot does not mean creating new gesture classes at runtime.

It means:

- train on the existing gesture label set
- collect a small labeled support set from the current user for those same gestures
- build prototypes from that support set
- classify live windows by nearest prototype

In the current realtime code, one prompted gesture-calibration recording per gesture is collected, then internally windowed into many support windows.

## 12. Per-Subject Training Pipeline

`train_per_subject.py` trains one arm for one subject.

### 12.1 Current checked-in defaults

At the moment this file is checked in with:

- `ARM = "left"`
- `TARGET_SUBJECT = "Matthew"`
- strict layout mode
- included gestures = `{"neutral", "left_turn", "right_turn", "horn"}`
- `MODEL_FAMILY = "metric_tcn"`
- `MODEL_OUT = models/strict/per_subject/<arm>/<subject>_tcn_4_gestures.pt`

Core training hyperparameters:

- batch size `512`
- epochs `40`
- learning rate `1e-4`
- dropout `0.25`
- label smoothing `0.05`

### 12.2 Split strategy

Per-subject evaluation is now grouped cross-validation over session files, not a single random holdout.

The splitter is:

- `StratifiedGroupKFold`

Grouping unit:

- session file path

This means windows from the same session do not appear in both train and test for a fold.

### 12.3 Final fit policy

After grouped CV:

- each fold records its best epoch
- the final full-dataset fit uses the median best epoch across folds

So reported metrics come from out-of-fold evaluation, while the exported model is trained on all subject windows.

### 12.4 Per-subject augmentation

Per-subject augmentation runs on GPU tensors and includes:

- amplitude scaling
- additive Gaussian noise
- temporal shift
- single-channel dropout
- temporal stretch

If salvage layout mode is active, per-subject augmentation can also apply permutation augmentation across exchangeable single-channel sensor blocks.

### 12.5 Exported metadata

The saved bundle stores:

- model state
- normalization mean/std
- label maps
- architecture description
- family metadata
- grouped CV metrics
- confusion matrices
- channel layout metadata
- final-fit metadata

## 13. Cross-Subject Training Pipeline

`train_cross_subject.py` trains one arm using pooled subjects.

### 13.1 Current checked-in defaults

At the moment this file is checked in with:

- `ARM = "right"`
- strict layout mode
- included gestures = `{"neutral", "left_turn", "right_turn", "horn"}`
- `MODEL_FAMILY = "metric_tcn"`
- `LOSO_EVAL = True`
- `CALIBRATED_LOSO_EVAL = True`
- `TRAIN_FINAL_MODEL = True`

Core training hyperparameters:

- batch size `512`
- epochs `40`
- learning rate `1e-4`
- dropout `0.25`
- label smoothing `0.05`

### 13.2 Subject-balanced sampling

Cross-subject training uses a `WeightedRandomSampler` so subjects with more windows do not dominate the batch stream. Each subject gets sample weights inversely proportional to its window count.

### 13.3 Cross-subject augmentation

Cross-subject augmentation uses the same general GPU-native family as per-subject training, but with a wider amplitude range:

- amplitude scale range `(0.5, 2.0)`

That is meant to reflect stronger inter-subject amplitude variation.

### 13.4 Zero-shot LOSO

If `LOSO_EVAL = True`, the script first performs leave-one-subject-out evaluation:

- one subject is held out entirely
- the model trains on all remaining subjects
- evaluation runs on the held-out subject

The script prints per-subject classification reports and builds a summary with:

- mean LOSO accuracy
- std, min, max

### 13.5 Calibrated LOSO

If `CALIBRATED_LOSO_EVAL = True` and the model family supports prototype calibration, the script also runs calibrated LOSO.

Current calibrated LOSO design:

- hold out one subject
- train the base model on all other subjects
- inside the held-out subject, use session files as support/query split units
- each session file is tried as the support set
- the remaining held-out sessions become the query set
- support must contain all active classes or that split is skipped
- support windows are embedded and converted into prototypes
- query windows are classified by prototype similarity

This is the current code path for few-shot cross-subject evaluation.

### 13.6 Final exported cross-subject bundle

After evaluation, if `TRAIN_FINAL_MODEL = True`, the script still trains and saves one pooled bundle for deployment.

That exported bundle includes:

- in-distribution group-file test metrics from a `GroupShuffleSplit`
- family metadata
- channel layout metadata
- an `evaluation` metadata block containing LOSO summaries

So the saved model contains both deployment weights and separate cross-subject evaluation summaries.

## 14. Realtime Inference Pipeline

The active realtime entrypoint is:

```bash
python realtime_gesture_cnn.py --model <bundle.pt>
```

Dual-arm mode uses:

```bash
python realtime_gesture_cnn.py --model-right <right.pt> --model-left <left.pt>
```

### 14.1 Current checked-in realtime defaults

`realtime_gesture_cnn.py` is currently checked in with:

- `MODE = "dual"`
- included gestures = `{"neutral", "left_turn", "right_turn", "horn"}`
- `CALIBRATE = True`
- `GESTURE_CALIB = False`
- `USE_PROTOTYPE_CLASSIFIER = False`
- `REALTIME_FILTER_MODE = "scipy_stateful"`
- `REALTIME_RESAMPLE = True`
- `REALTIME_TARGET_FS_HZ = 2000.0`

Default bundle paths currently point at:

- `models/strict/cross_subject/right/v6_4_gestures.pt`
- `models/strict/cross_subject/left/v6_4_gestures.pt`

### 14.2 Filtering and resampling

Realtime applies a causal approximation of the offline filter stack:

- notch 60 Hz
- notch 120 Hz
- bandpass 20-450 Hz

Realtime also supports timestamp-based resampling through `_RealtimeTimestampResampler`, which accumulates per-channel timestamps and interpolates them onto a common fixed-rate grid. It uses:

- `REALTIME_TARGET_FS_HZ` if set
- otherwise bundle metadata fallback if available

In the current trainers, target sampling rate is not explicitly exported into bundle metadata, so the checked-in realtime path is mainly controlled by `REALTIME_TARGET_FS_HZ = 2000.0`.

### 14.3 Strict-layout enforcement

If the loaded bundle advertises strict layout metadata:

- realtime validates the live channel labels against the strict pair contract
- derives ordered indices from actual pair identity
- fails closed on mismatch

That applies to both single-arm and dual-arm runs.

### 14.4 Calibration stages

Realtime calibration has two different meanings:

1. neutral/MVC calibration
   - signal normalization
2. gesture calibration
   - user/session-specific class adaptation

Neutral/MVC calibration is controlled by `CALIBRATE`.

Per-gesture calibration collects:

- one prompted recording per active gesture
- default duration `GESTURE_CALIB_S = 5.0 s`
- prep countdown `GESTURE_CALIB_PREP_S = 2.0 s`

For the current 4-gesture setup, that means:

- `neutral`
- `left_turn`
- `right_turn`
- `horn`

one prompt each.

### 14.5 Automatic prototype behavior for `metric_tcn`

Realtime now reads bundle metadata:

- `model_family`
- `supports_prototype_calibration`
- `decoder_preference`

If a bundle prefers prototype decoding:

- per-gesture calibration is auto-enabled even if `GESTURE_CALIB = False`
- prototype decoding is auto-enabled even if `USE_PROTOTYPE_CLASSIFIER = False`

The only required gate is still:

- `CALIBRATE = True`

So `metric_tcn` bundles automatically trigger the few-shot gesture-calibration path. Old CNN bundles do not.

### 14.6 Adaptation behavior after gesture calibration

There are two possible post-calibration paths:

- softmax-head quick fine-tuning
- prototype calibration

Softmax quick adaptation uses `quick_finetune(...)`:

- freezes all parameters except the final head
- trains only that head on the calibration windows

Prototype calibration:

- standardizes calibration windows with the bundle
- embeds them
- fits a `PrototypeClassifier`
- classifies live windows by prototype similarity

For `metric_tcn`, prototype decoding is the intended default.

### 14.7 Confidence gating, smoothing, and hysteresis

Realtime includes three layers of output stabilization:

1. probability smoothing
   - moving average over the last `SMOOTHING` probability vectors
2. reject gates
   - softmax reject gate
   - prototype reject gate
   - both can demote ambiguous outputs to `neutral`
3. output hysteresis
   - optional label latching to suppress one-frame flips

Dual-arm fusion then combines left and right arm outputs.

### 14.8 Dual-arm published output contract

`realtime_gesture_cnn.py` exposes:

- `get_latest_dual_state()`
- `get_latest_published_gestures()`
- legacy `get_latest_gesture()`

Published semantics are:

- if both arms agree, output mode is `single` with arm `dual`
- if arms differ, output mode is `split` with separate left and right labels

This is the interface CARLA reads.

## 15. Runtime Tuning

`emg/runtime_tuning.py` is the active tuning source of truth.

Current realtime tuning:

- smoothing `1`
- min confidence `0.80`
- dual-arm agree threshold `0.55`
- dual-arm single threshold inherits min confidence
- output hysteresis `False`
- softmax reject enabled `False`
- softmax reject min confidence `0.55`
- softmax reject min margin `0.08`
- prototype reject min confidence `0.55`
- prototype reject min margin `0.08`

Current CARLA tuning:

- gesture max age `0.75 s`
- active steer dwell frames `1`
- neutral steer dwell frames `1`
- steer left `-0.08`
- steer right `0.08`
- steer left strong `-0.30`
- steer right strong `0.30`
- steer neutral `0.0`

The runtime tuning label currently written into logs is:

- `RUNTIME_TUNING_NAME = "manual"`

## 16. CARLA Integration

The active CARLA entrypoint is:

```bash
python carla_integration/manual_control_emg.py
```

Named launch wrappers currently in `carla_integration/` include:

- `test_run_manual_control_emg_0_9_16.bat`
- `test_start_carla_server_0_9_16.bat`
- `test_run_lane_keep_5min_emg_0_9_16.cmd`
- `test_run_highway_overtake_emg_0_9_16.cmd`

### 16.1 Control policy

Current control policy:

- `CONTROL_POLICY_NAME = "split_strength_latched_aux_v1"`

Under that policy:

- left arm `left_turn` -> `left_strong`
- left arm `right_turn` -> `right_strong`
- right arm `left_turn` -> `left`
- right arm `right_turn` -> `right`
- otherwise -> `neutral`

So the left arm is treated as the stronger steering side and the right arm as the lighter steering side.

### 16.2 Turn signals and reverse

Current CARLA-side behavior:

- strong steering states drive turn-signal state while held
  - `left_strong` -> left signal on
  - `right_strong` -> right signal on
  - otherwise signal off
- reverse is toggled only when both arms publish `horn` at the same time
- horn is not currently mapped to a separate vehicle actuator in the active control loop

Reverse toggle protections:

- cooldown `1.0 s`
- max speed for toggle `0.75 m/s`

### 16.3 Realtime launch ownership

`manual_control_emg.py` imports and launches `realtime_gesture_cnn.py` internally. For normal CARLA testing, realtime should not be started separately unless you are explicitly debugging the interface boundary.

### 16.4 Named scenarios

`carla_integration/scenario_presets.py` defines the active named scenarios.

`lane_keep_5min`:

- kind: `lane_keep`
- map: `Town04_Opt`
- start checkpoint index: `3`
- checkpoint spacing: `35.0 m`
- route length: `2600.0 m`
- timeout: `None`

`highway_overtake`:

- kind: `overtake`
- map: `Town04_Opt`
- checkpoint spacing: `25.0 m`
- route length: `420.0 m`
- timeout: `None`
- lead spawn distance: `25.0 m`
- `lead_hold_until_start = True`
- lead speed reduction: `60%`

If a named scenario is requested, the scenario preset overrides the manual `--map` argument to keep map selection consistent with the preset.

### 16.5 CARLA logging

`manual_control_emg.py` can emit per-tick drive logs and forward realtime prediction logs.

Preferred switch:

```bash
--eval-log-dir <dir>
```

This auto-creates:

- `carla_drive_<timestamp>.csv`
- `realtime_predictions_<timestamp>.csv`

Drive logs include:

- prediction sequence and timestamps
- published gesture mode and labels
- steer key and applied steer key
- signal state
- reverse state
- lane error
- lane invasion events
- scenario state and progress

## 17. Evaluation Toolchain

Active evaluation scripts under `eval_metrics/` include:

- `harvest_model_metrics.py`
- `realtime_behavior_metrics.py`
- `analyze_latency.py`
- `analyze_drive_metrics.py`
- `build_eval_tables.py`
- `compare_realtime_runs.py`
- `diagnose_session_recall.py`
- `plot_filter_effect.py`

### 17.1 Offline bundle harvesting

`harvest_model_metrics.py` scans saved bundles and extracts stored metadata and offline metrics into CSV and JSON tables.

### 17.2 Realtime behavior metrics

`realtime_behavior_metrics.py` summarizes prompted realtime runs into:

- overall accuracy
- balanced accuracy
- time to first correct prediction
- time to stable prediction
- label flip rate
- stale carryover rate
- per-label breakdowns

### 17.3 Latency analysis

`analyze_latency.py` joins realtime and CARLA logs on `prediction_seq` and computes:

- classifier latency
- publish latency
- control latency
- end-to-end latency

### 17.4 Drive metrics

`analyze_drive_metrics.py` summarizes CARLA drive logs into:

- lane error mean
- lane error RMSE
- lane invasions
- completion time
- steering smoothness
- command success rate when available
- scenario success/failure fields

### 17.5 Table builder

`build_eval_tables.py` joins:

- harvested offline model metrics
- realtime behavior JSON
- latency JSON
- drive metrics JSON

using a manifest CSV, then writes:

- master evaluation table
- report-specific tables
- paper-specific tables

### 17.6 Diagnostics

Additional active diagnostics:

- `compare_realtime_runs.py`
  - compares two prompted realtime runs and reports deltas
- `diagnose_session_recall.py`
  - finds weak per-session recall collapse for selected labels
- `plot_filter_effect.py`
  - plots raw vs filtered EMG and PSD for one session/channel

## 18. Current Repo-State Notes

These are not archived-path notes; they are current working-tree facts that matter when reading or running the code.

1. The active resampling script is `emg/resample_raw_dataset.py`, even though some comments still mention `tools/resample_raw_dataset.py`.
2. Active comments also still mention `tools/recalibrate.py`, but there is no non-archived recalibration utility at that path in this working tree.
3. The checked-in trainer defaults currently point at `metric_tcn`, not `cnn_v2`.
4. `train_cross_subject.py` still uses the default output filename `v6_4_gestures.pt`, even though the checked-in model family is `metric_tcn`.
5. The checked-in realtime defaults currently point at 4-gesture cross-subject bundle paths and dual-arm mode.
6. `metric_tcn` few-shot behavior is now metadata-driven in realtime; it is not just a manual flag.

## 19. Practical Summary

If you describe the active system in one paragraph:

This repo now uses a strict-layout EMG pipeline where the GUI collects sample-level labeled raw recordings with channel labels, resampling aligns mixed sensor timestamps onto a shared 2000 Hz grid, filtering produces the trainable EMG stream plus filtered calibration arrays, training exports either a residual CNN (`GestureCNNv2`) or a supervised metric-learning TCN (`MetricTCN`) as a shared gesture bundle, realtime enforces strict sensor identity and can automatically run few-shot gesture calibration for `metric_tcn` bundles, and CARLA consumes the published split-or-single dual-arm gesture state to drive steering, signals, reverse, and evaluation logging.
