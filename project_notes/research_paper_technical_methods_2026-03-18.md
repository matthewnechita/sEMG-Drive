# Research Paper Technical Methods Note

Date: 2026-03-18
Branch reference: `v5-flicker-tuning`

## Purpose

This document is the paper-facing technical note for the current repo state. It is narrower than the final-report reference and is designed to answer two questions:

1. What technical details are solid enough to include in a research paper right now?
2. What should probably stay out of the paper unless matching experiments and results are actually reported?

The goal is to help scope the paper methods section around the active, defensible pipeline rather than the full engineering surface area of the repo.

## 1. Recommended Paper Scope

The cleanest current paper scope is:

- strict-layout EMG gesture classification
- Delsys-based data collection
- resampling and filtering pipeline
- CNN architecture and offline training
- prompted realtime evaluation
- optional CARLA demonstration or systems section only if matching logs/results are included

Recommended primary task scope:

- strict-layout 3-gesture setting
  - `neutral`
  - `left_turn`
  - `right_turn`

Reason:

- the current realtime defaults are already centered on this subset
- the active runtime tuning work is focused on neutral flicker within this subset
- it is the narrowest and most defensible story if the paper is still taking shape

If you already have strong 5- or 6-gesture results, those can be added, but the 3-gesture strict pipeline is the best default paper core.

## 2. Recommended Paper Story

A strong paper narrative for the current repo is:

1. Collect strict-layout EMG data with consistent sensor placement.
2. Resample heterogeneous multi-sensor data to a shared sample grid.
3. Filter the EMG stream with fixed notch and bandpass filters.
4. Train a residual 1D CNN with channel attention and an explicit energy bypass for neutral-vs-active separation.
5. Evaluate offline classification accuracy and realtime stability.
6. Optionally demonstrate closed-loop control in CARLA.

This story is specific, technically coherent, and matches the actual code.

## 3. Recommended Methods Content

### 3.1 Data acquisition

Safe to include:

- Delsys Trigno-based EMG acquisition through a custom Python GUI
- strict sensor placement policy using fixed pair-number identity
- per-session metadata logging including subject, arm, session, protocol name, gesture schedule, and EMG channel labels
- calibration segments for neutral rest and maximum voluntary contraction (MVC)

Useful collection details to state:

- standard protocol gestures:
  - `left_turn`
  - `right_turn`
  - `neutral`
  - `signal_left`
  - `signal_right`
  - `horn`
- standard protocol durations:
  - `5 s` gesture
  - `5 s` neutral
  - `5` repetitions
- calibration:
  - `5 s` neutral rest
  - `5 s` MVC
  - `2 s` preparation before MVC

If the paper only reports 3-gesture experiments, say clearly that training/evaluation used a subset of the collected labels rather than implying the collection protocol itself only used three labels.

### 3.2 Strict sensor-placement policy

This is paper-worthy because it is central to the current method.

Safe to include:

- right arm strict slots:
  - pairs `1`, `2`, `3` as Avanti
  - pair `7` as Maize
  - pair `9` as Galileo
  - pair `11` as Mini
- left arm strict slots:
  - pairs `4`, `5`, `6` as Avanti
  - pair `8` as Maize
  - pair `10` as Galileo
- strict channel counts:
  - right arm `17`
  - left arm `16`

Suggested wording:

> Channels were reordered at training and inference time according to fixed pair identity and sensor type. Sessions that did not satisfy the required strict-layout contract were rejected rather than silently remapped.

### 3.3 Preprocessing

Safe to include:

- per-channel timestamp resampling onto a shared common grid
- target sampling frequency: `2000 Hz`
- label remapping to the new time grid by nearest-neighbor transfer
- filtering stack:
  - notch `60 Hz`
  - notch `120 Hz`
  - bandpass `20–450 Hz`, order `6`
- optional retroactive recalibration for failed MVC sessions using labeled session statistics

Suggested wording:

> Because heterogeneous Delsys sensors can produce different effective per-channel sample rates, each raw recording was resampled onto a common 2000 Hz time grid using per-channel linear interpolation over the shared valid overlap window. The resulting aligned EMG stream was then filtered with 60 Hz and 120 Hz notch filters followed by a 20–450 Hz sixth-order bandpass filter.

### 3.4 Windowing and label generation

Safe to include:

- window size: `200` samples
- step size: `100` samples
- at `2000 Hz`, this corresponds to:
  - `100 ms` windows
  - `50 ms` stride
- window labels assigned by majority label within each window
- `neutral_buffer` excluded from training
- low-confidence windows dropped using per-window label-confidence thresholds

Suggested wording:

> The filtered EMG stream was segmented into overlapping 100 ms windows with 50 ms stride. Each window received the majority label of its constituent samples, and inter-gesture rest (`neutral_buffer`) was excluded from model training.

### 3.5 Model architecture

This should definitely be in the paper if the paper is about the algorithm.

Safe to include:

- architecture name: `GestureCNNv2`
- input: EMG windows shaped `(channels, time)`
- input `InstanceNorm1d`
- residual 1D CNN backbone
- three feature stages with channel sizes `32`, `64`, `128`
- squeeze-and-excitation channel attention in each stage
- global pooling and linear head
- explicit pre-normalization energy scalar concatenated to the classifier head

Suggested wording:

> The classifier was a residual 1D CNN with channel attention. To improve separation between rest-like and active windows, a scalar energy feature computed before input normalization was concatenated to the learned embedding before the final linear classifier.

### 3.6 Training configuration

Safe to include, but distinguish per-subject and cross-subject settings.

Per-subject defaults:

- epochs: `60`
- batch size: `512`
- optimizer: Adam
- learning rate: `1e-4`
- dropout: `0.25`
- label smoothing: `0.05`
- augmentation enabled
- calibration percentile: `95`
- minimum label confidence: `0.85`

Cross-subject defaults:

- epochs: `80`
- batch size: `512`
- optimizer: Adam
- learning rate: `1e-4`
- dropout: `0.25`
- label smoothing: `0.05`
- augmentation enabled
- subject-balanced sampling via `WeightedRandomSampler`
- minimum label confidence: `0.75`

Current augmentation operations:

- amplitude scaling
- additive Gaussian noise
- temporal shift
- channel dropout
- temporal stretch

### 3.7 Evaluation metrics

Safe offline metrics:

- accuracy
- balanced accuracy
- macro precision / recall / F1
- weighted precision / recall / F1
- worst-class recall
- confusion-to-neutral rate
- neutral prediction false-positive rate

Recommended primary offline metrics for the paper:

- balanced accuracy
- macro F1

Reason:

- they are already computed and stored
- they are more defensible than accuracy alone when class balance varies

### 3.8 Realtime evaluation

The paper can include realtime evaluation if you have prompted-run logs and summaries.

The current repo supports:

- per-prediction logging from `realtime_gesture_cnn.py`
- behavior summaries from `eval_metrics/realtime_behavior_metrics.py`

Safe realtime metrics:

- time to first correct prediction
- time to stable prediction
- label flip rate
- carryover stale rate
- prompt-conditioned balanced accuracy

This is especially relevant if the paper emphasizes control stability or neutral flicker reduction.

## 4. Current Realtime Method Details

If the paper includes a realtime method section, the active defaults on this branch are:

- dual-arm realtime mode
- active gesture subset:
  - `neutral`
  - `left_turn`
  - `right_turn`
- current runtime preset:
  - `flicker_mild_margin`
- smoothing:
  - `3`
- minimum confidence:
  - `0.80`
- hysteresis:
  - enabled
- softmax ambiguity rejection:
  - enabled

These settings are part of the deployed inference behavior on the current branch. Include them only if the reported experiments actually used this preset.

## 5. Current CARLA Method Details

CARLA should be framed carefully in the paper.

Recommended use:

- as a system demonstration or downstream-control evaluation section

Safe CARLA details to include if you have matching runs/logs:

- the simulator client launches realtime inference internally
- gesture outputs are converted into steering, signal, and reverse actions
- throttle and brake remain manual or wheel-controlled
- both arms must simultaneously emit `horn` to request reverse
- low-graphics client mode is supported for stable testing
- the client can log per-tick drive state and control outputs

If you do not have robust CARLA quantitative results yet, keep CARLA to a short systems demo paragraph rather than making it the center of the paper.

## 6. What To Keep Out Unless You Have Matching Results

Do not center the paper on these unless you have actual experiments to support them:

- prototype-classifier realtime mode
  - exists in code but is disabled by default
- salvage-layout training path
  - present in per-subject training but not the preferred current path
- archived SVM / handcrafted-feature pipeline
  - not the active method
- claims about 5- or 6-gesture superiority
  - unless those experiments are explicitly rerun and tabulated
- claims about cross-subject deployment readiness
  - unless LOSO or equivalent generalization metrics are reported

## 7. Recommended Figures And Tables

If you need the paper structure now, these are the most defensible artifacts to include.

### 7.1 Figures

- system pipeline diagram:
  - collection -> resampling -> filtering -> windowing -> CNN -> realtime post-processing -> CARLA
- strict sensor layout diagram:
  - right and left arm pair mapping
- example raw vs filtered EMG / PSD figure:
  - can be built from `eval_metrics/plot_filter_effect.py`
- confusion matrix figure for the main offline model
- optional realtime stability figure:
  - prompted label vs published label over time

### 7.2 Tables

- offline performance table:
  - balanced accuracy
  - macro F1
  - worst-class recall
  - neutral FP rate
- optional realtime table:
  - balanced accuracy
  - time to first correct prediction
  - time to stable prediction
  - label flip fraction
- optional latency table:
  - classifier latency
  - publish latency
  - control latency
  - end-to-end latency

The current `eval_metrics/build_eval_tables.py` already anticipates separate output tables for a capstone final report and a research paper.

## 8. Recommended Paper Configuration Right Now

If you need a single concrete methods configuration to anchor the paper draft, use this:

- strict-layout pipeline only
- right-arm per-subject and cross-subject results as the main offline comparison
- 3-gesture subset:
  - `neutral`
  - `left_turn`
  - `right_turn`
- preprocessing:
  - 2000 Hz resampling
  - 60/120 Hz notch
  - 20–450 Hz bandpass
- windows:
  - 200 samples
  - 100-sample step
- model:
  - `GestureCNNv2`
- realtime post-processing:
  - `flicker_mild_margin`

This is the narrowest paper configuration that matches the current branch cleanly.

## 9. Suggested Methods Paragraph Skeleton

You can adapt the following structure directly into the paper:

1. Data acquisition:
   - Delsys hardware, strict placement, protocol durations, calibration
2. Preprocessing:
   - resampling, filtering, calibration normalization, windowing
3. Model:
   - residual 1D CNN with attention and energy bypass
4. Training:
   - optimizer, epochs, augmentation, split policy
5. Realtime post-processing:
   - confidence thresholds, smoothing, hysteresis, ambiguity rejection
6. Evaluation:
   - offline metrics, prompted realtime behavior metrics, optional CARLA metrics

## 10. Recommended Writing Discipline

When drafting the paper:

- describe only the path used for the reported experiments
- explicitly state when a 3-gesture subset is used
- keep the strict-layout policy visible, because it is central to reproducibility
- separate offline classification results from realtime behavior results
- treat CARLA as downstream validation, not as the definition of model quality

If you want, the next step can be turning this methods note into paper-ready prose sections rather than keeping it as a technical reference.
