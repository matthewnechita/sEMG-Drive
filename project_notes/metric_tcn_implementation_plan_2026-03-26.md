# MetricTCN Implementation Plan

## Goal

Add one new switchable model family for EMG gesture recognition that can be compared directly against the current CNN path without replacing it.

Target model stack:

- TCN encoder
- supervised contrastive training
- prototype-based few-shot calibration before realtime inference

The current CNN workflow remains intact and selectable.

## High-Level Design

Use a shared training pipeline with a model-family switch:

```python
MODEL_FAMILY = "cnn_v2"      # or "metric_tcn"
```

This switch should exist in:

- `train_cross_subject.py`
- `train_per_subject.py`

Shared between model families:

- strict-layout loading
- windowing
- label handling
- train/eval split logic
- metrics reporting
- bundle save/load shape

Model-family-specific behavior:

- model construction
- loss function
- bundle metadata
- preferred inference decoder

## New Components

### 1. Model file

Add:

- `emg/gesture_model_metric_tcn.py`

Contents:

- small causal or dilated TCN encoder
- embedding output, recommended `embedding_dim = 128`
- classifier head for supervised training
- `extract_embedding(...)` method for prototype calibration

### 2. Loss file

Add:

- `emg/metric_losses.py`

Contents:

- supervised contrastive loss

Recommended training loss for `metric_tcn`:

```python
loss = cross_entropy_loss + lambda_supcon * supervised_contrastive_loss
```

## Trainer Changes

### `train_cross_subject.py`

Add config switches:

```python
MODEL_FAMILY = "cnn_v2"          # or "metric_tcn"
LOSO_EVAL = True
CALIBRATED_LOSO_EVAL = True
TRAIN_FINAL_MODEL = True
```

Rules:

- `LOSO_EVAL` controls zero-shot LOSO
- `CALIBRATED_LOSO_EVAL` controls few-shot calibrated LOSO
- `CALIBRATED_LOSO_EVAL` requires `LOSO_EVAL = True`
- `TRAIN_FINAL_MODEL` controls final pooled model fit after evaluation

Cross-subject evaluation outputs should be split into:

- `zero_shot_loso`
- `calibrated_loso`
- final pooled model fit metadata

### `train_per_subject.py`

Add:

```python
MODEL_FAMILY = "cnn_v2"          # or "metric_tcn"
```

Keep grouped cross-validation over session files as the default evaluation path.

Per-subject training does not need LOSO, but it should be able to train either:

- current CNN
- new `metric_tcn`

## Runtime / Calibration Behavior

Calibration means:

1. run neutral calibration
2. run MVC calibration
3. collect short labeled gesture reps immediately before realtime inference
4. build class prototypes from those reps
5. classify live windows by nearest prototype in embedding space

The existing runtime already has the right structure:

- gesture-calibration collection in `realtime_gesture_cnn.py`
- prototype decoder in `emg/prototype_classifier.py`

For `metric_tcn`, prototype decoding should be the preferred decoder after calibration.
Softmax should remain available as a fallback only.

## Offline Evaluation Protocol

### Zero-shot LOSO

Standard leave-one-subject-out:

- train on all but one subject
- test on the held-out subject
- no target-subject gesture calibration

### Calibrated LOSO

This should simulate the real deployment flow.

For each held-out subject:

1. train the base model on all other subjects
2. choose one held-out session or repetition block as calibration support
3. build prototypes from that support set
4. evaluate on different held-out data from that same subject

Important constraint:

- support and query must be separated by session or repetition
- do not split overlapping windows from the same recording into both support and query

Report at minimum:

- accuracy
- balanced accuracy
- macro F1
- per-class recall
- confusion into `neutral`

## Bundle Requirements

Save the new family in the same general bundle style as the current CNN path, with extra metadata such as:

- `model_family = "metric_tcn"`
- `supports_prototype_calibration = True`
- `decoder_preference = "prototype"`

Use separate filenames so comparisons are easy:

- `..._cnn_v2.pt`
- `..._metric_tcn.pt`

## Minimal Implementation Order

1. Add `emg/gesture_model_metric_tcn.py`
2. Add `emg/metric_losses.py`
3. Add `MODEL_FAMILY` switch to both trainers
4. Add save/load support for `metric_tcn`
5. Reuse existing runtime prototype calibration path
6. Add calibrated LOSO reporting in `train_cross_subject.py`
7. Compare `cnn_v2` vs `metric_tcn`

## Scope Guardrails

Do not include self-supervised pretraining in v1.

Rationale:

- it adds a second training stage
- it increases tuning risk
- it makes attribution of gains harder

The first implementation should stay focused on:

- one new backbone
- one new loss family
- one calibration decoder

## Success Criteria

The new path is successful if it provides:

- a clean one-line switch between `cnn_v2` and `metric_tcn`
- reproducible zero-shot LOSO metrics
- reproducible calibrated LOSO metrics
- runtime compatibility with the existing calibration flow
- no regression to the current CNN training path
