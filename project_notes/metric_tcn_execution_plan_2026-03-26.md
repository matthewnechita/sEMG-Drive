# Metric TCN Execution Plan

## Goal

Add a switchable `metric_tcn` model family that can be trained from the existing strict-layout EMG pipeline and used by `realtime_gesture_cnn.py` with prototype-based few-shot gesture calibration.

## Scope

- Keep the current `cnn_v2` workflow working unchanged by default.
- Add one new model family:
  - `MODEL_FAMILY = "metric_tcn"`
- Use the existing gesture label set only.
- Reuse the current realtime calibration flow:
  - neutral calibration
  - MVC calibration
  - per-gesture calibration reps
  - prototype decoding for live inference

## Implementation Phases

### 1. Shared model/bundle layer

- Extend the current bundle loader so it can load more than CNN bundles.
- Add metadata fields needed by realtime:
  - `model_family`
  - `supports_prototype_calibration`
  - `decoder_preference`
  - normalization mode details
- Keep backward compatibility for existing `cnn_v2` bundles.

### 2. New model/loss modules

- Add `emg/gesture_model_metric_tcn.py`
  - residual dilated TCN encoder
  - embedding output
  - classifier head
  - `extract_embedding(...)`
  - `forward_with_embedding(...)`
- Add `emg/metric_losses.py`
  - supervised contrastive loss over batch embeddings

### 3. Trainer integration

- Add `MODEL_FAMILY` to:
  - `train_per_subject.py`
  - `train_cross_subject.py`
- Make model-family-specific pieces switchable:
  - model construction
  - input normalization
  - training loss
  - architecture metadata
  - bundle metadata
- Keep shared pieces unchanged:
  - strict layout
  - windowing
  - calibration normalization
  - grouped CV
  - zero-shot LOSO

### 4. Cross-subject calibrated LOSO

- Keep existing zero-shot LOSO.
- Add optional calibrated LOSO for prototype-based few-shot evaluation.
- Use held-out session files as support/query split units to avoid overlap leakage.
- For each held-out subject:
  - train on all other subjects
  - choose one held-out session as support if it covers all trained labels
  - build prototypes from support windows
  - evaluate on the remaining held-out sessions

### 5. Realtime integration

- Replace CNN-only bundle loading with generic gesture-bundle loading.
- Read bundle metadata to decide whether prototype decoding should be preferred.
- For `metric_tcn`, gesture calibration should default to prototype building before live inference.
- Keep softmax decoding available as a fallback.

## Verification

- Syntax-check the new modules and edited entrypoints.
- Confirm existing `cnn_v2` training scripts still import and run.
- Confirm `realtime_gesture_cnn.py` still loads old CNN bundles.
- Confirm a `metric_tcn` bundle can be loaded, calibrated, embedded, and decoded through the prototype path.
