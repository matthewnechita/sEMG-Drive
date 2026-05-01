# EMG Driving Control Capstone

This repository contains the maintained codebase for a capstone project on EMG-driven vehicle control. The project uses Delsys surface EMG sensors, CNN-based gesture recognition, and CARLA driving scenarios to test whether upper-limb muscle activity can serve as a practical control interface for simple driving tasks.

The repo is organized around the full system. It covers data collection, preprocessing, training, live inference, CARLA control, and evaluation.

## Why This Project Was Built

Traditional driving controls assume that the user can reliably operate a steering wheel, pedals, and other physical interfaces. This project explores a different control pathway: using electromyography (EMG) from the forearms to recognize intentional gestures and convert them into vehicle commands in simulation.

The goal is to evaluate whether a small, well-defined EMG gesture set can support usable vehicle control in a constrained driving environment, and to measure the limits of that approach with both model metrics and driving metrics.

## What The System Does

The maintained project state uses a fixed 4-gesture contract:

- `left_turn`
- `right_turn`
- `neutral`
- `horn`

In the current CARLA client, `horn` is repurposed as the reverse request when both arms publish it together.

The active stack is:

- EMG collection with fixed sensor placement
- resampling and filtering
- CNN training with `GestureCNNv2`
- dual-arm realtime inference
- CARLA free-roam and scenario evaluation
- offline, latency, and driving-metric analysis

## How The Pipeline Works

### 1. Collection

Raw EMG data is collected through the Delsys GUI entrypoint:

```powershell
python DelsysPythonGUI.py
```

Collection is done with fixed sensor placement. That means the numbered sensors are expected to stay on the same arm and in the same physical locations across sessions. This matters because the maintained training and realtime code assume consistent channel identity, not just similar signal shape.

In plain terms, the system works best when the sensor setup is repeated the same way each time. If the sensors move around between sessions, the channel patterns the model sees can change enough to hurt both training consistency and live control performance.

Raw collections for the maintained workflow are stored under:

- `data_strict/<arm> arm/<subject>/raw/*.npz`

### 2. Preprocessing

After collection, the data is processed in two main stages:

```powershell
python emg/resample_raw_dataset.py
python emg/filtering.py
```

Resampling places each channel onto a consistent time grid. Filtering applies the maintained EMG signal-processing chain while preserving the metadata needed by the training and sensor-layout logic.

The maintained training root is:

- `data_resampled_strict/<arm> arm/<subject>/{raw,filtered}`

### 3. Training

The repo keeps two active training entrypoints:

- `train_per_subject.py`
- `train_cross_subject.py`

Both train `GestureCNNv2` models on filtered EMG windows shaped as `channels x time`. The output bundles store:

- model weights
- label maps
- normalization statistics
- architecture metadata
- sensor-layout metadata
- evaluation metrics

Saved models live under:

- `models/strict/`

This gives the project two complementary views of performance:

- per-subject models for best-case personalized control
- cross-subject models for generalization across participants

### 4. Realtime Inference

Live inference runs through:

```powershell
python realtime_gesture_cnn.py --model-right models/strict/per_subject/right/Matthewv6_4_gestures.pt --model-left models/strict/per_subject/left/Matthewv6_4_gestures.pt
```

At runtime, the system:

1. resolves channel identities from the Delsys metadata
2. performs the maintained calibration and normalization steps
3. windows the incoming EMG stream
4. runs the per-arm CNNs
5. publishes a dual-arm gesture output for downstream consumers

The realtime path is dual-arm only in the maintained repo state.

The design idea is straightforward: each arm is processed independently first, then the two outputs are combined into the control signal that CARLA uses. That keeps the realtime pipeline modular and makes it easier to inspect per-arm behavior when tuning the system.

### 5. CARLA Control

The canonical CARLA client is:

- `carla_integration/manual_control_emg.py`

It launches realtime inference internally, reads the published gesture output, and converts it into vehicle commands inside CARLA.

The maintained CARLA workflow supports:

- free-roam practice
- `lane_keep_5min`
- `highway_overtake`
- evaluation logging through `--eval-log-dir`

The practical reason for keeping CARLA in the same repo is that the project is meant to evaluate control, not only recognition. CARLA provides a controlled environment where the gesture interface can be tested on structured tasks instead of being judged only by offline classification numbers.

### 6. Evaluation

The maintained evaluation pipeline lives in `eval_metrics/` and is centered on three metric groups:

- offline model metrics
- end-to-end latency
- CARLA driving metrics

The final maintained metric set is:

- Offline:
  - balanced accuracy
  - macro precision
  - macro recall
  - macro F1
  - worst-class recall
- Latency:
  - end-to-end latency mean
  - end-to-end latency p95
- CARLA:
  - scenario success
  - completion time
  - mean velocity
  - lane offset mean
  - steering angle mean
  - steering entropy
  - lane error RMSE
  - lane invasions

The evaluation scripts are designed around participant-first aggregation. In other words, runs are summarized per participant first, then aggregated across participants for final tables and plots. That keeps the results aligned with how the study is actually conducted.

## Repo Layout

- `emg/`
  - preprocessing, sensor-layout helpers, model loading, training helpers, and runtime tuning
- `carla_integration/`
  - CARLA client, scenario presets, and launcher scripts
- `eval_metrics/`
  - maintained evaluation pipeline and plotting scripts
- `project_notes/`
  - technical reference and maintained project notes
- `models/strict/`
  - saved CNN bundles for the maintained workflow

## Current Maintained State

The current repo has been cleaned down to the active workflow only:

- CNN-only model stack
- fixed sensor placement workflow
- fixed 4-gesture contract
- dual-arm realtime inference
- keyboard-first + EMG CARLA control, with optional steering-wheel throttle/brake support still present in the client
- one preferred CARLA camera view for evaluation, while the default camera toggle remains available in code

## Quick Start

### 1. Collect and preprocess data

```powershell
python DelsysPythonGUI.py
python emg/resample_raw_dataset.py
python emg/filtering.py
```

### 2. Train models

```powershell
python train_per_subject.py
python train_cross_subject.py
```

### 3. Run realtime inference

```powershell
python realtime_gesture_cnn.py --model-right models/strict/per_subject/right/Matthewv6_4_gestures.pt --model-left models/strict/per_subject/left/Matthewv6_4_gestures.pt
```

### 4. Run CARLA

Start the CARLA server:

```bat
carla_integration\test_start_carla_server_0_9_16.bat
```

Run free-roam practice:

```bat
carla_integration\test_run_manual_control_emg_0_9_16.bat
```

Run the named scenarios:

```bat
carla_integration\lane_keep_5min.cmd
carla_integration\highway_overtake.cmd
```

Run the logging wrappers for evaluation:

```bat
carla_integration\lane_keep_5min_eval.cmd
carla_integration\highway_overtake_eval.cmd
```

## CARLA Notes

The maintained CARLA client keeps a practical, simplified interface:

- standard RGB camera view preferred for evaluation, with `Tab` still wired to the inherited camera toggle
- keyboard weather control with `C` and `Shift+C`
- `Backspace` restarts the vehicle
- in free roam, `Backspace` respawns a random vehicle

Free roam currently defaults to:

- map: `Town03_Opt`
- `90` ambient vehicles
- `0` pedestrians

Named scenarios currently use:

- `lane_keep_5min` on `Town04_Opt`
- `highway_overtake` on `Town04_Opt`

More CARLA-specific detail lives in [carla_integration/README.md](/C:/Users/matth/Desktop/capstone/capstone-emg/carla_integration/README.md).

## Evaluation Workflow

Use this order when collecting and processing final evaluation runs.

For normal use, you should not need any CLI flags.

If the maintained model filenames changed, update only these lines in [eval_metrics/config.py](/C:/Users/matth/Desktop/capstone/capstone-emg/eval_metrics/config.py):

- `ACTIVE_PER_SUBJECT_MODEL_NAME`
- `ACTIVE_CROSS_SUBJECT_MODEL_NAME`

Then the normal full refresh is just:

```powershell
python eval_metrics/run_current_eval.py
```

If you only want to refresh the report-grade figures from already staged data, run:

```powershell
python eval_metrics/plot_scripts/make_report_figures.py
```

### 1. Collect raw evaluation runs

Run the named eval wrappers:

```bat
carla_integration\lane_keep_5min_eval.cmd
carla_integration\highway_overtake_eval.cmd
```

Each wrapper launches `manual_control_emg.py` with `--eval-log-dir`, which auto-saves:

- `carla_drive_<timestamp>.csv`
- `realtime_predictions_<timestamp>.csv`

Saved locations:

- `eval_metrics/out/lane_keep_eval/`
- `eval_metrics/out/highway_overtake_eval/`

The filenames are timestamp-based, not manually chosen, so the raw logs are preserved automatically as long as the run starts cleanly.

### 2. Harvest offline model metrics

For the normal no-flag path, the maintained model filenames come from `eval_metrics/config.py`.

If you want to refresh only the offline bundle harvest, run:

```powershell
python eval_metrics/pipeline_scripts/harvest_model_metrics.py
```

This exports:

- `balanced_accuracy`
- `macro_precision`
- `macro_recall`
- `macro_f1`
- `worst_class_recall`

### 3. Analyze CARLA latency and driving metrics

If you want the convenience path after a batch of runs, use:

```powershell
python eval_metrics/pipeline_scripts/gather_current_metrics.py
```

That stages discovered run summaries under:

- `eval_metrics/out/current_metrics/`

If you want the manual per-run path instead, run:

```powershell
python eval_metrics/pipeline_scripts/analyze_latency.py --realtime-log <realtime_csv> --carla-log <carla_csv> --output-json <latency_json> --output-csv <latency_joined_csv>
python eval_metrics/pipeline_scripts/analyze_drive_metrics.py --log <carla_csv> --output-json <drive_summary_json>
```

These produce the maintained live-run metrics:

- Latency:
  - mean end-to-end latency
  - p95 end-to-end latency
- CARLA:
  - scenario success
  - completion time
  - mean velocity
  - lane offset mean
  - steering angle mean
  - steering entropy
  - lane error RMSE
  - lane invasions

For `highway_overtake`, success/fail is carried by `scenario_success` on those overtake rows.

## Technical Reference

The canonical technical reference for the maintained repo state is:

- [technical_reference.md](/C:/Users/matth/Desktop/capstone/capstone-emg/project_notes/technical_reference.md)

That file is the source of truth for the current sensor placement contract, active training and realtime setup, CARLA integration details, and evaluation structure.
