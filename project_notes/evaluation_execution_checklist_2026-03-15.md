# Evaluation Execution Checklist

Date created: 2026-03-15

## Purpose

This is the single working document for evaluation planning and execution.

Use it to track:

- what metrics belong in the capstone final report
- what extra metrics belong in the research paper
- which 3-gesture and 5-gesture runs are already available
- which per-subject and cross-subject results still need to be collected
- which scripts, logs, figures, and tables still need to be created

## Scope Lock

### Capstone final report

The capstone final report will include the metrics outlined in the design roadmap.

### Research paper

The research paper will include:

- the same core metrics used for the capstone final report
- comparison metrics drawn from the previous research paper
- no eye-tracker-related metrics

### Evaluation dimensions that must be covered for both deliverables

- 3 gestures
- 5 gestures
- per-subject
- cross-subject

## Fixed Evaluation Matrix

This matrix is the first thing to keep stable.
Every result collected later should map to one row below.

| Row | Deliverable | Gesture Set | Model Scope | Required Metric Bundles | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Capstone final report | 3-gesture | Per-subject | Offline bundle + roadmap system metrics | Pending | |
| 2 | Capstone final report | 3-gesture | Cross-subject | Offline bundle + roadmap system metrics | Pending | |
| 3 | Capstone final report | 5-gesture | Per-subject | Offline bundle + roadmap system metrics | Pending | |
| 4 | Capstone final report | 5-gesture | Cross-subject | Offline bundle + roadmap system metrics | Pending | |
| 5 | Research paper | 3-gesture | Per-subject | Offline bundle + roadmap system metrics + prior-paper comparison bundle | Pending | No eye-tracker metrics |
| 6 | Research paper | 3-gesture | Cross-subject | Offline bundle + roadmap system metrics + prior-paper comparison bundle | Pending | No eye-tracker metrics |
| 7 | Research paper | 5-gesture | Per-subject | Offline bundle + roadmap system metrics + prior-paper comparison bundle | Pending | No eye-tracker metrics |
| 8 | Research paper | 5-gesture | Cross-subject | Offline bundle + roadmap system metrics + prior-paper comparison bundle | Pending | No eye-tracker metrics |

## Metric Bundle Definitions

Use these definitions when deciding the matrix status for each row.

### Offline bundle

Mark this bundle complete only if all of the following exist for that row:

- accuracy
- balanced accuracy
- macro F1
- weighted F1
- per-class recall
- confusion matrix
- worst-class recall

Optional but useful when available:

- LOSO summary
- cross-subject spread

### Roadmap system metrics

These are the main non-offline metrics that support the design-roadmap style evaluation.
Mark this bundle complete only if the row has the applicable system-level results you plan to report.

- prompted-segment accuracy
- time to first correct prediction
- time to stable prediction
- average prediction confidence
- label-flip rate
- stale-prediction rate
- end-to-end software latency
- mean latency
- median latency
- p90 latency
- p95 latency
- max latency
- average lane keeping error
- lane keeping RMSE
- lane invasions
- collisions
- completion time
- steering smoothness
- command success rate

### Prior-paper comparison bundle

These are required for the research-paper rows only.
Mark this bundle complete only if the row includes the subset of prior-paper-comparable metrics you actually intend to discuss.

- classification-performance metrics that can be compared fairly
- driving or control outcome metrics
- lane-keeping-related metrics
- collisions or error events
- completion time or task-performance summaries
- steering or control-response summaries

Explicitly excluded:

- blink rate
- eye-tracker metrics
- eye-gaze-derived metrics

## Status Rules

Use these rules for the matrix `Status` column.

### `Already have`

Use this only if:

- the exact gesture set matches the row
- the exact model scope matches the row
- the metric bundles required for that row are already computed or logged
- the outputs are in a form you can directly use in the report or paper

### `Need to rerun`

Use this if:

- you have the right model or setup already
- but the run used the wrong gesture subset, wrong data split, wrong subjects, or outdated settings
- or the result exists but is not trustworthy enough to cite

### `Need new logging/script`

Use this if:

- the metric cannot be produced from current saved outputs
- the repo does not yet log the required data
- or you still need a script to extract the metric from raw logs or CARLA runs

## Core Metric Set

These are the main metrics that should feed both the capstone report and the research paper.

### Offline model metrics

- accuracy
- balanced accuracy
- macro F1
- weighted F1
- per-class recall
- confusion matrix
- worst-class recall

### Realtime behavior metrics

- prompted-segment accuracy
- time to first correct prediction
- time to stable prediction
- average prediction confidence
- label-flip rate
- stale-prediction rate

### Latency metrics

- end-to-end software latency
- mean latency
- median latency
- p90 latency
- p95 latency
- max latency

### Driving / control metrics

- average lane keeping error
- lane keeping RMSE
- lane invasions
- collisions
- completion time
- steering smoothness
- command success rate

### Signal-processing / methods figures

- raw vs filtered EMG overlay
- PSD before filtering
- PSD after filtering

## Prior-Paper Comparison Metrics

These are for the research paper only.
Use the previous paper for comparison where applicable, excluding eye-tracker-related metrics.

### Include when available

- classification performance metrics that can be compared fairly
- driving-control outcome metrics
- lane-keeping-related metrics
- collisions or error events
- completion time or task-performance summaries
- steering or control-response summaries

### Exclude

- blink rate
- eye-tracker metrics
- eye-gaze-derived workload or attention metrics

## Current Status Snapshot

### Already available or mostly available

- [x] Offline deep-learning metrics from `train_per_subject.py`
- [x] Offline deep-learning metrics from `train_cross_subject.py`
- [x] Prompted realtime logging from `realtime_confidence_analysis.py`
- [x] Realtime run comparison utility from `eval_metrics/compare_realtime_runs.py`
- [x] Dual-arm realtime publishing path from `realtime_gesture_cnn.py`
- [x] CARLA integration reading published gestures from `carla wheel z.py`

### Still missing or not yet organized

- [ ] One official matrix status update for all 8 evaluation rows
- [ ] One consolidated offline metrics table for 3-gesture and 5-gesture runs
- [ ] Confirmation of which per-subject results already exist
- [ ] Confirmation of which cross-subject results already exist
- [ ] Dedicated CARLA run logger
- [ ] Dedicated latency analysis script
- [ ] Driving-metrics postprocessing script
- [ ] Filter-effect plotting script
- [ ] Final capstone tables and figures
- [ ] Final research-paper tables and figures

## First Execution Step

Before collecting anything new, fill the matrix above with the status of what already exists.

Allowed status labels:

- `Already have`
- `Need to rerun`
- `Need new logging/script`

This prevents mixing:

- 3-gesture and 5-gesture results
- per-subject and cross-subject results
- capstone-only outputs and paper-only outputs

### Practical way to fill the matrix

For each row:

1. Check whether the offline bundle exists.
2. Check whether the roadmap system metrics exist.
3. For research-paper rows, check whether the prior-paper comparison bundle exists.
4. Then assign the row status:
   - `Already have` if all required bundles exist
   - `Need to rerun` if the bundles are possible with current models/runs but the current outputs are mismatched or outdated
   - `Need new logging/script` if one or more required bundles cannot yet be produced from current artifacts

## Inventory Checklist

Use this section to audit what already exists.

### 3-gesture results

#### Per-subject

- [ ] Model files identified
- [ ] Offline metrics identified
- [ ] Realtime metrics identified
- [ ] CARLA metrics identified

#### Cross-subject

- [ ] Model files identified
- [ ] Offline metrics identified
- [ ] Realtime metrics identified
- [ ] CARLA metrics identified

### 5-gesture results

#### Per-subject

- [ ] Model files identified
- [ ] Offline metrics identified
- [ ] Realtime metrics identified
- [ ] CARLA metrics identified

#### Cross-subject

- [ ] Model files identified
- [ ] Offline metrics identified
- [ ] Realtime metrics identified
- [ ] CARLA metrics identified

## Workstream A: Offline Metrics Harvest

### Goal

Collect and organize the offline results already produced by the training scripts.

### Existing sources

- `train_per_subject.py`
- `train_cross_subject.py`

### Required outputs

- [ ] 3-gesture per-subject offline table
- [ ] 3-gesture cross-subject offline table
- [ ] 5-gesture per-subject offline table
- [ ] 5-gesture cross-subject offline table
- [ ] One combined comparison table for writing
- [ ] Confusion matrix figures selected

### Notes

- This is the first real data-gathering step because most of it likely already exists.
- Do this before writing new scripts.

## Workstream B: Realtime Metrics

### Goal

Collect realtime control-behavior evidence for the same 3-gesture and 5-gesture conditions.

### Existing sources

- `realtime_confidence_analysis.py`
- `eval_metrics/compare_realtime_runs.py`
- `eval_metrics/realtime_behavior_metrics.py`

### Required outputs

- [ ] 3-gesture per-subject realtime summary
- [ ] 3-gesture cross-subject realtime summary
- [ ] 5-gesture per-subject realtime summary
- [ ] 5-gesture cross-subject realtime summary

### Required metrics

- [ ] Prompted-segment accuracy
- [ ] Time to first correct prediction
- [ ] Time to stable prediction
- [ ] Average prediction confidence
- [ ] Label-flip rate
- [ ] Stale-prediction rate

## Workstream C: Latency

### Goal

Measure the software delay through the realtime pipeline.

### Current status

- Realtime file has commented latency code ideas
- No dedicated analysis script yet

### Required implementation

- [ ] Add lightweight timestamps to the realtime path
- [ ] Add lightweight timestamps to the CARLA control path
- [x] Create `eval_metrics/analyze_latency.py`

### Required outputs

- [ ] 3-gesture latency summary
- [ ] 5-gesture latency summary
- [ ] Mean / median / p95 / max latency
- [ ] One latency figure

## Workstream D: CARLA Driving Metrics

### Goal

Collect system-level driving or control metrics that support both the roadmap results and the paper comparison.

### Required implementation

- [ ] Add a CARLA logger to `carla wheel z.py`
- [x] Create `eval_metrics/analyze_drive_metrics.py`

### Required metrics

- [ ] Average lane keeping error
- [ ] Lane keeping RMSE
- [ ] Lane invasions
- [ ] Collisions
- [ ] Completion time
- [ ] Steering smoothness
- [ ] Command success rate

### Required outputs

- [ ] 3-gesture CARLA summary
- [ ] 5-gesture CARLA summary
- [ ] Prior-paper comparison table entries filled where applicable

## Workstream E: Signal-Processing Figures

### Goal

Generate low-cost figures that make the filtering and preprocessing visible.

### Required implementation

- [x] Create `eval_metrics/plot_filter_effect.py`

### Required outputs

- [ ] Raw vs filtered overlay
- [ ] PSD before filtering
- [ ] PSD after filtering

## Workstream F: Final Deliverables

### Capstone final report

- [ ] Core metrics table for 3-gesture and 5-gesture
- [ ] Per-subject and cross-subject comparison table
- [ ] Realtime/latency summary
- [ ] CARLA/control summary
- [ ] Filtering figure
- [ ] Use-case section

### Research paper

- [ ] Core metrics table for 3-gesture and 5-gesture
- [ ] Per-subject and cross-subject comparison table
- [ ] Prior-paper comparison table
- [ ] Realtime/latency figure
- [ ] Driving/control figure or table
- [ ] Filtering figure
- [ ] Use-case section

## Use Cases To Target In Writing

Use these consistently in both deliverables.

### Primary use cases

- adaptive control for upper-limb amputees
- adaptive control for users with reduced hand function
- driving simulation and gaming
- remote robot or vehicle teleoperation
- industrial or hazardous-environment command input

### Safe wording

- prototype control interface
- adaptive or assistive command interface
- simulator or teleoperation control system
- target application for amputees, not clinical validation

## Recommended Order Of Work

1. Fill the fixed evaluation matrix status column
2. Harvest offline metrics that already exist
3. Confirm which 3-gesture and 5-gesture rows are missing
4. Run or collect missing realtime results
5. Add latency logging and analysis
6. Add CARLA logging and drive-metrics analysis
7. Add filter-effect figure generation
8. Build final capstone tables and figures
9. Build final research-paper tables and figures

## Immediate Next Actions

- [ ] Mark each matrix row as `Already have`, `Need to rerun`, or `Need new logging/script`
- [ ] Identify the exact model files for 3-gesture per-subject and cross-subject
- [ ] Identify the exact model files for 5-gesture per-subject and cross-subject
- [ ] Export or copy the offline metrics you already have into one place

## File Targets

### Existing files to use

- `train_per_subject.py`
- `train_cross_subject.py`
- `realtime_confidence_analysis.py`
- `realtime_gesture_cnn.py`
- `carla wheel z.py`
- `eval_metrics/compare_realtime_runs.py`
- `eval_metrics/diagnose_session_recall.py`

### New files expected

- `eval_metrics/analyze_latency.py`
- `eval_metrics/analyze_drive_metrics.py`
- `eval_metrics/plot_filter_effect.py`
- `eval_metrics/harvest_model_metrics.py`
- `eval_metrics/realtime_behavior_metrics.py`

## Daily Update Template

Copy this block to the bottom of the file when a work session ends.

### Session update template

Date:

Matrix rows updated:

- 

Completed:

- 

Outputs created:

- 

Blocked on:

- 

Next action:

- 
