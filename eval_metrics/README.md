# Evaluation Metrics

This folder contains the maintained evaluation pipeline for the current fixed 4-gesture CNN workflow:

- `left_turn`
- `right_turn`
- `neutral`
- `horn`

Use this README as the runbook for collecting, processing, and plotting final evaluation results.

## Normal Use

For normal evaluation work, you should not need any CLI flags.

1. If the maintained model filenames changed, update only these lines in `eval_metrics/config.py`:
   - `ACTIVE_PER_SUBJECT_MODEL_NAME`
   - `ACTIVE_CROSS_SUBJECT_MODEL_NAME`
2. Put the raw CARLA eval logs in:
   - `eval_metrics/out/lane_keep_eval/`
   - `eval_metrics/out/highway_overtake_eval/`
3. Run the full refresh:

```powershell
python eval_metrics/run_current_eval.py
```

That one script:

- harvests the maintained offline model metrics
- stages latency and drive summaries under `eval_metrics/out/current_metrics/`
- refreshes `eval_metrics/table_manifest.csv`
- rebuilds the tables under `eval_metrics/out/tables/`
- writes the final aggregate tables you actually grab into:
  - `eval_metrics/final_capstone_table.csv`
  - `eval_metrics/final_research_table.csv`
- overwrites all figures under `eval_metrics/out/figures/`
- removes old legacy summary-bar figures that are no longer maintained

If you only want to refresh the report-grade figures from already staged data, run:

```powershell
python eval_metrics/plot_scripts/make_report_figures.py
```

## What You Collect

### CARLA metrics

- `scenario_success`
- `completion_time_s`
- `mean_velocity_mps`
- `lane_offset_mean_m`
- `steering_angle_mean_rad`
- `steering_entropy`
- `lane_error_rmse_m`
- `lane_invasions`

For `highway_overtake`, success/fail is carried by `scenario_success` on the overtake rows.

### CARLA latency

- `lat_e2e_mean_ms`
- `lat_e2e_p95_ms`

These are mean and p95 end-to-end latency from realtime window end to CARLA control apply.

### Model metrics

- `balanced_accuracy`
- `macro_precision`
- `macro_recall`
- `macro_f1`
- `worst_class_recall`

## Raw Run Collection

During testing, the main thing you need to do is run the eval wrappers:

```bat
carla_integration\lane_keep_5min_eval.cmd
carla_integration\highway_overtake_eval.cmd
```

Those wrappers launch `manual_control_emg.py` with `--eval-log-dir`, which auto-saves two CSVs per run:

- `carla_drive_<timestamp>.csv`
- `realtime_predictions_<timestamp>.csv`

Saved locations:

- `eval_metrics/out/lane_keep_eval/`
- `eval_metrics/out/highway_overtake_eval/`

The filenames are timestamp-based, so you do not need to name runs manually.

## Script Roles

### `harvest_model_metrics.py`

Reads saved model bundles under `models/strict/` and exports the offline metrics:

- `balanced_accuracy`
- `macro_precision`
- `macro_recall`
- `macro_f1`
- `worst_class_recall`

Outputs:

- CSV, usually `eval_metrics/out/current_metrics/model_metrics.csv`
- JSON, usually `eval_metrics/out/current_metrics/model_metrics.json`

Example:

```powershell
python eval_metrics/pipeline_scripts/harvest_model_metrics.py
```

### `analyze_latency.py`

Inputs:

- one `realtime_predictions_*.csv`
- one matching `carla_drive_*.csv`

Joins them on `prediction_seq` and computes:

- `lat_e2e_mean_ms`
- `lat_e2e_p95_ms`

Outputs:

- latency summary JSON
- joined latency CSV

### `analyze_drive_metrics.py`

Input:

- one `carla_drive_*.csv`

Computes:

- `scenario_success`
- `completion_time_s`
- `mean_velocity_mps`
- `lane_offset_mean_m`
- `steering_angle_mean_rad`
- `steering_entropy`
- `lane_error_rmse_m`
- `lane_invasions`

If the log contains the extra field, it can also emit:

- `command_success_rate`

That field is optional and is not part of the main required metric set.

### `build_eval_tables.py`

Merges:

- offline model metrics CSV
- latency summary JSON files
- drive summary JSON files

Builds:

- participant-level tables
- aggregate tables
- capstone-specific table exports
- research-paper-specific table exports

The current table builder preserves scenario identity from the drive summaries, so `lane_keep_5min` and `highway_overtake` can remain separate if you keep them as separate manifest rows.

### `gather_current_metrics.py`

Convenience script that:

1. harvests current model metrics
2. discovers paired CARLA/realtime run logs
3. runs latency analysis for each pair
4. runs drive analysis for each CARLA log
5. stages outputs under `eval_metrics/out/current_metrics/`

Use this after a batch of runs if you want a quick staged snapshot.

## Required Order

### 1. Collect raw runs

Run:

```bat
carla_integration\lane_keep_5min_eval.cmd
carla_integration\highway_overtake_eval.cmd
```

Repeat as needed for each participant and scenario.

### 2. Harvest offline model metrics

For the normal no-flag path, the maintained model filenames come from `eval_metrics/config.py`.

If you want just the offline bundle harvest refreshed by itself, run:

```powershell
python eval_metrics/pipeline_scripts/harvest_model_metrics.py
```

### 3. Analyze the CARLA run logs

#### Convenience path

```powershell
python eval_metrics/pipeline_scripts/gather_current_metrics.py
```

This stages discovered outputs under:

- `eval_metrics/out/current_metrics/`

#### Manual path

For each run pair:

```powershell
python eval_metrics/pipeline_scripts/analyze_latency.py --realtime-log <realtime_csv> --carla-log <carla_csv> --output-json <latency_json> --output-csv <latency_joined_csv>
python eval_metrics/pipeline_scripts/analyze_drive_metrics.py --log <carla_csv> --output-json <drive_summary_json>
```

### 4. Build the report tables

For the normal no-flag path, refresh the filled manifest from staged outputs:

```powershell
python eval_metrics/pipeline_scripts/fill_table_manifest.py
python eval_metrics/pipeline_scripts/build_eval_tables.py
```

Main outputs:

- `eval_metrics/out/tables/evaluation_participant_table.csv`
- `eval_metrics/out/tables/evaluation_aggregate_table.csv`
- `eval_metrics/final_capstone_table.csv`
- `eval_metrics/final_research_table.csv`

### 5. Generate plots

The maintained figure layer focuses on distributions, fold spread, and confusion structure rather than legacy summary bars.

#### Confusion matrix heatmap

Plots row-normalized confusion matrices directly from saved bundles and annotates counts:

```powershell
python eval_metrics/plot_scripts/plot_confusion_matrix.py
```

#### Latency ECDF

Plots end-to-end latency distributions by scenario from the staged `latency_joined.csv` files:

```powershell
python eval_metrics/plot_scripts/plot_latency_ecdf.py
```

#### CARLA run distributions

Plots run-level CARLA distributions with boxplots and per-run points. This is the maintained CARLA figure now, and it includes:

- `completion_time_s`
- `mean_velocity_mps`
- `lane_offset_mean_m`
- `steering_angle_mean_rad`
- `steering_entropy`
- `lane_error_rmse_m`
- `lane_invasions`

```powershell
python eval_metrics/plot_scripts/plot_carla_run_distributions.py
```

#### Offline balanced-accuracy forest plot

Plots the current offline model comparison as a point-and-interval figure. Where fold-level metrics are stored, it shows fold dots and a 95% interval:

```powershell
python eval_metrics/plot_scripts/plot_model_accuracy_forest.py
```

#### Offline metric heatmap

Plots the full offline metric set as an annotated heatmap:

```powershell
python eval_metrics/plot_scripts/plot_model_metric_heatmap.py
```

#### Gesture embedding clusters

Plots a 2D t-SNE projection of each current model's learned penultimate embedding space, colored by gesture:

```powershell
python eval_metrics/plot_scripts/plot_gesture_embedding_clusters.py
```

#### Representative drive traces

Plots one automatically selected representative lane-keep run and one representative highway-overtake run, with lane error, speed, and steering angle over time:

```powershell
python eval_metrics/plot_scripts/plot_drive_trace_representative.py
```

#### One-command figure pack

Generates the maintained report-grade figure set into `eval_metrics/out/figures/`:

```powershell
python eval_metrics/plot_scripts/make_report_figures.py
```

#### One-command full refresh

Generates the staged metrics, tables, and figure outputs in one pass:

```powershell
python eval_metrics/run_current_eval.py
```

## Core Output Locations

Raw run logs:

- `eval_metrics/out/lane_keep_eval/`
- `eval_metrics/out/highway_overtake_eval/`

Offline bundle harvest:

- `eval_metrics/out/model_metrics.csv`
- `eval_metrics/out/model_metrics.json`

Convenience staged snapshot:

- `eval_metrics/out/current_metrics/`

Final tables:

- `eval_metrics/out/tables/`
- `eval_metrics/final_capstone_table.csv`
- `eval_metrics/final_research_table.csv`

Report-grade figures:

- `eval_metrics/out/figures/`

## Important Notes

- `eval_metrics/out/` is generated output, not source data.
- normal use is no flags; the maintained filenames live in `eval_metrics/config.py`
- if you archive old bundles out of `models/strict/`, you should not need to touch model-selection flags at all
- top-level scripts you normally care about are:
  - `eval_metrics/config.py`
  - `eval_metrics/run_current_eval.py`
- implementation scripts are organized under:
  - `eval_metrics/pipeline_scripts/`
  - `eval_metrics/plot_scripts/`
- Keep `lane_keep_5min` and `highway_overtake` as separate manifest rows if you want them reported separately.
- The maintained evaluation flow is participant-first: repeated runs for one participant are averaged at the participant level before final aggregate rows are computed.
- The plotting layer now supports:
  - one offline model forest plot
  - one offline model heatmap
  - one gesture embedding cluster figure
  - one CARLA run-distribution figure
  - one representative drive-trace figure
  - one latency ECDF figure
