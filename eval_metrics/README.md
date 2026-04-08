# Evaluation Metrics

This folder contains the maintained evaluation pipeline for the current fixed 4-gesture CNN workflow:

- `left_turn`
- `right_turn`
- `neutral`
- `horn`

Use this README as the runbook for collecting, processing, and plotting final evaluation results.

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

- CSV, usually `eval_metrics/out/model_metrics.csv`
- JSON, usually `eval_metrics/out/model_metrics.json`

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

Run once for the current strict bundles:

```powershell
python eval_metrics/harvest_model_metrics.py --models-root models/strict --output-csv eval_metrics/out/model_metrics.csv --output-json eval_metrics/out/model_metrics.json
```

### 3. Analyze the CARLA run logs

#### Convenience path

```powershell
python eval_metrics/gather_current_metrics.py
```

This stages discovered outputs under:

- `eval_metrics/out/current_metrics/`

#### Manual path

For each run pair:

```powershell
python eval_metrics/analyze_latency.py --realtime-log <realtime_csv> --carla-log <carla_csv> --output-json <latency_json> --output-csv <latency_joined_csv>
python eval_metrics/analyze_drive_metrics.py --log <carla_csv> --output-json <drive_summary_json>
```

### 4. Build the report tables

Create or refresh a manifest:

```powershell
python eval_metrics/build_eval_tables.py --write-template-manifest eval_metrics/table_manifest.csv
```

Then fill in the manifest rows with:

- the model bundle path
- the latency summary JSON path
- the drive summary JSON path
- the scenario/condition split you want to report

Then build the tables:

```powershell
python eval_metrics/build_eval_tables.py --manifest eval_metrics/table_manifest.csv --model-metrics eval_metrics/out/model_metrics.csv --output-dir eval_metrics/out/tables
```

Main outputs:

- `eval_metrics/out/tables/evaluation_participant_table.csv`
- `eval_metrics/out/tables/evaluation_aggregate_table.csv`

### 5. Generate plots

#### Model summary plot

Plots all required offline model metrics together:

```powershell
python eval_metrics/plot_model_summary.py --input-csv eval_metrics/out/model_metrics.csv --gesture-bucket 4_gesture --latest-only
```

#### Optional focused model plot

Plots one selected offline metric:

```powershell
python eval_metrics/plot_model_accuracy_bars.py --input-csv eval_metrics/out/model_metrics.csv --gesture-bucket 4_gesture --latest-only
```

#### CARLA summary plot

Plots the maintained CARLA metric summary together:

```powershell
python eval_metrics/plot_carla_summary_bars.py --input-csv eval_metrics/out/tables/evaluation_aggregate_table.csv --deliverable-bucket capstone_report
```

#### Latency summary plot

Plots mean and p95 end-to-end latency:

```powershell
python eval_metrics/plot_latency_summary.py --input-csv eval_metrics/out/tables/evaluation_aggregate_table.csv --deliverable-bucket capstone_report
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

## Important Notes

- `eval_metrics/out/` is generated output, not source data.
- Keep `lane_keep_5min` and `highway_overtake` as separate manifest rows if you want them reported separately.
- The maintained evaluation flow is participant-first: repeated runs for one participant are averaged at the participant level before final aggregate rows are computed.
- The plotting layer now supports:
  - one model summary figure
  - one CARLA summary figure
  - one latency summary figure
