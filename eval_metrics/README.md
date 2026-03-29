# Evaluation Metrics

This folder holds the maintained evaluation pipeline for offline model metrics, end-to-end latency, and CARLA driving metrics.

## Core scripts

- `harvest_model_metrics.py`
  - Read saved CNN bundles and export the final offline metric set.
- `gather_current_metrics.py`
  - Stage a current snapshot from tracked model bundles and CARLA run artifacts.
- `plot_model_accuracy_bars.py`
  - Plot the final offline model metrics as one aggregate figure.
- `plot_carla_summary_bars.py`
  - Plot the final aggregate CARLA metrics as one summary figure.
- `analyze_latency.py`
  - Join realtime and CARLA timing logs and summarize end-to-end latency.
- `analyze_drive_metrics.py`
  - Summarize CARLA drive logs into the final driving metric set.
- `build_eval_tables.py`
  - Build participant-level and aggregate report tables from a manifest.

## Log collection

Prediction log only:

```powershell
python realtime_gesture_cnn.py --model-right models/strict/per_subject/right/Matthewv6_4_gestures.pt --model-left models/strict/per_subject/left/Matthewv6_4_gestures.pt --prediction-log eval_metrics/out/realtime_predictions.csv
```

CARLA drive log plus forwarded realtime prediction log:

```powershell
python carla_integration/manual_control_emg.py --scenario lane_keep_5min --eval-log-dir eval_metrics/out/lane_keep_eval
```

## Final metric set

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
  - lane error RMSE
  - lane invasions
- Scenario-specific:
  - `highway_overtake` command success rate, only when explicitly logged

## Offline metric harvest

```powershell
python eval_metrics/harvest_model_metrics.py --models-root models/strict --output-csv eval_metrics/out/model_metrics.csv --output-json eval_metrics/out/model_metrics.json
```

## Table assembly

```powershell
python eval_metrics/build_eval_tables.py --write-template-manifest eval_metrics/table_manifest.csv
python eval_metrics/build_eval_tables.py --manifest eval_metrics/table_manifest.csv
```

The checked-in [table_manifest_template.csv](/C:/Users/matth/Desktop/capstone/capstone-emg/eval_metrics/table_manifest_template.csv) is a maintained example of the expected manifest columns.

## Final plots

Offline figure:

```powershell
python eval_metrics/plot_model_accuracy_bars.py --input-csv eval_metrics/out/model_metrics.csv --gesture-bucket 4_gesture --latest-only
```

CARLA figure:

```powershell
python eval_metrics/plot_carla_summary_bars.py --input-csv eval_metrics/out/tables/evaluation_aggregate_table.csv --deliverable-bucket capstone_report
```

## Notes

- `eval_metrics/out/` and `eval_metrics/logs/` are generated outputs, not source data.
- The maintained evaluation flow is participant-first: summarize each participant/run first, then aggregate across participants for report tables.
- Standalone prompted realtime behavior metrics are no longer part of the maintained evaluation path.
- The final visualization plan is one offline figure, one CARLA figure, and latency kept as a table rather than a standalone plot.
- The evaluation scripts assume the cleaned CNN-only repo state with the fixed 4-gesture set: `left_turn`, `right_turn`, `neutral`, `horn`.
