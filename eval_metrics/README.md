# Evaluation Metrics Scripts

This folder holds evaluation and reporting utilities that are separate from the
top-level training and realtime entrypoints.

## Scripts

- `harvest_model_metrics.py`
  - Scan saved model bundles and export their stored offline metrics.
  - Useful for building the 3-gesture / 5-gesture and per-subject / cross-subject tables.

- `plot_filter_effect.py`
  - Compare one raw `.npz` file against its filtered `.npz` counterpart.
  - Saves a figure with a time-domain overlay and PSD comparison.

- `compare_realtime_runs.py`
  - Compare two prompted realtime CSV runs and summarize accuracy, balanced accuracy,
    confidence shifts, and top confusions.

- `realtime_behavior_metrics.py`
  - Compute segment-based realtime behavior metrics from prompted CSV runs.
  - Includes time to first correct prediction, time to stable prediction, label-flip
    rate, carryover stale rate, and confidence summaries.

- `diagnose_session_recall.py`
  - Break down model performance by filtered session file.
  - Useful for identifying weak sessions, label issues, or outlier recordings.

- `analyze_latency.py`
  - Join realtime and CARLA timing logs and compute latency summaries.
  - Expects timestamp columns to be present in the logs.

- `analyze_drive_metrics.py`
  - Summarize CARLA run logs into lane-keeping, collision, timing, and steering metrics.
  - Consumes the per-tick CARLA CSV emitted by `carla wheel z.py`.

- `build_eval_tables.py`
  - Merge offline model metrics with optional realtime/latency/drive summaries into copyable tables.
  - Writes separate `capstone_report_*` and `research_paper_*` tables plus a master table.
  - Uses `table_manifest_template.csv` as the starter input format.

Runtime log collection

- Realtime prediction log only:
  - `python realtime_gesture_cnn.py --model <bundle_path> --prediction-log eval_metrics/out/realtime_predictions.csv`
- CARLA drive log plus realtime prediction log:
  - `python "carla wheel z.py" --eval-log-dir eval_metrics/out/run_01`
- CARLA with explicit log paths:
  - `python "carla wheel z.py" --carla-log eval_metrics/out/carla_drive.csv --realtime-log eval_metrics/out/realtime_predictions.csv`

The CARLA entrypoint now forwards `--realtime-log` into `realtime_gesture_cnn.py` and writes
its own per-tick vehicle/control log to `--carla-log`.

Table assembly

- Write a fresh manifest from the template:
  - `python eval_metrics/build_eval_tables.py --write-template-manifest eval_metrics/table_manifest.csv`
- Fill `eval_metrics/table_manifest.csv` with the exact model rows and optional JSON summaries.
- Build the tables:
  - `python eval_metrics/build_eval_tables.py --manifest eval_metrics/table_manifest.csv`

## Expected workflow

1. Use top-level training scripts to generate or load model bundles.
2. Use `harvest_model_metrics.py` to export offline metrics into CSV/JSON.
3. Use `realtime_confidence_analysis.py` for prompted realtime runs.
4. Use `compare_realtime_runs.py`, `realtime_behavior_metrics.py`, and
   `diagnose_session_recall.py` to summarize and inspect those runs.
5. Add or collect realtime/CARLA logs.
6. Use `analyze_latency.py` and `analyze_drive_metrics.py` on those logs.
7. Use `plot_filter_effect.py` to generate signal-processing figures.

## Notes

- These scripts are intended for evaluation support, not model training.
- They are designed to be usable independently from the top-level scripts.
