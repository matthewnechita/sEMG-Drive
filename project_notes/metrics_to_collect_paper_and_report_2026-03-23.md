# Metrics To Collect

This note is the short source of truth for the research paper and final report metric set.

It supersedes older mentions of `steering smoothness`, `command success rate`, and `collision count`.

Yes: the core metric set should be the same for the research paper and the final report.

The final report can include more explanation and extra supporting plots, but the main reported metrics should stay aligned.

Recommended main metric set:

- offline model metrics
- latency metrics
- CARLA scenario metrics

Prompted realtime behavior metrics should be treated as optional backup analysis, not a required main section.

## 1. Offline model metrics

Primary:

- balanced accuracy
- macro F1

Supporting:

- accuracy
- weighted F1
- per-class recall
- worst-class recall
- confusion matrix
- confusion-to-neutral rate
- neutral prediction false-positive rate

How to collect:

1. Train and save the model bundle with `python train_per_subject.py` or `python train_cross_subject.py`
2. Harvest the stored offline metrics from the saved bundles:

```bash
python eval_metrics/harvest_model_metrics.py --models-root models/strict
```

Outputs:

- `eval_metrics/out/model_metrics.csv`
- `eval_metrics/out/model_metrics.json`

## 2. Optional backup: realtime behavior metrics

- prompt-conditioned balanced accuracy
- time to first correct prediction
- time to stable prediction
- label-flip rate
- stale-prediction rate
- average prediction confidence

Use only if you need extra evidence about flicker or prediction stability beyond CARLA and latency results.

How to collect:

1. Run a prompted realtime evaluation with:

```bash
python realtime_confidence_analysis.py --mode dual --model-right <right_bundle> --model-left <left_bundle> --duration-s 60 --segment-s 5 --sequence neutral,left_turn,neutral,right_turn --output eval_metrics/out/realtime_prompted.csv
```

2. Summarize the prompted run with:

```bash
python eval_metrics/realtime_behavior_metrics.py --input eval_metrics/out/realtime_prompted.csv --output-json eval_metrics/out/realtime_behavior_summary.json --output-segments-csv eval_metrics/out/realtime_behavior_segments.csv
```

Notes:

- this path requires a prompted CSV with `prompt_label`
- this is the correct path for behavior metrics, not the plain realtime `--prediction-log` CSV

## 3. Latency metrics

Collect:

- classifier latency
- publish latency
- control latency
- end-to-end latency

Report:

- mean
- median
- p90
- p95
- max

How to collect:

1. Run CARLA with both logs enabled:

```bash
python carla_integration/manual_control_emg.py --scenario lane_keep_5min --eval-log-dir eval_metrics/out/run_01
```

2. Summarize latency from the paired logs:

```bash
python eval_metrics/analyze_latency.py --realtime-log eval_metrics/out/run_01/realtime_predictions_<stamp>.csv --carla-log eval_metrics/out/run_01/carla_drive_<stamp>.csv --output-json eval_metrics/out/run_01/latency_summary.json --output-csv eval_metrics/out/run_01/latency_joined.csv
```

## 4. CARLA scenario metrics

- mean lane error
- lane error RMSE
- lane invasions
- scenario success / failure
- scenario completion time

Notes:

- lane error = distance from the vehicle to the nearest lane centerline
- completion time = from crossing the start checkpoint to reaching the final checkpoint or final scenario objective
- completion time is only valid for defined scenarios, not free-drive runs

How to collect:

1. Run a defined scenario with drive logging enabled:

```bash
python carla_integration/manual_control_emg.py --scenario lane_keep_5min --eval-log-dir eval_metrics/out/run_01
```

or

```bash
python carla_integration/manual_control_emg.py --scenario highway_overtake --eval-log-dir eval_metrics/out/run_02
```

2. Summarize the CARLA drive log:

```bash
python eval_metrics/analyze_drive_metrics.py --log eval_metrics/out/run_01/carla_drive_<stamp>.csv --output-json eval_metrics/out/run_01/drive_metrics_summary.json
```

What CARLA logs during the run:

- scenario started / finished
- scenario success / failure
- checkpoint progress
- completion time
- lane error
- lane invasions
- lead distance and gap for overtake

## 5. Table assembly

After collecting the offline, realtime, latency, and drive summaries:

```bash
python eval_metrics/build_eval_tables.py --manifest eval_metrics/table_manifest.csv
```

If you need a fresh manifest first:

```bash
python eval_metrics/build_eval_tables.py --write-template-manifest eval_metrics/table_manifest.csv
```

## 6. Do not include in the main metric set

- steering smoothness
- command success rate
- collision count
- prompted realtime behavior metrics as a required headline section
