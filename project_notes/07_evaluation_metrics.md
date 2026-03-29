# Evaluation Metrics

## What It Is Used For

The maintained evaluation subsystem turns trained bundles and CARLA logs into the final metrics, tables, and figures used for reporting.

The active scripts live in `eval_metrics/`:

- `harvest_model_metrics.py`
- `analyze_latency.py`
- `analyze_drive_metrics.py`
- `build_eval_tables.py`
- `gather_current_metrics.py`
- `plot_model_accuracy_bars.py`
- `plot_carla_summary_bars.py`

## Why It Was Made

The project needs evaluation at three levels:

- offline classification quality
- end-to-end timing
- simulator control performance

Here, end-to-end timing means the full delay from a gesture window finishing in the realtime pipeline to the corresponding vehicle control action being applied in CARLA.

No single script can answer all three well, and no single metric is enough. The evaluation subsystem exists so the repo can move from raw artifacts to reportable numbers in a repeatable way instead of relying on manual spreadsheet work.

It also enforces the maintained reporting position of the cleaned repo:

- CNN-only model stack
- fixed 4-gesture workflow
- participant-first aggregation
- no standalone prompted realtime-behavior analysis in the maintained path

## How It Works

`harvest_model_metrics.py` scans model bundles and reads stored metadata directly from them. It exports the maintained offline metrics:

- balanced accuracy
- macro precision
- macro recall
- macro F1
- worst-class recall

It also records bundle scope, subject, arm, label set, channel count, window settings, and creation timestamp so later scripts can filter or group results without reopening the model files.

`analyze_latency.py` joins the realtime prediction log and the CARLA drive log by `prediction_seq`. It computes end-to-end latency from `window_end_ts` in the prediction log to `control_apply_ts` in the CARLA log. The join key matters because it ensures the timing comparison is made between the matching prediction and control event rather than between unrelated rows. The maintained latency outputs are:

- mean end-to-end latency
- p95 end-to-end latency

`p95` means the 95th percentile. In practical terms, it answers: "How bad is latency on the slower end of normal operation?" It is useful because mean latency alone can hide occasional but important slow responses.

`analyze_drive_metrics.py` reads a CARLA drive log and summarizes the maintained driving metrics:

- scenario success
- completion time
- lane error RMSE
- lane invasions

For `highway_overtake`, it also reports `command_success_rate` if and only if the input log explicitly contains `command_correct`.

`build_eval_tables.py` is the report-assembly layer. It reads a manifest, flattens offline, latency, and drive summaries into participant rows, and then builds aggregate rows from those participant rows.

Here:

- the manifest is the table of evaluation inputs that tells the script which models and log summaries belong in which report rows
- participant rows are the per-participant outputs after the offline, latency, and drive summaries have been combined
- aggregate rows are the final summarized rows used for report-level tables and figures

The key design choice is participant-first aggregation. The script averages multiple rows for the same participant before computing the final aggregate means and standard deviations. That prevents participants with more runs from dominating the reported result.

The plotting scripts then summarize the final outputs:

- `plot_model_accuracy_bars.py` builds the offline summary figure
- `plot_carla_summary_bars.py` builds the CARLA summary figure

`gather_current_metrics.py` is a convenience script that stages a fresh current snapshot by running the core evaluation scripts and collecting discovered CARLA run pairs into one output area.

## How We Validate

Evaluation validation is mostly about traceability and correct aggregation.

Offline metric validation comes from the fact that the bundle already contains:

- the label map
- the exact metric payload saved at training time
- the bundle scope and subject metadata

That design improves traceability. It means the reported offline metrics can be traced back to the exact saved model artifact instead of depending on a separate handwritten record.

Latency validation is explicit. `analyze_latency.py` fails if the two logs do not share any `prediction_seq` values, because a latency number without a valid join is meaningless.

Drive-metric validation is also explicit:

- lane error is computed from the logged lane offset values
- lane invasions come from logged invasion events
- scenario success is taken from the scenario runtime state rather than guessed from the path alone
- command success rate is omitted unless the required field is actually present

Table validation comes from the manifest-plus-aggregation structure:

- model rows can be matched by path, filename, or metadata
- latency and drive JSON files are flattened only if they exist
- aggregate rows are computed from participant summaries, not raw run counts

This matters because reporting by raw run count can overweight participants who simply generated more logged runs. Participant-first aggregation keeps the reported summary aligned with the study design rather than the artifact count.

## What The Validation Is Used For

These checks are used to make the final reported numbers defensible.

Offline validation tells us whether the classifier is good enough.

Latency validation tells us whether the classifier is fast enough in the live control loop.

Drive validation tells us whether the user can actually complete the CARLA tasks with acceptable control quality.

Participant-first aggregation then turns those run-level facts into report-level facts without overweighting the participants who happened to contribute more rows. That is why the evaluation subsystem matters. It does not only compute metrics. It protects the interpretation of the final results.
