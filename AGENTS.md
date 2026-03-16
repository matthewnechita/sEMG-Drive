# AGENTS.md

Project context
- The active stack is the CNN workflow driven by `DelsysPythonGUI.py`, `train_per_subject.py`, `train_cross_subject.py`, and `realtime_gesture_cnn.py`.
- The repo copy of `carla wheel z.py` is the current CARLA integration reference for EMG-driven control testing.
- `carla_integration/` is the repo-owned CARLA workspace for tracked Python-side integration files and reference upstream examples.
- `carla_integration/manual_control_emg.py` is currently a copy of `carla wheel z.py`; keep those aligned until one canonical CARLA entrypoint is chosen.
- The collection GUI entrypoint is `python DelsysPythonGUI.py`; it uses `DataCollector/CollectDataWindow.py`.
- Raw GUI strict collections are stored under `data_strict/<arm> arm/<subject>/raw/*.npz`.
- The recommended preprocessing/training root is `data_resampled_strict/<arm> arm/<subject>/{raw,filtered}`.
- Legacy raw/filtered files may still exist under `data/...` and `data_resampled/...`, but the active strict workflow should stay on the dedicated strict roots.
- Expected environment is the `capstone-emg` conda env with `torch` and `libemg`.

Label contract
- Gesture labels: `left_turn`, `right_turn`, `neutral`, `signal_left`, `signal_right`, `horn`.
- Inter-gesture rest is labeled `neutral_buffer` and must not be trained as a gesture.
- Current strict-layout experiments often use the subset `{"neutral", "left_turn", "right_turn"}`.

CNN contract
- Input is filtered EMG windows shaped `(channels x time)`; no handcrafted features.
- Default windowing is 200 samples with 100-sample step unless bundle metadata overrides it.
- Standardization is per-channel z-score using train-only stats saved in the bundle.
- Bundles contain model state, normalization stats, label maps, architecture info, and metadata.
- Active trainers:
  - `train_per_subject.py`: one subject, one arm.
  - `train_cross_subject.py`: pooled subjects, one arm.
- Active realtime entrypoint: `realtime_gesture_cnn.py`.

Strict sensor placement workflow
- Strict fixed-position layout is implemented on this branch; it is no longer just a plan.
- Default trainer mode is `CHANNEL_LAYOUT_MODE = "strict"` in both `train_per_subject.py` and `train_cross_subject.py`.
- Strict mode requires fresh collection files with `metadata.emg_channel_labels`, which are saved by `CollectDataWindow.py`.
- Strict mode must use clean recollected data kept separate from old mixed-layout sessions.
- Default strict roots on this branch are `data_strict/`, `data_resampled_strict/`, and `models/strict/`.
- Strict placement is sensor-number-specific, not just sensor-type-specific: each numbered sensor must stay on the same arm and in the same physical location every session.
- Strict collection should be one arm at a time with the other arm disconnected.
- Fixed pair-to-slot contract:
  - Right arm: pairs `1, 2, 3` = Avanti, `7` = Maize, `9` = Galileo, `11` = Mini.
  - Left arm: pairs `4, 5, 6` = Avanti, `8` = Maize, `10` = Galileo.
- Strict channel counts:
  - Right arm = 17 channels.
  - Left arm = 16 channels.
- In strict mode, channels are reordered by fixed pair identity, not by Delsys pairing/stream order.
- Strict realtime should fail closed if required pairs are missing, duplicated, or mismatched.
- The strict layout helpers live in `emg/strict_layout.py`.

Legacy salvage path
- The earlier salvage approach still exists only in `train_per_subject.py` as `CHANNEL_LAYOUT_MODE = "salvage"`.
- Salvage mode uses sensor-type canonicalization plus permutation augmentation for repeated single-channel sensors.
- Salvage mode is for mixed historical layouts; it is not the preferred path for new strict recollection.

Preprocessing workflow
- Recommended order for new GUI collections:
  1. `python tools/resample_raw_dataset.py`
  2. `python emg/filtering.py`
  3. `python tools/recalibrate.py --data-root <resampled_root>` for a dry run
  4. `python tools/recalibrate.py --data-root <resampled_root> --apply` if recalibration fixes are needed
  5. Retrain
- `tools/resample_raw_dataset.py` preserves channel order and metadata while resampling each channel onto a common time grid.
- `emg/filtering.py` preserves metadata and calibration arrays when creating filtered files.
- If filter settings or layout policy changes, regenerate filtered files, rerun recalibration if needed, and retrain.

Common commands
- Collect data:
  - `python DelsysPythonGUI.py`
- Resample raw GUI collections:
  - `python tools/resample_raw_dataset.py`
- Filter resampled data:
  - `python emg/filtering.py`
- Recalibration dry run:
  - `python tools/recalibrate.py --data-root data_resampled_strict`
- Recalibration apply:
  - `python tools/recalibrate.py --data-root data_resampled_strict --apply`
- Train a per-subject CNN:
  - Edit `ARM`, `TARGET_SUBJECT`, `DATA_ROOT`, and `MODEL_OUT` in `train_per_subject.py`
  - Run `python train_per_subject.py`
- Train a cross-subject CNN:
  - Edit `ARM`, `DATA_ROOT`, and `MODEL_OUT` in `train_cross_subject.py`
  - Run `python train_cross_subject.py`
- Run live single-arm CNN inference:
  - Set `MODE` in `realtime_gesture_cnn.py` to `right` or `left`
  - Run `python realtime_gesture_cnn.py --model <bundle_path>`
- Run live dual-arm CNN inference:
  - `python realtime_gesture_cnn.py --model-right <right_bundle> --model-left <left_bundle>`
- Build evaluation tables from harvested metrics and JSON summaries:
  - `python eval_metrics/build_eval_tables.py --manifest eval_metrics/table_manifest.csv`

Evaluation workflow
- Evaluation scripts now live under `eval_metrics/`.
- Core scripts there include:
  - `harvest_model_metrics.py`
  - `realtime_behavior_metrics.py`
  - `analyze_latency.py`
  - `analyze_drive_metrics.py`
  - `build_eval_tables.py`
- The consolidated planning/tracking note is `project_notes/evaluation_execution_checklist_2026-03-15.md`.

Realtime notes
- `realtime_gesture_cnn.py` is the source of truth for live strict-layout behavior.
- `AUTO_DUAL_ARM_CHANNEL_MAPPING` does not override strict dual-arm mapping when both bundles advertise strict layout.
- Dual-arm realtime now keeps per-arm state internally and exposes:
  - `get_latest_dual_state()` for full left/right arm state plus compatibility combined state.
  - `get_latest_published_gestures()` for the testing/CARLA-facing published output contract.
  - Legacy `get_latest_gesture()` only for single-label compatibility with older callers.
- Published dual-arm output semantics:
  - If both arms agree, the published output is one gesture (`mode="single"`, arm `"dual"`).
  - If the arms differ, the published output is split (`mode="split"`) with separate left/right gestures.
- The repo copy of `carla wheel z.py` already reads `get_latest_published_gestures()` and resolves split-or-single dual-arm outputs into CARLA steering/signal/horn actions.
- `realtime_gesture_cnn.py` can now emit per-prediction timing CSVs with `--prediction-log` for evaluation work.
- The CARLA integration can now emit per-tick drive logs and forward realtime prediction logs for latency analysis.
- The current realtime file defaults are still tuned for strict 3-gesture testing; check `MODE`, `INCLUDED_GESTURES`, smoothing, confidence thresholds, and `OUTPUT_HYSTERESIS` before a run.
- `realtime_confidence_analysis.py` still uses older pair-membership logic and is not the source of truth for strict-layout validation.
- Bundles are loaded with `torch.load(weights_only=False)` for PyTorch 2.6+ compatibility.

Archived or deprecated
- The old SVM/feature pipeline is not the active workflow for this repo state.
- Archived scripts live under `code_archive/`, including:
  - `train_cnn.py`
  - `train_classifier.py`
  - `feature_extraction.py`
  - `realtime_gesture.py`
  - older filtering variants
- One-off cleanup scripts such as the relabel/drop-first-gesture utilities are historical only.
- The design rationale for strict placement is documented in `project_notes/strict_sensor_placement_branch_plan_2026-03-11.txt`, but the implementation now lives in the active scripts listed above.
