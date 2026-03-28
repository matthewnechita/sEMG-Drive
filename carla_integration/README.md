# CARLA Integration

This folder is the repo-owned home for the CARLA pieces we actually maintain.

Do not edit the external CARLA install directly unless you are intentionally testing
something temporary. Permanent changes should live here and be tracked with Git.

Current clean runtime reference

- Version: `CARLA 0.9.15`
- Clean local install used as the reference source: `C:\Users\matth\Desktop\CARLA_0.9.15`

Files in this folder

- `manual_control_emg.py`
  - Current EMG/CARLA control script copied from the repo root `carla wheel z.py`.
  - This should become the maintained CARLA control entrypoint going forward.
- `wheel_config.ini`
  - Wheel config kept next to the control script.
- `upstream/manual_control_steeringwheel.py`
  - Clean upstream CARLA example copied from the untouched install.
- `upstream/generate_traffic.py`
  - Clean upstream traffic-generation example copied from the untouched install.
- `test_start_carla_server_0_9_16.bat`
  - Restored-display test server launcher for the external CARLA install.
- `test_run_lane_keep_5min_emg_0_9_16.cmd`
  - Named lane-keep scenario wrapper.
- `test_run_highway_overtake_emg_0_9_16.cmd`
  - Named highway-overtake scenario wrapper.
- `CARLA_VERSION.txt`
  - Records the runtime version and reference install path.

Recommended workflow

1. Keep the full CARLA simulator outside this repo.
2. Treat `C:\Users\matth\Desktop\CARLA_0.9.15` as the clean runtime install.
3. Make Python-side CARLA changes in this folder, not inside the install.
4. If you need a fresh simulator elsewhere, copy or unzip the clean install.
5. Commit changes here so CARLA-side edits can be reverted normally with Git.

Suggested next cleanup

- Pick one canonical script path for future use:
  - either keep using the repo root `carla wheel z.py`
  - or switch to `carla_integration/manual_control_emg.py`
- Once the team agrees, remove the duplicate or turn one into a wrapper so the two
  copies do not drift.

Example runtime flow

1. Start CARLA with `carla_integration/test_start_carla_server_0_9_16.bat`
2. Run a named scenario wrapper or `python carla_integration\manual_control_emg.py --scenario ...`
3. For free-roam practice, use `test_run_manual_control_emg_0_9_16.bat`; it now starts with moderate ambient traffic by default unless `--scenario ...` is passed

Scenario CLI commands

Start the restored-display CARLA server:

```bat
carla_integration\test_start_carla_server_0_9_16.bat
```

Run the named lane-keep scenario:

```bat
carla_integration\test_run_lane_keep_5min_emg_0_9_16.cmd
```

Run the named overtake scenario:

```bat
carla_integration\test_run_highway_overtake_emg_0_9_16.cmd
```

Run either named scenario with evaluation logging enabled:

```bat
carla_integration\test_run_lane_keep_5min_emg_0_9_16.cmd --eval-log-dir eval_metrics\out\lane_keep_run_01
carla_integration\test_run_highway_overtake_emg_0_9_16.cmd --eval-log-dir eval_metrics\out\overtake_run_01
```

Run the maintained Python entrypoint directly:

```bat
python carla_integration\manual_control_emg.py --scenario lane_keep_5min
python carla_integration\manual_control_emg.py --scenario highway_overtake
```

Optional direct CLI examples:

```bat
python carla_integration\manual_control_emg.py --scenario lane_keep_5min --show-hud
python carla_integration\manual_control_emg.py --scenario lane_keep_5min --eval-log-dir eval_metrics\out\lane_keep_run_01
python carla_integration\manual_control_emg.py --scenario highway_overtake --eval-log-dir eval_metrics\out\overtake_run_01
python carla_integration\manual_control_emg.py --map Town03_Opt --ambient-vehicles 10 --ambient-pedestrians 18 --show-hud
```

Notes:

- Scenario presets set their own map, so `--scenario lane_keep_5min` and `--scenario highway_overtake` automatically load `Town04_Opt`.
- `lane_keep_5min` currently starts from checkpoint index `3`, so the scenario START is intentionally moved farther up the route instead of beginning at the first physical checkpoint.
- `manual_control_emg.py` now supports `--ambient-vehicles` and `--ambient-pedestrians` for integrated background traffic and walkers.
- `test_run_manual_control_emg_0_9_16.bat` applies default free-roam practice traffic only when no named `--scenario` is requested, so the evaluation wrappers keep their existing behavior.

The maps and simulator binaries are intentionally not stored in Git here.
