# CARLA Integration

This folder contains the Python-side CARLA files that are maintained in the repo. The simulator install itself stays outside the repo.

## Canonical entrypoint

- `manual_control_emg.py`
  - Maintained EMG/CARLA client.
  - Loads named scenarios, starts realtime inference, and can write evaluation logs.
  - Preferred evaluation path uses keyboard + EMG vehicle control, but the current client also retains optional steering-wheel throttle/brake support when one joystick is detected.

## Wrapper scripts

- `test_start_carla_server_0_9_16.bat`
  - Restored-display CARLA server launcher.
- `test_run_manual_control_emg_0_9_16.bat`
  - Free-roam practice launcher.
  - Defaults to `Town03_Opt` with `90` vehicles and `0` pedestrians.
- `lane_keep_5min.cmd`
  - Named lane-keep scenario wrapper.
- `highway_overtake.cmd`
  - Named highway-overtake scenario wrapper.
- `lane_keep_5min_eval.cmd`
  - Lane-keep evaluation wrapper with `--eval-log-dir`.
- `highway_overtake_eval.cmd`
  - Highway-overtake evaluation wrapper with `--eval-log-dir`.

## Named scenarios

- `lane_keep_5min`
  - Map: `Town04_Opt`
  - Ego spawns about `15 m` before the first active checkpoint.
- `highway_overtake`
  - Map: `Town04_Opt`
  - Lead vehicle reacts during passing attempts to make the overtake less static.

## Common commands

Start the CARLA server:

```bat
carla_integration\test_start_carla_server_0_9_16.bat
```

Run free-roam practice:

```bat
carla_integration\test_run_manual_control_emg_0_9_16.bat
```

Run the lane-keep scenario:

```bat
carla_integration\lane_keep_5min.cmd
```

Run the lane-keep scenario with evaluation logging:

```bat
carla_integration\lane_keep_5min_eval.cmd
```

Run the overtake scenario:

```bat
carla_integration\highway_overtake.cmd
```

Run the overtake scenario with evaluation logging:

```bat
carla_integration\highway_overtake_eval.cmd
```

Run a scenario directly:

```bat
python carla_integration\manual_control_emg.py --scenario lane_keep_5min
python carla_integration\manual_control_emg.py --scenario highway_overtake
```

Enable evaluation logging:

```bat
python carla_integration\manual_control_emg.py --scenario lane_keep_5min --eval-log-dir eval_metrics\out\lane_keep_eval
python carla_integration\manual_control_emg.py --scenario highway_overtake --eval-log-dir eval_metrics\out\highway_overtake_eval
```

The dedicated eval wrappers above apply those `--eval-log-dir` arguments automatically.

## Controls

- `Backspace`
  - Restart the vehicle.
  - In free roam, this respawns a random vehicle.
- `C`
  - Next weather preset.
- `Shift+C`
  - Previous weather preset.
- `F1`
  - Toggle the HUD.
- `Tab`
  - Toggle the inherited camera view.
- `P`
  - Toggle autopilot.
- `Q`, `M`, `,`, `.`
  - Retained manual-gear hotkeys inherited from the base manual-control client.

## Notes

- For evaluation runs, prefer the named scenario wrappers and eval wrappers instead of typing long CLI commands manually.
- Named scenario presets choose their own maps.
- `manual_control_emg.py` also supports direct free-roam runs with `--map`, `--ambient-vehicles`, and `--ambient-pedestrians`.
- The preferred evaluation view is the standard RGB camera, but the inherited camera toggle is still available.
