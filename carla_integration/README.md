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
- `start_carla_low.bat`
  - Convenience launcher for the external CARLA install in low-quality mode.
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

1. Start CARLA with `carla_integration/start_carla_low.bat`
2. Run the maintained control script from the repo
3. Keep traffic generation as a separate process if needed

The maps and simulator binaries are intentionally not stored in Git here.
