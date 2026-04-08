# CARLA Control

## What It Is Used For

`carla_integration/manual_control_emg.py` is the canonical CARLA client for the maintained repo state. It is responsible for:

- launching the realtime EMG inference thread
- loading free-roam or named-scenario worlds
- converting published gestures into vehicle control actions
- logging per-tick drive data for later evaluation

In the preferred evaluation path, this client is vehicle-only and keyboard/EMG-first. The current code still retains optional steering-wheel throttle/brake support when one joystick is connected.

Here, the client is the Python program that connects the EMG system to the CARLA simulator and applies the resulting control actions inside the simulated vehicle.

## Why It Was Made

The project is evaluating control, not only recognition. That means the system needs one place where all of the following meet:

- live EMG predictions
- a stable driving-control contract
- simulator state and map loading
- scenario execution
- evaluation logging

## How It Works

The client starts by loading the requested CARLA map, creating the world wrapper, and instantiating `DualControl`. If a named scenario is selected, the scenario preset overrides the map choice automatically.

`DualControl` launches `realtime_gesture_cnn.py` in a background thread and optionally forwards a prediction log path into that realtime process. The CARLA side then polls `get_latest_published_gestures()` every control tick. A control tick is one cycle of the client loop where the current vehicle state is updated, a fresh gesture output is read, and a new control command can be applied.

The maintained gesture-to-control contract is intentionally narrow:

- the left arm drives strong steering requests
- the right arm drives weaker steering requests
- no gesture directly applies throttle or brake
- a dual-arm `horn` command toggles reverse on or off

This narrow contract is deliberate. It keeps the live gesture mapping focused on steering and a small number of discrete control actions, which makes the interface easier to test and less likely to produce unsafe or confusing multi-command behavior.

Concretely, the steering keys are:

- `left_strong`
- `right_strong`
- `left`
- `right`
- `neutral`

Those steering keys are then mapped to actual steer values through `CARLA_TUNING` in `emg/runtime_tuning.py`.

The distinction between strong and weak steering exists because the maintained dual-arm design uses the left arm for stronger turn requests and the right arm for lighter steering influence. That gives the controller a simple way to represent different steering magnitudes without introducing many gesture classes.

The control path also adds guardrails:

- stale gesture output is ignored if it is older than the configured age limit
- requested steering changes must satisfy dwell rules before being applied
- reverse toggling is edge-triggered, rate-limited, and blocked while the vehicle is moving too fast

Those terms mean:

- stale means the published gesture is too old to trust as a current user intention
- dwell means a requested steering state must remain present for enough consecutive control ticks before it is accepted
- edge-triggered means reverse toggles only when the command first appears, not on every frame while it is being held
- rate-limited means toggling cannot happen repeatedly with no delay between requests

The client still keeps a small set of keyboard controls that matter operationally:

- `Backspace` restarts the vehicle
- in free roam, `Backspace` respawns a random vehicle
- `C` advances weather
- `Shift+C` goes to the previous weather
- `F1` toggles the HUD

The client also still retains several inherited manual-control hotkeys:

- `Tab` toggles the camera view
- `P` toggles autopilot
- `Q`, `M`, `,`, and `.` control manual gearing

For visualization, the preferred evaluation view is one standard RGB camera with a simplified HUD, even though the inherited camera toggle remains available in code.

For evaluation, the client can write:

- a per-tick CARLA drive log
- a forwarded realtime prediction log

If `--eval-log-dir` is provided, the client creates both logs with matching timestamps so later analysis can pair them automatically. This is what lets the evaluation pipeline connect a published gesture prediction to the vehicle action that followed it.

## How We Validate

Control validation is built into the runtime logic and the logs.

The runtime logic validates whether a published gesture is still fresh enough to trust. If it is too old, the gesture override is dropped instead of applying stale steering.

The steering dwell logic validates whether an output is stable enough to become the applied steer key. This protects the vehicle from reacting to a single transient frame.

The reverse-toggle logic validates whether the system is in a safe state to switch gears. A dual-horn command only toggles reverse on the rising edge and only when the vehicle speed is below the configured threshold.

The per-tick drive log validates the whole control loop by recording:

- `prediction_seq`
- requested and applied steering keys
- the actual steer value
- vehicle speed and posted speed-limit context
- an approximate steering angle in radians derived from the applied steer value and the vehicle steer range
- reverse state
- lane error estimate
- lane invasion events
- current scenario snapshot fields

Here:

- lane error estimate is the vehicle's lateral offset from the lane centerline
- lane invasion events are moments where the vehicle crosses a lane marking in a way the client treats as a meaningful lane-boundary violation

That logging makes the control layer inspectable instead of opaque.

## What The Validation Is Used For

These checks are used to answer two different questions.

First, is the control contract itself safe and stable enough to drive the simulator without obvious control glitches?

Second, when performance is poor, is the problem coming from gesture recognition, from control interpretation, or from scenario difficulty?

Without dwell checks, staleness checks, and per-tick logging, those questions collapse together. With them, the repo can separate prediction behavior from applied control behavior and evaluate the EMG interface as a real control system.
