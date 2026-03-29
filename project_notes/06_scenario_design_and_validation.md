# Scenario Design And Validation

## What It Is Used For

The maintained scenario subsystem defines the structured CARLA tasks used to evaluate whether the EMG control interface can support simple driving behaviors beyond free roam.

The active named scenarios are:

- `lane_keep_5min`
- `highway_overtake`

Scenario presets live in `carla_integration/scenario_presets.py`, and runtime execution lives inside `ScenarioRuntime` in `carla_integration/manual_control_emg.py`.

In simple terms, the preset defines the scenario configuration and the runtime enforces that configuration while the scenario is being driven.

## Why It Was Made

Free-roam driving is useful for practice, but it is a weak evaluation target. It does not give a clean success condition, and it does not isolate a specific control challenge.

The scenario subsystem exists to provide:

- repeatable routes
- explicit start and finish structure
- known task intent
- scenario-specific success and failure logic
- logs that can be compared across runs and participants

That is what turns CARLA from a demo environment into an evaluation environment.

## How It Works

Each `ScenarioPreset` defines the task configuration, including:

- map name
- ego spawn configuration
- route length
- checkpoint spacing and radius
- start checkpoint index
- timeout behavior
- lead-vehicle behavior for overtake scenarios
- optional debug markers

You can think of a scenario preset as a compact recipe for one evaluation task: where it happens, how long it is, where checkpoints go, and what counts as success.

At runtime, `ScenarioRuntime` builds a route starting from an anchor waypoint. A waypoint is a road-position reference supplied by CARLA. The runtime extends that route forward by choosing the next waypoint that minimizes unwanted road, lane, and yaw changes, which keeps the scenario aligned with the intended road segment instead of drifting onto an arbitrary branch.

Checkpoints are then derived from either:

- explicit checkpoint locations
- explicit route-progress markers
- evenly spaced progress intervals along the generated route

The ego vehicle does not start the scenario timer immediately at spawn. It starts when the car reaches the designated start checkpoint radius. That makes the measured scenario time reflect task execution instead of spawn overhead.

That detail matters because a scenario should measure driving performance, not setup delay.

### Lane keep

`lane_keep_5min` is a long structured route on `Town04_Opt`.

Its preset intentionally:

- starts the ego vehicle about `15 m` before the first active checkpoint
- skips early route markers by using `start_checkpoint_index = 3`
- spreads progress markers over a long route (`2600 m`)

The success condition is simple: reach the final checkpoint. This makes the scenario useful for measuring sustained lane-centering performance over time rather than only a single maneuver.

### Highway overtake

`highway_overtake` is a shorter route on `Town04_Opt` with a spawned lead vehicle.

Its preset includes:

- a fixed lead spawn distance ahead of the ego vehicle
- a hold-until-start option so the lead does not begin moving before the scenario starts
- a reactive speed-response policy during passing attempts
- an overtake-finish margin
- a requirement to return to the start lane before the overtake counts as complete

The reactive lead-vehicle behavior is important because it makes the scenario less static. Instead of passing a fully predictable obstacle, the user has to complete the maneuver while the lead vehicle responds in a limited but realistic way.

The overtake objective is not satisfied by merely reaching the finish. The ego vehicle must first progress far enough ahead of the lead vehicle and be back in the target lane. Only then does the runtime mark the overtake objective as complete and allow the final finish gate to count as success.

The return-to-lane requirement exists because a pass is not really complete until the vehicle has safely re-established itself in the intended lane rather than only drawing alongside or crossing the finish area in the passing lane.

If the vehicle reaches the finish without satisfying that overtake condition, the scenario fails with `finish_without_overtake`.

## How We Validate

Scenario validation happens in both the simulator and the logs.

Inside CARLA, the runtime draws checkpoint markers and active route-guide segments. The HUD also exposes scenario status, checkpoint progress, and overtake state so the operator can see whether the task logic is behaving as intended.

These visual aids are not just cosmetic. They help verify that the route, checkpoint spacing, and success logic match the intended scenario design during testing.

At runtime, the scenario records explicit state such as:

- scenario status
- elapsed and completion time
- checkpoint index and checkpoint count
- route progress
- overtake-objective status
- lead distance and lead gap
- whether the lead response logic is active

This state is what later lets the drive log describe not just where the vehicle went, but what the scenario believed was happening at each stage.

Success and failure are validated by explicit rules, not by visual guesswork:

- lane keep succeeds when the final checkpoint is reached
- overtake succeeds only after the lead has been passed by the configured margin and the ego vehicle has returned to the start lane
- setup failures, missing lead vehicles, and timeouts produce explicit failure reasons

The overtake scenario can also support `command_success_rate`, but only if the required `command_correct` field was explicitly logged into the drive data.

That metric is treated as optional because it depends on extra logging support. The maintained evaluation path should not fabricate it when the required evidence is missing.

## What The Validation Is Used For

Scenario validation is used to make sure the driving task actually measures what it claims to measure.

For lane keeping, that means separating long-horizon steering quality from simple route completion.

For overtaking, that means separating true maneuver completion from a shallow success like touching the finish area without really performing the pass.

The checkpoint logic, status fields, and failure reasons are what make the CARLA metrics interpretable later. Without them, lane error, completion time, and success rate would not clearly correspond to the intended driving task.
