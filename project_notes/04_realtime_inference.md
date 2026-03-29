# Realtime Inference

## What It Is Used For

`realtime_gesture_cnn.py` is the maintained live inference path for the project. It:

- loads the active right-arm and left-arm CNN bundles
- reads live Delsys streams
- resamples and filters incoming EMG in realtime
- performs dual-arm inference
- publishes the latest output through `get_latest_published_gestures()`

This is the source of truth for live gesture behavior in the cleaned repo state.

## Why It Was Made

Offline classification alone does not answer the project question. The system has to work as a continuous control loop:

- sensor order may vary at scan time even when physical placement is fixed
- mixed sensor families still need a common time base
- realtime filtering must stay close to the offline preprocessing contract
- the output has to be stable enough to drive CARLA without constant flicker (rapid one-frame label changes)

The realtime subsystem exists to enforce those constraints while still producing low-latency gesture output that downstream control code can trust.

## How It Works

The maintained realtime path is dual-arm only. It loads one bundle for the right arm and one for the left arm, and both bundles must carry strict channel-layout metadata. If a bundle is missing that metadata, the script stops instead of guessing.

After connecting to Delsys, the script reads `emgChannelNames` from the live stream and resolves arm-specific channel indices through `emg/strict_layout.py`. This is the key fixed-placement step at runtime: the code no longer trusts scan order. It trusts pair identity.

Before continuous inference begins, the script can run a live calibration sequence for each arm:

- neutral rest capture
- MVC capture
- MVC quality check

If the MVC-to-neutral ratio is too weak, normalization is disabled for that arm rather than applying unreliable scaling.

During streaming, the realtime path does four signal-processing steps:

1. timestamp resampling to a shared fixed rate
   Because the Delsys setup can include different sensor types, not all channels are guaranteed to arrive at exactly the same effective sampling rate or timestamp spacing. Resampling maps every channel onto one common time grid so that each row of the EMG matrix represents the same moment in time across all channels. This matters because filtering, windowing, and CNN inference all assume temporally aligned multichannel input. Without resampling, the model would see time-skewed channel data and the live pipeline would no longer match the training data format.

2. causal (realtime-safe) notch and bandpass filtering
   Here, causal means the filter only uses the current sample and past samples, not future samples. That matters in live inference because future samples do not exist yet. In other words, the filter is designed so it can run continuously on a stream instead of needing the whole signal in advance.

3. optional MVC normalization
   The neutral and MVC calibrations are used to make EMG amplitudes more comparable across sessions, arms, and users. The neutral calibration gives a per-channel resting baseline, and the MVC calibration gives a per-channel high-effort reference level. The pipeline then uses those values to scale the incoming signal so the model is less sensitive to differences caused by skin contact, placement pressure, or natural strength differences. The MVC quality check also prevents unreliable normalization by disabling it when the contraction is too weak.

4. windowing into `200`-sample windows with `100`-sample steps
   This means the live EMG stream is cut into short overlapping segments before each prediction. A `200`-sample window is one chunk of data given to the CNN, and a `100`-sample step means the next chunk starts `100` samples later, so consecutive windows overlap by `50%`. At the maintained `2000 Hz` target rate, that corresponds to a `100 ms` window with a new inference every `50 ms`. The overlap gives the model enough recent temporal context while still updating often enough for responsive control.

The filter design intentionally mirrors the maintained offline filtering chain as closely as possible:

- notch `60 Hz`
- notch `120 Hz`
- bandpass `20-450 Hz`

The reason for using those same cutoff frequencies is the same as in the offline preprocessing chain described in the data collection note: the realtime path is trying to match the signal conditions the model was trained on as closely as possible.

Each arm is inferred independently first. The script:

- restricts probabilities to the active gesture subset (keeps only the maintained labels active at runtime)
- optionally smooths probabilities over recent windows (reduces rapid frame-to-frame swings)
- applies a confidence gate (weak predictions fall back to `neutral`)
- can optionally apply softmax-margin rejection (rejects uncertain close-call predictions)
- can optionally apply output hysteresis (requires more consistent evidence before switching labels)

The current runtime tuning values live in `emg/runtime_tuning.py`, which is the maintained source of truth for thresholds, smoothing, and CARLA-side dwell values.

After per-arm decoding, the script fuses the two arm outputs. The fusion logic is simple and intentional:

- if both arms agree and the agreement confidence is strong enough, publish one shared output
- if they disagree, let the stronger confident active label win
- if neither arm has a confident active label, fall back to confident neutral

Published output is stored as `PublishedGestureOutput`, which can be:

- `single`, when both arms resolve to one published gesture
- `split`, when the right and left arms are published separately

That published representation is what CARLA consumes. The practical reason for keeping both modes is that some moments are best represented as one shared command, while others are better represented as separate left-arm and right-arm states.

## How We Validate

Realtime validation is mostly contract enforcement plus logging.

Contract enforcement includes:

- both bundles must use the maintained strict layout
- right and left bundles must agree on target sampling rate
- every live stream channel must have a label
- fixed pair order and expected channel counts must match the bundle metadata

Calibration validation includes:

- explicit neutral and MVC collection
- per-arm MVC quality checks
- disabling normalization when calibration quality is weak

Behavior validation includes:

- confidence gating to neutral for weak predictions
- optional smoothing and hysteresis for output stability
- optional probability rejection when the softmax margin is too small

Timing and state validation comes from the prediction logger. When `--prediction-log` is enabled, each published prediction records:

- `prediction_seq` (the shared identifier used to join prediction rows to CARLA control rows later)
- `window_end_ts` (when the input window ended)
- `prediction_ts` (when inference finished)
- `publish_ts` (when the result was made available to downstream code)
- combined and per-arm labels
- combined and per-arm confidences

## What The Validation Is Used For

These checks are used to make sure realtime output is both correct enough and structured enough for downstream use.

The layout and calibration checks protect the model from being fed the wrong channels or badly scaled data.

The confidence, smoothing, and hysteresis checks protect the control loop from unstable one-frame decisions.

The prediction logging protects evaluation. It creates the timing record that later gets joined with CARLA control logs for end-to-end latency analysis, and it makes it possible to trace a control action back to the exact published gesture state that produced it.
