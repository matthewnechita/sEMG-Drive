# Data Collection

## What It Is Used For

This subsystem turns live Delsys sessions into labeled EMG recordings that can move through the maintained training pipeline. In the cleaned repo state, that means:

- collecting raw fixed-placement sessions with `DelsysPythonGUI.py`
- preserving channel-label metadata needed for downstream sensor-layout checks
- storing calibration segments that can be reused during preprocessing and training
- resampling and filtering those sessions into the maintained training root

The maintained raw root is `data_strict/`, and the maintained resampled and filtered root is `data_resampled_strict/`.

## Why It Was Made

The project is not only a classifier problem. It depends on repeatable collection conditions:

- the same arm-specific sensor pairs must stay in the same physical positions across sessions
- multiple Trigno sensor families can stream at different effective sample rates
- calibration quality can make normalization reliable or unreliable
- labels need to reflect the prompt protocol, not manual after-the-fact reconstruction

Because of that, the collection path was built to save more than just EMG arrays. It also saves timestamps, event markers, channel labels, arm and session metadata, and calibration captures so later stages can fail closed instead of guessing.

## How It Works

`DataCollector/CollectDataWindow.py` defines the scripted collection flow through `TrialConfig`. The maintained protocol for the active workflow is `standard_4g`, which records the fixed trainable gesture set:

- `left_turn`
- `right_turn`
- `neutral`
- `horn`

The GUI still contains some focus and diagnostic protocols, but the maintained training and realtime stack are fixed to the 4-gesture contract above. Rest segments are labeled `neutral_buffer` during collection and are intentionally removed before training.

Each scripted run can include:

- a preparation countdown
- a neutral calibration segment (resting baseline capture)
- an MVC calibration segment (`MVC` = maximum voluntary contraction, meaning an intentional maximum-effort squeeze)
- prompted gesture segments with configurable durations and repetitions
- optional neutral recovery or rest buffers between gesture prompts

The raw `.npz` files are compressed NumPy archives. They store:

- `X` (raw multichannel EMG sample matrix)
- `timestamps` (per-sample acquisition times for each channel)
- `y` (gesture labels aligned to the recorded samples)
- `events` (protocol milestones such as prep, calibration, and gesture-start markers)
- `metadata` (session, arm, protocol, and channel-layout details)
- optional calibration arrays for neutral and MVC captures

The saved metadata includes subject, arm, session, protocol name, gesture timing, and `metadata.emg_channel_labels`. Those channel labels are the key link between collection and the fixed sensor-placement contract enforced later by `emg/strict_layout.py`.

After collection, `emg/resample_raw_dataset.py` builds one shared time grid for every file. It uses per-channel timestamps, finds the overlapping valid time region, linearly interpolates each EMG channel onto a common grid, transfers labels onto that grid with nearest-neighbor timing, and resamples the calibration segments the same way. The maintained target rate is `2000 Hz`.

Then `emg/filtering.py` applies the maintained offline filter chain:

- notch at `60 Hz`
- notch at `120 Hz`
- bandpass `20-450 Hz`

Here, a notch filter removes a very narrow unwanted frequency band while leaving nearby signal content mostly intact. In this pipeline:

- `60 Hz` targets power-line interference, which is a common source of electrical noise in EMG recordings
- `120 Hz` targets the second harmonic of that same interference, which can still remain after the fundamental is removed

The bandpass filter keeps only a broader frequency range that is likely to contain useful EMG activity. In practical terms:

- the lower cutoff at `20 Hz` helps suppress slow baseline drift and low-frequency motion contamination
- the upper cutoff at `450 Hz` helps suppress higher-frequency noise that is less useful for the maintained EMG classifier

The same filtering step is applied to the main EMG array and to any saved calibration arrays, so training and realtime normalization operate on similarly processed signals.

## How We Validate

Validation starts at capture time:

- the GUI exposes the detected channel labels so the operator can confirm the connected sensors
- calibration quality is checked with an MVC-to-neutral ratio and a warning is shown when the contraction is too weak
- event logs and protocol metadata are stored with the session instead of being reconstructed later

Validation continues in preprocessing:

- resampling fails if the channels do not have a usable shared time window
- filtering skips files that do not have a recoverable sampling rate
- calibration arrays are carried forward explicitly instead of being silently dropped

There is also an indirect validation step later in the pipeline: training and realtime both resolve fixed channel order from `metadata.emg_channel_labels`, so a collection that does not carry the required channel metadata will not be accepted by the maintained path.

## What The Validation Is Used For

These checks are used to separate data-quality failures from model failures.

In practice, they answer five important questions before training starts:

- was the right subject and arm recorded?
- were the channels labeled and preserved correctly?
- was the contraction quality good enough for MVC normalization?
- were mixed-rate channels aligned onto one valid time base?
- does the session still carry enough metadata for fixed sensor-placement enforcement?

If those answers are not reliable, any offline metric or CARLA result becomes hard to trust. The collection and preprocessing validation exists to stop that problem early.
