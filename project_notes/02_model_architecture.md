# Model Architecture

## What It Is Used For

The maintained model architecture is `GestureCNNv2`. It is the per-arm classifier used by:

- `train_per_subject.py`
- `train_cross_subject.py`
- `realtime_gesture_cnn.py`

Its job is to map one fixed-placement EMG window, shaped as `channels x time`, into probabilities (confidence scores across the possible gesture classes) over the maintained gesture set.

## Why It Was Made

The project needed a model that could do three things at the same time:

- learn short-timescale temporal patterns from raw filtered EMG
- preserve the meaning of fixed channel identity across sessions
- stay small and stable enough for live inference in the CARLA workflow

Older alternate model families were removed from the maintained path because the repo was finalized around one deployable stack. `GestureCNNv2` is the architecture that remained after that cleanup because it fits the actual deployment problem: short windows, fixed sensor placement, and realtime inference rather than offline experimentation alone.

## How It Works

`emg/gesture_model_cnn.py` defines the architecture.

The model takes input data shaped like `(batch, channels, time)` and processes it in five main stages.

Here:

- `batch` means how many windows are being processed together
- `channels` means the EMG sensor channels for one arm
- `time` means the sequence of time samples inside each window

First, it computes one global energy feature from the raw input window. This is the mean squared signal energy over all channels and time samples. In simple terms, it gives the network one compact summary of how strong the muscle activity in the window is overall. That scalar is kept aside and concatenated back in near the output head.

Second, it applies `InstanceNorm1d` to the input window. This is a normalization step that rescales each input window in a consistent way before feature extraction. This matters because the maintained CNN path does not currently depend on dataset-wide mean and standard-deviation scaling during inference. The bundle still stores normalization arrays for compatibility and metadata completeness, but the active model path standardizes behavior mainly through input instance normalization plus the calibration step used before the model sees the window.

Third, it passes the normalized window through a stem convolution (the first feature-extraction layer):

- `Conv1d(in_channels -> 32, kernel_size=11, padding=5)`
- `BatchNorm1d`
- `ReLU`

Fourth, it builds increasingly richer temporal features through three convolutional stages:

1. Stage 1 keeps width `32`, applies a residual block, applies channel attention, then downsamples with max pooling.
2. Stage 2 uses a `1x1` projection from `32 -> 64`, applies another residual block plus channel attention, then downsamples again.
3. Stage 3 uses a `1x1` projection from `64 -> 128`, applies the final residual block plus channel attention, and ends with adaptive average pooling over time.

Some of those terms need a short plain-language translation:

- `Conv1d` means a one-dimensional convolution, which scans across the time axis to detect short temporal patterns in the EMG signal
- `BatchNorm1d` helps stabilize training by keeping intermediate activations in a more controlled range
- `ReLU` is a simple non-linear activation that lets the network learn more complex patterns than a purely linear model
- `MaxPool1d` downsamples the time axis, which reduces computation and lets deeper layers focus on larger-scale temporal structure
- a `1x1` projection changes the number of feature channels without changing the time length, which is useful when moving into a wider stage of the network
- adaptive average pooling collapses the remaining time dimension into one summary value per learned feature channel, so the classifier gets a fixed-size representation even after several temporal operations

The residual blocks are straightforward two-layer `Conv1d + BatchNorm + ReLU` stacks with a skip connection and dropout. Their role is to learn local temporal structure without forcing the network to relearn a clean identity path at every depth.

The channel-attention blocks are lightweight squeeze-and-excitation style modules. In simple terms, they let the model decide which channels should matter more or less for the current input. That is useful in this project because not every sensor contributes equally to every gesture even under fixed placement.

After temporal pooling, the network has a learned `128`-dimensional feature vector. The raw-window energy scalar is concatenated to that feature vector, giving a `129`-dimensional representation. A final linear layer maps that representation to the class logits (the raw class scores before they are converted into probabilities).

This design is deliberate:

- large temporal kernels (`11`) capture short bursts and activation ramps
- residual blocks stabilize training
- channel attention lets the model emphasize gesture-relevant sensors
- adaptive pooling avoids tying the head to a fragile exact intermediate time length
- the explicit energy scalar keeps a simple amplitude summary available to the classifier

## How We Validate

Architecture validation happens through the maintained training and evaluation scripts, not through a separate model-benchmark harness.

For per-subject models:

- grouped cross-validation is run over session files
- the final stored metrics come from out-of-fold predictions rather than a single lucky split

For cross-subject models:

- leave-one-subject-out evaluation is run before the pooled final fit
- the code explicitly treats LOSO as the real cross-subject validation signal

For both cases, the model is judged using the maintained offline metric set:

- balanced accuracy (average recall across classes, so one common class does not dominate the score)
- macro precision (precision averaged evenly across classes)
- macro recall (recall averaged evenly across classes)
- macro F1 (balanced precision/recall summary averaged evenly across classes)
- worst-class recall (the recall of the hardest-performing class)

The evaluation utilities also keep class-level confusion detail, including confusion into `neutral` and false-positive neutral prediction rate. That matters because a weak gesture-controlled system can look acceptable on overall accuracy while still collapsing uncertain decisions into neutral too often.

Bundle loading also validates architecture identity. `load_gesture_bundle()` rejects unsupported architecture types, so realtime cannot silently load a bundle from an obsolete model family.

## What The Validation Is Used For

The architecture validation is used for three practical decisions.

First, it decides whether the current CNN is good enough to deploy into realtime at all.

Second, it exposes the kind of failure the model is making. Balanced accuracy and macro metrics show class fairness, while worst-class recall and neutral-confusion summaries show whether the model is failing on the gestures that matter most to control.

Third, it ensures the saved bundle is actually compatible with the maintained runtime. In this repo, architecture validation is not only about model quality. It is also about making sure the trained artifact can be loaded, interpreted, and trusted by the live system.
