# Training And Validation

## What It Is Used For

This subsystem converts filtered fixed-placement EMG sessions into deployable model bundles (saved model packages) and the offline metrics used to judge them.

The maintained training entrypoints are:

- `train_per_subject.py`
- `train_cross_subject.py`

Both scripts train `GestureCNNv2`, save bundle metadata under `models/strict/`, and attach the information that realtime needs later:

- label maps (the mapping between gesture names and model output indices)
- architecture metadata (the saved description of the model structure)
- training metadata (the saved description of how the model was trained)
- stored metrics
- channel-layout metadata

## Why It Was Made

The project needs two different deployment views:

- a per-subject model for best-case personalized control
- a cross-subject model for generalization to unseen participants

Those are not the same validation problem. A subject-specific model must avoid leakage between sessions from the same person. A cross-subject model must avoid over-crediting performance that only comes from seeing similar windows from the same people in both train and test splits.

Here, leakage means hidden overlap between training data and test data that makes performance look better than it really is.

The training subsystem was built around that distinction. It enforces fixed sensor placement, converts sample-level labels into window-level labels, and uses different split logic for personalized and generalized models.

## How It Works

Window construction lives in `emg/training_data.py`.

For every filtered file, the loader:

- resolves the required channel order from `metadata.emg_channel_labels`
- checks that the live file matches the expected pair order and channel counts for the selected arm
- optionally applies per-session calibration normalization using the session's neutral and MVC captures
- windows the signal with `WINDOW_SIZE = 200` and `WINDOW_STEP = 100` (the practical meaning of this windowing choice is explained in the realtime inference note)
- assigns one label per window through majority vote (whichever label occupies most of that window)
- drops windows that fail the minimum label-confidence rule
- drops `neutral_buffer` because it is not a trainable gesture label
- optionally restricts the dataset to `{"neutral", "left_turn", "right_turn", "horn"}`

The current minimum label-confidence threshold is `0.85`, which means a window is kept only when the prompt label dominates that window strongly enough to be trustworthy.

The shared CNN helpers in `emg/cnn_training.py` keep the training path consistent across both entrypoints. The active bundle metadata identifies the model family as `GestureCNNv2`, while the actual runtime path relies on the architecture definition in `emg/gesture_model_cnn.py`.

Training behavior is similar across both scripts:

- Adam optimizer
- `ReduceLROnPlateau` learning-rate scheduling (automatically lowers the learning rate when validation progress stalls)
- label smoothing (reduces overconfident fitting to the training labels)
- GPU-native augmentation (synthetic training variation generated on the GPU during training)

The augmentation policy includes:

- amplitude scaling (slightly changing signal strength)
- additive noise (injecting small random perturbations)
- temporal shift (sliding the pattern slightly earlier or later in time)
- channel dropout (temporarily blanking one channel during training)
- temporal stretch (slightly compressing or expanding the pattern in time)

The per-subject and cross-subject scripts use different augmentation strength because inter-subject amplitude variation is a larger problem in the cross-subject case.

Per-subject training uses session-grouped validation. `train_per_subject.py` runs `StratifiedGroupKFold` over session files so that windows from the same file do not leak across folds. Here, stratified means the script tries to keep the class mix reasonably balanced across folds, and grouped means all windows from one session file stay together in either train or test for that fold. The final saved metrics come from out-of-fold predictions, meaning each stored prediction was made only when that window belonged to the held-out split rather than the training split. After that, the script fits one final model on all windows for a number of epochs chosen from the median best epoch across folds.

Cross-subject training uses two validation layers. `train_cross_subject.py` first runs leave-one-subject-out evaluation when `LOSO_EVAL` is enabled. This means one participant is held out entirely, the model is trained on the others, and then it is tested on the unseen participant. That is the real zero-shot cross-subject test, where zero-shot means the model has not seen any training windows from the person it is being evaluated on. After that, if final training is enabled, it trains one pooled cross-subject model and records an additional grouped holdout split for bundle metadata. During cross-subject training, subject-balanced sampling is always on so subjects with more windows do not dominate the optimizer.

## How We Validate

Validation is built into the training scripts rather than bolted on afterward.

Fixed sensor-placement validation happens first:

- every file must resolve to the correct arm-specific pair order
- every file must have the expected channel count
- inconsistent pair mappings across files raise an error

Calibration validation also happens before model fitting:

- missing calibration arrays are reported
- weak MVC-to-neutral ratios disable normalization for that session instead of forcing bad scaling

Then the offline model validation begins.

Per-subject validation is session-grouped and out-of-fold, which answers: can the same participant reproduce gestures across separate sessions?

Cross-subject validation uses LOSO (`leave-one-subject-out`), which answers: can the model generalize to a participant whose data it has never seen during training?

The metrics are computed through `emg/eval_utils.py`, which records:

- overall accuracy
- balanced accuracy
- macro precision
- macro recall
- macro F1
- worst-class recall
- confusion matrices
- confusion-to-neutral rates
- neutral false-positive rate

The cross-subject training script also includes a deployment warning when mean LOSO accuracy falls below `65%`, because performance below that level suggests the model may not generalize reliably to unseen users.

## What The Validation Is Used For

The training validation is used for different decisions at different scopes.

For per-subject models, it tells us whether a personalized model is stable enough across separate sessions to trust in live control.

For cross-subject models, LOSO tells us whether the system is actually learning subject-transferable gesture structure or only memorizing the people already in the dataset.

At the bundle level, the stored metadata makes later auditing possible. When a model is used in realtime or in a report table, the repo can still recover:

- what labels were included
- how windows were built
- whether calibration was used
- what fixed sensor layout the model expects
- what offline metrics justified keeping that bundle

That validation record is what makes the trained artifact useful as a maintained system component instead of only a temporary experiment output.
