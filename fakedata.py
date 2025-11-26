import numpy as np
from pathlib import Path
import datetime as dt

# Tunables
fs = 2148  # sampling rate we are using for delsys
gestures = ["left_turn", "right_turn", "neutral"]
gesture_duration = 3.0
neutral_duration = 2.0
repetitions = 2
channel_count = 8
rng = np.random.default_rng(7)

segments = []
label_segments = []
ts_segments = []
events = []

start_wall = dt.datetime.now().timestamp()
events.append({"event": "session_start", "t_wall": start_wall})
t_cursor = 0.0

def synth_segment(label: str, duration: float, t0: float):
    samples = int(duration * fs)
    t = t0 + np.arange(samples) / fs
    
    base_noise = rng.normal(0, 0.02, size=(samples, channel_count))
    mains = np.sin(2 * np.pi * 60 * t) * (0.12 if label != "neutral" else 0.05)
    channel_scaling = rng.normal(1.0, 0.12, size=(channel_count,))
    signal = base_noise + mains[:, None] * channel_scaling
    if label != "neutral":
        envelope = np.clip(np.linspace(0, 1.0, samples), 0, 1.0)
        activation = envelope[:, None] * rng.normal(0.3, 0.06, size=(samples, channel_count))
        signal += activation
    return t, signal

for rep in range(repetitions):
    for label in gestures:
        duration = neutral_duration if label == "neutral" else gesture_duration
        events.append({"event": f"{label}_start", "t_wall": start_wall + t_cursor, "rep": rep + 1})
        t_vec, seg = synth_segment(label, duration, t_cursor)
        ts_segments.append(np.tile(t_vec[:, None], channel_count))
        label_segments.append(np.full(t_vec.shape, label, dtype=object))
        segments.append(seg)
        t_cursor = t_vec[-1] + 1 / fs

events.append({"event": "session_stop", "t_wall": start_wall + t_cursor})

X = np.vstack(segments)
timestamps = np.vstack(ts_segments)
y = np.concatenate(label_segments)

metadata = {
    "created_at": dt.datetime.now().isoformat(),
    "gestures": gestures,
    "gesture_duration_s": gesture_duration,
    "neutral_duration_s": neutral_duration,
    "repetitions": repetitions,
    "channel_count": channel_count,
    "prep_duration_s": 0.0,
    "ramp_style": "synthetic example with low-noise neutral and ramped gesture activation",
    "sampling_rate_hz": fs,
    "notes": "Synthetic Delsys-like EMG for quick testing; includes mains-like 50 Hz component.",
}

dest = Path("data") / "fake data" / "emg_subjectS00_session00_raw.npz"
dest.parent.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    dest,
    X=X,
    timestamps=timestamps,
    y=y,
    events=np.asarray(events, dtype=object),
    metadata=metadata,
    emg=X,  # convenience alias for filtering.py expectations
    fs=fs,
)

print(f"Wrote {dest} with {X.shape[0]} samples, {X.shape[1]} channels at {fs} Hz.")