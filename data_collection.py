import argparse
import datetime as dt
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from AeroPy.TrignoBase import TrignoBase
from AeroPy.DataManager import DataKernel


@dataclass
class TrialConfig:
    gestures: List[str]
    gesture_duration: float
    neutral_duration: float
    repetitions: int
    prep_duration: float = 3.0


class _CollectionHandler:
    """
    Minimal stub object so TrignoBase.ConfigureCollectionOutput can populate
    channel GUIDs. We keep this tiny and run the polling loop ourselves.
    """

    def __init__(self, stream_yt_data: bool):
        self.streamYTData = stream_yt_data
        self.pauseFlag = False
        self.DataHandler: DataKernel | None = None
        self.EMGplot = None  # not used here

    # TrignoBase.Start_Callback expects this symbol; keep it no-op.
    def threadManager(self, start_trigger: bool, stop_trigger: bool) -> None:
        return


def _pair_time_value(sample) -> Tuple[float, float]:
    """Handles both .Item1/.Item2 tuples and native (t, y) pairs."""
    if hasattr(sample, "Item1"):
        return float(sample.Item1), float(sample.Item2)
    return float(sample[0]), float(sample[1])


def collect_segment(
    kernel: DataKernel,
    label: str,
    duration_s: float,
    channel_count: int,
    stop_flag: Optional[Callable[[], bool]] = None,
):
    """
    Poll the Trigno data queue for a fixed duration, return timestamp/value arrays.
    """
    ts_buffer: List[List[float]] = []
    x_buffer: List[List[float]] = []
    labels: List[str] = []
    end_time = time.time() + duration_s

    while time.time() < end_time:
        if stop_flag and stop_flag():
            break
        out = kernel.GetYTData()
        if out is None:
            time.sleep(0.001)
            continue

        channel_times = []
        channel_values = []
        for channel in out:
            if not channel:
                continue
            chan_array = np.asarray(channel[0], dtype=object)
            if chan_array.size == 0:
                continue
            times, values = zip(*(_pair_time_value(s) for s in chan_array))
            channel_times.append(list(times))
            channel_values.append(list(values))

        if len(channel_times) < channel_count:
            # If a channel came back empty this tick, skip this packet to keep alignment.
            continue

        sample_count = min(len(c) for c in channel_values)
        if sample_count == 0:
            continue

        for idx in range(sample_count):
            ts_buffer.append([channel_times[ch][idx] for ch in range(channel_count)])
            x_buffer.append([channel_values[ch][idx] for ch in range(channel_count)])
            labels.append(label)

    return ts_buffer, x_buffer, labels


def run_protocol(config: TrialConfig, output_path: Path):
    # Wire up the AeroPy base with the minimal handler.
    handler = _CollectionHandler(stream_yt_data=True)
    base = TrignoBase(handler)
    handler.DataHandler = DataKernel(base)

    base.Connect_Callback()
    sensors = base.Scan_Callback()
    if not sensors:
        raise RuntimeError("No Trigno sensors found during scan.")

    base.start_trigger = False
    base.stop_trigger = False
    configured = base.ConfigureCollectionOutput()
    if not configured:
        raise RuntimeError("Failed to configure Trigno pipeline.")

    channel_count = base.channelcount
    print(f"Configured {channel_count} channels.")

    base.TrigBase.Start(handler.streamYTData)
    all_ts: List[List[float]] = []
    all_x: List[List[float]] = []
    all_labels: List[str] = []
    events = []
    aborted = False

    start_wall = time.time()
    events.append({"event": "session_start", "t_wall": start_wall})

    # Initial prep buffer so the subject can get ready before the first gesture.
    if config.prep_duration > 0:
        print(f"Prep: waiting {config.prep_duration:.1f}s before starting.")
        time.sleep(config.prep_duration)

    try:
        for rep in range(config.repetitions):
            for idx, gesture in enumerate(config.gestures):
                if idx + 1 < len(config.gestures):
                    next_gesture = config.gestures[idx + 1]
                elif rep + 1 < config.repetitions:
                    next_gesture = config.gestures[0]
                else:
                    next_gesture = "done"
                duration = config.neutral_duration if gesture == "neutral" else config.gesture_duration
                print(f"Rep {rep + 1}/{config.repetitions}: {gesture} (next: {next_gesture}) for {duration:.1f}s")
                events.append(
                    {
                        "event": f"{gesture}_start",
                        "t_wall": time.time(),
                        "rep": rep + 1,
                    }
                )
                seg_ts, seg_x, seg_labels = collect_segment(
                    handler.DataHandler,
                    gesture,
                    duration,
                    channel_count,
                )
                all_ts.extend(seg_ts)
                all_x.extend(seg_x)
                all_labels.extend(seg_labels)
    except KeyboardInterrupt:
        aborted = True
        events.append({"event": "session_abort", "t_wall": time.time()})
        print("Aborted early; no file will be written.")
    finally:
        base.Stop_Callback()
        events.append({"event": "session_stop", "t_wall": time.time()})

    X = np.asarray(all_x, dtype=float)
    timestamps = np.asarray(all_ts, dtype=float)
    y = np.asarray(all_labels, dtype=object)

    metadata = {
        "created_at": dt.datetime.now().isoformat(),
        "gestures": config.gestures,
        "gesture_duration_s": config.gesture_duration,
        "neutral_duration_s": config.neutral_duration,
        "repetitions": config.repetitions,
        "channel_count": channel_count,
        "prep_duration_s": config.prep_duration,
        "ramp_style": "ramp contractions (longer window for non-neutral gestures)",
    }

    if aborted:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=X,
        timestamps=timestamps,
        y=y,
        events=np.asarray(events, dtype=object),
        metadata=metadata,
    )
    print(f"Saved {X.shape[0]} samples to {output_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Collect labeled EMG data from Delsys Trigno.")
    parser.add_argument("--subject", required=True, help="Subject ID, e.g. S01")
    parser.add_argument("--session", required=True, help="Session ID, e.g. 01")
    parser.add_argument(
        "--gestures",
        nargs="+",
        default=["left_turn", "right_turn", "neutral"],
        help="List of gesture labels to present per repetition.",
    )
    parser.add_argument(
        "--gesture-duration",
        type=float,
        default=5.0,
        help="Seconds per non-neutral gesture window (longer to allow ramp contractions).",
    )
    parser.add_argument(
        "--neutral-duration",
        type=float,
        default=3.0,
        help="Seconds per neutral window.",
    )
    parser.add_argument(
        "--prep-duration",
        type=float,
        default=3.0,
        help="Seconds of prep time before the first gesture.",
    )
    parser.add_argument("--repetitions", type=int, default=10, help="Repetitions per gesture.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory to store the .npz session file.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    output_name = f"emg_subject{args.subject}_session{args.session}.npz"
    output_path = args.output_dir / output_name
    config = TrialConfig(
        gestures=args.gestures,
        gesture_duration=args.gesture_duration,
        neutral_duration=args.neutral_duration,
        repetitions=args.repetitions,
        prep_duration=args.prep_duration,
    )
    run_protocol(config, output_path)


if __name__ == "__main__":
    main(sys.argv[1:])