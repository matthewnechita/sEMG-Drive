# -*- coding: utf-8 -*-
# encoding: ascii
"""
Comprehensive Data Quality Analysis for EMG gesture dataset.
Outputs a structured DQ report covering:
  - Class balance per subject and globally
  - SNR per channel per session
  - Label quality (confidence, neutral ratio)
  - MVC/calibration data quality
  - Session-level outlier detection
  - Cross-subject consistency
  - Potential data issues
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_ROOT = Path("data")
WINDOW_SIZE = 200
WINDOW_STEP = 100

# ── helpers ───────────────────────────────────────────────────────────────────

def rms(x):
    return float(np.sqrt(np.mean(x ** 2)))

def snr_db(signal_rms, noise_rms):
    if noise_rms < 1e-12:
        return float("inf")
    return 20 * np.log10(max(signal_rms, 1e-12) / noise_rms)

def majority_label_confidence(segment):
    flat = segment.reshape(-1)
    if flat.dtype == object:
        cleaned = [str(x) for x in flat if x is not None and not isinstance(x, bytes)]
    else:
        cleaned = [str(x) for x in flat]
    if not cleaned:
        return None, 0.0
    from collections import Counter
    cnt = Counter(cleaned)
    top_label, top_count = cnt.most_common(1)[0]
    return top_label, top_count / len(cleaned)

# ── per-session analysis ───────────────────────────────────────────────────────

def analyse_session(fp):
    data = np.load(fp, allow_pickle=True)
    result = {
        "file": fp.name,
        "subject": fp.parts[-4] if len(fp.parts) >= 4 else "?",
        "issues": [],
    }

    if "emg" not in data.files:
        result["issues"].append("MISSING: emg array")
        return result
    if "y" not in data.files:
        result["issues"].append("MISSING: y labels")
        return result

    emg = np.asarray(data["emg"], dtype=np.float32)   # (T, C)
    y   = np.asarray(data["y"], dtype=object)

    n_samples, n_channels = emg.shape
    result["n_samples"] = n_samples
    result["n_channels"] = n_channels
    result["duration_s"] = n_samples / 2000.0   # Delsys @ 2 kHz

    # ── calibration ──────────────────────────────────────────────────────────
    has_calib = ("calib_neutral_emg" in data.files and "calib_mvc_emg" in data.files)
    result["has_calib"] = has_calib
    if has_calib:
        neutral = np.asarray(data["calib_neutral_emg"], dtype=np.float32)
        mvc     = np.asarray(data["calib_mvc_emg"],     dtype=np.float32)
        neutral_rms = np.sqrt(np.mean(neutral ** 2, axis=0))   # (C,)
        mvc_rms     = np.sqrt(np.mean(mvc     ** 2, axis=0))   # (C,)
        mvc_ratio   = mvc_rms / np.maximum(neutral_rms, 1e-8)
        result["neutral_rms_mean"]  = float(neutral_rms.mean())
        result["mvc_rms_mean"]      = float(mvc_rms.mean())
        result["mvc_ratio_mean"]    = float(mvc_ratio.mean())
        result["mvc_ratio_min"]     = float(mvc_ratio.min())
        result["mvc_ratio_max"]     = float(mvc_ratio.max())
        # Flag channels where MVC < 2× neutral (barely activated)
        weak_channels = int((mvc_ratio < 2.0).sum())
        result["weak_mvc_channels"] = weak_channels
        if weak_channels > 0:
            result["issues"].append(
                f"CALIB: {weak_channels}/{n_channels} channels have MVC/neutral < 2× "
                f"(possible lazy MVC contraction)"
            )
        if neutral_rms.max() > 500:
            result["issues"].append(
                f"CALIB: Very high neutral RMS ({neutral_rms.max():.1f}); possible motion artifact"
            )
    else:
        result["issues"].append("MISSING: calibration data (calib_neutral_emg / calib_mvc_emg)")

    # ── channel-level signal quality ─────────────────────────────────────────
    ch_rms = np.sqrt(np.mean(emg ** 2, axis=0))   # (C,)
    result["channel_rms_mean"] = float(ch_rms.mean())
    result["channel_rms_std"]  = float(ch_rms.std())
    result["channel_rms_min"]  = float(ch_rms.min())
    result["channel_rms_max"]  = float(ch_rms.max())

    # Dead / saturated channels
    dead_ch = int((ch_rms < 1.0).sum())          # effectively zero signal
    sat_ch  = int((ch_rms > 5000).sum())         # likely saturated
    result["dead_channels"]      = dead_ch
    result["saturated_channels"] = sat_ch
    if dead_ch > 0:
        result["issues"].append(f"CHANNEL: {dead_ch} channel(s) appear dead (RMS < 1 µV)")
    if sat_ch > 0:
        result["issues"].append(f"CHANNEL: {sat_ch} channel(s) appear saturated (RMS > 5000)")

    # Cross-channel RMS imbalance (coefficient of variation)
    cv = float(ch_rms.std() / max(ch_rms.mean(), 1e-8))
    result["channel_rms_cv"] = cv
    if cv > 2.0:
        result["issues"].append(
            f"CHANNEL: High amplitude imbalance across channels (CV={cv:.2f}); "
            "check electrode placement"
        )

    # ── label quality ─────────────────────────────────────────────────────────
    label_strs = np.array([str(v) for v in y if v is not None], dtype=object)
    if len(label_strs) == 0:
        result["issues"].append("LABELS: No valid labels found")
        return result

    from collections import Counter
    label_counts = Counter(label_strs)
    total = sum(label_counts.values())
    result["label_counts"] = dict(label_counts)

    neutral_count = label_counts.get("neutral", 0) + label_counts.get("neutral_buffer", 0)
    neutral_frac  = neutral_count / max(total, 1)
    result["neutral_fraction"] = neutral_frac
    if neutral_frac > 0.65:
        result["issues"].append(
            f"LABELS: Very high neutral fraction ({neutral_frac:.1%}); "
            "model may be biased towards neutral"
        )

    # Check label diversity
    gesture_labels = {k for k in label_counts if k not in ("neutral", "neutral_buffer")}
    result["n_gesture_classes"] = len(gesture_labels)
    if len(gesture_labels) < 3:
        result["issues"].append(
            f"LABELS: Only {len(gesture_labels)} gesture class(es) found; "
            "expected ≥ 4 for car control"
        )

    # Window count
    n_windows = max(0, (n_samples - WINDOW_SIZE) // WINDOW_STEP + 1)
    result["n_windows"] = n_windows
    if n_windows < 500:
        result["issues"].append(
            f"DATA: Only {n_windows} windows; recommend ≥ 500 for reliable training"
        )

    return result

# ── per-subject aggregation ────────────────────────────────────────────────────

def analyse_subject(subj_dir):
    files = sorted(subj_dir.rglob("*_filtered.npz"))
    if not files:
        return None
    sessions = [analyse_session(f) for f in files]
    return sessions

# ── cross-subject consistency ──────────────────────────────────────────────────

def cross_subject_consistency(all_sessions):
    """Check that all subjects have the same channel count and gesture labels."""
    ch_counts = {}
    label_sets = {}
    for s in all_sessions:
        subj = s.get("subject", "?")
        n_ch = s.get("n_channels")
        if n_ch is not None:
            ch_counts.setdefault(subj, set()).add(n_ch)
        lc = s.get("label_counts", {})
        gestures = frozenset(k for k in lc if k not in ("neutral", "neutral_buffer"))
        if gestures:
            label_sets.setdefault(subj, set()).update(gestures)
    return ch_counts, label_sets

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("EMG DATASET — DATA QUALITY REPORT")
    print("=" * 70)

    subjects = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir())
    all_sessions = []
    all_issues = []

    for subj in subjects:
        subj_dir = DATA_ROOT / subj
        sessions = analyse_subject(subj_dir)
        if sessions is None:
            print(f"\n[{subj}] No *_filtered.npz files found — skipping")
            continue

        print(f"\n{'─'*70}")
        print(f"SUBJECT: {subj}  ({len(sessions)} session(s))")
        print(f"{'─'*70}")

        subj_windows = 0
        subj_issues = []
        for s in sessions:
            all_sessions.append(s)
            n_win = s.get("n_windows", 0)
            subj_windows += n_win
            has_calib = "YES" if s.get("has_calib") else "NO "
            dead = s.get("dead_channels", 0)
            sat  = s.get("saturated_channels", 0)
            mvc_ratio = s.get("mvc_ratio_mean", float("nan"))
            neutral_frac = s.get("neutral_fraction", float("nan"))
            n_classes = s.get("n_gesture_classes", 0)
            print(
                f"  {s['file']:<55}"
                f"calib:{has_calib}  "
                f"wins:{n_win:4d}  "
                f"dead:{dead}  sat:{sat}  "
                f"mvc_ratio:{mvc_ratio:5.1f}×  "
                f"neutral:{neutral_frac:.0%}  "
                f"classes:{n_classes}"
            )
            for issue in s.get("issues", []):
                print(f"    WARN  {issue}")
                subj_issues.append(f"[{s['file']}] {issue}")

        print(f"  SUBTOTAL: {subj_windows} windows")
        all_issues.extend(subj_issues)

    # ── cross-subject consistency ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("CROSS-SUBJECT CONSISTENCY")
    print(f"{'='*70}")

    ch_counts, label_sets = cross_subject_consistency(all_sessions)
    print("\nChannel counts per subject:")
    for subj, chs in sorted(ch_counts.items()):
        flag = "" if len(chs) == 1 else "  ← INCONSISTENT WITHIN SUBJECT"
        print(f"  {subj}: {sorted(chs)}{flag}")

    all_ch = {c for chs in ch_counts.values() for c in chs}
    if len(all_ch) > 1:
        print(f"  WARN  Channel count inconsistency across subjects: {sorted(all_ch)}")
        all_issues.append(f"CROSS-SUBJECT: Channel count mismatch {sorted(all_ch)}")
    else:
        print(f"  OK  Consistent channel count: {all_ch.pop()}")

    print("\nGesture label sets per subject:")
    global_labels = None
    for subj, lbls in sorted(label_sets.items()):
        print(f"  {subj}: {sorted(lbls)}")
        if global_labels is None:
            global_labels = lbls
        elif lbls != global_labels:
            diff = lbls.symmetric_difference(global_labels)
            print(f"    WARN  Label mismatch vs others: {sorted(diff)}")
            all_issues.append(f"CROSS-SUBJECT: [{subj}] label mismatch {sorted(diff)}")
    if global_labels is not None and all(
        s.get("label_counts") and
        frozenset(k for k in s["label_counts"] if k not in ("neutral","neutral_buffer")) == global_labels
        for s in all_sessions
    ):
        print("  OK  Label sets consistent across all subjects")

    # ── global summary ─────────────────────────────────────────────────────────
    total_windows = sum(s.get("n_windows", 0) for s in all_sessions)
    total_sessions = len(all_sessions)
    sessions_with_calib = sum(1 for s in all_sessions if s.get("has_calib"))

    print(f"\n{'='*70}")
    print("GLOBAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Subjects         : {len(subjects)}")
    print(f"  Sessions total   : {total_sessions}")
    print(f"  Sessions w/ calib: {sessions_with_calib} / {total_sessions}")
    print(f"  Windows total    : {total_windows:,}")
    print(f"  Issues found     : {len(all_issues)}")

    # ── global class balance ──────────────────────────────────────────────────
    from collections import Counter
    global_labels_count = Counter()
    for s in all_sessions:
        global_labels_count.update(s.get("label_counts", {}))
    total_samples = sum(global_labels_count.values())
    print(f"\nGlobal label distribution ({total_samples:,} labeled samples):")
    for label, count in sorted(global_labels_count.items(), key=lambda x: -x[1]):
        pct = count / max(total_samples, 1)
        bar = "#" * int(pct * 40)
        flag = ""
        if label not in ("neutral", "neutral_buffer") and pct < 0.05:
            flag = "  ← UNDER-REPRESENTED"
        print(f"  {label:<25} {count:7,} ({pct:5.1%}) {bar}{flag}")

    # ── outlier sessions ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("OUTLIER SESSIONS (any ≥1 issue)")
    print(f"{'='*70}")
    sessions_with_issues = [s for s in all_sessions if s.get("issues")]
    if not sessions_with_issues:
        print("  OK  No sessions with data quality issues")
    for s in sessions_with_issues:
        print(f"\n  {s['subject']} / {s['file']}")
        for issue in s["issues"]:
            print(f"    FAIL {issue}")

    # ── DQ verdict ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("DQ VERDICT")
    print(f"{'='*70}")
    critical = [i for i in all_issues if any(
        kw in i for kw in ["MISSING: emg", "MISSING: y", "Channel count mismatch", "dead"]
    )]
    warnings = [i for i in all_issues if i not in critical]

    if critical:
        print(f"  STATUS: CRITICAL  {len(critical)} CRITICAL issue(s) — review before training")
        for c in critical:
            print(f"    FAIL {c}")
    else:
        print("  STATUS: PASS  No critical issues — data is usable for training")

    if warnings:
        print(f"  WARNINGS: {len(warnings)} non-critical warning(s)")
        for w in warnings[:10]:
            print(f"    WARN  {w}")
        if len(warnings) > 10:
            print(f"    ... and {len(warnings)-10} more")

    print()

if __name__ == "__main__":
    main()
