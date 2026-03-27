import argparse
import csv
import math
import re
from pathlib import Path


def _load_rows(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _infer_model_family(row):
    family = str(row.get("model_family") or "").strip().lower()
    if family:
        return family
    joined = " ".join(
        str(row.get(key) or "").strip().lower()
        for key in ("filename", "path")
    )
    if "tcn" in joined:
        return "metric_tcn"
    if "cnn" in joined or "v6" in joined:
        return "cnn_v2"
    return "unknown"


def _row_matches_tokens(row, include_tokens, exclude_tokens):
    haystack = " ".join(
        str(row.get(key) or "").strip().lower()
        for key in ("filename", "path", "bundle_scope", "arm", "subject", "gesture_bucket")
    )
    if include_tokens and not all(token in haystack for token in include_tokens):
        return False
    if exclude_tokens and any(token in haystack for token in exclude_tokens):
        return False
    return True


def _scope_order(value):
    text = str(value or "").strip().lower()
    if text == "cross_subject":
        return 0
    if text == "per_subject":
        return 1
    return 2


def _arm_order(value):
    text = str(value or "").strip().lower()
    if text == "left":
        return 0
    if text == "right":
        return 1
    return 2


def _label_for_row(row):
    scope = str(row.get("bundle_scope") or "").strip().replace("_", " ")
    arm = str(row.get("arm") or "").strip().title() or "Unknown arm"
    subject = str(row.get("subject") or "").strip()
    if subject:
        return f"{arm}\n{scope}\n{subject}"
    return f"{arm}\n{scope}"


def _compact_label_for_row(row):
    arm = str(row.get("arm") or "").strip().title() or "Unknown arm"
    return arm


def _latest_rows(rows):
    latest = {}
    for row in rows:
        key = (
            str(row.get("bundle_scope") or "").strip().lower(),
            str(row.get("arm") or "").strip().lower(),
            str(row.get("subject") or "").strip().lower(),
            str(row.get("gesture_bucket") or "").strip().lower(),
            _infer_model_family(row),
        )
        created_at = str(row.get("created_at") or "").strip()
        current = latest.get(key)
        if current is None or created_at > str(current.get("created_at") or "").strip():
            latest[key] = row
    return list(latest.values())


def _parse_cross_subject_breakdown(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    per_arm = {"left": [], "right": []}
    current_arm = None
    in_breakdown = False
    legacy_header_re = re.compile(r"^\s*cross subject\s+([a-z]+)\s*:\s*$", re.IGNORECASE)
    training_header_re = re.compile(
        r"^\s*Training\s+([a-z]+)\s+arm\s+cross-subject\s+model\b",
        re.IGNORECASE,
    )
    row_re = re.compile(
        r"^\s*([A-Za-z0-9_]+)\s*:\s*([0-9]*\.?[0-9]+)\s*\(\s*(\d+)\s+windows\s*\)\s*$"
    )

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        legacy_header_match = legacy_header_re.match(line)
        training_header_match = training_header_re.match(line)
        header_match = legacy_header_match or training_header_match
        if header_match:
            current_arm = header_match.group(1).strip().lower()
            in_breakdown = False
            continue

        if line.strip() == "Per-subject breakdown:":
            in_breakdown = True
            continue

        if in_breakdown:
            row_match = row_re.match(line)
            if row_match and current_arm in per_arm:
                per_arm[current_arm].append(
                    {
                        "subject": row_match.group(1).strip(),
                        "accuracy": float(row_match.group(2)),
                        "windows": int(row_match.group(3)),
                    }
                )
                continue

            if line.strip().startswith("Saved to ") or not line.strip():
                in_breakdown = False

    for arm in per_arm:
        per_arm[arm].sort(key=lambda item: (-float(item["accuracy"]), str(item["subject"]).lower()))
    return per_arm


def _find_cross_subject_subject(rankings, arm, subject_name):
    candidates = rankings.get(arm) or []
    target = str(subject_name or "").strip().lower()
    for item in candidates:
        if str(item.get("subject") or "").strip().lower() == target:
            return item
    return None


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot offline accuracy bars from harvested model metrics."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("eval_metrics") / "out" / "current_metrics" / "model_metrics.csv",
        help="CSV from harvest_model_metrics.py",
    )
    parser.add_argument(
        "--metric",
        default="test_accuracy",
        choices=["test_accuracy", "balanced_accuracy"],
        help="Metric column to plot.",
    )
    parser.add_argument(
        "--gesture-bucket",
        default="",
        help="Optional gesture bucket filter such as 4_gesture.",
    )
    parser.add_argument(
        "--model-family",
        default="",
        help="Optional model family filter such as cnn_v2 or metric_tcn.",
    )
    parser.add_argument(
        "--include-token",
        action="append",
        default=[],
        help="Optional token that must appear in the filename/path metadata. Repeatable.",
    )
    parser.add_argument(
        "--exclude-token",
        action="append",
        default=[],
        help="Optional token to exclude from the filename/path metadata. Repeatable.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("eval_metrics") / "out" / "current_metrics" / "model_accuracy_bars.png",
        help="Path to save the output PNG.",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Optional custom plot title.",
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Keep only the latest row per arm/scope/subject/gesture bucket/model family.",
    )
    parser.add_argument(
        "--compact-labels",
        action="store_true",
        help="Use generic x-axis labels without subject names.",
    )
    parser.add_argument(
        "--cross-subject-rank",
        type=int,
        default=0,
        help="If > 0, replace each cross-subject aggregate bar with the Nth-best subject from the training results note.",
    )
    parser.add_argument(
        "--cross-subject-subject",
        default="",
        help="If set, replace each cross-subject aggregate bar with the named held-out subject from the training results note.",
    )
    parser.add_argument(
        "--training-results-txt",
        type=Path,
        default=Path("project_notes") / "model_training_results.txt",
        help="Training results note used to recover per-subject cross-subject rankings.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    rows = _load_rows(args.input_csv)
    if not rows:
        raise ValueError(f"No rows found in {args.input_csv}")

    include_tokens = [str(token).strip().lower() for token in args.include_token if str(token).strip()]
    exclude_tokens = [str(token).strip().lower() for token in args.exclude_token if str(token).strip()]
    target_family = str(args.model_family or "").strip().lower()
    target_bucket = str(args.gesture_bucket or "").strip().lower()

    filtered = []
    for row in rows:
        metric_value = _to_float(row.get(args.metric))
        if metric_value is None:
            continue
        if target_bucket and str(row.get("gesture_bucket") or "").strip().lower() != target_bucket:
            continue
        if target_family and _infer_model_family(row) != target_family:
            continue
        if not _row_matches_tokens(row, include_tokens, exclude_tokens):
            continue
        filtered.append(row)

    if not filtered:
        raise ValueError("No model rows matched the requested filters.")

    if args.latest_only:
        filtered = _latest_rows(filtered)

    cross_subject_rankings = None
    cross_subject_subject = str(args.cross_subject_subject or "").strip()
    if int(args.cross_subject_rank) > 0 and cross_subject_subject:
        raise ValueError("Use only one of --cross-subject-rank or --cross-subject-subject.")
    if int(args.cross_subject_rank) > 0 or cross_subject_subject:
        if args.metric != "test_accuracy":
            raise ValueError("Cross-subject subject/rank substitution currently supports --metric test_accuracy only.")
        cross_subject_rankings = _parse_cross_subject_breakdown(args.training_results_txt)

    filtered.sort(
        key=lambda row: (
            _arm_order(row.get("arm")),
            _scope_order(row.get("bundle_scope")),
            str(row.get("subject") or "").strip().lower(),
            str(row.get("created_at") or "").strip().lower(),
        )
    )

    plot_rows = []
    for row in filtered:
        scope = str(row.get("bundle_scope") or "").strip().lower()
        arm = str(row.get("arm") or "").strip().lower()
        if scope == "cross_subject" and cross_subject_rankings is not None:
            if cross_subject_subject:
                ranked_row = _find_cross_subject_subject(cross_subject_rankings, arm, cross_subject_subject)
                if ranked_row is None:
                    available = ", ".join(item["subject"] for item in (cross_subject_rankings.get(arm) or []))
                    raise ValueError(
                        f"Subject {cross_subject_subject!r} was not found for arm {arm!r} in "
                        f"{args.training_results_txt}. Available: {available or 'none'}."
                    )
            else:
                ranked_rows = cross_subject_rankings.get(arm) or []
                rank_index = int(args.cross_subject_rank) - 1
                if rank_index < 0 or rank_index >= len(ranked_rows):
                    raise ValueError(
                        f"Requested cross-subject rank {args.cross_subject_rank} for arm {arm!r}, "
                        f"but only found {len(ranked_rows)} subject result(s) in {args.training_results_txt}."
                    )
                ranked_row = ranked_rows[rank_index]
            plot_rows.append(
                {
                    "scope": scope,
                    "arm": arm,
                    "value": float(ranked_row["accuracy"]) * 100.0,
                }
            )
            continue

        plot_rows.append(
            {
                "scope": scope,
                "arm": arm,
                "value": _to_float(row.get(args.metric)) * 100.0,
            }
        )

    value_by_scope_arm = {}
    for item in plot_rows:
        key = (item["scope"], item["arm"])
        if key in value_by_scope_arm:
            raise ValueError(
                f"Multiple rows matched scope={item['scope']!r}, arm={item['arm']!r}. "
                "Use --latest-only or add filters so each scope/arm pair is unique."
            )
        value_by_scope_arm[key] = float(item["value"])

    scope_keys = sorted({item["scope"] for item in plot_rows}, key=_scope_order)
    arm_keys = sorted({item["arm"] for item in plot_rows}, key=_arm_order)
    scope_labels = {
        "cross_subject": "Cross-subject",
        "per_subject": "Per-subject",
    }
    arm_labels = {
        "left": "Left arm",
        "right": "Right arm",
    }
    arm_colors = {
        "left": "#f5b60a",
        "right": "#e0001b",
    }

    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    x = list(range(len(scope_keys)))
    bar_width = 0.34
    all_values = []
    legend_handles = []
    for arm_index, arm in enumerate(arm_keys):
        offset = (arm_index - (len(arm_keys) - 1) / 2.0) * bar_width
        arm_positions = []
        arm_values = []
        for scope_index, scope in enumerate(scope_keys):
            value = value_by_scope_arm.get((scope, arm))
            if value is None:
                continue
            arm_positions.append(float(scope_index) + offset)
            arm_values.append(float(value))
        bars = ax.bar(
            arm_positions,
            arm_values,
            color=arm_colors.get(arm, "#666666"),
            width=bar_width,
            label=arm_labels.get(arm, arm.title()),
        )
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, color=arm_colors.get(arm, "#666666"), label=arm_labels.get(arm, arm.title()))
        )
        all_values.extend(arm_values)
        for bar, value in zip(bars, arm_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.8,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#333333",
            )

    metric_label = "Test accuracy" if args.metric == "test_accuracy" else "Balanced accuracy"
    title = args.title.strip()
    if title:
        ax.set_title(title)
    ax.set_ylabel(f"{metric_label} (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([scope_labels.get(scope, scope.replace("_", " ").title()) for scope in scope_keys])
    min_value = min(all_values) if all_values else 0.0
    lower_bound = 75.0 if min_value >= 75.0 else max(0.0, 5.0 * math.floor(min_value / 5.0))
    ax.set_ylim(lower_bound, max(100.0, max(all_values) + 4.0))
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=False,
    )

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=180, bbox_inches="tight")
    print(f"Saved figure to {args.output_png}")


if __name__ == "__main__":
    main()
