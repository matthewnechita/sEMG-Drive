from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval_metrics.config import CURRENT_METRICS_ROOT, FIGURES_ROOT, PLOT_SCRIPTS_ROOT


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str]) -> None:
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the report-grade figure set from current staged metrics."
    )
    parser.add_argument(
        "--run-index",
        type=Path,
        default=CURRENT_METRICS_ROOT / "carla_run_index.csv",
        help="Run index CSV from gather_current_metrics.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURES_ROOT,
        help="Directory for the generated figures.",
    )
    return parser


def _remove_obsolete_outputs(output_dir: Path) -> None:
    for path in output_dir.glob("*.svg"):
        path.unlink(missing_ok=True)
    (output_dir / "carla_summary_bars.png").unlink(missing_ok=True)
    (output_dir / "latency_summary_bars.png").unlink(missing_ok=True)
    (output_dir / "model_accuracy_bars.png").unlink(missing_ok=True)
    (output_dir / "model_summary.png").unlink(missing_ok=True)


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_obsolete_outputs(output_dir)

    confusion_cmd = [
        sys.executable,
        str(PLOT_SCRIPTS_ROOT / "plot_confusion_matrix.py"),
        "--output-png",
        str(output_dir / "confusion_matrix_row_norm.png"),
    ]

    _run(confusion_cmd)
    _run(
        [
            sys.executable,
            str(PLOT_SCRIPTS_ROOT / "plot_latency_ecdf.py"),
            "--run-index",
            str(Path(args.run_index)),
            "--output-png",
            str(output_dir / "latency_ecdf.png"),
        ]
    )
    _run(
        [
            sys.executable,
            str(PLOT_SCRIPTS_ROOT / "plot_carla_run_distributions.py"),
            "--run-index",
            str(Path(args.run_index)),
            "--output-png",
            str(output_dir / "carla_run_distributions.png"),
        ]
    )
    _run(
        [
            sys.executable,
            str(PLOT_SCRIPTS_ROOT / "plot_model_accuracy_forest.py"),
            "--output-png",
            str(output_dir / "model_accuracy_forest.png"),
        ]
    )
    _run(
        [
            sys.executable,
            str(PLOT_SCRIPTS_ROOT / "plot_model_metric_heatmap.py"),
            "--output-png",
            str(output_dir / "model_metric_heatmap.png"),
        ]
    )
    _run(
        [
            sys.executable,
            str(PLOT_SCRIPTS_ROOT / "plot_gesture_embedding_clusters.py"),
            "--output-png",
            str(output_dir / "gesture_embedding_clusters.png"),
        ]
    )
    _run(
        [
            sys.executable,
            str(PLOT_SCRIPTS_ROOT / "plot_drive_trace_representative.py"),
            "--output-png",
            str(output_dir / "drive_trace_representative.png"),
        ]
    )

    print(f"\nSaved figure pack under: {output_dir}")


if __name__ == "__main__":
    main()
