from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval_metrics.config import (
    ACTIVE_OFFLINE_MODEL_NAMES,
    CURRENT_METRICS_ROOT,
    MODELS_ROOT,
    PIPELINE_SCRIPTS_ROOT,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = CURRENT_METRICS_ROOT
CARLA_RUN_ROOT = EVAL_ROOT / "out"


def _run(cmd: list[str]) -> None:
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def _run_optional(cmd: list[str]) -> tuple[bool, str]:
    print(">", " ".join(cmd))
    completed = subprocess.run(
        cmd,
        check=False,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.stderr:
        print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n")
    if completed.returncode == 0:
        return True, ""
    error_text = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
    return False, error_text


def _extract_stamp(path: Path, prefix: str) -> str | None:
    name = path.stem
    if not name.startswith(prefix):
        return None
    stamp = name[len(prefix) :]
    return stamp or None


def _discover_carla_pairs() -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for run_dir in sorted(CARLA_RUN_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_dir == OUT_ROOT:
            continue

        carla_by_stamp: dict[str, Path] = {}
        rt_by_stamp: dict[str, Path] = {}

        for path in run_dir.glob("carla_drive_*.csv"):
            stamp = _extract_stamp(path, "carla_drive_")
            if stamp:
                carla_by_stamp[stamp] = path
        for path in run_dir.glob("realtime_predictions_*.csv"):
            stamp = _extract_stamp(path, "realtime_predictions_")
            if stamp:
                rt_by_stamp[stamp] = path

        # Pair runs by the shared filename stamp so latency analysis can join
        # realtime and CARLA rows from the same simulator session.
        for stamp in sorted(set(carla_by_stamp) & set(rt_by_stamp)):
            pairs.append(
                {
                    "run_dir": run_dir.name,
                    "stamp": stamp,
                    "carla_log": str(carla_by_stamp[stamp]),
                    "realtime_log": str(rt_by_stamp[stamp]),
                }
            )
    return pairs


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Harvest current offline metrics and stage CARLA/latency outputs."
    )
    parser.add_argument(
        "--model-name",
        action="append",
        default=[],
        help=(
            "Optional exact bundle filename or stem to keep when harvesting model metrics. "
            "Repeatable. Examples: v6_4_gestures_2.pt or v6_4_gestures_2"
        ),
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    requested_model_names = [str(name).strip() for name in args.model_name if str(name).strip()]
    if not requested_model_names:
        requested_model_names = [str(name).strip() for name in ACTIVE_OFFLINE_MODEL_NAMES if str(name).strip()]

    # Keep one machine-readable index of every artifact generated in this pass
    # so the report scripts can find staged outputs without re-discovering them.
    inventory: dict[str, object] = {
        "models_root": str(MODELS_ROOT),
        "output_root": str(OUT_ROOT),
        "requested_model_names": requested_model_names,
    }

    # 1. Harvest offline bundle metrics for active strict models only.
    model_metrics_csv = OUT_ROOT / "model_metrics.csv"
    model_metrics_json = OUT_ROOT / "model_metrics.json"
    harvest_cmd = [
        sys.executable,
        str(PIPELINE_SCRIPTS_ROOT / "harvest_model_metrics.py"),
        "--models-root",
        str(MODELS_ROOT),
        "--output-csv",
        str(model_metrics_csv),
        "--output-json",
        str(model_metrics_json),
    ]
    for model_name in requested_model_names:
        harvest_cmd.extend(["--model-name", model_name])
    _run(harvest_cmd)

    # 2. Build a fresh manifest template in the staging area.
    manifest_template = OUT_ROOT / "table_manifest_template.csv"
    _run(
        [
            sys.executable,
            str(PIPELINE_SCRIPTS_ROOT / "build_eval_tables.py"),
            "--write-template-manifest",
            str(manifest_template),
        ]
    )

    # 3. CARLA drive/latency summaries for every paired run.
    pairs = _discover_carla_pairs()
    run_rows: list[dict[str, object]] = []
    skipped_latency_rows: list[dict[str, object]] = []
    for pair in pairs:
        run_dir = str(pair["run_dir"])
        stamp = str(pair["stamp"])
        pair_root = OUT_ROOT / "carla_runs" / run_dir / stamp
        latency_json = pair_root / "latency_summary.json"
        latency_csv = pair_root / "latency_joined.csv"
        drive_json = pair_root / "drive_metrics_summary.json"

        latency_ok, latency_error = _run_optional(
            [
                sys.executable,
                str(PIPELINE_SCRIPTS_ROOT / "analyze_latency.py"),
                "--realtime-log",
                str(pair["realtime_log"]),
                "--carla-log",
                str(pair["carla_log"]),
                "--output-json",
                str(latency_json),
                "--output-csv",
                str(latency_csv),
            ]
        )
        if not latency_ok:
            latency_json = None
            latency_csv = None
            skipped_latency_rows.append(
                {
                    "run_dir": run_dir,
                    "stamp": stamp,
                    "realtime_log": str(pair["realtime_log"]),
                    "carla_log": str(pair["carla_log"]),
                    "error": latency_error,
                }
            )
        _run(
            [
                sys.executable,
                str(PIPELINE_SCRIPTS_ROOT / "analyze_drive_metrics.py"),
                "--log",
                str(pair["carla_log"]),
                "--output-json",
                str(drive_json),
            ]
        )

        run_rows.append(
            {
                "run_dir": run_dir,
                "stamp": stamp,
                "realtime_log": str(pair["realtime_log"]),
                "carla_log": str(pair["carla_log"]),
                "latency_ok": latency_ok,
                "latency_json": str(latency_json) if latency_json else "",
                "latency_csv": str(latency_csv) if latency_csv else "",
                "latency_error": latency_error,
                "drive_json": str(drive_json),
            }
        )

    inventory["carla_run_pairs"] = run_rows
    inventory["skipped_latency_pairs"] = skipped_latency_rows
    inventory["outputs"] = {
        "model_metrics_csv": str(model_metrics_csv),
        "model_metrics_json": str(model_metrics_json),
        "manifest_template": str(manifest_template),
        "carla_run_index_csv": str(OUT_ROOT / "carla_run_index.csv"),
        "skipped_latency_csv": str(OUT_ROOT / "skipped_latency_runs.csv"),
    }

    _write_csv(OUT_ROOT / "carla_run_index.csv", run_rows or [{"note": "No paired CARLA run logs found."}])
    _write_csv(
        OUT_ROOT / "skipped_latency_runs.csv",
        skipped_latency_rows or [{"note": "No latency runs were skipped."}],
    )
    _write_json(OUT_ROOT / "artifact_inventory.json", inventory)

    print("\nCurrent metric gathering complete.")
    print(f"Staged outputs under: {OUT_ROOT}")
    if skipped_latency_rows:
        print(
            f"Skipped latency analysis for {len(skipped_latency_rows)} run(s). "
            f"See {OUT_ROOT / 'skipped_latency_runs.csv'}"
        )


if __name__ == "__main__":
    main()
