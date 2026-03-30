from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = REPO_ROOT / "eval_metrics"
OUT_ROOT = EVAL_ROOT / "out" / "current_metrics"
MODELS_ROOT = REPO_ROOT / "models" / "strict"
CARLA_RUN_ROOT = EVAL_ROOT / "out"


def _run(cmd: list[str]) -> None:
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


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


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Keep one machine-readable index of every artifact generated in this pass
    # so the report scripts can find staged outputs without re-discovering them.
    inventory: dict[str, object] = {
        "models_root": str(MODELS_ROOT),
        "output_root": str(OUT_ROOT),
    }

    # 1. Harvest offline bundle metrics for active strict models only.
    model_metrics_csv = OUT_ROOT / "model_metrics.csv"
    model_metrics_json = OUT_ROOT / "model_metrics.json"
    _run(
        [
            sys.executable,
            str(EVAL_ROOT / "harvest_model_metrics.py"),
            "--models-root",
            str(MODELS_ROOT),
            "--output-csv",
            str(model_metrics_csv),
            "--output-json",
            str(model_metrics_json),
        ]
    )

    # 2. Build a fresh manifest template in the staging area.
    manifest_template = OUT_ROOT / "table_manifest_template.csv"
    _run(
        [
            sys.executable,
            str(EVAL_ROOT / "build_eval_tables.py"),
            "--write-template-manifest",
            str(manifest_template),
        ]
    )

    # 3. CARLA drive/latency summaries for every paired run.
    pairs = _discover_carla_pairs()
    run_rows: list[dict[str, object]] = []
    for pair in pairs:
        run_dir = str(pair["run_dir"])
        stamp = str(pair["stamp"])
        pair_root = OUT_ROOT / "carla_runs" / run_dir / stamp
        latency_json = pair_root / "latency_summary.json"
        latency_csv = pair_root / "latency_joined.csv"
        drive_json = pair_root / "drive_metrics_summary.json"

        _run(
            [
                sys.executable,
                str(EVAL_ROOT / "analyze_latency.py"),
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
        _run(
            [
                sys.executable,
                str(EVAL_ROOT / "analyze_drive_metrics.py"),
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
                "latency_json": str(latency_json),
                "latency_csv": str(latency_csv),
                "drive_json": str(drive_json),
            }
        )

    inventory["carla_run_pairs"] = run_rows
    inventory["outputs"] = {
        "model_metrics_csv": str(model_metrics_csv),
        "model_metrics_json": str(model_metrics_json),
        "manifest_template": str(manifest_template),
        "carla_run_index_csv": str(OUT_ROOT / "carla_run_index.csv"),
    }

    _write_csv(OUT_ROOT / "carla_run_index.csv", run_rows or [{"note": "No paired CARLA run logs found."}])
    _write_json(OUT_ROOT / "artifact_inventory.json", inventory)

    print("\nCurrent metric gathering complete.")
    print(f"Staged outputs under: {OUT_ROOT}")


if __name__ == "__main__":
    main()
