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
PROMPT_LOG_ROOT = EVAL_ROOT / "logs"
CARLA_RUN_ROOT = EVAL_ROOT / "out"


def _run(cmd: list[str]) -> None:
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def _csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        row = next(reader, [])
    return [str(value).strip() for value in row]


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


def _discover_prompt_logs() -> tuple[list[Path], list[Path]]:
    behavior_ready: list[Path] = []
    skipped: list[Path] = []
    if not PROMPT_LOG_ROOT.exists():
        return behavior_ready, skipped

    for path in sorted(PROMPT_LOG_ROOT.rglob("*.csv")):
        header = _csv_header(path)
        if "prompt_label" in header:
            behavior_ready.append(path)
        else:
            skipped.append(path)
    return behavior_ready, skipped


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

    # 3. Prompted realtime behavior metrics, only when prompt labels exist.
    prompt_logs, prompt_logs_skipped = _discover_prompt_logs()
    behavior_rows: list[dict[str, object]] = []
    behavior_root = OUT_ROOT / "realtime_behavior"
    for path in prompt_logs:
        rel_slug = "_".join(path.relative_to(PROMPT_LOG_ROOT).with_suffix("").parts)
        out_json = behavior_root / f"{rel_slug}.json"
        out_segments = behavior_root / f"{rel_slug}_segments.csv"
        _run(
            [
                sys.executable,
                str(EVAL_ROOT / "realtime_behavior_metrics.py"),
                "--input",
                str(path),
                "--output-json",
                str(out_json),
                "--output-segments-csv",
                str(out_segments),
            ]
        )
        behavior_rows.append(
            {
                "input_csv": str(path),
                "summary_json": str(out_json),
                "segments_csv": str(out_segments),
            }
        )

    # 4. CARLA drive/latency summaries for every paired run.
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

    inventory["realtime_behavior_ready_logs"] = [str(path) for path in prompt_logs]
    inventory["realtime_behavior_skipped_logs"] = [str(path) for path in prompt_logs_skipped]
    inventory["carla_run_pairs"] = run_rows
    inventory["outputs"] = {
        "model_metrics_csv": str(model_metrics_csv),
        "model_metrics_json": str(model_metrics_json),
        "manifest_template": str(manifest_template),
        "realtime_behavior_index_csv": str(OUT_ROOT / "realtime_behavior_index.csv"),
        "carla_run_index_csv": str(OUT_ROOT / "carla_run_index.csv"),
    }

    _write_csv(OUT_ROOT / "realtime_behavior_index.csv", behavior_rows or [{"note": "No prompted realtime logs with prompt_label found."}])
    _write_csv(OUT_ROOT / "carla_run_index.csv", run_rows or [{"note": "No paired CARLA run logs found."}])
    _write_json(OUT_ROOT / "artifact_inventory.json", inventory)

    print("\nCurrent metric gathering complete.")
    print(f"Staged outputs under: {OUT_ROOT}")
    if not prompt_logs:
        print("No prompted realtime logs with prompt labels were found, so behavior metrics were skipped.")


if __name__ == "__main__":
    main()
