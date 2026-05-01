from __future__ import annotations

import subprocess
import sys
from pathlib import Path

try:
    from eval_metrics.config import (
        FIGURES_ROOT,
        FINAL_CAPSTONE_TABLE_CSV,
        FINAL_RESEARCH_TABLE_CSV,
        PIPELINE_SCRIPTS_ROOT,
        PLOT_SCRIPTS_ROOT,
        TABLES_ROOT,
    )
except ModuleNotFoundError:
    from config import (
        FIGURES_ROOT,
        FINAL_CAPSTONE_TABLE_CSV,
        FINAL_RESEARCH_TABLE_CSV,
        PIPELINE_SCRIPTS_ROOT,
        PLOT_SCRIPTS_ROOT,
        TABLES_ROOT,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = REPO_ROOT / "eval_metrics"


def _run(script_path: Path) -> None:
    cmd = [sys.executable, str(script_path)]
    print(">", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def _remove_obsolete_outputs() -> None:
    if not FIGURES_ROOT.exists():
        return
    for path in FIGURES_ROOT.glob("*.svg"):
        path.unlink(missing_ok=True)
    (FIGURES_ROOT / "carla_summary_bars.png").unlink(missing_ok=True)
    (FIGURES_ROOT / "latency_summary_bars.png").unlink(missing_ok=True)
    (FIGURES_ROOT / "model_accuracy_bars.png").unlink(missing_ok=True)
    (FIGURES_ROOT / "model_summary.png").unlink(missing_ok=True)


def main() -> None:
    _remove_obsolete_outputs()
    _run(PIPELINE_SCRIPTS_ROOT / "gather_current_metrics.py")
    _run(PIPELINE_SCRIPTS_ROOT / "fill_table_manifest.py")
    _run(PIPELINE_SCRIPTS_ROOT / "build_research_runtime_selection.py")
    _run(PIPELINE_SCRIPTS_ROOT / "build_research_control_stability.py")
    _run(PIPELINE_SCRIPTS_ROOT / "build_eval_tables.py")
    _run(PLOT_SCRIPTS_ROOT / "make_report_figures.py")

    print("\nEval refresh complete.")
    print(f"Figures: {FIGURES_ROOT}")
    print(f"Tables: {TABLES_ROOT}")
    print(f"Final capstone table: {FINAL_CAPSTONE_TABLE_CSV}")
    print(f"Final research table: {FINAL_RESEARCH_TABLE_CSV}")


if __name__ == "__main__":
    main()
