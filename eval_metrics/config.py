from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = REPO_ROOT / "eval_metrics"
OUT_ROOT = EVAL_ROOT / "out"
CURRENT_METRICS_ROOT = OUT_ROOT / "current_metrics"
TABLES_ROOT = OUT_ROOT / "tables"
FIGURES_ROOT = OUT_ROOT / "figures"
PIPELINE_SCRIPTS_ROOT = EVAL_ROOT / "pipeline_scripts"
PLOT_SCRIPTS_ROOT = EVAL_ROOT / "plot_scripts"
MODELS_ROOT = REPO_ROOT / "models" / "strict"
FINAL_CAPSTONE_TABLE_CSV = EVAL_ROOT / "final_capstone_table.csv"
FINAL_RESEARCH_TABLE_CSV = EVAL_ROOT / "final_research_table.csv"

# Normal use: only edit these two lines when the maintained model selection
# changes. Each filename covers the correlated left/right pair for that family.
ACTIVE_PER_SUBJECT_MODEL_NAME = "Matthewv6_4_gestures.pt"
ACTIVE_CROSS_SUBJECT_MODEL_NAME = "v6_4_gestures_2.pt"

# Offline model harvesting/plots use both the current per-subject and current
# evaluated model families by default.
ACTIVE_OFFLINE_MODEL_NAMES = [
    ACTIVE_PER_SUBJECT_MODEL_NAME,
    ACTIVE_CROSS_SUBJECT_MODEL_NAME,
]

# Table-building and run-linked evaluation defaults use the actively evaluated
# model family only, since CARLA/realtime logs correspond to one deployed pair.
ACTIVE_TABLE_MODEL_NAME = ACTIVE_CROSS_SUBJECT_MODEL_NAME
