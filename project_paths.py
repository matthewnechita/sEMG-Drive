from pathlib import Path


# Keep the maintained strict dataset and model roots in one place so collection,
# preprocessing, and training scripts all resolve the same folder contract.
STRICT_DATA_ROOT = Path("data_strict")
STRICT_RESAMPLED_ROOT = Path("data_resampled_strict")
STRICT_MODELS_ROOT = Path("models") / "strict"


def strict_arm_root(root: Path | str, arm: str) -> Path:
    # The on-disk naming contract is "<arm> arm", not just the raw arm token.
    arm_name = str(arm).strip().lower()
    if arm_name not in {"left", "right"}:
        raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
    return Path(root) / f"{arm_name} arm"


def strict_subject_root(root: Path | str, arm: str, subject: str) -> Path:
    subject_name = str(subject).strip()
    if not subject_name:
        raise ValueError("subject must not be empty.")
    return strict_arm_root(root, arm) / subject_name


def strict_raw_dir(root: Path | str, arm: str, subject: str) -> Path:
    return strict_subject_root(root, arm, subject) / "raw"


def strict_filtered_dir(root: Path | str, arm: str, subject: str) -> Path:
    return strict_subject_root(root, arm, subject) / "filtered"
