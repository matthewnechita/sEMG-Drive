from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np


STRICT_LAYOUT_VERSION = "strict_pair_v1"

PAIR_RE = re.compile(r"^\((\d+)\)")


@dataclass(frozen=True)
class StrictSlot:
    slot_name: str
    pair_number: int
    sensor_kind: str
    channel_count: int


@dataclass(frozen=True)
class StrictLayoutResolution:
    arm: str
    ordered_indices: np.ndarray
    slot_names: tuple[str, ...]
    pair_numbers: tuple[int, ...]
    sensor_kinds: tuple[str, ...]
    channel_counts: tuple[int, ...]


STRICT_ARM_LAYOUTS: dict[str, tuple[StrictSlot, ...]] = {
    "right": (
        StrictSlot("R_Avanti_1", 1, "avanti", 1),
        StrictSlot("R_Avanti_2", 2, "avanti", 1),
        StrictSlot("R_Avanti_3", 3, "avanti", 1),
        StrictSlot("R_Maize", 7, "maize", 9),
        StrictSlot("R_Galileo", 9, "galileo", 4),
        StrictSlot("R_Mini", 11, "mini", 1),
    ),
    "left": (
        StrictSlot("L_Avanti_1", 4, "avanti", 1),
        StrictSlot("L_Avanti_2", 5, "avanti", 1),
        StrictSlot("L_Avanti_3", 6, "avanti", 1),
        StrictSlot("L_Maize", 8, "maize", 9),
        StrictSlot("L_Galileo", 10, "galileo", 4),
    ),
}


def _normalize_arm(arm: str) -> str:
    value = str(arm).strip().lower()
    if value not in STRICT_ARM_LAYOUTS:
        raise ValueError(f"Unsupported strict layout arm: {arm!r}")
    return value


def parse_pair_number(channel_label: Any) -> int | None:
    text = str(channel_label).strip()
    match = PAIR_RE.match(text)
    return int(match.group(1)) if match else None


def infer_sensor_kind_from_label(channel_label: Any) -> str | None:
    text = str(channel_label).strip().lower()
    if "galileo" in text:
        return "galileo"
    if "maize" in text:
        return "maize"
    if "mini" in text:
        return "mini"
    if "avanti" in text:
        return "avanti"
    return None


def strict_slots_for_arm(arm: str) -> tuple[StrictSlot, ...]:
    return STRICT_ARM_LAYOUTS[_normalize_arm(arm)]


def strict_pair_numbers_for_arm(arm: str) -> list[int]:
    return [slot.pair_number for slot in strict_slots_for_arm(arm)]


def strict_channel_count_for_arm(arm: str) -> int:
    return int(sum(slot.channel_count for slot in strict_slots_for_arm(arm)))


def strict_layout_bundle_metadata(arm: str) -> dict[str, Any]:
    slots = strict_slots_for_arm(arm)
    return {
        "layout_mode": "strict",
        "strict_layout_version": STRICT_LAYOUT_VERSION,
        "arm": _normalize_arm(arm),
        "slot_order": [slot.slot_name for slot in slots],
        "pair_order": [int(slot.pair_number) for slot in slots],
        "sensor_kinds": [str(slot.sensor_kind) for slot in slots],
        "channel_counts_per_slot": [int(slot.channel_count) for slot in slots],
        "expected_channel_count": int(sum(slot.channel_count for slot in slots)),
    }


def _metadata_as_dict(metadata: Any) -> dict[str, Any]:
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, np.ndarray) and metadata.shape == () and metadata.dtype == object:
        try:
            value = metadata.item()
        except Exception:
            return {}
        return value if isinstance(value, dict) else {}
    return {}


def channel_labels_from_metadata(metadata: Any) -> list[str]:
    md = _metadata_as_dict(metadata)
    for key in ("emg_channel_labels", "emg_channel_names", "channel_labels"):
        labels = md.get(key)
        if labels is None:
            continue
        return [str(value) for value in labels]
    return []


def resolve_strict_channel_indices(
    channel_labels: list[str] | tuple[str, ...],
    arm: str,
) -> StrictLayoutResolution:
    labels = [str(value) for value in channel_labels]
    arm_name = _normalize_arm(arm)
    slots = strict_slots_for_arm(arm_name)
    pair_to_indices: dict[int, list[int]] = {}
    pair_to_kind: dict[int, str | None] = {}

    for idx, label in enumerate(labels):
        pair = parse_pair_number(label)
        if pair is None:
            continue
        pair_to_indices.setdefault(pair, []).append(int(idx))
        kind = infer_sensor_kind_from_label(label)
        if kind is not None:
            existing = pair_to_kind.get(pair)
            if existing is not None and existing != kind:
                raise ValueError(
                    f"Inconsistent sensor kind labels for pair {pair}: {existing!r} vs {kind!r}"
                )
            pair_to_kind[pair] = kind

    missing_pairs = [slot.pair_number for slot in slots if slot.pair_number not in pair_to_indices]
    if missing_pairs:
        raise ValueError(
            f"Strict layout for {arm_name} missing required pair(s): {missing_pairs}."
        )

    ordered_indices: list[int] = []
    for slot in slots:
        idxs = pair_to_indices[slot.pair_number]
        if len(idxs) != slot.channel_count:
            raise ValueError(
                f"Strict layout mismatch for {slot.slot_name}: pair {slot.pair_number} has "
                f"{len(idxs)} channel(s), expected {slot.channel_count}."
            )
        observed_kind = pair_to_kind.get(slot.pair_number)
        if observed_kind is not None and observed_kind != slot.sensor_kind:
            raise ValueError(
                f"Strict layout mismatch for {slot.slot_name}: pair {slot.pair_number} "
                f"looks like {observed_kind!r}, expected {slot.sensor_kind!r}."
            )
        ordered_indices.extend(idxs)

    ordered = np.asarray(ordered_indices, dtype=int)
    expected_channels = strict_channel_count_for_arm(arm_name)
    if ordered.size != expected_channels:
        raise ValueError(
            f"Strict layout mismatch for {arm_name}: resolved {ordered.size} channel(s), "
            f"expected {expected_channels}."
        )

    return StrictLayoutResolution(
        arm=arm_name,
        ordered_indices=ordered,
        slot_names=tuple(slot.slot_name for slot in slots),
        pair_numbers=tuple(int(slot.pair_number) for slot in slots),
        sensor_kinds=tuple(str(slot.sensor_kind) for slot in slots),
        channel_counts=tuple(int(slot.channel_count) for slot in slots),
    )


def resolve_strict_indices_from_metadata(metadata: Any, arm: str) -> StrictLayoutResolution:
    channel_labels = channel_labels_from_metadata(metadata)
    if not channel_labels:
        raise ValueError(
            f"Strict layout for {arm!r} requires metadata.emg_channel_labels from a new collection session."
        )
    return resolve_strict_channel_indices(channel_labels, arm=arm)
