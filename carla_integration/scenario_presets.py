from dataclasses import dataclass
from typing import Dict, Optional, Tuple


FIXED_SEDAN_BLUEPRINT = "vehicle.lincoln.mkz_2020"
CheckpointXYZ = Tuple[float, float, float]


def _progress_markers(start_m: float, end_m: float, spacing_m: float) -> Tuple[float, ...]:
    markers = []
    current = float(start_m)
    end = float(end_m)
    spacing = max(1.0, float(spacing_m))
    while current < end:
        markers.append(round(current, 3))
        current += spacing
    if not markers or abs(markers[-1] - end) > 1e-3:
        markers.append(round(end, 3))
    return tuple(markers)


@dataclass(frozen=True)
class ScenarioPreset:
    name: str
    kind: str
    map_name: str
    ego_blueprint_id: str = FIXED_SEDAN_BLUEPRINT
    ego_spawn_index: int = 0
    start_offset_m: float = 18.0
    checkpoint_spacing_m: float = 40.0
    checkpoint_radius_m: float = 10.0
    route_length_m: float = 0.0
    checkpoint_progress_m: Tuple[float, ...] = ()
    checkpoint_locations_xyz: Tuple[CheckpointXYZ, ...] = ()
    timeout_s: Optional[float] = 300.0
    lead_blueprint_id: str = FIXED_SEDAN_BLUEPRINT
    lead_spawn_distance_m: float = 40.0
    lead_hold_until_start: bool = False
    lead_speed_reduction_pct: float = 55.0
    overtake_finish_margin_m: float = 12.0
    require_return_to_start_lane: bool = True
    draw_debug_markers: bool = True


SCENARIO_PRESETS: Dict[str, ScenarioPreset] = {
    "lane_keep_5min": ScenarioPreset(
        name="lane_keep_5min",
        kind="lane_keep",
        map_name="Town04_Opt",
        ego_spawn_index=0,
        start_offset_m=18.0,
        checkpoint_spacing_m=35.0,
        checkpoint_radius_m=10.0,
        route_length_m=2600.0,
        checkpoint_progress_m=_progress_markers(18.0, 2600.0, 140.0),
        timeout_s=None,
    ),
    "highway_overtake": ScenarioPreset(
        name="highway_overtake",
        kind="overtake",
        map_name="Town04_Opt",
        ego_spawn_index=0,
        start_offset_m=18.0,
        checkpoint_spacing_m=25.0,
        checkpoint_radius_m=8.0,
        route_length_m=420.0,
        checkpoint_progress_m=_progress_markers(18.0, 420.0, 85.0),
        timeout_s=None,
        lead_spawn_distance_m=25.0,
        lead_hold_until_start=True,
        lead_speed_reduction_pct=60.0,
        overtake_finish_margin_m=15.0,
        require_return_to_start_lane=True,
    ),
}


def get_scenario_preset(name: str) -> Optional[ScenarioPreset]:
    key = str(name or "").strip().lower()
    if not key:
        return None
    return SCENARIO_PRESETS.get(key)


def scenario_choices():
    return sorted(SCENARIO_PRESETS.keys())
