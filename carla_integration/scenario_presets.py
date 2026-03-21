from dataclasses import dataclass
from typing import Dict, Optional


FIXED_SEDAN_BLUEPRINT = "vehicle.lincoln.mkz_2020"


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
    timeout_s: float = 300.0
    lead_blueprint_id: str = FIXED_SEDAN_BLUEPRINT
    lead_spawn_distance_m: float = 40.0
    lead_speed_reduction_pct: float = 55.0
    overtake_finish_margin_m: float = 12.0
    require_return_to_start_lane: bool = True


SCENARIO_PRESETS: Dict[str, ScenarioPreset] = {
    "lane_keep_5min": ScenarioPreset(
        name="lane_keep_5min",
        kind="lane_keep",
        map_name="Town02_Opt",
        ego_spawn_index=0,
        start_offset_m=18.0,
        checkpoint_spacing_m=35.0,
        checkpoint_radius_m=10.0,
        route_length_m=2600.0,
        timeout_s=420.0,
    ),
    "highway_overtake": ScenarioPreset(
        name="highway_overtake",
        kind="overtake",
        map_name="Town04_Opt",
        ego_spawn_index=0,
        start_offset_m=18.0,
        checkpoint_spacing_m=25.0,
        checkpoint_radius_m=8.0,
        route_length_m=700.0,
        timeout_s=120.0,
        lead_spawn_distance_m=45.0,
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
