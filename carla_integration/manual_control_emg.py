#!/usr/bin/env python

# Copyright (c) 2019 Intel Labs
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""CARLA manual control adapted for EMG-driven vehicle testing."""

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc

import argparse
import collections
import csv
import datetime
import logging
import math
import random
import re
import weakref
from configparser import ConfigParser
from pathlib import Path

from carla_integration.scenario_presets import get_scenario_preset, scenario_choices
from emg.runtime_tuning import CARLA_TUNING, RUNTIME_TUNING_NAME

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_PERIOD
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_c
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import realtime_gesture_cnn as realtime_gesture
except Exception as exc:
    print("[gesture] failed to import realtime_gesture_cnn:", exc)
    realtime_gesture = None
import threading
import time
try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

# Gesture/CARLA timing diagnostics.
# Frame rate does not need to match EMG sample rate; this only controls how
# often CARLA polls and applies the latest gesture label.
DEFAULT_CLIENT_FPS_LIMIT = 30
CLIENT_FPS_LIMIT = DEFAULT_CLIENT_FPS_LIMIT
CLIENT_USE_BUSY_LOOP = False
GESTURE_TIMING_DEBUG = True
GESTURE_TIMING_WARN_MS = 200.0
DEFAULT_LOW_GRAPHICS_MODE = True
LOW_GRAPHICS_MODE = DEFAULT_LOW_GRAPHICS_MODE
DEFAULT_LOW_GRAPHICS_NO_RENDERING = False  # True = lowest possible GPU use, but no camera view.
LOW_GRAPHICS_NO_RENDERING = DEFAULT_LOW_GRAPHICS_NO_RENDERING
DEFAULT_CLIENT_RES = '640x360'
DEFAULT_CAMERA_RES = '640x360'
DEFAULT_CAMERA_FPS = 10.0
CAMERA_IMAGE_WIDTH = 640
CAMERA_IMAGE_HEIGHT = 360
CAMERA_SENSOR_TICK_S = 1.0 / DEFAULT_CAMERA_FPS
DEFAULT_HUD_VISIBLE = False
HUD_VISIBLE_BY_DEFAULT = DEFAULT_HUD_VISIBLE
REVERSE_TOGGLE_COOLDOWN_S = 1.0
REVERSE_TOGGLE_MAX_SPEED_MPS = 0.75
MANUAL_EARLY_CLOSE_REASON = "closed_early"


def _now_stamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_dual_arm_steer_key(left_label: str, right_label: str) -> str:
    left_label = str(left_label)
    right_label = str(right_label)
    # Final dual-arm steering contract: left arm drives strong steering,
    # right arm drives weak steering.
    if left_label == "left_turn":
        return "left_strong"
    if left_label == "right_turn":
        return "right_strong"
    if right_label == "left_turn":
        return "left"
    if right_label == "right_turn":
        return "right"
    return "neutral"


def _map_basename(map_name):
    map_text = str(map_name or '').strip()
    if not map_text:
        return ''
    return map_text.rsplit('/', 1)[-1].split('.')[0]


def _resolve_world(client, requested_map=''):
    requested = _map_basename(requested_map)
    current_world = client.get_world()
    if not requested:
        return current_world

    current_name = _map_basename(current_world.get_map().name)
    if current_name == requested:
        print(f"[carla] map already loaded: {current_name}")
        return current_world

    print(f"[carla] loading map: {requested} (current: {current_name})")
    client.set_timeout(20.0)
    try:
        world = client.load_world(requested)
    finally:
        client.set_timeout(2.0)
    loaded_name = _map_basename(world.get_map().name)
    print(f"[carla] loaded map: {loaded_name}")
    return world


class DriveCSVLogger(object):
    def __init__(self, path):
        self.path = str(path)
        out_path = Path(self.path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = out_path.open('w', newline='', encoding='utf-8')
        self._fieldnames = [
            'timestamp',
            'control_apply_ts',
            'simulation_time',
            'prediction_seq',
            'steer_key',
            'applied_steer_key',
            'steer',
            'throttle',
            'brake',
            'reverse',
            'hand_brake',
            'speed_mps',
            'speed_limit_mps',
            'velocity_deviation_mps',
            'steering_angle_rad',
            'lane_error_m',
            'lane_invasion_event',
            'scenario_name',
            'scenario_kind',
            'scenario_status',
            'scenario_finished',
            'scenario_success',
            'scenario_failure_reason',
            'scenario_elapsed_s',
            'scenario_completion_time_s',
        ]
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
        self._writer.writeheader()

    def write_row(self, row):
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        self._file.close()


def _location_distance(a, b):
    return math.sqrt(
        (float(a.x) - float(b.x)) ** 2
        + (float(a.y) - float(b.y)) ** 2
        + (float(a.z) - float(b.z)) ** 2
    )


def _velocity_speed_mps(velocity):
    return math.sqrt(
        float(velocity.x) ** 2
        + float(velocity.y) ** 2
        + float(velocity.z) ** 2
    )


def _speed_limit_mps(actor):
    try:
        speed_limit_kph = float(actor.get_speed_limit())
    except RuntimeError:
        return None
    if speed_limit_kph <= 0.0:
        return None
    return speed_limit_kph / 3.6


def _vehicle_max_steer_angle_rad(actor):
    try:
        physics = actor.get_physics_control()
    except RuntimeError:
        return None
    wheels = tuple(getattr(physics, "wheels", ()) or ())
    if not wheels:
        return None
    max_deg = max(float(getattr(wheel, "max_steer_angle", 0.0) or 0.0) for wheel in wheels)
    if max_deg <= 0.0:
        return None
    return math.radians(max_deg)


def _shift_location(location, z=0.0):
    return carla.Location(
        x=float(location.x),
        y=float(location.y),
        z=float(location.z) + float(z),
    )


def _yaw_delta_deg(lhs, rhs):
    delta = (float(lhs) - float(rhs) + 180.0) % 360.0 - 180.0
    return abs(delta)


class ScenarioRuntime(object):
    def __init__(self, preset, traffic_manager):
        self.preset = preset
        self._traffic_manager = traffic_manager
        self._actors = []
        self._lead_vehicle = None
        self._route_waypoints = []
        self._route_progress_m = []
        self._checkpoint_locations = []
        self._checkpoint_progress_m = []
        self._start_location = None
        self._finish_location = None
        self._start_lane_id = None
        self._start_road_id = None
        self._status = "idle"
        self._started = False
        self._finished = False
        self._success = False
        self._failure_reason = ""
        self._start_sim_time = None
        self._finish_sim_time = None
        self._completion_time_s = None
        self._next_checkpoint_index = 0
        self._start_checkpoint_index = 0
        self._last_progress_m = None
        self._overtake_objective_met = False
        self._last_next_checkpoint_distance_m = None
        self._last_lead_distance_m = None
        self._last_lead_gap_m = None
        self._last_current_lane_id = None
        self._last_sim_time = 0.0
        self._should_exit = False
        self._lead_autopilot_enabled = False
        self._lead_response_active = False
        self._lead_speed_reduction_pct = None

    def get_ego_spawn_transform(self, carla_map):
        anchor_transform = self._get_anchor_spawn_transform(carla_map)
        spawn_before_start_m = max(
            0.0,
            float(getattr(self.preset, "ego_spawn_before_start_checkpoint_m", 0.0) or 0.0),
        )
        if spawn_before_start_m <= 0.0:
            return anchor_transform

        anchor_waypoint = self._get_anchor_waypoint(carla_map)
        if anchor_waypoint is None:
            return anchor_transform

        route_waypoints, route_progress = self._build_route(anchor_waypoint)
        if not route_waypoints or not route_progress:
            return anchor_transform

        start_checkpoint_progress = self._resolve_start_checkpoint_progress(route_waypoints, route_progress)
        if start_checkpoint_progress is None:
            return anchor_transform

        target_progress = max(0.0, float(start_checkpoint_progress) - spawn_before_start_m)
        spawn_transform = self._transform_at_route_progress(route_waypoints, route_progress, target_progress)
        return spawn_transform if spawn_transform is not None else anchor_transform

    def _get_anchor_spawn_transform(self, carla_map):
        spawn_points = list(carla_map.get_spawn_points())
        if not spawn_points:
            return carla.Transform()
        index = int(self.preset.ego_spawn_index) % len(spawn_points)
        selected = spawn_points[index]
        return carla.Transform(
            carla.Location(
                x=float(selected.location.x),
                y=float(selected.location.y),
                z=float(selected.location.z) + 0.2,
            ),
            carla.Rotation(
                pitch=float(selected.rotation.pitch),
                yaw=float(selected.rotation.yaw),
                roll=float(selected.rotation.roll),
            ),
        )

    def _get_anchor_waypoint(self, carla_map):
        anchor_transform = self._get_anchor_spawn_transform(carla_map)
        return carla_map.get_waypoint(
            anchor_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

    def _resolve_start_checkpoint_progress(self, route_waypoints, route_progress):
        if not route_waypoints or not route_progress:
            return None
        start_checkpoint_index = max(
            0,
            int(getattr(self.preset, "start_checkpoint_index", 0) or 0),
        )

        explicit_progress = [
            float(x)
            for x in getattr(self.preset, 'checkpoint_progress_m', ())
            if x is not None
        ]
        if explicit_progress:
            if start_checkpoint_index >= len(explicit_progress):
                return None
            return min(float(route_progress[-1]), explicit_progress[start_checkpoint_index])

        explicit_locations = self._locations_from_explicit_checkpoints()
        if explicit_locations:
            if start_checkpoint_index >= len(explicit_locations):
                return None
            checkpoint_location = explicit_locations[start_checkpoint_index]
            best_progress = None
            best_distance = None
            for waypoint, progress in zip(route_waypoints, route_progress):
                distance = _location_distance(checkpoint_location, waypoint.transform.location)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_progress = float(progress)
            return best_progress

        checkpoint_spacing_m = max(10.0, float(self.preset.checkpoint_spacing_m))
        start_progress = float(self.preset.start_offset_m) + start_checkpoint_index * checkpoint_spacing_m
        return min(float(route_progress[-1]), start_progress)

    def _transform_at_route_progress(self, route_waypoints, route_progress, target_progress_m):
        if not route_waypoints or not route_progress:
            return None
        target_progress = max(0.0, float(target_progress_m))
        route_index = len(route_progress) - 1
        for idx, progress in enumerate(route_progress):
            if float(progress) >= target_progress:
                route_index = idx
                break
        waypoint = route_waypoints[route_index]
        return carla.Transform(
            carla.Location(
                x=float(waypoint.transform.location.x),
                y=float(waypoint.transform.location.y),
                z=float(waypoint.transform.location.z) + 0.2,
            ),
            carla.Rotation(
                pitch=float(waypoint.transform.rotation.pitch),
                yaw=float(waypoint.transform.rotation.yaw),
                roll=float(waypoint.transform.rotation.roll),
            ),
        )

    def _choose_next_waypoint(self, waypoint, step_m):
        candidates = list(waypoint.next(float(step_m)))
        if not candidates:
            return None
        current_yaw = float(waypoint.transform.rotation.yaw)

        def _score(candidate):
            yaw_penalty = _yaw_delta_deg(current_yaw, candidate.transform.rotation.yaw)
            road_penalty = 0.0 if int(candidate.road_id) == int(waypoint.road_id) else 90.0
            lane_penalty = abs(int(candidate.lane_id) - int(waypoint.lane_id)) * 10.0
            return road_penalty + lane_penalty + yaw_penalty

        return min(candidates, key=_score)

    def _build_route(self, start_waypoint):
        explicit_progress = [float(x) for x in getattr(self.preset, 'checkpoint_progress_m', ()) if x is not None]
        total_length = max(
            float(self.preset.route_length_m),
            float(self.preset.start_offset_m) + 20.0,
            max(explicit_progress) if explicit_progress else 0.0,
            float(self.preset.lead_spawn_distance_m) + 25.0,
        )
        step_m = min(max(float(self.preset.checkpoint_spacing_m) * 0.5, 8.0), 15.0)
        route_waypoints = [start_waypoint]
        route_progress = [0.0]
        current = start_waypoint
        traveled = 0.0

        while traveled < total_length:
            nxt = self._choose_next_waypoint(current, step_m)
            if nxt is None:
                break
            step_dist = _location_distance(current.transform.location, nxt.transform.location)
            if step_dist <= 1e-3:
                break
            traveled += step_dist
            route_waypoints.append(nxt)
            route_progress.append(traveled)
            current = nxt

        return route_waypoints, route_progress

    def _progress_to_location(self, target_progress_m):
        if not self._route_progress_m:
            return None
        target = float(target_progress_m)
        for idx, progress in enumerate(self._route_progress_m):
            if progress >= target:
                return self._route_waypoints[idx].transform.location
        return self._route_waypoints[-1].transform.location

    def _lead_spawn_waypoint(self, target_progress_m):
        route_index = self._progress_to_route_index(target_progress_m)
        if route_index is None or not self._route_waypoints:
            return None

        if self._start_lane_id is None or self._start_road_id is None:
            return self._route_waypoints[route_index]

        candidate = self._route_waypoints[route_index]
        if (
            int(candidate.road_id) == int(self._start_road_id)
            and int(candidate.lane_id) == int(self._start_lane_id)
        ):
            return candidate

        best_match = None
        best_offset = None
        for idx, waypoint in enumerate(self._route_waypoints):
            if (
                int(waypoint.road_id) != int(self._start_road_id)
                or int(waypoint.lane_id) != int(self._start_lane_id)
            ):
                continue
            offset = abs(int(idx) - int(route_index))
            if best_offset is None or offset < best_offset:
                best_match = waypoint
                best_offset = offset
        return best_match if best_match is not None else candidate

    def _locations_from_explicit_checkpoints(self):
        checkpoints = []
        explicit_locations = list(getattr(self.preset, 'checkpoint_locations_xyz', ()) or ())
        for xyz in explicit_locations:
            if xyz is None or len(xyz) != 3:
                continue
            checkpoints.append(
                carla.Location(
                    x=float(xyz[0]),
                    y=float(xyz[1]),
                    z=float(xyz[2]),
                )
            )
        return checkpoints

    def _build_checkpoints(self):
        explicit_locations = self._locations_from_explicit_checkpoints()
        if explicit_locations:
            return explicit_locations

        explicit_progress = [float(x) for x in getattr(self.preset, 'checkpoint_progress_m', ()) if x is not None]
        if explicit_progress:
            checkpoints = []
            for progress in explicit_progress:
                location = self._progress_to_location(progress)
                if location is None:
                    continue
                if checkpoints and _location_distance(checkpoints[-1], location) <= 1.0:
                    continue
                checkpoints.append(location)
            return checkpoints

        checkpoints = []
        next_progress = float(self.preset.start_offset_m)
        route_end = float(self._route_progress_m[-1]) if self._route_progress_m else 0.0
        spacing = max(10.0, float(self.preset.checkpoint_spacing_m))
        while next_progress < route_end:
            location = self._progress_to_location(next_progress)
            if location is not None:
                checkpoints.append(location)
            next_progress += spacing
        finish_location = self._progress_to_location(route_end)
        if finish_location is not None and (
            not checkpoints or _location_distance(checkpoints[-1], finish_location) > 1.0
        ):
            checkpoints.append(finish_location)
        return checkpoints

    def _project_progress_m(self, location):
        if not self._route_waypoints:
            return None
        best_index = None
        best_distance = None
        for idx, waypoint in enumerate(self._route_waypoints):
            dist = _location_distance(location, waypoint.transform.location)
            if best_distance is None or dist < best_distance:
                best_distance = dist
                best_index = idx
        if best_index is None:
            return None
        return float(self._route_progress_m[best_index])

    def _distance_to_next_checkpoint(self, ego_location):
        if self._next_checkpoint_index >= len(self._checkpoint_locations):
            return None
        checkpoint = self._checkpoint_locations[self._next_checkpoint_index]
        return float(_location_distance(ego_location, checkpoint))

    def _active_checkpoint_count(self):
        return max(0, len(self._checkpoint_locations) - int(self._start_checkpoint_index))

    def _checkpoint_display_index(self):
        checkpoint_count = self._active_checkpoint_count()
        raw_index = max(0, int(self._next_checkpoint_index) - int(self._start_checkpoint_index))
        return min(raw_index, checkpoint_count)

    def _progress_to_route_index(self, target_progress_m):
        if not self._route_progress_m:
            return None
        target = float(target_progress_m)
        for idx, progress in enumerate(self._route_progress_m):
            if progress >= target:
                return idx
        return len(self._route_progress_m) - 1

    def _build_checkpoint_progress(self):
        progress_values = []
        for checkpoint in self._checkpoint_locations:
            progress_values.append(self._project_progress_m(checkpoint))
        return progress_values

    def _active_checkpoint_pairs(self):
        if self._finished or len(self._checkpoint_locations) < 2:
            return []
        if not self._started:
            start_idx = int(self._start_checkpoint_index)
            end_idx = min(start_idx + 1, len(self._checkpoint_locations) - 1)
            if end_idx <= start_idx:
                return []
            current_pair = (start_idx, end_idx)
        elif self._next_checkpoint_index >= len(self._checkpoint_locations):
            return []
        else:
            start_idx = max(0, self._next_checkpoint_index - 1)
            end_idx = min(start_idx + 1, len(self._checkpoint_locations) - 1)
            if end_idx <= start_idx:
                return []
            current_pair = (start_idx, end_idx)

        pairs = [current_pair]
        next_start_idx = current_pair[1]
        next_end_idx = next_start_idx + 1
        if next_end_idx < len(self._checkpoint_locations):
            pairs.append((next_start_idx, next_end_idx))
        return pairs

    def _route_segment_locations(self, start_idx, end_idx):
        if start_idx < 0 or end_idx >= len(self._checkpoint_locations) or end_idx <= start_idx:
            return []

        progress_values = self._checkpoint_progress_m
        start_progress = progress_values[start_idx] if start_idx < len(progress_values) else None
        end_progress = progress_values[end_idx] if end_idx < len(progress_values) else None
        if start_progress is None or end_progress is None:
            return []

        route_start_idx = self._progress_to_route_index(start_progress)
        route_end_idx = self._progress_to_route_index(end_progress)
        if route_start_idx is None or route_end_idx is None or route_end_idx <= route_start_idx:
            return []

        segment_locations = []
        for idx in range(route_start_idx, route_end_idx + 1):
            segment_locations.append(self._route_waypoints[idx].transform.location)
        return segment_locations

    def _draw_checkpoint_guide_pair(self, debug, lifetime, start_idx, end_idx, guide_color, thickness):
        segment_locations = self._route_segment_locations(start_idx, end_idx)

        if len(segment_locations) >= 2:
            for idx in range(len(segment_locations) - 1):
                debug.draw_line(
                    _shift_location(segment_locations[idx], z=0.55),
                    _shift_location(segment_locations[idx + 1], z=0.55),
                    thickness,
                    guide_color,
                    lifetime,
                    False,
                )
            return

        debug.draw_line(
            _shift_location(self._checkpoint_locations[start_idx], z=0.8),
            _shift_location(self._checkpoint_locations[end_idx], z=0.8),
            max(0.10, thickness - 0.02),
            guide_color,
            lifetime,
            False,
        )

    def _draw_active_checkpoint_guide(self, debug, lifetime):
        pairs = self._active_checkpoint_pairs()
        if not pairs:
            return

        active_guide_color = carla.Color(83, 69, 18)
        preview_guide_color = carla.Color(58, 48, 14)
        for pair_index, (start_idx, end_idx) in enumerate(pairs):
            guide_color = active_guide_color if pair_index == 0 else preview_guide_color
            thickness = 0.12 if pair_index == 0 else 0.09
            self._draw_checkpoint_guide_pair(
                debug,
                lifetime,
                start_idx,
                end_idx,
                guide_color,
                thickness,
            )

    def _advance_checkpoints(self, ego_location):
        radius = float(self.preset.checkpoint_radius_m)
        while self._next_checkpoint_index < len(self._checkpoint_locations):
            checkpoint = self._checkpoint_locations[self._next_checkpoint_index]
            if _location_distance(ego_location, checkpoint) > radius:
                break
            self._next_checkpoint_index += 1

    def _draw_checkpoint_markers(self, world):
        if not self._checkpoint_locations or not bool(getattr(self.preset, 'draw_debug_markers', True)):
            return
        debug = getattr(world.world, 'debug', None)
        if debug is None:
            return

        lifetime = max(0.2, 1.0 / max(1.0, float(CLIENT_FPS_LIMIT)) * 2.5)
        next_color = carla.Color(255, 220, 0)

        if not self._finished and self._next_checkpoint_index < len(self._checkpoint_locations):
            idx = int(self._next_checkpoint_index)
            checkpoint = self._checkpoint_locations[idx]
            point_location = _shift_location(checkpoint, z=1.2)
            text_location = _shift_location(checkpoint, z=2.2)

            if idx == int(self._start_checkpoint_index) and not self._started:
                label = "START"
            elif idx == len(self._checkpoint_locations) - 1:
                label = "FINISH"
            else:
                label = "CP %d" % max(1, idx - int(self._start_checkpoint_index))

            debug.draw_point(point_location, 0.35, next_color, lifetime, False)
            debug.draw_string(text_location, label, False, next_color, lifetime, False)
        self._draw_active_checkpoint_guide(debug, lifetime)

    def _spawn_lead_vehicle(self, world):
        if not self._route_progress_m:
            return
        target_progress = min(
            float(self.preset.lead_spawn_distance_m),
            float(self._route_progress_m[-1]),
        )
        waypoint = self._lead_spawn_waypoint(target_progress)
        if waypoint is None:
            return
        transform = carla.Transform(
            carla.Location(
                x=float(waypoint.transform.location.x),
                y=float(waypoint.transform.location.y),
                z=float(waypoint.transform.location.z) + 0.2,
            ),
            carla.Rotation(
                pitch=float(waypoint.transform.rotation.pitch),
                yaw=float(waypoint.transform.rotation.yaw),
                roll=float(waypoint.transform.rotation.roll),
            ),
        )
        blueprint = world._resolve_blueprint(self.preset.lead_blueprint_id, 'vehicle.*')
        blueprint.set_attribute('role_name', 'scenario_lead')
        if blueprint.has_attribute('color'):
            colors = list(blueprint.get_attribute('color').recommended_values)
            if colors:
                blueprint.set_attribute('color', colors[0])
        actor = world.world.try_spawn_actor(blueprint, transform)
        if actor is None:
            return
        self._lead_vehicle = actor
        self._actors.append(actor)
        if bool(getattr(self.preset, "lead_hold_until_start", False)):
            self._freeze_lead_vehicle()
            return
        self._enable_lead_autopilot()

    def _enable_lead_autopilot(self):
        actor = self._lead_vehicle
        if actor is None or not actor.is_alive or self._traffic_manager is None:
            return
        tm_port = int(self._traffic_manager.get_port())
        try:
            actor.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0, hand_brake=False))
        except RuntimeError:
            pass
        actor.set_autopilot(True, tm_port)
        try:
            self._traffic_manager.auto_lane_change(actor, False)
        except RuntimeError:
            pass
        try:
            self._traffic_manager.distance_to_leading_vehicle(actor, 1.0)
        except RuntimeError:
            pass
        self._lead_response_active = False
        self._set_lead_speed_reduction_pct(float(self.preset.lead_speed_reduction_pct))
        self._lead_autopilot_enabled = True

    def _set_lead_speed_reduction_pct(self, reduction_pct):
        actor = self._lead_vehicle
        if actor is None or not actor.is_alive or self._traffic_manager is None:
            return
        reduction_pct = max(-100.0, min(100.0, float(reduction_pct)))
        if (
            self._lead_speed_reduction_pct is not None
            and abs(float(self._lead_speed_reduction_pct) - reduction_pct) < 0.25
        ):
            return
        try:
            self._traffic_manager.vehicle_percentage_speed_difference(actor, reduction_pct)
        except RuntimeError:
            return
        self._lead_speed_reduction_pct = float(reduction_pct)

    def _update_lead_speed_response(self, ego_waypoint, ego_speed_mps):
        base_reduction_pct = float(self.preset.lead_speed_reduction_pct)
        if not bool(getattr(self.preset, "lead_reactive_speed_enabled", False)):
            self._lead_response_active = False
            self._set_lead_speed_reduction_pct(base_reduction_pct)
            return

        actor = self._lead_vehicle
        if actor is None or not actor.is_alive or self._traffic_manager is None:
            self._lead_response_active = False
            return

        lead_gap_m = self._last_lead_gap_m
        lead_distance_m = self._last_lead_distance_m
        if lead_gap_m is None or lead_distance_m is None or ego_waypoint is None:
            self._lead_response_active = False
            self._set_lead_speed_reduction_pct(base_reduction_pct)
            return

        trigger_distance_m = max(1.0, float(getattr(self.preset, "lead_reactive_trigger_distance_m", 24.0)))
        trigger_gap_m = max(1.0, float(getattr(self.preset, "lead_reactive_trigger_gap_m", 28.0)))
        release_gap_m = max(0.0, float(getattr(self.preset, "lead_reactive_release_gap_m", 6.0)))

        in_start_lane = (
            self._start_lane_id is not None
            and self._start_road_id is not None
            and int(ego_waypoint.road_id) == int(self._start_road_id)
            and int(ego_waypoint.lane_id) == int(self._start_lane_id)
        )
        in_passing_lane = (
            self._start_lane_id is not None
            and self._start_road_id is not None
            and int(ego_waypoint.road_id) == int(self._start_road_id)
            and int(ego_waypoint.lane_id) != int(self._start_lane_id)
        )

        close_behind = (
            in_start_lane
            and lead_gap_m >= 0.0
            and lead_gap_m <= trigger_gap_m
            and lead_distance_m <= trigger_distance_m
        )
        passing_attempt = (
            in_passing_lane
            and lead_gap_m >= -release_gap_m
            and lead_gap_m <= (trigger_gap_m + 10.0)
            and lead_distance_m <= (trigger_distance_m + 8.0)
        )
        keep_active = (
            self._lead_response_active
            and lead_gap_m > -release_gap_m
            and lead_distance_m <= (trigger_distance_m + 8.0)
        )
        response_active = bool(close_behind or passing_attempt or keep_active)
        if not response_active:
            self._lead_response_active = False
            self._set_lead_speed_reduction_pct(base_reduction_pct)
            return

        speed_limit_kph = float(actor.get_speed_limit())
        speed_limit_mps = float(speed_limit_kph) / 3.6 if speed_limit_kph > 1.0 else 0.0
        if speed_limit_mps <= 0.0:
            self._lead_response_active = False
            self._set_lead_speed_reduction_pct(base_reduction_pct)
            return

        base_target_speed_mps = speed_limit_mps * max(0.0, 1.0 - (base_reduction_pct / 100.0))
        target_speed_ratio = max(
            0.0,
            float(getattr(self.preset, "lead_reactive_target_speed_ratio", 0.85)),
        )
        min_reduction_pct = max(
            -100.0,
            min(100.0, float(getattr(self.preset, "lead_reactive_min_reduction_pct", 22.0))),
        )

        # A ratio of 1.12 means the lead tries to hold 112% of ego speed
        # while the response is active, subject to the configured TM cap.
        desired_target_speed_mps = max(
            base_target_speed_mps,
            float(ego_speed_mps) * target_speed_ratio,
        )
        desired_target_speed_mps = min(
            desired_target_speed_mps,
            speed_limit_mps * max(0.0, 1.0 - (min_reduction_pct / 100.0)),
        )
        if desired_target_speed_mps <= (base_target_speed_mps + 0.1):
            self._lead_response_active = False
            self._set_lead_speed_reduction_pct(base_reduction_pct)
            return

        desired_reduction_pct = 100.0 * (1.0 - (desired_target_speed_mps / speed_limit_mps))
        self._lead_response_active = True
        self._set_lead_speed_reduction_pct(desired_reduction_pct)

    def _freeze_lead_vehicle(self):
        actor = self._lead_vehicle
        if actor is None or not actor.is_alive:
            return
        try:
            if self._traffic_manager is not None:
                actor.set_autopilot(False, int(self._traffic_manager.get_port()))
            else:
                actor.set_autopilot(False)
        except RuntimeError:
            pass
        try:
            actor.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=True))
        except RuntimeError:
            pass
        self._lead_autopilot_enabled = False
        self._lead_response_active = False

    def destroy(self):
        for actor in reversed(self._actors):
            try:
                if actor is not None and actor.is_alive:
                    actor.destroy()
            except RuntimeError:
                pass
        self._actors = []
        self._lead_vehicle = None
        self._lead_autopilot_enabled = False
        self._lead_response_active = False
        self._lead_speed_reduction_pct = None

    def setup(self, world):
        self.destroy()
        self._route_waypoints = []
        self._route_progress_m = []
        self._checkpoint_locations = []
        self._checkpoint_progress_m = []
        self._start_location = None
        self._finish_location = None
        self._status = "waiting_start"
        self._started = False
        self._finished = False
        self._success = False
        self._failure_reason = ""
        self._start_sim_time = None
        self._finish_sim_time = None
        self._completion_time_s = None
        self._next_checkpoint_index = 0
        self._start_checkpoint_index = 0
        self._last_progress_m = None
        self._overtake_objective_met = False
        self._last_next_checkpoint_distance_m = None
        self._last_lead_distance_m = None
        self._last_lead_gap_m = None
        self._last_current_lane_id = None
        self._last_sim_time = 0.0
        self._should_exit = False
        self._lead_autopilot_enabled = False
        self._lead_response_active = False
        self._lead_speed_reduction_pct = None

        start_waypoint = self._get_anchor_waypoint(world.world.get_map())
        if start_waypoint is None:
            start_waypoint = world.world.get_map().get_waypoint(
                world.player.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
        if start_waypoint is None:
            self._status = "setup_failed"
            self._failure_reason = "no_start_waypoint"
            return

        self._start_lane_id = int(start_waypoint.lane_id)
        self._start_road_id = int(start_waypoint.road_id)
        self._route_waypoints, self._route_progress_m = self._build_route(start_waypoint)
        self._checkpoint_locations = self._build_checkpoints()
        self._checkpoint_progress_m = self._build_checkpoint_progress()
        configured_start_checkpoint_index = max(
            0,
            int(getattr(self.preset, "start_checkpoint_index", 0) or 0),
        )
        if self._checkpoint_locations:
            remaining_checkpoints = len(self._checkpoint_locations) - configured_start_checkpoint_index
            if remaining_checkpoints < 2:
                self._status = "setup_failed"
                self._failure_reason = "start_checkpoint_index_out_of_range"
                world.hud.notification("Scenario setup failed: invalid start checkpoint", seconds=4.0)
                print(
                    "[scenario] setup failed: start checkpoint index %d leaves fewer than 2 checkpoints"
                    % configured_start_checkpoint_index
                )
                return
            self._start_checkpoint_index = configured_start_checkpoint_index
            self._next_checkpoint_index = int(self._start_checkpoint_index)
            self._start_location = self._checkpoint_locations[self._start_checkpoint_index]
            self._finish_location = self._checkpoint_locations[-1]
        elif self._route_waypoints:
            self._start_location = self._route_waypoints[0].transform.location
            self._finish_location = self._route_waypoints[-1].transform.location
        else:
            self._status = "setup_failed"
            self._failure_reason = "no_route_generated"
            world.hud.notification("Scenario setup failed: no route", seconds=4.0)
            print("[scenario] setup failed: no route generated")
            return

        if self.preset.kind == "overtake":
            self._spawn_lead_vehicle(world)
            if self._lead_vehicle is None:
                self._status = "setup_failed"
                self._failure_reason = "lead_vehicle_spawn_failed"
                world.hud.notification("Scenario setup failed: no lead vehicle", seconds=4.0)
                print("[scenario] setup failed: lead vehicle spawn failed")
                return

        checkpoint_count = self._active_checkpoint_count()
        world.hud.notification(
            "Scenario: %s (%d checkpoints)" % (self.preset.name, checkpoint_count),
            seconds=4.0,
        )
        print(
            "[scenario] prepared %s on %s with %d active checkpoints (start checkpoint index %d)" % (
                self.preset.name,
                self.preset.map_name,
                checkpoint_count,
                int(self._start_checkpoint_index),
            )
        )
        self._draw_checkpoint_markers(world)

    def _finish(self, world, sim_time, success, reason=""):
        if self._finished:
            return
        self._finished = True
        self._success = bool(success)
        self._failure_reason = str(reason or "")
        self._finish_sim_time = float(sim_time)
        self._completion_time_s = (
            float(self._finish_sim_time - self._start_sim_time)
            if self._start_sim_time is not None
            else None
        )
        self._status = "success" if self._success else "failed"
        self._should_exit = True
        message = (
            "%s complete" % self.preset.name
            if self._success
            else "%s failed: %s" % (self.preset.name, self._failure_reason or "unknown")
        )
        world.hud.notification(message, seconds=4.0)
        print("[scenario] %s" % message)
        self._draw_checkpoint_markers(world)

    def should_exit(self):
        return bool(self._should_exit)

    def abort(self, world, reason=MANUAL_EARLY_CLOSE_REASON):
        if self._finished:
            return False
        if self._status == "setup_failed":
            self._should_exit = True
            return False
        sim_time = float(getattr(world.hud, 'simulation_time', self._last_sim_time))
        self._finish(world, sim_time, False, reason)
        return True

    def tick(self, world):
        if self._status == "setup_failed":
            self._should_exit = True
            return
        if self._finished:
            self._draw_checkpoint_markers(world)
            return

        sim_time = float(getattr(world.hud, 'simulation_time', 0.0))
        self._last_sim_time = float(sim_time)
        ego_location = world.player.get_location()
        ego_waypoint = world.world.get_map().get_waypoint(
            ego_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        self._last_current_lane_id = int(ego_waypoint.lane_id) if ego_waypoint is not None else None
        self._last_progress_m = self._project_progress_m(ego_location)
        self._last_next_checkpoint_distance_m = self._distance_to_next_checkpoint(ego_location)
        self._draw_checkpoint_markers(world)

        if not self._started:
            if self._start_location is not None and _location_distance(ego_location, self._start_location) <= float(self.preset.checkpoint_radius_m):
                self._started = True
                self._status = "active"
                self._start_sim_time = float(sim_time)
                self._next_checkpoint_index = (
                    min(self._start_checkpoint_index + 1, len(self._checkpoint_locations))
                    if self._checkpoint_locations
                    else 0
                )
                self._last_next_checkpoint_distance_m = self._distance_to_next_checkpoint(ego_location)
                if self.preset.kind == "overtake" and bool(getattr(self.preset, "lead_hold_until_start", False)):
                    self._enable_lead_autopilot()
                world.hud.notification("Scenario started", seconds=2.0)
        else:
            self._advance_checkpoints(ego_location)
            self._last_next_checkpoint_distance_m = self._distance_to_next_checkpoint(ego_location)

        if self._started and not self._finished and self._start_sim_time is not None:
            elapsed = float(sim_time - self._start_sim_time)
            timeout_s = getattr(self.preset, "timeout_s", None)
            if timeout_s is not None and float(timeout_s) > 0.0 and elapsed > float(timeout_s):
                self._finish(world, sim_time, False, "timeout")
                return

        if not self._started or self._finished:
            return

        if self.preset.kind == "lane_keep":
            if self._checkpoint_locations and self._next_checkpoint_index >= len(self._checkpoint_locations):
                self._finish(world, sim_time, True, "")
            return

        if self.preset.kind != "overtake":
            return

        if self._lead_vehicle is None or not self._lead_vehicle.is_alive:
            self._finish(world, sim_time, False, "lead_vehicle_missing")
            return

        lead_location = self._lead_vehicle.get_location()
        ego_speed_mps = _velocity_speed_mps(world.player.get_velocity())
        self._last_lead_distance_m = _location_distance(ego_location, lead_location)
        lead_progress_m = self._project_progress_m(lead_location)
        if self._last_progress_m is not None and lead_progress_m is not None:
            self._last_lead_gap_m = float(lead_progress_m - self._last_progress_m)
        else:
            self._last_lead_gap_m = None
        self._update_lead_speed_response(ego_waypoint, ego_speed_mps)

        in_start_lane = True
        if bool(self.preset.require_return_to_start_lane) and ego_waypoint is not None:
            in_start_lane = (
                int(ego_waypoint.road_id) == int(self._start_road_id)
                and int(ego_waypoint.lane_id) == int(self._start_lane_id)
            )

        if (
            self._last_progress_m is not None
            and lead_progress_m is not None
            and self._last_progress_m >= lead_progress_m + float(self.preset.overtake_finish_margin_m)
            and in_start_lane
        ):
            if not self._overtake_objective_met:
                self._overtake_objective_met = True
                self._status = "finish_gate_pending"
                world.hud.notification("Overtake complete, reach FINISH", seconds=2.5)

        if self._checkpoint_locations and self._next_checkpoint_index >= len(self._checkpoint_locations):
            if self._overtake_objective_met:
                self._finish(world, sim_time, True, "")
            else:
                self._finish(world, sim_time, False, "finish_without_overtake")
            return

        if not self._checkpoint_locations and self._overtake_objective_met:
            self._finish(world, sim_time, True, "")

    def snapshot(self):
        elapsed_s = None
        if self._started and self._start_sim_time is not None:
            if self._finished and self._completion_time_s is not None:
                elapsed_s = float(self._completion_time_s)
            else:
                elapsed_s = max(0.0, float(self._last_sim_time - self._start_sim_time))
        checkpoint_count = self._active_checkpoint_count()
        checkpoint_index = self._checkpoint_display_index()
        return {
            "scenario_name": str(self.preset.name),
            "scenario_kind": str(self.preset.kind),
            "scenario_status": str(self._status),
            "scenario_started": bool(self._started),
            "scenario_finished": bool(self._finished),
            "scenario_success": bool(self._success) if self._finished else None,
            "scenario_failure_reason": str(self._failure_reason),
            "scenario_elapsed_s": (
                float(elapsed_s)
                if elapsed_s is not None
                else None
            ),
            "scenario_completion_time_s": (
                float(self._completion_time_s)
                if self._completion_time_s is not None
                else None
            ),
            "scenario_checkpoint_index": int(checkpoint_index),
            "scenario_checkpoint_count": int(checkpoint_count),
            "scenario_progress_m": (
                float(self._last_progress_m)
                if self._last_progress_m is not None
                else None
            ),
            "scenario_objective_met": bool(self._overtake_objective_met),
            "scenario_next_checkpoint_distance_m": (
                float(self._last_next_checkpoint_distance_m)
                if self._last_next_checkpoint_distance_m is not None
                else None
            ),
            "scenario_lead_distance_m": (
                float(self._last_lead_distance_m)
                if self._last_lead_distance_m is not None
                else None
            ),
            "scenario_lead_gap_m": (
                float(self._last_lead_gap_m)
                if self._last_lead_gap_m is not None
                else None
            ),
            "scenario_lead_response_active": bool(self._lead_response_active),
            "scenario_lead_speed_reduction_pct": (
                float(self._lead_speed_reduction_pct)
                if self._lead_speed_reduction_pct is not None
                else None
            ),
            "scenario_current_lane_id": (
                int(self._last_current_lane_id)
                if self._last_current_lane_id is not None
                else None
            ),
            "scenario_target_lane_id": (
                int(self._start_lane_id)
                if self._start_lane_id is not None
                else None
            ),
        }

    def hud_lines(self, sim_time):
        if not self.preset:
            return []
        lines = [
            "Scenario: %s" % self.preset.name,
            "Scenario status: %s" % self._status,
        ]
        if self._checkpoint_locations:
            lines.append(
                "Checkpoints: %d/%d" % (
                    self._checkpoint_display_index(),
                    self._active_checkpoint_count(),
                )
            )
        if self._last_next_checkpoint_distance_m is not None:
            lines.append("Next checkpoint: %.1fm" % self._last_next_checkpoint_distance_m)
        if self._started and self._start_sim_time is not None:
            elapsed = (
                float(self._completion_time_s)
                if self._finished and self._completion_time_s is not None
                else float(sim_time - self._start_sim_time)
            )
            lines.append("Scenario time: %.1fs" % max(0.0, elapsed))
        if self.preset.kind == "overtake":
            lines.append(
                "Overtake objective: %s" % ("done" if self._overtake_objective_met else "pending")
            )
            if self._lead_speed_reduction_pct is not None:
                lines.append(
                    "Lead response: %s (TM %.1f%%)"
                    % (
                        "active" if self._lead_response_active else "base",
                        float(self._lead_speed_reduction_pct),
                    )
                )
        if self._last_lead_distance_m is not None:
            lines.append("Lead distance: %.1fm" % self._last_lead_distance_m)
        return lines


class AmbientTrafficManager(object):
    def __init__(self, client, carla_world, traffic_manager=None, vehicle_count=0, pedestrian_count=0):
        self._client = client
        self._world = carla_world
        self._traffic_manager = traffic_manager
        self._target_vehicle_count = max(0, int(vehicle_count or 0))
        self._target_pedestrian_count = max(0, int(pedestrian_count or 0))
        self._vehicle_actor_ids = []
        self._walker_actor_ids = []
        self._walker_controller_ids = []
        self._spawned_vehicle_count = 0
        self._spawned_pedestrian_count = 0

    def enabled(self):
        return self._target_vehicle_count > 0 or self._target_pedestrian_count > 0

    def summary_text(self):
        if not self.enabled():
            return ""
        return "Ambient traffic: %d vehicles, %d pedestrians" % (
            int(self._spawned_vehicle_count),
            int(self._spawned_pedestrian_count),
        )

    def spawn(self, hero_actor=None):
        self.destroy()
        if not self.enabled():
            return
        hero_location = hero_actor.get_location() if hero_actor is not None else None
        self._spawned_vehicle_count = self._spawn_vehicles(hero_location)
        self._spawned_pedestrian_count = self._spawn_pedestrians(hero_location)

    def destroy(self):
        controller_ids = list(self._walker_controller_ids)
        for controller_id in controller_ids:
            controller = self._world.get_actor(int(controller_id))
            if controller is None:
                continue
            try:
                controller.stop()
            except RuntimeError:
                pass

        destroy_ids = controller_ids + list(self._walker_actor_ids) + list(self._vehicle_actor_ids)
        if self._client is not None and destroy_ids:
            try:
                self._client.apply_batch([carla.command.DestroyActor(actor_id) for actor_id in destroy_ids])
            except RuntimeError:
                pass

        self._vehicle_actor_ids = []
        self._walker_actor_ids = []
        self._walker_controller_ids = []
        self._spawned_vehicle_count = 0
        self._spawned_pedestrian_count = 0

    def _spawn_vehicles(self, hero_location):
        if self._target_vehicle_count <= 0 or self._client is None or self._traffic_manager is None:
            return 0

        spawn_points = list(self._world.get_map().get_spawn_points())
        if not spawn_points:
            return 0

        if hero_location is not None:
            filtered_spawn_points = [
                spawn_point
                for spawn_point in spawn_points
                if _location_distance(spawn_point.location, hero_location) >= 20.0
            ]
            if filtered_spawn_points:
                spawn_points = filtered_spawn_points

        random.shuffle(spawn_points)
        blueprints = [
            blueprint
            for blueprint in self._world.get_blueprint_library().filter('vehicle.*')
            if (
                not blueprint.has_attribute('number_of_wheels')
                or int(blueprint.get_attribute('number_of_wheels')) == 4
            )
        ]
        if not blueprints:
            return 0

        tm_port = int(self._traffic_manager.get_port())
        batch = []
        for spawn_point in spawn_points[: self._target_vehicle_count]:
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                colors = list(blueprint.get_attribute('color').recommended_values)
                if colors:
                    blueprint.set_attribute('color', random.choice(colors))
            if blueprint.has_attribute('driver_id'):
                driver_ids = list(blueprint.get_attribute('driver_id').recommended_values)
                if driver_ids:
                    blueprint.set_attribute('driver_id', random.choice(driver_ids))
            if blueprint.has_attribute('role_name'):
                blueprint.set_attribute('role_name', 'ambient_traffic')
            batch.append(
                carla.command.SpawnActor(blueprint, spawn_point).then(
                    carla.command.SetAutopilot(carla.command.FutureActor, True, tm_port)
                )
            )

        spawned_ids = []
        for response in self._client.apply_batch_sync(batch, False):
            if response.error:
                continue
            spawned_ids.append(int(response.actor_id))
        self._vehicle_actor_ids = spawned_ids
        return len(spawned_ids)

    def _sample_pedestrian_spawn_points(self, hero_location):
        spawn_points = []
        attempts_remaining = max(20, int(self._target_pedestrian_count) * 8)
        while len(spawn_points) < self._target_pedestrian_count and attempts_remaining > 0:
            attempts_remaining -= 1
            location = self._world.get_random_location_from_navigation()
            if location is None:
                continue
            if hero_location is not None and _location_distance(location, hero_location) < 12.0:
                continue
            if any(_location_distance(location, point.location) < 1.5 for point in spawn_points):
                continue
            spawn_points.append(
                carla.Transform(
                    carla.Location(
                        x=float(location.x),
                        y=float(location.y),
                        z=float(location.z),
                    )
                )
            )
        return spawn_points

    def _spawn_pedestrians(self, hero_location):
        if self._target_pedestrian_count <= 0 or self._client is None:
            return 0

        walker_blueprints = list(self._world.get_blueprint_library().filter('walker.pedestrian.*'))
        if not walker_blueprints:
            return 0

        try:
            controller_blueprint = self._world.get_blueprint_library().find('controller.ai.walker')
        except RuntimeError:
            return 0

        walker_spawn_points = self._sample_pedestrian_spawn_points(hero_location)
        if not walker_spawn_points:
            return 0

        pedestrian_running_fraction = 0.05
        pedestrian_crossing_fraction = 0.20
        walker_speeds = []
        walker_batch = []
        for spawn_point in walker_spawn_points:
            walker_blueprint = random.choice(walker_blueprints)
            if walker_blueprint.has_attribute('is_invincible'):
                walker_blueprint.set_attribute('is_invincible', 'false')
            speed = 1.4
            if walker_blueprint.has_attribute('speed'):
                speed_values = list(walker_blueprint.get_attribute('speed').recommended_values)
                if speed_values:
                    walk_speed = float(speed_values[1]) if len(speed_values) > 1 else float(speed_values[0])
                    run_speed = float(speed_values[2]) if len(speed_values) > 2 else walk_speed
                    speed = run_speed if random.random() < pedestrian_running_fraction else walk_speed
            walker_speeds.append(float(speed))
            walker_batch.append(carla.command.SpawnActor(walker_blueprint, spawn_point))

        spawned_walkers = []
        for response, speed in zip(self._client.apply_batch_sync(walker_batch, True), walker_speeds):
            if response.error:
                continue
            spawned_walkers.append((int(response.actor_id), float(speed)))
        if not spawned_walkers:
            return 0

        controller_batch = []
        for walker_actor_id, _ in spawned_walkers:
            controller_batch.append(
                carla.command.SpawnActor(controller_blueprint, carla.Transform(), walker_actor_id)
            )

        walker_actor_ids = []
        walker_controller_ids = []
        walker_controller_speeds = []
        failed_walker_ids = []
        for response, (walker_actor_id, speed) in zip(
            self._client.apply_batch_sync(controller_batch, True),
            spawned_walkers,
        ):
            if response.error:
                failed_walker_ids.append(int(walker_actor_id))
                continue
            walker_actor_ids.append(int(walker_actor_id))
            walker_controller_ids.append(int(response.actor_id))
            walker_controller_speeds.append(float(speed))

        if failed_walker_ids:
            try:
                self._client.apply_batch([carla.command.DestroyActor(actor_id) for actor_id in failed_walker_ids])
            except RuntimeError:
                pass

        self._walker_actor_ids = walker_actor_ids
        self._walker_controller_ids = walker_controller_ids

        try:
            self._world.set_pedestrians_cross_factor(float(pedestrian_crossing_fraction))
        except RuntimeError:
            pass

        try:
            world_settings = self._world.get_settings()
            if bool(getattr(world_settings, "synchronous_mode", False)):
                self._world.tick()
            else:
                self._world.wait_for_tick()
        except RuntimeError:
            pass

        controller_actors = {}
        if self._walker_controller_ids:
            try:
                controller_actors = {
                    int(actor.id): actor
                    for actor in self._world.get_actors(list(self._walker_controller_ids))
                    if actor is not None
                }
            except (RuntimeError, TypeError):
                controller_actors = {}

        for controller_id, speed in zip(self._walker_controller_ids, walker_controller_speeds):
            controller = controller_actors.get(int(controller_id))
            if controller is None:
                controller = self._world.get_actor(int(controller_id))
            if controller is None:
                continue
            try:
                controller.start()
                destination = self._world.get_random_location_from_navigation()
                if destination is not None:
                    controller.go_to_location(destination)
                controller.set_max_speed(float(speed))
            except RuntimeError:
                pass

        return len(self._walker_actor_ids)

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


def parse_resolution(value):
    try:
        width_str, height_str = value.lower().split('x', 1)
        width = int(width_str)
        height = int(height_str)
    except ValueError as exc:
        raise ValueError('resolution must look like WIDTHxHEIGHT') from exc
    if width <= 0 or height <= 0:
        raise ValueError('resolution values must be positive')
    return width, height


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(
        self,
        carla_world,
        hud,
        actor_filter,
        client=None,
        scenario_preset=None,
        ambient_vehicle_count=0,
        ambient_pedestrian_count=0,
    ):
        self.world = carla_world
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = actor_filter
        self._client = client
        self._traffic_manager = client.get_trafficmanager() if client is not None else None
        self._scenario_preset = scenario_preset
        self._scenario_runtime = ScenarioRuntime(scenario_preset, self._traffic_manager) if scenario_preset else None
        self._ambient_traffic = AmbientTrafficManager(
            client,
            carla_world,
            self._traffic_manager,
            vehicle_count=ambient_vehicle_count,
            pedestrian_count=ambient_pedestrian_count,
        )
        self._scenario_exit_requested = False
        self._apply_runtime_world_settings()
        self.restart()
        self.world.on_tick(hud.on_world_tick)

    def _resolve_blueprint(self, preferred_id='', fallback_filter='vehicle.*'):
        library = self.world.get_blueprint_library()
        blueprint = None
        preferred_text = str(preferred_id or '').strip()
        if preferred_text:
            try:
                blueprint = library.find(preferred_text)
            except RuntimeError:
                blueprint = None
        if blueprint is None:
            candidates = list(library.filter(str(fallback_filter or self._actor_filter)))
            if not candidates:
                raise RuntimeError('no matching CARLA vehicle blueprints found')
            blueprint = random.choice(candidates)
        return blueprint

    def _apply_runtime_world_settings(self):
        if not LOW_GRAPHICS_MODE:
            return
        settings = self.world.get_settings()
        changed = False
        if settings.no_rendering_mode != bool(LOW_GRAPHICS_NO_RENDERING):
            settings.no_rendering_mode = bool(LOW_GRAPHICS_NO_RENDERING)
            changed = True
        if changed:
            self.world.apply_settings(settings)

    def restart(self):
        cam_transform_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        self._scenario_exit_requested = False
        if self.player is not None:
            self.destroy()
        if self._scenario_preset is not None:
            blueprint = self._resolve_blueprint(self._scenario_preset.ego_blueprint_id, self._actor_filter)
        else:
            blueprint = self._resolve_blueprint('', self._actor_filter)
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            colors = list(blueprint.get_attribute('color').recommended_values)
            if colors:
                blueprint.set_attribute('color', colors[0] if self._scenario_preset is not None else random.choice(colors))
        while self.player is None:
            if self._scenario_runtime is not None:
                spawn_point = self._scenario_runtime.get_ego_spawn_transform(self.world.get_map())
            else:
                spawn_points = self.world.get_map().get_spawn_points()
                spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.camera_manager = CameraManager(self.player, self.hud, transform_index=cam_transform_index)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)
        if self._scenario_runtime is not None:
            self._scenario_runtime.setup(self)
        if self._ambient_traffic.enabled():
            self._ambient_traffic.spawn(self.player)
            summary_text = self._ambient_traffic.summary_text()
            if summary_text:
                self.hud.notification(summary_text, seconds=4.0)
                print("[traffic] %s" % summary_text)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock):
        if self._scenario_runtime is not None:
            self._scenario_runtime.tick(self)
            self._scenario_exit_requested = self._scenario_runtime.should_exit()
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def get_scenario_snapshot(self):
        if self._scenario_runtime is None:
            return {}
        return self._scenario_runtime.snapshot()

    def get_scenario_hud_lines(self):
        if self._scenario_runtime is None:
            return []
        return self._scenario_runtime.hud_lines(getattr(self.hud, 'simulation_time', 0.0))

    def scenario_exit_requested(self):
        return bool(self._scenario_exit_requested)

    def abort_active_scenario(self, reason=MANUAL_EARLY_CLOSE_REASON):
        if self._scenario_runtime is None:
            return False
        aborted = self._scenario_runtime.abort(self, reason=reason)
        self._scenario_exit_requested = self._scenario_runtime.should_exit()
        return bool(aborted)

    def destroy(self):
        if self._scenario_runtime is not None:
            self._scenario_runtime.destroy()
        if self._ambient_traffic is not None:
            self._ambient_traffic.destroy()
        sensors = [
            self.camera_manager.sensor if self.camera_manager is not None else None,
            self.collision_sensor.sensor if self.collision_sensor is not None else None,
            self.lane_invasion_sensor.sensor if self.lane_invasion_sensor is not None else None,
        ]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()
            self.player = None

# ==============================================================================
# -- DualControl -----------------------------------------------------------
# ==============================================================================


class DualControl(object):
    def __init__(self, world, start_in_autopilot, carla_log_path='', realtime_log_path=''):
        self._autopilot_enabled = start_in_autopilot
        if not isinstance(world.player, carla.Vehicle):
            raise NotImplementedError("Actor type not supported")
        self._control = carla.VehicleControl()
        world.player.set_autopilot(self._autopilot_enabled)
        
        # Store references so gesture actions can affect vehicle + HUD
        self._player = world.player
        self._hud = world.hud
        self._joystick = None
        self._throttle_idx = None
        self._brake_idx = None
        self._handbrake_idx = None
        self._vehicle_max_steer_angle_rad = _vehicle_max_steer_angle_rad(world.player)
        self._init_wheel_controls()
        
        # Gesture freshness (optional safety: ignore stale labels)
        self._gesture_max_age = float(CARLA_TUNING.gesture_max_age_s)
        self._latest_published = None
        self._latest_steer_key = "neutral"
        self._latest_applied_steer_key = "neutral"
        self._last_lane_invasion_count = 0
        self._drive_logger = DriveCSVLogger(carla_log_path) if carla_log_path else None
        self._exit_log_written = False
        self._realtime_log_path = str(realtime_log_path or '').strip()
        
        # Gestures -> steering only (NO throttle/brake from gestures)
        self._gesture_to_steer = {
            "left": float(CARLA_TUNING.steer_left),
            "right": float(CARLA_TUNING.steer_right),
            "left_strong": float(CARLA_TUNING.steer_left_strong),
            "right_strong": float(CARLA_TUNING.steer_right_strong),
            "neutral": float(CARLA_TUNING.steer_neutral),
        }
        self._active_steer_dwell_frames = max(1, int(CARLA_TUNING.active_steer_dwell_frames))
        self._neutral_steer_dwell_frames = max(1, int(CARLA_TUNING.neutral_steer_dwell_frames))
        self._applied_steer_key = "neutral"
        self._pending_steer_key = None
        self._pending_steer_count = 0
        self._reverse_toggle_cooldown_s = float(REVERSE_TOGGLE_COOLDOWN_S)
        self._reverse_toggle_max_speed_mps = float(REVERSE_TOGGLE_MAX_SPEED_MPS)
        self._last_reverse_toggle_ts = -1e9
        self._prev_reverse_toggle_active = False
        print(
            f"[gesture] runtime tuning: {RUNTIME_TUNING_NAME} "
            f"(gesture_max_age={self._gesture_max_age:.2f}s, "
            f"active_dwell={self._active_steer_dwell_frames}, "
            f"neutral_dwell={self._neutral_steer_dwell_frames})"
        )
        scenario_name = getattr(getattr(world, "_scenario_preset", None), "name", "")
        if scenario_name:
            print(f"[scenario] active scenario: {scenario_name}")
        
        # Strong steering holds the indicators on; reverse is edge-triggered.
        self._signal_state = "off"  # "off" | "left" | "right"
        
        self._gesture_thread = None
        self._start_gesture_thread()

    def close(self):
        if self._drive_logger is not None:
            self._drive_logger.close()
            self._drive_logger = None

    def finalize_exit(self, world):
        self._write_exit_log(world)

    def _init_wheel_controls(self):
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count <= 0:
            print("[wheel] no steering wheel detected; using keyboard throttle/brake")
            return
        if joystick_count > 1:
            raise ValueError("Please connect just one joystick")

        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()

        parser = ConfigParser()
        wheel_config_path = os.path.join(SCRIPT_DIR, 'wheel_config.ini')
        if not parser.read(wheel_config_path):
            raise FileNotFoundError("wheel config not found: %s" % wheel_config_path)

        self._throttle_idx = int(parser.get('G29 Racing Wheel', 'throttle'))
        self._brake_idx = int(parser.get('G29 Racing Wheel', 'brake'))
        self._handbrake_idx = int(parser.get('G29 Racing Wheel', 'handbrake'))
        print("[wheel] joystick controls enabled from %s" % wheel_config_path)

    def _get_live_player(self):
        player = self._player
        if player is None:
            return None
        if hasattr(player, "is_alive") and not player.is_alive:
            return None
        return player

    def _reset_vehicle_state_for_respawn(self):
        self._signal_state = "off"
        self._last_reverse_toggle_ts = -1e9
        self._last_lane_invasion_count = 0
        self._prev_reverse_toggle_active = False
        self._reset_steer_dwell_candidate()
        self._applied_steer_key = "neutral"
        self._latest_steer_key = "neutral"
        self._latest_applied_steer_key = "neutral"

    def _sync_world_refs(self, world, reset_state_on_change: bool = False):
        self._hud = getattr(world, "hud", None)
        player = getattr(world, "player", None)
        if player is self._player:
            return False

        self._player = player
        if not isinstance(player, carla.Vehicle):
            raise NotImplementedError("Vehicle actor required")
        try:
            self._control = player.get_control()
        except RuntimeError:
            self._control = carla.VehicleControl()
        self._vehicle_max_steer_angle_rad = _vehicle_max_steer_angle_rad(player)
        player.set_autopilot(self._autopilot_enabled)

        if reset_state_on_change:
            self._reset_vehicle_state_for_respawn()
        return True

    def parse_events(self, world, clock):
        self._sync_world_refs(world, reset_state_on_change=True)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                world.abort_active_scenario()
                self.finalize_exit(world)
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    world.abort_active_scenario()
                    self.finalize_exit(world)
                    return True
                elif event.key == K_BACKSPACE:
                    world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_TAB and world.camera_manager is not None:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_q:
                    self._control.gear = 1 if self._control.reverse else -1
                elif event.key == K_m:
                    self._control.manual_gear_shift = not self._control.manual_gear_shift
                    self._control.gear = world.player.get_control().gear
                    world.hud.notification('%s Transmission' %
                                           ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                elif self._control.manual_gear_shift and event.key == K_COMMA:
                    self._control.gear = max(-1, self._control.gear - 1)
                elif self._control.manual_gear_shift and event.key == K_PERIOD:
                    self._control.gear = self._control.gear + 1
                elif event.key == K_p:
                    self._autopilot_enabled = not self._autopilot_enabled
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))

        self._sync_world_refs(world, reset_state_on_change=True)
        if not self._autopilot_enabled:
            self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
            self._parse_vehicle_wheel()
            self._apply_gesture_override()
            self._control.reverse = self._control.gear < 0
            world.player.apply_control(self._control)
            self._log_drive_step(world)

    def _parse_vehicle_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def _parse_vehicle_wheel(self):
        if self._joystick is None:
            return

        num_axes = self._joystick.get_numaxes()
        js_inputs = [float(self._joystick.get_axis(i)) for i in range(num_axes)]
        if max(self._throttle_idx, self._brake_idx) >= len(js_inputs):
            return

        js_buttons = [
            float(self._joystick.get_button(i))
            for i in range(self._joystick.get_numbuttons())
        ]

        k2 = 1.6
        throttle_cmd = k2 + (2.05 * math.log10(-0.7 * js_inputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
        throttle_cmd = max(0.0, min(1.0, throttle_cmd))

        brake_cmd = k2 + (2.05 * math.log10(-0.7 * js_inputs[self._brake_idx] + 1.4) - 1.2) / 0.92
        brake_cmd = max(0.0, min(1.0, brake_cmd))

        hand_brake_pressed = False
        if self._handbrake_idx < len(js_buttons):
            hand_brake_pressed = bool(js_buttons[self._handbrake_idx])

        self._control.throttle = max(float(self._control.throttle), throttle_cmd)
        self._control.brake = max(float(self._control.brake), brake_cmd)
        self._control.hand_brake = bool(self._control.hand_brake) or hand_brake_pressed

    def _start_gesture_thread(self):
        if realtime_gesture is None:
            return
        if self._gesture_thread is not None and self._gesture_thread.is_alive():
            return
        
        def _runner():
            try:
                rt_args = []
                if self._realtime_log_path:
                    rt_args.extend(["--prediction-log", self._realtime_log_path])
                realtime_gesture.main(rt_args)
            except Exception as e:
                print("[gesture] thread stopped:", e)

        self._gesture_thread = threading.Thread(target=_runner, daemon=True)
        self._gesture_thread.start()

    def _resolve_dual_arm_actions(self, left_label: str, right_label: str):
        """
        Returns:
            steer_key: "left", "right", "left_strong", "right_strong", or "neutral"
            reverse_toggle_requested: bool
        """
        left_label = str(left_label)
        right_label = str(right_label)

        reverse_toggle_requested = left_label == "horn" and right_label == "horn"
        steer_key = _resolve_dual_arm_steer_key(left_label, right_label)

        return steer_key, reverse_toggle_requested

    def _reset_steer_dwell_candidate(self):
        self._pending_steer_key = None
        self._pending_steer_count = 0

    def _apply_steer_dwell(self, requested_steer_key: str):
        requested_steer_key = str(requested_steer_key)
        applied_steer_key = str(self._applied_steer_key)
        dwell_required = (
            self._neutral_steer_dwell_frames
            if requested_steer_key == "neutral"
            else self._active_steer_dwell_frames
        )
        dwell_required = max(1, int(dwell_required))

        if requested_steer_key == applied_steer_key:
            self._reset_steer_dwell_candidate()
            return applied_steer_key

        if self._pending_steer_key == requested_steer_key:
            self._pending_steer_count += 1
        else:
            self._pending_steer_key = requested_steer_key
            self._pending_steer_count = 1

        if self._pending_steer_count >= dwell_required:
            self._applied_steer_key = requested_steer_key
            applied_steer_key = requested_steer_key
            self._reset_steer_dwell_candidate()
        return applied_steer_key

    def _update_reverse_toggle(self, reverse_toggle_requested: bool):
        reverse_active = bool(reverse_toggle_requested)
        if not isinstance(self._control, carla.VehicleControl):
            self._prev_reverse_toggle_active = reverse_active
            return
        player = self._get_live_player()
        if not isinstance(player, carla.Vehicle):
            self._prev_reverse_toggle_active = reverse_active
            return
        reverse_edge = reverse_active and not self._prev_reverse_toggle_active
        now = time.monotonic()
        if reverse_edge and (now - self._last_reverse_toggle_ts) >= self._reverse_toggle_cooldown_s:
            try:
                velocity = player.get_velocity()
            except RuntimeError:
                self._prev_reverse_toggle_active = reverse_active
                return
            speed_mps = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
            if speed_mps > self._reverse_toggle_max_speed_mps:
                if self._hud is not None:
                    self._hud.notification(
                        "Reverse toggle blocked while moving",
                        seconds=2.0,
                    )
            else:
                currently_reverse = bool(self._control.gear < 0)
                self._control.gear = 1 if currently_reverse else -1
                self._control.reverse = bool(self._control.gear < 0)
                if self._hud is not None:
                    self._hud.notification(
                        "Reverse %s" % ("Off" if currently_reverse else "On"),
                        seconds=2.0,
                    )
            self._last_reverse_toggle_ts = float(now)
        self._prev_reverse_toggle_active = reverse_active

    def _apply_gesture_override(self):
        if realtime_gesture is None:
            self._reset_steer_dwell_candidate()
            self._set_turn_signal("off")
            self._prev_reverse_toggle_active = False
            return
    
        # Read the new published dual-arm output from realtime_gesture_cnn.py
        published, age = realtime_gesture.get_latest_published_gestures()
        self._latest_published = published
        if age > self._gesture_max_age:
            self._reset_steer_dwell_candidate()
            self._set_turn_signal("off")
            self._prev_reverse_toggle_active = False
            return
    
        left_label = "neutral"
        right_label = "neutral"
    
        gestures = tuple(getattr(published, "gestures", ()))
    
        if getattr(published, "mode", "single") == "split":
            for gesture in gestures:
                arm = str(getattr(gesture, "arm", "")).lower()
                label = str(getattr(gesture, "label", "neutral"))
                if arm == "left":
                    left_label = label
                elif arm == "right":
                    right_label = label
        else:
            # single output means both arms agreed, or only one arm is active/published
            if gestures:
                label = str(getattr(gestures[0], "label", "neutral"))
                arm = str(getattr(gestures[0], "arm", "")).lower()
    
                if arm == "left":
                    left_label = label
                elif arm == "right":
                    right_label = label
                elif arm == "dual":
                    left_label = label
                    right_label = label
                else:
                    # safe fallback
                    left_label = label
                    right_label = label
    
        steer_key, reverse_toggle_requested = self._resolve_dual_arm_actions(left_label, right_label)
        self._latest_steer_key = steer_key
        applied_steer_key = self._apply_steer_dwell(steer_key)
        self._latest_applied_steer_key = applied_steer_key
    
        # Apply steering
        self._control.steer = float(self._gesture_to_steer[applied_steer_key])

        # Signals are on only while a hard turn is being held.
        if applied_steer_key == "left_strong":
            target_signal = "left"
        elif applied_steer_key == "right_strong":
            target_signal = "right"
        else:
            target_signal = "off"

        if target_signal != self._signal_state:
            self._set_turn_signal(target_signal)

        # Reverse is a dual-horn toggle, not a hold-to-reverse action.
        self._update_reverse_toggle(reverse_toggle_requested)

    def _estimate_lane_error_m(self, world):
        try:
            waypoint = world.world.get_map().get_waypoint(
                world.player.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
        except RuntimeError:
            return None
        if waypoint is None:
            return None
        vehicle_location = world.player.get_location()
        wp_location = waypoint.transform.location
        return math.hypot(vehicle_location.x - wp_location.x, vehicle_location.y - wp_location.y)

    def _log_drive_step(self, world):
        if self._drive_logger is None:
            return
        player = self._get_live_player()
        if player is None:
            return
        published = self._latest_published
        if published is None:
            published = getattr(realtime_gesture, "PublishedGestureOutput", None)
            published = published() if callable(published) else None
        lane_invasion_count = int(getattr(world.lane_invasion_sensor, "event_count", 0))
        lane_invasion_event = lane_invasion_count > self._last_lane_invasion_count
        self._last_lane_invasion_count = lane_invasion_count

        apply_ts = time.time()
        scenario = world.get_scenario_snapshot()
        try:
            speed_mps = _velocity_speed_mps(player.get_velocity())
        except RuntimeError:
            speed_mps = None
        speed_limit_mps = _speed_limit_mps(player)
        velocity_deviation_mps = (
            abs(float(speed_mps) - float(speed_limit_mps))
            if speed_mps is not None and speed_limit_mps is not None
            else None
        )
        if self._vehicle_max_steer_angle_rad is None:
            self._vehicle_max_steer_angle_rad = _vehicle_max_steer_angle_rad(player)
        steering_angle_rad = (
            float(getattr(self._control, 'steer', 0.0)) * float(self._vehicle_max_steer_angle_rad)
            if self._vehicle_max_steer_angle_rad is not None
            else None
        )
        row = {
            'timestamp': datetime.datetime.now().isoformat(),
            'control_apply_ts': apply_ts,
            'simulation_time': float(getattr(world.hud, 'simulation_time', 0.0)),
            'prediction_seq': int(getattr(published, 'prediction_seq', 0)) if published is not None else 0,
            'steer_key': str(self._latest_steer_key),
            'applied_steer_key': str(self._latest_applied_steer_key),
            'steer': float(getattr(self._control, 'steer', 0.0)),
            'throttle': float(getattr(self._control, 'throttle', 0.0)),
            'brake': float(getattr(self._control, 'brake', 0.0)),
            'reverse': bool(getattr(self._control, 'reverse', False)),
            'hand_brake': bool(getattr(self._control, 'hand_brake', False)),
            'speed_mps': speed_mps,
            'speed_limit_mps': speed_limit_mps,
            'velocity_deviation_mps': velocity_deviation_mps,
            'steering_angle_rad': steering_angle_rad,
            'lane_error_m': self._estimate_lane_error_m(world),
            'lane_invasion_event': bool(lane_invasion_event),
            'scenario_name': str(scenario.get('scenario_name', '')),
            'scenario_kind': str(scenario.get('scenario_kind', '')),
            'scenario_status': str(scenario.get('scenario_status', '')),
            'scenario_finished': scenario.get('scenario_finished', ''),
            'scenario_success': scenario.get('scenario_success', ''),
            'scenario_failure_reason': str(scenario.get('scenario_failure_reason', '')),
            'scenario_elapsed_s': scenario.get('scenario_elapsed_s', ''),
            'scenario_completion_time_s': scenario.get('scenario_completion_time_s', ''),
        }
        self._drive_logger.write_row(row)

    def _write_exit_log(self, world):
        if self._exit_log_written:
            return
        self._log_drive_step(world)
        self._exit_log_written = True

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

    def _set_turn_signal(self, side: str):
        """
        side: "off" | "left" | "right"
        """
        player = self._get_live_player()
        if not isinstance(player, carla.Vehicle):
            self._signal_state = "off"
            return

        # Preserve any existing lights, but replace blinker bits.
        try:
            current = int(player.get_light_state())
        except RuntimeError:
            self._signal_state = "off"
            return
        left_bit = int(carla.VehicleLightState.LeftBlinker)
        right_bit = int(carla.VehicleLightState.RightBlinker)
        mask = left_bit | right_bit

        new_bits = 0
        if side == "left":
            new_bits = left_bit
        elif side == "right":
            new_bits = right_bit

        new_state = (current & ~mask) | new_bits
        try:
            player.set_light_state(carla.VehicleLightState(new_state))
        except RuntimeError:
            self._signal_state = "off"
            return
        self._signal_state = side
    
# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = bool(HUD_VISIBLE_BY_DEFAULT) if LOW_GRAPHICS_MODE else True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        v = world.player.get_velocity()
        c = world.player.get_control()
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.world.get_map().name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            '']
        self._info_text += [
            ('Throttle:', c.throttle, 0.0, 1.0),
            ('Steer:', c.steer, -1.0, 1.0),
            ('Brake:', c.brake, 0.0, 1.0),
            ('Reverse:', c.reverse),
            ('Hand brake:', c.hand_brake),
            ('Manual:', c.manual_gear_shift),
            'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        scenario_lines = world.get_scenario_hud_lines()
        if scenario_lines:
            self._info_text += ['']
            self._info_text += scenario_lines
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            player_location = world.player.get_location()
            distance = lambda l: _location_distance(l, player_location)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self.event_count = 0
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.event_count += 1
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.event_count = 0
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _enum_value_name(value):
        if value is None:
            return ""
        text = str(value).strip()
        if "." in text:
            text = text.split(".")[-1]
        if text.startswith("LaneMarkingType.") or text.startswith("LaneChange."):
            text = text.split(".", 1)[-1]
        return text.strip()

    @staticmethod
    def _lane_marking_type(marking):
        value = getattr(marking, "type", None)
        if callable(value):
            value = value()
        return LaneInvasionSensor._enum_value_name(value)

    @staticmethod
    def _lane_marking_allows_change(marking):
        lane_change = getattr(marking, "lane_change", None)
        if callable(lane_change):
            lane_change = lane_change()
        if lane_change is not None:
            try:
                return int(lane_change) != 0
            except (TypeError, ValueError):
                pass
            lane_change_name = LaneInvasionSensor._enum_value_name(lane_change).lower()
            if lane_change_name in {"left", "right", "both"}:
                return True
            if lane_change_name in {"none", "0", ""}:
                return False

        return LaneInvasionSensor._lane_marking_type(marking).lower() in {"broken", "brokenbroken"}

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        markings = tuple(getattr(event, "crossed_lane_markings", ()) or ())
        if markings and all(LaneInvasionSensor._lane_marking_allows_change(marking) for marking in markings):
            return
        self.event_count += 1
        lane_types = sorted({LaneInvasionSensor._lane_marking_type(marking) for marking in markings if marking is not None})
        if lane_types:
            self.hud.notification('Crossed line %s' % ' and '.join(repr(text) for text in lane_types))
        else:
            self.hud.notification('Crossed line')

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, transform_index=0):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self._camera_transforms = [
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            carla.Transform(carla.Location(x=1.6, z=1.7)),
        ]
        self.transform_index = int(transform_index) % len(self._camera_transforms)
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        bp = bp_library.find('sensor.camera.rgb')
        if LOW_GRAPHICS_MODE:
            bp.set_attribute('image_size_x', str(CAMERA_IMAGE_WIDTH))
            bp.set_attribute('image_size_y', str(CAMERA_IMAGE_HEIGHT))
            if bp.has_attribute('sensor_tick'):
                bp.set_attribute('sensor_tick', str(CAMERA_SENSOR_TICK_S))
            if bp.has_attribute('enable_postprocess_effects'):
                bp.set_attribute('enable_postprocess_effects', 'False')
        else:
            bp.set_attribute('image_size_x', str(hud.dim[0]))
            bp.set_attribute('image_size_y', str(hud.dim[1]))
        self.sensor = self._parent.get_world().spawn_actor(
            bp,
            self._camera_transforms[self.transform_index],
            attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        if self.sensor is not None:
            self.sensor.set_transform(self._camera_transforms[self.transform_index])
        if self.hud is not None:
            label = 'Third-person' if self.transform_index == 0 else 'Dash'
            self.hud.notification('Camera: %s' % label)

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(cc.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    controller = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)
        sim_world = _resolve_world(client, getattr(args, 'map', ''))

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.FULLSCREEN)

        hud = HUD(args.width, args.height)
        world = World(
            sim_world,
            hud,
            args.filter,
            client=client,
            scenario_preset=getattr(args, 'scenario_preset', None),
            ambient_vehicle_count=getattr(args, 'ambient_vehicles', 0),
            ambient_pedestrian_count=getattr(args, 'ambient_pedestrians', 0),
        )
        controller = DualControl(
            world,
            args.autopilot,
            carla_log_path=getattr(args, 'carla_log', ''),
            realtime_log_path=getattr(args, 'realtime_log', ''),
        )

        clock = pygame.time.Clock()
        while True:
            if CLIENT_USE_BUSY_LOOP:
                clock.tick_busy_loop(CLIENT_FPS_LIMIT)
            else:
                clock.tick(CLIENT_FPS_LIMIT)
            if controller.parse_events(world, clock):
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()
            if world.scenario_exit_requested():
                controller.finalize_exit(world)
                time.sleep(1.0)
                return

    finally:
        if controller is not None:
            controller.close()

        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--map',
        metavar='NAME',
        default='',
        help='Optional CARLA map name to load via client API before spawning the vehicle.')
    argparser.add_argument(
        '--scenario',
        choices=scenario_choices(),
        default='',
        help='Optional named CARLA evaluation scenario.')
    argparser.add_argument(
        '--ambient-vehicles',
        metavar='N',
        default=0,
        type=int,
        help='Number of ambient autopilot vehicles to spawn (default: 0)')
    argparser.add_argument(
        '--ambient-pedestrians',
        metavar='N',
        default=0,
        type=int,
        help='Number of ambient pedestrians to spawn (default: 0)')
    argparser.add_argument(
        '--graphics',
        choices=['low', 'normal'],
        default='low' if DEFAULT_LOW_GRAPHICS_MODE else 'normal',
        help='client graphics preset (default: low)')
    argparser.add_argument(
        '--camera-res',
        metavar='WIDTHxHEIGHT',
        default=DEFAULT_CAMERA_RES,
        help='camera sensor resolution in low mode (default: %s)' % DEFAULT_CAMERA_RES)
    argparser.add_argument(
        '--camera-fps',
        metavar='FPS',
        default=DEFAULT_CAMERA_FPS,
        type=float,
        help='camera sensor FPS in low mode (default: %.1f)' % DEFAULT_CAMERA_FPS)
    argparser.add_argument(
        '--client-fps',
        metavar='FPS',
        default=DEFAULT_CLIENT_FPS_LIMIT,
        type=int,
        help='client loop FPS cap (default: %d)' % DEFAULT_CLIENT_FPS_LIMIT)
    argparser.add_argument(
        '--no-rendering',
        action='store_true',
        help='disable server rendering entirely; camera feeds will stop updating')
    argparser.add_argument(
        '--show-hud',
        action='store_true',
        help='show the HUD on startup')
    argparser.add_argument(
        '--hide-hud',
        action='store_true',
        help='hide the HUD on startup')
    argparser.add_argument(
        '--eval-log-dir',
        default='',
        help='Directory for evaluation CSV logs. If set, default CARLA and realtime log files are created there.')
    argparser.add_argument(
        '--carla-log',
        default='',
        help='Optional CSV path for per-tick CARLA drive logging.')
    argparser.add_argument(
        '--realtime-log',
        default='',
        help='Optional CSV path passed to realtime_gesture_cnn.py for per-prediction logging.')
    args = argparser.parse_args()

    if args.show_hud and args.hide_hud:
        argparser.error('use only one of --show-hud or --hide-hud')
    if args.ambient_vehicles < 0:
        argparser.error('--ambient-vehicles must be >= 0')
    if args.ambient_pedestrians < 0:
        argparser.error('--ambient-pedestrians must be >= 0')

    global CLIENT_FPS_LIMIT
    global LOW_GRAPHICS_MODE
    global LOW_GRAPHICS_NO_RENDERING
    global CAMERA_IMAGE_WIDTH
    global CAMERA_IMAGE_HEIGHT
    global CAMERA_SENSOR_TICK_S
    global HUD_VISIBLE_BY_DEFAULT

    try:
        args.width, args.height = parse_resolution(args.res)
        CAMERA_IMAGE_WIDTH, CAMERA_IMAGE_HEIGHT = parse_resolution(args.camera_res)
    except ValueError as exc:
        argparser.error(str(exc))

    LOW_GRAPHICS_MODE = args.graphics == 'low'
    LOW_GRAPHICS_NO_RENDERING = bool(args.no_rendering)
    CLIENT_FPS_LIMIT = max(1, int(args.client_fps))
    CAMERA_SENSOR_TICK_S = 1.0 / max(1.0, float(args.camera_fps))

    eval_log_dir = str(getattr(args, 'eval_log_dir', '') or '').strip()
    carla_log = str(getattr(args, 'carla_log', '') or '').strip()
    realtime_log = str(getattr(args, 'realtime_log', '') or '').strip()
    scenario_preset = get_scenario_preset(getattr(args, 'scenario', ''))
    args.scenario_preset = scenario_preset
    if scenario_preset is not None:
        if args.map and _map_basename(args.map) != _map_basename(scenario_preset.map_name):
            print(
                "[scenario] overriding requested map %s with scenario map %s" % (
                    args.map,
                    scenario_preset.map_name,
                )
            )
        args.map = scenario_preset.map_name
    if eval_log_dir:
        eval_dir = Path(eval_log_dir)
        eval_dir.mkdir(parents=True, exist_ok=True)
        stamp = _now_stamp()
        if not carla_log:
            carla_log = str(eval_dir / ('carla_drive_%s.csv' % stamp))
        if not realtime_log:
            realtime_log = str(eval_dir / ('realtime_predictions_%s.csv' % stamp))
    args.carla_log = carla_log
    args.realtime_log = realtime_log

    if args.show_hud:
        HUD_VISIBLE_BY_DEFAULT = True
    elif args.hide_hud:
        HUD_VISIBLE_BY_DEFAULT = False
    else:
        HUD_VISIBLE_BY_DEFAULT = DEFAULT_HUD_VISIBLE if LOW_GRAPHICS_MODE else True

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
