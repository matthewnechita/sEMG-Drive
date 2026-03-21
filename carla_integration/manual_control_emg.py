#!/usr/bin/env python

# Copyright (c) 2019 Intel Labs
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control with steering wheel Logitech G29.

To drive start by preshing the brake pedal.
Change your wheel_config.ini according to your steering wheel.

To find out the values of your steering wheel use jstest-gtk in Ubuntu.

"""

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
from pathlib import Path

from carla_integration.scenario_presets import get_scenario_preset, scenario_choices
from emg.runtime_tuning import get_runtime_tuning_preset

if sys.version_info >= (3, 0):

    from configparser import ConfigParser

else:

    from ConfigParser import RawConfigParser as ConfigParser

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
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

RUNTIME_TUNING_PRESET = get_runtime_tuning_preset()
RUNTIME_TUNING_PRESET_NAME = RUNTIME_TUNING_PRESET.name
CARLA_TUNING = RUNTIME_TUNING_PRESET.carla

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


def _now_stamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


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
            'runtime_preset',
            'timestamp',
            'wall_time_s',
            'control_apply_ts',
            'simulation_time',
            'frame',
            'prediction_seq',
            'window_end_ts',
            'prediction_ts',
            'publish_ts',
            'gesture_age_s',
            'gesture_fresh',
            'published_mode',
            'pred_label',
            'pred_conf',
            'right_label',
            'right_conf',
            'left_label',
            'left_conf',
            'steer_key',
            'applied_steer_key',
            'steer_dwell_pending_key',
            'steer_dwell_pending_count',
            'steer_dwell_required',
            'signal_state',
            'horn_on',
            'steer',
            'throttle',
            'brake',
            'reverse',
            'hand_brake',
            'vehicle_x',
            'vehicle_y',
            'vehicle_z',
            'speed_mps',
            'lane_error_m',
            'lane_invasion_event',
            'lane_invasion_count_total',
            'collision_event',
            'collision_count_total',
            'scenario_name',
            'scenario_kind',
            'scenario_status',
            'scenario_started',
            'scenario_finished',
            'scenario_success',
            'scenario_failure_reason',
            'scenario_elapsed_s',
            'scenario_completion_time_s',
            'scenario_checkpoint_index',
            'scenario_checkpoint_count',
            'scenario_progress_m',
            'scenario_lead_distance_m',
            'scenario_lead_gap_m',
            'scenario_current_lane_id',
            'scenario_target_lane_id',
            'published_labels',
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
        self._last_progress_m = None
        self._last_lead_distance_m = None
        self._last_lead_gap_m = None
        self._last_current_lane_id = None
        self._last_sim_time = 0.0
        self._should_exit = False

    def get_ego_spawn_transform(self, carla_map):
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
        total_length = max(float(self.preset.route_length_m), float(self.preset.start_offset_m) + 20.0)
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

    def _build_checkpoints(self):
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

    def _advance_checkpoints(self, ego_location):
        radius = float(self.preset.checkpoint_radius_m)
        while self._next_checkpoint_index < len(self._checkpoint_locations):
            checkpoint = self._checkpoint_locations[self._next_checkpoint_index]
            if _location_distance(ego_location, checkpoint) > radius:
                break
            self._next_checkpoint_index += 1

    def _spawn_lead_vehicle(self, world):
        if not self._route_progress_m:
            return
        target_progress = min(
            float(self.preset.lead_spawn_distance_m),
            float(self._route_progress_m[-1]),
        )
        spawn_location = self._progress_to_location(target_progress)
        if spawn_location is None:
            return
        waypoint = world.world.get_map().get_waypoint(
            spawn_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
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
        if self._traffic_manager is None:
            return
        tm_port = int(self._traffic_manager.get_port())
        actor.set_autopilot(True, tm_port)
        try:
            self._traffic_manager.auto_lane_change(actor, False)
        except RuntimeError:
            pass
        try:
            self._traffic_manager.distance_to_leading_vehicle(actor, 1.0)
        except RuntimeError:
            pass
        try:
            self._traffic_manager.vehicle_percentage_speed_difference(
                actor,
                float(self.preset.lead_speed_reduction_pct),
            )
        except RuntimeError:
            pass

    def destroy(self):
        for actor in reversed(self._actors):
            try:
                if actor is not None and actor.is_alive:
                    actor.destroy()
            except RuntimeError:
                pass
        self._actors = []
        self._lead_vehicle = None

    def setup(self, world):
        self.destroy()
        self._route_waypoints = []
        self._route_progress_m = []
        self._checkpoint_locations = []
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
        self._last_progress_m = None
        self._last_lead_distance_m = None
        self._last_lead_gap_m = None
        self._last_current_lane_id = None
        self._last_sim_time = 0.0
        self._should_exit = False

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
        if self._checkpoint_locations:
            self._start_location = self._checkpoint_locations[0]
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

        checkpoint_count = len(self._checkpoint_locations)
        world.hud.notification(
            "Scenario: %s (%d checkpoints)" % (self.preset.name, checkpoint_count),
            seconds=4.0,
        )
        print(
            "[scenario] prepared %s on %s with %d checkpoints" % (
                self.preset.name,
                self.preset.map_name,
                checkpoint_count,
            )
        )

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

    def should_exit(self):
        return bool(self._should_exit)

    def tick(self, world):
        if self._status == "setup_failed":
            self._should_exit = True
            return
        if self._finished:
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

        if not self._started:
            if self._start_location is not None and _location_distance(ego_location, self._start_location) <= float(self.preset.checkpoint_radius_m):
                self._started = True
                self._status = "active"
                self._start_sim_time = float(sim_time)
                self._next_checkpoint_index = 1 if self._checkpoint_locations else 0
                world.hud.notification("Scenario started", seconds=2.0)
        else:
            self._advance_checkpoints(ego_location)

        if self._started and not self._finished and self._start_sim_time is not None:
            elapsed = float(sim_time - self._start_sim_time)
            if elapsed > float(self.preset.timeout_s):
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
        self._last_lead_distance_m = _location_distance(ego_location, lead_location)
        lead_progress_m = self._project_progress_m(lead_location)
        if self._last_progress_m is not None and lead_progress_m is not None:
            self._last_lead_gap_m = float(lead_progress_m - self._last_progress_m)
        else:
            self._last_lead_gap_m = None

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
            self._finish(world, sim_time, True, "")

    def snapshot(self):
        elapsed_s = None
        if self._started and self._start_sim_time is not None:
            if self._finished and self._completion_time_s is not None:
                elapsed_s = float(self._completion_time_s)
            else:
                elapsed_s = max(0.0, float(self._last_sim_time - self._start_sim_time))
        checkpoint_count = len(self._checkpoint_locations)
        checkpoint_index = min(self._next_checkpoint_index, checkpoint_count)
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
                    min(self._next_checkpoint_index, len(self._checkpoint_locations)),
                    len(self._checkpoint_locations),
                )
            )
        if self._started and self._start_sim_time is not None:
            elapsed = (
                float(self._completion_time_s)
                if self._finished and self._completion_time_s is not None
                else float(sim_time - self._start_sim_time)
            )
            lines.append("Scenario time: %.1fs" % max(0.0, elapsed))
        if self._last_lead_distance_m is not None:
            lines.append("Lead distance: %.1fm" % self._last_lead_distance_m)
        return lines

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
    def __init__(self, carla_world, hud, actor_filter, client=None, scenario_preset=None):
        self.world = carla_world
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = actor_filter
        self._client = client
        self._traffic_manager = client.get_trafficmanager() if client is not None else None
        self._scenario_preset = scenario_preset
        self._scenario_runtime = ScenarioRuntime(scenario_preset, self._traffic_manager) if scenario_preset else None
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
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
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
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)
        if self._scenario_runtime is not None:
            self._scenario_runtime.setup(self)

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

    def destroy(self):
        if self._scenario_runtime is not None:
            self._scenario_runtime.destroy()
        sensors = [
            self.camera_manager.sensor if self.camera_manager is not None else None,
            self.collision_sensor.sensor if self.collision_sensor is not None else None,
            self.lane_invasion_sensor.sensor if self.lane_invasion_sensor is not None else None,
            self.gnss_sensor.sensor if self.gnss_sensor is not None else None]
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
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            world.player.set_autopilot(self._autopilot_enabled)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

        # initialize steering wheel
        pygame.joystick.init()

        joystick_count = pygame.joystick.get_count()
        if joystick_count > 1:
            raise ValueError("Please Connect Just One Joystick")

        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()

        self._parser = ConfigParser()
        wheel_config_path = os.path.join(SCRIPT_DIR, 'wheel_config.ini')
        self._parser.read(wheel_config_path)
        self._steer_idx = int(
            self._parser.get('G29 Racing Wheel', 'steering_wheel'))
        self._throttle_idx = int(
            self._parser.get('G29 Racing Wheel', 'throttle'))
        self._brake_idx = int(self._parser.get('G29 Racing Wheel', 'brake'))
        self._reverse_idx = int(self._parser.get('G29 Racing Wheel', 'reverse'))
        self._handbrake_idx = int(
            self._parser.get('G29 Racing Wheel', 'handbrake'))
        
        # Store references so gesture actions can affect vehicle + HUD
        self._player = world.player
        self._hud = world.hud
        
        # Gesture freshness (optional safety: ignore stale labels)
        self._gesture_max_age = float(CARLA_TUNING.gesture_max_age_s)
        self._client_fps = 0.0
        self._last_applied_gesture = None
        self._last_gesture_debug_t = 0.0
        self._latest_published = None
        self._latest_gesture_age = float('inf')
        self._latest_right_label = "neutral"
        self._latest_right_conf = 0.0
        self._latest_left_label = "neutral"
        self._latest_left_conf = 0.0
        self._latest_pred_label = "neutral"
        self._latest_pred_conf = 0.0
        self._latest_steer_key = "neutral"
        self._latest_applied_steer_key = "neutral"
        self._latest_steer_dwell_pending_key = ""
        self._latest_steer_dwell_pending_count = 0
        self._latest_steer_dwell_required = 1
        self._latest_signal_state = "off"
        self._latest_horn_on = False
        self._gesture_reverse_active = False
        self._latest_gesture_fresh = False
        self._last_lane_invasion_count = 0
        self._last_collision_count = 0
        self._drive_logger = DriveCSVLogger(carla_log_path) if carla_log_path else None
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
        print(
            f"[gesture] runtime preset: {RUNTIME_TUNING_PRESET_NAME} "
            f"(gesture_max_age={self._gesture_max_age:.2f}s, "
            f"active_dwell={self._active_steer_dwell_frames}, "
            f"neutral_dwell={self._neutral_steer_dwell_frames})"
        )
        scenario_name = getattr(getattr(world, "_scenario_preset", None), "name", "")
        if scenario_name:
            print(f"[scenario] active scenario: {scenario_name}")
        
        # Track signals so they behave like a real car (stay on until changed/canceled)
        self._signal_state = "off"  # "off" | "left" | "right"
        
        self._gesture_thread = None
        self._start_gesture_thread()

    def close(self):
        if self._drive_logger is not None:
            self._drive_logger.close()
            self._drive_logger = None

    def parse_events(self, world, clock):
        self._client_fps = float(clock.get_fps())
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == 0:
                    world.restart()
                elif event.button == 1:
                    world.hud.toggle_info()
                elif event.button == 2:
                    world.camera_manager.toggle_camera()
                elif event.button == 3:
                    world.next_weather()
                elif event.button == self._reverse_idx:
                    self._control.gear = 1 if self._control.reverse else -1
                elif event.button == 23:
                    world.camera_manager.next_sensor()

            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r:
                    world.camera_manager.toggle_recording()
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
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

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._parse_vehicle_wheel()
                self._apply_gesture_override()
                self._control.reverse = self._control.gear < 0
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time())
            world.player.apply_control(self._control)
            self._log_drive_step(world)

    def _parse_vehicle_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def _parse_vehicle_wheel(self):
        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
        # print (jsInputs)
        jsButtons = [float(self._joystick.get_button(i)) for i in
                     range(self._joystick.get_numbuttons())]

        
        K2 = 1.6  # 1.6
        throttleCmd = K2 + (2.05 * math.log10(
            -0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
        if throttleCmd <= 0:
            throttleCmd = 0
        elif throttleCmd > 1:
            throttleCmd = 1

        brakeCmd = 1.6 + (2.05 * math.log10(
            -0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92
        if brakeCmd <= 0:
            brakeCmd = 0
        elif brakeCmd > 1:
            brakeCmd = 1

        self._control.brake = brakeCmd
        self._control.throttle = throttleCmd

        #toggle = jsButtons[self._reverse_idx]

        self._control.hand_brake = bool(jsButtons[self._handbrake_idx])

    def _parse_walker_keys(self, keys, milliseconds):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = 5.556 if pygame.key.get_mods() & KMOD_SHIFT else 2.778
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

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
            signal_state: "left", "right", or "off"
            reverse_requested: bool
        """
    
        labels = {str(left_label), str(right_label)}
    
        # --- reverse ---
        reverse_requested = str(left_label) == "horn" and str(right_label) == "horn"
    
        # --- signal ---
        has_signal_left = "signal_left" in labels
        has_signal_right = "signal_right" in labels
    
        if has_signal_left and has_signal_right:
            signal_state = "off"
        elif has_signal_left:
            signal_state = "left"
        elif has_signal_right:
            signal_state = "right"
        else:
            signal_state = "off"
    
        # --- steer ---
        has_left_turn = "left_turn" in labels
        has_right_turn = "right_turn" in labels
    
        if has_left_turn and has_right_turn:
            steer_key = "neutral"
        elif left_label == "left_turn" and right_label == "left_turn":
            steer_key = "left_strong"
        elif left_label == "right_turn" and right_label == "right_turn":
            steer_key = "right_strong"
        elif has_left_turn:
            steer_key = "left"
        elif has_right_turn:
            steer_key = "right"
        else:
            steer_key = "neutral"

        return steer_key, signal_state, reverse_requested

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
            self._latest_steer_dwell_pending_key = ""
            self._latest_steer_dwell_pending_count = 0
            self._latest_steer_dwell_required = dwell_required
            return applied_steer_key

        if self._pending_steer_key == requested_steer_key:
            self._pending_steer_count += 1
        else:
            self._pending_steer_key = requested_steer_key
            self._pending_steer_count = 1

        self._latest_steer_dwell_pending_key = str(self._pending_steer_key or "")
        self._latest_steer_dwell_pending_count = int(self._pending_steer_count)
        self._latest_steer_dwell_required = int(dwell_required)

        if self._pending_steer_count >= dwell_required:
            self._applied_steer_key = requested_steer_key
            applied_steer_key = requested_steer_key
            self._reset_steer_dwell_candidate()
            self._latest_steer_dwell_pending_key = ""
            self._latest_steer_dwell_pending_count = 0
        return applied_steer_key

    def _apply_reverse_override(self, reverse_requested: bool):
        if not isinstance(self._control, carla.VehicleControl):
            return

        reverse_requested = bool(reverse_requested)
        if reverse_requested:
            if self._control.gear >= 0:
                self._control.gear = -1
            self._gesture_reverse_active = True
            return

        if self._gesture_reverse_active and self._control.gear < 0:
            self._control.gear = 1
        self._gesture_reverse_active = False

    def _apply_gesture_override(self):
        if realtime_gesture is None:
            self._latest_gesture_fresh = False
            self._reset_steer_dwell_candidate()
            self._latest_steer_dwell_pending_key = ""
            self._latest_steer_dwell_pending_count = 0
            self._apply_reverse_override(False)
            return
    
        # Read the new published dual-arm output from realtime_gesture_cnn.py
        published, age = realtime_gesture.get_latest_published_gestures()
        self._latest_published = published
        self._latest_gesture_age = float(age)
        if age > self._gesture_max_age:
            self._latest_gesture_fresh = False
            self._reset_steer_dwell_candidate()
            self._latest_steer_dwell_pending_key = ""
            self._latest_steer_dwell_pending_count = 0
            self._apply_reverse_override(False)
            return
    
        left_label = "neutral"
        right_label = "neutral"
        left_conf = 0.0
        right_conf = 0.0
    
        gestures = tuple(getattr(published, "gestures", ()))
    
        if getattr(published, "mode", "single") == "split":
            for gesture in gestures:
                arm = str(getattr(gesture, "arm", "")).lower()
                label = str(getattr(gesture, "label", "neutral"))
                confidence = float(getattr(gesture, "confidence", 0.0))
                if arm == "left":
                    left_label = label
                    left_conf = confidence
                elif arm == "right":
                    right_label = label
                    right_conf = confidence
        else:
            # single output means both arms agreed, or only one arm is active/published
            if gestures:
                label = str(getattr(gestures[0], "label", "neutral"))
                arm = str(getattr(gestures[0], "arm", "")).lower()
                confidence = float(getattr(gestures[0], "confidence", 0.0))
    
                if arm == "left":
                    left_label = label
                    left_conf = confidence
                elif arm == "right":
                    right_label = label
                    right_conf = confidence
                elif arm == "dual":
                    left_label = label
                    right_label = label
                    left_conf = confidence
                    right_conf = confidence
                else:
                    # safe fallback
                    left_label = label
                    right_label = label
                    left_conf = confidence
                    right_conf = confidence

        self._latest_left_label = left_label
        self._latest_left_conf = left_conf
        self._latest_right_label = right_label
        self._latest_right_conf = right_conf
    
        steer_key, signal_state, reverse_requested = self._resolve_dual_arm_actions(left_label, right_label)
        self._latest_steer_key = steer_key
        applied_steer_key = self._apply_steer_dwell(steer_key)
        self._latest_applied_steer_key = applied_steer_key
        self._latest_signal_state = signal_state
        self._latest_horn_on = False
        self._latest_pred_label = "split" if getattr(published, "mode", "single") == "split" else (
            left_label if left_label == right_label else right_label
        )
        self._latest_pred_conf = max(left_conf, right_conf)
        self._latest_gesture_fresh = True
    
        # Apply steering
        self._control.steer = float(self._gesture_to_steer[applied_steer_key])
    
        # Apply turn signal
        if signal_state != self._signal_state:
            self._set_turn_signal(signal_state)

        # Reverse engages only when both arms collapse to horn.
        self._apply_reverse_override(reverse_requested)

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
        published = self._latest_published
        if published is None:
            published = getattr(realtime_gesture, "PublishedGestureOutput", None)
            published = published() if callable(published) else None
        transform = world.player.get_transform()
        velocity = world.player.get_velocity()
        speed_mps = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        lane_invasion_count = int(getattr(world.lane_invasion_sensor, "event_count", 0))
        collision_count = int(getattr(world.collision_sensor, "event_count", 0))
        lane_invasion_event = lane_invasion_count > self._last_lane_invasion_count
        collision_event = collision_count > self._last_collision_count
        self._last_lane_invasion_count = lane_invasion_count
        self._last_collision_count = collision_count

        gestures = tuple(getattr(published, "gestures", ())) if published is not None else ()
        if gestures:
            published_labels = "|".join(
                f"{getattr(gesture, 'arm', '')}:{getattr(gesture, 'label', '')}"
                for gesture in gestures
            )
        else:
            published_labels = ""

        if str(getattr(published, "mode", "single")) == "single" and gestures:
            self._latest_pred_label = str(getattr(gestures[0], "label", self._latest_pred_label))
            self._latest_pred_conf = float(getattr(gestures[0], "confidence", self._latest_pred_conf))
        elif str(getattr(published, "mode", "single")) == "split":
            self._latest_pred_label = "split"
            self._latest_pred_conf = max(self._latest_left_conf, self._latest_right_conf)

        apply_ts = time.time()
        scenario = world.get_scenario_snapshot()
        row = {
            'runtime_preset': RUNTIME_TUNING_PRESET_NAME,
            'timestamp': datetime.datetime.now().isoformat(),
            'wall_time_s': apply_ts,
            'control_apply_ts': apply_ts,
            'simulation_time': float(getattr(world.hud, 'simulation_time', 0.0)),
            'frame': int(getattr(world.hud, 'frame', 0)),
            'prediction_seq': int(getattr(published, 'prediction_seq', 0)) if published is not None else 0,
            'window_end_ts': float(getattr(published, 'window_end_ts', 0.0)) if published is not None else 0.0,
            'prediction_ts': float(getattr(published, 'prediction_ts', 0.0)) if published is not None else 0.0,
            'publish_ts': float(getattr(published, 'publish_ts', 0.0)) if published is not None else 0.0,
            'gesture_age_s': float(self._latest_gesture_age),
            'gesture_fresh': bool(self._latest_gesture_fresh),
            'published_mode': str(getattr(published, 'mode', 'single')) if published is not None else '',
            'pred_label': str(self._latest_pred_label),
            'pred_conf': float(self._latest_pred_conf),
            'right_label': str(self._latest_right_label),
            'right_conf': float(self._latest_right_conf),
            'left_label': str(self._latest_left_label),
            'left_conf': float(self._latest_left_conf),
            'steer_key': str(self._latest_steer_key),
            'applied_steer_key': str(self._latest_applied_steer_key),
            'steer_dwell_pending_key': str(self._latest_steer_dwell_pending_key),
            'steer_dwell_pending_count': int(self._latest_steer_dwell_pending_count),
            'steer_dwell_required': int(self._latest_steer_dwell_required),
            'signal_state': str(self._latest_signal_state),
            'horn_on': bool(self._latest_horn_on),
            'steer': float(getattr(self._control, 'steer', 0.0)),
            'throttle': float(getattr(self._control, 'throttle', 0.0)),
            'brake': float(getattr(self._control, 'brake', 0.0)),
            'reverse': bool(getattr(self._control, 'reverse', False)),
            'hand_brake': bool(getattr(self._control, 'hand_brake', False)),
            'vehicle_x': float(transform.location.x),
            'vehicle_y': float(transform.location.y),
            'vehicle_z': float(transform.location.z),
            'speed_mps': float(speed_mps),
            'lane_error_m': self._estimate_lane_error_m(world),
            'lane_invasion_event': bool(lane_invasion_event),
            'lane_invasion_count_total': lane_invasion_count,
            'collision_event': bool(collision_event),
            'collision_count_total': collision_count,
            'scenario_name': str(scenario.get('scenario_name', '')),
            'scenario_kind': str(scenario.get('scenario_kind', '')),
            'scenario_status': str(scenario.get('scenario_status', '')),
            'scenario_started': scenario.get('scenario_started', ''),
            'scenario_finished': scenario.get('scenario_finished', ''),
            'scenario_success': scenario.get('scenario_success', ''),
            'scenario_failure_reason': str(scenario.get('scenario_failure_reason', '')),
            'scenario_elapsed_s': scenario.get('scenario_elapsed_s', ''),
            'scenario_completion_time_s': scenario.get('scenario_completion_time_s', ''),
            'scenario_checkpoint_index': scenario.get('scenario_checkpoint_index', ''),
            'scenario_checkpoint_count': scenario.get('scenario_checkpoint_count', ''),
            'scenario_progress_m': scenario.get('scenario_progress_m', ''),
            'scenario_lead_distance_m': scenario.get('scenario_lead_distance_m', ''),
            'scenario_lead_gap_m': scenario.get('scenario_lead_gap_m', ''),
            'scenario_current_lane_id': scenario.get('scenario_current_lane_id', ''),
            'scenario_target_lane_id': scenario.get('scenario_target_lane_id', ''),
            'published_labels': published_labels,
        }
        self._drive_logger.write_row(row)

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

    def _set_turn_signal(self, side: str):
        """
        side: "off" | "left" | "right"
        """
        if not isinstance(self._player, carla.Vehicle):
            return
    
        # Preserve any existing lights, but replace blinker bits.
        current = int(self._player.get_light_state())
        left_bit = int(carla.VehicleLightState.LeftBlinker)
        right_bit = int(carla.VehicleLightState.RightBlinker)
        mask = left_bit | right_bit
    
        new_bits = 0
        if side == "left":
            new_bits = left_bit
        elif side == "right":
            new_bits = right_bit
    
        new_state = (current & ~mask) | new_bits
        self._player.set_light_state(carla.VehicleLightState(new_state))
        self._signal_state = side
    
    def _honk(self):
        # CARLA doesn't have a built-in horn actuator for standard vehicles,
        # so we simulate it with a HUD notification (and optional sound if you add one).
        if hasattr(self, "_hud") and self._hud is not None:
            self._hud.notification("HORN!", seconds=0.2)
        else:
            print("HORN!")


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
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
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
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
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
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
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
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
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
        self.help.render(display)


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
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
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
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.event_count += 1
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self._camera_transforms = [
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            carla.Transform(carla.Location(x=1.6, z=1.7))]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
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
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '15' if LOW_GRAPHICS_MODE else '50')
            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self.transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None \
            else self.sensors[index][0] != self.sensors[self.index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index],
                attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data) # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)


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
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(
            sim_world,
            hud,
            args.filter,
            client=client,
            scenario_preset=getattr(args, 'scenario_preset', None),
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
