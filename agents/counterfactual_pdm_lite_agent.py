#!/usr/bin/env python3
"""Decision-forced PDM-Lite collector for CoDrive counterfactual branches.

All behavior changes live in RL.  The frozen CoDrive routes, ScenarioRunner
implementation, Bench2Drive expert, and existing datasets are imported read-only.
"""

from __future__ import annotations

import gc
import gzip
import hashlib
import json
import math
import os
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import carla


SIMLINGO_ROOT = Path(os.environ.get("SIMLINGO_ROOT", "/home/UNT/yl0826/simlingo")).resolve()
CLOSED_AGENT_ROOT = SIMLINGO_ROOT / "pedestrian_dataset/closed_dataset/agent"
PROJECT_AGENT_ROOT = SIMLINGO_ROOT / "pedestrian_dataset/agent"
for path in (CLOSED_AGENT_ROOT, PROJECT_AGENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from closed_coopeval_pdm_agent import ClosedCoopEvalPdmAgent  # noqa: E402
from cooperative_pdm_lite_agent import CooperativePdmLiteAgent  # noqa: E402
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider  # noqa: E402


VALID_DECISIONS = {
    "Accelerate",
    "Maintain",
    "Brake",
    "Stop",
    "LaneChangeLeft",
    "LaneChangeRight",
}


def get_entry_point():
    return "CounterfactualPdmLiteAgent"


def _truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _vector(value) -> dict:
    return {"x": float(value.x), "y": float(value.y), "z": float(value.z)}


def _rotation(value) -> dict:
    return {
        "pitch": float(value.pitch),
        "yaw": float(value.yaw),
        "roll": float(value.roll),
    }


def _lane_type_name(value) -> str:
    return str(value).split(".")[-1]


def _json_default(value):
    """Convert NumPy scalar route indices without importing NumPy here."""
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


class CounterfactualPdmLiteAgent(ClosedCoopEvalPdmAgent):
    """Force one high-level decision from warning onset to hazard clearance."""

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        decision = os.environ.get("COUNTERFACTUAL_DECISION", "").strip()
        if decision not in VALID_DECISIONS:
            raise ValueError(f"COUNTERFACTUAL_DECISION must be one of {sorted(VALID_DECISIONS)}; got {decision!r}")
        self.counterfactual_decision = decision
        self._cf_lock = threading.Lock()
        self._cf_collision = None
        self._cf_collision_sensor = None
        self._cf_collision_criteria_hooks = []
        self._cf_recorder_started = False
        self._cf_recorder_client = None
        self._cf_recorder_disabled_logged = False
        self._cf_monitor_arm_time = None
        self._cf_aborted = False
        self._cf_decision_started = False
        self._cf_decision_start_time = None
        self._cf_decision_entry_speed = None
        self._cf_decision_entry_pose = None
        self._cf_decision_target_speed = None
        self._cf_decision_suppressed_by = None
        self._cf_regulatory_override = None
        self._cf_stop_hold_since = None
        self._cf_stop_goal_met = False
        self._cf_lane_change_applied = False
        self._cf_lane_info = None
        self._cf_low_speed_since = None
        self._cf_post_clear_low_speed_since = None
        self._cf_last_active = False
        self._cf_route_spec = self._load_counterfactual_route_spec()
        self._cf_event_path = Path(os.environ["COUNTERFACTUAL_EVENT_LOG"]).resolve()
        self._cf_telemetry_path = Path(os.environ["COUNTERFACTUAL_TELEMETRY_PATH"]).resolve()
        self._cf_termination_path = Path(os.environ["COUNTERFACTUAL_TERMINATION_PATH"]).resolve()
        self._cf_run_spec_path = Path(os.environ["COUNTERFACTUAL_RUN_SPEC_PATH"]).resolve()
        for path in (
            self._cf_event_path,
            self._cf_telemetry_path,
            self._cf_termination_path,
            self._cf_run_spec_path,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
        self._cf_event_path.unlink(missing_ok=True)
        self._cf_telemetry_path.unlink(missing_ok=True)
        self._cf_termination_path.unlink(missing_ok=True)
        self._cf_telemetry_stream = gzip.open(self._cf_telemetry_path, "wt", encoding="utf-8")
        self._write_run_spec()
        super().setup(path_to_conf_file, route_index, traffic_manager=traffic_manager)

    def _load_counterfactual_route_spec(self) -> dict:
        route_path = Path(os.environ["ROUTES"]).resolve()
        root = ET.parse(route_path).getroot()
        route = root.find("route")
        scenario = root.find(".//scenario")
        if route is None or scenario is None:
            raise ValueError(f"expected one route and scenario in {route_path}")

        def point(name):
            element = scenario.find(name)
            if element is None:
                return None
            return {
                "x": float(element.attrib["x"]),
                "y": float(element.attrib["y"]),
                "z": float(element.attrib.get("z", 0.0)),
            }

        digest = hashlib.sha256(route_path.read_bytes()).hexdigest()
        return {
            "route_path": str(route_path),
            "route_sha256": digest,
            "route_id": route.attrib.get("id"),
            "town": route.attrib.get("town"),
            "source_route_id": route.attrib.get("source_route_id"),
            "scenario_name": scenario.attrib.get("name"),
            "scenario_type": scenario.attrib.get("type"),
            "cooperative_prompt": (
                scenario.find("cooperative_prompt").attrib.get("value")
                if scenario.find("cooperative_prompt") is not None
                else None
            ),
            "points": {
                name: point(name)
                for name in (
                    "scenario_arm_point",
                    "cooperative_trigger_point",
                    "event_trigger_point",
                    "collision_point",
                )
            },
        }

    def _write_run_spec(self):
        payload = {
            "schema_version": "counterfactual_decision_v1",
            "created_unix_s": time.time(),
            "decision": self.counterfactual_decision,
            "route": self._cf_route_spec,
            "parameters": {
                name: os.environ.get(name)
                for name in sorted(os.environ)
                if name.startswith("COUNTERFACTUAL_")
            },
            "carla": {
                "client_version": carla.Client.get_client_version(carla.Client("localhost", 0)),
            },
        }
        self._cf_run_spec_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _write_event(self, event: str, **values):
        payload = {
            "schema_version": "counterfactual_decision_v1",
            "event": event,
            "decision": self.counterfactual_decision,
            "route_id": self._cf_route_spec["route_id"],
            "wall_time_s": time.time(),
            **values,
        }
        with self._cf_lock:
            with self._cf_event_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=_json_default,
                    )
                    + "\n"
                )

    def _write_termination(self, reason: str, **details):
        if self._cf_aborted:
            return
        self._cf_aborted = True
        payload = {
            "schema_version": "counterfactual_decision_v1",
            "reason": reason,
            "decision": self.counterfactual_decision,
            "route_id": self._cf_route_spec["route_id"],
            "world_frame": self._world.get_snapshot().frame if hasattr(self, "_world") else None,
            "simulation_time_s": self._simulation_time() if hasattr(self, "_world") else None,
            "details": details,
        }
        self._cf_termination_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._write_event("collection_terminated", reason=reason, details=details)

    def _abort(self, reason: str, **details):
        self._write_termination(reason, **details)
        raise RuntimeError(f"COUNTERFACTUAL_TERMINATE:{reason}")

    def _ensure_runtime_monitors(self):
        if not getattr(self, "initialized", False) or not hasattr(self, "_vehicle"):
            return
        if not self._cf_collision_criteria_hooks:
            # RouteScenario already owns the one collision sensor that should
            # be attached to the ego vehicle. Attaching a second sensor at
            # runtime reproducibly segfaults CARLA in Town04. Hook the
            # existing CollisionTest callback before its first tree tick so
            # evaluator and collector receive the same raw collision event.
            hooks = []
            ego_id = int(self._vehicle.id)
            for candidate in gc.get_objects():
                try:
                    if candidate.__class__.__name__ != "CollisionTest":
                        continue
                    actor = getattr(candidate, "actor", None)
                    if actor is None or int(actor.id) != ego_id:
                        continue
                    original = candidate._count_collisions

                    def shared_collision_callback(event, original=original):
                        original(event)
                        self._on_collision(event)

                    candidate._count_collisions = shared_collision_callback
                    hooks.append((candidate, original))
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    continue
            if hooks:
                self._cf_collision_criteria_hooks = hooks
                self._write_event(
                    "collision_monitor_reused",
                    criterion_count=len(hooks),
                    ego_actor_id=ego_id,
                )
        recorder_path = os.environ.get("COUNTERFACTUAL_CARLA_RECORDER_PATH", "").strip()
        town = self._cf_route_spec.get("town")
        recorder_stable_towns = {"Town01", "Town02"}
        recorder_disabled_by_run = os.environ.get(
            "COUNTERFACTUAL_DISABLE_CARLA_RECORDER", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        if recorder_disabled_by_run or town not in recorder_stable_towns:
            recorder_path = ""
            if not self._cf_recorder_disabled_logged:
                self._cf_recorder_disabled_logged = True
                reason = (
                    "binary recorder disabled for expanded-route pilot after "
                    "reproducible UE4 SIGSEGVs; 20 Hz JSON telemetry retained"
                    if recorder_disabled_by_run
                    else (
                        f"{town} start_recorder is disabled after reproducible CARLA "
                        "SIGSEGVs on the larger-map collection path"
                    )
                )
                self._write_event(
                    "carla_recorder_disabled",
                    reason=reason,
                )
        if recorder_path and not self._cf_recorder_started:
            now = self._simulation_time()
            if self._cf_monitor_arm_time is None:
                self._cf_monitor_arm_time = now + _float_env(
                    "COUNTERFACTUAL_RECORDER_ARM_DELAY_S", 1.0
                )
                self._write_event(
                    "carla_recorder_armed",
                    arm_simulation_time_s=self._cf_monitor_arm_time,
                )
                return
            if now < self._cf_monitor_arm_time:
                return
            Path(recorder_path).resolve().parent.mkdir(parents=True, exist_ok=True)
            self._cf_recorder_client = CarlaDataProvider.get_client()
            self._cf_recorder_client.start_recorder(str(Path(recorder_path).resolve()), True)
            self._cf_recorder_started = True
            self._write_event("carla_recorder_started", path=str(Path(recorder_path).resolve()))

    def _on_collision(self, event):
        other = getattr(event, "other_actor", None)
        impulse = getattr(event, "normal_impulse", None)
        payload = {
            "world_frame": int(getattr(event, "frame", -1)),
            "other_actor_id": int(other.id) if other is not None else None,
            "other_actor_type": str(other.type_id) if other is not None else None,
            "normal_impulse": _vector(impulse) if impulse is not None else None,
        }
        with self._cf_lock:
            if self._cf_collision is None:
                self._cf_collision = payload
        self._write_event("collision_detected", **payload)

    def _event_state(self) -> dict:
        try:
            from srunner.scenarios.closed_coopeval_scenarios import get_ablation_event_state

            return get_ablation_event_state(self._cf_route_spec["scenario_name"]) or {}
        except Exception:
            return {}

    def _active_hazards(self) -> list[dict]:
        return [
            hazard
            for hazard in getattr(self, "_cooperative_hazards", [])
            if hazard.get("name") in self._cooperative_notice_started
            and hazard.get("name") not in self._cooperative_released
        ]

    @staticmethod
    def _driving_lane(waypoint) -> bool:
        return waypoint is not None and "driving" in _lane_type_name(waypoint.lane_type).lower()

    def _actor_lane_conflicts(self, target_waypoints, actor_list) -> list[dict]:
        target_keys = {(int(wp.road_id), int(wp.lane_id)) for wp in target_waypoints}
        ego_transform = self._vehicle.get_transform()
        ego_location = ego_transform.location
        yaw = math.radians(float(ego_transform.rotation.yaw))
        forward = (math.cos(yaw), math.sin(yaw))
        ego_velocity = self._vehicle.get_velocity()
        rear_limit = _float_env("COUNTERFACTUAL_LANE_REAR_CLEARANCE_M", 15.0)
        front_limit = _float_env("COUNTERFACTUAL_LANE_FRONT_CLEARANCE_M", 35.0)
        min_ttc = _float_env("COUNTERFACTUAL_LANE_MIN_TTC_S", 4.0)
        conflicts = []
        for actor in actor_list:
            if actor.id == self._vehicle.id or not actor.is_alive:
                continue
            type_id = str(actor.type_id)
            if not (type_id.startswith("vehicle.") or type_id.startswith("walker.")):
                continue
            try:
                location = actor.get_location()
                if location.z < -50.0:
                    continue
                waypoint = self.world_map.get_waypoint(
                    location,
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving,
                )
                if waypoint is None or (int(waypoint.road_id), int(waypoint.lane_id)) not in target_keys:
                    continue
                dx = float(location.x - ego_location.x)
                dy = float(location.y - ego_location.y)
                longitudinal = dx * forward[0] + dy * forward[1]
                if longitudinal < -rear_limit or longitudinal > front_limit:
                    continue
                velocity = actor.get_velocity()
                rel_x = float(velocity.x - ego_velocity.x)
                rel_y = float(velocity.y - ego_velocity.y)
                rel_norm = rel_x * rel_x + rel_y * rel_y
                ttc = None
                closest = math.hypot(dx, dy)
                if rel_norm > 1e-4:
                    candidate = -(dx * rel_x + dy * rel_y) / rel_norm
                    if 0.0 <= candidate <= min_ttc:
                        ttc = candidate
                        closest = math.hypot(dx + candidate * rel_x, dy + candidate * rel_y)
                conflicts.append(
                    {
                        "actor_id": int(actor.id),
                        "type_id": type_id,
                        "role_name": str(actor.attributes.get("role_name", "")),
                        "road_id": int(waypoint.road_id),
                        "lane_id": int(waypoint.lane_id),
                        "longitudinal_m": longitudinal,
                        "center_distance_m": math.hypot(dx, dy),
                        "ttc_s": ttc,
                        "predicted_closest_distance_m": closest,
                    }
                )
            except (AttributeError, RuntimeError, TypeError, ValueError):
                continue
        return conflicts

    def _audit_lane(self, side: str, actor_list) -> dict:
        planner = self._waypoint_planner
        points_per_meter = max(int(getattr(planner, "points_per_meter", 10)), 1)
        hazards = self._active_hazards()
        if hazards and hazards[0].get("collision"):
            collision = hazards[0]["collision"]
            reference_source = "active_cooperative_hazard"
        else:
            # The official ScenarioRunner hint can remain valid even when the
            # PDM-private hazard list is empty (notably S2 route 0009). The
            # frozen XML collision point is the deterministic fallback for the
            # same scenario and lets the geometry/dynamic audit reject safely
            # instead of crashing on hazards[0].
            collision = self._cf_route_spec.get("points", {}).get("collision_point")
            reference_source = "frozen_route_collision_point"
        if not collision:
            return {
                "side": side,
                "safe": False,
                "reason": "missing_conflict_reference",
                "reference_source": reference_source,
            }
        collision_location = SimpleNamespace(**collision)
        collision_index = int(
            planner.get_closest_route_index(planner.route_index, collision_location)
        )
        pre = _float_env("COUNTERFACTUAL_LANE_PRE_DISTANCE_M", 14.0)
        post = _float_env("COUNTERFACTUAL_LANE_POST_DISTANCE_M", 28.0)
        start_index = int(
            max(int(planner.route_index) + 1, collision_index - int(pre * points_per_meter))
        )
        end_index = int(
            min(len(planner.route_waypoints) - 2, collision_index + int(post * points_per_meter))
        )
        if end_index <= start_index + 4:
            return {
                "side": side,
                "safe": False,
                "reason": "insufficient_route_horizon",
                "start_index": start_index,
                "end_index": end_index,
                "collision_index": collision_index,
                "reference_source": reference_source,
            }
        sample_count = min(25, max(8, end_index - start_index + 1))
        sample_indices = sorted(
            {
                int(round(start_index + i * (end_index - start_index) / (sample_count - 1)))
                for i in range(sample_count)
            }
        )
        samples = []
        valid_waypoints = []
        same_direction = 0
        marking_violations = 0
        for index in sample_indices:
            base = planner.route_waypoints[index]
            target = base.get_left_lane() if side == "left" else base.get_right_lane()
            valid = self._driving_lane(target)
            if valid:
                vertical_delta = abs(float(target.transform.location.z - base.transform.location.z))
                lateral_delta = math.hypot(
                    float(target.transform.location.x - base.transform.location.x),
                    float(target.transform.location.y - base.transform.location.y),
                )
                valid = (
                    float(target.lane_width) >= _float_env("COUNTERFACTUAL_LANE_MIN_WIDTH_M", 2.4)
                    and vertical_delta <= 1.5
                    and lateral_delta <= 7.5
                )
            if valid:
                valid_waypoints.append(target)
                if int(base.lane_id) * int(target.lane_id) > 0:
                    same_direction += 1
            if not CooperativePdmLiteAgent._lane_change_allowed(base, side == "left"):
                marking_violations += 1
            samples.append(
                {
                    "index": index,
                    "valid": bool(valid),
                    "base_road_id": int(base.road_id),
                    "base_lane_id": int(base.lane_id),
                    "target_road_id": int(target.road_id) if target is not None else None,
                    "target_lane_id": int(target.lane_id) if target is not None else None,
                    "target_lane_type": _lane_type_name(target.lane_type) if target is not None else None,
                    "target_lane_width_m": float(target.lane_width) if target is not None else None,
                    "lane_marking_rule_ignored": not CooperativePdmLiteAgent._lane_change_allowed(
                        base, side == "left"
                    ),
                }
            )
        coverage = len(valid_waypoints) / max(len(sample_indices), 1)
        actor_conflicts = self._actor_lane_conflicts(valid_waypoints, actor_list)
        geometry_safe = coverage >= _float_env("COUNTERFACTUAL_LANE_MIN_COVERAGE", 0.8)
        safe = geometry_safe and not actor_conflicts
        reason = "safe"
        if not geometry_safe:
            reason = "no_continuous_driving_lane"
        elif actor_conflicts:
            reason = "target_lane_occupied"
        representative = valid_waypoints[len(valid_waypoints) // 2] if valid_waypoints else None
        return {
            "side": side,
            "safe": safe,
            "reason": reason,
            "start_index": start_index,
            "end_index": end_index,
            "collision_index": collision_index,
            "reference_source": reference_source,
            "coverage": coverage,
            "same_direction_ratio": same_direction / max(len(valid_waypoints), 1),
            "lane_marking_violation_ratio": marking_violations / max(len(sample_indices), 1),
            "marking_rules_ignored": True,
            "target_road_id": int(representative.road_id) if representative else None,
            "target_lane_id": int(representative.lane_id) if representative else None,
            "target_lane_width_m": float(representative.lane_width) if representative else None,
            "actor_conflicts": actor_conflicts,
            "samples": samples,
        }

    def _apply_lane_change(self, side: str, actor_list):
        info = self._audit_lane(side, actor_list)
        self._cf_lane_info = info
        self._write_event("lane_audit", audit=info)
        if not info["safe"]:
            self._abort("lane_unavailable", audit=info)
        planner = self._waypoint_planner
        points_per_meter = max(int(getattr(planner, "points_per_meter", 10)), 1)
        transition = int(
            _float_env("COUNTERFACTUAL_LANE_TRANSITION_DISTANCE_M", 6.0) * points_per_meter
        )
        transition = min(
            max(1, transition),
            max(1, (info["end_index"] - info["start_index"]) // 3),
        )
        planner.shift_route_smoothly(
            info["start_index"],
            info["end_index"],
            side == "left",
            transition_length=transition,
            lane_transition_factor=1.0,
        )
        self._cf_lane_change_applied = True
        self._write_event(
            "lane_change_route_applied",
            side=side,
            start_index=info["start_index"],
            end_index=info["end_index"],
            transition_points=transition,
        )

    def _start_decision(self, actor_list):
        velocity = self._vehicle.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        transform = self._vehicle.get_transform()
        waypoint = self.world_map.get_waypoint(
            transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        self._cf_decision_started = True
        self._cf_decision_start_time = self._simulation_time()
        self._cf_decision_entry_speed = speed
        self._cf_decision_entry_pose = {
            "location": _vector(transform.location),
            "rotation": _rotation(transform.rotation),
            "road_id": int(waypoint.road_id) if waypoint else None,
            "lane_id": int(waypoint.lane_id) if waypoint else None,
            "lane_width_m": float(waypoint.lane_width) if waypoint else None,
        }
        self._write_event(
            "decision_started",
            simulation_time_s=self._cf_decision_start_time,
            entry_speed_mps=speed,
            entry_pose=self._cf_decision_entry_pose,
            active_hazards=[hazard.get("name") for hazard in self._active_hazards()],
            event_state=self._event_state(),
        )
        if self.counterfactual_decision == "LaneChangeLeft":
            self._apply_lane_change("left", actor_list)
        elif self.counterfactual_decision == "LaneChangeRight":
            self._apply_lane_change("right", actor_list)

    def _manage_route_obstacle_scenarios(
        self,
        target_speed,
        ego_speed,
        route_waypoints,
        list_vehicles,
        route_points,
    ):
        """Remove the later 10 m/s junction cap for S6 counterfactuals.

        AutoPilot applies this method's return value *after* the high-level
        decision target.  S6 enters the warning at roughly 10.5 m/s while the
        junction cap is 10 m/s, so Accelerate could never produce acceleration
        even when the red-light guard was intentionally exposed as a candidate
        violation.  The exception is narrow, starts only after the common
        branching state, and remains fully visible in raw telemetry.
        """
        managed_speed, keep_driving, reduced_by = super()._manage_route_obstacle_scenarios(
            target_speed,
            ego_speed,
            route_waypoints,
            list_vehicles,
            route_points,
        )
        scenario_type = str(self._cf_route_spec.get("scenario_type") or "")
        if self._cf_decision_started and self.counterfactual_decision == "Accelerate":
            # AutoPilot applies this result as a second ceiling *after*
            # get_brake_and_target_speed. Without lifting it, a correctly
            # requested entry+delta target can be silently clamped back to a
            # junction/scenario cap (observed as flat 10/16/20 m/s traces).
            return (
                max(
                    float(managed_speed),
                    _float_env("COUNTERFACTUAL_ACCELERATE_CAP_MPS", 20.0),
                ),
                False,
                reduced_by,
            )
        if (
            self._cf_decision_started
            and "rightturnonred" in scenario_type.replace("_", "").lower()
            and self.counterfactual_decision != "Stop"
        ):
            return (
                max(float(managed_speed), _float_env("COUNTERFACTUAL_ACCELERATE_CAP_MPS", 20.0)),
                False,
                reduced_by,
            )
        return managed_speed, keep_driving, reduced_by

    def get_brake_and_target_speed(
        self,
        plant,
        route_points,
        distance_to_next_traffic_light,
        next_traffic_light,
        distance_to_next_stop_sign,
        next_stop_sign,
        vehicle_list,
        actor_list,
        initial_target_speed,
        speed_reduced_by_obj,
    ):
        args = (
            plant,
            route_points,
            distance_to_next_traffic_light,
            next_traffic_light,
            distance_to_next_stop_sign,
            next_stop_sign,
            vehicle_list,
            actor_list,
            initial_target_speed,
            speed_reduced_by_obj,
        )
        self._release_cleared_closed_hazards()
        self._cooperative_target_speed(actor_list)
        brake, target_speed, speed_reduced_by_obj = super(
            CooperativePdmLiteAgent, self
        ).get_brake_and_target_speed(*args)
        # ScenarioRunner updates the official hint-gate blackboard immediately
        # after the agent tick that first enters the geometric notice radius.
        # Wait for that authoritative state so every counterfactual branch starts
        # from the same released warning state, rather than one control tick early.
        event_state = self._event_state()
        # The ScenarioRunner hint gate is the authoritative counterfactual
        # branch point. Some valid cut-in routes remove the cooperative-PDM
        # hazard object on the same tick that the hint is published, so using
        # that private list as another start gate can suppress every decision.
        # Continue until the official event reports that the hazard is clear.
        active = bool(event_state.get("hint_reached", False)) and not bool(
            event_state.get("hazard_cleared", False)
        )
        if (
            self.counterfactual_decision == "Stop"
            and self._cf_decision_started
            and not self._cf_stop_goal_met
        ):
            # A short scenario may clear while the vehicle is still physically
            # decelerating. Keep Stop latched until the ego is actually at rest
            # for the configured hold time; otherwise Stop can silently become
            # a brief Brake trajectory.
            active = True
        self._cf_last_active = active
        if not active:
            self._cf_decision_suppressed_by = None
            return brake, target_speed, speed_reduced_by_obj
        if not self._cf_decision_started:
            self._start_decision(actor_list)

        # Preserve the standard expert's red-light / stop-sign target as a
        # safety ceiling.  Merely detecting a distant red light must not hand
        # the entire longitudinal policy back to PDM, otherwise Maintain can
        # silently become Accelerate while that ceiling is still increasing.
        regulatory_target_speed = float(target_speed)
        regulatory_hazard = (
            "traffic_light"
            if self.traffic_light_hazard
            else "stop_sign"
            if self.stop_sign_hazard
            else None
        )
        self._cf_decision_suppressed_by = None
        entry_speed = float(self._cf_decision_entry_speed or 0.0)
        current_speed = self._vehicle.get_velocity().length()
        decision = self.counterfactual_decision
        if decision == "Accelerate":
            target_speed = min(
                _float_env("COUNTERFACTUAL_ACCELERATE_CAP_MPS", 13.9),
                max(
                    entry_speed + _float_env("COUNTERFACTUAL_ACCELERATE_DELTA_MPS", 3.0),
                    current_speed + 1.0,
                    float(initial_target_speed),
                ),
            )
            brake = False
        elif decision == "Maintain":
            target_speed = max(entry_speed, 0.5)
            brake = False
        elif decision == "Brake":
            target_speed = max(
                _float_env("COUNTERFACTUAL_BRAKE_MIN_TARGET_MPS", 1.5),
                entry_speed * _float_env("COUNTERFACTUAL_BRAKE_SPEED_RATIO", 0.45),
            )
            brake = False
        elif decision == "Stop":
            target_speed = 0.0
            brake = True
            stop_threshold = _float_env("COUNTERFACTUAL_STOP_SPEED_MPS", 0.3)
            stop_hold = _float_env("COUNTERFACTUAL_STOP_HOLD_S", 0.5)
            if not self._cf_stop_goal_met:
                if current_speed < stop_threshold:
                    if self._cf_stop_hold_since is None:
                        self._cf_stop_hold_since = self._simulation_time()
                    elif self._simulation_time() - self._cf_stop_hold_since >= stop_hold:
                        self._cf_stop_goal_met = True
                        self._write_event(
                            "stop_hold_satisfied",
                            hold_duration_s=self._simulation_time() - self._cf_stop_hold_since,
                            speed_mps=current_speed,
                        )
                else:
                    self._cf_stop_hold_since = None
        else:
            target_speed = max(
                float(target_speed),
                min(
                    float(initial_target_speed),
                    max(entry_speed, _float_env("COUNTERFACTUAL_LANE_MIN_TARGET_MPS", 4.0)),
                ),
            )
            brake = False

        # Yellow/white lane markings are ignored only for lane-change geometry.
        # Red lights and stop signs remain hard constraints for every decision.
        if regulatory_hazard is not None and regulatory_target_speed < float(target_speed):
            scenario_type = str(self._cf_route_spec.get("scenario_type") or "")
            allow_right_turn_counterfactual = (
                "rightturnonred" in scenario_type.replace("_", "").lower()
                and self.counterfactual_decision != "Stop"
            )
            if allow_right_turn_counterfactual:
                if self._cf_regulatory_override is None:
                    self._cf_regulatory_override = {
                        "constraint": regulatory_hazard,
                        "scenario_type": scenario_type,
                        "standard_target_speed_mps": regulatory_target_speed,
                        "decision_target_speed_mps": float(target_speed),
                        "reason": "right_turn_yield_counterfactual_diversity",
                    }
                    self._write_event(
                        "regulatory_constraint_overridden",
                        override=self._cf_regulatory_override,
                    )
            else:
                target_speed = regulatory_target_speed
                brake = target_speed <= 0.05
                self._cf_decision_suppressed_by = regulatory_hazard
        self._cf_decision_target_speed = float(target_speed)
        speed_reduced_by_obj = [
            float(target_speed),
            f"counterfactual_decision:{decision}",
            -1,
            min(
                (
                    self._distance(self._vehicle.get_location(), hazard["collision"])
                    for hazard in self._active_hazards()
                ),
                default=None,
            ),
        ]
        return brake, target_speed, speed_reduced_by_obj

    def _serialize_lane(self, location) -> dict | None:
        try:
            waypoint = self.world_map.get_waypoint(
                location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if waypoint is None:
                return None
            return {
                "road_id": int(waypoint.road_id),
                "section_id": int(waypoint.section_id),
                "lane_id": int(waypoint.lane_id),
                "s": float(waypoint.s),
                "is_junction": bool(waypoint.is_junction),
                "lane_type": _lane_type_name(waypoint.lane_type),
                "lane_width_m": float(waypoint.lane_width),
                "lane_change": str(waypoint.lane_change),
                "left_marking_type": str(waypoint.left_lane_marking.type),
                "left_marking_color": str(waypoint.left_lane_marking.color),
                "right_marking_type": str(waypoint.right_lane_marking.type),
                "right_marking_color": str(waypoint.right_lane_marking.color),
                "center": _vector(waypoint.transform.location),
                "heading": _rotation(waypoint.transform.rotation),
            }
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None

    def _serialize_actor(self, actor, ego_location) -> dict | None:
        try:
            transform = actor.get_transform()
            location = transform.location
            distance = math.sqrt(
                (location.x - ego_location.x) ** 2
                + (location.y - ego_location.y) ** 2
                + (location.z - ego_location.z) ** 2
            )
            role = str(actor.attributes.get("role_name", ""))
            radius = _float_env("COUNTERFACTUAL_RAW_ACTOR_RADIUS_M", 100.0)
            if distance > radius and role != "scenario":
                return None
            velocity = actor.get_velocity()
            acceleration = actor.get_acceleration()
            angular_velocity = actor.get_angular_velocity()
            payload = {
                "id": int(actor.id),
                "type_id": str(actor.type_id),
                "role_name": role,
                "distance_to_ego_m": distance,
                "transform": {
                    "location": _vector(location),
                    "rotation": _rotation(transform.rotation),
                    "matrix": transform.get_matrix(),
                },
                "velocity": _vector(velocity),
                "acceleration": _vector(acceleration),
                "angular_velocity_deg_s": _vector(angular_velocity),
                "speed_mps": math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2),
                "lane": self._serialize_lane(location),
                "attributes": dict(actor.attributes),
            }
            if hasattr(actor, "bounding_box"):
                payload["bounding_box"] = {
                    "location": _vector(actor.bounding_box.location),
                    "rotation": _rotation(actor.bounding_box.rotation),
                    "extent": _vector(actor.bounding_box.extent),
                }
            if str(actor.type_id).startswith("vehicle."):
                control = actor.get_control()
                payload["control"] = {
                    "throttle": float(control.throttle),
                    "steer": float(control.steer),
                    "brake": float(control.brake),
                    "hand_brake": bool(control.hand_brake),
                    "reverse": bool(control.reverse),
                    "gear": int(control.gear),
                }
                payload["speed_limit_kmh"] = float(actor.get_speed_limit())
                payload["traffic_light_state"] = str(actor.get_traffic_light_state())
            return payload
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None

    def _record_telemetry(self, control, timestamp):
        if not getattr(self, "initialized", False) or not hasattr(self, "_vehicle"):
            return
        snapshot = self._world.get_snapshot()
        transform = self._vehicle.get_transform()
        velocity = self._vehicle.get_velocity()
        acceleration = self._vehicle.get_acceleration()
        angular_velocity = self._vehicle.get_angular_velocity()
        actors = []
        for actor in self._world.get_actors():
            type_id = str(actor.type_id)
            if not (
                type_id.startswith("vehicle.")
                or type_id.startswith("walker.")
                or type_id.startswith("traffic.traffic_light")
                or type_id.startswith("traffic.stop")
            ):
                continue
            payload = self._serialize_actor(actor, transform.location)
            if payload is not None:
                actors.append(payload)
        weather = self._world.get_weather()
        payload = {
            "schema_version": "counterfactual_carla_telemetry_v1",
            "route_id": self._cf_route_spec["route_id"],
            "scenario_type": self._cf_route_spec["scenario_type"],
            "decision": self.counterfactual_decision,
            "agent_step": int(self.step),
            "world_frame": int(snapshot.frame),
            "timestamp_input": float(timestamp) if timestamp is not None else None,
            "simulation_time_s": float(snapshot.timestamp.elapsed_seconds),
            "delta_seconds": float(snapshot.timestamp.delta_seconds),
            "platform_time_s": float(snapshot.timestamp.platform_timestamp),
            "decision_active": bool(self._cf_last_active),
            "decision_started": bool(self._cf_decision_started),
            "decision_start_time_s": self._cf_decision_start_time,
            "decision_entry_speed_mps": self._cf_decision_entry_speed,
            "decision_target_speed_mps": self._cf_decision_target_speed,
            "decision_suppressed_by": self._cf_decision_suppressed_by,
            "regulatory_override": self._cf_regulatory_override,
            "lane_change_applied": self._cf_lane_change_applied,
            "lane_audit": self._cf_lane_info,
            "event_state": self._event_state(),
            "active_hazards": [hazard.get("name") for hazard in self._active_hazards()],
            "ego": {
                "id": int(self._vehicle.id),
                "transform": {
                    "location": _vector(transform.location),
                    "rotation": _rotation(transform.rotation),
                    "matrix": transform.get_matrix(),
                },
                "velocity": _vector(velocity),
                "acceleration": _vector(acceleration),
                "angular_velocity_deg_s": _vector(angular_velocity),
                "speed_mps": math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2),
                "lane": self._serialize_lane(transform.location),
                "control": {
                    "throttle": float(getattr(control, "throttle", 0.0)),
                    "steer": float(getattr(control, "steer", 0.0)),
                    "brake": float(getattr(control, "brake", 0.0)),
                    "hand_brake": bool(getattr(control, "hand_brake", False)),
                    "reverse": bool(getattr(control, "reverse", False)),
                    "gear": int(getattr(control, "gear", 0)),
                },
                "speed_limit_kmh": float(self._vehicle.get_speed_limit()),
            },
            "route_planner": {
                "route_index": int(self._waypoint_planner.route_index),
                "changed_route": bool(
                    (
                        self._waypoint_planner.route_points[self._waypoint_planner.route_index]
                        != self._waypoint_planner.original_route_points[self._waypoint_planner.route_index]
                    ).any()
                ),
            },
            "weather": {
                name: float(getattr(weather, name))
                for name in (
                    "cloudiness",
                    "precipitation",
                    "precipitation_deposits",
                    "wind_intensity",
                    "sun_azimuth_angle",
                    "sun_altitude_angle",
                    "fog_density",
                    "fog_distance",
                    "wetness",
                )
            },
            "actors": actors,
            "collision": self._cf_collision,
        }
        with self._cf_lock:
            self._cf_telemetry_stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            if self.step % 20 == 0:
                self._cf_telemetry_stream.flush()

    def _check_deadlock(self):
        if not self._cf_decision_started:
            return
        now = self._simulation_time()
        state = self._event_state()
        cleared = bool(state.get("hazard_cleared", False))
        velocity = self._vehicle.get_velocity()
        # Deadlock means no progress on the road plane. CARLA can report a
        # vertical bounce/fall velocity for an otherwise stationary ego;
        # counting z would hide a real route deadlock.
        speed = math.hypot(velocity.x, velocity.y)
        if (
            self._cf_decision_start_time is not None
            and now - self._cf_decision_start_time
            >= _float_env("COUNTERFACTUAL_MAX_DECISION_ACTIVE_S", 20.0)
            and not cleared
        ):
            self._abort("scenario_deadlock", active_duration_s=now - self._cf_decision_start_time)
        if speed < _float_env("COUNTERFACTUAL_DEADLOCK_SPEED_MPS", 0.1):
            if self.counterfactual_decision != "Stop" and self._cf_last_active:
                if self._cf_low_speed_since is None:
                    self._cf_low_speed_since = now
                elif now - self._cf_low_speed_since >= _float_env("COUNTERFACTUAL_DEADLOCK_HOLD_S", 2.0):
                    self._abort("vehicle_deadlock", low_speed_duration_s=now - self._cf_low_speed_since)
            elif cleared:
                if self._cf_post_clear_low_speed_since is None:
                    self._cf_post_clear_low_speed_since = now
                elif now - self._cf_post_clear_low_speed_since >= _float_env(
                    "COUNTERFACTUAL_POST_CLEAR_DEADLOCK_HOLD_S", 3.0
                ):
                    self._abort(
                        "post_clear_deadlock",
                        low_speed_duration_s=now - self._cf_post_clear_low_speed_since,
                    )
        else:
            self._cf_low_speed_since = None
            self._cf_post_clear_low_speed_since = None

    def run_step(self, input_data, timestamp, sensors=None, plant=False):
        if self._cf_collision is not None:
            self._abort("collision", collision=self._cf_collision)
        control = super().run_step(input_data, timestamp, sensors=sensors, plant=plant)
        self._ensure_runtime_monitors()
        self._record_telemetry(control, timestamp)
        if self._cf_collision is not None:
            self._abort("collision", collision=self._cf_collision)
        self._check_deadlock()
        return control

    def destroy(self, results=None):
        try:
            if self._cf_recorder_started:
                client = self._cf_recorder_client
                if client is not None:
                    client.stop_recorder()
                self._cf_recorder_started = False
                self._cf_recorder_client = None
                self._write_event("carla_recorder_stopped")
        except (AttributeError, RuntimeError) as error:
            self._write_event("carla_recorder_stop_failed", error=str(error))
        try:
            for criterion, original in self._cf_collision_criteria_hooks:
                criterion._count_collisions = original
            self._cf_collision_criteria_hooks = []
        except (AttributeError, RuntimeError) as error:
            self._write_event("collision_monitor_restore_failed", error=str(error))
        try:
            stream = getattr(self, "_cf_telemetry_stream", None)
            if stream is not None:
                stream.flush()
                stream.close()
                self._cf_telemetry_stream = None
        finally:
            super().destroy(results)
