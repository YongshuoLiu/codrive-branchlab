#!/usr/bin/env python3
"""Add factorized decision labels and Decision Reward v2 to branch metadata.

The migration is deliberately limited to canonical branch directories whose
name exactly matches the configured decision taxonomy.  Historical
``*_incomplete_*`` directories are never modified.

Writes performed with ``--write``:

* ``metadata/run_spec.json`` receives factorized decision-label fields.
* ``quality_report.json`` receives flat training fields plus a complete nested
  ``decision_reward`` audit record.
* a compact branch/group index is written below
  ``_annotations/decision_reward_v2``.

The operation is atomic per JSON file and idempotent.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import statistics
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

from score_decision_reward_v1 import (
    actor_radius,
    base_branch,
    clip,
    evaluation_start,
    finalize_group,
    hazard_clear_index,
    percentile,
    read_json,
    read_telemetry,
    vector,
)


RL_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = RL_ROOT / "data/counterfactual_decision_v1"
REWARD_VERSION = "decision_reward_v2.0"
LABEL_SCHEMA_VERSION = "factorized_decision_4x3_v1"
ANNOTATION_SCHEMA_VERSION = "counterfactual_decision_training_annotation_v2"

EXPECTED_DECISIONS = {
    "S1": ("Accelerate", "Maintain", "Brake", "Stop", "LaneChangeLeft", "LaneChangeRight"),
    "S2": ("Accelerate", "Maintain", "Brake", "Stop", "LaneChangeLeft", "LaneChangeRight"),
    "S3": ("Accelerate", "Maintain", "Brake", "Stop", "LaneChangeLeft", "LaneChangeRight"),
    "S4": ("Accelerate", "Maintain", "Brake", "Stop"),
    "S5": ("Accelerate", "Maintain", "Brake", "Stop"),
    "S6": ("Accelerate", "Maintain", "Brake", "Stop"),
}

LONGITUDINAL_CLASSES = ("Accelerate", "Maintain", "Brake", "Stop")
LATERAL_CLASSES = ("RouteFollow", "LaneChangeLeft", "LaneChangeRight")
DECISION_LABELS = {
    "Accelerate": ("Accelerate", "RouteFollow"),
    "Maintain": ("Maintain", "RouteFollow"),
    "Brake": ("Brake", "RouteFollow"),
    "Stop": ("Stop", "RouteFollow"),
    "LaneChangeLeft": ("Maintain", "LaneChangeLeft"),
    "LaneChangeRight": ("Maintain", "LaneChangeRight"),
}

SAFE_CLEARANCE_M = {
    "pedestrian": 3.0,
    "vehicle": 4.0,
    "static": 2.0,
}
TTC_HORIZON_S = 4.0
TTC_FULL_RISK_S = 0.5
PET_SAFE_S = 2.0
PET_FULL_RISK_S = 0.5
MAX_VERTICAL_SEPARATION_M = 2.5
HAZARD_PENALTY_WEIGHT = 70.0
SOFT_TARGET_WINDOW = 10.0
TOP_DECISION_WINDOW = 5.0
SOFTMAX_TEMPERATURE = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=pathlib.Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--route-id", action="append", help="Limit to an exact route id; repeatable")
    parser.add_argument("--scenario", action="append", choices=tuple(EXPECTED_DECISIONS))
    parser.add_argument("--write", action="store_true", help="Apply annotations; otherwise dry-run")
    parser.add_argument("--sample-limit", type=int, default=24)
    return parser.parse_args()


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    for key in (
        "decision_label",
        "decision_reward",
        "decision_longitudinal",
        "decision_lateral",
        "decision_reward_score",
        "decision_reward_version",
        "decision_target_probability",
        "decision_top_label",
        "decision_training_eligible",
    ):
        clean.pop(key, None)
    raw = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def decision_label(decision: str) -> dict[str, Any]:
    longitudinal, lateral = DECISION_LABELS[decision]
    return {
        "schema_version": LABEL_SCHEMA_VERSION,
        "branch_decision": decision,
        "joint_decision": f"{longitudinal}+{lateral}",
        "longitudinal": longitudinal,
        "longitudinal_id": LONGITUDINAL_CLASSES.index(longitudinal),
        "lateral": lateral,
        "lateral_id": LATERAL_CLASSES.index(lateral),
        "longitudinal_classes": list(LONGITUDINAL_CLASSES),
        "lateral_classes": list(LATERAL_CLASSES),
    }


def actor_kind(actor: dict[str, Any]) -> str:
    type_id = str(actor.get("type_id", ""))
    if type_id.startswith("walker."):
        return "pedestrian"
    if type_id.startswith("vehicle."):
        return "vehicle"
    return "static"


def evaluation_window(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    start = evaluation_start(frames)
    if start is None:
        return []
    clear_index = hazard_clear_index(frames, start)
    end = len(frames)
    if clear_index is not None:
        clear_time = float(frames[clear_index].get("simulation_time_s", 0.0))
        for index in range(clear_index, len(frames)):
            now = float(frames[index].get("simulation_time_s", clear_time))
            if now > clear_time + 3.0:
                end = index + 1
                break
    return frames[start:end]


def lane_geometry(frame: dict[str, Any]) -> dict[str, Any]:
    ego = frame.get("ego") or {}
    lane = ego.get("lane") or {}
    heading = lane.get("heading") or {}
    yaw = math.radians(float(heading.get("yaw", 0.0)))
    forward = (math.cos(yaw), math.sin(yaw))
    return {
        "center": vector(lane.get("center")),
        "forward": forward,
        "left": (-forward[1], forward[0]),
        "width": max(float(lane.get("lane_width_m") or 4.0), 1.0),
        "road_id": lane.get("road_id"),
    }


def actor_on_driving_lane(actor: dict[str, Any], fallback_width: float) -> bool:
    lane = actor.get("lane") or {}
    if lane.get("lane_type") != "Driving":
        return False
    location = vector((actor.get("transform") or {}).get("location"))
    center = vector(lane.get("center"))
    width = max(float(lane.get("lane_width_m") or fallback_width), 1.0)
    offset = math.hypot(location[0] - center[0], location[1] - center[1])
    return offset <= width / 2.0 + actor_radius(actor, 0.3)


def scenario_actor(actor: dict[str, Any]) -> bool:
    return actor.get("role_name") == "scenario"


def frame_hazard(scenario_id: str, frame: dict[str, Any]) -> dict[str, Any]:
    ego = frame.get("ego") or {}
    ego_location = vector((ego.get("transform") or {}).get("location"))
    ego_velocity = vector(ego.get("velocity"))
    geometry = lane_geometry(frame)
    hero = next(
        (actor for actor in frame.get("actors", []) if actor.get("role_name") == "hero"),
        {},
    )
    ego_radius = actor_radius(hero, 2.6)
    best = {
        "risk": 0.0,
        "distance_risk": 0.0,
        "ttc_risk": 0.0,
        "actor_id": None,
        "actor_type": None,
        "actor_kind": None,
        "clearance_m": None,
        "ttc_s": None,
        "predicted_clearance_m": None,
        "longitudinal_m": None,
        "lateral_m": None,
    }

    for actor in frame.get("actors", []):
        if not scenario_actor(actor):
            continue
        kind = actor_kind(actor)
        if scenario_id == "S1" and kind != "pedestrian":
            continue

        location = vector((actor.get("transform") or {}).get("location"))
        actor_location_raw = (actor.get("transform") or {}).get("location") or {}
        ego_location_raw = (ego.get("transform") or {}).get("location") or {}
        vertical_separation = abs(
            float(actor_location_raw.get("z", 0.0))
            - float(ego_location_raw.get("z", 0.0))
        )
        velocity = vector(actor.get("velocity"))
        radius = actor_radius(actor, 0.3 if kind == "pedestrian" else 1.5)
        relative = (location[0] - ego_location[0], location[1] - ego_location[1])
        distance = math.hypot(*relative)
        if distance <= 1e-6 or distance > 100.0:
            continue
        longitudinal = (
            relative[0] * geometry["forward"][0]
            + relative[1] * geometry["forward"][1]
        )
        from_lane_center = (
            location[0] - geometry["center"][0],
            location[1] - geometry["center"][1],
        )
        lateral = (
            from_lane_center[0] * geometry["left"][0]
            + from_lane_center[1] * geometry["left"][1]
        )

        same_road = (actor.get("lane") or {}).get("road_id") == geometry["road_id"]
        within_ego_or_adjacent = abs(lateral) <= 1.5 * geometry["width"] + radius
        on_lane = actor_on_driving_lane(actor, geometry["width"])
        if scenario_id == "S1":
            # At junctions CARLA can assign the pedestrian and ego to different
            # connected road IDs even while their paths physically intersect.
            # The S1 policy is geometric (ahead + ego/adjacent driving-lane
            # corridor), so exact road-ID equality would hide genuine near
            # misses.  The Z gate prevents geometrically overlapping overpasses
            # from becoming false hazards.
            corridor_relevant = (
                longitudinal > 0.0
                and vertical_separation <= MAX_VERTICAL_SEPARATION_M
                and within_ego_or_adjacent
                and on_lane
            )
        else:
            corridor_relevant = (
                longitudinal > 0.0
                and same_road
                and within_ego_or_adjacent
                and on_lane
            )

        relative_velocity = (
            velocity[0] - ego_velocity[0],
            velocity[1] - ego_velocity[1],
        )
        velocity_norm_sq = sum(value * value for value in relative_velocity)
        ttc = None
        predicted_clearance = None
        predicted_conflict = False
        safe_clearance = SAFE_CLEARANCE_M[kind]
        if velocity_norm_sq > 1e-4:
            candidate = -sum(
                relative[axis] * relative_velocity[axis] for axis in range(2)
            ) / velocity_norm_sq
            if 0.0 <= candidate <= TTC_HORIZON_S:
                predicted_distance = math.hypot(
                    relative[0] + candidate * relative_velocity[0],
                    relative[1] + candidate * relative_velocity[1],
                )
                predicted_clearance = predicted_distance - ego_radius - radius
                predicted_conflict = predicted_clearance <= safe_clearance
                if predicted_conflict:
                    ttc = candidate

        # S1 has the explicit user-specified pedestrian gate: only an ahead
        # pedestrian physically occupying the ego/adjacent driving lanes counts.
        # Other scenarios additionally admit side/rear actors when their motion
        # predicts a real conflict (cut-in, cross traffic, target-lane vehicle).
        relevant = corridor_relevant if scenario_id == "S1" else (
            corridor_relevant or predicted_conflict
        )
        if not relevant:
            continue

        clearance = distance - ego_radius - radius
        distance_risk = clip((safe_clearance - clearance) / safe_clearance)
        ttc_risk = 0.0
        if ttc is not None:
            ttc_risk = clip(
                (TTC_HORIZON_S - ttc) / (TTC_HORIZON_S - TTC_FULL_RISK_S)
            )
        risk = 0.35 * distance_risk + 0.65 * ttc_risk
        if risk > float(best["risk"]):
            best = {
                "risk": risk,
                "distance_risk": distance_risk,
                "ttc_risk": ttc_risk,
                "actor_id": actor.get("id"),
                "actor_type": actor.get("type_id"),
                "actor_kind": kind,
                "clearance_m": clearance,
                "ttc_s": ttc,
                "predicted_clearance_m": predicted_clearance,
                "longitudinal_m": longitudinal,
                "lateral_m": lateral,
            }
    return best


def approximate_pet(
    frames: list[dict[str, Any]], relevant_actor_ids: set[int]
) -> float | None:
    if not relevant_actor_ids or not frames:
        return None
    actor_paths: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    ego_path: list[tuple[float, float, float, float]] = []
    for index, frame in enumerate(frames):
        if index % 2:
            continue
        now = float(frame.get("simulation_time_s", 0.0))
        ego = frame.get("ego") or {}
        ego_location = vector((ego.get("transform") or {}).get("location"))
        hero = next(
            (actor for actor in frame.get("actors", []) if actor.get("role_name") == "hero"),
            {},
        )
        ego_path.append((ego_location[0], ego_location[1], now, actor_radius(hero, 2.6)))
        for actor in frame.get("actors", []):
            actor_id = actor.get("id")
            if actor_id not in relevant_actor_ids:
                continue
            location = vector((actor.get("transform") or {}).get("location"))
            actor_paths[int(actor_id)].append(
                (location[0], location[1], now, actor_radius(actor, 0.5))
            )

    minimum = None
    cell_size = 6.0
    for points in actor_paths.values():
        buckets: dict[tuple[int, int], list[tuple[float, float, float, float]]] = defaultdict(list)
        for point in points:
            buckets[(math.floor(point[0] / cell_size), math.floor(point[1] / cell_size))].append(point)
        for ex, ey, ego_time, ego_radius_m in ego_path:
            cell = (math.floor(ex / cell_size), math.floor(ey / cell_size))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for ax, ay, actor_time, actor_radius_m in buckets.get(
                        (cell[0] + dx, cell[1] + dy), []
                    ):
                        conflict_radius = ego_radius_m + actor_radius_m + 1.0
                        if math.hypot(ex - ax, ey - ay) > conflict_radius:
                            continue
                        pet = abs(ego_time - actor_time)
                        minimum = pet if minimum is None else min(minimum, pet)
    return minimum


def hazard_metrics(scenario_id: str, run_dir: pathlib.Path) -> dict[str, Any]:
    frames = read_telemetry(run_dir / "raw/carla_telemetry.jsonl.gz")
    window = evaluation_window(frames)
    frame_rows = [frame_hazard(scenario_id, frame) for frame in window]
    risks = [float(row["risk"]) for row in frame_rows]
    relevant = [row for row in frame_rows if row["actor_id"] is not None]
    actor_ids = {int(row["actor_id"]) for row in relevant if row["actor_id"] is not None}
    clearances = [float(row["clearance_m"]) for row in relevant if row["clearance_m"] is not None]
    ttcs = [float(row["ttc_s"]) for row in relevant if row["ttc_s"] is not None]
    predicted = [
        float(row["predicted_clearance_m"])
        for row in relevant
        if row["predicted_clearance_m"] is not None
    ]
    p95 = percentile(risks, 0.95)
    mean = statistics.fmean(risks) if risks else 0.0
    pet = approximate_pet(window, actor_ids)
    pet_risk = 0.0
    if pet is not None:
        pet_risk = clip((PET_SAFE_S - pet) / (PET_SAFE_S - PET_FULL_RISK_S))
    aggregate = clip(0.65 * p95 + 0.25 * mean + 0.10 * pet_risk)

    exposure_s = 0.0
    for index in range(1, len(window)):
        if risks[index] <= 0.6:
            continue
        previous = float(window[index - 1].get("simulation_time_s", 0.0))
        now = float(window[index].get("simulation_time_s", previous))
        exposure_s += min(max(now - previous, 0.0), 0.25)

    return {
        "schema_version": "hazard_risk_v2",
        "evaluation_frames": len(window),
        "relevant_frames": len(relevant),
        "relevant_actor_ids": sorted(actor_ids),
        "minimum_clearance_m": min(clearances, default=None),
        "minimum_ttc_s": min(ttcs, default=None),
        "minimum_predicted_clearance_m": min(predicted, default=None),
        "approximate_pet_s": pet,
        "pet_method": "downsampled_trajectory_point_pair",
        "p95_frame_risk": p95,
        "mean_frame_risk": mean,
        "pet_risk": pet_risk,
        "risk_exposure_above_0_6_s": exposure_s,
        "aggregate_risk": aggregate,
        "penalty": HAZARD_PENALTY_WEIGHT * aggregate,
        "frame_risk_formula": "0.35*distance_risk + 0.65*ttc_risk",
        "aggregate_formula": "0.65*p95 + 0.25*mean + 0.10*pet_risk",
    }


def compute_static(job: tuple[str, str, str]) -> tuple[str, dict[str, Any]]:
    scenario_id, decision, raw_path = job
    path = pathlib.Path(raw_path)
    quality = read_json(path / "quality_report.json")
    try:
        row = base_branch(scenario_id, decision, path)
        hazard = hazard_metrics(scenario_id, path)
        row["hazard"] = hazard
        if row.get("components"):
            row["risk"] = hazard["aggregate_risk"]
            row["components"]["risk_margin"] = 1.0 - hazard["aggregate_risk"]
        row["source_quality_sha256"] = canonical_json_sha256(quality)
        row["static_compute_error"] = None
    except Exception as exc:  # Preserve a branch-level failure instead of aborting all data.
        row = {
            "scenario_id": scenario_id,
            "route_id": quality.get("route_id", path.parent.name),
            "decision": decision,
            "run_dir": str(path.resolve()),
            "outcome": quality.get("outcome", "missing"),
            "quality_accepted": quality.get("accepted") is True,
            "reward_valid": False,
            "exclusion_reason": "annotation_compute_error",
            "source_quality_sha256": canonical_json_sha256(quality),
            "static_compute_error": f"{type(exc).__name__}: {exc}",
            "hazard": {},
        }
    return str(path.resolve()), row


def task_score(row: dict[str, Any]) -> float | None:
    component = row.get("components") or {}
    required = {
        "progress",
        "recovery",
        "efficiency",
        "comfort",
        "decision_fidelity",
        "legality",
        "unnecessary_intervention",
    }
    if not required.issubset(component):
        return None
    weighted = (
        0.22 * float(component["progress"])
        + 0.16 * float(component["recovery"])
        + 0.12 * float(component["efficiency"])
        + 0.10 * float(component["comfort"])
        + 0.05 * float(component["decision_fidelity"])
        + 0.03 * float(component["legality"])
    )
    value = 100.0 * weighted / 0.68
    value -= 15.0 * float(component["unnecessary_intervention"])
    return clip(value, 0.0, 100.0)


def hard_outcome(row: dict[str, Any]) -> tuple[str, int, float | None]:
    if not row.get("quality_accepted") or not row.get("reward_valid"):
        return "invalid_quality", -1, None
    counts = row.get("collision_counts") or {}
    if int(counts.get("collisions_pedestrian", 0)) > 0:
        return "pedestrian_collision", 0, -100.0
    if int(counts.get("collisions_vehicle", 0)) > 0:
        return "vehicle_collision", 1, -95.0
    if int(counts.get("collisions_layout", 0)) > 0:
        return "layout_collision", 2, -90.0
    if row.get("offroad_or_route_violation"):
        return "route_violation", 3, -85.0
    if row.get("lane_unavailable") or row.get("decision_feasible") is False:
        return "action_infeasible", 4, -75.0
    if row.get("deadlock"):
        return "deadlock", 5, -60.0
    return "safe", 6, None


def apply_safety_cap(score: float, hazard: dict[str, Any]) -> tuple[float, float | None, list[str]]:
    cap = None
    reasons = []
    clearance = hazard.get("minimum_clearance_m")
    pet = hazard.get("approximate_pet_s")
    exposure = float(hazard.get("risk_exposure_above_0_6_s") or 0.0)
    risk = float(hazard.get("aggregate_risk") or 0.0)
    if (clearance is not None and float(clearance) < 1.0) or (
        pet is not None and float(pet) < 0.5
    ):
        cap = 30.0
        reasons.append("severe_near_miss")
    elif (clearance is not None and float(clearance) < 2.0) or (
        pet is not None and float(pet) < 1.0
    ):
        cap = 60.0
        reasons.append("near_miss")
    if exposure > 1.0 and risk > 0.6:
        cap = min(cap if cap is not None else 100.0, 50.0)
        reasons.append("sustained_high_risk")
    return (min(score, cap), cap, reasons) if cap is not None else (score, None, reasons)


def finalize_reward_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    finalize_group(rows)
    for row in rows:
        base = task_score(row)
        tier, rank, fixed = hard_outcome(row)
        hazard = row.get("hazard") or {}
        cap = None
        cap_reasons: list[str] = []
        if fixed is not None:
            final = fixed
        elif tier == "safe" and base is not None:
            final = max(0.0, base - HAZARD_PENALTY_WEIGHT * float(hazard.get("aggregate_risk") or 0.0))
            final, cap, cap_reasons = apply_safety_cap(final, hazard)
        else:
            final = None
        row.update(
            {
                "task_score_before_hazard_penalty": round(base, 6) if base is not None else None,
                "hard_safety_tier": tier,
                "hard_safety_tier_rank": rank,
                "safety_cap": cap,
                "safety_cap_reasons": cap_reasons,
                "final_reward": round(final, 6) if final is not None else None,
                "action_feasible": tier != "action_infeasible",
                "candidate_for_decision_ranking": (
                    final is not None and tier != "action_infeasible"
                ),
            }
        )

    for row in rows:
        row["target_probability"] = 0.0
        row["top_decision"] = False
    safe = [
        row
        for row in rows
        if row.get("candidate_for_decision_ranking")
        and row.get("hard_safety_tier") == "safe"
    ]
    if not safe:
        return {
            "label_valid": False,
            "label_reason": "no_safe_candidate",
            "longitudinal_target_distribution": {key: 0.0 for key in LONGITUDINAL_CLASSES},
            "lateral_target_distribution": {key: 0.0 for key in LATERAL_CLASSES},
        }
    best = max(float(row["final_reward"]) for row in safe)
    targets = [row for row in safe if float(row["final_reward"]) >= best - SOFT_TARGET_WINDOW]
    weights = [
        math.exp((float(row["final_reward"]) - best) / SOFTMAX_TEMPERATURE)
        for row in targets
    ]
    denominator = sum(weights)
    for row, weight in zip(targets, weights):
        row["target_probability"] = weight / denominator
        row["top_decision"] = float(row["final_reward"]) >= best - TOP_DECISION_WINDOW

    lon = {key: 0.0 for key in LONGITUDINAL_CLASSES}
    lat = {key: 0.0 for key in LATERAL_CLASSES}
    for row in rows:
        label = decision_label(str(row["decision"]))
        probability = float(row["target_probability"])
        lon[label["longitudinal"]] += probability
        lat[label["lateral"]] += probability
    return {
        "label_valid": True,
        "label_reason": "safe_candidates_ranked",
        "best_reward": best,
        "longitudinal_target_distribution": {key: round(value, 8) for key, value in lon.items()},
        "lateral_target_distribution": {key: round(value, 8) for key, value in lat.items()},
    }


def discover_groups(
    data_root: pathlib.Path,
    scenarios: set[str],
    route_filter: set[str] | None,
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, str]]]:
    groups = []
    all_paths: dict[str, tuple[str, str]] = {}
    v2_root = data_root / "_recollected/S1_PED_SPEED_V2/S1"
    for scenario_id in sorted(scenarios):
        scenario_root = data_root / scenario_id
        if not scenario_root.is_dir():
            continue
        for route_dir in sorted(path for path in scenario_root.iterdir() if path.is_dir()):
            route_id = route_dir.name
            if route_filter is not None and route_id not in route_filter:
                continue
            baseline = {
                decision: route_dir / decision
                for decision in EXPECTED_DECISIONS[scenario_id]
                if (route_dir / decision / "quality_report.json").is_file()
            }
            if not baseline:
                continue
            if scenario_id == "S1":
                active = dict(baseline)
                for decision in ("Accelerate", "Maintain"):
                    candidate = v2_root / route_id / decision
                    if (candidate / "quality_report.json").is_file():
                        active[decision] = candidate
                groups.append(
                    {
                        "group_id": f"S1/{route_id}",
                        "scenario_id": "S1",
                        "route_id": route_id,
                        "variant": "active_s1_v2_accelerate_maintain",
                        "active_for_training": True,
                        "paths": active,
                        "annotate_decisions": set(active),
                        "policy_alignment": "explicit_s1_v2_override",
                    }
                )
                legacy_decisions = {key for key in ("Accelerate", "Maintain") if key in baseline}
                if legacy_decisions:
                    groups.append(
                        {
                            "group_id": f"S1_LEGACY/{route_id}",
                            "scenario_id": "S1",
                            "route_id": route_id,
                            "variant": "legacy_s1_superseded",
                            "active_for_training": False,
                            "paths": baseline,
                            "annotate_decisions": legacy_decisions,
                            "policy_alignment": "legacy_baseline",
                        }
                    )
            else:
                groups.append(
                    {
                        "group_id": f"{scenario_id}/{route_id}",
                        "scenario_id": scenario_id,
                        "route_id": route_id,
                        "variant": "active_baseline",
                        "active_for_training": True,
                        "paths": baseline,
                        "annotate_decisions": set(baseline),
                        "policy_alignment": "baseline_same_route",
                    }
                )
            for decision, path in baseline.items():
                all_paths[str(path.resolve())] = (scenario_id, decision)
            if scenario_id == "S1":
                for decision, path in active.items():
                    all_paths[str(path.resolve())] = (scenario_id, decision)
    return groups, all_paths


def atomic_write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = (path.stat().st_mode & 0o777) if path.exists() else 0o644
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def atomic_write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = (path.stat().st_mode & 0o777) if path.exists() else 0o644
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def compact_branch_record(annotation: dict[str, Any]) -> dict[str, Any]:
    reward = annotation["decision_reward"]
    label = annotation["decision_label"]
    return {
        "scenario_id": reward["scenario_id"],
        "route_id": reward["route_id"],
        "group_id": reward["group_id"],
        "branch_decision": label["branch_decision"],
        "longitudinal": label["longitudinal"],
        "lateral": label["lateral"],
        "run_dir": reward["run_dir"],
        "dataset_variant": reward["dataset_variant"],
        "active_for_training": reward["active_for_training"],
        "quality_valid": reward["quality_valid"],
        "action_feasible": reward["action_feasible"],
        "candidate_for_decision_ranking": reward["candidate_for_decision_ranking"],
        "hard_safety_tier": reward["hard_safety_tier"],
        "final_reward": reward["final_reward"],
        "target_probability": reward["target_probability"],
        "top_decision": reward["top_decision"],
    }


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    if RL_ROOT.resolve() not in data_root.parents and data_root != RL_ROOT.resolve():
        raise SystemExit(f"data root must stay under {RL_ROOT.resolve()}")
    scenarios = set(args.scenario or EXPECTED_DECISIONS)
    route_filter = set(args.route_id) if args.route_id else None
    groups, physical_paths = discover_groups(data_root, scenarios, route_filter)
    if not groups:
        raise SystemExit("no canonical branch groups found")

    jobs = [
        (scenario_id, decision, path)
        for path, (scenario_id, decision) in sorted(physical_paths.items())
    ]
    print(
        json.dumps(
            {
                "phase": "compute_static_metrics",
                "mode": "write" if args.write else "dry_run",
                "groups": len(groups),
                "physical_branches": len(jobs),
                "workers": max(args.workers, 1),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    started = time.monotonic()
    cache: dict[str, dict[str, Any]] = {}
    issues = []
    with ProcessPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        futures = {executor.submit(compute_static, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            path, row = future.result()
            cache[path] = row
            if row.get("static_compute_error"):
                issues.append(
                    {
                        "run_dir": path,
                        "error": row["static_compute_error"],
                        "kind": "reward_compute_error",
                        "severity": "error" if row.get("quality_accepted") else "warning",
                    }
                )
            if index == 1 or index % 100 == 0 or index == len(futures):
                elapsed = max(time.monotonic() - started, 1e-6)
                print(
                    f"[compute] {index:5d}/{len(futures):5d}  "
                    f"{index / len(futures) * 100:6.2f}%  "
                    f"{index / elapsed:6.1f} branch/s  issues={len(issues)}",
                    flush=True,
                )

    annotations: dict[str, dict[str, Any]] = {}
    group_index = []
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    for group_number, group in enumerate(groups, 1):
        rows = []
        for decision in EXPECTED_DECISIONS[group["scenario_id"]]:
            path = group["paths"].get(decision)
            if path is None:
                continue
            cached = cache.get(str(path.resolve()))
            if cached is None:
                continue
            rows.append(copy.deepcopy(cached))
        preference = finalize_reward_rows(rows)
        group_index.append(
            {
                "group_id": group["group_id"],
                "scenario_id": group["scenario_id"],
                "route_id": group["route_id"],
                "dataset_variant": group["variant"],
                "active_for_training": group["active_for_training"],
                "policy_alignment": group["policy_alignment"],
                **preference,
                "branch_rewards": {
                    row["decision"]: row.get("final_reward") for row in rows
                },
                "branch_target_probabilities": {
                    row["decision"]: round(float(row.get("target_probability", 0.0)), 8)
                    for row in rows
                },
            }
        )
        rows_by_decision = {row["decision"]: row for row in rows}
        for decision in group["annotate_decisions"]:
            row = rows_by_decision.get(decision)
            path = group["paths"].get(decision)
            if row is None or path is None:
                continue
            active = bool(group["active_for_training"])
            label = decision_label(decision)
            reward_payload = {
                "schema_version": ANNOTATION_SCHEMA_VERSION,
                "reward_version": REWARD_VERSION,
                "computed_at_utc": generated_at,
                "scenario_id": group["scenario_id"],
                "route_id": group["route_id"],
                "group_id": group["group_id"],
                "dataset_variant": group["variant"],
                "policy_alignment": group["policy_alignment"],
                "run_dir": str(path.resolve()),
                "active_for_training": active,
                "superseded": not active,
                "superseded_reason": (
                    "replaced_by_S1_PED_SPEED_V2" if not active else None
                ),
                "quality_valid": bool(row.get("quality_accepted") and row.get("reward_valid")),
                "quality_exclusion_reason": row.get("exclusion_reason"),
                "action_feasible": bool(row.get("action_feasible")),
                "candidate_for_decision_ranking": bool(
                    active and row.get("candidate_for_decision_ranking")
                ),
                "group_label_valid": bool(active and preference["label_valid"]),
                "group_label_reason": preference["label_reason"],
                "hard_safety_tier": row.get("hard_safety_tier"),
                "hard_safety_tier_rank": row.get("hard_safety_tier_rank"),
                "task_score_before_hazard_penalty": row.get(
                    "task_score_before_hazard_penalty"
                ),
                "hazard_penalty": round(
                    float((row.get("hazard") or {}).get("penalty") or 0.0), 6
                ),
                "safety_cap": row.get("safety_cap"),
                "safety_cap_reasons": row.get("safety_cap_reasons") or [],
                "final_reward": row.get("final_reward"),
                "target_probability": (
                    round(float(row.get("target_probability", 0.0)), 8) if active else 0.0
                ),
                "top_decision": bool(active and row.get("top_decision")),
                "longitudinal_target_distribution": (
                    preference["longitudinal_target_distribution"]
                    if active else {key: 0.0 for key in LONGITUDINAL_CLASSES}
                ),
                "lateral_target_distribution": (
                    preference["lateral_target_distribution"]
                    if active else {key: 0.0 for key in LATERAL_CLASSES}
                ),
                "components": row.get("components") or {},
                "hazard": row.get("hazard") or {},
                "collision_counts": row.get("collision_counts") or {},
                "outcome": row.get("outcome"),
                "route_completion": row.get("route_completion"),
                "source_quality_sha256": row.get("source_quality_sha256"),
                "compute_error": row.get("static_compute_error"),
                "thresholds": {
                    "safe_clearance_m": SAFE_CLEARANCE_M,
                    "ttc_horizon_s": TTC_HORIZON_S,
                    "pet_safe_s": PET_SAFE_S,
                    "max_vertical_separation_m": MAX_VERTICAL_SEPARATION_M,
                    "hazard_penalty_weight": HAZARD_PENALTY_WEIGHT,
                    "soft_target_window": SOFT_TARGET_WINDOW,
                    "top_decision_window": TOP_DECISION_WINDOW,
                    "softmax_temperature": SOFTMAX_TEMPERATURE,
                },
            }
            annotations[str(path.resolve())] = {
                "decision_label": label,
                "decision_reward": reward_payload,
            }
        if group_number % 200 == 0 or group_number == len(groups):
            print(f"[group] {group_number:4d}/{len(groups):4d}", flush=True)

    branch_rows = [compact_branch_record(value) for _, value in sorted(annotations.items())]
    summary = {
        "schema_version": "decision_reward_v2_migration_summary",
        "reward_version": REWARD_VERSION,
        "generated_at_utc": generated_at,
        "mode": "write" if args.write else "dry_run",
        "data_root": str(data_root),
        "canonical_groups_scored": len(groups),
        "physical_branches_computed": len(cache),
        "branches_annotated": len(annotations),
        "active_branches": sum(row["active_for_training"] for row in branch_rows),
        "superseded_branches": sum(not row["active_for_training"] for row in branch_rows),
        "quality_valid_branches": sum(row["quality_valid"] for row in branch_rows),
        "candidate_branches": sum(row["candidate_for_decision_ranking"] for row in branch_rows),
        "groups_with_valid_labels": sum(
            row["active_for_training"] and row["label_valid"] for row in group_index
        ),
        "compute_errors": sum(issue["kind"] == "reward_compute_error" for issue in issues),
        "accepted_branch_compute_errors": sum(
            issue["kind"] == "reward_compute_error" and issue["severity"] == "error"
            for issue in issues
        ),
        "metadata_warnings": sum(issue["kind"] == "missing_run_spec" for issue in issues),
        "hard_safety_tier_counts": dict(
            sorted(
                (
                    tier,
                    sum(row["hard_safety_tier"] == tier for row in branch_rows),
                )
                for tier in {row["hard_safety_tier"] for row in branch_rows}
            )
        ),
    }

    if args.write:
        print(f"[write] annotating {len(annotations)} canonical branches", flush=True)
        for index, (raw_path, annotation) in enumerate(sorted(annotations.items()), 1):
            path = pathlib.Path(raw_path)
            quality_path = path / "quality_report.json"
            quality = read_json(quality_path)
            label = annotation["decision_label"]
            reward = annotation["decision_reward"]
            quality.update(
                {
                    "decision_label": label,
                    "decision_reward": reward,
                    "decision_longitudinal": label["longitudinal"],
                    "decision_lateral": label["lateral"],
                    "decision_reward_score": reward["final_reward"],
                    "decision_reward_version": REWARD_VERSION,
                    "decision_target_probability": reward["target_probability"],
                    "decision_top_label": reward["top_decision"],
                    "decision_training_eligible": reward[
                        "candidate_for_decision_ranking"
                    ],
                }
            )
            atomic_write_json(quality_path, quality)

            run_spec_path = path / "metadata/run_spec.json"
            run_spec = read_json(run_spec_path)
            if run_spec:
                run_spec.update(
                    {
                        "decision_label": label,
                        "decision_longitudinal": label["longitudinal"],
                        "decision_lateral": label["lateral"],
                    }
                )
                atomic_write_json(run_spec_path, run_spec)
            else:
                issues.append(
                    {
                        "run_dir": raw_path,
                        "error": "missing metadata/run_spec.json",
                        "kind": "missing_run_spec",
                        "severity": "warning",
                    }
                )
            if index == 1 or index % 250 == 0 or index == len(annotations):
                print(f"[write] {index:5d}/{len(annotations):5d}", flush=True)

        output_root = data_root / "_annotations/decision_reward_v2"
        summary["compute_errors"] = sum(
            issue["kind"] == "reward_compute_error" for issue in issues
        )
        summary["accepted_branch_compute_errors"] = sum(
            issue["kind"] == "reward_compute_error" and issue["severity"] == "error"
            for issue in issues
        )
        summary["metadata_warnings"] = sum(
            issue["kind"] == "missing_run_spec" for issue in issues
        )
        atomic_write_json(output_root / "summary.json", summary)
        atomic_write_jsonl(output_root / "branch_index.jsonl", branch_rows)
        atomic_write_jsonl(output_root / "group_index.jsonl", group_index)
        atomic_write_jsonl(output_root / "errors.jsonl", issues)

    samples = sorted(
        branch_rows,
        key=lambda row: (
            row["scenario_id"],
            row["route_id"],
            row["branch_decision"],
            not row["active_for_training"],
        ),
    )[: max(args.sample_limit, 0)]
    print(json.dumps({"summary": summary, "samples": samples}, ensure_ascii=False, indent=2), flush=True)
    fatal_issues = [issue for issue in issues if issue["severity"] == "error"]
    return 0 if not fatal_issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
