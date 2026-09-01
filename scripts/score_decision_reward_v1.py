#!/usr/bin/env python3
"""Prototype the constrained decision reward on reproducibly sampled routes."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


RL_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DECISIONS = {
    "S1": ("Accelerate", "Maintain", "Brake", "Stop", "LaneChangeLeft", "LaneChangeRight"),
    "S2": ("Accelerate", "Maintain", "Brake", "Stop", "LaneChangeLeft", "LaneChangeRight"),
    "S3": ("Accelerate", "Maintain", "Brake", "Stop", "LaneChangeLeft", "LaneChangeRight"),
    "S4": ("Accelerate", "Maintain", "Brake", "Stop"),
    "S5": ("Accelerate", "Maintain", "Brake", "Stop"),
    "S6": ("Accelerate", "Maintain", "Brake", "Stop"),
}


def clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(max(float(value), low), high)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = clip(quantile) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    ratio = position - lower
    return ordered[lower] * (1.0 - ratio) + ordered[upper] * ratio


def vector(node: dict[str, Any] | None) -> tuple[float, float, float]:
    node = node or {}
    return float(node.get("x", 0.0)), float(node.get("y", 0.0)), float(node.get("z", 0.0))


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}


def read_telemetry(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                try:
                    rows.append(json.loads(line))
                except (ValueError, TypeError):
                    continue
    except OSError:
        pass
    return rows


def result_record(run_dir: Path) -> dict[str, Any]:
    payload = read_json(run_dir / "results.json")
    records = (payload.get("_checkpoint") or {}).get("records") or []
    return records[0] if records else {}


def evaluation_start(frames: list[dict[str, Any]]) -> int | None:
    for index, frame in enumerate(frames):
        if frame.get("decision_started") is True:
            return index
    for index, frame in enumerate(frames):
        if (frame.get("event_state") or {}).get("hint_reached") is True:
            return index
    return None


def hazard_clear_index(frames: list[dict[str, Any]], start: int) -> int | None:
    for index in range(start, len(frames)):
        if (frames[index].get("event_state") or {}).get("hazard_cleared") is True:
            return index
    return None


def actor_radius(actor: dict[str, Any], default: float) -> float:
    extent = ((actor.get("bounding_box") or {}).get("extent") or {})
    try:
        return math.hypot(float(extent.get("x", 0.0)), float(extent.get("y", 0.0)))
    except (TypeError, ValueError):
        return default


def frame_risk(frame: dict[str, Any]) -> float:
    ego = frame.get("ego") or {}
    ego_location = vector((ego.get("transform") or {}).get("location"))
    ego_velocity = vector(ego.get("velocity"))
    hero = next(
        (
            actor
            for actor in frame.get("actors", [])
            if actor.get("role_name") == "hero"
        ),
        {},
    )
    ego_radius = actor_radius(hero, 2.6)
    maximum = 0.0
    for actor in frame.get("actors", []):
        if actor.get("role_name") != "scenario":
            continue
        type_id = str(actor.get("type_id", ""))
        location = vector((actor.get("transform") or {}).get("location"))
        velocity = vector(actor.get("velocity"))
        relative = tuple(location[axis] - ego_location[axis] for axis in range(3))
        center_distance = math.sqrt(sum(value * value for value in relative))
        if center_distance <= 1e-6 or center_distance > 80.0:
            continue
        clearance = center_distance - ego_radius - actor_radius(actor, 0.5)
        if type_id.startswith("walker."):
            safe_clearance = 3.0
        elif type_id.startswith("vehicle."):
            safe_clearance = 4.0
        else:
            safe_clearance = 1.5
        clearance_risk = clip((safe_clearance - clearance) / safe_clearance)
        relative_velocity = tuple(velocity[axis] - ego_velocity[axis] for axis in range(3))
        closing_speed = -sum(relative[axis] * relative_velocity[axis] for axis in range(3)) / center_distance
        ttc_risk = 0.0
        if closing_speed > 0.5:
            ttc = max(clearance, 0.0) / closing_speed
            ttc_risk = clip((4.0 - ttc) / 3.5)
        maximum = max(maximum, 0.65 * ttc_risk + 0.35 * clearance_risk)
    return maximum


def rolling_median(values: list[float], radius: int = 2) -> list[float]:
    return [
        statistics.median(values[max(0, index - radius) : index + radius + 1])
        for index in range(len(values))
    ]


def comfort_score(frames: list[dict[str, Any]], collision: bool) -> tuple[float, dict[str, float]]:
    if len(frames) < 12:
        return 0.0, {"a_long_p95": 0.0, "a_lat_p95": 0.0, "jerk_p95": 0.0}
    frames = frames[5 : -5 if collision and len(frames) > 15 else None]
    along = []
    lateral = []
    times = []
    for frame in frames:
        ego = frame.get("ego") or {}
        acceleration = vector(ego.get("acceleration"))
        velocity = vector(ego.get("velocity"))
        speed_xy = math.hypot(velocity[0], velocity[1])
        if speed_xy > 0.5:
            heading = (velocity[0] / speed_xy, velocity[1] / speed_xy)
        else:
            yaw = math.radians(float(((ego.get("transform") or {}).get("rotation") or {}).get("yaw", 0.0)))
            heading = (math.cos(yaw), math.sin(yaw))
        along.append(acceleration[0] * heading[0] + acceleration[1] * heading[1])
        lateral.append(-acceleration[0] * heading[1] + acceleration[1] * heading[0])
        times.append(float(frame.get("simulation_time_s", len(times) * 0.05)))
    along = rolling_median(along)
    lateral = rolling_median(lateral)
    jerk = []
    for index in range(1, len(along)):
        delta = max(times[index] - times[index - 1], 0.02)
        jerk.append(abs(along[index] - along[index - 1]) / delta)
    a_long_p95 = percentile([abs(value) for value in along], 0.95)
    a_lat_p95 = percentile([abs(value) for value in lateral], 0.95)
    jerk_p95 = percentile(jerk, 0.95)
    penalty = (
        0.45 * clip((a_long_p95 - 2.5) / 3.5)
        + 0.25 * clip((a_lat_p95 - 2.0) / 3.0)
        + 0.30 * clip((jerk_p95 - 4.0) / 8.0)
    )
    return 1.0 - penalty, {
        "a_long_p95": a_long_p95,
        "a_lat_p95": a_lat_p95,
        "jerk_p95": jerk_p95,
    }


def fidelity_score(decision: str, quality: dict[str, Any]) -> float:
    metrics = quality.get("decision_fidelity") or {}
    if decision == "Accelerate":
        return clip(float(metrics.get("maximum_speed_gain_mps", 0.0)) / 3.0)
    if decision == "Maintain":
        return 1.0 - clip(float(metrics.get("median_speed_deviation_mps", 3.0)) / 3.0)
    if decision == "Brake":
        entry = max(float(metrics.get("entry_speed_mps", 0.0)), 1.0)
        expected = max(entry * 0.55, 1.0)
        return clip(float(metrics.get("maximum_speed_reduction_mps", 0.0)) / expected)
    if decision == "Stop":
        return clip(float(metrics.get("full_stop_hold_s", 0.0)) / 0.5)
    if decision.startswith("LaneChange"):
        if metrics.get("lane_audit_safe") is not True:
            return 0.0
        return clip(float(metrics.get("target_lane_hold_s", 0.0)) / 0.5)
    return 0.0


def legality_score(
    record: dict[str, Any], scenario_id: str, decision: str
) -> tuple[float, bool]:
    infractions = record.get("infractions") or {}
    # A commanded lane change is allowed to cross lane markings and to leave the
    # evaluator's original route-lane envelope.  Real route deviation remains a
    # hard failure; traffic-control infractions only reduce the soft legality term.
    outside_route = bool(infractions.get("outside_route_lanes")) and not decision.startswith("LaneChange")
    route_deviation = bool(infractions.get("route_dev"))
    soft_count = int(outside_route) + int(route_deviation) + int(bool(infractions.get("stop_infraction")))
    if scenario_id != "S6":
        soft_count += int(bool(infractions.get("red_light")))
    severe_route_violation = outside_route or route_deviation
    return 1.0 - clip(soft_count / 2.0), severe_route_violation


def base_branch(scenario_id: str, decision: str, run_dir: Path) -> dict[str, Any]:
    quality = read_json(run_dir / "quality_report.json")
    record = result_record(run_dir)
    frames = read_telemetry(run_dir / "raw/carla_telemetry.jsonl.gz")
    official = (record.get("scores") or {}).get("score_composed")
    row: dict[str, Any] = {
        "scenario_id": scenario_id,
        "route_id": quality.get("route_id", run_dir.parent.name),
        "decision": decision,
        "run_dir": str(run_dir.resolve()),
        "outcome": quality.get("outcome", "missing"),
        "quality_accepted": quality.get("accepted") is True,
        "official_score": float(official) if isinstance(official, (int, float)) else None,
        "route_completion": clip(float(quality.get("route_completion") or 0.0) / 100.0),
        "collision": quality.get("collision") is True,
        "deadlock": quality.get("deadlock") is True,
        "lane_unavailable": quality.get("lane_unavailable") is True,
        "collision_counts": quality.get("collision_counts") or {},
    }
    if not quality or quality.get("accepted") is not True:
        row.update({"reward_valid": False, "exclusion_reason": "quality_not_accepted"})
        return row
    if row["lane_unavailable"]:
        row.update(
            {
                "reward_valid": True,
                "decision_feasible": False,
                "safety_tier": "lane_unavailable",
                "safety_tier_rank": 4,
                "decision_reward": -70.0,
                "components": {},
            }
        )
        return row
    start = evaluation_start(frames)
    if start is None or not record:
        row.update({"reward_valid": False, "exclusion_reason": "missing_evaluable_rollout"})
        return row
    clear_index = hazard_clear_index(frames, start)
    end_index = len(frames)
    if clear_index is not None:
        clear_time = float(frames[clear_index].get("simulation_time_s", 0.0))
        for index in range(clear_index, len(frames)):
            if float(frames[index].get("simulation_time_s", clear_time)) > clear_time + 3.0:
                end_index = index + 1
                break
    window = frames[start:end_index]
    risks = [frame_risk(frame) for frame in window]
    risk_margin = 1.0 - clip(0.7 * percentile(risks, 0.90) + 0.3 * (statistics.fmean(risks) if risks else 0.0))
    comfort, comfort_metrics = comfort_score(window, row["collision"])
    recovery = 0.0
    if clear_index is not None:
        clear_time = float(frames[clear_index].get("simulation_time_s", 0.0))
        recovery += 0.4
        resume_delay = None
        for frame in frames[clear_index:]:
            ego = frame.get("ego") or {}
            speed_limit = max(float(ego.get("speed_limit_kmh", 30.0)) / 3.6, 1.0)
            if float(ego.get("speed_mps", 0.0)) >= 0.6 * speed_limit:
                resume_delay = max(float(frame.get("simulation_time_s", clear_time)) - clear_time, 0.0)
                break
        if resume_delay is not None:
            recovery += 0.35 * clip(1.0 - resume_delay / 3.0)
            recovery += 0.25 if not row["deadlock"] else 0.0
    elif row["route_completion"] >= 0.995 and not row["collision"] and not row["deadlock"]:
        # Some scenario implementations end the route without emitting an
        # explicit hazard_cleared edge.  Successful route completion is the
        # conservative fallback evidence that the hazard was resolved.
        final_ego = (frames[-1].get("ego") or {}) if frames else {}
        final_speed = float(final_ego.get("speed_mps", 0.0))
        final_limit = max(float(final_ego.get("speed_limit_kmh", 30.0)) / 3.6, 1.0)
        recovery = 0.75 + 0.25 * clip(final_speed / (0.6 * final_limit))
    legality, offroad = legality_score(record, scenario_id, decision)
    fidelity = fidelity_score(decision, quality)
    duration = float((record.get("meta") or {}).get("duration_game") or 0.0)
    speeds = [float((frame.get("ego") or {}).get("speed_mps", 0.0)) for frame in window]
    entry_speed = max(speeds[0] if speeds else 0.0, 1.0)
    speed_loss = statistics.fmean(clip((entry_speed - speed) / entry_speed) for speed in speeds) if speeds else 1.0
    intervention = clip(0.7 * speed_loss + 0.3 * float(decision.startswith("LaneChange")))
    row.update(
        {
            "reward_valid": True,
            "decision_feasible": fidelity > 0.0,
            "duration_game_s": duration,
            "progress_rate": row["route_completion"] / max(duration, 0.1),
            "risk": 1.0 - risk_margin,
            "intervention": intervention,
            "offroad_or_route_violation": offroad,
            "components": {
                "risk_margin": risk_margin,
                "recovery": recovery,
                "comfort": comfort,
                "decision_fidelity": fidelity,
                "legality": legality,
            },
            "comfort_metrics": comfort_metrics,
        }
    )
    return row


def finalize_group(rows: list[dict[str, Any]]) -> None:
    evaluable = [row for row in rows if row.get("reward_valid") and not row.get("lane_unavailable")]
    safe = [
        row
        for row in evaluable
        if not row.get("collision")
        and not row.get("deadlock")
        and not row.get("offroad_or_route_violation")
    ]
    best_completion = max((row["route_completion"] for row in evaluable), default=1.0)
    best_rate = max((row["progress_rate"] for row in safe), default=0.0)
    complete_safe = [row for row in safe if row["route_completion"] >= 0.995]
    best_time = min((row["duration_game_s"] for row in complete_safe if row["duration_game_s"] > 0), default=0.0)
    maintain = next((row for row in evaluable if row["decision"] == "Maintain"), None)
    maintain_risk = float(maintain.get("risk", 0.0)) if maintain else 0.0

    for row in evaluable:
        relative_progress = clip(row["route_completion"] / max(best_completion, 1e-6))
        progress = 0.7 * row["route_completion"] + 0.3 * relative_progress
        rate_score = clip(row["progress_rate"] / max(best_rate, 1e-6)) if best_rate > 0 else 0.0
        if row["route_completion"] >= 0.995 and best_time > 0:
            time_score = clip(best_time / max(row["duration_game_s"], 0.1))
        else:
            time_score = row["route_completion"]
        efficiency = 0.6 * rate_score + 0.4 * time_score
        risk_benefit = max(0.0, maintain_risk - row["risk"])
        unnecessary = row["intervention"] * math.exp(-risk_benefit / 0.15)
        row["components"].update(
            {
                "progress": progress,
                "efficiency": efficiency,
                "unnecessary_intervention": unnecessary,
            }
        )
        component = row["components"]
        utility = 100.0 * (
            0.32 * component["risk_margin"]
            + 0.22 * component["progress"]
            + 0.16 * component["recovery"]
            + 0.12 * component["efficiency"]
            + 0.10 * component["comfort"]
            + 0.05 * component["decision_fidelity"]
            + 0.03 * component["legality"]
        ) - 15.0 * unnecessary
        utility = clip(utility, 0.0, 100.0)
        counts = row.get("collision_counts") or {}
        if row.get("collision"):
            if int(counts.get("collisions_pedestrian", 0)) > 0:
                base, tier, rank = -100.0, "pedestrian_collision", 0
            elif int(counts.get("collisions_vehicle", 0)) > 0:
                base, tier, rank = -95.0, "vehicle_collision", 1
            else:
                base, tier, rank = -90.0, "layout_collision", 2
            reward = base + 4.0 * utility / 100.0
        elif row.get("offroad_or_route_violation"):
            tier, rank = "route_violation", 3
            reward = -85.0 + 9.0 * utility / 100.0
        elif row.get("deadlock"):
            tier, rank = "deadlock", 5
            reward = -50.0 + 10.0 * utility / 100.0
        else:
            tier, rank = "safe", 6
            reward = utility
        row.update(
            {
                "continuous_utility": round(utility, 4),
                "decision_reward": round(reward, 4),
                "safety_tier": tier,
                "safety_tier_rank": rank,
            }
        )

    valid = [
        row
        for row in rows
        if row.get("reward_valid") and row.get("decision_feasible") is not False
    ]
    if not valid:
        return
    highest_tier = max(int(row.get("safety_tier_rank", -1)) for row in valid)
    target_rows = [row for row in valid if int(row.get("safety_tier_rank", -1)) == highest_tier]
    best_reward = max(float(row["decision_reward"]) for row in target_rows)
    target_rows = [row for row in target_rows if float(row["decision_reward"]) >= best_reward - 10.0]
    weights = [math.exp((float(row["decision_reward"]) - best_reward) / 5.0) for row in target_rows]
    denominator = sum(weights)
    for row in rows:
        row["target_probability"] = 0.0
        row["top_decision"] = False
    for row, weight in zip(target_rows, weights):
        row["target_probability"] = round(weight / denominator, 6)
        row["top_decision"] = float(row["decision_reward"]) >= best_reward - 5.0


def complete_route_groups(data_root: Path, scenario_id: str) -> dict[str, dict[str, Path]]:
    decisions = EXPECTED_DECISIONS[scenario_id]
    groups: dict[str, dict[str, Path]] = defaultdict(dict)
    for decision in decisions:
        for path in (data_root / scenario_id).glob(f"*/{decision}/quality_report.json"):
            if "_incomplete_" in str(path):
                continue
            quality = read_json(path)
            route_id = str(quality.get("route_id", path.parents[1].name))
            if quality.get("accepted") is True:
                groups[route_id][decision] = path.parent
    return {
        route_id: branches
        for route_id, branches in groups.items()
        if set(branches) == set(decisions)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=RL_ROOT / "data/counterfactual_decision_v1")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--routes-per-scenario", type=int, default=3)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    all_rows = []
    selected = {}
    for scene_index, scenario_id in enumerate(EXPECTED_DECISIONS):
        groups = complete_route_groups(args.data_root.resolve(), scenario_id)
        if len(groups) < args.routes_per_scenario:
            raise RuntimeError(f"{scenario_id}: only {len(groups)} complete reward-evaluable routes")
        generator = random.Random(args.seed + scene_index)
        route_ids = sorted(generator.sample(sorted(groups), args.routes_per_scenario))
        selected[scenario_id] = route_ids
        for route_id in route_ids:
            rows = [
                base_branch(scenario_id, decision, groups[route_id][decision])
                for decision in EXPECTED_DECISIONS[scenario_id]
            ]
            finalize_group(rows)
            all_rows.extend(rows)

    payload = {
        "schema_version": "decision_reward_v1_prototype",
        "seed": args.seed,
        "routes_per_scenario": args.routes_per_scenario,
        "selected_routes": selected,
        "branch_count": len(all_rows),
        "rows": all_rows,
    }
    output_json = args.output_json.resolve()
    output_csv = args.output_csv.resolve()
    for output in (output_json, output_csv):
        if RL_ROOT.resolve() not in output.parents:
            raise SystemExit(f"output must stay under {RL_ROOT.resolve()}")
        output.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = [
        "scenario_id", "route_id", "decision", "decision_reward", "safety_tier",
        "target_probability", "top_decision", "outcome", "official_score",
        "route_completion", "continuous_utility", "reward_valid", "decision_feasible",
        "risk_margin", "progress", "recovery", "efficiency", "comfort",
        "decision_fidelity", "legality", "unnecessary_intervention", "run_dir",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            flat = dict(row)
            flat.update(row.get("components") or {})
            writer.writerow({key: flat.get(key) for key in fields})
    print(json.dumps({"selected_routes": selected, "branches": len(all_rows), "json": str(output_json), "csv": str(output_csv)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
