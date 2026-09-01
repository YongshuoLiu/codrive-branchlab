#!/usr/bin/env python3
"""Validate one counterfactual branch and compute reward-ready raw metrics."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
from pathlib import Path

from counterfactual_contract import validate_run_contract


DECISIONS = {
    "Accelerate",
    "Maintain",
    "Brake",
    "Stop",
    "LaneChangeLeft",
    "LaneChangeRight",
}


def read_json(path: Path, default=None):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def read_jsonl_gz(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except (gzip.BadGzipFile, EOFError, OSError):
        return rows
    return rows


def result_record(path: Path) -> dict | None:
    payload = read_json(path, {}) or {}
    records = payload.get("_checkpoint", {}).get("records", [])
    return records[-1] if records else None


def collision_summary(record: dict | None) -> tuple[dict, int]:
    infractions = (record or {}).get("infractions", {})
    counts = {}
    for key, value in infractions.items():
        if key.startswith("collisions_"):
            counts[key] = len(value) if isinstance(value, list) else int(bool(value))
    return counts, sum(counts.values())


def modality_counts(run_dir: Path) -> dict:
    counts = {}
    data_dirs = [path for path in (run_dir / "data").rglob("*") if path.is_dir()]
    for name in (
        "measurements",
        "rgb",
        "rgb_augmented",
        "semantics",
        "semantics_augmented",
        "depth",
        "depth_augmented",
        "bev_semantics",
        "bev_semantics_augmented",
        "lidar",
        "boxes",
    ):
        directories = [path for path in data_dirs if path.name == name]
        counts[name] = sum(sum(1 for child in path.iterdir() if child.is_file()) for path in directories)
    return counts


def actor_pair_metrics(row: dict) -> tuple[float | None, float | None]:
    ego = row.get("ego", {})
    ego_loc = ego.get("transform", {}).get("location", {})
    ego_vel = ego.get("velocity", {})
    ego_extent = (2.3, 1.0)
    minimum_clearance = None
    minimum_ttc = None
    for actor in row.get("actors", []):
        if actor.get("id") == ego.get("id"):
            continue
        location = actor.get("transform", {}).get("location", {})
        velocity = actor.get("velocity", {})
        try:
            dx = float(location["x"]) - float(ego_loc["x"])
            dy = float(location["y"]) - float(ego_loc["y"])
            rvx = float(velocity["x"]) - float(ego_vel["x"])
            rvy = float(velocity["y"]) - float(ego_vel["y"])
        except (KeyError, TypeError, ValueError):
            continue
        extent = actor.get("bounding_box", {}).get("extent", {})
        other_radius = math.hypot(float(extent.get("x", 0.5)), float(extent.get("y", 0.5)))
        clearance = math.hypot(dx, dy) - math.hypot(*ego_extent) - other_radius
        minimum_clearance = clearance if minimum_clearance is None else min(minimum_clearance, clearance)
        rel_norm = rvx * rvx + rvy * rvy
        if rel_norm > 1e-4:
            ttc = -(dx * rvx + dy * rvy) / rel_norm
            if 0.0 <= ttc <= 10.0:
                closest = math.hypot(dx + ttc * rvx, dy + ttc * rvy)
                if closest <= math.hypot(*ego_extent) + other_radius + 1.0:
                    minimum_ttc = ttc if minimum_ttc is None else min(minimum_ttc, ttc)
    return minimum_clearance, minimum_ttc


def consecutive_duration(rows: list[dict], predicate) -> float:
    longest = 0.0
    start = None
    previous = None
    for row in rows:
        now = float(row.get("simulation_time_s", 0.0))
        if predicate(row):
            if start is None or (previous is not None and now - previous > 0.25):
                start = now
            longest = max(longest, now - start)
        else:
            start = None
        previous = now
    return longest


def validate(run_dir: Path, expected_job: dict | None = None) -> dict:
    spec = read_json(run_dir / "metadata/run_spec.json", {}) or {}
    decision = spec.get("decision")
    termination = read_json(run_dir / "metadata/termination.json")
    runner = read_json(run_dir / "metadata/runner_status.json", {}) or {}
    events = read_jsonl(run_dir / "raw/decision_events.jsonl")
    telemetry = read_jsonl_gz(run_dir / "raw/carla_telemetry.jsonl.gz")
    record = result_record(run_dir / "results.json")
    modality = modality_counts(run_dir)
    contract_errors = validate_run_contract(run_dir, expected_job)
    errors = list(contract_errors)
    warnings = []
    termination_reason = termination.get("reason") if termination else None
    lane_unavailable = termination_reason == "lane_unavailable"

    if decision not in DECISIONS:
        errors.append(f"missing or invalid decision: {decision}")
    event_names = [row.get("event") for row in events]
    if event_names.count("decision_started") != 1:
        errors.append(f"decision_started occurs {event_names.count('decision_started')} times")
    if not telemetry:
        errors.append("missing readable CARLA telemetry")
    recorder_path = run_dir / "raw/carla_recorder.log"
    town = (spec.get("route") or {}).get("town")
    recorder_disabled = "carla_recorder_disabled" in event_names
    if not recorder_path.is_file():
        if recorder_disabled:
            warnings.append(
                f"CARLA recorder intentionally disabled on {town} because start_recorder "
                "can reproducibly segfault this collection path; per-tick telemetry remains available"
            )
        else:
            errors.append("missing CARLA recorder")
    # Frozen CoDrive routes are intentionally short.  A fast but valid branch
    # can complete with 15--19 saved sensor frames at the standard 4 Hz save
    # cadence, while raw CARLA telemetry remains available at every policy tick.
    # A rejected lane change must stop immediately after its geometry/dynamic
    # safety audit. At the 4 Hz save cadence that can legitimately leave only
    # 5 sensor frames, while 20 Hz raw telemetry still contains the audit.
    minimum_frames = 5 if lane_unavailable else 10 if termination_reason else 15
    if modality.get("measurements", 0) < minimum_frames:
        errors.append(f"only {modality.get('measurements', 0)} measurement frames")
    nonzero_modalities = [value for value in modality.values() if value > 0]
    if nonzero_modalities and max(nonzero_modalities) - min(nonzero_modalities) > 2:
        errors.append(f"sensor modality counts are not synchronized: {modality}")

    collision_counts, evaluator_collision_total = collision_summary(record)
    collision_event = "collision_detected" in event_names
    collision = collision_event or evaluator_collision_total > 0 or termination_reason == "collision"
    deadlock = termination_reason in {
        "vehicle_deadlock",
        "post_clear_deadlock",
        "scenario_deadlock",
    }
    if collision and termination_reason != "collision":
        errors.append("collision was observed but did not trigger immediate agent termination")
    # Collision, deadlock, and lane-unavailable are valid terminal episode
    # outcomes when the agent wrote an explicit termination marker and stopped.
    # They are not equivalent to a completed safe trajectory, but must remain
    # usable labels rather than being retried into a different world state.

    active = [row for row in telemetry if row.get("decision_active")]
    speeds = [float(row.get("ego", {}).get("speed_mps", 0.0)) for row in active]
    target_speeds = [
        float(row["decision_target_speed_mps"])
        for row in active
        if row.get("decision_target_speed_mps") is not None
    ]
    accelerations = []
    jerks = []
    previous_acceleration = None
    previous_time = None
    minimum_clearance = None
    minimum_ttc = None
    for row in telemetry:
        transform = row.get("ego", {}).get("transform", {})
        yaw = math.radians(float(transform.get("rotation", {}).get("yaw", 0.0)))
        acceleration = row.get("ego", {}).get("acceleration", {})
        lon_acceleration = (
            float(acceleration.get("x", 0.0)) * math.cos(yaw)
            + float(acceleration.get("y", 0.0)) * math.sin(yaw)
        )
        accelerations.append(lon_acceleration)
        now = float(row.get("simulation_time_s", 0.0))
        if previous_acceleration is not None and previous_time is not None and now > previous_time:
            jerks.append((lon_acceleration - previous_acceleration) / (now - previous_time))
        previous_acceleration = lon_acceleration
        previous_time = now
        clearance, ttc = actor_pair_metrics(row)
        if clearance is not None:
            minimum_clearance = clearance if minimum_clearance is None else min(minimum_clearance, clearance)
        if ttc is not None:
            minimum_ttc = ttc if minimum_ttc is None else min(minimum_ttc, ttc)

    decision_fidelity = {}
    if active:
        entry_speed = float(active[0].get("decision_entry_speed_mps") or speeds[0])
        final_window = speeds[-min(len(speeds), 20):]
        if decision == "Accelerate":
            gain = max(speeds) - entry_speed
            running_minimum = speeds[0]
            recovery_gain = 0.0
            for speed in speeds[1:]:
                recovery_gain = max(recovery_gain, speed - running_minimum)
                running_minimum = min(running_minimum, speed)
            decision_fidelity = {
                "entry_speed_mps": entry_speed,
                "maximum_speed_gain_mps": gain,
                "maximum_reacceleration_gain_mps": recovery_gain,
            }
            # A mandatory stop sign/red light can force the trajectory below
            # its branch-entry speed before the Accelerate target is allowed.
            # In that case require an actual positive re-acceleration segment.
            if max(gain, recovery_gain) < 0.75 and not collision:
                errors.append(
                    f"Accelerate gained only {gain:.2f} m/s from entry and "
                    f"re-accelerated only {recovery_gain:.2f} m/s"
                )
        elif decision == "Maintain":
            median_deviation = statistics.median(abs(speed - entry_speed) for speed in speeds)
            decision_fidelity = {"entry_speed_mps": entry_speed, "median_speed_deviation_mps": median_deviation}
            if median_deviation > 2.0 and not collision:
                errors.append(f"Maintain median speed deviation is {median_deviation:.2f} m/s")
        elif decision == "Brake":
            reduction = entry_speed - min(speeds)
            stop_hold = consecutive_duration(active, lambda row: row["ego"]["speed_mps"] < 0.3)
            unsuppressed_stop_hold = consecutive_duration(
                active,
                lambda row: (
                    row["ego"]["speed_mps"] < 0.3
                    and row.get("decision_suppressed_by") is None
                ),
            )
            decision_fidelity = {
                "entry_speed_mps": entry_speed,
                "maximum_speed_reduction_mps": reduction,
                "full_stop_hold_s": stop_hold,
                "unsuppressed_full_stop_hold_s": unsuppressed_stop_hold,
            }
            if reduction < 1.0 and not collision:
                errors.append(f"Brake reduced speed by only {reduction:.2f} m/s")
            # A collision can pin the ego at zero even though the selected
            # longitudinal target remains Brake. That is Brake+collision, not
            # evidence that the decision silently became Stop.
            if unsuppressed_stop_hold >= 0.5 and not collision:
                errors.append(
                    f"Brake became Stop without a regulatory override for "
                    f"{unsuppressed_stop_hold:.2f} s"
                )
        elif decision == "Stop":
            stop_hold = consecutive_duration(active, lambda row: row["ego"]["speed_mps"] < 0.3)
            decision_fidelity = {"full_stop_hold_s": stop_hold}
            if stop_hold < 0.5 and not collision:
                errors.append(f"Stop held zero speed for only {stop_hold:.2f} s")
        elif decision in {"LaneChangeLeft", "LaneChangeRight"}:
            audit = active[0].get("lane_audit") or {}
            entry_pose = next(
                (row.get("entry_pose") for row in events if row.get("event") == "decision_started"),
                {},
            )
            entry_lane = entry_pose.get("lane_id")
            target_lane = audit.get("target_lane_id")
            reached_rows = [
                row
                for row in active
                if row.get("ego", {}).get("lane", {}).get("lane_id") == target_lane
            ]
            reached_hold = consecutive_duration(
                reached_rows,
                lambda row: row.get("ego", {}).get("lane", {}).get("lane_id") == target_lane,
            )
            decision_fidelity = {
                "lane_audit_safe": audit.get("safe"),
                "entry_lane_id": entry_lane,
                "target_lane_id": target_lane,
                "target_lane_reached_frames": len(reached_rows),
                "target_lane_hold_s": reached_hold,
                "same_direction_ratio": audit.get("same_direction_ratio"),
                "lane_marking_violation_ratio": audit.get("lane_marking_violation_ratio"),
            }
            if not audit.get("safe"):
                errors.append("lane audit was not safe")
            if not reached_rows and not collision:
                errors.append("ego never reached the requested adjacent lane")
            elif reached_hold < 0.5 and not collision:
                errors.append(f"ego held the target lane for only {reached_hold:.2f} s")
    elif lane_unavailable:
        audit = (termination or {}).get("details", {}).get("audit", {})
        decision_fidelity = {
            "lane_audit_safe": audit.get("safe"),
            "lane_audit_reason": audit.get("reason"),
            "requested_side": audit.get("side"),
            "lane_geometry_coverage": audit.get("coverage"),
            "actor_conflict_count": len(audit.get("actor_conflicts", [])),
        }
        if audit.get("safe") is not False:
            errors.append("lane-unavailable termination is missing a failed safety audit")
    else:
        errors.append("no decision-active telemetry frames")

    status = (record or {}).get("status")
    route_completion = float((record or {}).get("scores", {}).get("score_route", 0.0))
    if not collision and not deadlock and not lane_unavailable:
        if status not in {"Completed", "Perfect"}:
            errors.append(f"route status is {status or 'missing'}")
        if route_completion < 99.0:
            errors.append(f"route completion is {route_completion:.2f}")
    if runner.get("runner_exit_status", 0) != 0 and not termination:
        errors.append(f"runner exited {runner.get('runner_exit_status')} without a termination marker")

    report = {
        "schema_version": "counterfactual_branch_quality_v1",
        "run_dir": str(run_dir),
        "route_id": spec.get("route", {}).get("route_id"),
        "scenario_type": spec.get("route", {}).get("scenario_type"),
        "decision": decision,
        "contract_verified": not contract_errors,
        "contract_errors": contract_errors,
        "accepted": not errors,
        "outcome": (
            "collision" if collision else "deadlock" if deadlock else "lane_unavailable" if lane_unavailable else "completed"
        ),
        "errors": errors,
        "warnings": warnings,
        "termination": termination,
        "runner_exit_status": runner.get("runner_exit_status"),
        "route_status": status,
        "route_completion": route_completion,
        "collision": collision,
        "collision_counts": collision_counts,
        "deadlock": deadlock,
        "lane_unavailable": lane_unavailable,
        "telemetry_frames": len(telemetry),
        "active_frames": len(active),
        "modality_counts": modality,
        "decision_fidelity": decision_fidelity,
        "raw_metrics": {
            "minimum_actor_clearance_m": minimum_clearance,
            "minimum_predicted_ttc_s": minimum_ttc,
            "minimum_speed_mps": min(speeds) if speeds else None,
            "maximum_speed_mps": max(speeds) if speeds else None,
            "mean_target_speed_mps": statistics.fmean(target_speeds) if target_speeds else None,
            "maximum_abs_longitudinal_acceleration_mps2": max(map(abs, accelerations), default=None),
            "maximum_abs_jerk_mps3": max(map(abs, jerks), default=None),
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--job-id")
    args = parser.parse_args()
    if bool(args.manifest) != bool(args.job_id):
        parser.error("--manifest and --job-id must be supplied together")
    expected_job = None
    if args.manifest:
        rows = [
            json.loads(line)
            for line in args.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        matches = [row for row in rows if row.get("job_id") == args.job_id]
        if len(matches) != 1:
            parser.error(f"expected exactly one manifest row for {args.job_id}, found {len(matches)}")
        expected_job = matches[0]
    run_dir = args.run_dir.resolve()
    report = validate(run_dir, expected_job)
    output = args.output or run_dir / "quality_report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
