#!/usr/bin/env python3
"""Validate completeness and cross-decision alignment of decision-v1."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from counterfactual_contract import sha256, validate_manifest_job, validate_run_contract


RL_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    rows.append(json.loads(line))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return rows


def first_active_telemetry(path: Path):
    last_row = None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                last_row = row
                if row.get("decision_active"):
                    return row
    except (FileNotFoundError, gzip.BadGzipFile, json.JSONDecodeError, OSError):
        return None
    # A lane change can be rejected by the safety audit inside the onset
    # control tick, before that tick reaches telemetry serialization. Its last
    # pre-onset row is still the closest actor snapshot and avoids treating an
    # intentional immediate rejection as a missing actor signature.
    return last_row


def distance(first: dict, second: dict) -> float:
    return math.sqrt(sum((float(first[k]) - float(second[k])) ** 2 for k in ("x", "y", "z")))


def yaw_distance(first: float, second: float) -> float:
    return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)


def scenario_actor_signature(telemetry: dict | None) -> dict[str, dict]:
    if not telemetry:
        return {}
    grouped = defaultdict(list)
    for actor in telemetry.get("actors", []):
        role = actor.get("role_name", "")
        if role != "scenario":
            continue
        location = actor.get("transform", {}).get("location")
        if location:
            grouped[str(actor.get("type_id"))].append(location)
    signature = {}
    for actor_type, locations in grouped.items():
        locations.sort(key=lambda item: (item["x"], item["y"], item["z"]))
        for index, location in enumerate(locations):
            signature[f"{actor_type}#{index}"] = location
    return signature


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=RL_ROOT / "manifests/collection_manifest.jsonl",
    )
    parser.add_argument(
        "--config", type=Path, default=RL_ROOT / "config/decision_v1.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RL_ROOT / "data/counterfactual_decision_v1/campaign_quality_report.json",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=RL_ROOT / "data/counterfactual_decision_v1/campaign_quality_table.csv",
    )
    args = parser.parse_args()

    config = read_json(args.config, {}) or {}
    thresholds = config.get("quality_thresholds", {})
    position_tolerance = float(thresholds.get("onset_position_tolerance_m", 0.5))
    yaw_tolerance = float(thresholds.get("onset_yaw_tolerance_deg", 1.0))
    speed_tolerance = float(thresholds.get("onset_speed_tolerance_mps", 0.75))
    actor_tolerance = float(thresholds.get("onset_hazard_position_tolerance_m", 0.75))

    jobs = read_jsonl(args.manifest)
    by_group = defaultdict(list)
    branch_rows = []
    campaign_errors = []
    campaign_warnings = []
    for job in jobs:
        manifest_contract_errors = validate_manifest_job(job)
        run_dir = Path(job["output_dir"])
        quality = read_json(run_dir / "quality_report.json", {}) or {}
        run_contract_errors = validate_run_contract(run_dir, job)
        contract_errors = [*manifest_contract_errors, *run_contract_errors]
        events = read_jsonl(run_dir / "raw/decision_events.jsonl")
        onset = next((row for row in events if row.get("event") == "decision_started"), None)
        active = first_active_telemetry(run_dir / "raw/carla_telemetry.jsonl.gz")
        row = {
            "job_id": job["job_id"],
            "group_id": job["group_id"],
            "scenario_id": job["scenario_id"],
            "route_id": job["route_id"],
            "town": job["town"],
            "decision": job["decision"],
            "run_dir": str(run_dir),
            "quality_present": bool(quality),
            "accepted": quality.get("accepted") is True and not contract_errors,
            "contract_verified": not contract_errors,
            "contract_errors": contract_errors,
            "outcome": quality.get("outcome", "missing"),
            "route_completion": quality.get("route_completion"),
            "telemetry_frames": quality.get("telemetry_frames"),
            "saved_frames": (quality.get("modality_counts") or {}).get("measurements"),
            "onset": onset,
            "scenario_actors": scenario_actor_signature(active),
            "decision_fidelity": quality.get("decision_fidelity", {}),
            "warnings": quality.get("warnings", []),
            "errors": [
                *(quality.get("errors", ["missing quality report"])),
                *contract_errors,
            ],
        }
        branch_rows.append(row)
        by_group[job["group_id"]].append(row)
        if not row["accepted"]:
            campaign_errors.append(f"{job['job_id']}: branch quality was not accepted")
        campaign_errors.extend(
            f"{job['job_id']}: contract: {error}" for error in contract_errors
        )

    group_reports = []
    for group_id in sorted(by_group):
        rows = sorted(by_group[group_id], key=lambda row: row["decision"])
        scene_id = rows[0]["scenario_id"]
        expected = set(config.get("scenarios", {}).get(scene_id, {}).get("decisions", []))
        observed = {row["decision"] for row in rows}
        errors = []
        warnings = []
        if observed != expected:
            errors.append(
                f"decision set mismatch: expected {sorted(expected)}, observed {sorted(observed)}"
            )
        usable = [row for row in rows if row["accepted"] and row["onset"]]
        if len(usable) != len(rows):
            errors.append(f"only {len(usable)}/{len(rows)} branches have an accepted official onset")

        onset_metrics = {
            "maximum_position_deviation_m": None,
            "maximum_yaw_deviation_deg": None,
            "maximum_speed_deviation_mps": None,
            "maximum_scenario_actor_deviation_m": None,
        }
        if usable:
            reference = next((row for row in usable if row["decision"] == "Maintain"), usable[0])
            reference_onset = reference["onset"]
            reference_pose = reference_onset.get("entry_pose", {})
            reference_location = reference_pose.get("location", {})
            reference_rotation = reference_pose.get("rotation", {})
            reference_speed = float(reference_onset.get("entry_speed_mps", 0.0))
            reference_actors = reference["scenario_actors"]
            position_deviations = []
            yaw_deviations = []
            speed_deviations = []
            actor_deviations = []
            for row in usable:
                onset = row["onset"]
                state = onset.get("event_state", {})
                if state.get("hint_reached") is not True:
                    errors.append(f"{row['decision']}: decision did not begin at hint_reached")
                pose = onset.get("entry_pose", {})
                try:
                    position_deviations.append(distance(reference_location, pose["location"]))
                    yaw_deviations.append(
                        yaw_distance(reference_rotation["yaw"], pose["rotation"]["yaw"])
                    )
                    speed_deviations.append(
                        abs(reference_speed - float(onset["entry_speed_mps"]))
                    )
                except (KeyError, TypeError, ValueError):
                    errors.append(f"{row['decision']}: incomplete onset pose/speed")
                actors = row["scenario_actors"]
                if set(actors) != set(reference_actors):
                    errors.append(f"{row['decision']}: scenario actor signature differs at onset")
                else:
                    actor_deviations.extend(
                        distance(reference_actors[key], actors[key]) for key in actors
                    )
            if position_deviations:
                onset_metrics["maximum_position_deviation_m"] = max(position_deviations)
                if max(position_deviations) > position_tolerance:
                    errors.append(
                        f"onset position deviation {max(position_deviations):.3f} m exceeds {position_tolerance:.3f} m"
                    )
            if yaw_deviations:
                onset_metrics["maximum_yaw_deviation_deg"] = max(yaw_deviations)
                if max(yaw_deviations) > yaw_tolerance:
                    errors.append(
                        f"onset yaw deviation {max(yaw_deviations):.3f} deg exceeds {yaw_tolerance:.3f} deg"
                    )
            if speed_deviations:
                onset_metrics["maximum_speed_deviation_mps"] = max(speed_deviations)
                if max(speed_deviations) > speed_tolerance:
                    errors.append(
                        f"onset speed deviation {max(speed_deviations):.3f} m/s exceeds {speed_tolerance:.3f} m/s"
                    )
            if actor_deviations:
                onset_metrics["maximum_scenario_actor_deviation_m"] = max(actor_deviations)
                if max(actor_deviations) > actor_tolerance:
                    errors.append(
                        f"scenario actor deviation {max(actor_deviations):.3f} m exceeds {actor_tolerance:.3f} m"
                    )

        report = {
            "group_id": group_id,
            "scenario_id": scene_id,
            "route_id": rows[0]["route_id"],
            "town": rows[0]["town"],
            "accepted": not errors,
            "branch_count": len(rows),
            "decisions": sorted(observed),
            "outcomes": dict(Counter(row["outcome"] for row in rows)),
            "onset_alignment": onset_metrics,
            "errors": errors,
            "warnings": warnings,
        }
        group_reports.append(report)
        campaign_errors.extend(f"{group_id}: {error}" for error in errors)

    scene_counts = {}
    for scene_id in sorted(config.get("scenarios", {})):
        routes_per_scenario = int(
            config["scenarios"][scene_id].get(
                "routes_per_scenario", config.get("routes_per_scenario", 10)
            )
        )
        rows = [row for row in branch_rows if row["scenario_id"] == scene_id]
        groups = {row["group_id"] for row in rows}
        expected_branches = routes_per_scenario * len(
            config["scenarios"][scene_id]["decisions"]
        )
        scene_counts[scene_id] = {
            "groups": len(groups),
            "branches": len(rows),
            "expected_branches": expected_branches,
            "accepted_branches": sum(row["accepted"] for row in rows),
            "outcomes": dict(Counter(row["outcome"] for row in rows)),
            "decisions": dict(Counter(row["decision"] for row in rows)),
            "lane_change_executed": sum(
                row["decision"].startswith("LaneChange") and row["outcome"] != "lane_unavailable"
                for row in rows
            ),
            "lane_change_safety_rejected": sum(
                row["outcome"] == "lane_unavailable" for row in rows
            ),
        }
        if len(groups) != routes_per_scenario:
            campaign_errors.append(
                f"{scene_id}: expected {routes_per_scenario} route groups, found {len(groups)}"
            )
        if len(rows) != expected_branches:
            campaign_errors.append(
                f"{scene_id}: expected {expected_branches} branches, found {len(rows)}"
            )

    duplicate_jobs = len(jobs) - len({job["job_id"] for job in jobs})
    if duplicate_jobs:
        campaign_errors.append(f"manifest contains {duplicate_jobs} duplicate job ids")
    expected_groups = sum(
        int(scenario.get("routes_per_scenario", config.get("routes_per_scenario", 10)))
        for scenario in config.get("scenarios", {}).values()
    )
    expected_total_branches = sum(
        int(scenario.get("routes_per_scenario", config.get("routes_per_scenario", 10)))
        * len(scenario.get("decisions", []))
        for scenario in config.get("scenarios", {}).values()
    )
    if len(jobs) != expected_total_branches:
        campaign_errors.append(
            f"manifest contains {len(jobs)} jobs, expected {expected_total_branches}"
        )

    result = {
        "schema_version": "counterfactual_campaign_quality_v1",
        "accepted": not campaign_errors,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "expected_groups": expected_groups,
        "expected_branches": expected_total_branches,
        "observed_groups": len(by_group),
        "observed_branches": len(branch_rows),
        "accepted_branches": sum(row["accepted"] for row in branch_rows),
        "outcomes": dict(Counter(row["outcome"] for row in branch_rows)),
        "branch_reports": branch_rows,
        "scene_summary": scene_counts,
        "group_reports": group_reports,
        "errors": campaign_errors,
        "warnings": campaign_warnings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "job_id", "group_id", "scenario_id", "route_id", "town", "decision",
            "accepted", "outcome", "route_completion", "telemetry_frames", "saved_frames",
            "contract_verified", "run_dir",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in branch_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    print(json.dumps({
        "accepted": result["accepted"],
        "groups": result["observed_groups"],
        "branches": result["observed_branches"],
        "accepted_branches": result["accepted_branches"],
        "error_count": len(campaign_errors),
        "report": str(args.output),
        "table": str(args.csv),
    }, indent=2))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
