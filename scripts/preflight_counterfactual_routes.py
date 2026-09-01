#!/usr/bin/env python3
"""Batch preflight frozen counterfactual routes before expensive CARLA collection."""

from __future__ import annotations

import argparse
import json
import math
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from counterfactual_contract import (
    finite_point,
    read_json,
    route_xml_contract,
    sha256,
    validate_manifest_job,
)


RL_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_EVENT_POINTS = (
    "scenario_arm_point",
    "cooperative_trigger_point",
    "event_trigger_point",
    "collision_point",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def point(element: ET.Element) -> tuple[float, float, float]:
    return tuple(float(element.attrib.get(key, 0.0)) for key in ("x", "y", "z"))


def polyline_projection(
    target: tuple[float, float, float],
    points: list[tuple[float, float, float]],
) -> tuple[float, float]:
    """Return straight-line polyline distance and arc position."""
    best_distance = math.inf
    best_arc = 0.0
    accumulated = 0.0
    for start, end in zip(points, points[1:]):
        vector = tuple(end[index] - start[index] for index in range(3))
        offset = tuple(target[index] - start[index] for index in range(3))
        length_squared = sum(value * value for value in vector)
        ratio = (
            max(
                0.0,
                min(
                    1.0,
                    sum(offset[index] * vector[index] for index in range(3))
                    / length_squared,
                ),
            )
            if length_squared
            else 0.0
        )
        projected = tuple(
            start[index] + ratio * vector[index] for index in range(3)
        )
        distance = math.dist(target, projected)
        segment_length = math.sqrt(length_squared)
        if distance < best_distance:
            best_distance = distance
            best_arc = accumulated + ratio * segment_length
        accumulated += segment_length
    return best_distance, best_arc


def static_route_check(route_path: Path, decisions: set[str]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    contract, contract_errors = route_xml_contract(route_path)
    errors.extend(contract_errors)
    if contract is None:
        return {
            "route_path": str(route_path),
            "accepted": False,
            "errors": errors,
            "warnings": warnings,
        }
    root = ET.parse(route_path).getroot()
    route = root.find("route")
    scenario = root.find(".//scenario")
    assert route is not None and scenario is not None
    if not str(contract["scenario_type"]).startswith("Closed"):
        errors.append(
            f"scenario type must be a production Closed* rule, found {contract['scenario_type']}"
        )
    waypoint_elements = route.findall("./waypoints/position")
    if len(waypoint_elements) < 2:
        errors.append(f"route needs at least two XML waypoints, found {len(waypoint_elements)}")
        route_points: list[tuple[float, float, float]] = []
    elif not all(finite_point(element) for element in waypoint_elements):
        errors.append("route contains a non-finite XML waypoint")
        route_points = []
    else:
        route_points = [point(element) for element in waypoint_elements]

    event_metrics: dict[str, Any] = {}
    event_arcs: dict[str, float] = {}
    for name in REQUIRED_EVENT_POINTS:
        element = scenario.find(name)
        if not finite_point(element):
            errors.append(f"missing or non-finite {name}")
            continue
        if route_points:
            distance_m, arc_m = polyline_projection(point(element), route_points)
            event_arcs[name] = arc_m
            event_metrics[name] = {
                "raw_polyline_distance_m": round(distance_m, 3),
                "raw_polyline_arc_m": round(arc_m, 3),
            }
            # Expanded routes can be curved while their XML contains only sparse
            # endpoints. This is diagnostic only; --live checks the CARLA trace.
            if distance_m > 5.0:
                warnings.append(
                    f"{name} is {distance_m:.2f} m from the sparse raw XML polyline; "
                    "require live trace verification"
                )
    ordered = [event_arcs.get(name) for name in REQUIRED_EVENT_POINTS]
    if all(value is not None for value in ordered) and any(
        float(ordered[index]) + 0.5 < float(ordered[index - 1])
        for index in range(1, len(ordered))
    ):
        errors.append(
            "event points are not ordered arm <= cooperative <= event <= collision"
        )
    route_length_m = (
        sum(math.dist(first, second) for first, second in zip(route_points, route_points[1:]))
        if route_points
        else 0.0
    )
    if route_points and route_length_m < 10.0:
        errors.append(f"raw XML route is only {route_length_m:.2f} m long")

    cooperative_enabled = scenario.find("cooperative_enabled")
    if cooperative_enabled is None or cooperative_enabled.attrib.get("value", "").lower() != "true":
        errors.append("cooperative_enabled must be true")
    if decisions & {"LaneChangeLeft", "LaneChangeRight"}:
        if scenario.find("cooperative_trigger_point") is None:
            errors.append("lane decisions require cooperative_trigger_point")

    return {
        **contract,
        "route_path": str(route_path.resolve()),
        "decisions": sorted(decisions),
        "xml_waypoints": len(waypoint_elements),
        "raw_route_length_m": round(route_length_m, 3),
        "event_points": event_metrics,
        "accepted": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def carla_location(carla: Any, element: ET.Element) -> Any:
    return carla.Location(
        x=float(element.attrib["x"]),
        y=float(element.attrib["y"]),
        z=float(element.attrib.get("z", 0.0)),
    )


def live_checks(
    route_reports: list[dict[str, Any]], host: str, port: int, timeout: float,
    trace_tolerance_m: float,
) -> None:
    try:
        import carla  # type: ignore
        from agents.navigation.global_route_planner import GlobalRoutePlanner  # type: ignore
        from srunner.tools.scenario_helper import (  # type: ignore
            filter_junction_wp_direction,
            get_junction_topology,
        )
    except ImportError as exc:
        raise RuntimeError(
            "live preflight needs the simlingo Python environment and CARLA/scenario_runner PYTHONPATH"
        ) from exc

    client = carla.Client(host, port)
    client.set_timeout(timeout)
    world = client.get_world()
    current_town = world.get_map().name.split("/")[-1]
    planner_town = None
    planner = None
    # Loading a CARLA town dominates preflight time. Group by town so each map
    # is loaded once even when route files are organized by scenario family.
    for report in sorted(
        route_reports, key=lambda item: (str(item.get("town")), str(item.get("route_id")))
    ):
        if report["errors"]:
            report["live"] = {"accepted": False, "errors": ["static check failed"]}
            report["accepted"] = False
            continue
        root = ET.parse(report["route_path"]).getroot()
        route = root.find("route")
        scenario = root.find(".//scenario")
        assert route is not None and scenario is not None
        town = route.attrib["town"]
        if current_town != town:
            world = client.load_world(town)
            loaded_names = []
            stable_reads = 0
            deadline = time.time() + 30.0
            while time.time() < deadline and stable_reads < 2:
                world = client.get_world()
                observed_name = world.get_map().name.split("/")[-1]
                loaded_names.append(observed_name)
                stable_reads = stable_reads + 1 if observed_name == town else 0
                time.sleep(1.0)
            if stable_reads < 2:
                raise RuntimeError(
                    f"CARLA map did not stabilize on {town}: observed {loaded_names}"
                )
            current_town = loaded_names[-1]
        carla_map = world.get_map()
        if planner is None or planner_town != current_town:
            planner = GlobalRoutePlanner(carla_map, 1.0)
            planner_town = current_town
        xml_points = route.findall("./waypoints/position")
        start = carla_location(carla, xml_points[0])
        end = carla_location(carla, xml_points[-1])
        trace = planner.trace_route(start, end)
        trace_waypoints = [item[0] for item in trace]
        anchor_locations = [
            carla_location(carla, scenario.find(name))
            for name in REQUIRED_EVENT_POINTS
        ]
        anchored_trace_waypoints = []
        for segment_start, segment_end in zip(
            [start, *anchor_locations], [*anchor_locations, end]
        ):
            segment = planner.trace_route(segment_start, segment_end)
            anchored_trace_waypoints.extend(item[0] for item in segment)
        live_errors: list[str] = []
        live_warnings: list[str] = []
        point_metrics: dict[str, Any] = {}
        if not trace_waypoints:
            live_errors.append("CARLA planner returned an empty route trace")
        for name in REQUIRED_EVENT_POINTS:
            element = scenario.find(name)
            if element is None or not trace_waypoints:
                continue
            location = carla_location(carla, element)
            endpoint_trace_distance_m = min(
                location.distance(item.transform.location) for item in trace_waypoints
            )
            anchored_trace_distance_m = min(
                location.distance(item.transform.location)
                for item in anchored_trace_waypoints
            )
            projected = carla_map.get_waypoint(
                location, project_to_road=True, lane_type=carla.LaneType.Driving
            )
            road_projection_distance_m = (
                location.distance(projected.transform.location)
                if projected is not None
                else math.inf
            )
            point_metrics[name] = {
                "endpoint_trace_distance_m": round(endpoint_trace_distance_m, 3),
                "anchored_trace_distance_m": round(anchored_trace_distance_m, 3),
                "driving_lane_projection_distance_m": round(road_projection_distance_m, 3),
            }
            if anchored_trace_distance_m > trace_tolerance_m:
                live_errors.append(
                    f"{name} is {anchored_trace_distance_m:.2f} m from the anchored CARLA trace "
                    f"(limit {trace_tolerance_m:.2f} m)"
                )
            if road_projection_distance_m > trace_tolerance_m:
                live_errors.append(
                    f"{name} is {road_projection_distance_m:.2f} m from a CARLA driving lane "
                    f"(limit {trace_tolerance_m:.2f} m)"
                )
            if endpoint_trace_distance_m > trace_tolerance_m:
                live_warnings.append(
                    f"{name} is {endpoint_trace_distance_m:.2f} m from the endpoint-only "
                    "planner trace; route has an ambiguous alternate path and should be "
                    "reviewed or frozen with an intermediate XML waypoint"
                )

        lane_metrics: dict[str, Any] = {}
        cooperative = scenario.find("cooperative_trigger_point")
        if cooperative is not None and report["decisions"]:
            location = carla_location(carla, cooperative)
            waypoint = carla_map.get_waypoint(
                location, project_to_road=True, lane_type=carla.LaneType.Driving
            )
            for decision, method_name in (
                ("LaneChangeLeft", "get_left_lane"),
                ("LaneChangeRight", "get_right_lane"),
            ):
                if decision not in report["decisions"]:
                    continue
                adjacent = getattr(waypoint, method_name)() if waypoint else None
                same_direction = bool(
                    adjacent
                    and adjacent.lane_type == carla.LaneType.Driving
                    and waypoint.lane_id * adjacent.lane_id > 0
                )
                lane_metrics[decision] = {
                    "adjacent_driving_lane": bool(adjacent and adjacent.lane_type == carla.LaneType.Driving),
                    "same_direction": same_direction,
                    "target_lane_id": adjacent.lane_id if adjacent else None,
                }
                if not same_direction:
                    live_warnings.append(
                        f"{decision} has no same-direction adjacent lane at the trigger; "
                        "lane_unavailable is the expected recorded outcome"
                    )

        junction_metrics = None
        if report["scenario_type"] == "ClosedSignalizedJunctionRightTurnOnRed":
            cooperative_location = carla_location(
                carla, scenario.find("cooperative_trigger_point")
            )
            current = carla_map.get_waypoint(
                cooperative_location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            distance_ahead = 0
            while current is not None and not current.is_junction and distance_ahead < 100:
                candidates = current.next(1.0)
                current = candidates[0] if candidates else None
                distance_ahead += 1
            left_entries = []
            entry_count = 0
            if current is not None and current.is_junction:
                entries, _ = get_junction_topology(current.get_junction())
                entry_count = len(entries)
                left_entries = filter_junction_wp_direction(current, entries, "left")
            junction_metrics = {
                "found": bool(current and current.is_junction),
                "distance_ahead_m": distance_ahead,
                "entry_count": entry_count,
                "left_cross_traffic_entries": len(left_entries),
            }
            if not left_entries:
                live_errors.append(
                    "right-turn-on-red rule has no left cross-traffic junction entry"
                )

        report["live"] = {
            "accepted": not live_errors,
            "loaded_map": current_town,
            "endpoint_trace_waypoints": len(trace_waypoints),
            "anchored_trace_waypoints": len(anchored_trace_waypoints),
            "event_points": point_metrics,
            "lane_topology": lane_metrics,
            "junction": junction_metrics,
            "errors": live_errors,
            "warnings": live_warnings,
        }
        report["errors"].extend(f"live: {error}" for error in live_errors)
        report["warnings"].extend(f"live: {warning}" for warning in live_warnings)
        report["accepted"] = not report["errors"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=32100)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--trace-tolerance-m", type=float, default=2.0)
    args = parser.parse_args()

    config = read_json(args.config, {}) or {}
    jobs = read_jsonl(args.manifest)
    errors: list[str] = []
    warnings: list[str] = []
    job_contract_errors = {
        job.get("job_id", f"row_{index}"): contract_errors
        for index, job in enumerate(jobs)
        if (contract_errors := validate_manifest_job(job))
    }
    errors.extend(
        f"{job_id}: {error}"
        for job_id, contract_errors in job_contract_errors.items()
        for error in contract_errors
    )
    for field in ("job_id", "output_dir"):
        values = [job.get(field) for job in jobs]
        duplicate_count = len(values) - len(set(values))
        if duplicate_count:
            errors.append(f"manifest contains {duplicate_count} duplicate {field} values")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    route_jobs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        groups[str(job.get("group_id"))].append(job)
        route_jobs[str(Path(str(job.get("route_path", ""))).resolve())].append(job)
    for group_id, rows in sorted(groups.items()):
        scenario_id = rows[0].get("scenario_id")
        expected = set((config.get("scenarios", {}).get(scenario_id) or {}).get("decisions", []))
        observed = {row.get("decision") for row in rows}
        if observed != expected:
            errors.append(
                f"{group_id}: decisions differ from config: expected {sorted(expected)}, "
                f"observed {sorted(observed)}"
            )
        identities = {
            (row.get("route_id"), row.get("route_sha256"), row.get("route_path"))
            for row in rows
        }
        if len(identities) != 1:
            errors.append(f"{group_id}: decisions do not share one frozen route identity")

    route_reports = [
        static_route_check(
            Path(route_path), {str(job.get("decision")) for job in route_rows}
        )
        for route_path, route_rows in sorted(route_jobs.items())
    ]
    errors.extend(
        f"{report.get('route_id', report['route_path'])}: {error}"
        for report in route_reports
        for error in report["errors"]
    )
    if args.live:
        live_checks(
            route_reports,
            args.host,
            args.port,
            args.timeout,
            args.trace_tolerance_m,
        )
        # Rebuild route errors because live checks were appended after static aggregation.
        errors = [error for error in errors if not error.startswith("live:")]
        errors.extend(
            f"{report.get('route_id', report['route_path'])}: {error}"
            for report in route_reports
            for error in report["errors"]
            if f"{report.get('route_id', report['route_path'])}: {error}" not in errors
        )

    for scenario_id in sorted(config.get("scenarios", {})):
        scenario = config["scenarios"][scenario_id]
        expected_routes = int(
            scenario.get("routes_per_scenario", config.get("routes_per_scenario", 0))
        )
        scene_routes = [
            report for report in route_reports
            if any(
                job.get("scenario_id") == scenario_id
                for job in route_jobs[report["route_path"]]
            )
        ]
        if len(scene_routes) != expected_routes:
            errors.append(
                f"{scenario_id}: expected {expected_routes} unique routes, found {len(scene_routes)}"
            )
        route_ids = [report.get("route_id") for report in scene_routes]
        route_hashes = [report.get("route_sha256") for report in scene_routes]
        if len(route_ids) != len(set(route_ids)):
            errors.append(f"{scenario_id}: route ids are not unique")
        if len(route_hashes) != len(set(route_hashes)):
            errors.append(f"{scenario_id}: route XML hashes are not unique")

    result = {
        "schema_version": "counterfactual_route_preflight_v1",
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "accepted": not errors,
        "live_checked": bool(args.live),
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "validator_sha256": sha256(Path(__file__).resolve()),
        "contract_validator_sha256": sha256(
            Path(__file__).resolve().with_name("counterfactual_contract.py")
        ),
        "jobs": len(jobs),
        "groups": len(groups),
        "routes": len(route_reports),
        "towns": dict(Counter(report.get("town") for report in route_reports)),
        "errors": errors,
        "warnings": warnings,
        "route_reports": route_reports,
    }
    output = args.output.resolve()
    if RL_ROOT.resolve() not in output.parents:
        raise SystemExit(f"preflight output must stay under {RL_ROOT.resolve()}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "accepted": result["accepted"],
                "live_checked": result["live_checked"],
                "jobs": result["jobs"],
                "groups": result["groups"],
                "routes": result["routes"],
                "error_count": len(errors),
                "route_warning_count": sum(len(row["warnings"]) for row in route_reports),
                "report": str(output),
            },
            indent=2,
        )
    )
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
