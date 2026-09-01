#!/usr/bin/env python3
"""Build a topology-planned 100-route-per-family CARLA Garage candidate pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SOURCE_TYPES = {
    "S1": (
        "PedestrianCrossing",
        "ParkingCrossingPedestrian",
        "DynamicObjectCrossing",
        "VehicleTurningRoutePedestrian",
    ),
    "S2": (
        "StaticCutIn",
        "ParkingCutIn",
        "HighwayCutIn",
        "MergerIntoSlowTraffic",
        "MergerIntoSlowTrafficV2",
    ),
    "S3": (
        "Accident",
        "ConstructionObstacle",
        "ParkedObstacle",
        "HazardAtSideLane",
        "AccidentTwoWays",
        "ConstructionObstacleTwoWays",
        "ParkedObstacleTwoWays",
        "HazardAtSideLaneTwoWays",
    ),
    "S4": ("NonSignalizedJunctionLeftTurn", "SignalizedJunctionLeftTurn"),
    "S5": ("NonSignalizedJunctionRightTurn", "SignalizedJunctionRightTurn"),
    "S6": ("SignalizedJunctionRightTurn", "NonSignalizedJunctionRightTurn"),
}
MIN_APPROACH = {"S1": 30.0, "S2": 35.0, "S3": 26.0, "S4": 22.0, "S5": 22.0, "S6": 22.0}
MIN_RECOVERY = {"S1": 35.0, "S2": 35.0, "S3": 30.0, "S4": 30.0, "S5": 30.0, "S6": 30.0}
TARGET_APPROACH = {"S1": 45.0, "S2": 50.0, "S3": 42.0, "S4": 35.0, "S5": 35.0, "S6": 35.0}
TARGET_RECOVERY = {family: 45.0 for family in MIN_RECOVERY}


def add_carla_paths(carla_root: Path, bench2drive_root: Path) -> None:
    paths = (
        carla_root / "PythonAPI",
        carla_root / "PythonAPI/carla",
        bench2drive_root,
        bench2drive_root / "scenario_runner",
    )
    for path in paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def xodr_path(carla_root: Path, town: str) -> Path:
    candidates = (
        carla_root / f"CarlaUE4/Content/Carla/Maps/{town}/OpenDrive/{town}.xodr",
        carla_root / f"CarlaUE4/Content/Carla/Maps/OpenDrive/{town}.xodr",
        carla_root / f"CarlaUE4/Content/Carla/Maps/OpenDrive/{town}_Opt.xodr",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing OpenDRIVE for {town}")


def distance(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    return math.hypot(second[0] - first[0], second[1] - first[1])


def cumulative(points: list[tuple[float, float, float]]) -> list[float]:
    result = [0.0]
    for first, second in zip(points, points[1:]):
        result.append(result[-1] + distance(first, second))
    return result


def circular_mean(values: list[float]) -> float:
    x = sum(math.cos(math.radians(value)) for value in values)
    y = sum(math.sin(math.radians(value)) for value in values)
    return math.degrees(math.atan2(y, x)) if values else 0.0


def angle_delta(first: float, second: float) -> float:
    return (second - first + 180.0) % 360.0 - 180.0


def headings(points: list[tuple[float, float, float]]) -> list[float]:
    return [
        math.degrees(math.atan2(second[1] - first[1], second[0] - first[0]))
        for first, second in zip(points, points[1:])
        if distance(first, second) >= 0.2
    ]


def turn_entry_s(points: list[tuple[float, float, float]], lengths: list[float]) -> float | None:
    segments = headings(points)
    if len(segments) < 8:
        return None
    baseline = circular_mean(segments[: min(6, len(segments))])
    segment_index = 0
    for index, (first, second) in enumerate(zip(points, points[1:])):
        if distance(first, second) < 0.2:
            continue
        if abs(angle_delta(baseline, segments[segment_index])) >= 16.0:
            return lengths[index]
        segment_index += 1
    return None


def point_at_s(points: list[tuple[float, float, float]], along: float) -> tuple[float, float, float]:
    lengths = cumulative(points)
    along = min(max(along, 0.0), lengths[-1])
    for index in range(len(points) - 1):
        if lengths[index + 1] + 1e-6 < along:
            continue
        span = max(lengths[index + 1] - lengths[index], 1e-6)
        ratio = (along - lengths[index]) / span
        return tuple(
            points[index][axis] + ratio * (points[index + 1][axis] - points[index][axis])
            for axis in range(3)
        )
    return points[-1]


def crop_polyline(
    points: list[tuple[float, float, float]], start_s: float, end_s: float
) -> list[tuple[float, float, float]]:
    lengths = cumulative(points)
    cropped = [point_at_s(points, start_s)]
    cropped.extend(
        point for point, along in zip(points, lengths) if start_s < along < end_s
    )
    cropped.append(point_at_s(points, end_s))
    return [
        point
        for index, point in enumerate(cropped)
        if index == 0 or distance(cropped[index - 1], point) >= 0.2
    ]


def parse_source(path: Path) -> tuple[ET.Element, list[tuple[float, float, float]]]:
    route = ET.parse(path).getroot().find("route")
    if route is None:
        raise ValueError("missing route")
    points = [
        (float(node.attrib["x"]), float(node.attrib["y"]), float(node.attrib.get("z", 0.0)))
        for node in route.findall("./waypoints/position")
    ]
    if len(points) < 2:
        raise ValueError("fewer than two source waypoints")
    return route, points


def planned_candidate(
    path: Path,
    family: str,
    source_type: str,
    town_map: Any,
    planner: Any,
    carla: Any,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        route, source_points = parse_source(path)
    except (ET.ParseError, OSError, ValueError) as exc:
        return None, f"source_parse:{exc}"
    town = route.attrib.get("town", "")
    start = carla.Location(x=source_points[0][0], y=source_points[0][1], z=source_points[0][2])
    end = carla.Location(x=source_points[-1][0], y=source_points[-1][1], z=source_points[-1][2])
    trace = planner.trace_route(start, end)
    points: list[tuple[float, float, float]] = []
    topology = []
    route_options = []
    for waypoint, option in trace:
        location = waypoint.transform.location
        point = (float(location.x), float(location.y), float(location.z))
        if points and distance(points[-1], point) < 0.5:
            continue
        points.append(point)
        topo = (int(waypoint.road_id), int(waypoint.section_id), int(waypoint.lane_id))
        if not topology or topology[-1] != topo:
            topology.append(topo)
        name = getattr(option, "name", str(option))
        if not route_options or route_options[-1] != name:
            route_options.append(name)
    if len(points) < 12:
        return None, "empty_or_short_planner_trace"
    maximum_anchor_error = max(
        min(distance(anchor, point) for point in points) for anchor in source_points
    )
    if maximum_anchor_error > 4.0:
        return None, "endpoint_trace_misses_source_anchor"
    lengths = cumulative(points)
    total = lengths[-1]
    segment_headings = headings(points)
    heading_change = angle_delta(
        circular_mean(segment_headings[:4]), circular_mean(segment_headings[-4:])
    )
    if family in {"S4", "S5", "S6"}:
        reference_s = turn_entry_s(points, lengths)
        if reference_s is None or abs(heading_change) < 45.0:
            return None, "missing_turn"
    else:
        if abs(heading_change) > 35.0:
            return None, "unexpected_turn"
        low = MIN_APPROACH[family]
        high = total - MIN_RECOVERY[family]
        reference_s = max(low, min(total * 0.55, high))
    if reference_s < MIN_APPROACH[family]:
        return None, "insufficient_approach"
    if total - reference_s < MIN_RECOVERY[family]:
        return None, "insufficient_recovery"
    if family == "S2" and any(name in {"CHANGELANELEFT", "CHANGELANERIGHT"} for name in route_options):
        return None, "ego_route_contains_lane_change"
    full_route_length = total
    crop_start_s = max(0.0, float(reference_s) - TARGET_APPROACH[family])
    crop_end_s = min(total, float(reference_s) + TARGET_RECOVERY[family])
    points = crop_polyline(points, crop_start_s, crop_end_s)
    reference_s = float(reference_s) - crop_start_s
    total = cumulative(points)[-1]
    if reference_s < MIN_APPROACH[family] or total - reference_s < MIN_RECOVERY[family]:
        return None, "cropped_route_lost_required_context"
    geometry = [[round(value, 4) for value in point] for point in points]
    geometry_hash = hashlib.sha256(json.dumps(geometry, separators=(",", ":")).encode()).hexdigest()
    source_key = f"{town}:{source_type}:{path.name}"
    numeric_source_id = str(int(hashlib.sha256(source_key.encode()).hexdigest()[:12], 16))
    candidate_id = f"{family.lower()}_{town}_{source_type}_{path.stem}"
    return {
        "family": family,
        "candidate_id": candidate_id,
        "source_dataset": "CARLA-Garage-generated-routes",
        "source_clip": str(path.resolve()),
        "source_clip_name": candidate_id,
        "source_scenario": source_type,
        "source_route_id": numeric_source_id,
        "source_weather_id": path.stem.rsplit("_", 1)[-1],
        "town": town,
        "frame_count": 0,
        "moving_point_count": len(points),
        "route_length_m": round(total, 3),
        "full_source_route_length_m": round(full_route_length, 3),
        "crop_start_s": round(crop_start_s, 3),
        "crop_end_s": round(crop_end_s, 3),
        "reference_s": round(float(reference_s), 3),
        "recovery_m": round(total - float(reference_s), 3),
        "heading_change_deg": round(heading_change, 3),
        "turn_entry_s": round(float(reference_s), 3) if family in {"S4", "S5", "S6"} else None,
        "topology": [list(value) for value in topology],
        "route_options": route_options,
        "source_track_overlap": 1.0,
        "maximum_source_anchor_error_m": round(maximum_anchor_error, 3),
        "points": geometry,
        "source_signature": hashlib.sha256(path.read_bytes()).hexdigest(),
        "geometry_sha256": geometry_hash,
        "reference_point": [round(value, 4) for value in point_at_s(points, float(reference_s))],
    }, None


def diverse_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    remaining = list(rows)
    selected = []
    town_count: Counter = Counter()
    type_count: Counter = Counter()
    while remaining:
        remaining.sort(
            key=lambda row: (
                town_count[row["town"]] * 3 + type_count[row["source_scenario"]],
                -float(row["route_length_m"]),
                row["candidate_id"],
            )
        )
        item = remaining.pop(0)
        selected.append(item)
        town_count[item["town"]] += 1
        type_count[item["source_scenario"]] += 1
    return selected


def adjacent_lane_options(candidate: dict[str, Any], town_map: Any, carla: Any) -> list[str]:
    point = candidate["reference_point"]
    waypoint = town_map.get_waypoint(
        carla.Location(x=point[0], y=point[1], z=point[2]),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if waypoint is None:
        return []
    options = []
    for side, adjacent in (("left", waypoint.get_left_lane()), ("right", waypoint.get_right_lane())):
        if adjacent is None or adjacent.lane_type != carla.LaneType.Driving:
            continue
        if int(adjacent.lane_id) * int(waypoint.lane_id) <= 0:
            continue
        options.append(side)
    return options


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/mnt/SSD/Coop_closed_loop/carla_garage/data"),
    )
    parser.add_argument(
        "--carla-root", type=Path, default=Path("/mnt/SSD/Coop_closed_loop/carla")
    )
    parser.add_argument(
        "--bench2drive-root",
        type=Path,
        default=Path("/mnt/SSD/Coop_closed_loop/Bench2Drive"),
    )
    parser.add_argument("--count-per-family", type=int, default=140)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    roots = (
        args.source_root / "50x38_Town12",
        args.source_root / "50x36_Town13",
    )
    add_carla_paths(args.carla_root, args.bench2drive_root)
    import carla  # type: ignore
    from agents.navigation.global_route_planner import GlobalRoutePlanner  # type: ignore

    maps = {}
    planners = {}
    for town in ("Town12", "Town13"):
        maps[town] = carla.Map(town, xodr_path(args.carla_root, town).read_text(encoding="utf-8"))
        planners[town] = GlobalRoutePlanner(maps[town], 1.0)

    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected = []
    for family, source_types in SOURCE_TYPES.items():
        seen_geometry = set()
        geometry_variants = []
        for root in roots:
            town = "Town12" if "Town12" in root.name else "Town13"
            for source_type in source_types:
                for path in sorted((root / source_type).glob("*.xml")):
                    candidate, reason = planned_candidate(
                        path, family, source_type, maps[town], planners[town], carla
                    )
                    if candidate is None:
                        rejected.append({"family": family, "source": str(path), "reason": reason})
                        continue
                    if family in {"S2", "S3"}:
                        options = adjacent_lane_options(candidate, maps[town], carla)
                        if not options:
                            rejected.append({"family": family, "source": str(path), "reason": "no_same_direction_adjacent_lane"})
                            continue
                        candidate["adjacent_lane_options_prechecked"] = options
                    geometry_hash = candidate["geometry_sha256"]
                    if geometry_hash in seen_geometry:
                        candidate["geometry_variant"] = True
                        geometry_variants.append(candidate)
                        rejected.append({"family": family, "source": str(path), "reason": "exact_geometry_variant_reserved"})
                        continue
                    seen_geometry.add(geometry_hash)
                    candidates[family].append(candidate)
        unique = diverse_order(candidates[family])
        variants = diverse_order(geometry_variants)
        candidates[family] = (unique + variants)[: args.count_per_family]

    payload = {
        "version": "counterfactual_scaleup_candidate_pool_v1",
        "source_roots": [str(path.resolve()) for path in roots],
        "requested_per_family": args.count_per_family,
        "family_counts": {family: len(candidates[family]) for family in SOURCE_TYPES},
        "candidates": dict(candidates),
        "rejection_counts": dict(Counter(row["reason"] for row in rejected)),
        "rejected": rejected,
    }
    args.output = args.output.resolve()
    if Path(__file__).resolve().parents[1] not in args.output.parents:
        raise SystemExit("output must stay under RL")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "family_counts": payload["family_counts"], "rejection_counts": payload["rejection_counts"]}, indent=2))
    return 0 if all(len(candidates[family]) >= 100 for family in SOURCE_TYPES) else 2


if __name__ == "__main__":
    raise SystemExit(main())
