#!/usr/bin/env python3
"""Inspect CARLA lane topology and route-trace proximity for a frozen XML route."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import carla

from agents.navigation.global_route_planner import GlobalRoutePlanner
from srunner.tools.scenario_helper import (
    filter_junction_wp_direction,
    get_junction_topology,
)


def waypoint_dict(waypoint: carla.Waypoint | None) -> dict | None:
    if waypoint is None:
        return None
    transform = waypoint.transform
    return {
        "road_id": waypoint.road_id,
        "section_id": waypoint.section_id,
        "lane_id": waypoint.lane_id,
        "s": round(waypoint.s, 3),
        "is_junction": waypoint.is_junction,
        "junction_id": waypoint.junction_id if waypoint.is_junction else None,
        "lane_type": str(waypoint.lane_type),
        "location": {
            "x": round(transform.location.x, 3),
            "y": round(transform.location.y, 3),
            "z": round(transform.location.z, 3),
        },
        "yaw": round(transform.rotation.yaw, 3),
    }


def location_from(element: ET.Element) -> carla.Location:
    return carla.Location(
        x=float(element.attrib["x"]),
        y=float(element.attrib["y"]),
        z=float(element.attrib.get("z", 0.0)),
    )


def nearest_distance(location: carla.Location, waypoints: list[carla.Waypoint]) -> float | None:
    if not waypoints:
        return None
    return min(location.distance(item.transform.location) for item in waypoints)


def first_junction(waypoint: carla.Waypoint, limit_m: int = 100) -> tuple[carla.Waypoint | None, int]:
    current = waypoint
    for distance in range(limit_m + 1):
        if current.is_junction:
            return current, distance
        candidates = current.next(1.0)
        if not candidates:
            return None, distance
        current = candidates[0]
    return None, limit_m


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=32100)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    tree = ET.parse(args.route)
    route = tree.getroot().find("route")
    scenario = tree.getroot().find(".//scenario")
    if route is None or scenario is None:
        raise RuntimeError("route XML must contain one route and one scenario")

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    requested_town = route.attrib["town"]
    current_town = world.get_map().name.split("/")[-1]
    if current_town != requested_town:
        world = client.load_world(requested_town)
    carla_map = world.get_map()

    xml_waypoints = list(route.findall("./waypoints/position"))
    start = location_from(xml_waypoints[0])
    end = location_from(xml_waypoints[-1])
    trigger_element = scenario.find("trigger_point")
    if trigger_element is None:
        raise RuntimeError("missing trigger_point")
    trigger = location_from(trigger_element)

    planner = GlobalRoutePlanner(carla_map, 1.0)
    trace = planner.trace_route(start, end)
    trace_waypoints = [item[0] for item in trace]
    trigger_wp = carla_map.get_waypoint(trigger, project_to_road=True, lane_type=carla.LaneType.Driving)
    junction_wp, junction_distance = first_junction(trigger_wp)

    payload = {
        "route": str(args.route.resolve()),
        "route_id": route.attrib.get("id"),
        "town": requested_town,
        "xml_waypoints": len(xml_waypoints),
        "start": waypoint_dict(carla_map.get_waypoint(start, project_to_road=True, lane_type=carla.LaneType.Driving)),
        "end": waypoint_dict(carla_map.get_waypoint(end, project_to_road=True, lane_type=carla.LaneType.Driving)),
        "trigger": {
            "xml": {"x": trigger.x, "y": trigger.y, "z": trigger.z},
            "projected": waypoint_dict(trigger_wp),
            "projection_distance_m": round(trigger.distance(trigger_wp.transform.location), 3),
            "minimum_trace_distance_m": round(nearest_distance(trigger, trace_waypoints), 3),
        },
        "trace": {
            "waypoints": len(trace_waypoints),
            "length_m": round(
                sum(
                    trace_waypoints[index - 1].transform.location.distance(item.transform.location)
                    for index, item in enumerate(trace_waypoints)
                    if index
                ),
                3,
            ),
        },
        "junction": None,
    }
    if junction_wp is not None:
        junction = junction_wp.get_junction()
        entries, exits = get_junction_topology(junction)
        payload["junction"] = {
            "id": junction.id,
            "distance_ahead_m": junction_distance,
            "ego_entry": waypoint_dict(trigger_wp),
            "entries": [waypoint_dict(item) for item in entries],
            "left_entries": [
                waypoint_dict(item)
                for item in filter_junction_wp_direction(junction_wp, entries, "left")
            ],
            "right_entries": [
                waypoint_dict(item)
                for item in filter_junction_wp_direction(junction_wp, entries, "right")
            ],
            "opposite_entries": [
                waypoint_dict(item)
                for item in filter_junction_wp_direction(junction_wp, entries, "opposite")
            ],
            "exits": [waypoint_dict(item) for item in exits],
        }

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
