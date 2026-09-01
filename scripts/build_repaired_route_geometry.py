#!/usr/bin/env python3
"""Repair a frozen route using validated geometry from the same source route."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path


POINT_NAMES = (
    "trigger_point",
    "scenario_arm_point",
    "cooperative_trigger_point",
    "collision_point",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--geometry-template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--event-from",
        choices=("event_trigger_point", "cooperative_trigger_point"),
        default="event_trigger_point",
    )
    args = parser.parse_args()
    source = args.source.resolve()
    template = args.geometry_template.resolve()
    output = args.output.resolve()
    source_tree = ET.parse(source)
    template_tree = ET.parse(template)
    source_route = source_tree.getroot().find("route")
    template_route = template_tree.getroot().find("route")
    source_scenario = source_tree.getroot().find(".//scenario")
    template_scenario = template_tree.getroot().find(".//scenario")
    if None in (source_route, template_route, source_scenario, template_scenario):
        raise RuntimeError("source/template must each contain one route and scenario")
    for key in ("town", "source_route_id"):
        if source_route.attrib.get(key) != template_route.attrib.get(key):
            raise RuntimeError(
                f"geometry provenance mismatch for {key}: "
                f"{source_route.attrib.get(key)} != {template_route.attrib.get(key)}"
            )

    source_waypoints = source_route.find("waypoints")
    template_waypoints = template_route.find("waypoints")
    if source_waypoints is None or template_waypoints is None:
        raise RuntimeError("missing waypoints")
    route_children = list(source_route)
    waypoint_index = route_children.index(source_waypoints)
    source_route.remove(source_waypoints)
    source_route.insert(waypoint_index, copy.deepcopy(template_waypoints))

    for name in POINT_NAMES:
        source_point = source_scenario.find(name)
        template_point = template_scenario.find(name)
        if source_point is None or template_point is None:
            raise RuntimeError(f"missing geometry point: {name}")
        source_point.attrib.clear()
        source_point.attrib.update(template_point.attrib)
    event_template = template_scenario.find(args.event_from)
    source_event = source_scenario.find("event_trigger_point")
    if event_template is None or source_event is None:
        raise RuntimeError("missing event trigger point")
    source_event.attrib.clear()
    source_event.attrib.update(event_template.attrib)

    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(source_tree, space="  ")
    source_tree.write(output, encoding="utf-8", xml_declaration=True)
    provenance = {
        "repair": "replace route/trigger geometry while preserving scenario semantics",
        "source": str(source),
        "source_sha256": sha256(source),
        "geometry_template": str(template),
        "geometry_template_sha256": sha256(template),
        "output": str(output),
        "output_sha256": sha256(output),
        "town": source_route.attrib.get("town"),
        "source_route_id": source_route.attrib.get("source_route_id"),
        "scenario_name": source_scenario.attrib.get("name"),
        "scenario_type": source_scenario.attrib.get("type"),
        "event_from": args.event_from,
    }
    output.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
