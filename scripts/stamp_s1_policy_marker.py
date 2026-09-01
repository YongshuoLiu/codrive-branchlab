#!/usr/bin/env python3
"""Stamp an S1 branch with the pedestrian-policy version encoded in its route XML."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


TARGET_POLICY = "S1_PED_SPEED_V2"
TARGET_SPEED_MPS = 8.5
TARGET_DECISIONS = {"Accelerate", "Maintain"}


def xml_value(scenario: ET.Element, tag: str) -> str | None:
    node = scenario.find(tag)
    return node.get("value") if node is not None else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--route", type=Path)
    parser.add_argument("--decision")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    route_path = (args.route or run_dir / "metadata/source_route.xml").resolve()
    if not route_path.is_file():
        return 0

    root = ET.parse(route_path).getroot()
    route = root.find("route")
    if route is None:
        return 0
    scenario = next(
        (
            item
            for item in route.findall("./scenarios/scenario")
            if item.get("type") == "ClosedOccludedPedestrian"
            or xml_value(item, "closed_loop_family") == "occluded_pedestrian"
        ),
        None,
    )
    if scenario is None:
        return 0

    speed_text = xml_value(scenario, "ped_speed")
    try:
        speed = float(speed_text) if speed_text is not None else None
    except ValueError:
        speed = None
    explicit_version = xml_value(scenario, "s1_policy_version") or route.get("s1_policy_version")
    if explicit_version == TARGET_POLICY and speed is not None and speed >= TARGET_SPEED_MPS:
        policy_version = TARGET_POLICY
    elif speed is not None and speed >= TARGET_SPEED_MPS:
        policy_version = "S1_UNVERSIONED_FAST_PEDESTRIAN"
    else:
        policy_version = "S1_LEGACY_PED_SPEED"

    decision = args.decision or run_dir.name
    marker = {
        "schema_version": "s1_pedestrian_policy_marker_v1",
        "scenario_id": "S1",
        "route_id": route.get("id"),
        "scenario_name": scenario.get("name"),
        "decision": decision,
        "ped_speed_mps": speed,
        "policy_version": policy_version,
        "target_policy_version": TARGET_POLICY,
        "target_ped_speed_mps": TARGET_SPEED_MPS,
        "is_recollected_v2": policy_version == TARGET_POLICY,
        "needs_recollection": decision in TARGET_DECISIONS and policy_version != TARGET_POLICY,
        "source_route_xml": str(route_path),
    }
    marker_path = run_dir / "metadata/s1_policy_marker.json"
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
