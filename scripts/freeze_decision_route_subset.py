#!/usr/bin/env python3
"""Freeze a numbered subset of LANTERN closed-loop routes inside RL."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import xml.etree.ElementTree as ET


FAMILIES = (
    "occluded_pedestrian",
    "lane_cut_in",
    "reveal_obstacle",
    "ghost_driver_left_turn",
    "ghost_driver_right_turn",
    "right_turn_on_red",
)

DEFAULT_SOURCE = pathlib.Path(
    "/home/UNT/yl0826/simlingo/pedestrian_dataset/"
    "huggingface_release/LANTERN/closedloop/routes"
)
DEFAULT_OUTPUT = pathlib.Path(
    "/home/UNT/yl0826/simlingo/RL/routes/decision_v1_pilot_0011_0015"
)


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_route(path: pathlib.Path, expected_family: str, expected_index: int) -> dict:
    tree = ET.parse(path)
    route = tree.getroot().find("route")
    scenarios = tree.getroot().findall(".//scenario")
    if route is None or len(scenarios) != 1:
        raise RuntimeError(f"expected one route and one scenario in {path}")
    expected_id = f"closed_{expected_family}_{expected_index:04d}"
    if route.attrib.get("id") != expected_id:
        raise RuntimeError(
            f"route id mismatch in {path}: {route.attrib.get('id')} != {expected_id}"
        )
    positions = route.findall("./waypoints/position")
    if len(positions) < 2:
        raise RuntimeError(f"route has fewer than two waypoints: {path}")
    scenario = scenarios[0]
    required_points = (
        "trigger_point",
        "scenario_arm_point",
        "cooperative_trigger_point",
        "event_trigger_point",
        "collision_point",
    )
    missing = [name for name in required_points if scenario.find(name) is None]
    if missing:
        raise RuntimeError(f"route is missing scenario points {missing}: {path}")
    return {
        "route_id": expected_id,
        "town": route.attrib.get("town"),
        "source_route_id": route.attrib.get("source_route_id"),
        "scenario_name": scenario.attrib.get("name"),
        "scenario_type": scenario.attrib.get("type"),
        "waypoint_count": len(positions),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=pathlib.Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", type=int, default=11)
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    rl_routes = pathlib.Path("/home/UNT/yl0826/simlingo/RL/routes").resolve()
    if rl_routes not in output_root.parents:
        raise SystemExit(f"output must remain below {rl_routes}: {output_root}")

    rows = []
    for family in FAMILIES:
        for index in range(args.start, args.start + args.count):
            filename = f"closed_{family}_{index:04d}.xml"
            source = source_root / family / filename
            if not source.is_file():
                raise RuntimeError(f"missing source route: {source}")
            metadata = inspect_route(source, family, index)
            target = output_root / family / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            source_hash = sha256(source)
            if target.exists() and sha256(target) != source_hash:
                raise RuntimeError(f"refusing to replace a changed frozen route: {target}")
            if not target.exists():
                shutil.copy2(source, target)
            rows.append(
                {
                    "family": family,
                    "index": index,
                    "source": str(source),
                    "source_sha256": source_hash,
                    "frozen_route": str(target),
                    "frozen_sha256": sha256(target),
                    **metadata,
                }
            )

    provenance = {
        "schema_version": "frozen_decision_route_subset_v1",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "start": args.start,
        "count_per_family": args.count,
        "families": list(FAMILIES),
        "routes": rows,
    }
    manifest = output_root / "frozen_routes.manifest.json"
    manifest.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"routes": len(rows), "manifest": str(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
