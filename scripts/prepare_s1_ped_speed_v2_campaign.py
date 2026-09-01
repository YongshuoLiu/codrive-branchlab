#!/usr/bin/env python3
"""Materialize S1 V2 routes and build an Accelerate/Maintain recollection campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


RL_ROOT = Path(__file__).resolve().parents[1]
POLICY_VERSION = "S1_PED_SPEED_V2"
DECISIONS = ("Accelerate", "Maintain")
SOURCE_DECISION_ORDER = (
    "Maintain", "Accelerate", "Brake", "Stop", "LaneChangeLeft", "LaneChangeRight"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_route(route_dir: Path) -> Path | None:
    for decision in SOURCE_DECISION_ORDER:
        candidate = route_dir / decision / "metadata/source_route.xml"
        if candidate.is_file():
            return candidate
    candidates = sorted(
        path
        for path in route_dir.glob("*/metadata/source_route.xml")
        if "_incomplete_" not in str(path)
    )
    return candidates[0] if candidates else None


def update_parameter(scenario: ET.Element, name: str, value: str) -> None:
    node = scenario.find(name)
    if node is None:
        node = ET.SubElement(scenario, name)
    node.set("value", value)


def write_route(source: Path, output: Path, speed: float) -> dict:
    tree = ET.parse(source)
    root = tree.getroot()
    route = root.find("route")
    if route is None:
        raise ValueError(f"missing route node: {source}")
    scenarios = [
        item
        for item in route.findall("./scenarios/scenario")
        if item.get("type") == "ClosedOccludedPedestrian"
    ]
    if len(scenarios) != 1:
        raise ValueError(f"expected one ClosedOccludedPedestrian scenario: {source}")
    scenario = scenarios[0]
    route.set("s1_policy_version", POLICY_VERSION)
    update_parameter(scenario, "ped_speed", f"{speed:g}")
    update_parameter(scenario, "s1_policy_version", POLICY_VERSION)
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return {
        "route_id": route.get("id"),
        "scenario_name": scenario.get("name"),
        "source_xml": str(source.resolve()),
        "source_sha256": sha256(source),
        "v2_xml": str(output.resolve()),
        "v2_sha256": sha256(output),
        "ped_speed_mps": speed,
        "policy_version": POLICY_VERSION,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-root", type=Path,
        default=RL_ROOT / "data/counterfactual_decision_v1/S1",
    )
    parser.add_argument(
        "--route-root", type=Path,
        default=RL_ROOT / "routes/decision_v2_s1_ped_speed_8p5",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=RL_ROOT / "data/counterfactual_decision_v1/_recollected/S1_PED_SPEED_V2",
    )
    parser.add_argument(
        "--config", type=Path,
        default=RL_ROOT / "config/s1_ped_speed_v2_recollection.json",
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=RL_ROOT / "manifests/s1_ped_speed_v2_recollection.jsonl",
    )
    parser.add_argument("--ped-speed", type=float, default=8.5)
    args = parser.parse_args()

    baseline_root = args.baseline_root.resolve()
    route_root = args.route_root.resolve()
    output_root = args.output_root.resolve()
    if RL_ROOT.resolve() not in route_root.parents or RL_ROOT.resolve() not in output_root.parents:
        raise SystemExit("route and output roots must remain inside RL")

    rows = []
    missing = []
    route_dirs = sorted(
        path
        for path in baseline_root.glob("closed_occluded_pedestrian_*")
        if path.is_dir()
    )
    for route_dir in route_dirs:
        source = source_route(route_dir)
        if source is None:
            missing.append(route_dir.name)
            continue
        output = route_root / "occluded_pedestrian" / f"{route_dir.name}.xml"
        rows.append(write_route(source, output, args.ped_speed))
    if missing:
        raise RuntimeError(f"S1 routes without source_route.xml: {missing}")
    if not rows:
        raise RuntimeError("no S1 routes discovered")

    base_config = json.loads((RL_ROOT / "config/decision_v1.json").read_text(encoding="utf-8"))
    config = {
        "schema_version": "counterfactual_decision_v1",
        "project_root": str(RL_ROOT.parent.resolve()),
        "output_root": str(output_root),
        "source_route_root": str(route_root),
        "routes_per_scenario": len(rows),
        "route_index_start": 1,
        "scenarios": {
            "S1": {
                "family": "occluded_pedestrian",
                "name": "Pedestrian Emergence S1_PED_SPEED_V2",
                "decisions": list(DECISIONS),
            }
        },
        "decision_parameters": base_config["decision_parameters"],
        "quality_thresholds": base_config["quality_thresholds"],
        "workers": [
            {"slot": 0, "gpu": 0, "graphics_adapter": 1, "port": 32400, "tm_port": 42400},
            {"slot": 1, "gpu": 1, "graphics_adapter": 2, "port": 32500, "tm_port": 42500},
            {"slot": 2, "gpu": 2, "graphics_adapter": 0, "port": 32600, "tm_port": 42600}
        ],
        "s1_policy": {
            "policy_version": POLICY_VERSION,
            "ped_speed_mps": args.ped_speed,
            "recollected_decisions": list(DECISIONS),
            "legacy_marker_rule": "missing marker or policy_version != S1_PED_SPEED_V2",
        },
    }
    args.config.parent.mkdir(parents=True, exist_ok=True)
    args.config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            str(RL_ROOT / "scripts/build_collection_manifest.py"),
            "--config", str(args.config),
            "--output", str(args.manifest),
        ],
        check=True,
    )
    plan = {
        "schema_version": "s1_ped_speed_v2_recollection_plan_v1",
        "policy_version": POLICY_VERSION,
        "ped_speed_mps": args.ped_speed,
        "route_count": len(rows),
        "branch_count": len(rows) * len(DECISIONS),
        "decisions": list(DECISIONS),
        "baseline_root": str(baseline_root),
        "output_root": str(output_root),
        "config": str(args.config.resolve()),
        "manifest": str(args.manifest.resolve()),
        "routes": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "recollection_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: plan[key] for key in ("policy_version", "ped_speed_mps", "route_count", "branch_count", "output_root", "manifest")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
