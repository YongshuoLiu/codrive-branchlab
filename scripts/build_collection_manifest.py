#!/usr/bin/env python3
"""Build a frozen counterfactual collection manifest from a decision-v1 config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from counterfactual_contract import route_xml_contract, sha256, validate_manifest_job


RL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = RL_ROOT / "config/decision_v1.json"
DEFAULT_OUTPUT = RL_ROOT / "manifests/collection_manifest.jsonl"


def scenario_route_count(config: dict, scenario: dict) -> int:
    return int(scenario.get("routes_per_scenario", config["routes_per_scenario"]))


def route_spec(path: Path) -> dict:
    contract, errors = route_xml_contract(path)
    if errors or contract is None:
        raise ValueError(f"invalid frozen route {path}: {'; '.join(errors)}")
    return {key: value for key, value in contract.items() if key != "route_sha256"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    route_root = Path(config["source_route_root"]).resolve()
    rows = []
    group_index = 0
    route_index_start = int(config.get("route_index_start", 1))
    for scenario_id, scenario in config["scenarios"].items():
        family = scenario["family"]
        routes = sorted((route_root / family).glob("*.xml"))
        expected = scenario_route_count(config, scenario)
        if len(routes) != expected:
            raise RuntimeError(
                f"{scenario_id}/{family}: expected {expected} frozen routes, found {len(routes)}"
            )
        replacements = scenario.get("route_replacements", {})
        routes = [Path(replacements.get(route.name, route)).resolve() for route in routes]
        missing = [str(route) for route in routes if not route.is_file()]
        if missing:
            raise RuntimeError(f"{scenario_id}/{family}: missing replacement routes: {missing}")
        if len({str(route) for route in routes}) != expected:
            raise RuntimeError(f"{scenario_id}/{family}: replacement routes are not unique")
        for route_index, route in enumerate(routes, route_index_start):
            spec = route_spec(route)
            worker_slot = group_index % len(config["workers"])
            for decision_index, decision in enumerate(scenario["decisions"]):
                job_id = f"{scenario_id}_{route_index:02d}_{decision}"
                output_dir = (
                    Path(config["output_root"])
                    / scenario_id
                    / spec["route_id"]
                    / decision
                )
                rows.append(
                    {
                        "schema_version": config["schema_version"],
                        "job_id": job_id,
                        "group_id": f"{scenario_id}_{route_index:02d}",
                        "group_index": group_index,
                        "scenario_id": scenario_id,
                        "scenario_name": scenario["name"],
                        "family": family,
                        "route_index": route_index,
                        "decision_index": decision_index,
                        "decision": decision,
                        "route_path": str(route.resolve()),
                        "route_sha256": sha256(route),
                        "output_dir": str(output_dir.resolve()),
                        "worker_slot": worker_slot,
                        **spec,
                    }
                )
            group_index += 1

    expected_jobs = sum(
        scenario_route_count(config, scenario) * len(scenario["decisions"])
        for scenario in config["scenarios"].values()
    )
    if len(rows) != expected_jobs:
        raise RuntimeError(f"expected {expected_jobs} jobs, built {len(rows)}")
    duplicate_jobs = len(rows) - len({row["job_id"] for row in rows})
    duplicate_outputs = len(rows) - len({row["output_dir"] for row in rows})
    if duplicate_jobs or duplicate_outputs:
        raise RuntimeError(
            f"manifest uniqueness failed: duplicate_jobs={duplicate_jobs}, "
            f"duplicate_outputs={duplicate_outputs}"
        )
    for scenario_id in config["scenarios"]:
        scene_rows = [row for row in rows if row["scenario_id"] == scenario_id]
        unique_routes = {
            (row["route_id"], row["route_sha256"]) for row in scene_rows
        }
        expected = scenario_route_count(config, config["scenarios"][scenario_id])
        if len(unique_routes) != expected:
            raise RuntimeError(
                f"{scenario_id}: route identity/hash pairs are not unique"
            )
    invalid = {
        row["job_id"]: errors
        for row in rows
        if (errors := validate_manifest_job(row))
    }
    if invalid:
        raise RuntimeError(f"built manifest failed contract validation: {invalid}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "config": str(config_path),
        "manifest": str(args.output.resolve()),
        "config_sha256": sha256(config_path),
        "manifest_sha256": sha256(args.output),
        "groups": group_index,
        "jobs": len(rows),
        "jobs_by_scenario": {
            scenario_id: sum(row["scenario_id"] == scenario_id for row in rows)
            for scenario_id in config["scenarios"]
        },
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
