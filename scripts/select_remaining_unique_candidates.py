#!/usr/bin/env python3
"""Freeze candidate geometries that have not received multi-branch collection."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any


RL_ROOT = Path(__file__).resolve().parents[1]
FAMILY_TO_SLUG = {
    "S1": "occluded_pedestrian",
    "S2": "lane_cut_in",
    "S3": "reveal_obstacle",
    "S4": "ghost_driver_left_turn",
    "S5": "ghost_driver_right_turn",
    "S6": "right_turn_on_red",
}


def route_geometry_hashes(root: Path, slug: str) -> set[str]:
    paths = list((root / slug).glob("*.xml")) if (root / slug).is_dir() else []
    hashes: set[str] = set()
    for path in paths:
        try:
            route = ET.parse(path).getroot().find("route")
        except (ET.ParseError, OSError):
            continue
        if route is not None and route.get("candidate_geometry_sha256"):
            hashes.add(str(route.get("candidate_geometry_sha256")))
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--exclude-route-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pool_path = args.candidate_pool.resolve()
    payload: dict[str, Any] = json.loads(pool_path.read_text(encoding="utf-8"))
    selected: dict[str, list[dict[str, Any]]] = {}
    audit: dict[str, dict[str, int]] = {}
    all_selected_hashes: set[tuple[str, str]] = set()

    for family, slug in FAMILY_TO_SLUG.items():
        used_hashes: set[str] = set()
        for root in args.exclude_route_root:
            used_hashes.update(route_geometry_hashes(root.resolve(), slug))

        representatives: dict[str, dict[str, Any]] = {}
        for candidate in payload.get("candidates", {}).get(family, []):
            geometry_hash = str(candidate["geometry_sha256"])
            representatives.setdefault(geometry_hash, candidate)
        remaining = [
            candidate
            for geometry_hash, candidate in representatives.items()
            if geometry_hash not in used_hashes
        ]
        remaining.sort(key=lambda row: str(row["candidate_id"]))
        selected[family] = remaining
        all_selected_hashes.update(
            (family, str(candidate["geometry_sha256"])) for candidate in remaining
        )
        audit[family] = {
            "pool_records": len(payload.get("candidates", {}).get(family, [])),
            "pool_unique_geometries": len(representatives),
            "collected_geometries": len(set(representatives) & used_hashes),
            "remaining_unique_geometries": len(remaining),
        }

    if sum(len(rows) for rows in selected.values()) != len(all_selected_hashes):
        raise RuntimeError("family/geometry selection is not unique")

    result = {
        "version": "counterfactual_remaining_unique_candidates_v1",
        "candidate_source": str(pool_path),
        "excluded_route_roots": [str(path.resolve()) for path in args.exclude_route_root],
        "family_counts": {family: len(rows) for family, rows in selected.items()},
        "total_unique_geometries": sum(len(rows) for rows in selected.values()),
        "audit": audit,
        "candidates": selected,
        "selection_rejections": [],
    }
    output = args.output.resolve()
    if RL_ROOT.resolve() not in output.parents:
        raise SystemExit(f"output must stay under {RL_ROOT.resolve()}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "family_counts": result["family_counts"],
                "total_unique_geometries": result["total_unique_geometries"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
