#!/usr/bin/env python3
"""Audit legacy versus S1_PED_SPEED_V2 Accelerate/Maintain branches."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


RL_ROOT = Path(__file__).resolve().parents[1]
DECISIONS = ("Accelerate", "Maintain")
TARGET_POLICY = "S1_PED_SPEED_V2"
TARGET_SPEED_MPS = 8.5


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}


def branch_state(path: Path) -> dict:
    quality = read_json(path / "quality_report.json")
    marker = read_json(path / "metadata/s1_policy_marker.json")
    collision_counts = quality.get("collision_counts") or {}
    pedestrian_collision = int(collision_counts.get("collisions_pedestrian", 0)) > 0
    return {
        "exists": path.is_dir(),
        "accepted": quality.get("accepted") is True,
        "outcome": quality.get("outcome"),
        "policy_version": marker.get("policy_version"),
        "ped_speed_mps": marker.get("ped_speed_mps"),
        "is_recollected_v2": (
            marker.get("policy_version") == TARGET_POLICY
            and isinstance(marker.get("ped_speed_mps"), (int, float))
            and float(marker["ped_speed_mps"]) >= TARGET_SPEED_MPS
        ),
        "pedestrian_collision": pedestrian_collision,
        "expected_outcome": quality.get("outcome") == "collision" and pedestrian_collision,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-root", type=Path,
        default=RL_ROOT / "data/counterfactual_decision_v1/S1",
    )
    parser.add_argument(
        "--v2-root", type=Path,
        default=RL_ROOT / "data/counterfactual_decision_v1/_recollected/S1_PED_SPEED_V2/S1",
    )
    parser.add_argument(
        "--output", type=Path,
        default=RL_ROOT / "data/counterfactual_decision_v1/_recollected/S1_PED_SPEED_V2/recollection_status.json",
    )
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    route_ids = sorted(
        {path.name for path in args.baseline_root.glob("closed_occluded_pedestrian_*") if path.is_dir()}
        | {path.name for path in args.v2_root.glob("closed_occluded_pedestrian_*") if path.is_dir()}
    )
    rows = []
    for route_id in route_ids:
        decisions = {}
        pending = []
        for decision in DECISIONS:
            legacy = branch_state(args.baseline_root / route_id / decision)
            v2 = branch_state(args.v2_root / route_id / decision)
            complete = v2["accepted"] and v2["is_recollected_v2"] and v2["expected_outcome"]
            if not complete:
                pending.append(decision)
            decisions[decision] = {"legacy": legacy, "v2": v2, "complete": complete}
        rows.append(
            {
                "route_id": route_id,
                "status": "complete" if not pending else "partial" if len(pending) == 1 else "pending",
                "needs_recollection": pending,
                "decisions": decisions,
            }
        )

    summary = {
        "schema_version": "s1_ped_speed_v2_recollection_status_v1",
        "target_policy_version": TARGET_POLICY,
        "target_ped_speed_mps": TARGET_SPEED_MPS,
        "route_count": len(rows),
        "complete_routes": sum(row["status"] == "complete" for row in rows),
        "partial_routes": sum(row["status"] == "partial" for row in rows),
        "pending_routes": sum(row["status"] == "pending" for row in rows),
        "complete_branches": sum(item["complete"] for row in rows for item in row["decisions"].values()),
        "required_branches": len(rows) * len(DECISIONS),
        "unexpected_safe_v2_branches": sum(
            item["v2"]["accepted"]
            and item["v2"]["is_recollected_v2"]
            and not item["v2"]["expected_outcome"]
            for row in rows
            for item in row["decisions"].values()
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = args.csv or args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["route_id", "status", "needs_recollection", "accelerate_v2", "maintain_v2"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "route_id": row["route_id"],
                    "status": row["status"],
                    "needs_recollection": "|".join(row["needs_recollection"]),
                    "accelerate_v2": row["decisions"]["Accelerate"]["complete"],
                    "maintain_v2": row["decisions"]["Maintain"]["complete"],
                }
            )
    print(json.dumps({key: summary[key] for key in ("route_count", "complete_routes", "partial_routes", "pending_routes", "complete_branches", "required_branches", "unexpected_safe_v2_branches")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
