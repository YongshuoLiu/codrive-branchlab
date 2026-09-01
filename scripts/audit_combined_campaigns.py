#!/usr/bin/env python3
"""Audit the official branches referenced by all decision-v1 campaign manifests."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


RL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGNS = (
    (
        "routes_0001_0010",
        RL_ROOT / "manifests/collection_manifest.jsonl",
        RL_ROOT / "data/counterfactual_decision_v1/routes_0001_0010_campaign_quality_report.json",
    ),
    (
        "routes_0011_0015",
        RL_ROOT / "manifests/pilot_0011_0015_manifest.jsonl",
        RL_ROOT / "data/counterfactual_decision_v1/pilot_0011_0015_campaign_quality_report.json",
    ),
    (
        "routes_0016_0020",
        RL_ROOT / "manifests/pilot_0016_0020_manifest.jsonl",
        RL_ROOT / "data/counterfactual_decision_v1/pilot_0016_0020_campaign_quality_report.json",
    ),
)
DEFAULT_OUTPUT = RL_ROOT / "data/counterfactual_decision_v1/routes_0001_0020_combined_audit.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def directory_bytes(path: Path) -> int:
    total = 0
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            candidate = Path(root) / filename
            if candidate.is_file() and not candidate.is_symlink():
                total += candidate.stat().st_size
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    all_jobs: list[dict[str, Any]] = []
    campaign_rows: list[dict[str, Any]] = []
    alignment_maxima = {
        "maximum_position_deviation_m": 0.0,
        "maximum_yaw_deviation_deg": 0.0,
        "maximum_speed_deviation_mps": 0.0,
        "maximum_scenario_actor_deviation_m": 0.0,
    }

    for label, manifest_path, report_path in DEFAULT_CAMPAIGNS:
        jobs = read_jsonl(manifest_path)
        report = read_json(report_path)
        if report.get("accepted") is not True:
            errors.append(f"{label}: campaign quality report is not accepted")
        if int(report.get("accepted_branches", -1)) != len(jobs):
            errors.append(
                f"{label}: report accepted_branches={report.get('accepted_branches')} "
                f"but manifest has {len(jobs)} jobs"
            )
        for group in report.get("group_reports", []):
            alignment = group.get("onset_alignment") or {}
            for key in alignment_maxima:
                value = alignment.get(key)
                if value is not None:
                    alignment_maxima[key] = max(alignment_maxima[key], float(value))
        campaign_rows.append(
            {
                "label": label,
                "manifest": str(manifest_path.resolve()),
                "quality_report": str(report_path.resolve()),
                "accepted": report.get("accepted") is True,
                "groups": len({job["group_id"] for job in jobs}),
                "branches": len(jobs),
                "outcomes": report.get("outcomes", {}),
            }
        )
        all_jobs.extend(jobs)

    job_ids = [job["job_id"] for job in all_jobs]
    output_dirs = [str(Path(job["output_dir"]).resolve()) for job in all_jobs]
    if len(set(job_ids)) != len(job_ids):
        errors.append("combined manifests contain duplicate job ids")
    if len(set(output_dirs)) != len(output_dirs):
        errors.append("combined manifests contain duplicate output directories")

    outcomes: Counter[str] = Counter()
    modality_totals: Counter[str] = Counter()
    telemetry_frames = 0
    official_bytes = 0
    accepted_branches = 0
    branch_warning_count = 0
    branch_warning_messages: Counter[str] = Counter()
    scene_rows: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "branches": 0,
            "groups": set(),
            "route_ids": set(),
            "route_sha256": set(),
            "decisions": Counter(),
            "outcomes": Counter(),
        }
    )

    for job in all_jobs:
        run_dir = Path(job["output_dir"])
        report_path = run_dir / "quality_report.json"
        if not report_path.is_file():
            errors.append(f"{job['job_id']}: missing quality_report.json")
            continue
        quality = read_json(report_path)
        if quality.get("accepted") is not True:
            errors.append(f"{job['job_id']}: official branch is not accepted")
        else:
            accepted_branches += 1
        outcome = str(quality.get("outcome", "missing"))
        outcomes[outcome] += 1
        telemetry_frames += int(quality.get("telemetry_frames") or 0)
        modality_totals.update(
            {
                key: int(value or 0)
                for key, value in (quality.get("modality_counts") or {}).items()
                if not key.endswith("_augmented")
            }
        )
        quality_warnings = [str(item) for item in (quality.get("warnings") or [])]
        branch_warning_count += len(quality_warnings)
        branch_warning_messages.update(quality_warnings)
        official_bytes += directory_bytes(run_dir)

        scene = scene_rows[job["scenario_id"]]
        scene["branches"] += 1
        scene["groups"].add(job["group_id"])
        scene["route_ids"].add(job["route_id"])
        scene["route_sha256"].add(job["route_sha256"])
        scene["decisions"][job["decision"]] += 1
        scene["outcomes"][outcome] += 1

    serialized_scenes = {}
    for scene_id in sorted(scene_rows):
        row = scene_rows[scene_id]
        if len(row["route_ids"]) != len(row["groups"]):
            errors.append(f"{scene_id}: route ids are not unique across groups")
        if len(row["route_sha256"]) != len(row["groups"]):
            errors.append(f"{scene_id}: frozen route hashes are not unique across groups")
        serialized_scenes[scene_id] = {
            "groups": len(row["groups"]),
            "unique_route_ids": len(row["route_ids"]),
            "unique_route_sha256": len(row["route_sha256"]),
            "branches": row["branches"],
            "decisions": dict(sorted(row["decisions"].items())),
            "outcomes": dict(sorted(row["outcomes"].items())),
        }

    sensor_modalities = (
        "rgb",
        "depth",
        "semantics",
        "bev_semantics",
        "lidar",
        "boxes",
        "measurements",
    )
    sensor_counts = {name: modality_totals.get(name, 0) for name in sensor_modalities}
    if len(set(sensor_counts.values())) != 1:
        errors.append(f"combined synchronized modality totals differ: {sensor_counts}")

    payload = {
        "schema_version": "counterfactual_decision_v1_combined_audit",
        "accepted": not errors,
        "campaigns": campaign_rows,
        "totals": {
            "groups": len({job["group_id"] for job in all_jobs}),
            "branches": len(all_jobs),
            "accepted_branches": accepted_branches,
            "unique_output_directories": len(set(output_dirs)),
            "official_bytes": official_bytes,
            "official_gib": round(official_bytes / (1024 ** 3), 3),
            "telemetry_frames": telemetry_frames,
            "synchronized_sensor_frames": sensor_counts.get("measurements", 0),
            "modality_totals": sensor_counts,
            "branch_warning_count": branch_warning_count,
            "branch_warning_summary": dict(sorted(branch_warning_messages.items())),
            "outcomes": dict(sorted(outcomes.items())),
        },
        "alignment_maxima": alignment_maxima,
        "scenes": serialized_scenes,
        "errors": errors,
        "warnings": warnings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "accepted": payload["accepted"],
                "groups": payload["totals"]["groups"],
                "branches": payload["totals"]["branches"],
                "accepted_branches": payload["totals"]["accepted_branches"],
                "official_gib": payload["totals"]["official_gib"],
                "errors": errors,
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0 if payload["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
