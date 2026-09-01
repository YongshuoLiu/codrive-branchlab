#!/usr/bin/env python3
"""Select accepted branch attempts that satisfy counterfactual onset alignment."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

from counterfactual_contract import validate_run_contract

from validate_counterfactual_campaign import (
    distance,
    first_active_telemetry,
    read_json,
    read_jsonl,
    scenario_actor_signature,
    yaw_distance,
)


RL_ROOT = Path(__file__).resolve().parents[1]


def load_candidate(path: Path, decision: str, job: dict) -> dict | None:
    quality = read_json(path / "quality_report.json", {}) or {}
    if quality.get("accepted") is not True:
        return None
    if validate_run_contract(path, job):
        return None
    onset = next(
        (
            row
            for row in read_jsonl(path / "raw/decision_events.jsonl")
            if row.get("event") == "decision_started"
        ),
        None,
    )
    if not onset or onset.get("decision") != decision:
        return None
    telemetry = first_active_telemetry(path / "raw/carla_telemetry.jsonl.gz")
    pose = onset.get("entry_pose", {})
    try:
        location = pose["location"]
        yaw = float(pose["rotation"]["yaw"])
        speed = float(onset["entry_speed_mps"])
        for key in ("x", "y", "z"):
            float(location[key])
    except (KeyError, TypeError, ValueError):
        return None
    return {
        "path": path,
        "decision": decision,
        "location": location,
        "yaw": yaw,
        "speed_mps": speed,
        "actors": scenario_actor_signature(telemetry),
        "outcome": quality.get("outcome"),
    }


def compare(reference: dict, candidate: dict) -> dict | None:
    if set(reference["actors"]) != set(candidate["actors"]):
        return None
    actor_deviations = [
        distance(reference["actors"][key], candidate["actors"][key])
        for key in reference["actors"]
    ]
    return {
        "position_m": distance(reference["location"], candidate["location"]),
        "yaw_deg": yaw_distance(reference["yaw"], candidate["yaw"]),
        "speed_mps": abs(reference["speed_mps"] - candidate["speed_mps"]),
        "actor_m": max(actor_deviations, default=0.0),
    }


def normalized_score(metrics: dict, thresholds: dict) -> tuple[float, float]:
    ratios = [
        metrics["position_m"] / thresholds["position_m"],
        metrics["yaw_deg"] / thresholds["yaw_deg"],
        metrics["speed_mps"] / thresholds["speed_mps"],
        metrics["actor_m"] / thresholds["actor_m"],
    ]
    return max(ratios), sum(ratios)


def serializable(candidate: dict, metrics: dict) -> dict:
    return {
        "path": str(candidate["path"]),
        "decision": candidate["decision"],
        "entry_speed_mps": candidate["speed_mps"],
        "outcome": candidate["outcome"],
        "alignment_to_maintain": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--group", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    config = read_json(args.config, {}) or {}
    quality = config.get("quality_thresholds", {})
    thresholds = {
        "position_m": float(quality.get("onset_position_tolerance_m", 0.5)),
        "yaw_deg": float(quality.get("onset_yaw_tolerance_deg", 1.0)),
        "speed_mps": float(quality.get("onset_speed_tolerance_mps", 0.75)),
        "actor_m": float(quality.get("onset_hazard_position_tolerance_m", 0.75)),
    }
    wanted = set(args.group)
    jobs_by_group: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(args.manifest):
        if row.get("group_id") in wanted:
            jobs_by_group[row["group_id"]].append(row)
    missing = sorted(wanted - set(jobs_by_group))
    if missing:
        raise SystemExit(f"groups absent from manifest: {missing}")

    plans = []
    unresolved = []
    internal_plans: dict[str, dict[str, dict]] = {}
    for group_id in sorted(jobs_by_group):
        candidates_by_decision: dict[str, list[dict]] = {}
        official_by_decision: dict[str, Path] = {}
        for job in jobs_by_group[group_id]:
            decision = job["decision"]
            official = Path(job["output_dir"]).resolve()
            official_by_decision[decision] = official
            attempts = [official, *sorted(official.parent.glob(f"{decision}_incomplete_*"))]
            candidates_by_decision[decision] = [
                candidate
                for path in attempts
                if (candidate := load_candidate(path, decision, job)) is not None
            ]

        best = None
        reference_diagnostics = []
        for reference in candidates_by_decision.get("Maintain", []):
            selected = {"Maintain": reference}
            selected_metrics = {
                "Maintain": {
                    "position_m": 0.0,
                    "yaw_deg": 0.0,
                    "speed_mps": 0.0,
                    "actor_m": 0.0,
                }
            }
            scores = []
            feasible = True
            compatibility = {"Maintain": 1}
            for decision in sorted(candidates_by_decision):
                if decision == "Maintain":
                    continue
                options = []
                for candidate in candidates_by_decision[decision]:
                    metrics = compare(reference, candidate)
                    if metrics is None or any(
                        metrics[key] > thresholds[key] for key in thresholds
                    ):
                        continue
                    options.append((normalized_score(metrics, thresholds), candidate, metrics))
                compatibility[decision] = len(options)
                if not options:
                    feasible = False
                    continue
                score, candidate, metrics = min(options, key=lambda item: item[0])
                selected[decision] = candidate
                selected_metrics[decision] = metrics
                scores.append(score)
            reference_diagnostics.append(
                {
                    "maintain_path": str(reference["path"]),
                    "maintain_speed_mps": reference["speed_mps"],
                    "compatible_candidates": compatibility,
                    "feasible": feasible,
                }
            )
            if not feasible:
                continue
            group_score = (
                max((score[0] for score in scores), default=0.0),
                sum(score[1] for score in scores),
                sum(
                    selected[decision]["path"] != official_by_decision[decision]
                    for decision in selected
                ),
            )
            if best is None or group_score < best[0]:
                best = (group_score, selected, selected_metrics)

        candidate_counts = {
            decision: len(candidates) for decision, candidates in candidates_by_decision.items()
        }
        if best is None:
            unresolved.append(group_id)
            plans.append(
                {
                    "group_id": group_id,
                    "resolved": False,
                    "candidate_counts": candidate_counts,
                    "maintain_reference_diagnostics": reference_diagnostics,
                }
            )
            continue
        score, selected, selected_metrics = best
        internal_plans[group_id] = selected
        plans.append(
            {
                "group_id": group_id,
                "resolved": True,
                "candidate_counts": candidate_counts,
                "maintain_reference_diagnostics": reference_diagnostics,
                "score": list(score),
                "selections": [
                    serializable(selected[decision], selected_metrics[decision])
                    for decision in sorted(selected)
                ],
            }
        )

    applied = []
    if args.apply:
        if unresolved:
            raise SystemExit(
                "refusing partial apply; unresolved groups: " + ", ".join(unresolved)
            )
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        for group_id in sorted(internal_plans):
            official_by_decision = {
                job["decision"]: Path(job["output_dir"]).resolve()
                for job in jobs_by_group[group_id]
            }
            for decision, candidate in sorted(internal_plans[group_id].items()):
                source = candidate["path"]
                destination = official_by_decision[decision]
                if source == destination:
                    continue
                displaced = destination.with_name(
                    f"{decision}_incomplete_{timestamp}_alignment_replaced"
                )
                counter = 1
                while displaced.exists():
                    displaced = destination.with_name(
                        f"{decision}_incomplete_{timestamp}_alignment_replaced_{counter:02d}"
                    )
                    counter += 1
                shutil.move(str(destination), str(displaced))
                shutil.move(str(source), str(destination))
                applied.append(
                    {
                        "group_id": group_id,
                        "decision": decision,
                        "selected_source": str(source),
                        "official_destination": str(destination),
                        "displaced_official": str(displaced),
                    }
                )

    report = {
        "schema_version": "counterfactual_alignment_selection_v1",
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": str(args.config.resolve()),
        "manifest": str(args.manifest.resolve()),
        "thresholds": thresholds,
        "requested_groups": sorted(wanted),
        "resolved_groups": len(plans) - len(unresolved),
        "unresolved_groups": unresolved,
        "applied": bool(args.apply),
        "applied_replacements": applied,
        "plans": plans,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "groups": len(plans),
                "resolved": len(plans) - len(unresolved),
                "unresolved": unresolved,
                "applied_replacements": len(applied),
                "report": str(args.output),
            },
            indent=2,
        )
    )
    return 0 if not unresolved else 2


if __name__ == "__main__":
    raise SystemExit(main())
