#!/usr/bin/env python3
"""Audit Decision Reward v2 coverage and training-label consistency."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import pathlib
from typing import Any

from annotate_counterfactual_decision_reward_v2 import (
    DECISION_LABELS,
    EXPECTED_DECISIONS,
    REWARD_VERSION,
    RL_ROOT,
    atomic_write_json,
    discover_groups,
    read_json,
)


DEFAULT_ROOT = RL_ROOT / "data/counterfactual_decision_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=pathlib.Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args()


def issue(kind: str, path: pathlib.Path, detail: str) -> dict[str, str]:
    return {"kind": kind, "path": str(path.resolve()), "detail": detail}


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    output = args.output or data_root / "_annotations/decision_reward_v2/audit.json"
    groups, physical = discover_groups(data_root, set(EXPECTED_DECISIONS), None)
    active_paths = set()
    for group in groups:
        if not group["active_for_training"]:
            continue
        for path in group["paths"].values():
            active_paths.add(str(path.resolve()))

    issues = []
    tier_counts: collections.Counter[str] = collections.Counter()
    scenario_counts: collections.Counter[str] = collections.Counter()
    mapping_counts: collections.Counter[str] = collections.Counter()
    active_count = 0
    superseded_count = 0
    quality_valid = 0
    scored = 0
    candidates = 0
    missing_run_specs = 0
    accepted_compute_errors = 0
    invalid_compute_warnings = 0

    for raw_path, (scenario_id, decision) in sorted(physical.items()):
        path = pathlib.Path(raw_path)
        quality_path = path / "quality_report.json"
        quality = read_json(quality_path)
        label = quality.get("decision_label") or {}
        reward = quality.get("decision_reward") or {}
        expected_longitudinal, expected_lateral = DECISION_LABELS[decision]
        scenario_counts[scenario_id] += 1
        mapping_counts[f"{expected_longitudinal}+{expected_lateral}"] += 1

        if not label:
            issues.append(issue("missing_decision_label", quality_path, decision))
            continue
        if label.get("branch_decision") != decision:
            issues.append(issue("branch_decision_mismatch", quality_path, str(label)))
        if label.get("longitudinal") != expected_longitudinal:
            issues.append(issue("longitudinal_mismatch", quality_path, str(label)))
        if label.get("lateral") != expected_lateral:
            issues.append(issue("lateral_mismatch", quality_path, str(label)))
        if quality.get("decision_longitudinal") != expected_longitudinal:
            issues.append(issue("flat_longitudinal_mismatch", quality_path, decision))
        if quality.get("decision_lateral") != expected_lateral:
            issues.append(issue("flat_lateral_mismatch", quality_path, decision))

        if not reward:
            issues.append(issue("missing_decision_reward", quality_path, decision))
            continue
        if reward.get("reward_version") != REWARD_VERSION:
            issues.append(issue("reward_version_mismatch", quality_path, str(reward.get("reward_version"))))
        active = bool(reward.get("active_for_training"))
        expected_active = raw_path in active_paths
        if active != expected_active:
            issues.append(issue("active_flag_mismatch", quality_path, f"actual={active} expected={expected_active}"))
        active_count += int(active)
        superseded_count += int(not active)
        quality_valid += int(bool(reward.get("quality_valid")))
        candidates += int(bool(reward.get("candidate_for_decision_ranking")))
        tier = str(reward.get("hard_safety_tier"))
        tier_counts[tier] += 1
        score = reward.get("final_reward")
        if score is not None:
            scored += 1
            if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
                issues.append(issue("nonfinite_reward", quality_path, str(score)))
            elif not -100.0 <= float(score) <= 100.0:
                issues.append(issue("reward_out_of_range", quality_path, str(score)))
        elif reward.get("quality_valid"):
            issues.append(issue("valid_branch_missing_reward", quality_path, tier))

        compute_error = reward.get("compute_error")
        if compute_error:
            if quality.get("accepted") is True:
                accepted_compute_errors += 1
                issues.append(issue("accepted_branch_compute_error", quality_path, str(compute_error)))
            else:
                invalid_compute_warnings += 1

        run_spec = path / "metadata/run_spec.json"
        if not run_spec.is_file():
            missing_run_specs += 1
        else:
            spec = read_json(run_spec)
            if (spec.get("decision_label") or {}).get("longitudinal") != expected_longitudinal:
                issues.append(issue("run_spec_longitudinal_mismatch", run_spec, decision))
            if (spec.get("decision_label") or {}).get("lateral") != expected_lateral:
                issues.append(issue("run_spec_lateral_mismatch", run_spec, decision))

    incomplete_modified = []
    expected_names = {decision for values in EXPECTED_DECISIONS.values() for decision in values}
    canonical_path_set = set(physical)
    for quality_path in data_root.rglob("quality_report.json"):
        if str(quality_path.parent.resolve()) in canonical_path_set:
            continue
        if quality_path.parent.name in expected_names:
            # A canonical-looking branch outside the selected production roots
            # is not a historical incomplete directory.
            continue
        quality = read_json(quality_path)
        if quality.get("decision_reward_version") == REWARD_VERSION:
            incomplete_modified.append(str(quality_path.resolve()))

    group_index_path = data_root / "_annotations/decision_reward_v2/group_index.jsonl"
    group_rows = []
    if group_index_path.is_file():
        with group_index_path.open("r", encoding="utf-8") as stream:
            group_rows = [json.loads(line) for line in stream if line.strip()]
    active_groups = [row for row in group_rows if row.get("active_for_training")]
    valid_label_groups = [row for row in active_groups if row.get("label_valid")]
    for row in valid_label_groups:
        total = sum(float(value) for value in (row.get("branch_target_probabilities") or {}).values())
        if abs(total - 1.0) > 1e-5:
            issues.append(issue("group_probability_sum", group_index_path, f"{row.get('group_id')} sum={total}"))
        lon_total = sum(float(value) for value in (row.get("longitudinal_target_distribution") or {}).values())
        lat_total = sum(float(value) for value in (row.get("lateral_target_distribution") or {}).values())
        if abs(lon_total - 1.0) > 1e-5 or abs(lat_total - 1.0) > 1e-5:
            issues.append(
                issue(
                    "factorized_probability_sum",
                    group_index_path,
                    f"{row.get('group_id')} lon={lon_total} lat={lat_total}",
                )
            )

    audit = {
        "schema_version": "decision_reward_v2_audit",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "data_root": str(data_root),
        "reward_version": REWARD_VERSION,
        "status": "pass" if not issues and accepted_compute_errors == 0 else "fail",
        "physical_canonical_branches": len(physical),
        "active_branches": active_count,
        "superseded_branches": superseded_count,
        "quality_valid_branches": quality_valid,
        "scored_branches": scored,
        "candidate_branches": candidates,
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "factorized_mapping_counts": dict(sorted(mapping_counts.items())),
        "hard_safety_tier_counts": dict(sorted(tier_counts.items())),
        "group_rows": len(group_rows),
        "active_groups": len(active_groups),
        "valid_label_groups": len(valid_label_groups),
        "groups_without_safe_label": len(active_groups) - len(valid_label_groups),
        "accepted_branch_compute_errors": accepted_compute_errors,
        "invalid_branch_compute_warnings": invalid_compute_warnings,
        "missing_run_spec_warnings": missing_run_specs,
        "historical_incomplete_directories_modified": len(incomplete_modified),
        "issue_count": len(issues),
        "issues": issues,
        "historical_incomplete_modified_paths": incomplete_modified,
    }
    atomic_write_json(output.resolve(), audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
