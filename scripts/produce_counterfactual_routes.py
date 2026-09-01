#!/usr/bin/env python3
"""Safe staged controller for repeatable multi-decision route production."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from counterfactual_contract import read_json, sha256, validate_run_contract


RL_ROOT = Path(__file__).resolve().parents[1]
BUILD = RL_ROOT / "scripts/build_collection_manifest.py"
PREFLIGHT = RL_ROOT / "scripts/preflight_counterfactual_routes.py"
CONTRACT = RL_ROOT / "scripts/counterfactual_contract.py"
COLLECT = RL_ROOT / "scripts/collect_counterfactual_v1.py"
VALIDATE = RL_ROOT / "scripts/validate_counterfactual_campaign.py"
SELECT = RL_ROOT / "scripts/select_alignment_candidates.py"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def branch_is_cached(job: dict[str, Any]) -> bool:
    run_dir = Path(job["output_dir"])
    quality = read_json(run_dir / "quality_report.json", {}) or {}
    return quality.get("accepted") is True and not validate_run_contract(run_dir, job)


def write_status(path: Path, status: dict[str, Any]) -> None:
    status["updated_at_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def run(command: list[str], status: dict[str, Any], stage: str) -> int:
    print(f"\n[production:{stage}] {' '.join(command)}", flush=True)
    started = time.time()
    status["current_stage"] = stage
    status["current_stage_started_at_epoch"] = started
    status["current_command"] = command
    write_status(Path(status["status_path"]), status)
    returncode = subprocess.run(command, start_new_session=True).returncode
    status["stages"].append(
        {
            "stage": stage,
            "returncode": returncode,
            "elapsed_s": round(time.time() - started, 2),
            "command": command,
        }
    )
    status["current_stage"] = "between_stages"
    status.pop("current_command", None)
    write_status(Path(status["status_path"]), status)
    return returncode


def collection_command(args: argparse.Namespace, production_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(COLLECT),
        "--config",
        str(args.config),
        "--manifest",
        str(args.manifest),
        "--workers",
        args.workers,
        "--walltime",
        str(args.walltime),
        "--technical-retries",
        str(args.technical_retries),
        "--retry-backoff",
        str(args.retry_backoff),
        "--status-output",
        str(production_dir / "latest_collection_status.json"),
        "--progress-output",
        str(production_dir / "runtime_progress.json"),
    ]
    for decision in args.decision or []:
        command.extend(["--decision", decision])
    for decision in args.exclude_decision or []:
        command.extend(["--exclude-decision", decision])
    if args.skip_existing_results:
        command.append("--skip-existing-results")
    return command


def validate_campaign(
    args: argparse.Namespace,
    production_dir: Path,
    status: dict[str, Any],
    label: str,
) -> tuple[int, dict[str, Any]]:
    report = production_dir / f"campaign_{label}.json"
    table = production_dir / f"campaign_{label}.csv"
    command = [
        sys.executable,
        str(VALIDATE),
        "--config",
        str(args.config),
        "--manifest",
        str(args.manifest),
        "--output",
        str(report),
        "--csv",
        str(table),
    ]
    returncode = run(command, status, f"validate_{label}")
    return returncode, read_json(report, {}) or {}


def repair_branch_failures(
    args: argparse.Namespace,
    jobs: list[dict[str, Any]],
    production_dir: Path,
    status: dict[str, Any],
) -> None:
    for round_index in range(1, args.max_branch_repair_rounds + 1):
        failed = [job["job_id"] for job in jobs if not branch_is_cached(job)]
        if not failed:
            return
        command = collection_command(args, production_dir)
        for job_id in failed:
            command.extend(["--job-id", job_id])
        command.append("--rerun")
        run(command, status, f"branch_repair_{round_index}")
    failed = [job["job_id"] for job in jobs if not branch_is_cached(job)]
    if failed:
        raise RuntimeError(
            f"{len(failed)} branches remain rejected after repair: {', '.join(failed[:20])}"
        )


def unresolved_rerun_jobs(
    selection: dict[str, Any],
    jobs_by_group: dict[str, list[dict[str, Any]]],
    round_index: int,
) -> list[str]:
    targets: list[str] = []
    plans = {plan["group_id"]: plan for plan in selection.get("plans", [])}
    for group_id in selection.get("unresolved_groups", []):
        plan = plans.get(group_id, {})
        diagnostics = plan.get("maintain_reference_diagnostics", [])
        decisions: set[str]
        if diagnostics:
            best = max(
                diagnostics,
                key=lambda item: (
                    sum(count > 0 for count in item.get("compatible_candidates", {}).values()),
                    sum(item.get("compatible_candidates", {}).values()),
                ),
            )
            compatibility = best.get("compatible_candidates", {})
            decisions = {
                decision for decision, count in compatibility.items() if int(count) == 0
            }
            if not decisions:
                decisions = {"Maintain"}
            # If several decisions have already accumulated candidates but all
            # remain incompatible with one Maintain speed cluster, refreshing
            # the reference is cheaper than repeatedly sampling every branch.
            if round_index >= 2 and len(decisions) >= 2:
                decisions.add("Maintain")
        else:
            decisions = {"Maintain"}
        for job in jobs_by_group[group_id]:
            if job["decision"] in decisions:
                targets.append(job["job_id"])
    return sorted(set(targets))


def repair_alignment(
    args: argparse.Namespace,
    jobs: list[dict[str, Any]],
    campaign: dict[str, Any],
    production_dir: Path,
    status: dict[str, Any],
) -> dict[str, Any]:
    jobs_by_group: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        jobs_by_group.setdefault(job["group_id"], []).append(job)
    current_campaign = campaign
    for round_index in range(1, args.max_alignment_rounds + 1):
        failed_groups = sorted(
            report["group_id"]
            for report in current_campaign.get("group_reports", [])
            if not report.get("accepted")
        )
        if not failed_groups:
            return current_campaign
        selection_path = production_dir / f"alignment_round_{round_index}.json"
        select_command = [
            sys.executable,
            str(SELECT),
            "--config",
            str(args.config),
            "--manifest",
            str(args.manifest),
            "--output",
            str(selection_path),
        ]
        for group_id in failed_groups:
            select_command.extend(["--group", group_id])
        select_status = run(select_command, status, f"alignment_select_{round_index}")
        selection = read_json(selection_path, {}) or {}
        unresolved = selection.get("unresolved_groups", failed_groups)
        if select_status == 0 and not unresolved:
            apply_command = [*select_command, "--apply"]
            if run(apply_command, status, f"alignment_apply_{round_index}") != 0:
                raise RuntimeError("alignment selection resolved but apply failed")
        else:
            targets = unresolved_rerun_jobs(selection, jobs_by_group, round_index)
            if not targets:
                raise RuntimeError(
                    "alignment is unresolved but selector produced no targeted rerun jobs"
                )
            collect_command = collection_command(args, production_dir)
            for job_id in targets:
                collect_command.extend(["--job-id", job_id])
            collect_command.append("--rerun")
            run(collect_command, status, f"alignment_rerun_{round_index}")
            repair_branch_failures(args, jobs, production_dir, status)
        _, current_campaign = validate_campaign(
            args, production_dir, status, f"alignment_round_{round_index}"
        )

    # The last rerun can create the missing compatible candidate. Always give
    # the selector one final chance to promote that combination before failing
    # the campaign merely because the loop boundary was reached.
    failed_groups = sorted(
        report["group_id"]
        for report in current_campaign.get("group_reports", [])
        if not report.get("accepted")
    )
    if not failed_groups:
        return current_campaign
    selection_path = production_dir / "alignment_final_selection.json"
    select_command = [
        sys.executable,
        str(SELECT),
        "--config",
        str(args.config),
        "--manifest",
        str(args.manifest),
        "--output",
        str(selection_path),
    ]
    for group_id in failed_groups:
        select_command.extend(["--group", group_id])
    selection_status = run(select_command, status, "alignment_final_select")
    selection = read_json(selection_path, {}) or {}
    if selection_status == 0 and not selection.get("unresolved_groups"):
        if run([*select_command, "--apply"], status, "alignment_final_apply") != 0:
            raise RuntimeError("final alignment selection resolved but apply failed")
        _, current_campaign = validate_campaign(
            args, production_dir, status, "alignment_final_selection"
        )
    return current_campaign


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", default="0,1,2")
    parser.add_argument("--walltime", type=int, default=300)
    parser.add_argument("--technical-retries", type=int, default=2)
    parser.add_argument("--retry-backoff", type=float, default=15.0)
    parser.add_argument("--max-branch-repair-rounds", type=int, default=2)
    parser.add_argument("--max-alignment-rounds", type=int, default=5)
    parser.add_argument("--live-preflight-report", type=Path)
    parser.add_argument(
        "--allow-static-only",
        action="store_true",
        help="Allow new collection without a matching successful live CARLA preflight",
    )
    parser.add_argument("--skip-maintain-smoke", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--decision", action="append")
    parser.add_argument("--exclude-decision", action="append")
    parser.add_argument("--skip-existing-results", action="store_true")
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.manifest = args.manifest.resolve()

    if args.rebuild_manifest:
        subprocess.run(
            [
                sys.executable,
                str(BUILD),
                "--config",
                str(args.config),
                "--output",
                str(args.manifest),
            ],
            check=True,
        )
    config = read_json(args.config, {}) or {}
    output_root = Path(config["output_root"]).resolve()
    if RL_ROOT.resolve() not in output_root.parents:
        raise SystemExit(f"config output_root must stay under {RL_ROOT.resolve()}")
    jobs = read_jsonl(args.manifest)
    production_dir = output_root / "_production" / args.manifest.stem
    production_dir.mkdir(parents=True, exist_ok=True)
    status_path = production_dir / "production_status.json"
    status: dict[str, Any] = {
        "schema_version": "counterfactual_production_status_v1",
        "config": str(args.config),
        "manifest": str(args.manifest),
        "config_sha256": sha256(args.config),
        "manifest_sha256": sha256(args.manifest),
        "status_path": str(status_path),
        "started_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dry_run": args.dry_run,
        "stages": [],
        "accepted": False,
    }
    write_status(status_path, status)

    static_report = production_dir / "static_preflight.json"
    preflight_status = run(
        [
            sys.executable,
            str(PREFLIGHT),
            "--config",
            str(args.config),
            "--manifest",
            str(args.manifest),
            "--output",
            str(static_report),
        ],
        status,
        "static_preflight",
    )
    if preflight_status != 0:
        status["failure"] = "static_preflight_failed"
        write_status(status_path, status)
        return 2

    uncached = [job["job_id"] for job in jobs if not branch_is_cached(job)]
    live_verified = False
    if args.live_preflight_report:
        live = read_json(args.live_preflight_report.resolve(), {}) or {}
        live_verified = bool(
            live.get("accepted")
            and live.get("live_checked")
            and live.get("manifest_sha256") == status["manifest_sha256"]
            and live.get("config_sha256") == status["config_sha256"]
            and live.get("validator_sha256") == sha256(PREFLIGHT)
            and live.get("contract_validator_sha256") == sha256(CONTRACT)
        )
        if not live_verified:
            status["failure"] = "live_preflight_report_does_not_match_current_config_manifest"
            write_status(status_path, status)
            return 2
    status["live_preflight_verified"] = live_verified
    status["initial_cached_branches"] = len(jobs) - len(uncached)
    status["initial_uncached_branches"] = len(uncached)

    if args.dry_run:
        command = collection_command(args, production_dir)
        command.append("--dry-run")
        dry_status = run(command, status, "collection_dry_run")
        status["ready_for_collection"] = bool(
            dry_status == 0
            and (not uncached or live_verified or args.allow_static_only)
        )
        status["accepted"] = status["ready_for_collection"]
        if uncached and not live_verified and not args.allow_static_only:
            status["failure"] = "new_routes_require_matching_live_preflight"
        write_status(status_path, status)
        print(json.dumps({
            "ready_for_collection": status["ready_for_collection"],
            "cached": len(jobs) - len(uncached),
            "uncached": len(uncached),
            "live_preflight_verified": live_verified,
            "status": str(status_path),
        }, indent=2))
        return 0 if status["ready_for_collection"] else 2

    if uncached and not live_verified and not args.allow_static_only:
        status["failure"] = "new_routes_require_matching_live_preflight"
        write_status(status_path, status)
        print(
            "Refusing new collection without a matching accepted --live-preflight-report; "
            "use --allow-static-only only for an intentional supervised exception.",
            file=sys.stderr,
        )
        return 2

    try:
        if args.collect_only:
            collection_code = run(
                collection_command(args, production_dir), status, "collect_only"
            )
            collection_report_path = production_dir / "latest_collection_status.json"
            collection_report = read_json(collection_report_path, {}) or {}
            runtime_report = read_json(production_dir / "runtime_progress.json", {}) or {}
            selected_jobs = int(runtime_report.get("selected_jobs", 0))
            finished_jobs = int(collection_report.get("finished_jobs", 0))
            complete = bool(
                runtime_report.get("phase") == "finished"
                and finished_jobs == selected_jobs
                and not any(
                    result.get("status") == "disk_gate"
                    for result in collection_report.get("results", [])
                )
            )
            status.update(
                {
                    "accepted": complete,
                    "collect_only": True,
                    "deferred_quality_review": True,
                    "requested_decisions": args.decision or [],
                    "excluded_decisions": args.exclude_decision or [],
                    "selected_jobs": selected_jobs,
                    "finished_jobs": finished_jobs,
                    "accepted_jobs": int(collection_report.get("accepted", 0)),
                    "rejected_jobs": int(collection_report.get("rejected", 0)),
                    "collection_returncode": collection_code,
                    "collection_report": str(collection_report_path),
                }
            )
            if not complete:
                status["failure"] = "collect_only_did_not_attempt_every_selected_job"
            elif collection_report.get("rejected", 0):
                status["warning"] = "collection complete; rejected branches deferred for final review"
            write_status(status_path, status)
            print(
                json.dumps(
                    {
                        "collection_complete": complete,
                        "selected_jobs": selected_jobs,
                        "finished_jobs": finished_jobs,
                        "accepted_jobs": status["accepted_jobs"],
                        "rejected_jobs": status["rejected_jobs"],
                        "status": str(status_path),
                    },
                    indent=2,
                )
            )
            return 0 if complete else 1

        if not args.skip_maintain_smoke:
            smoke = collection_command(args, production_dir)
            smoke.extend(["--decision", "Maintain"])
            run(smoke, status, "maintain_smoke")
            maintain_jobs = [job for job in jobs if job["decision"] == "Maintain"]
            repair_branch_failures(args, maintain_jobs, production_dir, status)

        run(collection_command(args, production_dir), status, "all_decisions")
        repair_branch_failures(args, jobs, production_dir, status)
        _, campaign = validate_campaign(args, production_dir, status, "initial")
        if not campaign.get("accepted"):
            campaign = repair_alignment(
                args, jobs, campaign, production_dir, status
            )
        final_code, campaign = validate_campaign(args, production_dir, status, "final")
        status["accepted"] = bool(final_code == 0 and campaign.get("accepted"))
        status["final_report"] = str(production_dir / "campaign_final.json")
        if not status["accepted"]:
            status["failure"] = "final_campaign_validation_failed"
    except (RuntimeError, subprocess.SubprocessError) as exc:
        status["failure"] = str(exc)
        status["accepted"] = False
    write_status(status_path, status)
    print(json.dumps({
        "accepted": status["accepted"],
        "failure": status.get("failure"),
        "status": str(status_path),
        "final_report": status.get("final_report"),
    }, indent=2))
    return 0 if status["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
