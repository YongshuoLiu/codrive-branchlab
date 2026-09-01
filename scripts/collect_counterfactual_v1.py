#!/usr/bin/env python3
"""Resume-safe three-GPU scheduler for counterfactual decision collection."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from counterfactual_contract import (
    read_json,
    sha256,
    validate_manifest_job,
    validate_run_contract,
)


RL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = RL_ROOT / "config/decision_v1.json"
DEFAULT_MANIFEST = RL_ROOT / "manifests/collection_manifest.jsonl"
RUN_ONE = RL_ROOT / "scripts/run_counterfactual_branch.sh"
VALIDATE_ONE = RL_ROOT / "scripts/validate_counterfactual_branch.py"
BUILD_MANIFEST = RL_ROOT / "scripts/build_collection_manifest.py"
DERIVED_AUGMENTED_MODALITIES = {
    "rgb_augmented",
    "depth_augmented",
    "semantics_augmented",
    "bev_semantics_augmented",
}

DECISION_ENV_NAMES = {
    "accelerate_delta_mps": "COUNTERFACTUAL_ACCELERATE_DELTA_MPS",
    "accelerate_cap_mps": "COUNTERFACTUAL_ACCELERATE_CAP_MPS",
    "brake_speed_ratio": "COUNTERFACTUAL_BRAKE_SPEED_RATIO",
    "brake_min_target_mps": "COUNTERFACTUAL_BRAKE_MIN_TARGET_MPS",
    "stop_speed_mps": "COUNTERFACTUAL_STOP_SPEED_MPS",
    "stop_hold_s": "COUNTERFACTUAL_STOP_HOLD_S",
    "lane_change_min_target_mps": "COUNTERFACTUAL_LANE_MIN_TARGET_MPS",
    "lane_change_pre_distance_m": "COUNTERFACTUAL_LANE_PRE_DISTANCE_M",
    "lane_change_post_distance_m": "COUNTERFACTUAL_LANE_POST_DISTANCE_M",
    "lane_change_transition_distance_m": "COUNTERFACTUAL_LANE_TRANSITION_DISTANCE_M",
    "lane_geometry_min_coverage": "COUNTERFACTUAL_LANE_MIN_COVERAGE",
    "lane_min_width_m": "COUNTERFACTUAL_LANE_MIN_WIDTH_M",
    "lane_actor_rear_clearance_m": "COUNTERFACTUAL_LANE_REAR_CLEARANCE_M",
    "lane_actor_front_clearance_m": "COUNTERFACTUAL_LANE_FRONT_CLEARANCE_M",
    "lane_actor_min_ttc_s": "COUNTERFACTUAL_LANE_MIN_TTC_S",
    "deadlock_speed_mps": "COUNTERFACTUAL_DEADLOCK_SPEED_MPS",
    "deadlock_hold_s": "COUNTERFACTUAL_DEADLOCK_HOLD_S",
    "post_clear_deadlock_hold_s": "COUNTERFACTUAL_POST_CLEAR_DEADLOCK_HOLD_S",
    "max_decision_active_s": "COUNTERFACTUAL_MAX_DECISION_ACTIVE_S",
    "raw_actor_radius_m": "COUNTERFACTUAL_RAW_ACTOR_RADIUS_M",
}


def atomic_write_json(path: Path, payload: dict) -> None:
    """Publish machine-readable progress without exposing partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


class RuntimeProgress:
    """Thread-safe live state shared by the three collection workers."""

    def __init__(
        self,
        path: Path,
        config_path: Path,
        manifest_path: Path,
        jobs: list[dict],
        jobs_by_worker: dict[int, list[dict]],
        workers: list[dict],
        cache_rows: list[dict],
    ) -> None:
        self.path = path
        self.lock = threading.Lock()
        now = time.time()
        self.state = {
            "schema_version": "counterfactual_runtime_progress_v1",
            "collector_pid": os.getpid(),
            "config": str(config_path.resolve()),
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": sha256(manifest_path),
            "started_at_epoch": now,
            "started_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "updated_at_epoch": now,
            "phase": "initializing",
            "selected_jobs": len(jobs),
            "cached_at_start": sum(row["cached"] for row in cache_rows),
            "pending_at_start": sum(not row["cached"] for row in cache_rows),
            "result_counts": {},
            "outcome_counts": {},
            "recent_results": [],
            "workers": {
                str(worker["slot"]): {
                    "slot": int(worker["slot"]),
                    "gpu": int(worker["gpu"]),
                    "graphics_adapter": int(worker["graphics_adapter"]),
                    "port": int(worker["port"]),
                    "tm_port": int(worker["tm_port"]),
                    "assigned_jobs": len(jobs_by_worker[int(worker["slot"])]),
                    "processed_jobs": 0,
                    "accepted_jobs": 0,
                    "state": "waiting",
                    "stage": "waiting",
                    "current_job": None,
                    "last_result": None,
                }
                for worker in workers
            },
        }
        self._publish_locked()

    def _publish_locked(self) -> None:
        self.state["updated_at_epoch"] = time.time()
        self.state["updated_at_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        atomic_write_json(self.path, self.state)

    def phase(self, phase: str) -> None:
        with self.lock:
            self.state["phase"] = phase
            self._publish_locked()

    def worker_state(
        self, slot: int, state: str, stage: str | None = None, **extra
    ) -> None:
        with self.lock:
            row = self.state["workers"][str(slot)]
            row["state"] = state
            row["stage"] = stage or state
            row.update(extra)
            self._publish_locked()

    def start_job(
        self,
        worker: dict,
        job: dict,
        index: int,
        total: int,
        attempt: int,
    ) -> None:
        now = time.time()
        with self.lock:
            row = self.state["workers"][str(worker["slot"])]
            existing = row.get("current_job") or {}
            started = (
                existing.get("started_at_epoch", now)
                if existing.get("job_id") == job["job_id"]
                else now
            )
            row.update(
                {
                    "state": "running",
                    "stage": "preparing",
                    "current_job": {
                        "job_id": job["job_id"],
                        "group_id": job["group_id"],
                        "scenario_id": job["scenario_id"],
                        "scenario_name": job.get("scenario_name"),
                        "route_id": job.get("route_id"),
                        "decision": job["decision"],
                        "town": job.get("town"),
                        "output_dir": job["output_dir"],
                        "worker_index": index,
                        "worker_total": total,
                        "attempt": attempt,
                        "started_at_epoch": started,
                        "attempt_started_at_epoch": now,
                        "evaluator_pid": None,
                    },
                }
            )
            self._publish_locked()

    def job_stage(self, slot: int, stage: str, **extra) -> None:
        with self.lock:
            row = self.state["workers"][str(slot)]
            row["stage"] = stage
            current = row.get("current_job")
            if current is not None:
                current.update(extra)
            self._publish_locked()

    def heartbeat(self, slot: int) -> None:
        with self.lock:
            row = self.state["workers"][str(slot)]
            row["heartbeat_at_epoch"] = time.time()
            self._publish_locked()

    def result(self, slot: int, result: dict) -> None:
        status = str(result.get("status", "unknown"))
        outcome = str(result.get("outcome", status))
        with self.lock:
            row = self.state["workers"][str(slot)]
            row["processed_jobs"] += 1
            row["accepted_jobs"] += int(bool(result.get("accepted")))
            row["state"] = "idle"
            row["stage"] = "idle"
            row["last_result"] = {
                "job_id": result.get("job_id"),
                "accepted": bool(result.get("accepted")),
                "status": status,
                "outcome": outcome,
                "finished_at_epoch": time.time(),
            }
            row["current_job"] = None
            counts = self.state["result_counts"]
            counts[status] = int(counts.get(status, 0)) + 1
            outcomes = self.state["outcome_counts"]
            outcomes[outcome] = int(outcomes.get(outcome, 0)) + 1
            self.state["recent_results"].append(row["last_result"])
            self.state["recent_results"] = self.state["recent_results"][-20:]
            self._publish_locked()

    def finish(self, accepted: int, rejected: int) -> None:
        with self.lock:
            self.state["phase"] = "finished"
            self.state["finished_at_epoch"] = time.time()
            self.state["accepted_this_invocation"] = accepted
            self.state["rejected_this_invocation"] = rejected
            for row in self.state["workers"].values():
                if row["state"] != "disk_gate":
                    row["state"] = "finished"
                    row["stage"] = "finished"
                    row["current_job"] = None
            self._publish_locked()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def cache_assessment(job: dict) -> tuple[bool, list[str]]:
    """Only reuse an accepted branch that is still bound to this manifest row."""
    report_path = Path(job["output_dir"]) / "quality_report.json"
    if not report_path.is_file():
        return False, ["missing quality_report.json"]
    report = read_json(report_path, {}) or {}
    if report.get("accepted") is not True:
        return False, ["quality_report.json is not accepted"]
    errors = validate_run_contract(Path(job["output_dir"]), job)
    return not errors, errors


def cached(job: dict) -> bool:
    return cache_assessment(job)[0]


def free_disk_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / (1024 ** 3)


def prune_run_derived_augmented(run_dir: Path) -> None:
    """Remove reproducible views/runtime caches and inventory them per branch."""
    campaign_root = (RL_ROOT / "data/counterfactual_decision_v1").resolve()
    resolved_run = run_dir.resolve()
    if campaign_root not in resolved_run.parents:
        raise RuntimeError(f"refusing derived-modality prune outside campaign: {resolved_run}")
    rows = []
    data_root = resolved_run / "data"
    if data_root.is_dir():
        directories = sorted(
            path
            for path in data_root.rglob("*")
            if path.is_dir()
            and not path.is_symlink()
            and path.name in DERIVED_AUGMENTED_MODALITIES
        )
        for directory in directories:
            files = [
                path
                for path in directory.rglob("*")
                if path.is_file() and not path.is_symlink()
            ]
            rows.append(
                {
                    "path": str(directory.relative_to(resolved_run)),
                    "modality": directory.name,
                    "file_count": len(files),
                    "bytes": sum(path.stat().st_size for path in files),
                }
            )
            shutil.rmtree(directory)
    for relative in (Path("home/carlaCache"), Path("home/.cache"), Path("home/.triton")):
        directory = run_dir / relative
        if not directory.is_dir() or directory.is_symlink():
            continue
        files = [
            path
            for path in directory.rglob("*")
            if path.is_file() and not path.is_symlink()
        ]
        rows.append(
            {
                "path": str(relative),
                "modality": "reproducible_runtime_cache",
                "file_count": len(files),
                "bytes": sum(path.stat().st_size for path in files),
            }
        )
        shutil.rmtree(directory)
    metadata = resolved_run / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "derived_modalities_policy.json").write_text(
        json.dumps(
            {
                "policy": "remove_reproducible_views_and_runtime_caches_keep_original_carla_modalities",
                "removed_directories": rows,
                "removed_files": sum(row["file_count"] for row in rows),
                "removed_bytes": sum(row["bytes"] for row in rows),
                "retained": [
                    "rgb",
                    "depth",
                    "semantics",
                    "bev_semantics",
                    "lidar",
                    "boxes",
                    "measurements",
                    "carla_telemetry_20hz",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run_job(
    job: dict,
    worker: dict,
    config: dict,
    walltime: int,
    rerun: bool,
    args,
    progress: RuntimeProgress,
    worker_index: int,
    worker_total: int,
) -> dict:
    run_dir = Path(job["output_dir"])
    cache_ok, _ = cache_assessment(job)
    if cache_ok and not rerun:
        return {"job_id": job["job_id"], "status": "cached", "accepted": True}
    maximum_attempts = 1 + max(int(args.technical_retries), 0)
    for attempt in range(1, maximum_attempts + 1):
        progress.start_job(worker, job, worker_index, worker_total, attempt)
        if run_dir.exists() and any(run_dir.iterdir()):
            progress.job_stage(int(worker["slot"]), "quarantining_previous_attempt")
            suffix = time.strftime("%Y%m%d_%H%M%S")
            quarantine = run_dir.with_name(f"{run_dir.name}_incomplete_{suffix}")
            counter = 1
            while quarantine.exists():
                quarantine = run_dir.with_name(f"{run_dir.name}_incomplete_{suffix}_{counter:02d}")
                counter += 1
            shutil.move(str(run_dir), str(quarantine))
        run_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(
            {
                "ROUTE": job["route_path"],
                "DECISION": job["decision"],
                "RUN_DIR": str(run_dir),
                "GPU_RANK": str(worker["gpu"]),
                "CARLA_GRAPHICS_ADAPTER": str(worker["graphics_adapter"]),
                "PORT": str(worker["port"]),
                "TM_PORT": str(worker["tm_port"]),
                "EVAL_WALLTIME_SECONDS": str(walltime),
            }
        )
        parameters = config.get("decision_parameters", {})
        env.update(
            {
                environment_name: str(parameters[config_name])
                for config_name, environment_name in DECISION_ENV_NAMES.items()
                if config_name in parameters
            }
        )
        progress.job_stage(int(worker["slot"]), "launching_carla")
        process = subprocess.Popen(
            ["bash", str(RUN_ONE)], env=env, start_new_session=True
        )
        progress.job_stage(
            int(worker["slot"]), "carla_simulation", evaluator_pid=process.pid
        )
        while process.poll() is None:
            progress.heartbeat(int(worker["slot"]))
            time.sleep(1.0)
        run_status = int(process.returncode)
        progress.job_stage(int(worker["slot"]), "pruning_derived_modalities")
        prune_run_derived_augmented(run_dir)
        progress.job_stage(int(worker["slot"]), "validating_branch")
        validate_status = subprocess.run(
            [
                sys.executable,
                str(VALIDATE_ONE),
                str(run_dir),
                "--manifest",
                str(args.manifest.resolve()),
                "--job-id",
                job["job_id"],
            ],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
        ).returncode
        report_path = run_dir / "quality_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
        errors = report.get("errors", [])
        termination_exists = (run_dir / "metadata/termination.json").is_file()
        technical_failure = (
            not bool(report.get("accepted"))
            and not termination_exists
            and (
                run_status != 0
                or any(
                    marker in str(error)
                    for error in errors
                    for marker in (
                        "missing readable CARLA telemetry",
                        "route status is missing",
                        "without a termination marker",
                    )
                )
            )
        )
        result = {
            "job_id": job["job_id"],
            "status": "finished",
            "attempts": attempt,
            "technical_failure": technical_failure,
            "run_status": run_status,
            "validate_status": validate_status,
            "accepted": bool(report.get("accepted")),
            "outcome": report.get("outcome", "missing_report"),
            "errors": errors,
            "output_dir": str(run_dir),
        }
        if not technical_failure or attempt >= maximum_attempts:
            return result
        delay = float(args.retry_backoff) * attempt + float(worker["slot"]) * 5.0
        print(
            f"[worker {worker['slot']}] {job['job_id']}: technical failure "
            f"on attempt {attempt}/{maximum_attempts}; retrying in {delay:.0f}s",
            flush=True,
        )
        progress.job_stage(
            int(worker["slot"]),
            "technical_retry_backoff",
            retry_delay_s=round(delay, 1),
        )
        time.sleep(delay)
    raise AssertionError("unreachable retry loop")


def run_worker(
    jobs: list[dict],
    worker: dict,
    config: dict,
    args,
    progress: RuntimeProgress,
) -> list[dict]:
    results = []
    initial_delay = float(args.startup_stagger) * int(worker["slot"])
    if initial_delay > 0:
        progress.worker_state(
            int(worker["slot"]),
            "startup_stagger",
            startup_delay_s=round(initial_delay, 1),
        )
        print(f"[worker {worker['slot']}] startup stagger {initial_delay:.0f}s", flush=True)
        time.sleep(initial_delay)
    minimum_gb = float(config["quality_thresholds"]["minimum_free_disk_gb"])
    output_root = Path(config["output_root"])
    if not jobs:
        progress.worker_state(int(worker["slot"]), "finished")
    for index, job in enumerate(jobs, 1):
        available = free_disk_gb(output_root)
        if available < minimum_gb:
            result = {
                "job_id": job["job_id"],
                "status": "disk_gate",
                "accepted": False,
                "errors": [f"free disk {available:.2f} GB is below {minimum_gb:.2f} GB"],
            }
            results.append(result)
            progress.result(int(worker["slot"]), result)
            progress.worker_state(
                int(worker["slot"]),
                "disk_gate",
                free_disk_gb=round(available, 3),
                minimum_free_disk_gb=minimum_gb,
            )
            print(f"[worker {worker['slot']}] disk gate: {result['errors'][0]}", flush=True)
            break
        print(
            f"[worker {worker['slot']} {index}/{len(jobs)}] "
            f"{job['job_id']} gpu={worker['gpu']} free={available:.1f}GB",
            flush=True,
        )
        result = run_job(
            job,
            worker,
            config,
            args.walltime,
            args.rerun,
            args,
            progress,
            index,
            len(jobs),
        )
        results.append(result)
        progress.result(int(worker["slot"]), result)
        print(
            f"[worker {worker['slot']}] {job['job_id']}: "
            f"accepted={result.get('accepted')} outcome={result.get('outcome', result['status'])}",
            flush=True,
        )
        if args.fail_fast and not result.get("accepted"):
            break
    if not results or results[-1].get("status") != "disk_gate":
        progress.worker_state(int(worker["slot"]), "finished")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--scenario", action="append")
    parser.add_argument("--job-id", action="append")
    parser.add_argument("--decision", action="append")
    parser.add_argument("--exclude-decision", action="append")
    parser.add_argument("--group", action="append")
    parser.add_argument("--town", action="append")
    parser.add_argument("--exclude-town", action="append")
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--walltime", type=int, default=300)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument(
        "--skip-existing-results",
        action="store_true",
        help="Do not revisit branches that already have a quality report, accepted or rejected",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", default="0,1,2", help="Comma-separated worker slots")
    parser.add_argument("--startup-stagger", type=float, default=15.0)
    parser.add_argument("--technical-retries", type=int, default=2)
    parser.add_argument("--retry-backoff", type=float, default=15.0)
    parser.add_argument(
        "--status-output",
        type=Path,
        help="Write scheduler status here instead of replacing campaign_status.json",
    )
    parser.add_argument(
        "--progress-output",
        type=Path,
        help="Atomically refreshed live worker/CARLA progress JSON",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not args.manifest.is_file():
        subprocess.run(
            [sys.executable, str(BUILD_MANIFEST), "--config", str(args.config), "--output", str(args.manifest)],
            check=True,
        )
    jobs = load_jsonl(args.manifest)
    manifest_errors = {}
    for index, job in enumerate(jobs):
        contract_errors = validate_manifest_job(job)
        if contract_errors:
            manifest_errors[job.get("job_id", f"row_{index}")] = contract_errors
    duplicate_job_ids = len(jobs) - len({job.get("job_id") for job in jobs})
    duplicate_output_dirs = len(jobs) - len({job.get("output_dir") for job in jobs})
    if duplicate_job_ids:
        manifest_errors["__manifest_job_ids__"] = [
            f"manifest has {duplicate_job_ids} duplicate job ids"
        ]
    if duplicate_output_dirs:
        manifest_errors["__manifest_output_dirs__"] = [
            f"manifest has {duplicate_output_dirs} duplicate output directories"
        ]
    if manifest_errors:
        print(
            json.dumps(
                {
                    "accepted": False,
                    "reason": "manifest_contract_failed",
                    "manifest": str(args.manifest.resolve()),
                    "manifest_sha256": sha256(args.manifest),
                    "errors": manifest_errors,
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    if args.job_id:
        selected = set(args.job_id)
        jobs = [job for job in jobs if job["job_id"] in selected]
    if args.scenario:
        selected = set(args.scenario)
        jobs = [job for job in jobs if job["scenario_id"] in selected]
    if args.decision:
        selected = set(args.decision)
        jobs = [job for job in jobs if job["decision"] in selected]
    if args.exclude_decision:
        excluded = set(args.exclude_decision)
        jobs = [job for job in jobs if job["decision"] not in excluded]
    if args.group:
        selected = set(args.group)
        jobs = [job for job in jobs if job["group_id"] in selected]
    if args.town:
        selected = set(args.town)
        jobs = [job for job in jobs if job.get("town") in selected]
    if args.exclude_town:
        excluded = set(args.exclude_town)
        jobs = [job for job in jobs if job.get("town") not in excluded]
    skipped_existing_results = 0
    if args.skip_existing_results:
        selected_count = len(jobs)
        jobs = [
            job
            for job in jobs
            if not (Path(job["output_dir"]) / "quality_report.json").is_file()
        ]
        skipped_existing_results = selected_count - len(jobs)
    if args.max_jobs > 0:
        jobs = jobs[: args.max_jobs]
    worker_slots = {int(value) for value in args.workers.split(",") if value.strip()}
    workers = [worker for worker in config["workers"] if int(worker["slot"]) in worker_slots]
    if not workers:
        raise SystemExit("no workers selected")
    jobs_by_worker = {int(worker["slot"]): [] for worker in workers}
    selected_slots = sorted(jobs_by_worker)
    for job in jobs:
        preferred = int(job["worker_slot"])
        slot = preferred if preferred in jobs_by_worker else selected_slots[job["group_index"] % len(selected_slots)]
        jobs_by_worker[slot].append(job)

    cache_rows = [
        {
            "job_id": job["job_id"],
            "cached": assessment[0],
            "reasons": assessment[1],
            "output_exists": Path(job["output_dir"]).exists(),
        }
        for job in jobs
        for assessment in [cache_assessment(job)]
    ]
    stale_cache = [
        row
        for row in cache_rows
        if not row["cached"] and row["output_exists"]
    ]
    summary = {
        "job_count": len(jobs),
        "workers": {str(slot): len(items) for slot, items in jobs_by_worker.items()},
        "manifest_sha256": sha256(args.manifest),
        "cached": sum(row["cached"] for row in cache_rows),
        "stale_or_rejected_existing": len(stale_cache),
        "skipped_existing_results": skipped_existing_results,
        "stale_or_rejected_examples": stale_cache[:20],
        "free_disk_gb": free_disk_gb(Path(config["output_root"])),
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return 0

    output_root = Path(config["output_root"])
    status_output = (args.status_output or output_root / "campaign_status.json").resolve()
    progress_output = (
        args.progress_output or status_output.with_name("runtime_progress.json")
    ).resolve()
    if output_root.resolve() not in progress_output.parents:
        raise SystemExit(f"progress output must remain under {output_root.resolve()}")
    progress = RuntimeProgress(
        progress_output,
        args.config,
        args.manifest,
        jobs,
        jobs_by_worker,
        workers,
        cache_rows,
    )
    progress.phase("collecting")

    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(workers)) as pool:
        futures = {
            pool.submit(
                run_worker,
                jobs_by_worker[int(worker["slot"])],
                worker,
                config,
                args,
                progress,
            ): worker
            for worker in workers
        }
        for future in concurrent.futures.as_completed(futures):
            all_results.extend(future.result())
    campaign = {
        "schema_version": config["schema_version"],
        "manifest": str(args.manifest.resolve()),
        "finished_jobs": len(all_results),
        "accepted": sum(bool(result.get("accepted")) for result in all_results),
        "rejected": sum(not bool(result.get("accepted")) for result in all_results),
        "results": sorted(all_results, key=lambda item: item["job_id"]),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    if output_root.resolve() not in status_output.parents:
        raise SystemExit(f"status output must remain under {output_root.resolve()}")
    status_output.parent.mkdir(parents=True, exist_ok=True)
    status_output.write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    progress.finish(campaign["accepted"], campaign["rejected"])
    print(json.dumps({key: value for key, value in campaign.items() if key != "results"}, indent=2))
    return 0 if campaign["rejected"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
