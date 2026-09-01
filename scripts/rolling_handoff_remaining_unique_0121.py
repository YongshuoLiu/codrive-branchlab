#!/usr/bin/env python3
"""Start GPU1 now and let GPU2 join after its original queue finishes."""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


RL_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = (
    RL_ROOT
    / "data/counterfactual_decision_v1/_production/remaining_unique_0121_manifest"
)
ORIGINAL_PROGRESS = PRODUCTION_DIR / "runtime_progress.json"
SUPERVISOR_STATUS = PRODUCTION_DIR / "supervisor_status.json"
HANDOFF_STATUS = PRODUCTION_DIR / "gpu12_handoff_status.json"
BACKGROUND_LOG = PRODUCTION_DIR / "background.log"
CONFIG = RL_ROOT / "config/decision_v1_remaining_unique_0121.json"
SOURCE_MANIFEST = RL_ROOT / "manifests/remaining_unique_0121_manifest.jsonl"
GPU1_MANIFEST = RL_ROOT / "manifests/remaining_unique_0121_gpu1_takeover.jsonl"
GPU2_MANIFEST = RL_ROOT / "manifests/remaining_unique_0121_gpu2_takeover.jsonl"
COLLECTOR = RL_ROOT / "scripts/collect_counterfactual_v1.py"


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}


def atomic_write_json(path: Path, payload: dict) -> None:
    payload["updated_at_epoch"] = time.time()
    payload["updated_at_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        stat_fields = (Path("/proc") / str(pid) / "stat").read_text().split()
        if len(stat_fields) > 2 and stat_fields[2] == "Z":
            return False
        return True
    except (OSError, IndexError):
        return False


def cmdline(pid: int) -> str:
    try:
        return (Path("/proc") / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode()
    except OSError:
        return ""


def terminate_exact(pid: Optional[int], marker: str, timeout_s: float = 15.0) -> None:
    if not pid_alive(pid):
        return
    assert pid is not None
    command = cmdline(pid)
    if marker not in command:
        raise RuntimeError(
            f"refusing to terminate PID {pid}: expected {marker!r}, got {command!r}"
        )
    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + timeout_s
    while pid_alive(pid) and time.time() < deadline:
        time.sleep(0.5)
    if pid_alive(pid):
        os.kill(pid, signal.SIGKILL)


def launch_takeover(slot: int, manifest: Path) -> subprocess.Popen:
    command = [
        sys.executable,
        "-u",
        str(COLLECTOR),
        "--config",
        str(CONFIG),
        "--manifest",
        str(manifest),
        "--workers",
        str(slot),
        "--walltime",
        "300",
        "--technical-retries",
        "2",
        "--retry-backoff",
        "15",
        "--status-output",
        str(PRODUCTION_DIR / f"takeover_gpu{slot}_collection_status.json"),
        "--progress-output",
        str(PRODUCTION_DIR / f"takeover_gpu{slot}_progress.json"),
        "--skip-existing-results",
    ]
    with BACKGROUND_LOG.open("a", encoding="utf-8") as log_handle:
        log_handle.write(
            f"\n[rolling-handoff] starting {len(read_jsonl(manifest))} "
            f"non-overlapping takeover jobs on GPU{slot}\n"
        )
        log_handle.flush()
        return subprocess.Popen(
            command,
            cwd=RL_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def snapshot_pending_gpu0_jobs() -> tuple[list[dict], list[dict]]:
    pending = [
        job
        for job in read_jsonl(SOURCE_MANIFEST)
        if int(job["worker_slot"]) == 0
        and not (Path(job["output_dir"]) / "quality_report.json").is_file()
    ]
    groups: list[tuple[str, list[dict]]] = []
    by_group: dict[str, list[dict]] = {}
    for job in pending:
        group_id = job["group_id"]
        if group_id not in by_group:
            by_group[group_id] = []
            groups.append((group_id, by_group[group_id]))
        by_group[group_id].append(job)

    # GPU1 starts while GPU2 still has a few original jobs. Give GPU1 53% of
    # whole route groups so their expected finishing times remain close.
    target_gpu1 = math.ceil(len(pending) * 0.53)
    gpu1: list[dict] = []
    gpu2: list[dict] = []
    for _, jobs in groups:
        if len(gpu1) < target_gpu1:
            gpu1.extend(jobs)
        else:
            gpu2.extend(jobs)
    return gpu1, gpu2


def publish_finished_original_progress() -> None:
    progress = read_json(ORIGINAL_PROGRESS)
    progress["phase"] = "finished"
    for worker in (progress.get("workers") or {}).values():
        worker["state"] = "finished"
        worker["stage"] = "finished"
        worker.pop("current_job", None)
    atomic_write_json(ORIGINAL_PROGRESS, progress)


def main() -> int:
    original = read_json(ORIGINAL_PROGRESS)
    old_collector = int(original.get("collector_pid") or 0)
    workers = original.get("workers") or {}
    worker0 = workers.get("0") or {}
    paused_worker0 = int((worker0.get("current_job") or {}).get("evaluator_pid") or 0)
    if (workers.get("1") or {}).get("state") != "finished":
        raise SystemExit("GPU1 original queue is not finished")
    if not pid_alive(old_collector) or "collect_counterfactual_v1.py" not in cmdline(old_collector):
        raise SystemExit("original collector is not alive")

    gpu1_jobs, gpu2_jobs = snapshot_pending_gpu0_jobs()
    if not gpu1_jobs or not gpu2_jobs:
        raise SystemExit(
            f"cannot split pending GPU0 work: gpu1={len(gpu1_jobs)} gpu2={len(gpu2_jobs)}"
        )
    atomic_write_jsonl(GPU1_MANIFEST, gpu1_jobs)
    atomic_write_jsonl(GPU2_MANIFEST, gpu2_jobs)

    state = {
        "schema_version": "remaining_unique_rolling_gpu12_handoff_v1",
        "state": "gpu1_takeover_running_gpu2_finishing_original",
        "handoff_pid": os.getpid(),
        "old_collector_pid": old_collector,
        "paused_worker0_pid": paused_worker0,
        "gpu1_manifest": str(GPU1_MANIFEST),
        "gpu2_manifest": str(GPU2_MANIFEST),
        "gpu1_assigned_jobs": len(gpu1_jobs),
        "gpu2_assigned_jobs": len(gpu2_jobs),
    }
    gpu1_process = launch_takeover(1, GPU1_MANIFEST)
    state["gpu1_collector_pid"] = gpu1_process.pid
    atomic_write_json(HANDOFF_STATUS, state)

    while True:
        original = read_json(ORIGINAL_PROGRESS)
        worker2 = (original.get("workers") or {}).get("2") or {}
        state["gpu2_original_state"] = worker2.get("state")
        state["gpu2_original_processed_jobs"] = worker2.get("processed_jobs")
        state["gpu2_original_assigned_jobs"] = worker2.get("assigned_jobs")
        state["gpu1_takeover_returncode"] = gpu1_process.poll()
        atomic_write_json(HANDOFF_STATUS, state)
        if worker2.get("state") == "finished":
            break
        if not pid_alive(old_collector):
            state.update({"state": "failed", "failure": "original collector exited early"})
            atomic_write_json(HANDOFF_STATUS, state)
            return 2
        time.sleep(5)

    gpu2_process = launch_takeover(2, GPU2_MANIFEST)
    state.update(
        {
            "state": "gpu1_gpu2_takeover_running",
            "gpu2_collector_pid": gpu2_process.pid,
        }
    )
    atomic_write_json(HANDOFF_STATUS, state)

    old_supervisor_state = read_json(SUPERVISOR_STATUS)
    old_producer = int(old_supervisor_state.get("producer_pid") or 0)
    old_supervisor = int(old_supervisor_state.get("supervisor_pid") or 0)
    terminate_exact(old_collector, "collect_counterfactual_v1.py")
    terminate_exact(paused_worker0, "run_counterfactual_branch.sh", timeout_s=2.0)
    terminate_exact(old_producer, "produce_counterfactual_routes.py")
    terminate_exact(old_supervisor, "supervise_counterfactual_campaign.py")
    state["old_campaign_retired"] = True
    atomic_write_json(HANDOFF_STATUS, state)

    while gpu1_process.poll() is None or gpu2_process.poll() is None:
        state["gpu1_takeover_returncode"] = gpu1_process.poll()
        state["gpu2_takeover_returncode"] = gpu2_process.poll()
        atomic_write_json(HANDOFF_STATUS, state)
        time.sleep(10)

    state.update(
        {
            "state": "takeover_complete",
            "gpu1_takeover_returncode": int(gpu1_process.returncode),
            "gpu2_takeover_returncode": int(gpu2_process.returncode),
            "finished_at_epoch": time.time(),
        }
    )
    publish_finished_original_progress()
    atomic_write_json(HANDOFF_STATUS, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
