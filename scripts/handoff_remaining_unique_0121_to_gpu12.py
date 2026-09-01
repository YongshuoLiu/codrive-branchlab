#!/usr/bin/env python3
"""Wait for workers 1/2, retire the paused GPU0 worker, and resume on GPUs 1/2."""

from __future__ import annotations

import json
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
PROGRESS = PRODUCTION_DIR / "runtime_progress.json"
SUPERVISOR_STATUS = PRODUCTION_DIR / "supervisor_status.json"
HANDOFF_STATUS = PRODUCTION_DIR / "gpu12_handoff_status.json"
BACKGROUND_LOG = PRODUCTION_DIR / "background.log"
CONFIG = RL_ROOT / "config/decision_v1_remaining_unique_0121.json"
MANIFEST = RL_ROOT / "manifests/remaining_unique_0121_manifest.jsonl"
LIVE_PREFLIGHT = PRODUCTION_DIR / "live_preflight.json"
SUPERVISOR = RL_ROOT / "scripts/supervise_counterfactual_campaign.py"
VALIDATOR = RL_ROOT / "scripts/validate_counterfactual_branch.py"


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}


def atomic_write(path: Path, payload: dict) -> None:
    payload["updated_at_epoch"] = time.time()
    payload["updated_at_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def cmdline(pid: int) -> str:
    try:
        return (Path("/proc") / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode()
    except OSError:
        return ""


def terminate_exact(pid: Optional[int], required_marker: str, timeout_s: float = 15.0) -> None:
    if not pid_alive(pid):
        return
    assert pid is not None
    command = cmdline(pid)
    if required_marker not in command:
        raise RuntimeError(
            f"refusing to terminate PID {pid}: expected {required_marker!r}, got {command!r}"
        )
    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + timeout_s
    while pid_alive(pid) and time.time() < deadline:
        time.sleep(0.5)
    if pid_alive(pid):
        os.kill(pid, signal.SIGKILL)


def main() -> int:
    initial = read_json(PROGRESS)
    old_collector = int(initial.get("collector_pid") or 0)
    worker0 = (initial.get("workers") or {}).get("0", {})
    interrupted_job = (worker0.get("current_job") or {}).get("job_id")
    interrupted_dir = Path((worker0.get("current_job") or {}).get("output_dir", ""))
    paused_worker_root = int((worker0.get("current_job") or {}).get("evaluator_pid") or 0)
    if not old_collector or "collect_counterfactual_v1.py" not in cmdline(old_collector):
        raise SystemExit("the expected original collector is not alive")

    state = {
        "schema_version": "remaining_unique_gpu12_handoff_v1",
        "state": "waiting_for_workers_1_2",
        "handoff_pid": os.getpid(),
        "old_collector_pid": old_collector,
        "paused_worker0_pid": paused_worker_root,
        "interrupted_job": interrupted_job,
        "target_workers": [1, 2],
    }
    atomic_write(HANDOFF_STATUS, state)

    while True:
        progress = read_json(PROGRESS)
        workers = progress.get("workers") or {}
        worker_states = {
            slot: (workers.get(slot) or {}).get("state") for slot in ("1", "2")
        }
        state["worker_states"] = worker_states
        atomic_write(HANDOFF_STATUS, state)
        if all(value == "finished" for value in worker_states.values()):
            break
        if not pid_alive(old_collector):
            state.update({"state": "failed", "failure": "old collector exited before handoff"})
            atomic_write(HANDOFF_STATUS, state)
            return 2
        time.sleep(10)

    state["state"] = "retiring_old_collector"
    atomic_write(HANDOFF_STATUS, state)
    old_supervisor_state = read_json(SUPERVISOR_STATUS)
    old_producer = int(old_supervisor_state.get("producer_pid") or 0)
    old_supervisor = int(old_supervisor_state.get("supervisor_pid") or 0)

    terminate_exact(old_collector, "collect_counterfactual_v1.py")
    terminate_exact(paused_worker_root, "run_counterfactual_branch.sh", timeout_s=2.0)
    terminate_exact(old_producer, "produce_counterfactual_routes.py")
    terminate_exact(old_supervisor, "supervise_counterfactual_campaign.py")

    # The interrupted branch had normally finished CARLA but had not reached
    # validation because worker 0 was frozen. Preserve it if it validates;
    # otherwise archive it so the two-GPU resume recollects it.
    validation_accepted = False
    if interrupted_job and interrupted_dir.is_dir():
        subprocess.run(
            [
                sys.executable,
                str(VALIDATOR),
                str(interrupted_dir),
                "--manifest",
                str(MANIFEST),
                "--job-id",
                interrupted_job,
            ],
            cwd=RL_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        validation_accepted = bool(
            read_json(interrupted_dir / "quality_report.json").get("accepted")
        )
        if not validation_accepted:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            archive = interrupted_dir.with_name(
                f"{interrupted_dir.name}_interrupted_gpu0_{stamp}"
            )
            interrupted_dir.rename(archive)
            state["interrupted_archive"] = str(archive)
    state["interrupted_validation_accepted"] = validation_accepted

    command = [
        sys.executable,
        "-u",
        str(SUPERVISOR),
        "--config",
        str(CONFIG),
        "--manifest",
        str(MANIFEST),
        "--live-preflight-report",
        str(LIVE_PREFLIGHT),
        "--workers",
        "1,2",
        "--collect-only",
        "--skip-existing-results",
    ]
    with BACKGROUND_LOG.open("a", encoding="utf-8") as log_handle:
        log_handle.write(
            "\n[gpu12-handoff] workers 1/2 finished their original queues; "
            "resuming all unfinished jobs on GPUs 1/2\n"
        )
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=RL_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_tmp = PRODUCTION_DIR / f".supervisor.pid.{os.getpid()}.tmp"
    pid_tmp.write_text(f"{process.pid}\n", encoding="utf-8")
    os.replace(pid_tmp, PRODUCTION_DIR / "supervisor.pid")
    state.update(
        {
            "state": "gpu12_resume_started",
            "new_supervisor_pid": process.pid,
            "command": command,
        }
    )
    atomic_write(HANDOFF_STATUS, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
