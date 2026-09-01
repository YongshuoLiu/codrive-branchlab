#!/usr/bin/env python3
"""Wait for the active campaign, then run the S1 V2 recollection on GPUs 1/2."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


RL_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
UPSTREAM_PROGRESS = (
    RL_ROOT
    / "data/counterfactual_decision_v1/_production/remaining_unique_0121_manifest/runtime_progress.json"
)
OUTPUT_ROOT = RL_ROOT / "data/counterfactual_decision_v1/_recollected/S1_PED_SPEED_V2"
PRODUCTION_ROOT = OUTPUT_ROOT / "_production"
STATUS = PRODUCTION_ROOT / "queue_status.json"
CONFIG = RL_ROOT / "config/s1_ped_speed_v2_recollection.json"
MANIFEST = RL_ROOT / "manifests/s1_ped_speed_v2_recollection.jsonl"
COLLECTOR = RL_ROOT / "scripts/collect_counterfactual_v1.py"
AUDIT = RL_ROOT / "scripts/audit_s1_ped_speed_v2.py"
PORTS = (32400, 32500, 32600, 42400, 42500, 42600)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}


def pid_alive(pid: object) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def port_free(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.2)
    try:
        return sock.connect_ex(("127.0.0.1", port)) != 0
    finally:
        sock.close()


def publish(state: dict) -> None:
    PRODUCTION_ROOT.mkdir(parents=True, exist_ok=True)
    state["updated_at_epoch"] = time.time()
    state["updated_at_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    temporary = STATUS.with_name(f".{STATUS.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, STATUS)


def main() -> int:
    state = {
        "schema_version": "s1_ped_speed_v2_queue_v1",
        "queue_pid": os.getpid(),
        "target_policy_version": "S1_PED_SPEED_V2",
        "target_ped_speed_mps": 8.5,
        "config": str(CONFIG),
        "manifest": str(MANIFEST),
        "state": "waiting_for_upstream",
    }
    publish(state)

    while True:
        upstream = read_json(UPSTREAM_PROGRESS)
        active_workers = [
            row.get("current_job", {}).get("job_id")
            for row in (upstream.get("workers") or {}).values()
            if row.get("state") not in {"finished", "waiting", "idle"} and row.get("current_job")
        ]
        upstream_alive = pid_alive(upstream.get("collector_pid"))
        upstream_finished = upstream.get("phase") == "finished"
        state.update(
            {
                "state": "waiting_for_upstream",
                "upstream_phase": upstream.get("phase"),
                "upstream_collector_pid": upstream.get("collector_pid"),
                "upstream_alive": upstream_alive,
                "upstream_active_jobs": active_workers,
            }
        )
        publish(state)
        if (upstream_finished or not upstream_alive) and not active_workers:
            break
        time.sleep(15)

    while True:
        occupied = [port for port in PORTS if not port_free(port)]
        if not occupied:
            break
        state.update({"state": "waiting_for_ports", "occupied_ports": occupied})
        publish(state)
        time.sleep(10)

    subprocess.run([str(PYTHON), str(AUDIT)], cwd=RL_ROOT, check=False)
    command = [
        str(PYTHON), "-u", str(COLLECTOR),
        "--config", str(CONFIG),
        "--manifest", str(MANIFEST),
        "--workers", "1,2",
        "--walltime", "300",
        "--technical-retries", "2",
        "--retry-backoff", "15",
        "--status-output", str(PRODUCTION_ROOT / "campaign_status.json"),
        "--progress-output", str(PRODUCTION_ROOT / "runtime_progress.json"),
    ]
    process = subprocess.Popen(command, cwd=RL_ROOT)
    state.update({"state": "collecting", "collector_pid": process.pid, "command": command})
    publish(state)
    while process.poll() is None:
        subprocess.run(
            [str(PYTHON), str(AUDIT)],
            cwd=RL_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        progress = read_json(PRODUCTION_ROOT / "runtime_progress.json")
        state.update(
            {
                "collector_phase": progress.get("phase"),
                "result_counts": progress.get("result_counts", {}),
                "outcome_counts": progress.get("outcome_counts", {}),
            }
        )
        publish(state)
        time.sleep(30)
    code = int(process.returncode)
    subprocess.run([str(PYTHON), str(AUDIT)], cwd=RL_ROOT, check=False)
    state.update(
        {
            "state": "complete" if code == 0 else "failed",
            "collector_returncode": code,
            "finished_at_epoch": time.time(),
        }
    )
    publish(state)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
