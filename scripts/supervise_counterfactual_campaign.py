#!/usr/bin/env python3
"""Resume-safe background supervisor for the production controller."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


RL_ROOT = Path(__file__).resolve().parents[1]
PRODUCER = RL_ROOT / "scripts/produce_counterfactual_routes.py"


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at_epoch"] = time.time()
    payload["updated_at_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def free_disk_gib(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / (1024 ** 3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--live-preflight-report", type=Path, required=True)
    parser.add_argument("--workers", default="0,1,2")
    parser.add_argument("--walltime", type=int, default=300)
    parser.add_argument("--technical-retries", type=int, default=2)
    parser.add_argument("--retry-backoff", type=float, default=15.0)
    parser.add_argument("--max-branch-repair-rounds", type=int, default=2)
    parser.add_argument("--max-alignment-rounds", type=int, default=5)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--decision", action="append")
    parser.add_argument("--exclude-decision", action="append")
    parser.add_argument("--skip-existing-results", action="store_true")
    parser.add_argument("--disk-resume-margin-gib", type=float, default=5.0)
    parser.add_argument("--disk-check-interval", type=float, default=60.0)
    args = parser.parse_args()

    args.config = args.config.resolve()
    args.manifest = args.manifest.resolve()
    args.live_preflight_report = args.live_preflight_report.resolve()
    config = read_json(args.config, {}) or {}
    output_root = Path(config["output_root"]).resolve()
    if RL_ROOT.resolve() not in output_root.parents:
        raise SystemExit(f"output root must stay under {RL_ROOT.resolve()}")
    production_dir = output_root / "_production" / args.manifest.stem
    production_dir.mkdir(parents=True, exist_ok=True)
    status_path = production_dir / "supervisor_status.json"
    stop_path = production_dir / "STOP"
    minimum_gib = float(config.get("quality_thresholds", {}).get("minimum_free_disk_gb", 20.0))
    state: dict[str, Any] = {
        "schema_version": "counterfactual_supervisor_v1",
        "supervisor_pid": os.getpid(),
        "config": str(args.config),
        "manifest": str(args.manifest),
        "live_preflight_report": str(args.live_preflight_report),
        "started_at_epoch": time.time(),
        "started_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "state": "starting",
        "attempt": 0,
        "minimum_free_disk_gib": minimum_gib,
        "disk_resume_margin_gib": args.disk_resume_margin_gib,
        "stop_file": str(stop_path),
    }
    atomic_write_json(status_path, state)

    command = [
        sys.executable,
        str(PRODUCER),
        "--config", str(args.config),
        "--manifest", str(args.manifest),
        "--live-preflight-report", str(args.live_preflight_report),
        "--workers", args.workers,
        "--walltime", str(args.walltime),
        "--technical-retries", str(args.technical_retries),
        "--retry-backoff", str(args.retry_backoff),
        "--max-branch-repair-rounds", str(args.max_branch_repair_rounds),
        "--max-alignment-rounds", str(args.max_alignment_rounds),
    ]
    if args.collect_only:
        command.extend(["--collect-only", "--skip-maintain-smoke"])
    for decision in args.decision or []:
        command.extend(["--decision", decision])
    for decision in args.exclude_decision or []:
        command.extend(["--exclude-decision", decision])
    if args.skip_existing_results:
        command.append("--skip-existing-results")

    while True:
        if stop_path.exists():
            state.update({"state": "stopped", "failure": "STOP file requested shutdown"})
            atomic_write_json(status_path, state)
            return 130
        available = free_disk_gib(output_root)
        state["free_disk_gib"] = round(available, 3)
        if available < minimum_gib:
            state.update(
                {
                    "state": "paused_disk",
                    "failure": (
                        f"free disk {available:.2f} GiB is below the "
                        f"{minimum_gib:.2f} GiB safety gate"
                    ),
                    "producer_pid": None,
                }
            )
            atomic_write_json(status_path, state)
            while free_disk_gib(output_root) < minimum_gib + args.disk_resume_margin_gib:
                if stop_path.exists():
                    state.update({"state": "stopped", "failure": "STOP file requested shutdown"})
                    atomic_write_json(status_path, state)
                    return 130
                state["free_disk_gib"] = round(free_disk_gib(output_root), 3)
                atomic_write_json(status_path, state)
                time.sleep(max(args.disk_check_interval, 5.0))
            state.pop("failure", None)

        state["attempt"] += 1
        state.update({"state": "running", "command": command, "free_disk_gib": round(free_disk_gib(output_root), 3)})
        process = subprocess.Popen(command)
        state["producer_pid"] = process.pid
        atomic_write_json(status_path, state)
        while process.poll() is None:
            state["free_disk_gib"] = round(free_disk_gib(output_root), 3)
            atomic_write_json(status_path, state)
            time.sleep(5.0)
        code = int(process.returncode)
        state["last_returncode"] = code
        state["producer_pid"] = None
        state["free_disk_gib"] = round(free_disk_gib(output_root), 3)
        if code == 0:
            state.update({"state": "complete", "accepted": True, "finished_at_epoch": time.time()})
            state.pop("failure", None)
            atomic_write_json(status_path, state)
            return 0
        if state["free_disk_gib"] < minimum_gib + 0.5:
            state.update({"state": "paused_disk", "accepted": False})
            atomic_write_json(status_path, state)
            continue
        state.update(
            {
                "state": "failed",
                "accepted": False,
                "failure": f"production controller exited with code {code}; automatic retry is disabled for non-disk failures",
                "finished_at_epoch": time.time(),
            }
        )
        atomic_write_json(status_path, state)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
