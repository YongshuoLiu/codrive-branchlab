#!/usr/bin/env python3
"""Continuous terminal dashboard for strict S1 pedestrian-speed V2 progress."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


RL_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = RL_ROOT / "data/counterfactual_decision_v1/_recollected/S1_PED_SPEED_V2"
QUEUE_STATUS = CAMPAIGN_ROOT / "_production/queue_status.json"
RUNTIME_PROGRESS = CAMPAIGN_ROOT / "_production/runtime_progress.json"
AUDIT_STATUS = CAMPAIGN_ROOT / "recollection_status.json"
AUDIT_SCRIPT = RL_ROOT / "scripts/audit_s1_ped_speed_v2.py"

STATE_LABELS = {
    "waiting_for_upstream": "等待原始量产任务释放 GPU",
    "waiting_for_ports": "等待 CARLA 端口释放",
    "collecting": "S1 V2 正在采集",
    "complete": "采集器已结束",
    "failed": "采集器异常退出",
}


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}


def progress_bar(done: int, total: int, width: int = 44) -> str:
    ratio = min(max(done / total if total else 0.0, 0.0), 1.0)
    exact = ratio * width
    full = int(exact)
    partial = int((exact - full) * 8)
    ticks = " ▏▎▍▌▋▊▉█"
    body = "█" * full
    if full < width and partial:
        body += ticks[partial]
        full += 1
    return body + "░" * (width - full)


def render() -> str:
    subprocess.run(
        [sys.executable, str(AUDIT_SCRIPT)],
        cwd=RL_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    queue = read_json(QUEUE_STATUS)
    audit = read_json(AUDIT_STATUS)
    runtime = read_json(RUNTIME_PROGRESS)
    branches_done = int(audit.get("complete_branches", 0))
    branches_total = int(audit.get("required_branches", 0))
    routes_done = int(audit.get("complete_routes", 0))
    routes_total = int(audit.get("route_count", 0))
    branch_pct = branches_done / branches_total * 100 if branches_total else 0.0
    route_pct = routes_done / routes_total * 100 if routes_total else 0.0
    state = str(queue.get("state", "unknown"))
    lines = [
        "╔════════════════════════ S1 行人速度 V2 重采面板 ════════════════════════╗",
        f"  更新时间  {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"  当前状态  {STATE_LABELS.get(state, state)}",
        f"  行人速度  {queue.get('target_ped_speed_mps', 8.5)} m/s    策略 {queue.get('target_policy_version', 'S1_PED_SPEED_V2')}",
        "",
        f"  分支进度  [{progress_bar(branches_done, branches_total)}]",
        f"             {branches_done:3d}/{branches_total:<3d}  {branch_pct:6.2f}%",
        f"  Route进度 [{progress_bar(routes_done, routes_total)}]",
        f"             {routes_done:3d}/{routes_total:<3d}  {route_pct:6.2f}%",
        "",
        f"  完整 route {routes_done}    部分完成 {audit.get('partial_routes', 0)}    待采 {audit.get('pending_routes', 0)}",
        f"  异常安全分支 {audit.get('unexpected_safe_v2_branches', 0)}（必须为 0 才能通过）",
        "╚════════════════════════════════════════════════════════════════════════╝",
    ]
    if state == "waiting_for_upstream":
        active = queue.get("upstream_active_jobs") or []
        lines.extend(
            [
                "",
                f"原始量产阶段: {queue.get('upstream_phase', '--')}    collector alive={queue.get('upstream_alive', False)}",
                "正在占用 GPU: " + (" | ".join(active) if active else "等待状态刷新"),
            ]
        )
    elif state == "waiting_for_ports":
        lines.extend(["", f"占用中的端口: {queue.get('occupied_ports', [])}"])
    elif state in {"collecting", "complete", "failed"}:
        lines.extend(
            [
                "",
                f"采集器 PID: {queue.get('collector_pid', '--')}    phase={queue.get('collector_phase', runtime.get('phase', '--'))}",
                f"运行结果: {queue.get('result_counts', runtime.get('result_counts', {}))}",
                f"场景结果: {queue.get('outcome_counts', runtime.get('outcome_counts', {}))}",
            ]
        )
        workers = runtime.get("workers") or {}
        if workers:
            lines.extend(["", "Worker 实时状态"])
            for slot, worker in sorted(workers.items(), key=lambda item: int(item[0])):
                current = worker.get("current_job") or {}
                lines.append(
                    f"  W{slot} GPU{worker.get('gpu', '--')}  "
                    f"{worker.get('stage', worker.get('state', '--')):<22} "
                    f"{current.get('job_id', '--')}"
                )
    lines.extend(["", "严格完成条件：quality accepted + S1_PED_SPEED_V2 + ped_speed≥8.5 + 行人碰撞", "按 Ctrl+C 退出监视"])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    try:
        while True:
            sys.stdout.write("\033[2J\033[H" + render() + "\n")
            sys.stdout.flush()
            if args.once:
                return 0
            time.sleep(max(args.interval, 1.0))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
