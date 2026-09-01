#!/usr/bin/env python3
"""Live terminal dashboard for a counterfactual CARLA collection campaign."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


RL_ROOT = Path(__file__).resolve().parents[1]
STAGE_LABELS = {
    "waiting": "等待",
    "startup_stagger": "错峰启动",
    "preparing": "准备目录",
    "quarantining_previous_attempt": "隔离旧尝试",
    "launching_carla": "启动 CARLA",
    "carla_simulation": "CARLA 仿真",
    "pruning_derived_modalities": "清理派生模态",
    "validating_branch": "质量验证",
    "technical_retry_backoff": "技术重试等待",
    "idle": "切换任务",
    "disk_gate": "磁盘门禁暂停",
    "finished": "本阶段完成",
}
SCENE_NAMES = {
    "S1": "行人突然出现",
    "S2": "车辆切入",
    "S3": "障碍物显现",
    "S4": "左转逆行车",
    "S5": "右转逆行车",
    "S6": "右转让行",
}


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def pid_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def port_open(port: Any) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=0.08):
            return True
    except (OSError, TypeError, ValueError):
        return False


def line_count(path: Path) -> int:
    try:
        with path.open("rb") as stream:
            return sum(chunk.count(b"\n") for chunk in iter(lambda: stream.read(65536), b""))
    except OSError:
        return 0


def last_json_row(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            position = stream.tell()
            block = bytearray()
            while position > 0 and block.count(b"\n") < 3:
                size = min(4096, position)
                position -= size
                stream.seek(position)
                block[:0] = stream.read(size)
        for raw in reversed(block.splitlines()):
            if raw.strip():
                return json.loads(raw.decode("utf-8", errors="replace"))
    except (OSError, ValueError):
        pass
    return {}


def human_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds == float("inf"):
        return "--"
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d {hours:02d}h"
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def human_bytes(value: float) -> str:
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or suffix == "TiB":
            return f"{value:.1f}{suffix}"
        value /= 1024.0
    return f"{value:.1f}TiB"


def progress_bar(done: int, total: int, width: int = 24) -> str:
    ratio = min(max(done / total if total else 0.0, 0.0), 1.0)
    filled = int(ratio * width)
    partial = int((ratio * width - filled) * 8)
    ticks = " ▏▎▍▌▋▊▉█"
    body = "█" * filled
    if filled < width and partial:
        body += ticks[partial]
        filled += 1
    return body + "░" * (width - filled)


def table(headers: list[str], rows: list[list[Any]], widths: list[int]) -> list[str]:
    def display_width(text: str) -> int:
        return sum(
            2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
            for character in text
        )

    def truncate(text: str, width: int) -> str:
        if display_width(text) <= width:
            return text
        result = ""
        for character in text:
            if display_width(result + character + "…") > width:
                break
            result += character
        return result + "…"

    def cell(value: Any, width: int) -> str:
        text = truncate(str(value), width)
        return " " + text + " " * (width - display_width(text) + 1)

    top = "┌" + "┬".join("─" * (width + 2) for width in widths) + "┐"
    mid = "├" + "┼".join("─" * (width + 2) for width in widths) + "┤"
    bottom = "└" + "┴".join("─" * (width + 2) for width in widths) + "┘"
    output = [top, "│" + "│".join(cell(v, w) for v, w in zip(headers, widths)) + "│", mid]
    output.extend(
        "│" + "│".join(cell(v, w) for v, w in zip(row, widths)) + "│"
        for row in rows
    )
    output.append(bottom)
    return output


def gpu_rows() -> dict[int, dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.run(
            command, capture_output=True, text=True, timeout=2.0, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    result = {}
    for line in output.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 5:
            continue
        result[int(fields[0])] = {
            "memory": f"{int(fields[1]):>5}/{int(fields[2]):<5} MiB",
            "util": f"{fields[3]}%",
            "temp": f"{fields[4]}°C",
        }
    return result


def campaign_counts(jobs: list[dict[str, Any]]) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]], list[float]]:
    scenes: dict[str, Counter] = defaultdict(Counter)
    decisions: dict[str, Counter] = defaultdict(Counter)
    accepted_mtimes = []
    for job in jobs:
        scene = str(job["scenario_id"])
        decision = str(job["decision"])
        report_path = Path(job["output_dir"]) / "quality_report.json"
        report = read_json(report_path, {}) or {}
        run_spec = read_json(Path(job["output_dir"]) / "metadata/run_spec.json", {}) or {}
        route_spec = run_spec.get("route") or {}
        contract_matches = route_spec.get("route_sha256") == job.get("route_sha256")
        scenes[scene]["total"] += 1
        decisions[decision]["total"] += 1
        if report.get("accepted") is True and contract_matches:
            state = "accepted"
            try:
                accepted_mtimes.append(report_path.stat().st_mtime)
            except OSError:
                pass
        elif report.get("accepted") is True:
            state = "pending"
            scenes[scene]["stale"] += 1
            decisions[decision]["stale"] += 1
        elif report_path.exists():
            state = "rejected"
        else:
            state = "pending"
        scenes[scene][state] += 1
        decisions[decision][state] += 1
        outcome = str(report.get("outcome", ""))
        if outcome:
            scenes[scene][f"outcome:{outcome}"] += 1
            decisions[decision][f"outcome:{outcome}"] += 1
    return scenes, decisions, accepted_mtimes


def group_counts(jobs: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        groups[str(job["scenario_id"])].append(job)
    output = {}
    for scene, rows in groups.items():
        by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_group[str(row["group_id"])].append(row)
        complete = sum(
            all((read_json(Path(job["output_dir"]) / "quality_report.json", {}) or {}).get("accepted") is True for job in group)
            for group in by_group.values()
        )
        output[scene] = (complete, len(by_group))
    return output


def render(
    config: dict[str, Any],
    jobs: list[dict[str, Any]],
    runtime: dict[str, Any],
    production: dict[str, Any],
    supervisor: dict[str, Any],
) -> str:
    now = time.time()
    scenes, decisions, accepted_mtimes = campaign_counts(jobs)
    groups = group_counts(jobs)
    total = len(jobs)
    accepted = sum(row["accepted"] for row in scenes.values())
    rejected = sum(row["rejected"] for row in scenes.values())
    pending = total - accepted - rejected
    started = supervisor.get("started_at_epoch") or production.get("started_at_epoch") or runtime.get("started_at_epoch")
    accepted_since_start = sum(mtime >= float(started or now) for mtime in accepted_mtimes)
    elapsed = now - float(started) if started else 0.0
    rate = accepted_since_start / (elapsed / 3600.0) if elapsed >= 60 and accepted_since_start else 0.0
    eta = pending / rate * 3600.0 if rate > 0 else None
    output_root = Path(config["output_root"])
    disk = shutil.disk_usage(output_root)
    safety_gib = float(config.get("quality_thresholds", {}).get("minimum_free_disk_gb", 20.0))
    estimate_mib = float(config.get("dashboard_estimated_branch_mib", 49.9))
    estimated_remaining = pending * estimate_mib * 1024 * 1024
    stage = production.get("current_stage") or supervisor.get("state") or runtime.get("phase", "等待启动")
    supervisor_pid = supervisor.get("supervisor_pid")
    producer_pid = supervisor.get("producer_pid")
    process_state = (
        f"supervisor={supervisor_pid or '--'}:{'alive' if pid_alive(supervisor_pid) else 'down'}  "
        f"producer={producer_pid or '--'}:{'alive' if pid_alive(producer_pid) else 'down'}"
    )

    lines = [
        "╔════════════════════════════════ COUNTERFACTUAL CARLA 量产面板 ════════════════════════════════╗",
        f"  更新时间 {time.strftime('%Y-%m-%d %H:%M:%S')}   阶段 {stage}   {process_state}",
        f"  总进度 [{progress_bar(accepted, total, 42)}] {accepted:4d}/{total}  {accepted / total * 100 if total else 0:6.2f}%",
        f"  接受 {accepted}   拒绝待修 {rejected}   尚未完成 {pending}   本轮新增 {accepted_since_start}   速度 {rate:.1f} branches/h   ETA {human_duration(eta)}",
        f"  磁盘 free={human_bytes(disk.free)} / total={human_bytes(disk.total)}   预计剩余≈{human_bytes(estimated_remaining)}   安全门限={safety_gib:.1f}GiB",
        "╚══════════════════════════════════════════════════════════════════════════════════════════════════╝",
        "",
        "按场景统计（route 完成表示该 route 的所有 decision 均验收通过）",
    ]
    scene_rows = []
    for scene in sorted(scenes):
        row = scenes[scene]
        complete, group_total = groups.get(scene, (0, 0))
        done = int(row["accepted"])
        scene_rows.append(
            [
                scene,
                SCENE_NAMES.get(scene, scene),
                f"{complete}/{group_total}",
                f"{done}/{row['total']}",
                f"{done / row['total'] * 100 if row['total'] else 0:.1f}%",
                row["rejected"],
                row["pending"],
                progress_bar(done, int(row["total"]), 18),
            ]
        )
    lines.extend(table(["场景", "类别", "完整 route", "接受分支", "完成率", "拒绝", "等待", "进度"], scene_rows, [4, 14, 11, 11, 8, 6, 6, 18]))

    lines.extend(["", "按 decision 统计"])
    decision_rows = []
    for decision in sorted(decisions):
        row = decisions[decision]
        decision_rows.append(
            [decision, f"{row['accepted']}/{row['total']}", row["rejected"], row["pending"], progress_bar(int(row["accepted"]), int(row["total"]), 20)]
        )
    lines.extend(table(["Decision", "接受", "拒绝", "等待", "进度"], decision_rows, [18, 12, 7, 7, 20]))

    lines.extend(["", "GPU / CARLA worker 实时状态"])
    gpus = gpu_rows()
    worker_rows = []
    for slot, worker in sorted((runtime.get("workers") or {}).items(), key=lambda item: int(item[0])):
        current = worker.get("current_job") or {}
        run_dir = Path(current["output_dir"]) if current.get("output_dir") else None
        carla_pid = None
        frames = 0
        sim_time = None
        event = "--"
        if run_dir:
            try:
                carla_pid = int((run_dir / "logs/carla.pid").read_text().strip())
            except (OSError, ValueError):
                carla_pid = None
            clock = run_dir / "logs/frame_clock.jsonl"
            frames = line_count(clock)
            sim_time = last_json_row(clock).get("sim_time")
            event = last_json_row(run_dir / "raw/decision_events.jsonl").get("event", "--")
        gpu = gpus.get(int(worker["gpu"]), {})
        elapsed_job = now - float(current.get("started_at_epoch", now)) if current else 0
        carla_state = f"pid {carla_pid}:{'up' if pid_alive(carla_pid) else 'down'}" if carla_pid else ("port up" if port_open(worker.get("port")) else "--")
        worker_rows.append(
            [
                slot,
                worker.get("gpu"),
                gpu.get("util", "--"),
                gpu.get("memory", "--"),
                worker.get("port"),
                STAGE_LABELS.get(worker.get("stage"), worker.get("stage", "--")),
                current.get("job_id", "--"),
                f"{frames} / {float(sim_time):.1f}s" if sim_time is not None else str(frames),
                event,
                human_duration(elapsed_job) if current else "--",
                carla_state,
            ]
        )
    if not worker_rows:
        for worker in config.get("workers", []):
            gpu = gpus.get(int(worker["gpu"]), {})
            worker_rows.append([worker["slot"], worker["gpu"], gpu.get("util", "--"), gpu.get("memory", "--"), worker["port"], "等待启动", "--", "--", "--", "--", "port up" if port_open(worker["port"]) else "--"])
    lines.extend(table(["W", "GPU", "利用", "显存", "Port", "阶段", "当前 job", "帧/仿真时", "最近事件", "耗时", "CARLA"], worker_rows, [2, 3, 6, 17, 5, 12, 31, 12, 22, 8, 15]))

    recent = runtime.get("recent_results") or []
    if recent:
        lines.extend(["", "最近完成"])
        for item in reversed(recent[-8:]):
            flag = "✓" if item.get("accepted") else "✗"
            lines.append(f"  {flag} {item.get('job_id')}  outcome={item.get('outcome')}  status={item.get('status')}")
    failure = supervisor.get("failure") or production.get("failure")
    if failure:
        lines.extend(["", f"警告: {failure}"])
    return "\n".join(lines) + "\n"


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime-progress", type=Path)
    parser.add_argument("--production-status", type=Path)
    parser.add_argument("--supervisor-status", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config = read_json(args.config.resolve(), {}) or {}
    jobs = read_jsonl(args.manifest.resolve())
    production_dir = Path(config["output_root"]).resolve() / "_production" / args.manifest.stem
    runtime_path = (args.runtime_progress or production_dir / "runtime_progress.json").resolve()
    production_path = (args.production_status or production_dir / "production_status.json").resolve()
    supervisor_path = (args.supervisor_status or production_dir / "supervisor_status.json").resolve()
    snapshot_path = (args.snapshot or production_dir / "dashboard_snapshot.txt").resolve()
    allowed_root = Path(config["output_root"]).resolve()
    if allowed_root not in snapshot_path.parents:
        raise SystemExit(f"snapshot must stay under {allowed_root}")

    interactive = sys.stdout.isatty() and not args.quiet
    try:
        while True:
            dashboard = render(
                config,
                jobs,
                read_json(runtime_path, {}) or {},
                read_json(production_path, {}) or {},
                read_json(supervisor_path, {}) or {},
            )
            atomic_write_text(snapshot_path, dashboard)
            if not args.quiet:
                if interactive:
                    sys.stdout.write("\033[2J\033[H")
                sys.stdout.write(dashboard)
                sys.stdout.flush()
            if args.once:
                break
            time.sleep(max(args.interval, 0.5))
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
