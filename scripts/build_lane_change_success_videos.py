#!/usr/bin/env python3
"""Build Notion-ready videos for verified S2/S3 lane-change examples."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import shutil
import subprocess
import tempfile

from build_decision_comparison_video import (
    CLIP_DURATION_S,
    DECISION_COLORS,
    FONT,
    OUTPUT_FPS,
    SOURCE_FPS,
    load_branch,
    probe,
    stage_branch,
    write_text,
)


RL_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_INPUT = RL_ROOT / "data/counterfactual_decision_v1"
DEFAULT_OUTPUT = RL_ROOT / "videos/lane_change_success_cases"

CASES = (
    {
        "case_id": "S2_left",
        "scene_id": "S2",
        "scene_zh": "车辆切入",
        "scene_en": "Vehicle Cut-In",
        "route_id": "closed_lane_cut_in_0009",
        "decision": "LaneChangeLeft",
        "decision_zh": "向左变道",
        "expected_outcome": "collision",
        "status_zh": "到达目标车道；后续碰撞终止",
        "filename": "S2_left_lane_change_reached_then_collision_route0009.mp4",
    },
    {
        "case_id": "S2_right",
        "scene_id": "S2",
        "scene_zh": "车辆切入",
        "scene_en": "Vehicle Cut-In",
        "route_id": "closed_lane_cut_in_0002",
        "decision": "LaneChangeRight",
        "decision_zh": "向右变道",
        "expected_outcome": "completed",
        "status_zh": "变道成功；场景完成",
        "filename": "S2_right_lane_change_completed_route0002.mp4",
    },
    {
        "case_id": "S3_left",
        "scene_id": "S3",
        "scene_zh": "前车让出后暴露障碍物",
        "scene_en": "Obstacle Reveal",
        "route_id": "closed_reveal_obstacle_0002",
        "decision": "LaneChangeLeft",
        "decision_zh": "向左变道",
        "expected_outcome": "completed",
        "status_zh": "变道成功；场景完成",
        "filename": "S3_left_lane_change_completed_route0002.mp4",
    },
    {
        "case_id": "S3_right",
        "scene_id": "S3",
        "scene_zh": "前车让出后暴露障碍物",
        "scene_en": "Obstacle Reveal",
        "route_id": "closed_reveal_obstacle_0002",
        "decision": "LaneChangeRight",
        "decision_zh": "向右变道",
        "expected_outcome": "completed",
        "status_zh": "变道成功；场景完成",
        "filename": "S3_right_lane_change_completed_route0002.mp4",
    },
)


def drawtext(path: pathlib.Path, size: int, x: str, y: str, color: str = "white", enable: str | None = None) -> str:
    result = (
        f"drawtext=fontfile={FONT}:textfile={path}:fontcolor={color}:"
        f"fontsize={size}:x={x}:y={y}:borderw=2:bordercolor=black@0.85"
    )
    if enable:
        result += f":enable='{enable}'"
    return result


def run(command: list[str]) -> None:
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())


def build_case(case: dict, input_root: pathlib.Path, output_root: pathlib.Path, work_root: pathlib.Path, crf: int) -> dict:
    branch_dir = input_root / case["scene_id"] / case["route_id"] / case["decision"]
    branch = load_branch(branch_dir)
    report = branch["report"]
    fidelity = report.get("decision_fidelity", {})
    if report.get("decision") != case["decision"]:
        raise RuntimeError(f"decision mismatch: {branch_dir}")
    if report.get("outcome") != case["expected_outcome"]:
        raise RuntimeError(f"outcome mismatch: {branch_dir}")
    if fidelity.get("lane_audit_safe") is not True:
        raise RuntimeError(f"lane audit did not pass: {branch_dir}")
    if int(fidelity.get("target_lane_reached_frames") or 0) <= 0:
        raise RuntimeError(f"target lane was not reached: {branch_dir}")
    if float(fidelity.get("target_lane_hold_s") or 0.0) <= 0.0:
        raise RuntimeError(f"target lane was not held: {branch_dir}")

    case_work = work_root / case["case_id"]
    stage_dir = case_work / "frames"
    _, end_badge_s = stage_branch(branch, stage_dir)
    title = write_text(
        case_work / "title.txt",
        f"{case['scene_id']}  {case['scene_zh']}  ·  {case['scene_en']}",
    )
    route = write_text(case_work / "route.txt", f"route: {case['route_id']}")
    decision = write_text(
        case_work / "decision.txt",
        f"{case['decision_zh']}  {case['decision']}",
    )
    status = write_text(case_work / "status.txt", case["status_zh"])
    metrics = write_text(
        case_work / "metrics.txt",
        (
            f"目标车道保持 {float(fidelity['target_lane_hold_s']):.2f} s"
            f"  ·  到达帧 {int(fidelity['target_lane_reached_frames'])}"
        ),
    )
    onset = write_text(case_work / "onset.txt", "t=0  DECISION START  ·  变道开始")
    end = write_text(case_work / "end.txt", "末帧冻结  END")

    outcome_color = "0xFF6B6B" if report["outcome"] == "collision" else "0x5EE2A0"
    filters = [
        "[0:v]setpts=PTS-STARTPTS",
        "scale=1760:880:force_original_aspect_ratio=decrease",
        "pad=1760:880:(ow-iw)/2:(oh-ih)/2:color=0x000000",
        f"drawbox=x=0:y=0:w=iw:h=ih:color={DECISION_COLORS[case['decision']]}:t=7",
        "pad=1920:1080:80:110:color=0x0B0D10",
        drawtext(title, 43, "(w-text_w)/2", "20"),
        drawtext(route, 22, "(w-text_w)/2", "75", "0xAAB4C0"),
        "drawbox=x=0:y=995:w=iw:h=85:color=0x101216:t=fill",
        drawtext(decision, 31, "100", "1007", DECISION_COLORS[case["decision"]]),
        drawtext(metrics, 23, "100", "1048", "0xD8DEE6"),
        drawtext(status, 28, "w-text_w-100", "1019", outcome_color),
        "drawbox=x=0:y=920:w=iw:h=54:color=0xD7263D@0.92:t=fill:enable='between(t,0.95,1.55)'",
        drawtext(onset, 30, "(w-text_w)/2", "930", "white", "between(t,0.95,1.55)"),
        f"drawbox=x=1510:y=130:w=315:h=48:color=black@0.72:t=fill:enable='gte(t,{end_badge_s:.3f})'",
        drawtext(end, 22, "1530", "140", "0xF3F4F6", f"gte(t,{end_badge_s:.3f})"),
        f"drawbox=x=0:y=1070:w='iw*t/{CLIP_DURATION_S}':h=10:color=0x2DD4BF:t=fill",
        f"tpad=stop_mode=clone:stop_duration={1 / SOURCE_FPS}",
        f"fps={OUTPUT_FPS}",
        "format=yuv420p[out]",
    ]
    filter_path = case_work / "filter.txt"
    filter_path.write_text(",".join(filters), encoding="utf-8")

    output = output_root / case["filename"]
    partial = output.with_suffix(".partial.mp4")
    run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-framerate", str(SOURCE_FPS), "-i", str(stage_dir / "%04d.jpg"),
            "-filter_complex_script", str(filter_path), "-map", "[out]",
            "-t", str(CLIP_DURATION_S), "-an", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(partial),
        ]
    )
    os.replace(partial, output)
    return {
        **case,
        "output_video": str(output.resolve()),
        "source_branch": str(branch_dir.resolve()),
        "accepted": True,
        "outcome": report["outcome"],
        "lane_audit_safe": True,
        "target_lane_reached_frames": int(fidelity["target_lane_reached_frames"]),
        "target_lane_hold_s": float(fidelity["target_lane_hold_s"]),
        "same_direction_ratio": fidelity.get("same_direction_ratio"),
        "lane_marking_violation_ratio": fidelity.get("lane_marking_violation_ratio"),
        "probe": probe(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=pathlib.Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--crf", type=int, default=23)
    args = parser.parse_args()
    if not FONT.is_file() or shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("required CJK font/ffmpeg/ffprobe is unavailable")
    args.output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lane_success_", dir=args.output_root) as tmp:
        work_root = pathlib.Path(tmp)
        rows = [build_case(case, args.input_root, args.output_root, work_root, args.crf) for case in CASES]
    manifest = {
        "schema_version": "lane_change_success_video_bundle_v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "success_definition": "accepted branch, safe lane audit, route applied, target lane reached and held",
        "notion_format": {
            "container": "MP4", "codec": "H.264", "pixel_format": "yuv420p",
            "resolution": "1920x1080", "fps": OUTPUT_FPS, "audio": False,
            "faststart": True, "maximum_file_bytes": 5_000_000,
        },
        "cases": rows,
    }
    manifest_path = args.output_root / "lane_change_success_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"videos": len(rows), "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
