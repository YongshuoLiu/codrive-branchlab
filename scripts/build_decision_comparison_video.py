#!/usr/bin/env python3
"""Build a synchronized comparison video for one decision-v1 route index.

The video contains one accepted route from every S1-S6 scenario family.  Branches
within a scenario are aligned at the recorded ``decision_started`` simulation
time.  A branch that terminates before the end of the common window holds its
last RGB frame, so terminal outcomes remain visible without changing the shared
timeline.
"""

from __future__ import annotations

import argparse
import bisect
import datetime as dt
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
from typing import Any


DEFAULT_ROOT = pathlib.Path(
    "/home/UNT/yl0826/simlingo/RL/data/counterfactual_decision_v1"
)
DEFAULT_OUTPUT = pathlib.Path(
    "/home/UNT/yl0826/simlingo/RL/videos/decision_v1_route0001_comparison.mp4"
)
FONT = pathlib.Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")

SOURCE_FPS = 4
OUTPUT_FPS = 20
PRE_ROLL_S = 1.0
POST_ROLL_S = 6.0
CLIP_DURATION_S = PRE_ROLL_S + POST_ROLL_S
SAMPLE_COUNT = int(CLIP_DURATION_S * SOURCE_FPS)

DECISIONS_LONGITUDINAL = ["Accelerate", "Maintain", "Brake", "Stop"]
DECISIONS_ALL = DECISIONS_LONGITUDINAL + ["LaneChangeLeft", "LaneChangeRight"]

SCENES = {
    "S1": {
        "route": "closed_occluded_pedestrian_0001",
        "slug": "pedestrian_emergence",
        "title": "行人从遮挡物后出现",
        "english": "Pedestrian Emergence",
        "decisions": DECISIONS_ALL,
    },
    "S2": {
        "route": "closed_lane_cut_in_0001",
        "slug": "vehicle_cut_in",
        "title": "车辆切入",
        "english": "Vehicle Cut-In",
        "decisions": DECISIONS_ALL,
    },
    "S3": {
        "route": "closed_reveal_obstacle_0001",
        "slug": "obstacle_reveal",
        "title": "前车让出后暴露障碍物",
        "english": "Obstacle Reveal",
        "decisions": DECISIONS_ALL,
    },
    "S4": {
        "route": "closed_ghost_driver_left_turn_0001",
        "slug": "left_turn_wrong_way",
        "title": "左转遇逆行／错误方向车辆",
        "english": "Left-Turn Wrong-Way",
        "decisions": DECISIONS_LONGITUDINAL,
    },
    "S5": {
        "route": "closed_ghost_driver_right_turn_0001",
        "slug": "right_turn_wrong_way",
        "title": "右转遇逆行／错误方向车辆",
        "english": "Right-Turn Wrong-Way",
        "decisions": DECISIONS_LONGITUDINAL,
    },
    "S6": {
        "route": "closed_right_turn_on_red_0001",
        "slug": "right_turn_yield",
        "title": "红灯右转让行",
        "english": "Right-Turn Yield",
        "decisions": DECISIONS_LONGITUDINAL,
    },
}

DECISION_LABELS = {
    "Accelerate": "加速  Accelerate",
    "Maintain": "保持  Maintain",
    "Brake": "制动  Brake",
    "Stop": "停车  Stop",
    "LaneChangeLeft": "向左变道  LaneChangeLeft",
    "LaneChangeRight": "向右变道  LaneChangeRight",
}

DECISION_COLORS = {
    "Accelerate": "0x23C483",
    "Maintain": "0x4D9DE0",
    "Brake": "0xF59E42",
    "Stop": "0xF05252",
    "LaneChangeLeft": "0xA78BFA",
    "LaneChangeRight": "0x22D3EE",
}

OUTCOME_LABELS = {
    "completed": "结果：完成",
    "collision": "结果：碰撞终止",
    "lane_unavailable": "结果：无安全车道，拒绝变道",
    "deadlock": "结果：死锁终止",
}

OUTCOME_COLORS = {
    "completed": "0x5EE2A0",
    "collision": "0xFF6B6B",
    "lane_unavailable": "0xFFD166",
    "deadlock": "0xC084FC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=pathlib.Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--separate-output-dir",
        type=pathlib.Path,
        help="Also export one standalone MP4 per S1-S6 scene",
    )
    parser.add_argument(
        "--route-index",
        type=int,
        default=1,
        help="Select the same four-digit route index from every S1-S6 family",
    )
    parser.add_argument(
        "--scene",
        action="append",
        choices=tuple(SCENES),
        help="Export only the selected scene; repeat for multiple scenes",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=21,
        help="libx264 constant-rate-factor (lower is higher quality; default: 21)",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"command failed ({completed.returncode}): {message}")


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_text(path: pathlib.Path, value: str) -> pathlib.Path:
    path.write_text(value, encoding="utf-8")
    return path


def locate_rgb_dir(branch_dir: pathlib.Path) -> pathlib.Path:
    candidates = sorted(branch_dir.glob("data/*/rgb"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one RGB directory in {branch_dir}, found {len(candidates)}"
        )
    return candidates[0]


def load_branch(branch_dir: pathlib.Path) -> dict[str, Any]:
    report_path = branch_dir / "quality_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("accepted"):
        raise RuntimeError(f"branch did not pass quality validation: {branch_dir}")

    events = read_jsonl(branch_dir / "raw" / "decision_events.jsonl")
    starts = [event for event in events if event.get("event") == "decision_started"]
    if len(starts) != 1:
        raise RuntimeError(f"expected one decision_started event in {branch_dir}")
    decision_start_s = float(starts[0]["simulation_time_s"])

    rgb_dir = locate_rgb_dir(branch_dir)
    clocks = read_jsonl(branch_dir / "logs" / "frame_clock.jsonl")
    samples: list[tuple[float, int, pathlib.Path]] = []
    for clock in clocks:
        frame = int(clock["dataset_frame"])
        path = rgb_dir / f"{frame:04d}.jpg"
        if path.is_file():
            samples.append((float(clock["sim_time"]), frame, path))
    samples.sort()
    if not samples:
        raise RuntimeError(f"no clocked RGB frames in {branch_dir}")

    return {
        "branch_dir": branch_dir,
        "report": report,
        "decision_start_s": decision_start_s,
        "samples": samples,
    }


def stage_branch(
    branch: dict[str, Any], stage_dir: pathlib.Path
) -> tuple[list[pathlib.Path], float]:
    stage_dir.mkdir(parents=True)
    samples: list[tuple[float, int, pathlib.Path]] = branch["samples"]
    sample_times = [item[0] for item in samples]
    clip_start_s = branch["decision_start_s"] - PRE_ROLL_S
    selected: list[pathlib.Path] = []

    for index in range(SAMPLE_COUNT):
        desired_s = clip_start_s + index / SOURCE_FPS
        source_index = bisect.bisect_right(sample_times, desired_s + 1e-6) - 1
        source_index = max(0, min(source_index, len(samples) - 1))
        source = samples[source_index][2]
        target = stage_dir / f"{index:04d}.jpg"
        os.symlink(source.resolve(), target)
        selected.append(source)

    last_relative_s = samples[-1][0] - clip_start_s
    end_badge_s = max(0.0, min(CLIP_DURATION_S, last_relative_s + 1 / SOURCE_FPS))
    return selected, end_badge_s


def filter_text(
    text_file: pathlib.Path,
    *,
    fontsize: int,
    x: str,
    y: str,
    color: str = "white",
    enable: str | None = None,
    stroke: bool = False,
) -> str:
    value = (
        f"drawtext=fontfile={FONT}:textfile={text_file}:fontcolor={color}:"
        f"fontsize={fontsize}:x={x}:y={y}"
    )
    if stroke:
        value += ":borderw=2:bordercolor=black@0.85"
    if enable:
        value += f":enable='{enable}'"
    return value


def build_scene_filter(
    scene_id: str,
    scene: dict[str, Any],
    branches: list[dict[str, Any]],
    end_badges: list[float],
    work_dir: pathlib.Path,
) -> str:
    six_way = len(branches) == 6
    if six_way:
        image_w, image_h, cell_h = 620, 310, 405
        positions = [(0, 0), (650, 0), (1300, 0), (0, 420), (650, 420), (1300, 420)]
        pad_x, pad_y = 0, 135
    else:
        image_w, image_h, cell_h = 720, 360, 430
        positions = [(0, 0), (740, 0), (0, 440), (740, 440)]
        pad_x, pad_y = 230, 110

    parts: list[str] = []
    labels: list[str] = []
    for index, (decision, branch, end_s) in enumerate(
        zip(scene["decisions"], branches, end_badges)
    ):
        report = branch["report"]
        outcome = str(report["outcome"])
        decision_file = write_text(
            work_dir / f"{scene_id}_{index}_decision.txt", DECISION_LABELS[decision]
        )
        outcome_file = write_text(
            work_dir / f"{scene_id}_{index}_outcome.txt",
            OUTCOME_LABELS.get(outcome, f"结果：{outcome}"),
        )
        end_file = write_text(work_dir / f"{scene_id}_{index}_end.txt", "末帧冻结  END")

        chain = [
            f"[{index}:v]setpts=PTS-STARTPTS",
            f"scale={image_w}:{image_h}:force_original_aspect_ratio=decrease",
            f"pad={image_w}:{image_h}:(ow-iw)/2:(oh-ih)/2:color=0x000000",
            f"pad={image_w}:{cell_h}:0:0:color=0x101216",
            f"drawbox=x=0:y=0:w=iw:h=ih:color={DECISION_COLORS[decision]}:t=5",
            filter_text(
                decision_file, fontsize=27 if six_way else 30, x="16", y=str(image_h + 9)
            ),
            filter_text(
                outcome_file,
                fontsize=20 if six_way else 22,
                x="16",
                y=str(image_h + (51 if six_way else 48)),
                color=OUTCOME_COLORS.get(outcome, "0xD1D5DB"),
            ),
            (
                f"drawbox=x=iw-205:y=12:w=190:h=38:color=black@0.72:t=fill:"
                f"enable='gte(t,{end_s:.3f})'"
            ),
            filter_text(
                end_file,
                fontsize=19,
                x="w-192",
                y="18",
                color="0xF3F4F6",
                enable=f"gte(t,{end_s:.3f})",
            ),
        ]
        parts.append(",".join(chain) + f"[cell{index}]")
        labels.append(f"[cell{index}]")

    layout = "|".join(f"{x}_{y}" for x, y in positions)
    parts.append(
        "".join(labels)
        + f"xstack=inputs={len(labels)}:layout={layout}:fill=0x0B0D10[grid]"
    )

    title_file = write_text(
        work_dir / f"{scene_id}_title.txt",
        f"{scene_id}  {scene['title']}  ·  {scene['english']}",
    )
    route_file = write_text(
        work_dir / f"{scene_id}_route.txt", f"route: {scene['route']}"
    )
    footer_file = write_text(
        work_dir / f"{scene_id}_footer.txt",
        "同步窗口：t=-1 s  →  t=0 决策开始  →  t=+6 s    ｜    早终止轨迹保留末帧",
    )
    onset_file = write_text(
        work_dir / f"{scene_id}_onset.txt", "t=0  DECISION START  ·  决策开始"
    )

    global_chain = [
        f"[grid]pad=1920:1080:{pad_x}:{pad_y}:color=0x0B0D10",
        filter_text(title_file, fontsize=43, x="(w-text_w)/2", y="22", stroke=True),
        filter_text(route_file, fontsize=22, x="(w-text_w)/2", y="82", color="0xAAB4C0"),
        "drawbox=x=0:y=970:w=iw:h=48:color=0xD7263D@0.92:t=fill:enable='between(t,0.95,1.55)'",
        filter_text(
            onset_file,
            fontsize=29,
            x="(w-text_w)/2",
            y="977",
            enable="between(t,0.95,1.55)",
            stroke=True,
        ),
        filter_text(
            footer_file,
            fontsize=22,
            x="(w-text_w)/2",
            y="1032",
            color="0xD8DEE6",
        ),
        f"drawbox=x=0:y=1070:w='iw*t/{CLIP_DURATION_S}':h=10:color=0x2DD4BF:t=fill",
        f"tpad=stop_mode=clone:stop_duration={1 / SOURCE_FPS}",
        f"fps={OUTPUT_FPS}",
        "format=yuv420p[out]",
    ]
    parts.append(",".join(global_chain))
    return ";\n".join(parts)


def make_scene(
    scene_id: str,
    scene: dict[str, Any],
    root: pathlib.Path,
    work_dir: pathlib.Path,
    crf: int,
) -> tuple[pathlib.Path, dict[str, Any]]:
    route_dir = root / scene_id / scene["route"]
    if not route_dir.is_dir():
        raise RuntimeError(f"missing selected route: {route_dir}")

    branches: list[dict[str, Any]] = []
    end_badges: list[float] = []
    branch_manifest: list[dict[str, Any]] = []
    inputs: list[str] = []
    for decision in scene["decisions"]:
        branch = load_branch(route_dir / decision)
        if branch["report"].get("decision") != decision:
            raise RuntimeError(f"decision mismatch in {route_dir / decision}")
        stage_dir = work_dir / scene_id / decision
        selected, end_badge_s = stage_branch(branch, stage_dir)
        branches.append(branch)
        end_badges.append(end_badge_s)
        samples = branch["samples"]
        branch_manifest.append(
            {
                "decision": decision,
                "decision_label_zh": DECISION_LABELS[decision].split("  ")[0],
                "outcome": branch["report"]["outcome"],
                "accepted": True,
                "source_branch": str(branch["branch_dir"]),
                "source_rgb_dir": str(locate_rgb_dir(branch["branch_dir"])),
                "source_frame_count": len(samples),
                "source_first_frame": samples[0][1],
                "source_last_frame": samples[-1][1],
                "source_first_sim_time_s": samples[0][0],
                "source_last_sim_time_s": samples[-1][0],
                "decision_start_sim_time_s": branch["decision_start_s"],
                "last_frame_relative_to_decision_s": (
                    samples[-1][0] - branch["decision_start_s"]
                ),
                "video_end_badge_time_s": end_badge_s,
                "video_first_source_frame": selected[0].name,
                "video_last_source_frame": selected[-1].name,
            }
        )
        inputs.extend(["-framerate", str(SOURCE_FPS), "-i", str(stage_dir / "%04d.jpg")])

    filter_path = work_dir / f"filter_{scene_id}.txt"
    filter_path.write_text(
        build_scene_filter(scene_id, scene, branches, end_badges, work_dir),
        encoding="utf-8",
    )
    output = work_dir / f"scene_{scene_id}.mp4"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    command.extend(inputs)
    command.extend(
        [
            "-filter_complex_script",
            str(filter_path),
            "-map",
            "[out]",
            "-t",
            str(CLIP_DURATION_S),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    run(command)
    print(f"built {scene_id}: {scene['route']}", flush=True)
    return output, {
        "scene_id": scene_id,
        "title_zh": scene["title"],
        "title_en": scene["english"],
        "route_id": scene["route"],
        "branches": branch_manifest,
    }


def probe(path: pathlib.Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_frames:format=duration,size",
            "-of",
            "json",
            str(path),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(completed.stdout)


def main() -> int:
    args = parse_args()
    if args.route_index < 0 or args.route_index > 9999:
        raise RuntimeError("route index must be between 0 and 9999")
    scenes = {
        scene_id: {
            **scene,
            "route": f"{scene['route'].rsplit('_', 1)[0]}_{args.route_index:04d}",
        }
        for scene_id, scene in SCENES.items()
    }
    if args.scene:
        selected_scenes = set(args.scene)
        scenes = {
            scene_id: scene
            for scene_id, scene in scenes.items()
            if scene_id in selected_scenes
        }
    if not FONT.is_file():
        raise RuntimeError(f"required CJK font is missing: {FONT}")
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"required executable is missing: {executable}")
    if not args.input_root.is_dir():
        raise RuntimeError(f"input root does not exist: {args.input_root}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.separate_output_dir:
        args.separate_output_dir.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.stem + ".partial.mp4")
    manifest_path = args.output.with_suffix(".manifest.json")

    with tempfile.TemporaryDirectory(
        prefix=".decision_video_build_", dir=args.output.parent
    ) as temporary:
        work_dir = pathlib.Path(temporary)
        scene_clips: list[pathlib.Path] = []
        scene_manifest: list[dict[str, Any]] = []
        for scene_id, scene in scenes.items():
            clip, metadata = make_scene(
                scene_id, scene, args.input_root, work_dir, args.crf
            )
            if args.separate_output_dir:
                scene_output = args.separate_output_dir / (
                    f"{scene_id}_{scene['slug']}_route{args.route_index:04d}.mp4"
                )
                scene_partial = scene_output.with_suffix(".partial.mp4")
                shutil.copy2(clip, scene_partial)
                os.replace(scene_partial, scene_output)
                metadata["output_video"] = str(scene_output.resolve())
                metadata["probe"] = probe(scene_output)
            scene_clips.append(clip)
            scene_manifest.append(metadata)

        concat_path = work_dir / "concat.txt"
        concat_path.write_text(
            "".join(f"file '{path}'\n" for path in scene_clips), encoding="utf-8"
        )
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_path),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(partial),
            ]
        )
        os.replace(partial, args.output)

    metadata = {
        "schema_version": "decision_comparison_video_v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_root": str(args.input_root.resolve()),
        "output_video": str(args.output.resolve()),
        "selection_rule": (
            f"one accepted route for each selected scene {list(scenes)}; "
            f"frozen route index {args.route_index:04d}"
        ),
        "alignment": {
            "event": "decision_started",
            "pre_roll_s": PRE_ROLL_S,
            "post_roll_s": POST_ROLL_S,
            "source_fps": SOURCE_FPS,
            "output_fps": OUTPUT_FPS,
            "early_termination_policy": "hold final RGB frame and show END badge",
        },
        "scenes": scene_manifest,
        "probe": probe(args.output),
    }
    temporary_manifest = manifest_path.with_suffix(".partial.json")
    temporary_manifest.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_path)
    if args.separate_output_dir:
        notion_manifest_path = args.separate_output_dir / "notion_video_manifest.json"
        notion_manifest = {
            "schema_version": "notion_decision_video_bundle_v1",
            "created_at_utc": metadata["created_at_utc"],
            "route_index": args.route_index,
            "compatibility": {
                "container": "MP4",
                "video_codec": "H.264",
                "pixel_format": "yuv420p",
                "resolution": "1920x1080",
                "frame_rate": OUTPUT_FPS,
                "audio": False,
                "faststart": True,
                "maximum_target_file_bytes": 5000000,
            },
            "scenes": scene_manifest,
        }
        notion_partial = notion_manifest_path.with_suffix(".partial.json")
        notion_partial.write_text(
            json.dumps(notion_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(notion_partial, notion_manifest_path)
        print(f"notion manifest: {notion_manifest_path}", flush=True)
    print(f"video: {args.output}", flush=True)
    print(f"manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
