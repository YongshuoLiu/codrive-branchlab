#!/usr/bin/env python3
"""Build a synchronized S1 decision video with front-lane pedestrian scores.

Accelerate and Maintain are read from the S1 pedestrian-speed V2 recollection.
The other decisions are read from the retained baseline counterfactual dataset.
Pedestrian proximity risk is counted only when the pedestrian is:

* ahead of the ego vehicle along the ego lane heading;
* physically on a driving lane; and
* within the ego lane or either immediately adjacent lane.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import pathlib
import shutil
import statistics
import subprocess
import tempfile
from typing import Any

from build_decision_comparison_video import (
    DECISION_COLORS,
    FONT,
    OUTPUT_FPS,
    SOURCE_FPS,
    filter_text,
    load_branch,
    probe,
    run,
    stage_branch,
    write_text,
)
from score_decision_reward_v1 import (
    actor_radius,
    base_branch,
    clip,
    evaluation_start,
    finalize_group,
    hazard_clear_index,
    percentile,
    read_telemetry,
    vector,
)


RL_ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE_ROOT = RL_ROOT / "data/counterfactual_decision_v1"
V2_ROOT = BASE_ROOT / "_recollected/S1_PED_SPEED_V2"
DEFAULT_OUTPUT = (
    RL_ROOT
    / "videos/s1_ped_speed_v2_failures/S1_route0002_all_decisions_scored.mp4"
)

DECISIONS = (
    "Accelerate",
    "Maintain",
    "Brake",
    "Stop",
    "LaneChangeLeft",
    "LaneChangeRight",
)

DECISION_LABELS = {
    "Accelerate": "加速  Accelerate",
    "Maintain": "保持  Maintain",
    "Brake": "制动  Brake",
    "Stop": "停车  Stop",
    "LaneChangeLeft": "向左变道  LaneChangeLeft",
    "LaneChangeRight": "向右变道  LaneChangeRight",
}

OUTCOME_LABELS = {
    "completed": "完成",
    "collision": "碰撞终止",
    "lane_unavailable": "无安全车道",
    "deadlock": "死锁终止",
}

OUTCOME_COLORS = {
    "completed": "0x5EE2A0",
    "collision": "0xFF6B6B",
    "lane_unavailable": "0xFFD166",
    "deadlock": "0xC084FC",
}

DISTANCE_THRESHOLD_M = 3.0
TTC_HORIZON_S = 4.0
PREDICTED_CLEARANCE_THRESHOLD_M = 3.0
PEDESTRIAN_PENALTY_WEIGHT = 70.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-index", type=int, default=2)
    parser.add_argument("--base-root", type=pathlib.Path, default=BASE_ROOT)
    parser.add_argument("--v2-root", type=pathlib.Path, default=V2_ROOT)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--crf", type=int, default=21)
    return parser.parse_args()


def branch_path(
    decision: str,
    route_id: str,
    base_root: pathlib.Path,
    v2_root: pathlib.Path,
) -> pathlib.Path:
    root = v2_root if decision in {"Accelerate", "Maintain"} else base_root
    return root / "S1" / route_id / decision


def evaluation_window(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    start = evaluation_start(frames)
    if start is None:
        return []
    clear_index = hazard_clear_index(frames, start)
    end = len(frames)
    if clear_index is not None:
        clear_time = float(frames[clear_index].get("simulation_time_s", 0.0))
        for index in range(clear_index, len(frames)):
            now = float(frames[index].get("simulation_time_s", clear_time))
            if now > clear_time + 3.0:
                end = index + 1
                break
    return frames[start:end]


def lane_aware_frame_pedestrian_risk(frame: dict[str, Any]) -> dict[str, Any]:
    ego = frame.get("ego") or {}
    ego_location = vector((ego.get("transform") or {}).get("location"))
    ego_velocity = vector(ego.get("velocity"))
    ego_lane = ego.get("lane") or {}
    lane_center = vector(ego_lane.get("center"))
    lane_heading = ego_lane.get("heading") or {}
    lane_yaw = math.radians(float(lane_heading.get("yaw", 0.0)))
    forward = (math.cos(lane_yaw), math.sin(lane_yaw))
    left = (-forward[1], forward[0])
    lane_width = max(float(ego_lane.get("lane_width_m") or 4.0), 1.0)
    ego_road_id = ego_lane.get("road_id")

    hero = next(
        (actor for actor in frame.get("actors", []) if actor.get("role_name") == "hero"),
        {},
    )
    ego_radius = actor_radius(hero, 2.6)
    best = {
        "risk": 0.0,
        "clearance_m": None,
        "ttc_s": None,
        "predicted_clearance_m": None,
        "longitudinal_m": None,
        "lateral_from_ego_lane_m": None,
    }

    for actor in frame.get("actors", []):
        if not str(actor.get("type_id", "")).startswith("walker."):
            continue
        actor_lane = actor.get("lane") or {}
        if actor_lane.get("lane_type") != "Driving":
            continue
        if ego_road_id is not None and actor_lane.get("road_id") != ego_road_id:
            continue

        actor_location = vector((actor.get("transform") or {}).get("location"))
        actor_velocity = vector(actor.get("velocity"))
        actor_lane_center = vector(actor_lane.get("center"))
        actor_lane_width = max(float(actor_lane.get("lane_width_m") or lane_width), 1.0)
        actor_radius_m = actor_radius(actor, 0.3)

        relative = (
            actor_location[0] - ego_location[0],
            actor_location[1] - ego_location[1],
        )
        longitudinal = relative[0] * forward[0] + relative[1] * forward[1]
        if longitudinal <= 0.0:
            continue

        from_lane_center = (
            actor_location[0] - lane_center[0],
            actor_location[1] - lane_center[1],
        )
        lateral = from_lane_center[0] * left[0] + from_lane_center[1] * left[1]
        if abs(lateral) > 1.5 * lane_width + actor_radius_m:
            continue

        # A nearest-map waypoint can still be returned for a pedestrian on a
        # sidewalk.  This geometric gate requires the pedestrian body to touch
        # the actual driving-lane envelope before proximity is penalized.
        lane_offset = math.hypot(
            actor_location[0] - actor_lane_center[0],
            actor_location[1] - actor_lane_center[1],
        )
        if lane_offset > actor_lane_width / 2.0 + actor_radius_m:
            continue

        center_distance = math.hypot(*relative)
        clearance = center_distance - ego_radius - actor_radius_m
        distance_risk = clip(
            (DISTANCE_THRESHOLD_M - clearance) / DISTANCE_THRESHOLD_M
        )

        relative_velocity = (
            actor_velocity[0] - ego_velocity[0],
            actor_velocity[1] - ego_velocity[1],
        )
        velocity_norm_sq = sum(value * value for value in relative_velocity)
        ttc = None
        predicted_clearance = None
        ttc_risk = 0.0
        if velocity_norm_sq > 1e-4:
            candidate_ttc = -sum(
                relative[axis] * relative_velocity[axis] for axis in range(2)
            ) / velocity_norm_sq
            if 0.0 <= candidate_ttc <= TTC_HORIZON_S:
                predicted_distance = math.hypot(
                    relative[0] + candidate_ttc * relative_velocity[0],
                    relative[1] + candidate_ttc * relative_velocity[1],
                )
                predicted_clearance = predicted_distance - ego_radius - actor_radius_m
                if predicted_clearance <= PREDICTED_CLEARANCE_THRESHOLD_M:
                    ttc = candidate_ttc
                    ttc_risk = clip((TTC_HORIZON_S - ttc) / (TTC_HORIZON_S - 0.5))

        risk = 0.35 * distance_risk + 0.65 * ttc_risk
        if risk > float(best["risk"]):
            best = {
                "risk": risk,
                "clearance_m": clearance,
                "ttc_s": ttc,
                "predicted_clearance_m": predicted_clearance,
                "longitudinal_m": longitudinal,
                "lateral_from_ego_lane_m": lateral,
            }
    return best


def front_lane_pedestrian_metrics(run_dir: pathlib.Path) -> dict[str, Any]:
    frames = read_telemetry(run_dir / "raw/carla_telemetry.jsonl.gz")
    window = evaluation_window(frames)
    frame_rows = [lane_aware_frame_pedestrian_risk(frame) for frame in window]
    risks = [float(row["risk"]) for row in frame_rows]
    relevant = [row for row in frame_rows if row["clearance_m"] is not None]
    ttc_values = [float(row["ttc_s"]) for row in relevant if row["ttc_s"] is not None]
    predicted_values = [
        float(row["predicted_clearance_m"])
        for row in relevant
        if row["predicted_clearance_m"] is not None
    ]
    mean_risk = statistics.fmean(risks) if risks else 0.0
    p90_risk = percentile(risks, 0.90)
    aggregate_risk = clip(0.7 * p90_risk + 0.3 * mean_risk)
    return {
        "evaluation_frames": len(window),
        "relevant_front_lane_frames": len(relevant),
        "minimum_clearance_m": min(
            (float(row["clearance_m"]) for row in relevant), default=None
        ),
        "minimum_ttc_s": min(ttc_values, default=None),
        "minimum_predicted_clearance_m": min(predicted_values, default=None),
        "p90_frame_risk": p90_risk,
        "mean_frame_risk": mean_risk,
        "aggregate_risk": aggregate_risk,
        "penalty": PEDESTRIAN_PENALTY_WEIGHT * aggregate_risk,
    }


def task_score_without_pedestrian_risk(row: dict[str, Any]) -> float | None:
    component = row.get("components") or {}
    required = {
        "progress",
        "recovery",
        "efficiency",
        "comfort",
        "decision_fidelity",
        "legality",
        "unnecessary_intervention",
    }
    if not required.issubset(component):
        return None
    weighted = (
        0.22 * float(component["progress"])
        + 0.16 * float(component["recovery"])
        + 0.12 * float(component["efficiency"])
        + 0.10 * float(component["comfort"])
        + 0.05 * float(component["decision_fidelity"])
        + 0.03 * float(component["legality"])
    )
    score = 100.0 * weighted / 0.68
    score -= 15.0 * float(component["unnecessary_intervention"])
    return clip(score, 0.0, 100.0)


def final_front_lane_score(row: dict[str, Any], task_score: float | None) -> float:
    counts = row.get("collision_counts") or {}
    if int(counts.get("collisions_pedestrian", 0)) > 0:
        return -100.0
    if int(counts.get("collisions_vehicle", 0)) > 0:
        return -95.0
    if int(counts.get("collisions_layout", 0)) > 0:
        return -90.0
    if row.get("offroad_or_route_violation"):
        return -85.0
    if row.get("lane_unavailable"):
        return -70.0
    if row.get("deadlock"):
        return -50.0
    if task_score is None:
        return float(row.get("decision_reward", 0.0))
    risk = float(row["front_lane_pedestrian"]["aggregate_risk"])
    return max(0.0, task_score - PEDESTRIAN_PENALTY_WEIGHT * risk)


def score_branches(paths: dict[str, pathlib.Path]) -> list[dict[str, Any]]:
    rows = []
    for decision in DECISIONS:
        row = base_branch("S1", decision, paths[decision])
        pedestrian = front_lane_pedestrian_metrics(paths[decision])
        row["front_lane_pedestrian"] = pedestrian
        if row.get("components"):
            row["risk"] = pedestrian["aggregate_risk"]
            row["components"]["risk_margin"] = 1.0 - pedestrian["aggregate_risk"]
        rows.append(row)
    finalize_group(rows)
    for row in rows:
        task_score = task_score_without_pedestrian_risk(row)
        row["task_score_before_front_lane_pedestrian_penalty"] = (
            round(task_score, 4) if task_score is not None else None
        )
        row["front_lane_decision_score"] = round(
            final_front_lane_score(row, task_score), 4
        )
    return rows


def score_color(score: float) -> str:
    if score >= 75.0:
        return "0x5EE2A0"
    if score >= 50.0:
        return "0xFFD166"
    return "0xFF6B6B"


def fmt(value: float | None, digits: int = 2) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def build_filter(
    branches: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    end_badges: list[float],
    route_id: str,
    work_dir: pathlib.Path,
) -> str:
    image_w, image_h, cell_h = 620, 310, 405
    positions = [(0, 0), (650, 0), (1300, 0), (0, 420), (650, 420), (1300, 420)]
    parts = []
    labels = []
    for index, (decision, branch, row, end_s) in enumerate(
        zip(DECISIONS, branches, scores, end_badges)
    ):
        outcome = str(branch["report"].get("outcome", "missing"))
        source_label = "V2 8.5m/s" if decision in {"Accelerate", "Maintain"} else "原分支"
        decision_file = write_text(
            work_dir / f"{index}_decision.txt",
            f"{DECISION_LABELS[decision]}  ·  {source_label}",
        )
        score = float(row["front_lane_decision_score"])
        penalty = float(row["front_lane_pedestrian"]["penalty"])
        penalty_label = f"-{penalty:.1f}" if penalty >= 0.05 else "0.0"
        score_file = write_text(
            work_dir / f"{index}_score.txt",
            f"得分 {score:+.1f}   前方车道行人惩罚 {penalty_label}",
        )
        metrics = row["front_lane_pedestrian"]
        detail_file = write_text(
            work_dir / f"{index}_detail.txt",
            (
                f"结果：{OUTCOME_LABELS.get(outcome, outcome)}  "
                f"风险 {metrics['aggregate_risk']:.3f}  "
                f"dmin {fmt(metrics['minimum_clearance_m'])}m  "
                f"TTC {fmt(metrics['minimum_ttc_s'])}s"
            ),
        )
        end_file = write_text(work_dir / f"{index}_end.txt", "末帧冻结  END")
        chain = [
            f"[{index}:v]setpts=PTS-STARTPTS",
            f"scale={image_w}:{image_h}:force_original_aspect_ratio=decrease",
            f"pad={image_w}:{image_h}:(ow-iw)/2:(oh-ih)/2:color=0x000000",
            f"pad={image_w}:{cell_h}:0:0:color=0x101216",
            f"drawbox=x=0:y=0:w=iw:h=ih:color={DECISION_COLORS[decision]}:t=5",
            filter_text(decision_file, fontsize=24, x="14", y=str(image_h + 7)),
            filter_text(
                score_file,
                fontsize=22,
                x="14",
                y=str(image_h + 39),
                color=score_color(score),
                stroke=True,
            ),
            filter_text(
                detail_file,
                fontsize=17,
                x="14",
                y=str(image_h + 72),
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
    route_number = route_id.rsplit("_", 1)[-1]
    title_file = write_text(
        work_dir / "title.txt", f"S1 行人遮挡场景 · Route {route_number} · 六决策评分"
    )
    route_file = write_text(work_dir / "route.txt", route_id)
    method_file = write_text(
        work_dir / "method.txt",
        "行人惩罚范围：仅前方 ego lane 与左右相邻 lane；侧后方及人行道不惩罚",
    )
    source_file = write_text(
        work_dir / "source.txt",
        "Accelerate / Maintain：S1_PED_SPEED_V2 8.5m/s    其他决策：保留原分支",
    )
    onset_file = write_text(work_dir / "onset.txt", "t=0  DECISION START  ·  决策开始")
    global_chain = [
        "[grid]pad=1920:1080:0:135:color=0x0B0D10",
        filter_text(title_file, fontsize=40, x="(w-text_w)/2", y="18", stroke=True),
        filter_text(route_file, fontsize=21, x="(w-text_w)/2", y="70", color="0xAAB4C0"),
        filter_text(method_file, fontsize=22, x="(w-text_w)/2", y="98", color="0xFFD166"),
        "drawbox=x=0:y=970:w=iw:h=45:color=0xD7263D@0.92:t=fill:enable='between(t,0.95,1.55)'",
        filter_text(
            onset_file,
            fontsize=27,
            x="(w-text_w)/2",
            y="976",
            enable="between(t,0.95,1.55)",
            stroke=True,
        ),
        filter_text(source_file, fontsize=20, x="(w-text_w)/2", y="1028", color="0xD8DEE6"),
        "drawbox=x=0:y=1070:w='iw*t/7.0':h=10:color=0x2DD4BF:t=fill",
        f"tpad=stop_mode=clone:stop_duration={1 / SOURCE_FPS}",
        f"fps={OUTPUT_FPS}",
        "format=yuv420p[out]",
    ]
    parts.append(",".join(global_chain))
    return ";\n".join(parts)


def main() -> int:
    args = parse_args()
    if not 0 <= args.route_index <= 9999:
        raise RuntimeError("route index must be between 0 and 9999")
    if not FONT.is_file():
        raise RuntimeError(f"required CJK font is missing: {FONT}")
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"required executable is missing: {executable}")

    route_id = f"closed_occluded_pedestrian_{args.route_index:04d}"
    paths = {
        decision: branch_path(decision, route_id, args.base_root, args.v2_root)
        for decision in DECISIONS
    }
    missing = [str(path) for path in paths.values() if not path.is_dir()]
    if missing:
        raise RuntimeError("missing branch directories:\n" + "\n".join(missing))

    scores = score_branches(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(".partial.mp4")
    manifest_path = args.output.with_suffix(".manifest.json")

    with tempfile.TemporaryDirectory(prefix=".s1_scored_video_", dir=args.output.parent) as temp:
        work_dir = pathlib.Path(temp)
        branches = []
        end_badges = []
        inputs = []
        for decision in DECISIONS:
            branch = load_branch(paths[decision])
            selected, end_badge = stage_branch(branch, work_dir / decision)
            branches.append(branch)
            end_badges.append(end_badge)
            inputs.extend(
                ["-framerate", str(SOURCE_FPS), "-i", str(work_dir / decision / "%04d.jpg")]
            )
            branch["selected_first"] = selected[0].name
            branch["selected_last"] = selected[-1].name

        filter_path = work_dir / "filter.txt"
        filter_path.write_text(
            build_filter(branches, scores, end_badges, route_id, work_dir),
            encoding="utf-8",
        )
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs]
        command.extend(
            [
                "-filter_complex_script",
                str(filter_path),
                "-map",
                "[out]",
                "-t",
                "7.0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                str(args.crf),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(partial),
            ]
        )
        run(command)
        os.replace(partial, args.output)

    payload = {
        "schema_version": "s1_front_lane_pedestrian_scored_video_v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "route_id": route_id,
        "video": str(args.output.resolve()),
        "score_range": [-100.0, 100.0],
        "branch_source_policy": {
            "Accelerate": "S1_PED_SPEED_V2",
            "Maintain": "S1_PED_SPEED_V2",
            "others": "retained_baseline",
        },
        "front_lane_pedestrian_filter": {
            "ahead_only": True,
            "driving_lane_surface_only": True,
            "included_lanes": ["ego", "adjacent_left", "adjacent_right"],
            "excluded_regions": ["behind_ego", "sidewalk", "beyond_adjacent_lanes"],
            "distance_threshold_m": DISTANCE_THRESHOLD_M,
            "ttc_horizon_s": TTC_HORIZON_S,
            "predicted_clearance_threshold_m": PREDICTED_CLEARANCE_THRESHOLD_M,
            "aggregate": "0.7 * p90(frame_risk) + 0.3 * mean(frame_risk)",
            "frame_risk": "0.35 * clearance_risk + 0.65 * TTC_risk",
            "penalty": f"-{PEDESTRIAN_PENALTY_WEIGHT} * aggregate_risk",
        },
        "hard_outcome_scores": {
            "pedestrian_collision": -100.0,
            "vehicle_collision": -95.0,
            "layout_collision": -90.0,
            "route_violation": -85.0,
            "lane_unavailable": -70.0,
            "deadlock": -50.0,
        },
        "branches": scores,
        "probe": probe(args.output),
    }
    temporary_manifest = manifest_path.with_suffix(".partial.json")
    temporary_manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)

    summary = [
        {
            "decision": row["decision"],
            "outcome": row["outcome"],
            "task_score": row["task_score_before_front_lane_pedestrian_penalty"],
            "front_lane_pedestrian_risk": round(
                float(row["front_lane_pedestrian"]["aggregate_risk"]), 4
            ),
            "front_lane_pedestrian_penalty": round(
                float(row["front_lane_pedestrian"]["penalty"]), 4
            ),
            "final_score": row["front_lane_decision_score"],
        }
        for row in scores
    ]
    print(json.dumps({"video": str(args.output), "manifest": str(manifest_path), "scores": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
