#!/usr/bin/env python3
"""Sample complete decision groups and build one Reward-v2 video per route.

The default selection is reproducible: one route is sampled from each of S1,
S2 and S3 with seed 20260901.  A route is eligible only when all six active
branches are quality-valid, the group has a valid soft target, and the set
contains both a safe branch and a failed/infeasible branch.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import pathlib
import random
import shutil
import tempfile
from typing import Any

from build_decision_comparison_video import (
    CLIP_DURATION_S,
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
from annotate_counterfactual_decision_reward_v2 import (
    compute_static,
    decision_label,
    finalize_reward_rows,
)


RL_ROOT = pathlib.Path(__file__).resolve().parents[1]
ANNOTATION_ROOT = (
    RL_ROOT / "data/counterfactual_decision_v1/_annotations/decision_reward_v2"
)
DEFAULT_OUTPUT_DIR = RL_ROOT / "videos/decision_reward_v2_samples"
DEFAULT_SEED = 20260901

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

LONGITUDINAL_ZH = {
    "Accelerate": "加速",
    "Maintain": "保持",
    "Brake": "制动",
    "Stop": "停车",
}

LATERAL_ZH = {
    "RouteFollow": "沿路线",
    "LaneChangeLeft": "左变道",
    "LaneChangeRight": "右变道",
}

SCENE_TITLES = {
    "S1": "遮挡行人",
    "S2": "车辆切入",
    "S3": "前车让出后暴露障碍物",
    "S4": "左转遇异常来车",
    "S5": "右转遇异常来车",
    "S6": "红灯右转让行",
}

TIER_LABELS = {
    "safe": "安全完成",
    "severe_near_miss": "严重近失",
    "near_miss": "近失风险",
    "sustained_high_risk": "持续高风险",
    "pedestrian_collision": "行人碰撞",
    "vehicle_collision": "车辆碰撞",
    "layout_collision": "环境碰撞",
    "route_violation": "路线违规",
    "action_infeasible": "无安全车道",
    "deadlock": "死锁",
    "unscored": "未评分",
}

TIER_COLORS = {
    "safe": "0x5EE2A0",
    "severe_near_miss": "0xFF6B6B",
    "near_miss": "0xFF8A65",
    "sustained_high_risk": "0xFFD166",
    "pedestrian_collision": "0xFF6B6B",
    "vehicle_collision": "0xFF6B6B",
    "layout_collision": "0xFF6B6B",
    "route_violation": "0xFF8A65",
    "action_infeasible": "0xFFD166",
    "deadlock": "0xC084FC",
    "unscored": "0xAAB4C0",
}

FAILURE_TIERS = {
    "pedestrian_collision",
    "vehicle_collision",
    "layout_collision",
    "route_violation",
    "action_infeasible",
    "deadlock",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-root", type=pathlib.Path, default=ANNOTATION_ROOT)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--scene",
        action="append",
        choices=tuple(SCENE_TITLES),
        help="Sample one route from this scene; repeat as needed (default: S1-S3)",
    )
    parser.add_argument("--crf", type=int, default=21)
    parser.add_argument(
        "--recompute-reward",
        action="store_true",
        help="Recompute branch rewards from telemetry instead of using stored annotations",
    )
    parser.add_argument(
        "--merge-existing-manifest",
        action="store_true",
        help="Replace rendered groups in an existing selection_manifest.json",
    )
    return parser.parse_args()


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def choose_groups(
    annotation_root: pathlib.Path, scenes: list[str], seed: int
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    groups = read_jsonl(annotation_root / "group_index.jsonl")
    branches = read_jsonl(annotation_root / "branch_index.jsonl")
    branches_by_group: dict[str, list[dict[str, Any]]] = {}
    for branch in branches:
        if branch.get("active_for_training"):
            branches_by_group.setdefault(str(branch["group_id"]), []).append(branch)

    rng = random.Random(seed)
    selected: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for scene_id in scenes:
        candidates: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        for group in groups:
            if group.get("scenario_id") != scene_id:
                continue
            if not group.get("active_for_training") or not group.get("label_valid"):
                continue
            group_branches = branches_by_group.get(str(group["group_id"]), [])
            decisions = {str(branch["branch_decision"]) for branch in group_branches}
            tiers = {str(branch["hard_safety_tier"]) for branch in group_branches}
            if decisions != set(DECISIONS):
                continue
            if not all(branch.get("quality_valid") for branch in group_branches):
                continue
            if "safe" not in tiers or not tiers.intersection(FAILURE_TIERS):
                continue
            candidates.append((group, group_branches))
        if not candidates:
            raise RuntimeError(f"no eligible complete Reward-v2 group for {scene_id}")
        selected.append(rng.choice(candidates))
    return selected


def score_color(score: float) -> str:
    if score >= 75.0:
        return "0x5EE2A0"
    if score >= 50.0:
        return "0xFFD166"
    if score >= 0.0:
        return "0xFFB15C"
    return "0xFF6B6B"


def signed_penalty(value: float) -> str:
    return "0.0" if abs(value) < 0.05 else f"-{value:.1f}"


def build_filter(
    scene_id: str,
    route_id: str,
    branches: list[dict[str, Any]],
    end_badges: list[float],
    work_dir: pathlib.Path,
    seed: int,
    recomputed_reward: bool,
) -> str:
    image_w, image_h, cell_h = 620, 310, 405
    positions = [(0, 0), (650, 0), (1300, 0), (0, 420), (650, 420), (1300, 420)]
    parts: list[str] = []
    labels: list[str] = []

    for index, (branch, end_s) in enumerate(zip(branches, end_badges)):
        report = branch["report"]
        reward = report["decision_reward"]
        decision = str(report["decision"])
        longitudinal = str(report["decision_longitudinal"])
        lateral = str(report["decision_lateral"])
        score = float(report["decision_reward_score"])
        probability = float(report["decision_target_probability"])
        hazard_penalty = float(reward.get("hazard_penalty") or 0.0)
        tier = str(reward.get("hard_safety_tier") or "unscored")
        display_tier = tier
        cap_reasons = set(reward.get("safety_cap_reasons") or [])
        if tier == "safe":
            if "severe_near_miss" in cap_reasons:
                display_tier = "severe_near_miss"
            elif "near_miss" in cap_reasons:
                display_tier = "near_miss"
            elif "sustained_high_risk" in cap_reasons:
                display_tier = "sustained_high_risk"
        top = bool(report.get("decision_top_label"))
        source = "S1-V2" if "_recollected/S1_PED_SPEED_V2" in str(branch["branch_dir"]) else "BASE"

        decision_file = write_text(
            work_dir / f"{index}_decision.txt",
            f"{DECISION_LABELS[decision]}  ·  {source}",
        )
        action_file = write_text(
            work_dir / f"{index}_action.txt",
            f"纵向 {LONGITUDINAL_ZH.get(longitudinal, longitudinal)}  ｜  横向 {LATERAL_ZH.get(lateral, lateral)}",
        )
        score_file = write_text(
            work_dir / f"{index}_score.txt",
            f"得分 {score:+.1f}  风险惩罚 {signed_penalty(hazard_penalty)}  GT p={probability:.3f}",
        )
        tier_file = write_text(
            work_dir / f"{index}_tier.txt",
            TIER_LABELS.get(display_tier, display_tier),
        )
        end_file = write_text(work_dir / f"{index}_end.txt", "末帧冻结  END")
        top_file = write_text(work_dir / f"{index}_top.txt", "TOP  GT候选")

        chain = [
            f"[{index}:v]setpts=PTS-STARTPTS",
            f"scale={image_w}:{image_h}:force_original_aspect_ratio=decrease",
            f"pad={image_w}:{image_h}:(ow-iw)/2:(oh-ih)/2:color=0x000000",
            f"pad={image_w}:{cell_h}:0:0:color=0x101216",
            f"drawbox=x=0:y=0:w=iw:h=ih:color={DECISION_COLORS[decision]}:t=5",
            "drawbox=x=12:y=12:w=150:h=36:color=black@0.72:t=fill",
            filter_text(
                tier_file,
                fontsize=18,
                x="22",
                y="18",
                color=TIER_COLORS.get(display_tier, "0xD1D5DB"),
                stroke=True,
            ),
            filter_text(decision_file, fontsize=23, x="14", y=str(image_h + 5)),
            filter_text(
                action_file,
                fontsize=18,
                x="14",
                y=str(image_h + 36),
                color="0xCBD5E1",
            ),
            filter_text(
                score_file,
                fontsize=18,
                x="14",
                y=str(image_h + 65),
                color=score_color(score),
                stroke=True,
            ),
        ]
        if top:
            chain.extend(
                [
                    "drawbox=x=iw-155:y=12:w=140:h=36:color=0x0F766E@0.94:t=fill",
                    filter_text(
                        top_file,
                        fontsize=18,
                        x="w-145",
                        y="18",
                        color="white",
                        stroke=True,
                    ),
                ]
            )
        chain.extend(
            [
                (
                    f"drawbox=x=iw-205:y=56:w=190:h=36:color=black@0.72:t=fill:"
                    f"enable='gte(t,{end_s:.3f})'"
                ),
                filter_text(
                    end_file,
                    fontsize=18,
                    x="w-192",
                    y="62",
                    color="0xF3F4F6",
                    enable=f"gte(t,{end_s:.3f})",
                ),
            ]
        )
        parts.append(",".join(chain) + f"[cell{index}]")
        labels.append(f"[cell{index}]")

    layout = "|".join(f"{x}_{y}" for x, y in positions)
    parts.append(
        "".join(labels)
        + f"xstack=inputs={len(labels)}:layout={layout}:fill=0x0B0D10[grid]"
    )

    route_number = route_id.rsplit("_", 1)[-1]
    title_file = write_text(
        work_dir / "title.txt",
        (
            f"{scene_id} {SCENE_TITLES.get(scene_id, scene_id)} · Route {route_number} · "
            f"六分支 Reward v2{' 修正版' if recomputed_reward else ''}"
        ),
    )
    route_file = write_text(work_dir / "route.txt", route_id)
    method_file = write_text(
        work_dir / "method.txt",
        (
            "S1 路口修正：按空间车道走廊计算，不强制 pedestrian road_id = ego road_id"
            if recomputed_reward and scene_id == "S1"
            else "得分范围 [-100,100]  ｜  GT p = 组内安全候选 soft target  ｜  TOP = 最高分窗口内候选"
        ),
    )
    footer_file = write_text(
        work_dir / "footer.txt",
        f"固定随机种子 {seed}  ｜  t=-1s → t=0 决策开始 → t=+6s  ｜  早终止分支保留末帧",
    )
    onset_file = write_text(work_dir / "onset.txt", "t=0  DECISION START  ·  决策开始")
    global_chain = [
        "[grid]pad=1920:1080:0:135:color=0x0B0D10",
        filter_text(title_file, fontsize=39, x="(w-text_w)/2", y="17", stroke=True),
        filter_text(route_file, fontsize=20, x="(w-text_w)/2", y="67", color="0xAAB4C0"),
        filter_text(method_file, fontsize=20, x="(w-text_w)/2", y="96", color="0xFFD166"),
        "drawbox=x=0:y=970:w=iw:h=45:color=0xD7263D@0.92:t=fill:enable='between(t,0.95,1.55)'",
        filter_text(
            onset_file,
            fontsize=27,
            x="(w-text_w)/2",
            y="976",
            enable="between(t,0.95,1.55)",
            stroke=True,
        ),
        filter_text(footer_file, fontsize=19, x="(w-text_w)/2", y="1030", color="0xD8DEE6"),
        f"drawbox=x=0:y=1070:w='iw*t/{CLIP_DURATION_S}':h=10:color=0x2DD4BF:t=fill",
        f"tpad=stop_mode=clone:stop_duration={1 / SOURCE_FPS}",
        f"fps={OUTPUT_FPS}",
        "format=yuv420p[out]",
    ]
    parts.append(",".join(global_chain))
    return ";\n".join(parts)


def render_group(
    group: dict[str, Any],
    branch_rows: list[dict[str, Any]],
    output_dir: pathlib.Path,
    crf: int,
    seed: int,
    recompute_reward: bool,
) -> dict[str, Any]:
    scene_id = str(group["scenario_id"])
    route_id = str(group["route_id"])
    rows_by_decision = {str(row["branch_decision"]): row for row in branch_rows}
    fresh_rows_by_decision: dict[str, dict[str, Any]] = {}
    fresh_preference: dict[str, Any] | None = None
    if recompute_reward:
        fresh_rows: list[dict[str, Any]] = []
        for decision in DECISIONS:
            source = rows_by_decision[decision]
            _, row = compute_static((scene_id, decision, str(source["run_dir"])))
            if row.get("static_compute_error"):
                raise RuntimeError(
                    f"reward recomputation failed for {source['run_dir']}: "
                    f"{row['static_compute_error']}"
                )
            fresh_rows.append(row)
        fresh_preference = finalize_reward_rows(fresh_rows)
        if not fresh_preference.get("label_valid"):
            raise RuntimeError(f"recomputed group has no valid label: {group['group_id']}")
        fresh_rows_by_decision = {
            str(row["decision"]): row for row in fresh_rows
        }
    output = output_dir / f"{scene_id}_{route_id}_reward_v2.mp4"
    partial = output.with_suffix(".partial.mp4")

    with tempfile.TemporaryDirectory(prefix=f".{scene_id}_reward_video_", dir=output_dir) as temporary:
        work_dir = pathlib.Path(temporary)
        branches: list[dict[str, Any]] = []
        end_badges: list[float] = []
        inputs: list[str] = []
        manifest_branches: list[dict[str, Any]] = []
        for decision in DECISIONS:
            branch_row = rows_by_decision[decision]
            branch = load_branch(pathlib.Path(branch_row["run_dir"]))
            if recompute_reward:
                score_row = fresh_rows_by_decision[decision]
                label = decision_label(decision)
                report = copy.deepcopy(branch["report"])
                reward = copy.deepcopy(report.get("decision_reward") or {})
                reward.update(
                    {
                        "reward_version": "decision_reward_v2.0",
                        "runtime_recomputed_for_video": True,
                        "corridor_filter_revision": "s1_connected_road_geometry_v1",
                        "hard_safety_tier": score_row["hard_safety_tier"],
                        "hard_safety_tier_rank": score_row["hard_safety_tier_rank"],
                        "task_score_before_hazard_penalty": score_row[
                            "task_score_before_hazard_penalty"
                        ],
                        "hazard_penalty": round(
                            float((score_row.get("hazard") or {}).get("penalty") or 0.0),
                            6,
                        ),
                        "hazard": score_row.get("hazard") or {},
                        "safety_cap": score_row.get("safety_cap"),
                        "safety_cap_reasons": score_row.get("safety_cap_reasons") or [],
                        "final_reward": score_row["final_reward"],
                        "target_probability": round(
                            float(score_row.get("target_probability") or 0.0), 8
                        ),
                        "top_decision": bool(score_row.get("top_decision")),
                        "longitudinal_target_distribution": fresh_preference[
                            "longitudinal_target_distribution"
                        ],
                        "lateral_target_distribution": fresh_preference[
                            "lateral_target_distribution"
                        ],
                    }
                )
                report.update(
                    {
                        "decision_label": label,
                        "decision_reward": reward,
                        "decision_longitudinal": label["longitudinal"],
                        "decision_lateral": label["lateral"],
                        "decision_reward_score": score_row["final_reward"],
                        "decision_target_probability": reward["target_probability"],
                        "decision_top_label": reward["top_decision"],
                    }
                )
                branch["report"] = report
            report = branch["report"]
            if report.get("decision") != decision:
                raise RuntimeError(f"decision mismatch in {branch['branch_dir']}")
            if report.get("decision_reward", {}).get("reward_version") != "decision_reward_v2.0":
                raise RuntimeError(f"missing Reward-v2 annotation: {branch['branch_dir']}")
            selected, end_badge = stage_branch(branch, work_dir / decision)
            branches.append(branch)
            end_badges.append(end_badge)
            inputs.extend(
                ["-framerate", str(SOURCE_FPS), "-i", str(work_dir / decision / "%04d.jpg")]
            )
            reward = report["decision_reward"]
            manifest_branches.append(
                {
                    "decision": decision,
                    "longitudinal": report["decision_longitudinal"],
                    "lateral": report["decision_lateral"],
                    "final_reward": report["decision_reward_score"],
                    "hazard_penalty": reward.get("hazard_penalty"),
                    "target_probability": report["decision_target_probability"],
                    "top_decision": report["decision_top_label"],
                    "hard_safety_tier": reward.get("hard_safety_tier"),
                    "safety_cap": reward.get("safety_cap"),
                    "safety_cap_reasons": reward.get("safety_cap_reasons") or [],
                    "outcome": report.get("outcome"),
                    "source_branch": str(branch["branch_dir"]),
                    "video_first_source_frame": selected[0].name,
                    "video_last_source_frame": selected[-1].name,
                    "video_end_badge_time_s": end_badge,
                }
            )

        filter_path = work_dir / "filter.txt"
        filter_path.write_text(
            build_filter(
                scene_id,
                route_id,
                branches,
                end_badges,
                work_dir,
                seed,
                recompute_reward,
            ),
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
                str(partial),
            ]
        )
        run(command)
        os.replace(partial, output)

    print(f"完成 {scene_id}: {route_id} -> {output}", flush=True)
    return {
        "scenario_id": scene_id,
        "scenario_title_zh": SCENE_TITLES.get(scene_id, scene_id),
        "route_id": route_id,
        "group_id": group["group_id"],
        "output_video": str(output.resolve()),
        "score_source": (
            "telemetry_runtime_recompute_s1_connected_road_geometry_v1"
            if recompute_reward
            else "stored_decision_reward_v2_annotation"
        ),
        "branches": manifest_branches,
        "probe": probe(output),
    }


def main() -> int:
    args = parse_args()
    scenes = args.scene or ["S1", "S2", "S3"]
    if len(set(scenes)) != len(scenes):
        raise RuntimeError("each --scene may be specified only once")
    if not args.annotation_root.is_dir():
        raise RuntimeError(f"missing annotation root: {args.annotation_root}")
    if not FONT.is_file():
        raise RuntimeError(f"required CJK font is missing: {FONT}")
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"required executable is missing: {executable}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = choose_groups(args.annotation_root, scenes, args.seed)
    print(
        json.dumps(
            {
                "seed": args.seed,
                "selected_groups": [group["group_id"] for group, _ in selected],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    videos = [
        render_group(
            group,
            branches,
            args.output_dir,
            args.crf,
            args.seed,
            args.recompute_reward,
        )
        for group, branches in selected
    ]
    payload = {
        "schema_version": "sampled_decision_reward_v2_videos_v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selection": {
            "seed": args.seed,
            "scenes": scenes,
            "one_route_per_scene": True,
            "requirements": [
                "six_active_decisions",
                "all_branches_quality_valid",
                "group_label_valid",
                "contains_safe_and_failed_or_infeasible_branch",
            ],
        },
        "videos": videos,
    }
    manifest_path = args.output_dir / "selection_manifest.json"
    if args.merge_existing_manifest and manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        replacements = {video["group_id"]: video for video in videos}
        merged = []
        for video in existing.get("videos", []):
            merged.append(replacements.pop(video.get("group_id"), video))
        merged.extend(replacements.values())
        videos = merged
        payload["videos"] = videos
        payload["selection"] = existing.get("selection", payload["selection"])
        payload["last_partial_update"] = {
            "scenes": scenes,
            "recompute_reward": args.recompute_reward,
            "corridor_filter_revision": "s1_connected_road_geometry_v1",
        }
    manifest_partial = manifest_path.with_suffix(".partial.json")
    manifest_partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(manifest_partial, manifest_path)
    print(f"清单: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
