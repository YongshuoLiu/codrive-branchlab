#!/usr/bin/env bash
set -euo pipefail

RL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SIMLINGO_ROOT=${SIMLINGO_ROOT:-/home/UNT/yl0826/simlingo}
PYTHON=${PYTHON:-/home/UNT/yl0826/miniconda3/envs/simlingo/bin/python}

ROUTE=${ROUTE:?Set ROUTE to one frozen CoDrive XML}
DECISION=${DECISION:?Set DECISION}
RUN_DIR=${RUN_DIR:?Set RUN_DIR inside RL/data}
GPU_RANK=${GPU_RANK:?Set GPU_RANK}
CARLA_GRAPHICS_ADAPTER=${CARLA_GRAPHICS_ADAPTER:?Set CARLA_GRAPHICS_ADAPTER}
PORT=${PORT:?Set PORT}
TM_PORT=${TM_PORT:?Set TM_PORT}
EVAL_WALLTIME_SECONDS=${EVAL_WALLTIME_SECONDS:-300}

case "${DECISION}" in
  Accelerate|Maintain|Brake|Stop|LaneChangeLeft|LaneChangeRight) ;;
  *) echo "Invalid DECISION=${DECISION}" >&2; exit 2 ;;
esac

resolved_run_dir=$(${PYTHON} -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${RUN_DIR}")
resolved_data_root=$(${PYTHON} -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${RL_ROOT}/data")
case "${resolved_run_dir}/" in
  "${resolved_data_root}/"*) ;;
  *) echo "RUN_DIR must remain under ${resolved_data_root}: ${resolved_run_dir}" >&2; exit 2 ;;
esac

for required in \
  "${ROUTE}" \
  "${RL_ROOT}/agents/counterfactual_pdm_lite_agent.py" \
  "${SIMLINGO_ROOT}/pedestrian_dataset/scripts/run_s1_pdm_lite.sh" \
  "${SIMLINGO_ROOT}/pedestrian_dataset/closed_dataset/prepare_runtime_overlay.py"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required path: ${required}" >&2
    exit 2
  fi
done

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/raw" "${RUN_DIR}/metadata"
OVERLAY=${SCENARIO_RUNNER_OVERLAY:-${RL_ROOT}/runtime/scenario_runner_gpu${GPU_RANK}}
overlay_lock="${RL_ROOT}/runtime/scenario_runner_gpu${GPU_RANK}.lock"
mkdir -p "${RL_ROOT}/runtime"
if [[ ! -e "${OVERLAY}/srunner/scenarios/closed_coopeval_scenarios.py" ]]; then
  flock "${overlay_lock}" "${PYTHON}" \
    "${SIMLINGO_ROOT}/pedestrian_dataset/closed_dataset/prepare_runtime_overlay.py" \
    --output "${OVERLAY}" > "${RUN_DIR}/logs/runtime_overlay.txt"
fi

ROUTE_ID=$(
  "${PYTHON}" -c 'import sys,xml.etree.ElementTree as ET; print(ET.parse(sys.argv[1]).getroot().find("route").attrib["id"])' "${ROUTE}"
)
ROUTE_SHA256=$(sha256sum "${ROUTE}" | awk '{print $1}')
cp --preserve=timestamps "${ROUTE}" "${RUN_DIR}/metadata/source_route.xml"
"${PYTHON}" "${RL_ROOT}/scripts/stamp_s1_policy_marker.py" \
  "${RUN_DIR}" --route "${RUN_DIR}/metadata/source_route.xml" --decision "${DECISION}"

# CARLA's binary recorder reproducibly crashes UE4 on the expanded pilot
# routes, including Town01/Town02. Keep the richer per-tick JSON telemetry and
# all original sensor modalities, and explicitly record this scoped fallback.
case "${ROUTE}" in
  */routes/decision_v1_*/*)
    export COUNTERFACTUAL_DISABLE_CARLA_RECORDER=1
    ;;
esac

export SIMLINGO_ROOT
export COUNTERFACTUAL_DECISION="${DECISION}"
export COUNTERFACTUAL_EVENT_LOG="${RUN_DIR}/raw/decision_events.jsonl"
export COUNTERFACTUAL_TELEMETRY_PATH="${RUN_DIR}/raw/carla_telemetry.jsonl.gz"
export COUNTERFACTUAL_TERMINATION_PATH="${RUN_DIR}/metadata/termination.json"
export COUNTERFACTUAL_RUN_SPEC_PATH="${RUN_DIR}/metadata/run_spec.json"
export COUNTERFACTUAL_CARLA_RECORDER_PATH="${RUN_DIR}/raw/carla_recorder.log"
export COUNTERFACTUAL_ACCELERATE_DELTA_MPS=${COUNTERFACTUAL_ACCELERATE_DELTA_MPS:-3.0}
export COUNTERFACTUAL_ACCELERATE_CAP_MPS=${COUNTERFACTUAL_ACCELERATE_CAP_MPS:-20.0}
export COUNTERFACTUAL_BRAKE_SPEED_RATIO=${COUNTERFACTUAL_BRAKE_SPEED_RATIO:-0.45}
export COUNTERFACTUAL_BRAKE_MIN_TARGET_MPS=${COUNTERFACTUAL_BRAKE_MIN_TARGET_MPS:-1.5}
export COUNTERFACTUAL_STOP_SPEED_MPS=${COUNTERFACTUAL_STOP_SPEED_MPS:-0.3}
export COUNTERFACTUAL_STOP_HOLD_S=${COUNTERFACTUAL_STOP_HOLD_S:-0.5}
export COUNTERFACTUAL_LANE_MIN_TARGET_MPS=${COUNTERFACTUAL_LANE_MIN_TARGET_MPS:-4.0}
export COUNTERFACTUAL_LANE_PRE_DISTANCE_M=${COUNTERFACTUAL_LANE_PRE_DISTANCE_M:-14.0}
export COUNTERFACTUAL_LANE_POST_DISTANCE_M=${COUNTERFACTUAL_LANE_POST_DISTANCE_M:-28.0}
export COUNTERFACTUAL_LANE_TRANSITION_DISTANCE_M=${COUNTERFACTUAL_LANE_TRANSITION_DISTANCE_M:-6.0}
export COUNTERFACTUAL_LANE_MIN_COVERAGE=${COUNTERFACTUAL_LANE_MIN_COVERAGE:-0.8}
export COUNTERFACTUAL_LANE_MIN_WIDTH_M=${COUNTERFACTUAL_LANE_MIN_WIDTH_M:-2.4}
export COUNTERFACTUAL_LANE_REAR_CLEARANCE_M=${COUNTERFACTUAL_LANE_REAR_CLEARANCE_M:-15.0}
export COUNTERFACTUAL_LANE_FRONT_CLEARANCE_M=${COUNTERFACTUAL_LANE_FRONT_CLEARANCE_M:-35.0}
export COUNTERFACTUAL_LANE_MIN_TTC_S=${COUNTERFACTUAL_LANE_MIN_TTC_S:-4.0}
export COUNTERFACTUAL_DEADLOCK_SPEED_MPS=${COUNTERFACTUAL_DEADLOCK_SPEED_MPS:-0.1}
export COUNTERFACTUAL_DEADLOCK_HOLD_S=${COUNTERFACTUAL_DEADLOCK_HOLD_S:-2.0}
export COUNTERFACTUAL_POST_CLEAR_DEADLOCK_HOLD_S=${COUNTERFACTUAL_POST_CLEAR_DEADLOCK_HOLD_S:-3.0}
export COUNTERFACTUAL_MAX_DECISION_ACTIVE_S=${COUNTERFACTUAL_MAX_DECISION_ACTIVE_S:-20.0}
export COUNTERFACTUAL_RAW_ACTOR_RADIUS_M=${COUNTERFACTUAL_RAW_ACTOR_RADIUS_M:-100.0}
export COUNTERFACTUAL_RECORDER_ARM_DELAY_S=${COUNTERFACTUAL_RECORDER_ARM_DELAY_S:-1.0}

{
  echo "SCHEMA_VERSION=counterfactual_decision_v1"
  echo "ROUTE=${ROUTE}"
  echo "ROUTE_ID=${ROUTE_ID}"
  echo "ROUTE_SHA256=${ROUTE_SHA256}"
  echo "DECISION=${DECISION}"
  echo "RUN_DIR=${RUN_DIR}"
  echo "GPU_RANK=${GPU_RANK}"
  echo "CARLA_GRAPHICS_ADAPTER=${CARLA_GRAPHICS_ADAPTER}"
  echo "PORT=${PORT}"
  echo "TM_PORT=${TM_PORT}"
  echo "SCENARIO_RUNNER_ROOT=${OVERLAY}"
  echo "BACKGROUND_ACTIVITY=disabled_for_state_alignment"
  echo "CARLA_BINARY_RECORDER_DISABLED=${COUNTERFACTUAL_DISABLE_CARLA_RECORDER:-0}"
  echo "LANE_MARKING_RULES=ignored"
  echo "LANE_GEOMETRY_AND_DYNAMIC_OCCUPANCY=required"
} > "${RUN_DIR}/metadata/branch_env.txt"

set +e
env \
  GPU_RANK="${GPU_RANK}" \
  CARLA_GRAPHICS_ADAPTER="${CARLA_GRAPHICS_ADAPTER}" \
  PORT="${PORT}" \
  TM_PORT="${TM_PORT}" \
  EVAL_WALLTIME_SECONDS="${EVAL_WALLTIME_SECONDS}" \
  ROUTES="${ROUTE}" \
  ROUTE_BASENAME="${ROUTE_ID}_${DECISION}" \
  RUN_NAME="${ROUTE_ID}_${DECISION}_gpu${GPU_RANK}" \
  RUN_DIR="${RUN_DIR}" \
  TEAM_AGENT="${RL_ROOT}/agents/counterfactual_pdm_lite_agent.py" \
  TEAM_CONFIG=pdm_lite_traj \
  SCENARIO_RUNNER_ROOT="${OVERLAY}" \
  LEADERBOARD_DISABLE_RANDOM_WEATHER=1 \
  LEADERBOARD_DISABLE_BACKGROUND_ACTIVITY=1 \
  B2D_AGENT_BLOCKED_MAX_TIME=60 \
  CARLA_UNSET_CUDA_VISIBLE_DEVICES=1 \
  MAKE_WAYPOINT_VIDEO=0 \
  SKIP_VIDEO=1 \
  TEMP_VIDEO_INTERVAL_GAME_SECONDS=0 \
  bash "${SIMLINGO_ROOT}/pedestrian_dataset/scripts/run_s1_pdm_lite.sh" \
  > "${RUN_DIR}/logs/branch_console.log" 2>&1
status=$?
set -e

"${PYTHON}" - "${RUN_DIR}" "${status}" <<'PY'
import json
import pathlib
import sys
run_dir = pathlib.Path(sys.argv[1])
payload = {"runner_exit_status": int(sys.argv[2])}
(run_dir / "metadata/runner_status.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
exit "${status}"
