#!/usr/bin/env bash
set -euo pipefail

RL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-/home/UNT/yl0826/miniconda3/envs/simlingo/bin/python}

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 CONFIG MANIFEST LIVE_PREFLIGHT_REPORT [SUPERVISOR_ARGS...]" >&2
  exit 2
fi

CONFIG=$1
MANIFEST=$2
LIVE_REPORT=$3
shift 3
SUPERVISOR_ARGS=("$@")
for required in "${PYTHON}" "${CONFIG}" "${MANIFEST}" "${LIVE_REPORT}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required path: ${required}" >&2
    exit 2
  fi
done

PRODUCTION_DIR=$("${PYTHON}" - "${CONFIG}" "${MANIFEST}" "${RL_ROOT}" <<'PY'
import json, pathlib, sys
config = json.load(open(sys.argv[1], encoding="utf-8"))
root = pathlib.Path(sys.argv[3]).resolve()
output = pathlib.Path(config["output_root"]).resolve()
if root not in output.parents:
    raise SystemExit("config output_root must remain under RL")
print(output / "_production" / pathlib.Path(sys.argv[2]).stem)
PY
)
mkdir -p "${PRODUCTION_DIR}"
rm -f "${PRODUCTION_DIR}/STOP"

PID_FILE="${PRODUCTION_DIR}/supervisor.pid"
if [[ -f "${PID_FILE}" ]]; then
  existing_pid=$(tr -dc '0-9' < "${PID_FILE}")
  if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    echo "Campaign is already running: supervisor PID ${existing_pid}" >&2
    exit 2
  fi
fi

LOG="${PRODUCTION_DIR}/background.log"
nohup setsid "${PYTHON}" -u "${RL_ROOT}/scripts/supervise_counterfactual_campaign.py" \
  --config "${CONFIG}" \
  --manifest "${MANIFEST}" \
  --live-preflight-report "${LIVE_REPORT}" \
  "${SUPERVISOR_ARGS[@]}" \
  >> "${LOG}" 2>&1 < /dev/null &
supervisor_pid=$!
echo "${supervisor_pid}" > "${PID_FILE}"

DASHBOARD_LOG="${PRODUCTION_DIR}/dashboard_daemon.log"
dashboard_pid=""
if [[ -f "${PRODUCTION_DIR}/dashboard.pid" ]]; then
  dashboard_pid=$(tr -dc '0-9' < "${PRODUCTION_DIR}/dashboard.pid")
fi
if [[ -z "${dashboard_pid}" ]] || ! kill -0 "${dashboard_pid}" 2>/dev/null; then
  nohup setsid "${PYTHON}" -u "${RL_ROOT}/scripts/watch_counterfactual_campaign.py" \
    --config "${CONFIG}" \
    --manifest "${MANIFEST}" \
    --quiet \
    --interval 3 \
    >> "${DASHBOARD_LOG}" 2>&1 < /dev/null &
  dashboard_pid=$!
  echo "${dashboard_pid}" > "${PRODUCTION_DIR}/dashboard.pid"
fi

sleep 2
if ! kill -0 "${supervisor_pid}" 2>/dev/null; then
  echo "Supervisor exited during startup; inspect ${LOG}" >&2
  tail -80 "${LOG}" >&2 || true
  exit 1
fi

echo "Background campaign started"
echo "  supervisor PID : ${supervisor_pid}"
echo "  dashboard PID  : ${dashboard_pid}"
echo "  production dir : ${PRODUCTION_DIR}"
echo "  log            : ${LOG}"
echo ""
echo "Open the dynamic dashboard with:"
echo "  ${PYTHON} ${RL_ROOT}/scripts/watch_counterfactual_campaign.py --config ${CONFIG} --manifest ${MANIFEST}"
