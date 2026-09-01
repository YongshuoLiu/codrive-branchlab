#!/usr/bin/env bash
set -euo pipefail

RL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SIMLINGO_ROOT=${SIMLINGO_ROOT:-/mnt/DATA/yongshuoliu/simlingo}
CARLA_INSTALL=${CARLA_INSTALL:-/mnt/SSD/Coop_closed_loop/carla}
BENCH2DRIVE_ROOT=${BENCH2DRIVE_ROOT:-/mnt/SSD/Coop_closed_loop/Bench2Drive}
PYTHON=${PYTHON:-/home/UNT/yl0826/miniconda3/envs/simlingo/bin/python}
GPU_RANK=${GPU_RANK:-0}
CARLA_GRAPHICS_ADAPTER=${CARLA_GRAPHICS_ADAPTER:-1}
PORT=${PORT:-32100}
LIVE_TIMEOUT=${LIVE_TIMEOUT:-120}

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 CONFIG MANIFEST OUTPUT_REPORT" >&2
  exit 2
fi
CONFIG=$1
MANIFEST=$2
OUTPUT=$3
CARLA_BINARY=${CARLA_INSTALL}/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping

for required in "${PYTHON}" "${CARLA_BINARY}" "${CONFIG}" "${MANIFEST}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required path: ${required}" >&2
    exit 2
  fi
done

resolved_output=$("${PYTHON}" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${OUTPUT}")
resolved_rl=$("${PYTHON}" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${RL_ROOT}")
case "${resolved_output}/" in
  "${resolved_rl}/"*) ;;
  *) echo "OUTPUT_REPORT must stay under ${resolved_rl}: ${resolved_output}" >&2; exit 2 ;;
esac
mkdir -p "$(dirname "${resolved_output}")"
CARLA_LOG=${resolved_output%.json}.carla.log

export PYTHONPATH=${CARLA_INSTALL}/PythonAPI:${CARLA_INSTALL}/PythonAPI/carla:${BENCH2DRIVE_ROOT}/scenario_runner:${SIMLINGO_ROOT}:${PYTHONPATH:-}

if "${PYTHON}" - "${PORT}" <<'PY'
import socket, sys
sock = socket.socket()
sock.settimeout(0.5)
in_use = sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0
sock.close()
raise SystemExit(0 if in_use else 1)
PY
then
  echo "Refusing to launch: TCP port ${PORT} is already in use" >&2
  exit 2
fi

carla_pid=
cleanup() {
  if [[ -n "${carla_pid}" ]] && kill -0 "${carla_pid}" 2>/dev/null; then
    kill "${carla_pid}" 2>/dev/null || true
    for _ in $(seq 1 15); do
      if ! kill -0 "${carla_pid}" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "${carla_pid}" 2>/dev/null; then
      kill -KILL "${carla_pid}" 2>/dev/null || true
    fi
    wait "${carla_pid}" 2>/dev/null || true
  fi
  carla_pid=
}
trap cleanup EXIT INT TERM

for launch_attempt in 1 2 3; do
  echo "Launching isolated CARLA preflight server attempt ${launch_attempt}/3 on GPU ${GPU_RANK}, adapter ${CARLA_GRAPHICS_ADAPTER}, port ${PORT}"
  echo "===== launch attempt ${launch_attempt} =====" >> "${CARLA_LOG}"
  env -u CUDA_VISIBLE_DEVICES \
    "${CARLA_BINARY}" CarlaUE4 \
    -RenderOffScreen \
    -nosound \
    -carla-rpc-port="${PORT}" \
    -graphicsadapter="${CARLA_GRAPHICS_ADAPTER}" \
    -stdout -FullStdOutLogOutput \
    >> "${CARLA_LOG}" 2>&1 &
  carla_pid=$!

  set +e
  "${PYTHON}" - "${PORT}" "${carla_pid}" <<'PY'
import os, sys, time
import carla

port = int(sys.argv[1])
pid = int(sys.argv[2])
deadline = time.time() + 120.0
last_error = None
stable_reads = 0
while time.time() < deadline:
    try:
        client = carla.Client("127.0.0.1", port)
        client.set_timeout(10.0)
        world = client.get_world()
        world.get_map().name
        stable_reads += 1
        if stable_reads >= 3:
            raise SystemExit(0)
        time.sleep(3.0)
    except Exception as exc:
        last_error = exc
        stable_reads = 0
        try:
            os.kill(pid, 0)
        except OSError:
            raise SystemExit("CARLA exited before becoming ready")
        time.sleep(2.0)
raise SystemExit(f"CARLA did not become ready: {last_error}")
PY
  ready_status=$?
  set -e
  if [[ "${ready_status}" -ne 0 ]]; then
    echo "CARLA attempt ${launch_attempt} failed stable readiness" >&2
    cleanup
    continue
  fi

  attempt_output=${resolved_output%.json}.attempt_${$}_${launch_attempt}.json
  set +e
  "${PYTHON}" "${RL_ROOT}/scripts/preflight_counterfactual_routes.py" \
    --config "${CONFIG}" \
    --manifest "${MANIFEST}" \
    --output "${attempt_output}" \
    --live \
    --port "${PORT}" \
    --timeout "${LIVE_TIMEOUT}"
  preflight_status=$?
  set -e
  if [[ -f "${attempt_output}" ]]; then
    mv -f "${attempt_output}" "${resolved_output}"
    echo "Live preflight report finalized: ${resolved_output}"
    exit "${preflight_status}"
  fi
  echo "CARLA attempt ${launch_attempt} failed during live inspection; retrying" >&2
  cleanup
done

echo "Live preflight failed after 3 isolated CARLA launch attempts; see ${CARLA_LOG}" >&2
exit 1
