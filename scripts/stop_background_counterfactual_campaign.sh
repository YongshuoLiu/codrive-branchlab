#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 PRODUCTION_DIR" >&2
  exit 2
fi
PRODUCTION_DIR=$(realpath "$1")
RL_ROOT=$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")
case "${PRODUCTION_DIR}/" in
  "${RL_ROOT}/data/counterfactual_decision_v1/"*) ;;
  *) echo "Refusing production directory outside the RL campaign root: ${PRODUCTION_DIR}" >&2; exit 2 ;;
esac

touch "${PRODUCTION_DIR}/STOP"

runtime="${PRODUCTION_DIR}/runtime_progress.json"
if [[ -f "${runtime}" ]]; then
  while read -r evaluator_pid; do
    [[ -n "${evaluator_pid}" ]] || continue
    command=$(ps -p "${evaluator_pid}" -o args= 2>/dev/null || true)
    if [[ "${command}" == *"run_counterfactual_branch.sh"* ]]; then
      kill -TERM -- "-${evaluator_pid}" 2>/dev/null || true
      echo "Stopped active branch process group ${evaluator_pid}"
    fi
  done < <(python - "${runtime}" <<'PY'
import json, sys
try:
    payload = json.load(open(sys.argv[1], encoding="utf-8"))
except (OSError, ValueError):
    payload = {}
for worker in (payload.get("workers") or {}).values():
    current = worker.get("current_job") or {}
    if current.get("evaluator_pid"):
        print(current["evaluator_pid"])
PY
)
fi

supervisor_status="${PRODUCTION_DIR}/supervisor_status.json"
if [[ -f "${supervisor_status}" ]]; then
  producer_pid=$(python - "${supervisor_status}" <<'PY'
import json, sys
try:
    print(json.load(open(sys.argv[1], encoding="utf-8")).get("producer_pid") or "")
except (OSError, ValueError):
    print("")
PY
)
  if [[ -n "${producer_pid}" ]] && kill -0 "${producer_pid}" 2>/dev/null; then
    while read -r child_pid; do
      command=$(ps -p "${child_pid}" -o args= 2>/dev/null || true)
      if [[ "${command}" == *"collect_counterfactual_v1.py"* ]]; then
        kill -TERM -- "-${child_pid}" 2>/dev/null || true
        echo "Stopped collector process group ${child_pid}"
      fi
    done < <(pgrep -P "${producer_pid}" 2>/dev/null || true)
  fi
fi

for name in supervisor dashboard; do
  pid_file="${PRODUCTION_DIR}/${name}.pid"
  [[ -f "${pid_file}" ]] || continue
  pid=$(tr -dc '0-9' < "${pid_file}")
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    echo "Stopped ${name} PID ${pid}"
  fi
done
