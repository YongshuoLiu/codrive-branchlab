#!/usr/bin/env bash
set -euo pipefail

RL_ROOT=$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")
PYTHON=${PYTHON:-/home/UNT/yl0826/miniconda3/envs/simlingo/bin/python}

exec "${PYTHON}" "${RL_ROOT}/scripts/watch_counterfactual_campaign.py" \
  --config "${RL_ROOT}/config/decision_v1_scaleup_0021_0120.json" \
  --manifest "${RL_ROOT}/manifests/scaleup_0021_0120_manifest.jsonl" \
  "$@"
