#!/usr/bin/env bash
set -euo pipefail

RL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-/home/UNT/yl0826/miniconda3/envs/simlingo/bin/python}

exec "${PYTHON}" "${RL_ROOT}/scripts/watch_counterfactual_campaign.py" \
  --config "${RL_ROOT}/config/decision_v1_remaining_unique_0121.json" \
  --manifest "${RL_ROOT}/manifests/remaining_unique_0121_manifest.jsonl" \
  "$@"
