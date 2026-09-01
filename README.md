# CoDrive BranchLab

Counterfactual multi-branch decision collection, validation, scoring, and visualization for CARLA-based cooperative vision-language-action research.

CoDrive BranchLab studies how a driving agent can move beyond a single conservative imitation-learning expert. Given the same warning-time state, it collects multiple controlled outcomes, assigns auditable decision rewards, and trains a decision policy separately from a decision-conditioned trajectory executor.

> This repository contains source code, configuration, frozen route definitions, and research documentation only. Collected sensor data, raw CARLA telemetry, generated videos, model weights, logs, manifests, and reward-audit outputs are intentionally excluded.

## Method at a glance

The project follows a three-stage research program:

```text
Execution learning
    → supervised decision learning
    → reward-based outcome optimization
```

The policy is decomposed into:

- **Decision policy** `π_D`: selects the high-level lateral and longitudinal behavior.
- **Trajectory policy** `π_T`: executes a selected decision as path and speed trajectories.
- **Decision Reward v2**: scores safety, hazard exposure, progress, recovery, efficiency, comfort, legality, decision fidelity, and unnecessary intervention.

For a warning state, the collection pipeline replays a frozen route with counterfactual branches such as `Accelerate`, `Maintain`, `Brake`, `Stop`, `LaneChangeLeft`, and `LaneChangeRight`. Branches share an aligned decision onset so their outcomes can be compared directly.

The full research formulation is documented in [Cooperative VLA Final Design](<Cooperative_VLA_Final_Design (1).md>).

## Scenario families

| ID | Scenario | Collected decisions |
|---|---|---|
| S1 | Pedestrian emergence from occlusion | 4 longitudinal + left/right lane change |
| S2 | Vehicle cut-in | 4 longitudinal + left/right lane change |
| S3 | Lead vehicle reveals an obstacle | 4 longitudinal + left/right lane change |
| S4 | Left-turn conflict with wrong-way vehicle | 4 longitudinal |
| S5 | Right-turn conflict with wrong-way vehicle | 4 longitudinal |
| S6 | Right turn on red with conflicting traffic | 4 longitudinal |

The current factorized label space is:

```text
Longitudinal: Accelerate / Maintain / Brake / Stop
Lateral:     RouteFollow / LaneChangeLeft / LaneChangeRight
```

The route definitions permit lane changes across lane markings for counterfactual research, but a target must still be a geometrically valid driving lane and pass dynamic occupancy checks.

## What is included

```text
agents/       Counterfactual PDM-Lite agent and decision execution rules
config/       Collection campaigns, worker assignments, and quality thresholds
routes/       Frozen CARLA/Leaderboard XML route definitions
scripts/      Collection, supervision, validation, reward, audit, and video tools
*.md          Design notes, operational checklists, and campaign reports
```

Important entry points:

- `scripts/produce_counterfactual_routes.py` — staged, resume-safe campaign controller.
- `scripts/collect_counterfactual_v1.py` — multi-GPU branch scheduler with live progress.
- `scripts/validate_counterfactual_branch.py` — single-branch quality and contract checks.
- `scripts/validate_counterfactual_campaign.py` — route-group and campaign validation.
- `scripts/annotate_counterfactual_decision_reward_v2.py` — factorized labels, rewards, and soft targets.
- `scripts/audit_counterfactual_decision_reward_v2.py` — annotation integrity audit.
- `scripts/build_sampled_reward_v2_route_videos.py` — synchronized scored branch videos.

## External requirements

This is an extension package rather than a standalone CARLA distribution. It expects an existing Linux research environment containing:

- SimLingo/Coop-SimLingo;
- CARLA and its Python API;
- CARLA Leaderboard and ScenarioRunner;
- the parent project's PDM-Lite and CoDrive scenario assets;
- an NVIDIA GPU setup for closed-loop collection;
- Python packages already required by the parent SimLingo environment;
- `ffmpeg` and `ffprobe` for comparison videos.

Paths in the campaign JSON files reflect the original research machine. Update `project_root`, `source_route_root`, `output_root`, worker ports, and adapter assignments for a new installation.

Typical environment overrides are:

```bash
export SIMLINGO_ROOT=/path/to/simlingo
export PYTHON=/path/to/your/simlingo/python
```

## Collection workflow

Run commands from the repository root.

### 1. Build a frozen campaign manifest

```bash
python scripts/build_collection_manifest.py \
  --config config/decision_v1.json \
  --output manifests/collection_manifest.jsonl
```

### 2. Run static and live route preflight

```bash
python scripts/preflight_counterfactual_routes.py \
  --config config/decision_v1.json \
  --manifest manifests/collection_manifest.jsonl \
  --output data/counterfactual_decision_v1/_production/decision_v1/static_preflight.json

GPU_RANK=0 CARLA_GRAPHICS_ADAPTER=1 PORT=32100 \
bash scripts/run_live_route_preflight.sh \
  config/decision_v1.json \
  manifests/collection_manifest.jsonl \
  data/counterfactual_decision_v1/_production/decision_v1/live_preflight.json
```

### 3. Dry-run and launch the three-worker collector

```bash
python scripts/produce_counterfactual_routes.py \
  --config config/decision_v1.json \
  --manifest manifests/collection_manifest.jsonl \
  --live-preflight-report \
    data/counterfactual_decision_v1/_production/decision_v1/live_preflight.json \
  --workers 0,1,2 \
  --dry-run
```

After preflight reports `ready_for_collection=true`, remove `--dry-run` to begin collection. The controller supports safe interruption, strict cache reuse, bounded technical retries, branch repair, and decision-onset alignment repair.

Live terminal monitoring:

```bash
python scripts/watch_counterfactual_campaign.py \
  --config config/decision_v1.json \
  --manifest manifests/collection_manifest.jsonl \
  --runtime-progress \
    data/counterfactual_decision_v1/_production/decision_v1/runtime_progress.json
```

See [Route Mass Production](ROUTE_MASS_PRODUCTION.md) and the [Decision V1 Checklist](CHECKLIST_DECISION_V1.md) before starting a new campaign.

## Validation and reward annotation

Validate a completed campaign before computing training labels:

```bash
python scripts/validate_counterfactual_campaign.py \
  --config config/decision_v1.json \
  --manifest manifests/collection_manifest.jsonl \
  --output data/counterfactual_decision_v1/campaign_quality_report.json \
  --csv data/counterfactual_decision_v1/campaign_quality_table.csv
```

Compute factorized decisions, Decision Reward v2, and route-level soft targets:

```bash
python scripts/annotate_counterfactual_decision_reward_v2.py --write
python scripts/audit_counterfactual_decision_reward_v2.py
```

Reward annotation is atomic and idempotent. Collision, deadlock, and `lane_unavailable` are valid counterfactual outcomes, but they are not automatically positive imitation targets. Safe branches within the configured reward window receive a temperature-scaled soft-target distribution.

## Visualization

Build three reproducibly sampled S1–S3 route videos with synchronized branches and score overlays:

```bash
python scripts/build_sampled_reward_v2_route_videos.py
```

Outputs are written below `videos/`, which is deliberately ignored by Git.

## Data contract and safety properties

The collector preserves enough raw simulation state for later reward redesign, including ego control and motion, map/lane context, nearby actor states, collision events, decision onset, route progress, and 20 Hz telemetry.

Key safeguards include:

- route XML and manifest SHA-256 identity checks;
- isolated CARLA/Traffic Manager ports per worker;
- immediate termination on collision and bounded deadlock detection;
- explicit `lane_unavailable` outcomes instead of unsafe forced lane changes;
- strict sensor-count, event, trajectory-semantics, and onset-alignment validation;
- atomic promotion of repaired branches without silently deleting history.

## Dataset policy

The repository does **not** distribute the collected dataset. In particular, the following stay local:

- camera, BEV, depth, semantic, LiDAR, box, and measurement streams;
- compressed CARLA telemetry and recorder files;
- branch-level `quality_report.json` annotations generated inside the dataset;
- campaign manifests and runtime status;
- rendered videos and audit artifacts;
- checkpoints or other model binaries.

The root `.gitignore` enforces these boundaries. Before publishing changes, verify the payload with:

```bash
git status --short
git ls-files | grep -E '^(data|videos|logs|runtime|manifests|reward_audits)/' && \
  echo "ERROR: generated payload is tracked" || \
  echo "OK: generated payload is excluded"
```

## Research status

This repository is active research code. Interfaces and reward definitions may evolve as new counterfactual strategies and joint lateral/longitudinal decision branches are added. No dataset, pretrained model, or standalone CARLA environment is bundled.
