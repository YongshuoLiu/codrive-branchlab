# Multi-decision routes 0001–0020 collection report

Date: 2026-08-25

## Result

The official dataset now contains 120 counterfactual route groups and 600 accepted decision branches. All three campaign-level quality reports pass with zero errors and zero campaign warnings:

| Campaign | Route indices per scene | Groups | Branches | Accepted |
|---|---:|---:|---:|---:|
| Base | 0001–0010 | 60 | 300 | 300 |
| Pilot 1 | 0011–0015 | 30 | 150 | 150 |
| Pilot 2 | 0016–0020 | 30 | 150 | 150 |
| Total | 0001–0020 | 120 | 600 | 600 |

S1–S3 each contain 20 unique frozen routes and six decisions per route: `Accelerate`, `Maintain`, `Brake`, `Stop`, `LaneChangeLeft`, and `LaneChangeRight`. S4–S6 each contain 20 unique frozen routes and four decisions per route: `Accelerate`, `Maintain`, `Brake`, and `Stop`.

The request to continue with ten more routes per scene could only contribute five new unique closed-loop routes per scene in this phase: routes 0016–0020 were the last unused routes in the 20-route LANTERN closed-loop catalog. Routes were not duplicated or relabeled to pretend that 0021–0025 existed. The older 0001–0010 campaign was also audited and its 48 missing branches were completed, so the usable result is now 20 unique routes per scene.

## Strict quality result

The combined official-set audit reports:

- 600/600 accepted branch directories and 120/120 route groups.
- 20 unique route IDs and 20 unique frozen route hashes for every scene.
- Maximum onset position deviation: 0.683 m, threshold 0.75 m.
- Maximum onset yaw deviation: 0.727 degrees, threshold 1.0 degree.
- Maximum onset speed deviation: 0.742 m/s, threshold 0.75 m/s.
- Maximum scenario-actor deviation: 0.538 m, threshold 0.75 m.
- No combined-audit errors.

The official outcomes are:

| Scene | Branches | Completed | Collision | Deadlock | Lane unavailable |
|---|---:|---:|---:|---:|---:|
| S1 | 120 | 67 | 26 | 0 | 27 |
| S2 | 120 | 50 | 29 | 5 | 36 |
| S3 | 120 | 51 | 54 | 0 | 15 |
| S4 | 80 | 26 | 54 | 0 | 0 |
| S5 | 80 | 29 | 50 | 1 | 0 |
| S6 | 80 | 77 | 3 | 0 | 0 |
| Total | 600 | 300 | 216 | 6 | 78 |

Collision and deadlock are valid counterfactual outcomes and terminate collection immediately. `lane_unavailable` means the lane-change safety gate rejected the maneuver; no intentionally crashing lane change was generated.

## Raw-data inventory

Only manifest-referenced official directories are included in these totals; `_incomplete_*` and `_alignment_displaced_*` audit candidates are excluded.

- Official size: 29.949 GiB.
- Per-tick CARLA telemetry records: 123,359.
- Synchronized saved frames per modality: 24,913.
- Retained modalities: RGB, depth, semantics, BEV semantics, LiDAR, boxes, measurements, decision events, scenario events, frame clock, run specification, source route, evaluator result, and termination metadata.

There are 565 branch-level warnings, all documenting the intentional CARLA binary-recorder fallback. `start_recorder` reproducibly segfaulted UE4 on these collection paths, so binary `.log` recording was disabled. Per-tick telemetry and all listed original sensor modalities remain available and synchronized.

## S6 route0004 repair

The original Town04 source route used for S6 route0004 was unusable: CARLA junction 850 had no left-direction conflict entry and the scenario trigger was outside the accepted route-trace distance. It was replaced inside `RL/routes` with an unused Town01 closed-loop S6 candidate from the catalog, while retaining the frozen `closed_right_turn_on_red_0004` identifier.

The replacement passed live CARLA topology checks: trigger projection error 0.004 m, route-trace distance 0.455 m, and one valid left-direction conflict entry. Its event trigger was moved earlier on the same straight lane so the Maintain onset speed changed from 16.31 m/s to 11.98 m/s; the final Maintain median speed deviation is 0.24 m/s. All four S6 decisions then passed branch and group quality gates. Full source and hash provenance is recorded next to the XML.

## Main artifacts

- Combined audit: `data/counterfactual_decision_v1/routes_0001_0020_combined_audit.json`
- Routes 0001–0010 report: `data/counterfactual_decision_v1/routes_0001_0010_campaign_quality_report.json`
- Routes 0011–0015 report: `data/counterfactual_decision_v1/pilot_0011_0015_campaign_quality_report.json`
- Routes 0016–0020 report: `data/counterfactual_decision_v1/pilot_0016_0020_campaign_quality_report.json`
- Official manifests: `manifests/collection_manifest.jsonl`, `manifests/pilot_0011_0015_manifest.jsonl`, and `manifests/pilot_0016_0020_manifest.jsonl`
- Combined audit script: `scripts/audit_combined_campaigns.py`
- Campaign validator: `scripts/validate_counterfactual_campaign.py`
- Alignment candidate selector: `scripts/select_alignment_candidates.py`
- CARLA topology inspector: `scripts/inspect_carla_route_topology.py`
- S6 repair provenance: `routes/decision_v1/right_turn_on_red/closed_right_turn_on_red_0004.provenance.json`

## Reproduction

```bash
python3 scripts/validate_counterfactual_campaign.py \
  --config config/decision_v1.json \
  --manifest manifests/collection_manifest.jsonl \
  --output data/counterfactual_decision_v1/routes_0001_0010_campaign_quality_report.json \
  --csv data/counterfactual_decision_v1/routes_0001_0010_campaign_quality_table.csv

python3 scripts/audit_combined_campaigns.py \
  --output data/counterfactual_decision_v1/routes_0001_0020_combined_audit.json
```

Creating five additional unique routes per scene beyond this catalog requires a separate route-extension phase: adapt open-loop `GhostProbe*` candidates to the closed-loop `Closed*` event interfaces, validate CARLA topology and event timing, and then collect routes 0021–0025. That adaptation was not silently mixed into this accepted dataset.
