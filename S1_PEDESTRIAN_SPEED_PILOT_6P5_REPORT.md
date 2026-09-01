# S1 pedestrian-speed pilot: route0050 at 6.5 m/s

Date: 2026-08-31

## Scope

- Source geometry: `closed_occluded_pedestrian_0050`
- Isolated pilot route: `closed_occluded_pedestrian_0050_speed65`
- Only changed scenario parameter: `ped_speed`, from `3.0` to `6.5 m/s`
- Unchanged parameters: `cross_dist=22.0`, `reaction_time=1.5`, `min_trigger_dist=9.0`, `hint_lead_distance=18.0`, `event_lead_distance=9.0`
- Dedicated ports: CARLA `32400`, Traffic Manager `42400`
- Existing production data and the active production collector were not modified.

## Result

| Decision | Accepted | Outcome | Pedestrian collisions | Route completion | Fidelity evidence | Pilot expectation |
|---|---:|---|---:|---:|---|---|
| Accelerate | yes | collision | 1 | 54.60% | entry 16.44 m/s; speed gain 3.37 m/s | pass |
| Maintain | yes | collision | 1 | 52.33% | entry 16.40 m/s; median deviation 0.05 m/s | pass |
| Brake | yes | completed | 0 | 100.00% | speed reduction 10.68 m/s | pass |
| Stop | yes | completed | 0 | 100.00% | full-stop hold 0.50 s | pass |
| LaneChangeLeft | yes | completed | 0 | 100.00% | lane audit safe; target hold 2.15 s | pass |
| LaneChangeRight | yes | lane_unavailable | 0 | 28.50% | unavailable direction safely rejected | pass |

The right side of this route has no valid target lane. The correct behavior is therefore a safe refusal, not a forced lane change.

## Counterfactual alignment

All six branches passed the campaign-level contract and onset-alignment checks:

- maximum ego position deviation: `0.0816 m`
- maximum ego speed deviation: `0.0419 m/s`
- maximum ego yaw deviation: `0.0517 deg`
- maximum scenario-actor position deviation: `0.0 m`
- campaign errors: `0`

## Artifacts

- Route XML: `routes/decision_v1_s1_speed_pilot_6p5/occluded_pedestrian/closed_occluded_pedestrian_0050_speed65.xml`
- Config: `config/s1_ped_speed_pilot_6p5.json`
- Manifest: `manifests/s1_ped_speed_pilot_6p5.jsonl`
- Raw branches: `data/counterfactual_decision_v1/_pilots/s1_ped_speed_6p5/S1/closed_occluded_pedestrian_0050_speed65/`
- Campaign report: `data/counterfactual_decision_v1/_pilots/s1_ped_speed_6p5/campaign_quality_report.json`
- Campaign table: `data/counterfactual_decision_v1/_pilots/s1_ped_speed_6p5/campaign_quality_table.csv`

## Interpretation

This single-route pilot meets the requested action-outcome separation. It does not yet prove that 6.5 m/s works for every S1 route, especially routes with substantially different ego entry speed, lane width, or pedestrian spawn offset. A multi-route validation is required before replacing the production S1 scenario.
