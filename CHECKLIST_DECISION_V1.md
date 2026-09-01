# CoDrive Decision V1 采集与验收 Checklist

## 1. 本版本范围

- [ ] 使用冻结的 CoDrive 路线，每个场景固定 10 条，共 60 个路线组。
- [ ] S1、S2、S3 每条路线采集 `Accelerate / Maintain / Brake / Stop / LaneChangeLeft / LaneChangeRight` 六个分支，各 60 条。
- [ ] S4、S5、S6 每条路线采集 `Accelerate / Maintain / Brake / Stop` 四个分支，各 40 条。
- [ ] 总数必须是 300 条：`60 + 60 + 60 + 40 + 40 + 40`。
- [ ] 不使用 `OffsetLeft`、`OffsetRight`，变道必须是进入完整相邻车道的 decision。
- [ ] 源路线、现有数据集和项目代码只读；新脚本、运行时 overlay、日志与数据全部位于 `simlingo/RL`。

Decision 的执行含义如下：

| Decision | 中文含义 | V1 执行规则 | 核心验收信号 |
|---|---|---|---|
| Accelerate | 加速通过 | 从 warning/hint 时刻的入口速度增加 3 m/s，上限 20 m/s；红灯/停车标志等硬约束可先强制停车，解除后必须重新加速 | 主动段最大速度相对入口至少增加 0.75 m/s；若先被硬约束压低，则解除后速度至少回升 0.75 m/s；若先碰撞，则记录碰撞并立即结束 |
| Maintain | 保持当前速度 | 把 warning/hint 时刻的车速冻结为目标速度，不是“按原 PDM 自由驾驶” | 主动段相对入口速度的中位偏差不超过 2 m/s |
| Brake | 制动减速但不停住 | 目标速度为入口速度的 45%，最低 1.5 m/s | 非碰撞轨迹至少降速 1 m/s；排除红灯/停车标志硬约束的时段后，不得连续低于 0.3 m/s 达 0.5 s；碰撞把车物理钉停时仍标为 Brake+collision，不改写成 Stop |
| Stop | 完全停车 | 目标速度为 0，并保持停车 | 低于 0.3 m/s 的连续时间至少 0.5 s；Stop 的预期停车不算 decision 内死锁 |
| LaneChangeLeft | 向左完整变道 | 忽略车道线是否合法，平滑进入左侧完整相邻 Driving lane，危险解除后回归原路线 | 目标车道安全审计通过，实际到达并保持目标 lane 至少 0.5 s |
| LaneChangeRight | 向右完整变道 | 与左变道对称 | 同上；若不存在安全可驾驶车道，则应得到显式 `lane_unavailable`，不得硬变 |

## 2. 每条路线开始前

- [ ] 由 `manifests/collection_manifest.jsonl` 锁定 route XML、route id、town、decision、worker 和 SHA256。
- [ ] 同一路线的所有 decision 使用同一个冻结 XML；禁止为某个 decision 单独移动触发点或危险参与者。
- [ ] 关闭随机背景交通，保留场景定义的车辆、行人、障碍物、交通灯和天气，避免不同分支的随机交通破坏起点对齐。
- [ ] GPU 0/1/2 分别使用隔离的 CARLA / Traffic Manager 端口和独立 RL runtime overlay。
- [ ] 启动前满足 config 的 `minimum_free_disk_gb`（当前默认 20 GB）；低于阈值时调度器停止提交新任务。空间不足时优先清理可重建的 `*_augmented` 派生视图，绝不删除原始 RGB/Depth/Semantics/BEV/LiDAR/boxes/measurements/20 Hz telemetry。
- [ ] decision 只能在场景官方 `hint_reached=true` 的首个策略 tick 开始，不能根据固定时间或人工估计提前开始。
- [ ] 在 decision 施加前记录入口位置、姿态、车速、lane、场景 actor 状态，供六/四个分支做反事实起点一致性检查。

## 3. 变道安全审计

对 S1、S2、S3 的每个左右变道分支逐项检查：

- [ ] 允许跨实线、双黄线或驶入反向车流方向，不使用车道线规则否决变道。
- [ ] 相邻位置必须能投影到 CARLA `LaneType.Driving`，目标车道宽度至少 2.4 m。
- [ ] 从变道前 14 m 到危险点后 28 m 的目标 lane 几何覆盖率至少 80%。
- [ ] 生成 6 m 平滑横向过渡，而不是瞬移或只偏移半个车身。
- [ ] 目标 lane 前方 35 m、后方 15 m 范围内检查所有动态车辆和行人。
- [ ] 最小预测 TTC 必须不低于 4 s；保存每个冲突 actor 的 id、类型、相对纵横向距离和相对速度。
- [ ] 审计通过后，ego 必须实际进入目标 lane id，并至少保持 0.5 s。
- [ ] 审计失败时立即结束为 `lane_unavailable`，在 termination 中保存方向、覆盖率、车道宽度、冲突 actor 和失败原因；这代表“安全拒绝”，不是采集故障。
- [ ] 若 PDM 私有 hazard 列表在官方提示 tick 为空，车道审计必须回退到冻结 route XML 的 `collision_point`，并记录 `reference_source=frozen_route_collision_point`；不得索引空列表或跳过安全检查。
- [ ] 对跨黄线进入反向 Driving lane 的轨迹，保留 `same_direction_ratio` 和 `lane_marking_violation_ratio`，不因违反交通线规则而判错。

## 4. 六个场景的逐场景操作与检查

| 场景 | 每条路线需要运行 | 操作重点 | 查验重点 | 场景级通过条件 |
|---|---|---|---|---|
| S1 行人从遮挡物后出现 | 6 decisions | 在行人预警到达时施加纵向或完整左右变道；变道前同时检查行人与目标 lane | 确认 scenario walker 和遮挡物存在；hint 后行人进入 ego 冲突区；碰撞对象若为 walker，collision frame 与 termination frame 必须一致 | 10 组 × 6；纵向语义通过；左右各有安全审计结果，执行或安全拒绝均有证据 |
| S2 车辆切入 | 6 decisions | 在切入车预警时施加 decision；变道审计必须把切入车及目标 lane 后车纳入动态冲突 | 检查切入车的横向运动、速度、lane id 和与 ego 的 TTC；禁止把普通跟车减速误写成 Brake decision | 10 组 × 6；切入 actor 状态完整；执行变道时目标 lane 无冲突，碰撞/死锁即时终止 |
| S3 前车让出后暴露障碍物 | 6 decisions | 前车开始 reveal 时施加 decision；变道既要绕过静态/慢速障碍，也不能与 reveal 车辆冲突 | 检查 reveal 车辆、被暴露障碍和 ego 的相对位置；确认变道是完整邻 lane，而非 Offset；障碍最小 clearance 被保存 | 10 组 × 6；六个分支起点一致；安全变道达到目标 lane，不安全方向明确拒绝 |
| S4 左转遇逆行/错误方向车辆 | 4 longitudinal decisions | 只改变纵向目标速度，不加变道；保留错误方向车辆的真实轨迹 | 检查 ghost vehicle 从左转相关冲突方向接近；Accelerate/Maintain/Brake/Stop 的速度曲线可区分 | 10 组 × 4；危险 actor、相对速度/TTC 完整；碰撞或死锁有终止标记 |
| S5 右转遇逆行/错误方向车辆 | 4 longitudinal decisions | 与 S4 对称，只采纵向 decision | 检查右转冲突几何、ghost vehicle 的 lane/heading；确认 Brake 不变成 Stop | 10 组 × 4；四条轨迹语义可区分，场景 actor 在 decision 起点对齐 |
| S6 红灯右转让行 | 4 longitudinal decisions | 保留红灯和冲突交通；为形成反事实轨迹，非 Stop 分支在 warning 后仅解除该场景的红灯/路口速度上限，具体 override 写入原始遥测 | 检查 `regulatory_constraint_overridden`/`regulatory_override` 只出现在 S6 非 Stop；Stop 必须停车；仍保存交通灯状态、冲突 actor 和 TTC | 10 组 × 4；四种速度行为确实不同；override 可追溯且没有扩散到 S1–S5 |

## 5. 运行中立即终止规则

- [ ] 复用 Leaderboard `CollisionTest` 的唯一碰撞传感器；禁止动态再挂第二个 collision sensor。
- [ ] 碰撞回调记录 `world_frame`、对方 actor id/type、冲量；下一控制 tick 前终止，termination 的 world frame 必须等于碰撞 frame。
- [ ] 除 Stop 外，decision 主动阶段水平车速（只取 x/y，不把车身竖直抖动算作前进）低于 0.1 m/s 连续 2 s，结束为 `vehicle_deadlock`。
- [ ] 危险未解除且 decision 主动时间达到 20 s，结束为 `scenario_deadlock`。
- [ ] 危险解除后仍低速连续 3 s，结束为 `post_clear_deadlock`。
- [ ] `collision`、三类 `deadlock` 和 `lane_unavailable` 都是带明确标签的有效终止结果，不应重跑成另一个世界状态。
- [ ] CARLA 段错误、超时、无 termination 的非零 runner 状态、损坏 gzip 或丢失 decision onset 属于技术失败：隔离为 `_incomplete_TIMESTAMP`，自动重试，不得计入 300 条。

## 6. 单分支自动验收

- [ ] `quality_report.json` 中 `accepted=true`。
- [ ] `run_spec.json`、采集时复制的 `metadata/source_route.xml`、manifest route identity 与 SHA256 完全一致；route XML 修改后不得命中旧缓存。
- [ ] `decision_started` 恰好出现一次，且 `event_state.hint_reached=true`。
- [ ] 正常完成分支至少保存 15 个 measurement frame；collision/deadlock 至少 10 个；变道安全审计失败必须立即结束，因此 `lane_unavailable` 至少 5 个，同时要求 20 Hz raw telemetry 和完整失败审计存在。
- [ ] 原始 RGB、depth、semantics、BEV、lidar、boxes、measurements 的数量差不超过 2。
- [ ] `*_augmented` 是可从原始模态重建的派生视图；空间策略会在验收前记录其文件数/字节数并移除。不得把派生视图缺失误报为原始 CARLA 信息缺失。
- [ ] 原始 gzip telemetry 可完整解压，且存在 decision-active 帧。
- [ ] 正常完成分支 route completion 不低于 99%，status 为 `Completed` 或 `Perfect`。
- [ ] 碰撞结果同时出现在 raw collision event、termination 和 evaluator collision criterion 中；如果 evaluator 看到碰撞但 agent 没立即终止，判失败。
- [ ] Accelerate、Maintain、Brake、Stop 分别满足第 1 节速度语义阈值。
- [ ] 变道成功分支满足 `lane_audit_safe=true`、进入 target lane、保持至少 0.5 s；变道拒绝分支必须有 `safe=false` 的完整审计。
- [ ] 保存 reward 设计所需 raw 指标：每 tick ego control/pose/velocity/acceleration/angular velocity/lane、route index、天气、交通灯、100 m 内 actor 的 transform/velocity/acceleration/bbox/attributes/lane、clearance、TTC、碰撞和场景状态。
- [ ] 在已验证稳定的 Town01/Town02 保存 `raw/carla_recorder.log`。Town04/Town05 的 `start_recorder` 已复现 UE4 SIGSEGV，因此较大地图禁用 recorder，并要求事件中存在显式 `carla_recorder_disabled`；Town06/Town10HD 也按同一保守规则处理。逐 tick telemetry 和所有传感器模态仍必须完整。

## 7. 同一路线跨 decision 验收

- [ ] 每组的 decision 集合与配置完全一致，不缺、不多、不重复。
- [ ] 以 Maintain 为参考，所有分支 decision onset 的 ego 位置差不超过 0.75 m（20 Hz 控制下高速车辆一个离散 tick 的最大位移约 0.7 m）。
- [ ] onset yaw 差不超过 1°，入口速度差不超过 0.75 m/s。
- [ ] onset 时所有 `role_name=scenario` actor 的类型集合一致，匹配 actor 的位置差不超过 0.75 m。
- [ ] 比较速度曲线、lane id 序列、最小 clearance/TTC 和 outcome，确认不同 decision 产生可解释的轨迹差异，而不是同一轨迹复制。
- [ ] 同一组中任何一个分支技术失败，则该组保持未通过，直到该分支重采并单分支验收通过。

## 8. 人工抽查

- [ ] 每个场景至少抽查 2 个路线组；每个 decision 至少看 2 条，collision/deadlock/lane_unavailable 全部查看。
- [ ] 每条抽查 warning 前、decision onset、最近危险点、结束/终止四个时间点的 RGB、BEV、depth/semantics 与 measurement。
- [ ] 检查场景 actor 无瞬移、穿模、错误生成或明显不同步；ego 无横向瞬移、路线折返或控制抖动。
- [ ] 对成功变道，视觉确认整个车身进入相邻 lane；对安全拒绝，结合地图 lane 与 actor 列表确认拒绝原因合理。
- [ ] 对碰撞，视觉帧、collision actor 类型和冲量方向一致；对死锁，确认不是 Stop 的正常停车或红灯临时等待。
- [ ] 对 S6，确认 override 只用于构造 decision 反事实，不会删除冲突 actor 或伪造绿灯状态。

## 9. 执行与最终签收

```bash
cd /home/UNT/yl0826/simlingo/RL
python scripts/build_collection_manifest.py \
  --config config/decision_v1.json \
  --output manifests/collection_manifest.jsonl
bash scripts/run_live_route_preflight.sh \
  config/decision_v1.json manifests/collection_manifest.jsonl \
  data/counterfactual_decision_v1/_production/decision_v1/live_preflight.json
python scripts/produce_counterfactual_routes.py \
  --config config/decision_v1.json \
  --manifest manifests/collection_manifest.jsonl \
  --live-preflight-report \
    data/counterfactual_decision_v1/_production/decision_v1/live_preflight.json \
  --workers 0,1,2 --walltime 300
```

- [ ] `campaign_quality_report.json` 显示 `observed_groups=60`、`observed_branches=300`、`accepted_branches=300`、`accepted=true`。
- [ ] `campaign_quality_table.csv` 有 300 条数据行，可按 scene/route/decision/outcome 筛选。
- [ ] 每个场景正好 10 个路线组；S1–S3 各 60 条，S4–S6 各 40 条。
- [ ] 最终报告单独统计 `completed / collision / deadlock / lane_unavailable`，不把终止结果包装成“安全完成”。
- [ ] 最终确认所有新增或修改内容都在 `/home/UNT/yl0826/simlingo/RL` 内。
