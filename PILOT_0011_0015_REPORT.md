# Decision-v1 五路线扩采报告（0011–0015）

生成日期：2026-08-24

## 结论

本轮为 S1–S6 每类冻结并采集 5 条新 route，共 30 个 route 组、150 条 decision 分支。最终分支级验收为 150/150，跨 decision 组级验收为 30/30，整批报告 `accepted=true`、`error_count=0`。

碰撞、死锁和无安全相邻车道均是被明确记录并立即终止/拒绝的有效结果，不等同于采集技术失败。

## 数量与结果

| 场景 | route 数 | decision 分支 | 完成 | 碰撞终止 | 死锁终止 | 无安全车道拒绝 | 变道执行/尝试 |
|---|---:|---:|---:|---:|---:|---:|---:|
| S1 遮挡行人 | 5 | 30 | 14 | 8 | 0 | 8 | 2/10 |
| S2 车辆切入 | 5 | 30 | 15 | 3 | 2 | 10 | 0/10 |
| S3 暴露障碍物 | 5 | 30 | 8 | 17 | 0 | 5 | 5/10 |
| S4 左转遇错误方向车辆 | 5 | 20 | 7 | 13 | 0 | 0 | 不适用 |
| S5 右转遇错误方向车辆 | 5 | 20 | 6 | 14 | 0 | 0 | 不适用 |
| S6 红灯右转让行 | 5 | 20 | 20 | 0 | 0 | 0 | 不适用 |
| **合计** | **30** | **150** | **70** | **55** | **2** | **23** | **7/30** |

Decision 数量：Accelerate 30、Maintain 30、Brake 30、Stop 30、LaneChangeLeft 15、LaneChangeRight 15。

## 自动验收结果

| 项目 | 阈值 | 全批最大值 | 结果 |
|---|---:|---:|---|
| decision 起点位置偏差 | ≤ 0.75 m | 0.396 m | 通过 |
| decision 起点朝向偏差 | ≤ 1.00° | 0.105° | 通过 |
| decision 起点速度偏差 | ≤ 0.75 m/s | 0.732 m/s | 通过 |
| 场景 actor 起点位置偏差 | ≤ 0.75 m | 0.537 m | 通过 |
| 分支级质量 | 150 条全通过 | 150/150 | 通过 |
| 组级完整性 | 30 组 decision 集合完整 | 30/30 | 通过 |
| 原始模态同步 | 每条 7 个保存模态帧数一致 | 150/150 | 通过 |
| 碰撞/死锁停止 | 末条 telemetry 不晚于终止 tick | 57/57 | 通过 |

碰撞 55 条均有 `collision` 终止记录；死锁 2 条均有 `scenario_deadlock` 终止记录。30 次变道尝试中，7 次在安全审计通过后执行，23 次因无安全相邻车道而拒绝。

S6 的交通约束覆盖只出现在非 Stop 分支且仅在实际遇到约束时触发；Stop 分支没有覆盖红灯/停车规则。

## 保存的数据

正式 150 条数据约 10.30 GB（9.59 GiB），共 53,682 个文件。累计保存：

- 7,044 个同步数据帧，每帧包含原始 RGB、深度、语义、BEV 语义、LiDAR、3D boxes 和 measurements。
- 34,917 个 20 Hz CARLA telemetry tick，包含 ego 状态、控制量、附近 actor、交通灯、车道、TTC/clearance 等 reward 设计所需信息。
- decision 事件、lane safety audit、碰撞/死锁终止原因、route 结果和逐分支质量报告。

Town01/Town02 及本轮扩展 route 启动 CARLA binary recorder 时会稳定触发 UE4 SIGSEGV，因此本轮明确关闭 `.log` binary recorder；20 Hz JSON telemetry 和全部原始传感器模态均保留。该策略在每条分支的事件和质量报告中有显式 warning，不是静默丢失。

历次合格但起点未对齐的尝试保存在 60 个 `_incomplete_*` 隔离目录中（约 2.71 GB），另有 2 个未提升候选（约 77 MB）。它们不在正式 manifest 中，不会计入正式 150 条训练分支，可在确认无需追溯后单独清理。

## 人工可视化抽查

已为 route index 0011 生成 S1–S6 对比视频。每个场景内的不同 decision 按 `decision_started` 对齐，窗口为决策前 1 秒至决策后 6 秒；早终止分支冻结末帧并显示结果。已抽查六个场景的 onset 和 post-decision 画面，未发现错 route、错 decision、黑帧或布局异常。

## 关键文件

- 配置：`config/decision_v1_pilot_0011_0015.json`
- 冻结 route：`routes/decision_v1_pilot_0011_0015/`
- 正式 manifest：`manifests/pilot_0011_0015_manifest.jsonl`
- 整批质量报告：`data/counterfactual_decision_v1/pilot_0011_0015_campaign_quality_report.json`
- 逐组表格：`data/counterfactual_decision_v1/pilot_0011_0015_campaign_quality_table.csv`
- 对齐候选选择审计：`data/counterfactual_decision_v1/pilot_0011_0015_alignment_selection_audit.json`
- 对比视频：`videos/decision_v1_route0011_comparison.mp4`
- 视频来源 manifest：`videos/decision_v1_route0011_comparison.manifest.json`
