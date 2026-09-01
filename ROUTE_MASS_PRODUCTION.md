# Multi-decision route 量产流程

本流程的目标是把后续批次从“逐条人工盯采”改成“自动门禁 + 异常复核”。它不假设脚本永远不会出错：正常分支无需逐条看视频，但任何 route 变更、技术失败、对齐修复和异常指标都会留下报告并进入复核队列。

## 1. 安全边界

- 所有新增 route、manifest、脚本、报告和数据都写在 `simlingo/RL` 内。
- GPU 0/1/2 使用 config 中独立的 CARLA/TM 端口；live 预检默认使用 GPU0、adapter1、32100。
- `run_live_route_preflight.sh` 会先检查端口，只终止自己启动的 CARLA PID，不使用 `pkill`，不接管 21000 等外部端口。
- 缓存必须同时满足：`quality_report.accepted=true`、run spec、采集时复制的 `source_route.xml`、manifest identity 和 route SHA256 完全一致。
- route XML 或 manifest 被改动后，旧轨迹自动变为 stale，不会静默复用。
- `collision`、`deadlock`、`lane_unavailable` 是有效 outcome，不等于安全成功；只有没有明确 termination 的崩溃、超时或缺数据才自动技术重试。

## 2. 新批次的一次性流程

下面以新 config 和 manifest 为例。先冻结 route 到 `RL/routes/<batch>`，并令 config 的 `source_route_root` 指向该目录；不要让正式 manifest 直接引用仍会变化的候选目录。

```bash
cd /home/UNT/yl0826/simlingo/RL

python scripts/build_collection_manifest.py \
  --config config/decision_v1_batch_0021_0070.json \
  --output manifests/batch_0021_0070_manifest.jsonl
```

构建 manifest 会检查每个场景的 route 数量、单 XML 单 route/scenario、route id/hash 唯一性、job/output 唯一性，以及 XML identity 与 manifest 一致性。

先做静态预检；这一步不启动 CARLA：

```bash
python scripts/preflight_counterfactual_routes.py \
  --config config/decision_v1_batch_0021_0070.json \
  --manifest manifests/batch_0021_0070_manifest.jsonl \
  --output data/counterfactual_decision_v1/_production/batch_0021_0070/static_preflight.json
```

随后做一次整批 live 地图拓扑预检：

```bash
GPU_RANK=0 CARLA_GRAPHICS_ADAPTER=1 PORT=32100 \
bash scripts/run_live_route_preflight.sh \
  config/decision_v1_batch_0021_0070.json \
  manifests/batch_0021_0070_manifest.jsonl \
  data/counterfactual_decision_v1/_production/batch_0021_0070/live_preflight.json
```

live 预检按 Town 分组，只加载每张地图一次。硬门禁包括事件点可投影到 Driving lane、锚点 route 可达、S6 左侧冲突来车入口存在、实际加载 Town 稳定一致。没有同向相邻 lane 是警告而不是错误，因为该 decision 应在正式轨迹中记录为 `lane_unavailable`。稀疏首尾 waypoint 的替代路径歧义也会保留为复核警告。

先 dry-run，确认哈希报告匹配、任务数、缓存数、磁盘和 worker 分配：

```bash
python scripts/produce_counterfactual_routes.py \
  --config config/decision_v1_batch_0021_0070.json \
  --manifest manifests/batch_0021_0070_manifest.jsonl \
  --live-preflight-report data/counterfactual_decision_v1/_production/batch_0021_0070/live_preflight.json \
  --workers 0,1,2 \
  --dry-run
```

`ready_for_collection=true` 后去掉 `--dry-run` 启动量产：

```bash
python scripts/produce_counterfactual_routes.py \
  --config config/decision_v1_batch_0021_0070.json \
  --manifest manifests/batch_0021_0070_manifest.jsonl \
  --live-preflight-report data/counterfactual_decision_v1/_production/batch_0021_0070/live_preflight.json \
  --workers 0,1,2 \
  --walltime 300 \
  --technical-retries 2 \
  --max-branch-repair-rounds 2 \
  --max-alignment-rounds 5
```

不要把 `--allow-static-only` 用作常规量产参数。它仅用于人在场监督、明确接受尚未做 live 拓扑检查的例外。

## 3. 控制器自动执行的阶段

1. 对完整 manifest 做静态结构与 SHA256 契约检查。
2. 校验 live 报告绑定当前 config、manifest、预检器和共享契约器版本。
3. 先采集所有 route 的 `Maintain` 冒烟分支；技术失败自动隔离并重试。
4. 再采集所有 decision；严格缓存自动跳过已验收分支。
5. 做单分支 quality、route/source 契约和同 route 跨 decision onset 对齐验收。
6. 对被拒分支做有限轮定向重采，不重写 collision/deadlock/lane_unavailable outcome。
7. 对 onset 不对齐组扫描 official 与 `_incomplete_*` 历史候选；能组成对齐集合时原子提升最佳候选。
8. 没有可用对齐集合时，只重采与最佳 Maintain reference 不兼容的 decision，而不是整组盲目重跑。
9. 最终 campaign 必须 `accepted=true` 才签收。

控制器可安全中断和重新运行。旧 official 分支在替换前被移到带时间戳的 `_incomplete_*`，不会直接删除。

## 4. 报告与复核策略

每个 manifest 的控制器状态位于：

```text
data/counterfactual_decision_v1/_production/<manifest_stem>/
  production_status.json
  static_preflight.json
  latest_collection_status.json
  campaign_initial.json / .csv
  alignment_round_*.json
  campaign_final.json / .csv
```

不再要求每条正常轨迹都人工验证。建议复核规则如下：

| 情况 | 是否必须人工看 |
|---|---|
| 新 scenario/PDM-lite 规则首次出现 | 是；先看每个 decision 的少量 route，再放量 |
| 自动验证全通过、无重试、指标在既有分布内 | 不逐条看；每场景随机抽 route group |
| collision、deadlock | 是；核对即时终止与碰撞/低速证据 |
| lane_unavailable | 首批与新 Town 必看；稳定后按失败原因分层抽样 |
| 成功左右变道 | 每个场景/方向至少保留可视抽样，核对完整进入相邻 lane |
| branch repair、alignment replacement、stale cache | 是；这些是异常队列 |
| live endpoint-path 歧义或新 junction warning | 是；必要时给 XML 增加中间 waypoint 后重建 manifest/live 报告 |

量产通过的定义不是“脚本跑完”，而是 `production_status.json.accepted=true` 且 `campaign_final.json.accepted=true`。奖励高低不参与采集验收，也不能把碰撞轨迹误标成高质量 IL 真值。

## 5. 当前回归基线

- 三批现有数据：120 个 route group、600 个 decision branch。
- 严格 source/manifest 哈希回归：600/600 通过。
- campaign 对齐回归：600/600 通过。
- 最近 0016–0020 批次 live 拓扑：30/30 route、8 个 Town、0 hard error。
- 18 个 live 警告为没有同向相邻 lane，正式分支应对应安全拒绝的 `lane_unavailable`。
- GPU0 live 预检结束后 32100 已关闭；外部 21000 未触碰。
