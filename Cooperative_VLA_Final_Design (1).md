# Cooperative VLA Beyond Imitation
## Counterfactual Decision Learning with Pseudo-Simulation Reinforcement Learning

## 0. 文档定位

本文档给出当前项目的统一最终设计。核心目标是解决 Coop-SimLingo 中由规则型 expert（如 PDM-Lite）带来的保守决策偏置，并将训练范式从单纯 imitation learning 扩展为：

\[
\boxed{
\text{Execution Learning}
\rightarrow
\text{Decision Learning}
\rightarrow
\text{Outcome Optimization}
}
\]

正式采用三阶段训练：

1. **Stage 1：Decision-conditioned Trajectory Executor Training**
2. **Stage 2：High-level Decision Policy Supervised Initialization**
3. **Stage 3：Pseudo-Simulation-based Reinforcement Learning**

---

# 1. 研究动机

原始 cooperative driving 训练可以抽象为：

\[
(s_t,W_t)\rightarrow \tau_t^{expert}
\]

其中：

- \(s_t\)：当前驾驶状态；
- \(W_t\)：cooperative warning；
- \(\tau_t^{expert}\)：expert trajectory。

PDM-Lite 在 safety-critical 场景中通常偏保守，因此模型容易学到：

\[
\boxed{
Warning\rightarrow Brake
}
\]

形成：

\[
\text{Rule-based Expert Bias}
\rightarrow
\text{Imitation Learning}
\rightarrow
\text{Conservative Cooperative Policy}
\]

项目目标是让模型学习：

\[
\boxed{
Image+Warning+Ego+Route
\rightarrow
Adaptive Decision
\rightarrow
Trajectory
}
\]

而不是把 warning 当成动作命令。

---

# 2. 核心研究问题

- **RQ1 Expert Bias**：单一保守 expert 是否会使 cooperative VLA 继承 braking bias？
- **RQ2 Execution**：给定 high-level decision，模型能否生成稳定且明显不同的对应 trajectory？
- **RQ3 Decision Selection**：模型能否根据视觉、warning、ego、route 选择合适 decision？
- **RQ4 Beyond Imitation**：reward-based post-training 是否能超过 expert 的保守 decision preference？
- **RQ5 Warning Generalization**：不显式设计 hazard parser 时，VLM 是否能泛化到 paraphrase、unseen composition 和 task-unseen warning？

---

# 3. 总体模型

将 policy 分为：

\[
\boxed{\pi_D}
\]

和：

\[
\boxed{\pi_T}
\]

其中：

- \(\pi_D\)：high-level decision policy，回答“现在应该做什么？”
- \(\pi_T\)：decision-conditioned trajectory policy，回答“这个 decision 应该如何执行？”

完整关系：

\[
\boxed{
s_t
\xrightarrow{\pi_D}
m_t
\xrightarrow{\pi_T}
\tau_t
}
\]

---

# 4. 符号定义

| 符号 | 含义 |
|---|---|
| \(t\) | 底层驾驶时间索引 |
| \(s_t\) | 当前驾驶状态 |
| \(I_t\) | 当前视觉 observation |
| \(W_t\) | cooperative warning，无 warning 时为 \(\varnothing\) |
| \(E_t\) | ego state，例如速度、加速度 |
| \(N_t\) | route / navigation 信息 |
| \(F_\theta\) | SimLingo shared VLM backbone |
| \(H_t\) | backbone 输出的 multimodal hidden representation |
| \(\pi_D\) | high-level decision policy |
| \(m_t\) | high-level driving decision |
| \(m_t^{lon}\) | longitudinal decision |
| \(m_t^{lat}\) | lateral decision |
| \(m_t^{GT}\) | Stage 2 中 decision 的监督标签 |
| \(m_t^{exec}\) | 某条 trajectory 实际对应的 execution decision |
| \(\pi_T\) | decision-conditioned trajectory generator |
| \(\tau_t\) | 模型生成的未来 trajectory |
| \(\tau_t^{GT}\) | trajectory supervision |
| \(H\) | trajectory horizon，例如 10 steps |
| \(K\) | warning 起点下离线生成的 candidate 数量 |
| \(G\) | RL 中同一 state 采样的 decision 数量 |
| \(S_i\) | candidate \(i\) 的 open-loop surrogate score |
| \(r_{1,i}\) | candidate \(i\) 的 Stage-1 pseudo-simulation reward |
| \(r_{2,i,j}\) | candidate \(i\) 在第 \(j\) 个 Stage-2 observation 上的 reward |
| \(R_i^{PS}\) | candidate \(i\) 的总 pseudo-simulation reward |
| \(A_i\) | candidate \(i\) 的 group-relative advantage |
| \(\gamma\) | future reward discount factor |
| \(\mathcal L_T\) | trajectory imitation loss |
| \(\mathcal L_D\) | decision supervised loss |
| \(\mathcal L_{RL}\) | reinforcement-learning policy loss |
| \(\theta_D\) | decision-policy 参数 |
| \(\theta_T\) | trajectory-policy 参数 |

状态统一写为：

\[
\boxed{
s_t=(I_t,W_t,E_t,N_t)
}
\]

---

# 5. High-Level Decision Space

采用 factorized decision：

\[
\boxed{
m_t=(m_t^{lon},m_t^{lat})
}
\]

Longitudinal：

\[
m_t^{lon}\in
\{
Accelerate,\ Maintain,\ MildBrake,\ HardBrake,\ Stop
\}
\]

Lateral：

\[
m_t^{lat}\in
\{
RouteFollow,\ OffsetLeft,\ OffsetRight,\ LaneChangeLeft,\ LaneChangeRight,\ ReturnToRoute
\}
\]

例如：

\[
m_t=(MildBrake,OffsetLeft)
\]

表示轻度减速并向左偏移避险。

---

# 6. 数据集改造总原则

数据集要同时支持：

- trajectory execution learning；
- decision selection learning；
- pseudo-simulation RL。

数据分为：

1. **No-warning 数据**
2. **Warning counterfactual 数据**

---

# 7. No-Warning 数据

即使：

\[
W_t=\varnothing
\]

也让：

\[
\pi_D
\]

输出 decision。

因为 \(\pi_D\) 应当是 general driving decision policy，而不是 warning-response classifier。

No-warning decision labels 可从原 GT trajectory 自动提取，例如：

- 正常巡航：\(Maintain+RouteFollow\)
- 加速：\(Accelerate+RouteFollow\)
- 前车减速：\(MildBrake+RouteFollow\)
- 停车：\(Stop+RouteFollow\)
- 正常换道：\(Maintain+LaneChangeLeft\)
- hazard 后恢复：\(Accelerate+ReturnToRoute\)

不能把所有 no-warning 数据都标为 Maintain，否则会产生：

\[
NoWarning\rightarrow Maintain
\]

的新 shortcut。

---

# 8. Warning Counterfactual 数据

在 warning 的共同起点状态：

\[
s_0
\]

构造多个 candidate decisions：

\[
\mathcal M(s_0)=\{m_1,\dots,m_K\}
\]

例如：

\[
m_1=HardBrake+RouteFollow
\]

\[
m_2=MildBrake+RouteFollow
\]

\[
m_3=MildBrake+OffsetLeft
\]

\[
m_4=Maintain+RouteFollow
\]

\[
m_5=Maintain+LaneChangeLeft
\]

每个 candidate 对应：

\[
(s_0,m_i)\rightarrow\tau_i
\]

并计算 open-loop surrogate score：

\[
S_i
\]

以及原始 metrics：

- collision；
- drivable-area compliance；
- route progress；
- TTC；
- safety margin；
- comfort；
- legality。

数据中不要只保留一个 score，应保留原始 metrics，便于以后重定义 ranking。

---

# 9. Open-Loop 数据的硬约束

当前数据集不是可 reset 的闭环 simulator。

因此从共同状态：

\[
s_0
\]

分叉后：

\[
s_1^{Brake}
\neq
s_1^{Offset}
\neq
s_1^{Maintain}
\]

不同 branch 后续 state：

- 图像不同；
- 位置不同；
- 速度不同；
- heading 不同；
- hazard 相对位置不同。

因此它们不能再被错误地当成同一个 state group。

也就是说：

\[
\boxed{
\text{branch 后续 states 不做跨 branch grouped decision supervision}
}
\]

这也是为什么我们最终不采用 joint grouped IL，而改成严格三阶段训练。

---

# 10. Stage 1：Decision-conditioned Trajectory Executor Training

Stage 1 只训练：

\[
\boxed{
\pi_T
}
\]

目标：

> 给定 decision，学会如何执行。

训练数据：

\[
\boxed{
\mathcal D_T=
\{
(s_t,m_t^{exec},\tau_t^{GT})
\}
}
\]

前向：

\[
\boxed{
\hat\tau_t
=
\pi_T(s_t,m_t^{exec})
}
\]

其中：

- \(m_t^{exec}\)：当前这条 branch 实际对应的 decision；
- \(\tau_t^{GT}\)：该 branch 对应 trajectory。

这一阶段完全不训练“哪个 decision 更好”。

---

# 11. Stage 1 为什么可以使用 Sliding Window

因为 Stage 1 不比较不同 branch。

例如 Brake branch：

\[
s_1^B,s_2^B,\dots
\]

都可以形成：

\[
\pi_T(s_t^B,Brake)\rightarrow\tau_t^B
\]

Offset branch：

\[
s_1^O,s_2^O,\dots
\]

都可以形成：

\[
\pi_T(s_t^O,OffsetLeft)\rightarrow\tau_t^O
\]

所以不同 branch 后续 state 不需要对齐。

因此：

\[
\boxed{
Stage1:\ dense\ sliding-window\ trajectory\ supervision
}
\]

---

# 12. Stage 1 Loss

SimLingo 原有 path / speed prediction 可继续使用：

\[
\mathcal L_{path}
=
SmoothL1(\hat p_t,p_t^{GT})
\]

\[
\mathcal L_{speed}
=
SmoothL1(\hat v_t,v_t^{GT})
\]

总 trajectory loss：

\[
\boxed{
\mathcal L_T
=
\lambda_p\mathcal L_{path}
+
\lambda_v\mathcal L_{speed}
}
\]

优化：

\[
\boxed{
\min_{\theta,\theta_T}\mathcal L_T
}
\]

得到：

\[
\boxed{
\pi_T^{IL}
}
\]

---

# 13. Stage 1 的 SimLingo 结构改造

原 SimLingo 有：

- path queries；
- speed waypoint queries。

增加 execution decision embedding：

\[
e_D=E(m_t^{exec})
\]

然后 condition 原 hidden features：

\[
\tilde H_P
=
LN(H_P+W_Pe_D)
\]

\[
\tilde H_V
=
LN(H_V+W_Ve_D)
\]

再：

\[
\tilde H_P\rightarrow PathHead
\]

\[
\tilde H_V\rightarrow SpeedHead
\]

因此显式得到：

\[
\boxed{
\pi_T(s,m)
}
\]

Stage 1 结束后，模型应满足：

\[
(s,m_1)\rightarrow\tau_1
\]

\[
(s,m_2)\rightarrow\tau_2
\]

即同一个 state + 不同 decision 可以生成不同轨迹。

---

# 14. Stage 2：High-Level Decision Policy Supervised Initialization

Stage 1 完成以后，再添加：

\[
\boxed{
\pi_D
}
\]

Stage 2 只学习：

\[
\boxed{
s_t\rightarrow m_t^{GT}
}
\]

即：

> 当前 state 下应该选哪个 high-level decision？

---

# 15. Warning Decision Label

对于一个真正 state-aligned 的 warning branching point：

\[
s_0
\]

有：

\[
\{(m_i,\tau_i,S_i)\}_{i=1}^{K}
\]

使用：

\[
\boxed{
m_0^{GT}
=
\arg\max_iS_i
}
\]

作为 decision supervision。

如果一个 warning episode 只有一个共同 branching state，那么它只贡献一个高置信 warning-decision sample。

这是 open-loop 数据条件下严格且合理的做法。

---

# 16. Stage 2 Dataset

Decision dataset：

\[
\boxed{
\mathcal D_D
=
\mathcal D_{normal}
\cup
\mathcal D_{warning}
}
\]

其中：

- \(\mathcal D_{normal}\)：大量 no-warning decision samples；
- \(\mathcal D_{warning}\)：较少但高信息密度的 warning branching samples。

---

# 17. Stage 2 Decision Head

在 shared representation 上增加：

\[
H_D^{lon}
\rightarrow
\pi_{lon}(m^{lon}|s_t)
\]

\[
H_D^{lat}
\rightarrow
\pi_{lat}(m^{lat}|s_t)
\]

组合：

\[
\boxed{
\pi_D(m_t|s_t)
}
\]

---

# 18. Stage 2 Loss

Longitudinal：

\[
\mathcal L_D^{lon}
=
CE(
\pi_{lon},
m_{GT}^{lon}
)
\]

Lateral：

\[
\mathcal L_D^{lat}
=
CE(
\pi_{lat},
m_{GT}^{lat}
)
\]

总 loss：

\[
\boxed{
\mathcal L_D
=
\mathcal L_D^{lon}
+
\lambda_{lat}\mathcal L_D^{lat}
}
\]

---

# 19. Stage 2 参数冻结策略

第一版推荐：

\[
\boxed{
Freeze\ \pi_T
}
\]

并优先冻结：

\[
\boxed{
F_\theta
}
\]

只训练：

- decision query；
- decision head；
- decision-specific projection。

即：

\[
\boxed{
\min_{\theta_D}\mathcal L_D
}
\]

如果 decision performance 不足，再用较小 learning rate 解冻：

- top transformer layers；
- decision-specific LoRA；
- multimodal projection。

保持：

\[
\eta_{backbone}
\ll
\eta_D
\]

---

# 20. Stage 2 是否需要 Trajectory Rollout

如果训练信号只有 decision CE：

\[
\mathcal L_D
\]

那么数学上不需要调用 \(\pi_T\)。

也可以为了模拟真实 inference 做：

\[
\hat m_t
=
\arg\max\pi_D(m|s_t)
\]

\[
\hat\tau_t
=
\pi_T(s_t,\hat m_t)
\]

但这一 trajectory 只用于：

- monitoring；
- decision-trajectory consistency；
- end-to-end validation。

不参与 Stage 2 backward。

---

# 21. IL 结束后的推理

Stage 1 得到：

\[
\pi_T^{IL}
\]

Stage 2 得到：

\[
\pi_D^{IL}
\]

推理：

\[
s_t
\]

\[
\downarrow
\]

\[
\hat m_t
=
\arg\max_m\pi_D^{IL}(m|s_t)
\]

\[
\downarrow
\]

\[
\hat\tau_t
=
\pi_T^{IL}(s_t,\hat m_t)
\]

\[
\downarrow
\]

PID / low-level controller

\[
\downarrow
\]

CARLA。

---

# 22. IL 的本质

Stage 1：

\[
\boxed{
\text{Learn how to execute}
}
\]

Stage 2：

\[
\boxed{
\text{Learn what to choose}
}
\]

两者都还是 supervised / imitation learning：

- Stage 1 用 trajectory GT；
- Stage 2 用 decision GT。

真正 reward-based optimization 从 Stage 3 开始。


# 23. Stage 3：Pseudo-Simulation-based Reinforcement Learning

Stage 3 中模型结构不变：

\[
s_t
\rightarrow
\pi_D
\rightarrow
m_t
\rightarrow
\pi_T
\rightarrow
\tau_t
\]

变化的是训练信号：

\[
\boxed{
GT\rightarrow Reward
}
\]

第一版 RL：

\[
\boxed{
Freeze\ \pi_T,\ optimize\ \pi_D
}
\]

---

# 24. 为什么 RL 只优化 \(\pi_D\)

原因：

1. \(\pi_T\) 已经通过 Stage 1 学会 action execution；
2. 研究问题本身是 high-level decision bias；
3. \(\pi_D\) 是低维离散 categorical policy；
4. 高维 continuous trajectory RL 优化困难；
5. 与 PerlAD 的“IL continuous behavior + RL low-dimensional action”设计哲学一致。

RL action 定义为：

\[
\boxed{
a_t=m_t
}
\]

而不是 10 个 continuous waypoints。

---

# 25. 同一 State 下采样多个 Decision

对于：

\[
s_t
\]

从 policy 中采样：

\[
\boxed{
m_t^1,\dots,m_t^G
\sim
\pi_D(m|s_t)
}
\]

其中：

- \(G\)：group size，例如 8 或更多。

对第 \(i\) 个 decision：

\[
\boxed{
\tau_t^i
=
\pi_T(s_t,m_t^i)
}
\]

\(\pi_T\) 冻结，只负责把 decision 变成 trajectory。

---

# 26. Pseudo-Simulation Stage 1

对每条：

\[
\tau_t^i
\]

计算：

\[
r_{1,i}
\]

reward 可以由：

- collision；
- drivable-area compliance；
- progress；
- TTC；
- safety margin；
- comfort；
- lane legality；
- unnecessary intervention

构成。

整个 \(H\)-step trajectory 被看作 high-level action 的 physical realization。

---

# 27. Pseudo-Simulation Stage 2

Stage 1 trajectory \(i\) 的 endpoint：

\[
x_{end}^{i}
\]

包含：

- position；
- heading；
- speed；
- scene time。

然后从 observation bank 中检索未来 observation：

\[
s_{t+H}^{i,j}
\]

其中 \(j\) 表示多个候选 Stage-2 state。

不能只用位置距离。

建议：

\[
\boxed{
D_{i,j}
=
\lambda_pD_{pos}
+
\lambda_hD_{heading}
+
\lambda_vD_{speed}
+
\lambda_tD_{time}
}
\]

可进一步加入：

- acceleration；
- lane ID；
- route progress；
- motion history。

---

# 28. Stage-2 State 权重

定义：

\[
w_{i,j}
=
\frac{
\exp(-D_{i,j}/\sigma)
}{
\sum_k\exp(-D_{i,k}/\sigma)
}
\]

其中：

- \(w_{i,j}\)：Stage-2 observation matching weight；
- \(\sigma\)：匹配温度。

满足：

\[
\sum_jw_{i,j}=1
\]

---

# 29. Stage 2 再次调用 Policy

对于每个：

\[
s_{t+H}^{i,j}
\]

再次：

\[
m_2^{i,j}
\sim
\pi_D(m|s_{t+H}^{i,j})
\]

再：

\[
\tau_2^{i,j}
=
\pi_T(
s_{t+H}^{i,j},
m_2^{i,j}
)
\]

计算：

\[
r_{2,i,j}
\]

---

# 30. 两阶段总 Reward

对 Stage-1 candidate \(i\)：

\[
\boxed{
R_i^{PS}
=
r_{1,i}
+
\gamma^H
\sum_j
w_{i,j}r_{2,i,j}
}
\]

其中：

- \(R_i^{PS}\)：candidate \(i\) 的总 pseudo-simulation reward；
- \(\gamma\)：discount factor；
- \(H\)：Stage-1 trajectory horizon。

两阶段设计的价值在于识别：

\[
\text{Immediate Safety}
\neq
\text{Long-term Quality}
\]

例如 HardBrake 的 Stage-1 reward 可能很高，但 Stage-2 recovery/progress 很差；Offset 的即时 reward 略低，但后续状态明显更好。

---

# 31. Reward 设计原则

不奖励：

\[
Brake
\]

本身。

而奖励 outcome：

\[
\boxed{
r=
w_sR_{safety}
+
w_pR_{progress}
+
w_mR_{margin}
+
w_cR_{comfort}
+
w_rR_{recovery}
+
w_qR_{risk}
-
w_uP_{unnecessary}
}
\]

其中：

- \(R_{safety}\)：collision/off-road/violation；
- \(R_{progress}\)：route progress；
- \(R_{margin}\)：安全距离；
- \(R_{comfort}\)：acceleration / jerk / lateral acceleration；
- \(R_{recovery}\)：hazard 后恢复；
- \(R_{risk}\)：risk reduction / TTC；
- \(P_{unnecessary}\)：false/irrelevant warning 下不必要干预。

核心原则：

\[
\boxed{
Reward\ risk\ reduction,\ not\ braking
}
\]

---

# 32. Group-Relative Advantage

同一个 state 得到：

\[
R_1^{PS},\dots,R_G^{PS}
\]

均值：

\[
\mu_R
=
\frac1G\sum_iR_i^{PS}
\]

标准差：

\[
\sigma_R
=
\sqrt{
\frac1G
\sum_i
(R_i^{PS}-\mu_R)^2
}
\]

定义：

\[
\boxed{
A_i
=
\frac{
R_i^{PS}-\mu_R
}{
\sigma_R+\epsilon
}
}
\]

其中：

- \(A_i>0\)：比同组平均 candidate 更好；
- \(A_i<0\)：比同组平均 candidate 更差。

---

# 33. RL Loss

采用 REINFORCE / group-relative policy gradient：

\[
\boxed{
\mathcal L_{RL}
=
-
\frac1G
\sum_{i=1}^{G}
A_i
\log\pi_D(m_i|s_t)
-
\lambda_H\mathcal H(\pi_D)
}
\]

其中：

- \(\mathcal H(\pi_D)\)：policy entropy；
- \(\lambda_H\)：exploration weight。

Factorized decision：

\[
\boxed{
\log\pi_D(m_t|s_t)
=
\log\pi_{lon}(m_t^{lon}|s_t)
+
\log\pi_{lat}(m_t^{lat}|s_t)
}
\]

---

# 34. RL 是否还使用 GT

RL 主训练信号不再使用：

\[
m^{GT}
\]

或：

\[
\tau^{GT}
\]

而是 reward。

但可增加 weak regularization：

\[
\mathcal L
=
\mathcal L_{RL}
+
\beta
D_{KL}
(
\pi_D^{RL}
||
\pi_D^{IL}
)
\]

用于防止 policy 突然漂移。

---

# 35. PerlAD 给我们的启发

PerlAD 的重要 precedent：

\[
\boxed{
IL\ continuous\ execution
+
RL\ low-dimensional\ structured\ action
}
\]

它并不直接在高维 trajectory space 做 RL，而将可结构化的低维 action 留给 RL。

这支持我们：

\[
\boxed{
Stage1:\ IL\ on\ \pi_T
}
\]

\[
\boxed{
Stage3:\ RL\ on\ \pi_D
}
\]

我们的扩展是把 low-dimensional action 从 target speed 提升为：

\[
\boxed{
language-conditioned\ cooperative\ high-level\ decision
}
\]

---

# 36. Pseudo-Simulation 给我们的启发

Pseudo-Simulation 的核心范式：

\[
Stage1\ trajectory
\rightarrow
endpoint
\rightarrow
Stage2\ observation
\]

原本主要用于 evaluation。

我们借用该两阶段结构，将它转化成：

\[
\boxed{
Reward-based\ policy\ optimization
}
\]

---

# 37. 当前 Pseudo-Simulation 的局限

Pseudo-simulation 不是等价真实闭环。

主要 limitation：

- surrounding agents 未必对 ego action 做真实反应；
- retrieved state 只是 offline-data-supported approximation；
- observation-bank coverage 有限；
- strongly interactive scenarios 误差更明显。

因此：

\[
\boxed{
PseudoSim
\neq
True\ ClosedLoop
}
\]

最终必须使用 CARLA 做真实 closed-loop evaluation。

---

# 38. Version 2：Reactive World Model

未来可以借鉴 PerlAD：

\[
\boxed{
P_{agents}^{future}
=
f(s_t,\tau_{ego})
}
\]

让 surrounding-agent future 显式 condition ego future trajectory。

特别适合：

- cut-in；
- lane change；
- intersection conflict。

但第一版不强制加入。

---

# 39. Warning Language 设计

第一版不使用显式 hazard parser。

直接：

\[
Image+Warning
\rightarrow
VLM
\rightarrow
Decision
\]

利用 pretrained VLM/LLM 的语言理解能力。

Prompt 包含：

- current speed；
- route / navigation；
- cooperative warning；
- image。

例如：

```text
Current speed: 8.2 m/s.
Command: Go straight.
Cooperative warning:
A pedestrian may emerge from behind the parked vehicle on the right.
Predict the future driving trajectory.
```

预测 decision 不重新写回 textual prompt。

Decision 使用内部 embedding condition trajectory。

---

# 40. 为什么不把 Decision 再写回 Prompt

如果：

\[
s_t\rightarrow DecisionText
\]

然后重新：

\[
s_t+DecisionText
\rightarrow VLM
\]

会引入：

- 第二次 VLM inference；
- 更高延迟；
- 更复杂 RL pipeline；
- 不必要的 autoregressive language generation。

因此：

\[
\boxed{
Decision\ is\ an\ internal\ action\ condition
}
\]

---

# 41. Warning Generalization Evaluation

建议包括：

### Paraphrase consistency

同一语义不同 wording：

\[
\pi_D(s,W_1)\approx\pi_D(s,W_2)
\]

### Semantic sensitivity

left/right 等核心语义改变时，decision 应相应改变。

### Unseen composition

训练见过 pedestrian-right、cyclist-left，测试 cyclist-right。

### Task-unseen hazard

某 hazard 在 cooperative fine-tuning 中未作为 decision task 出现。

### False / irrelevant warning

理想：

\[
\pi_D(s,W_{false})
\approx
\pi_D(s,\varnothing)
\]

---

# 42. 防止 Warning Shortcut 的原则

训练数据应尽可能满足：

\[
\boxed{
Same\ Warning+Different\ Scene
\rightarrow
Different\ Decision
}
\]

以及：

\[
\boxed{
Same\ Scene+Different\ Warning
\rightarrow
Different\ Decision
}
\]

目标是迫使模型联合使用：

\[
Image+Warning+Ego+Route
\]

而不是仅依赖 warning 文本。

---

# 43. 最终 SimLingo 结构

```text
Image + Warning + Route + Ego State
                |
                v
        Shared SimLingo VLM
                |
        ---------------------
        |                   |
        v                   v
Decision Feature        Driving Feature
        |                   |
        v                   |
   Decision Heads            |
        |                   |
        v                   |
High-level Decision          |
        |                   |
        v                   v
       Decision Embedding Conditioning
                |
        ------------------
        |                |
        v                v
     Path Head        Speed Head
        |                |
        v                v
      Path         Speed Waypoints
        \                /
         \              /
              Trajectory
                  |
                  v
                 PID
                  |
                  v
                CARLA
```

---

# 44. 三阶段训练总结

| 阶段 | 输入 | 主要监督 | 优化对象 | 回答的问题 |
|---|---|---|---|---|
| Stage 1 | \(s,m^{exec}\) | trajectory GT | \(\pi_T\) | “怎么执行？” |
| Stage 2 | \(s\) | decision GT | \(\pi_D\) | “选什么？” |
| Stage 3 | \(s\) | reward | \(\pi_D\) | “哪个结果最好？” |

核心递进：

\[
\boxed{
Trajectory\ GT
\rightarrow
Decision\ GT
\rightarrow
Reward
}
\]

---

# 45. 完整训练流程

```text
                DATASET EXPANSION
                       |
        --------------------------------
        |                              |
        v                              v
No-warning decision labels      Warning candidates
                               m1, m2, ..., mK
                               τ1, τ2, ..., τK
                               S1, S2, ..., SK
        |                              |
        -------------------------------
                       |
                       v

       STAGE 1: EXECUTION LEARNING
       (s, decision) -> trajectory
                       |
                       v
               trained π_T
                       |
                       v

       STAGE 2: DECISION LEARNING
             s -> decision
                       |
                       v
               trained π_D
                       |
                       v

       STAGE 3: PSEUDO-SIM RL
                       |
             sample decisions
                       |
                       v
             frozen π_T -> τ
                       |
                       v
             Pseudo-Simulation
                       |
                       v
                    Reward
                       |
                       v
           policy-gradient update
                       |
                       v
             improved π_D
```

---

# 46. Evaluation

## Decision

- \(Acc_{lon}\)
- \(Acc_{lat}\)
- braking rate
- offset rate
- lane-change rate
- maintain rate

## Decision-Trajectory Consistency

例如：

\[
Decision=OffsetLeft
\]

trajectory 是否真的左偏。

## Closed-loop

使用 LANTERN / CARLA 原有：

- collision；
- route completion；
- progress；
- TTC；
- safety margin；
- recovery；
- false stopping。

## Cooperative Utility

\[
CIU
=
R_W-R_{NoW}
\]

## Unnecessary Intervention Rate

\[
UIR
=
\frac{
N_{unnecessary}
}{
N_{irrelevant-warning}
}
\]

## Warning Generalization Gap

\[
Gap
=
Score_{seen}
-
Score_{unseen}
\]

---

# 47. 关键 Ablations

1. Original Coop-SimLingo
2. + decision-conditioned \(\pi_T\)
3. + Stage-2 \(\pi_D\)
4. + Pseudo-Sim RL
5. decision conditioning on/off
6. no-warning decision supervision on/off
7. one-stage reward vs two-stage pseudo-simulation reward
8. longitudinal-only decision vs longitudinal+lateral
9. Top-1 decision GT vs soft preference
10. seen / paraphrase / unseen composition / task-unseen warning

---

# 48. 推荐实施顺序

1. 固定 decision taxonomy。
2. 给所有现有 branch trajectory 添加 \(m^{exec}\)。
3. Stage 1 训练 decision-conditioned \(\pi_T\)。
4. 验证同 state + 不同 decision 能生成明显不同 trajectory。
5. 为 no-warning 数据自动提取 decision labels。
6. 为 warning branching point 根据 \(S_i\) 得到 \(m^{GT}\)。
7. 添加 \(\pi_D\)，Stage 2 只训练 decision selection。
8. 验证 decision accuracy / diversity / consistency。
9. 建立 Pseudo-Simulation Stage-1 reward。
10. 建立 Stage-2 observation retrieval。
11. 做 group-relative policy-gradient RL。
12. 最终用 CARLA 做真实 closed-loop evaluation。

---

# 49. 最终研究逻辑

\[
\boxed{
\text{Expert trajectories teach execution}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{Counterfactual scores teach initial decision preference}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{Pseudo-simulation rewards refine decision policy}
}
\]

也就是：

\[
\boxed{
Execution\ Learning
\rightarrow
Decision\ Learning
\rightarrow
Outcome\ Optimization
}
\]

---

# 50. 一句话方法总结

> We decouple cooperative driving into decision selection and trajectory execution. We first train a decision-conditioned trajectory executor from diverse counterfactual trajectories, then supervise a high-level decision policy using reliable decision labels, and finally optimize the decision policy with pseudo-simulation-based reinforcement learning to move beyond conservative rule-based imitation.

---

# 51. 最终公式总览

## Stage 1

\[
\boxed{
\hat\tau_t
=
\pi_T(s_t,m_t^{exec})
}
\]

\[
\boxed{
\min\mathcal L_T(
\hat\tau_t,
\tau_t^{GT}
)
}
\]

## Stage 2

\[
\boxed{
\hat m_t
=
\pi_D(s_t)
}
\]

\[
\boxed{
\min
CE(
\hat m_t,
m_t^{GT}
)
}
\]

## Stage 3

\[
\boxed{
m_i
\sim
\pi_D(m|s_t)
}
\]

\[
\boxed{
\tau_i
=
\pi_T(s_t,m_i)
}
\]

\[
\boxed{
R_i^{PS}
=
r_{1,i}
+
\gamma^H
\sum_j
w_{i,j}r_{2,i,j}
}
\]

\[
\boxed{
A_i
=
\frac{
R_i^{PS}-\mu_R
}{
\sigma_R+\epsilon
}
}
\]

\[
\boxed{
\mathcal L_{RL}
=
-
\frac1G
\sum_i
A_i
\log\pi_D(m_i|s_t)
-
\lambda_H\mathcal H(\pi_D)
}
\]

---

# 52. 最终设计原则

1. Warning 是风险信息，不是动作命令。
2. Execution 与 Decision Selection 分阶段训练。
3. Open-loop branch 后续 states 不做错误的跨 branch state-group 对齐。
4. Stage 1 充分利用所有 branch 的 sliding-window trajectory 数据。
5. Stage 2 只使用可靠 decision labels。
6. Stage 3 不再模仿 GT，而是最大化 reward。
7. 第一版 RL 冻结 \(\pi_T\)，只优化 \(\pi_D\)。
8. Pseudo-Simulation 只是闭环近似，最终必须用 CARLA 验证。
9. 模型目标不是超过 expert trajectory similarity，而是超过 expert outcome quality。
10. 最终目标是让 cooperative information 只在它确实能带来更好行为时改变 decision。
