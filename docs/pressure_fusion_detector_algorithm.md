# Pressure Fusion Detector 具体算法说明

本文档只解释 `PressureFusionBlowoutDetector` 当前代码实际执行的算法。接口治理、
验证准入和功能安全要求见
[`pressure_fusion_detector_algorithm_spec.md`](pressure_fusion_detector_algorithm_spec.md)。

## 1. 算法解决的问题

车辆有四路校正轮速，但只有一组对角轮具备逐轮胎压爆胎判定：

```text
布局 A：FL、RR 有胎压；FR、RL 由轮速检测
布局 B：FR、RL 有胎压；FL、RR 由轮速检测
```

算法采用两条判定路径：

```text
胎压路径：胎压 True 上升沿 → 对应轮立即锁存报警

轮速路径：逐轮相对增速
        + 对角共同增速
        + 约 0.7 s 高位持续
        + 参考健康和工况门控
        → 对应轮锁存报警
```

它是决策级融合，不计算“胎压权重 × 胎压值 + 轮速权重 × 轮速值”。胎压对角既
负责直接报警，也充当另一对角轮速检测的健康参考。

## 2. 轮位、输入与输出

### 2.1 固定轮位顺序

```text
index 0 = FL = 左前
index 1 = FR = 右前
index 2 = RL = 左后
index 3 = RR = 右后
```

合法对角固定为：

```python
DIAGONALS = ((0, 3), (1, 2))  # (FL, RR), (FR, RL)
```

### 2.2 单帧输入

```python
PressureFusionFrame.from_sequences(
    t_sec,
    [fl_speed, fr_speed, rl_speed, rr_speed],
    [fl_pressure, fr_pressure, rl_pressure, rr_pressure],
)
```

胎压输入使用三态：

| 输入 | 精确语义 |
| --- | --- |
| `True` | 胎压子系统确认该轮爆胎 |
| `False` | 胎压子系统确认该轮健康 |
| `None` | 该轮没有信号或当前信号不可用 |

`None` 不等于正常。轮速参考必须是两个胎压轮都明确为 `False`。

### 2.3 关键输出

| 输出 | 用途 |
| --- | --- |
| `new_blowouts[i]` | 第 `i` 轮是否在本帧首次报警 |
| `blowout_alarms[i]` | 第 `i` 轮是否处于锁存报警状态 |
| `alarm_sources[i]` | `none`、`pressure` 或 `wheel_speed_confirmed` |
| `candidates[i]` | 第 `i` 轮是否仍在轮速候选观察期 |
| `speed_detection_available` | 本帧是否允许作新的轮速强判定 |
| `estimated_onset_times_s[i]` | 经滤波延迟回推后的估算起点 |

正式报警使用 `blowout_alarms`。`candidates` 只是内部疑似状态。

## 3. 一帧数据的精确执行顺序

`push(frame)` 的顺序非常重要。伪代码与当前实现一致：

```python
def push(frame):
    # 1. 接口校验
    assert timestamp_is_finite_and_strictly_increasing(frame.t_sec)
    assert all_wheel_speeds_are_finite(frame.wheels)

    frame_index += 1

    # 2. 从非 None 胎压位置识别并锁定传感器对角
    discover_pressure_diagonal(frame.pressure_blowouts)

    # 3. 胎压分支先执行；True 上升沿可在当前帧直接锁存
    new_blowouts = [False, False, False, False]
    apply_pressure_alarms(frame, new_blowouts)

    # 4. 轮速只使用幅值
    speed = [abs(value) for value in frame.wheels]

    # 5. 判断轮速数值和胎压参考是否可用
    speed_valid = min(speed) > 1e-9 and mean(speed) >= min_avg_speed
    reference_healthy = all(
        pressure[i] is False and alarm[i] is False
        for i in sensor_diagonal
    )
    speed_available = speed_valid and reference_healthy

    if not speed_available:
        cancel_all_speed_candidates()
        clear_all_edge_windows()
        if wheel_speed_has_been_invalid_for_50_frames():
            clear_smoothing_and_baseline_history()
        return result

    # 6. 计算共同量、逐轮量和对角量
    x = [log(value) for value in speed]
    common_log_speed = mean(x)
    sensor_mean = mean(x[i] for i in sensor_diagonal)
    diagonal_raw = (
        sum(x[i] for i in speed_diagonal)
        - sum(x[i] for i in sensor_diagonal)
    )

    # 7. 对角特征：平滑 → 减基线 → 边沿
    diagonal_smoothed = robust_smooth(diagonal_raw)
    if diagonal_baseline_is_ready():
        diagonal_gain = diagonal_smoothed - diagonal_baseline
        diagonal_edge = causal_edge(diagonal_gain)

    # 8. 两个轮速目标轮分别计算逐轮特征
    for wheel in speed_diagonal:
        individual_raw = x[wheel] - sensor_mean
        individual_smoothed = robust_smooth(individual_raw)
        if individual_and_diagonal_baselines_are_ready():
            gain[wheel] = individual_smoothed - individual_baseline[wheel]
            edge[wheel] = causal_edge(gain[wheel])

    # 9. 两个目标轮分别推进候选状态
    for wheel in speed_diagonal:
        advance_candidate(
            wheel=wheel,
            gain=gain[wheel],
            edge=edge[wheel],
            diagonal_gain=diagonal_gain,
            diagonal_edge=diagonal_edge,
            mate_gain=gain[other_speed_wheel],
            common_log_speed=common_log_speed,
        )

    # 10. 仅在没有候选、没有轮速报警时更新正常基线
    update_normal_baseline_if_safe()

    # 11. 组装当前帧结果
    return result
```

因为胎压路径先于轮速可用性判断执行，所以参考胎压在本帧变成 `True` 时，会先锁存
胎压报警，然后同一帧立即关闭轮速路径并撤销已有候选。

## 4. 胎压布局发现

每帧取胎压数组中所有非 `None` 的索引：

```python
available = tuple(i for i, value in enumerate(pressure) if value is not None)
```

处理规则：

```text
available == ()       → 暂不发现布局，轮速路径不可用
available == (0, 3)   → 胎压对角 FL+RR，轮速对角 FR+RL
available == (1, 2)   → 胎压对角 FR+RL，轮速对角 FL+RR
其他位置组合          → ValueError
```

布局发现后保持不变，直到 `reset()`。例如先输入 `(0, 3)`，后续改成 `(1, 2)` 会直接
报错，不会在运行中静默切换参考组。

## 5. 胎压直接报警算法

对每个轮保存上一帧是否为 `True`：

```python
active = pressure[i] is True
rising = active and not previous_pressure[i]
```

当 `rising=True`：

```python
new_blowouts[i] = not blowout_alarms[i]
blowout_alarms[i] = True
alarm_sources[i] = "pressure"
estimated_onset_index[i] = current_frame_index
estimated_onset_time_s[i] = current_time
candidate[i] = None
```

一个帧宽的 `True` 脉冲也足以锁存报警。后续 `False` 只更新边沿记忆，不清除报警。

## 6. 轮速可用性门控

### 6.1 基本轮速有效性

四轮先取绝对值：

```math
v_i=|wheel_i|
```

然后判断：

```math
speed\_valid=
[\min_i(v_i)>10^{-9}]
\land
[mean(v_i)\ge 20]
```

`20` 的单位与输入轮速单位一致，当前代码不进行单位换算。

### 6.2 参考健康条件

设胎压对角为 $S$：

```math
reference\_healthy=
\bigwedge_{i\in S}
([pressure_i=False]\land[alarm_i=False])
```

真值示例：

| 两个参考胎压 | 参考轮历史报警 | 轮速路径 |
| --- | --- | --- |
| `False, False` | 都没有 | 可用 |
| `False, None` | 任意 | 不可用 |
| `False, True` | 任意 | 不可用 |
| `False, False` | 任一已锁存 | 不可用 |

因此胎压 `True` 即使下一帧恢复为 `False`，由于报警已经锁存，参考仍然不可用，必须
显式 `reset()` 才能恢复。

## 7. 轮速特征

假设本车胎压对角是 FR+RL，轮速对角是 FL+RR。定义：

```math
x_{FL}=\ln(v_{FL}),\quad x_{FR}=\ln(v_{FR}),\quad
x_{RL}=\ln(v_{RL}),\quad x_{RR}=\ln(v_{RR})
```

胎压参考均值：

```math
\bar{x}_S=\frac{x_{FR}+x_{RL}}{2}
```

### 7.1 逐轮特征

```math
r_{FL}=x_{FL}-\bar{x}_S
```

```math
r_{RR}=x_{RR}-\bar{x}_S
```

`r_FL` 和 `r_RR` 分开计算，用于定位具体异常轮。

### 7.2 对角特征

```math
q=x_{FL}+x_{RR}-x_{FR}-x_{RL}
```

它也等于：

```math
q=r_{FL}+r_{RR}
```

逐轮特征过门限但对角特征不过门限时，不建立候选。

### 7.3 为什么取对数

对数差把乘法比例变成加法：

```math
\ln(v_{target})-\ln(v_{reference})
=\ln\left(\frac{v_{target}}{v_{reference}}\right)
```

小变化下：

```math
\ln(1+p)\approx p
```

因此 `0.0058` 可近似解释为 `0.58%` 相对变化，同时不会依赖当前轮速的绝对数值。

### 7.4 对角残差如何抵消常见运动

把每个轮的对数轮速分解成共同项 $c$、左右项 $l/r$、前后项 $f/b$ 和故障项
$e_i$。两条对角都各含一个左轮、右轮、前轮和后轮：

```text
FL + RR：左 + 右，前 + 后
FR + RL：右 + 左，前 + 后
```

所以在一阶对称近似下：

```math
(x_{FL}+x_{RR})-(x_{FR}+x_{RL})
\approx e_{FL}+e_{RR}-e_{FR}-e_{RL}
```

共同加减速、左右轮差和前后轴差近似抵消。极限转向、打滑和控制系统介入不一定
满足该近似，因此后面还需要共同速度瞬变门控。

## 8. 两级平滑

每个逐轮特征和对角特征都有独立历史。对新输入值 $z_t$：

第一层取最近 5 帧中位数：

```math
m_t=median(z_{t-4},\ldots,z_t)
```

第二层对最近 5 个中位数取均值：

```math
\widetilde{z}_t=mean(m_{t-4},\ldots,m_t)
```

代码在窗口未满时使用已有样本，但报警仍会被基线预热条件阻止。中值层抑制孤立
毛刺，均值层减小量化噪声。

## 9. 正常基线与增益

逐轮平滑特征和对角平滑特征各维护一个最长 500 帧的队列。至少积累 200 帧后，
基线才有效：

```math
b_i=median(B_i)
```

```math
b_D=median(B_D)
```

基线中位数每 10 帧重新计算一次。最终用于检测的增益为：

```math
g_i=\widetilde{r_i}-b_i
```

```math
g_D=\widetilde{q}-b_D
```

基线更新条件是：

```python
not any(candidates) and not any(speed_diagonal_alarms)
```

候选期间冻结基线；任一轮速目标轮报警后，两个目标轮和对角基线都停止更新。候选
确认失败并被清除后，当前帧可以重新进入正常基线。

## 10. 因果边沿

边沿窗口包含最近 12 个增益，前后各 6 帧：

```math
edge_t=
mean(g_{t-5},\ldots,g_t)
-mean(g_{t-11},\ldots,g_{t-6})
```

100 Hz 下，它比较最近 60 ms 和此前 60 ms 的平均水平。只有收满 12 个有效增益后
才产生边沿，之前输出 `NaN`。

逐轮增益产生 `edge_i`，对角增益产生 `edge_D`。

## 11. 逐轮候选状态机

轮速对角的两个轮各有一个独立候选对象：

```text
READY
  │ edge_i >= 0.0058 且 edge_D >= 0.0058
  ▼
CANDIDATE
  ├─ 当前值跌破下限 ──────────────→ READY
  ├─ 峰值超过上限 ────────────────→ READY
  ├─ 70 帧后确认条件失败 ─────────→ READY
  └─ 70 帧后全部确认条件通过 ─────→ LATCHED
```

### 11.1 建立条件

对目标轮 $i$：

```math
edge_i\ge0.0058 \land edge_D\ge0.0058
```

建立候选时保存当前逐轮增益、对角增益、同组另一轮增益和共同对数速度。

### 11.2 起点回推

候选建立帧不是事件起点。算法回推：

```math
delay=(5-1)+6=10\ frames=0.1\ s
```

```python
estimated_onset_index = current_index - 10
estimated_onset_time = current_time - 0.1
```

索引最小截断到 0；时间不作 0 下限截断。

### 11.3 候选期间保存的量

每帧追加：

```text
individual_values  = 目标轮增益序列
diagonal_values    = 对角增益序列
mate_values        = 同组另一轮增益序列
common_log_speeds  = 四轮对数轮速均值序列
max_individual     = 目标轮历史最大增益
max_diagonal       = 对角历史最大增益
```

## 12. 候选提前否决

候选期间逐帧检查：

```python
rejected = (
    gain_i < -0.0040
    or diagonal_gain < -0.0040
    or max_individual > 0.0250
    or max_diagonal > 0.0450
)
```

任一条件成立立即删除该轮候选，本帧不会报警。下一帧重新按边沿数值判断是否建立
候选；实现判断的是 `edge >= threshold`，没有额外保存“必须再次穿越阈值”的记忆。

注意上限使用严格大于 `>`，下限使用严格小于 `<`；正好等于阈值时不会被提前否决。

## 13. 70 帧确认判定

候选样本数达到 70 时，只执行一次最终判断。设：

```text
A = 最后 40 帧目标轮增益
D = 最后 40 帧对角增益
M = 最后 40 帧同组另一轮增益
C = 候选全部 70 帧共同对数速度
```

确认公式：

```python
confirmed = (
    max_individual >= 0.0070
    and max_diagonal >= 0.0070
    and median(A) >= 0.0055
    and median(D) >= 0.0055
    and fraction(A >= 0.0035) >= 0.75
    and fraction(D >= 0.0035) >= 0.75
    and median(M) >= -0.0035
    and max(C) - min(C) <= 0.050
)
```

各条件含义：

| 条件 | 排除的典型情况 |
| --- | --- |
| 逐轮峰值 | 目标轮变化不足 |
| 对角峰值 | 只有局部逐轮噪声 |
| 两个尾窗中位数 | 上升后没有保持高位 |
| 两个 75% 高位占比 | 短脉冲或高位不稳定 |
| 同组另一轮中位数 | 另一个目标轮明显反向变化 |
| 共同速度极差 | 候选期间整车剧烈加减速 |

共同速度量是：

```math
c_t=\frac{1}{4}\sum_i\ln(v_i)
=\ln\left(\sqrt[4]{\prod_i v_i}\right)
```

`max(C)-min(C) <= 0.050` 等价于候选期四轮几何平均速度的最大/最小比不超过
$e^{0.05}\approx1.0513$，即约 5.13%。

确认通过时：

```python
blowout_alarms[i] = True
alarm_sources[i] = "wheel_speed_confirmed"
new_blowouts[i] = True
```

确认通过或失败后都会删除候选。失败后不会延长当前观察窗；下一帧重新按边沿门限
判断，如果边沿仍然满足条件，可以立即建立一个新候选。

## 14. 数值示例：RR 单轮爆胎

假设胎压参考对角是 FR+RL，正常阶段四轮都是 50：

```text
FL = 50.00
FR = 50.00  pressure=False
RL = 50.00  pressure=False
RR = 50.00
```

正常基线约为 0。RR 事件后稳定变为 50.55，约增加 1.1%：

```text
FL = 50.00
FR = 50.00
RL = 50.00
RR = 50.55
```

忽略滤波过渡时：

```math
g_{RR}\approx\ln(50.55/50)=0.01094
```

```math
g_{FL}\approx\ln(50/50)=0
```

```math
g_D\approx\ln(50\times50.55/(50\times50))=0.01094
```

结果：

```text
RR 逐轮证据       0.01094 > 0.0070
对角证据          0.01094 > 0.0070
FL mate 证据      0       > -0.0035
持续保持 70 帧    可以确认 RR
FL 逐轮证据       不满足，不会确认 FL
```

如果 FL 和 RR 同时各增加 1%，两个逐轮增益都约为 `0.00995`，对角增益约为
`0.01990`，两个独立候选都可能在同一帧确认。

## 15. 时延拆解

默认 100 Hz 下：

| 阶段 | 帧数 | 作用 |
| --- | ---: | --- |
| 基线预热 | 最少 200 | 建立正常中位数，约 2.0 s |
| 平滑 | 两级各 5 | 抑制毛刺并带来群延迟 |
| 边沿窗口 | 12 | 比较前后各 60 ms |
| 候选确认 | 70 | 建立后持续观察约 0.69 s |
| 起点回推 | 10 | 把报告起点向前修正约 0.1 s |

“70 帧确认”包含建立候选时保存的第一个样本，因此从候选建立到第 70 个样本约为
69 个采样间隔，即 0.69 秒。实际事件到报警还包含平滑和边沿建立时间。当前 8 条 RR
实车回放的平均确认延迟为 0.828 秒，最大为 0.85 秒。

## 16. 不可用处理与恢复

### 16.1 每次不可用都执行

```python
candidates = [None, None, None, None]
clear(all_individual_edge_windows)
clear(diagonal_edge_window)
```

这意味着恢复后必须重新积累 12 个有效增益才能再次计算完整边沿。

### 16.2 连续轮速无效 50 帧

只有 `speed_valid=False` 才累计无效帧数。达到 50 帧后额外清除：

```text
逐轮原始和平滑历史
逐轮正常基线
对角原始和平滑历史
对角正常基线
所有基线缓存
```

恢复后需要重新完成至少 200 帧基线预热。

### 16.3 参考不健康但轮速数值有效

此时会清候选和边沿，但无效轮速计数保持为 0，原有平滑与基线历史不会因为持续
时间而被清除。不过参考轮一旦因胎压报警锁存，当前实例不会自动恢复轮速判定。

## 17. 多轮事件的具体行为

| 场景 | 行为 |
| --- | --- |
| 胎压对角单轮爆胎 | 该轮立即报警；整个轮速分支关闭 |
| 胎压对角两轮爆胎 | 两轮分别由各自 `True` 上升沿报警；轮速分支关闭 |
| 轮速对角单轮爆胎 | 对应轮候选持续确认，另一轮不报警 |
| 轮速对角两轮同时爆胎 | 两轮有独立候选，可同时确认 |
| 轮速对角两轮先后爆胎 | 第一轮报警后基线冻结；第二轮仍可建立独立候选 |
| 参考轮先爆、轮速轮后爆 | 参考污染门控生效，后续轮速轮不作强判定 |
| 四轮同幅变化 | 相对特征可能抵消，只能保留胎压可直接观测的轮位 |

轮速对角某一轮报警不会关闭另一轮的检测，因为健康参考来自胎压对角；但它会冻结
共享基线，防止已知异常进入正常模型。

## 18. 报警与复位语义

报警数组只会从 `False` 变为 `True`：

```text
none → pressure
none → wheel_speed_confirmed
```

当前实现没有自动清警逻辑。`reset()` 会同时清除：

- 已识别的胎压/轮速对角；
- 所有滤波和基线历史；
- 所有候选；
- 所有报警、来源和估算起点；
- 胎压上升沿记忆；
- 时间戳和帧索引。

复位后的实例等价于新建实例，必须重新发现布局和预热基线。

## 19. 参数改变时的联动关系

参数不能孤立调整：

| 调整项 | 必须联动检查 |
| --- | --- |
| `sample_rate_hz` | 所有以帧为单位的平滑、边沿、基线、确认和清除窗口 |
| `smooth_window` | 边沿幅度、候选触发时刻和起点回推公式 |
| `edge_half_window` | 边沿噪声、响应时延和起点回推公式 |
| `baseline_*` | 冷启动可用时间、慢漂移跟踪和异常吸收风险 |
| 边沿门限 | 候选数量、漏检率和后续确认负载 |
| 峰值上下限 | 可接受事件幅度包络 |
| 持续性门限 | 报警延迟、短脉冲抑制和缓慢回落容忍度 |
| `max_common_speed_range` | 加减速工况可用性与动态误报风险 |

代码只验证单个参数的基本范围，不验证整组参数是否在物理和统计上自洽。

## 20. 代码对照

| 算法环节 | 实现函数/位置 |
| --- | --- |
| 单帧总流程 | `PressureFusionBlowoutDetector.push()` |
| 胎压布局发现 | `_discover_diagonal()` |
| 胎压直判 | `_apply_pressure()` |
| 参考健康门控 | `_reference_healthy()` |
| 逐轮和对角特征 | `push()` 中 `logs`、`sensor_mean`、`diag_value` |
| 两级平滑 | `_smooth_value()` |
| 正常基线 | `_current_baseline()`、`_current_diagonal_baseline()` |
| 因果边沿 | `_edge()` |
| 候选与确认 | `_advance()` |
| 不可用处理 | `_handle_unavailable()` |
| 输入校验 | `_validate()` |
| 完整复位 | `reset()` |

源代码：`wavelet_shape_blowout_detector/pressure_fusion_detector.py`。

## 21. 最小调用示例

```python
from wavelet_shape_blowout_detector import (
    PressureFusionBlowoutDetector,
    PressureFusionFrame,
)

detector = PressureFusionBlowoutDetector()

# FR、RL 是胎压参考；FL、RR 使用轮速检测。
result = detector.push(
    PressureFusionFrame.from_sequences(
        t_sec,
        [fl_speed, fr_speed, rl_speed, rr_speed],
        [None, fr_blowout, rl_blowout, None],
    )
)

print(result.speed_detection_available)
print(result.candidates)
print(result.new_blowouts)
print(result.blowout_alarms)
print(result.alarm_sources)
```

在线系统应以固定轮位顺序逐帧调用同一个检测器实例，不得为每帧重新创建实例。
