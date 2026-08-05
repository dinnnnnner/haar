# 累计证据爆胎检测器设计草案

## 目标

新增一个与现有硬阈值检测器并行的累计证据检测器。保留现有轮速比、平滑、
正常基线和因果 Haar 特征，但不再要求信号严格满足一次性的
“上升沿 → 下降沿 → 水平平台”。

真实信号可能呈现：

```text
上升 → 小回落 → 再上升 → 再回落 → 创新高 → 高位波动
```

新检测器需要允许多峰和局部回撤，同时保留以下物理时序：

1. 首先出现足够的累计上升证据；
2. 上升之后至少观察到一次有效回撤；
3. 回撤后轮速增益持续处于正常基线之上；
4. 完整形态确认后锁存报警，只能显式复位。

## 实现策略

不要直接覆盖 `detector.py`。建议新增：

```text
wavelet_shape_blowout_detector/
├── detector.py                   # 现有硬阈值算法，保留用于对照
├── features.py                   # 公共轮速特征提取
├── evidence_detector.py          # 新累计证据算法
├── calibrate_evidence.py         # 参数标定工具
└── test_evidence_detector.py     # 新算法回归测试
```

新旧检测器保持相同的 `push(frame)` 和 `reset()` 使用方式，便于在同一输入上
进行 A/B 比较。

## 状态设计

不采用严格且不可回退的 `RISING → FALLING → PLATEAU`。使用更宽松的状态：

```text
WARMING
   ↓
NORMAL
   ↓
CANDIDATE_BUILDING
   ↓
ELEVATED_VERIFY
   ↓
LATCHED_ALARM
```

`CANDIDATE_BUILDING` 内部同时保存上升、回撤和高位持续证据。状态机不需要
为每一个局部波峰判断“当前究竟是上升还是下降”。

建议的逐轮状态：

```python
@dataclass
class WheelEvidenceState:
    phase: str = "warming"

    rise_evidence: float = 0.0
    pullback_evidence: float = 0.0
    persistence_evidence: float = 0.0

    running_low_gain: float = 0.0
    running_peak_gain: float = 0.0
    onset_time_s: float | None = None

    pullback_seen: bool = False
    provisional_alarm: bool = False
    latched_alarm: bool = False

    phase_frames: int = 0
    candidate_frames: int = 0
```

四个轮位分别保存独立状态和证据。

## 公共特征

特征提取层只处理信号，不执行报警判断。每帧至少输出：

```python
@dataclass(frozen=True)
class WheelFeatures:
    t_sec: float
    speed_valid: bool
    gains: tuple[float, float, float, float]
    haar: tuple[float, float, float, float]
    short_slopes: tuple[float, float, float, float]
    noise_scales: tuple[float, float, float, float]
    reference_sources: tuple[str, str, str, str]
```

预处理链路保持：

```text
四轮轮速
→ 数据有效性检查
→ 目标轮/参考轮轮速比
→ 中值滤波与移动平均
→ 正常历史基线
→ 相对正常增益
→ 因果 Haar 与短期趋势
```

紫色人工事件线只用于离线评估和绘图，不能进入在线检测器。

## 证据累计

所有证据使用统一的带衰减累加器：

```python
def accumulate(
    previous: float,
    increment: float,
    *,
    decay: float,
    lower: float = 0.0,
    upper: float = 20.0,
) -> float:
    value = decay * previous + increment
    return min(upper, max(lower, value))
```

建议用 `tanh` 将特征转为连续、有界的软评分：

```python
def signed_score(value: float, scale: float) -> float:
    return math.tanh(value / scale)


def positive_score(value: float, scale: float) -> float:
    return max(0.0, math.tanh(value / scale))
```

原来的阈值可作为软评分尺度，而不是一票否决边界。例如 Haar 为 `0.46%`
时仍然贡献明显的正证据，不会因为没有达到 `0.55%` 而完全失效。

## 累计上升证据

上升证据综合以下信息：

- 正 Haar；
- 短期拟合斜率为正；
- 相对候选起点或滚动低点的净增益；
- 相对运行峰值的回撤作为扣分项。

示意公式：

```python
rise_increment = (
    w_haar * signed_score(haar, haar_scale)
    + w_slope * signed_score(short_slope, slope_scale)
    + w_gain * signed_score(net_rise, gain_scale)
    - w_pullback * positive_score(drawdown, pullback_scale)
)
```

局部下降只扣除一部分上升证据，不清空候选。后续继续上升时，净增益和正趋势
可以继续累积。

上升证据达到快速报警分数后：

```python
state.phase = "candidate_building"
state.provisional_alarm = True
state.running_peak_gain = gain
```

## 运行峰值与回撤记忆

候选期间持续更新运行峰值：

```python
if gain > state.running_peak_gain:
    state.running_peak_gain = gain
    state.pullback_evidence *= new_peak_pullback_decay
```

创新高说明整体上升仍在继续，因此回撤证据适当衰减，但不完全清零。算法保留
“上升后曾经出现过有效回撤”这一历史信息。

回撤证据综合：

- 当前增益相对运行峰值的回撤量；
- 负 Haar；
- 短期负斜率；
- 连续一段时间没有创新高。

```python
drawdown = state.running_peak_gain - gain

pullback_increment = (
    w_drawdown * positive_score(drawdown, drawdown_scale)
    + w_haar * signed_score(-haar, haar_scale)
    + w_slope * signed_score(-short_slope, slope_scale)
)
```

累计回撤证据达到分数后：

```python
state.pullback_seen = True
```

这不会强迫状态机立刻进入不可逆的“下降阶段”。之后继续上涨或创新高仍然
允许，状态保持在候选构建中。

## 高位持续证据

“稳定平台”不定义成水平直线，而定义为持续高于正常基线。允许局部波动和
继续缓慢上升。

在最近窗口内计算：

- 增益中位数；
- 高于正常水平的帧占比；
- 较低分位数是否仍高于基线；
- 信号是否长时间跌回基线；
- 局部波动强度。

```python
window = recent_gains[-plateau_window:]
median_gain = median(window)
above_ratio = sum(gain > elevated_level for gain in window) / len(window)
lower_quantile = percentile(window, 20)

persistence_increment = (
    w_median * level_score(median_gain)
    + w_ratio * above_ratio_score(above_ratio)
    + w_lower * level_score(lower_quantile)
    - w_variance * variance_penalty(window)
)
```

单个窗口略低于原来的硬门槛只会少加分，不会直接失败；后续持续高位可以补充
证据。

## 完整确认与报警锁存

确认条件保持时间顺序：

```python
if (
    state.rise_evidence >= rise_confirm_score
    and state.pullback_seen
    and state.persistence_evidence >= persistence_confirm_score
):
    state.latched_alarm = True
    state.provisional_alarm = False
    state.phase = "latched_alarm"
```

必须使用独立的 `latched_alarm`，不能用当前总证据是否超过分数来代表锁存。

```python
fast_alarm = state.provisional_alarm or state.latched_alarm
confirmed_alarm = state.latched_alarm
```

进入 `LATCHED_ALARM` 后：

- 后续证据下降不能解除报警；
- 轮速恢复不能解除报警；
- 不再用该轮异常数据更新正常基线；
- 可以继续计算特征和证据用于诊断显示；
- 只能通过 `reset()` 或明确的单轮复位解除。

候选尚未确认时，如果增益长期回到基线、候选超时、数据持续无效或参考来源
改变，可以撤回 `provisional_alarm` 并清理候选证据。候选清理逻辑必须首先检查
`latched_alarm`，不得清除已确认报警。

## 输出建议

新结果应明确区分快速报警和确认报警：

```python
@dataclass(frozen=True)
class EvidenceResult:
    fast_alarms: tuple[bool, bool, bool, bool]
    confirmed_alarms: tuple[bool, bool, bool, bool]
    new_fast_alarms: tuple[bool, bool, bool, bool]
    new_confirmed_alarms: tuple[bool, bool, bool, bool]

    rise_evidence: tuple[float, float, float, float]
    pullback_evidence: tuple[float, float, float, float]
    persistence_evidence: tuple[float, float, float, float]
    phases: tuple[str, str, str, str]
```

为了兼容现有页面，可以映射：

```python
blowout_alarms = fast_alarms
new_blowouts = new_fast_alarms
shape_events = new_confirmed_alarms
```

## 参数标定与验证边界

报警参数只使用前六条原始爆胎数据标定：

```text
E01_event_000
E02_event_000
E03_event_000
E04_event_000
E05_event_000
E06_event_000
```

标定数据用途：

- 事件前正常段：估计 gain、Haar 和趋势噪声，约束预事件误报；
- 事件段：确定上升、回撤、高位持续证据的尺度、权重和确认分数。

验证边界：

- `E07_event_000`、`E08_event_000` 作为盲测，不参与调参；
- 其余增强事件和正常样本只做压力测试；
- 不允许随机拆分增强样本，否则同一真实事件的相关变体会同时出现在标定和验证中；
- 参数选择应按 `E01`～`E06` 来源分组做留一来源检查，避免只记住单条曲线。

标定目标同时考虑：

```text
漏报损失
+ 预事件误报损失
+ 错误轮位损失
+ 报警延迟损失
+ 临时报警反复开关损失
```

## 必要安全硬约束

形态判断改成累计软证据，但以下条件仍保留硬约束：

- 平均轮速太低时不检测；
- 有效轮速少于三个时不检测；
- 输入包含 NaN 或无穷值时不检测；
- 参考来源改变时重建相关候选和特征历史；
- 极端增益或跳变按传感器故障处理；
- 确认报警只能显式复位。

原则是：

```text
数据与物理安全边界使用硬约束
爆胎形态判断使用累计软证据
```

