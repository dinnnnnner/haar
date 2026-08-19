# Quant 四轮轮速爆胎检测算法

## 1. 文档范围

本文描述 `quant_wheel_blowout_detector` 当前实现，覆盖输入约束、逐帧计算、
Hadamard 单轮指纹、在线协方差、风险分、候选与报警状态机、默认参数、接口和
验证结果。代码入口为 [`detector.py`](detector.py)。

当前版本只使用四轮校正轮速，不读取胎压、爆胎标注、纵向加速度或车辆状态。
0818 文件中的爆胎信号只用于离线评价，不会传入检测器。48 齿相位校正属于上游
轮速预处理，不属于 Quant 检测器本身。

## 2. 目标与边界

算法要从共同加减速、转弯、前后轴差和道路冲击中识别某一个车轮持续独立变快的
模式，并输出具体轮位。轮位顺序固定为：

```text
0 = FL，1 = FR，2 = RL，3 = RR
```

当前默认参数按 100 Hz 输入设计。输入低于最低轮速、尚未完成在线模型预热或包含
非法值时，不应使用检测结果作爆胎判定。

## 3. 输入与输出

### 3.1 输入帧

```python
QuantFrame(
    t_sec: float,
    wheels: tuple[float, float, float, float],
)
```

要求：

- `t_sec` 和四轮轮速均为有限数；
- 时间戳严格递增；
- 四轮顺序必须为 FL、FR、RL、RR；
- 当前工程输入为校正后的 `rad/s` 轮速；
- 内部对轮速取绝对值，四轮平均值需至少为 `20 rad/s`，且每轮不能为零。

### 3.2 主要输出

`QuantResult` 每帧返回：

| 字段 | 含义 |
| --- | --- |
| `speed_valid` | 当前轮速是否满足检测条件 |
| `warmed_up` | 在线均值和协方差是否完成预热 |
| `market_factors` | 左右、前后、对角三个 Hadamard 因子 |
| `factor_residuals` | 三因子相对在线均值的持续残差 |
| `factor_edges` | 两个 6 帧均值之差构成的瞬时边沿 |
| `shock_z_scores` | 四轮瞬时冲击匹配 Z 分 |
| `level_z_scores` | 四轮持续偏离匹配 Z 分 |
| `shock_isolation` | 瞬时轮位相对其他轮的领先量 |
| `level_isolation` | 持续轮位相对其他轮的领先量 |
| `physical_levels` | 未按协方差归一化的四轮物理投影 |
| `physical_edges` | 未按协方差归一化的四轮边沿投影 |
| `risk_scores` | 0–100 内部证据分，不是概率 |
| `states` | `warming`、`monitoring`、`candidate` 或 `alarm` |
| `new_blowouts` | 仅在本帧首次确认时为真 |
| `blowout_alarms` | 锁存报警状态 |
| `estimated_onset_times_s` | 按滤波延迟回推的候选起点，不是报警时间 |

## 4. 逐帧信号处理

### 4.1 对数轮速与因果平滑

对每轮绝对轮速取自然对数：

```text
x_i(t) = ln(|wheel_i(t)|)
```

每轮先取最近 5 帧原始对数轮速的中位数，再对最近 5 个中位数结果求平均。
这个两级因果滤波降低单点毛刺影响，但会引入检测延迟。

### 4.2 三个非共同因子

对平滑后的 `FL, FR, RL, RR` 构造：

```text
s = ( FL - FR + RL - RR) / 4    # 左右因子
a = ( FL + FR - RL - RR) / 4    # 前后轴因子
d = ( FL - FR - RL + RR) / 4    # 对角因子
```

记三因子向量为：

```text
y = [s, a, d]ᵀ
```

四轮相同的共同加减速不会进入 `y`。四个单轮模板为：

```text
h_FL = (+1, +1, +1)
h_FR = (-1, +1, -1)
h_RL = (+1, -1, -1)
h_RR = (-1, -1, +1)
```

单轮相对变快时，三因子变化方向应与对应模板一致。

### 4.3 持续残差和瞬时边沿

在线模型维护三因子均值 `μ_level`。持续残差为：

```text
r_level(t) = y(t) - μ_level(t)
```

边沿使用前后两个 6 帧窗口：

```text
edge(t) = mean(y[t-5:t]) - mean(y[t-11:t-6])
r_edge(t) = edge(t) - μ_edge(t)
```

持续残差描述单轮轮速平台是否形成；边沿残差描述平台开始时的冲击。

## 5. 在线协方差与匹配滤波

### 5.1 在线统计

持续残差模型和边沿模型分别维护 `3×3` EWMA 协方差：

- 均值更新率：`mean_alpha = 0.001`；
- 持续协方差更新率：`level_cov_alpha = 0.004`；
- 边沿协方差更新率：`edge_cov_alpha = 0.012`；
- 前 100 个样本使用不小于 `1 / n` 的更新率；
- 每 10 帧刷新一次逆协方差。

求逆前对协方差做两项保护：

- 持续和边沿对角方差下限分别为 `2.5e-8` 和 `1.0e-8`；
- 非对角项乘以 `1 - 0.25`，降低小样本相关性导致的数值不稳定。

### 5.2 四轮匹配 Z 分

对任意三因子残差 `r` 和逆协方差 `Σ⁻¹`，第 `i` 轮得分为：

```text
z_i = h_iᵀ Σ⁻¹ r / sqrt(h_iᵀ Σ⁻¹ h_i)
```

持续残差产生 `level_z_i`，边沿残差产生 `shock_z_i`。协方差归一化会降低正常
道路中常见变化方向的权重，增强异常单轮指纹。

### 5.3 轮位隔离度

```text
isolation_i = z_i - max(z_j),  j != i
```

只有目标轮明显领先其他三轮时，隔离度才为较大的正数。

### 5.4 物理投影

算法同时计算不经过协方差缩放的物理量：

```text
physical_level_i = h_iᵀ r_level
physical_edge_i  = h_iᵀ r_edge
```

Z 分用于适应不同道路噪声，物理投影用于限制实际幅值，避免低方差环境中的微小
变化被放大为很高的 Z 分。

## 6. 累积证据与风险分

每轮维护两个正向累积量：

```text
C_i(t) = max(0, 0.94  * C_i(t-1) + max(0, shock_z_i - 1.0))
P_i(t) = max(0, 0.985 * P_i(t-1) + max(0, level_z_i - 0.5))
```

综合证据为：

```text
q_i = 0.12 * shock_z_i
    + 0.36 * level_z_i
    + 0.08 * shock_isolation_i
    + 0.16 * level_isolation_i
    + 0.10 * min(C_i / 6, 8)
    + 0.18 * min(P_i / 20, 8)

risk_i = 100 * sigmoid((q_i - 3) / 1.15)
```

`risk_i` 只用于量化内部证据。最终报警必须同时通过物理幅值、持续性、隔离度、
单轮纯度和共同速度约束，因此不能把 `risk_i = 80` 解释成 80% 爆胎概率。

## 7. 状态机

### 7.1 预热 `warming`

默认需要 300 帧持续因子样本，同时边沿模型也要获得足够样本。在 100 Hz 下约为
3 秒。预热期间只学习正常模型，不产生候选。

### 7.2 监控 `monitoring`

第 `i` 轮同时满足以下条件时进入候选：

```text
3.0 <= shock_z_i <= 4.5
shock_isolation_i >= 2.0
physical_edge_i >= 0.0039
```

上限 `4.5` 用于排除超强的轮滑或数据冲击。候选起点按滤波延迟回推：

```text
回推帧数 = smooth_window - 1 + edge_half_window = 10 帧
```

该时间只写入 `estimated_onset_*`，不会把报警时间回写到过去。

### 7.3 候选 `candidate`

候选期间冻结在线均值和协方差，避免把事件吸收到正常模型。默认至少收集 16 帧
候选证据。以下任一条件会清除候选：

- 物理平台连续 15 帧低于 `-0.0025`；
- 单轮物理峰值超过 `0.025`，按轮滑或异常冲击处理；
- 候选达到 120 帧仍未确认。

### 7.4 16 帧确认条件

候选达到 16 帧后，每帧检查全部条件：

1. `peak(physical_level) >= 0.006`；
2. 若候选共同速度范围超过 `0.020`，则物理峰值必须至少为 `0.010`；
3. 最近 16 帧 `median(physical_level) >= 0.006`；
4. 至少 75% 的帧满足 `physical_level >= 0.0028`；
5. 最近 16 帧 `median(level_z) >= 1.5`；
6. 最近 16 帧 `median(level_isolation) >= 1.0`；
7. 至少 82.5% 的帧满足 `level_isolation >= 1.0`；
8. 三个非目标轮各自的 `physical_level` 中位值均不大于 0；
9. 最近 16 帧风险分中位值至少为 `52.5`；
10. 候选风险分峰值至少为 `82.0`；
11. 共同对数轮速范围不超过当前工况上限。

共同速度工况按候选开始至当前帧判断：

```text
common(t) = mean(smoothed_log_wheels(t))
range     = max(common) - min(common)
delta     = common[-1] - common[0]

braking = delta < 0 and -delta >= 0.8 * range
```

- 明确制动：共同速度范围上限为 `0.250`；
- 其他工况：共同速度范围上限为 `0.050`。

第 8 条是本轮快速确认的关键纯度约束。正常道路短冲击常使多个轮位投影同时为正；
0818 的有效事件在确认窗口内保持单一 RR 指纹，因此可以把确认窗口从 55 帧缩短
到 16 帧，同时维持现有 RobustData 0 误报结果。

### 7.5 报警 `alarm`

全部条件成立后：

- 当前帧 `new_blowouts[i] = True`；
- `blowout_alarms[i]` 锁存为真；
- 状态变为 `alarm`；
- 调用 `reset()` 前不会自动撤销报警。

某轮报警后，依赖该轮作为对角参考的轮位会停止检测，避免已爆胎轮污染参考组后
产生连锁误报。

## 8. 失效和复位

轮速无效时会立即清除未确认候选。连续 50 帧无效后，还会清除平滑历史、边沿
历史和在线均值/协方差，恢复有效轮速后必须重新预热。已经锁存的报警不会因低速
或无效输入自动清除；需要显式调用：

```python
detector.reset()
```

## 9. 默认参数摘要

| 类别 | 参数 | 默认值 |
| --- | --- | ---: |
| 输入 | `sample_rate_hz` | 100 |
| 输入 | `min_avg_speed` | 20 rad/s |
| 平滑 | `smooth_window` | 5 帧 |
| 边沿 | `edge_half_window` | 6 帧 |
| 预热 | `warmup_frames` | 300 帧 |
| 触发 | `shock_trigger_z` / `max_shock_trigger_z` | 3.0 / 4.5 |
| 触发 | `shock_isolation_z` | 2.0 |
| 触发 | `min_physical_edge` | 0.0039 |
| 确认 | `confirm_frames` | 16 帧 |
| 确认 | `persistence_tail_frames` | 16 帧 |
| 确认 | `min_physical_peak` | 0.006 |
| 确认 | `min_physical_persistence` | 0.006 |
| 纯度 | `max_peer_physical_median` | 0.0 |
| 风险 | `min_median_risk` / `min_peak_risk` | 52.5 / 82.0 |
| 超时 | `candidate_timeout_frames` | 120 帧 |

完整配置以 `QuantBlowoutConfig` 为准，评估结果中的实际配置快照见
[`0818_algorithm_evaluation/summary.json`](../0818_algorithm_evaluation/summary.json)。

## 10. Python 接入

```python
from quant_wheel_blowout_detector import (
    QuantBlowoutDetector,
    QuantFrame,
)

detector = QuantBlowoutDetector()

result = detector.push(
    QuantFrame.from_sequences(
        t_sec,
        [fl_corrected, fr_corrected, rl_corrected, rr_corrected],
    )
)

if result.new_blowouts[3]:
    print("RR blowout confirmed")
```

检测器是有内部状态的在线对象。同一数据流应按时间顺序使用同一个实例，不应把
同一实例同时用于多台车或多条记录。

## 11. CSV 回放

```bash
python3 -m quant_wheel_blowout_detector.cli \
  --input wheel_speed.csv \
  --output quant_result.csv
```

默认输入列：

```text
time_s
wheel0_corrected_rad_s
wheel1_corrected_rad_s
wheel2_corrected_rad_s
wheel3_corrected_rad_s
```

可通过 `--time-column` 和 `--wheel-columns FL FR RL RR` 指定其他列名。

## 12. 当前验证结果

优化依据：

- 正样本：0818 的 `60kpa_RRBlowOut`、`Acc_RRBlowOut`、
  `Brk_RRBlowOut`；
- 暂缓：`40kph_RRBlowOut` 的标注和轮速反馈问题；
- 排除：`ly` 爆胎数据；
- 负样本：37 条 RobustData 正常道路记录。

结果：

| 0818 样本 | 报警轮位 | 延迟帧 | 延迟时间 |
| --- | --- | ---: | ---: |
| 60kpa | RR | 46 | 0.46 s |
| Acc | RR | 41 | 0.41 s |
| Brk | RR | 32 | 0.32 s |

- 三条有效正样本 3/3 检出；
- 无事件前报警，无错误轮位报警；
- 平均延迟 39.7 帧，即 0.397 秒；
- 最大延迟 46 帧，即 0.46 秒；
- RobustData 共 8,922,100 帧、24.7835 小时，0/37 记录误报；
- 20 帧端到端目标未找到 RobustData 零误报分界。

快速确认搜索过程见
[`0818_fast_detection_exploration.md`](../docs/0818_fast_detection_exploration.md)。

## 13. 局限与放行要求

- 当前 0818 正样本只有 3 条，且全部为 RR；
- 参数使用过相同 0818 正样本和 RobustData 反例开发，不是锁参后的独立盲测；
- 纯轮速相对特征无法可靠识别四轮完全同幅变化；
- 低速阶段禁用检测；
- 轮滑、道路冲击和爆胎初段在很短窗口内存在重叠，不能以当前证据承诺 20 帧报警；
- `risk_score` 未做概率校准；
- 量产前应补齐 FL/FR/RL、多轮事件、不同车辆与轮胎，并按日期和道路隔离做
  walk-forward 或锁参盲测。

## 14. 回归验证命令

```bash
python3 -m unittest \
  quant_wheel_blowout_detector.test_detector \
  test_build_0818_display -v

python3 evaluate_0818_algorithms.py --jobs 4
```

统一结果输出到 `0818_algorithm_evaluation/`。
