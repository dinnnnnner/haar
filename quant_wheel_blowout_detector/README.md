# 四轮轮速量化爆胎检测器

这是第三套完全独立的算法。它只读取四轮轮速，不导入或修改原有小波算法和
双空间硬门限算法。

## 量化模型

四轮取对数轮速后，用 Hadamard 变换构造三个非共同市场因子：

```text
左右因子 s = (FL - FR + RL - RR) / 4
前后因子 a = (FL + FR - RL - RR) / 4
对角因子 d = (FL - FR - RL + RR) / 4
```

共同加减速在三个因子中被消除。四个轮位对应四个固定指纹：

```text
FL = (+1, +1, +1)    FR = (-1, +1, -1)
RL = (+1, -1, -1)    RR = (-1, -1, +1)
```

检测器在线维护因子的 EWMA 均值以及 `3×3` 协方差矩阵。瞬时边沿和持续偏离
分别使用协方差逆矩阵进行匹配滤波：

```text
z_i = h_iᵀ Σ⁻¹ y / sqrt(h_iᵀ Σ⁻¹ h_i)
```

因此常见转弯或轴间变化会在协方差中降权；只有同时符合某个单轮三因子指纹的
变化才会取得明显的轮位隔离度。最终风险分综合：

- 协方差标准化瞬时冲击；
- 协方差标准化持续偏离；
- 瞬时和持续轮位隔离度；
- 正向 CUSUM；
- 高位持续证据。

冲击触发后观察约 0.7 秒。确认还要求物理轮速偏差、高位占比、共同速度范围，
以及确认窗口至少 95% 的帧保持同一轮位指纹。候选期间冻结在线统计模型，避免
把事件吸收到正常基线。

`risk_score` 是内部量化证据强度，不是经过独立数据校准的真实概率；最终报警
仍以完整的持续性和风险约束为准。

## Python API

```python
from quant_wheel_blowout_detector import QuantBlowoutDetector, QuantFrame

detector = QuantBlowoutDetector()
result = detector.push(
    QuantFrame.from_sequences(t_sec, [fl_speed, fr_speed, rl_speed, rr_speed])
)

print(result.risk_scores)       # FL/FR/RL/RR，0–100 量化证据分
print(result.leading_wheel)     # 当前领先轮位索引
print(result.leading_margin)    # 第一名与第二名的风险分差
print(result.new_blowouts)      # 本帧新确认
print(result.blowout_alarms)    # 锁存报警
```

## CSV CLI

```bash
python3 -m quant_wheel_blowout_detector.cli \
  --input augmented_event_dataset_v2/samples/E01_event_000.csv \
  --output quant_blowout_result.csv
```

默认输入列为 `time_s` 和 `wheel0_corrected_rad_s` 至
`wheel3_corrected_rad_s`。其他列名可通过 `--time-column` 和
`--wheel-columns FL FR RL RR` 指定。

## 交互式 Display

单条 CSV 可生成量化算法的完整证据链展示：四轮轮速、Hadamard 三因子残差、
逐轮物理投影、冲击/持续匹配 Z 分、轮位隔离度、风险分，以及候选和锁存报警
状态。图表支持拖动、滚轮缩放、统一悬浮提示和底部范围条。

```bash
haar/bin/python -m quant_wheel_blowout_detector.display \
  --input augmented_event_dataset_v2/samples/E01_event_000.csv \
  --event-time 40.0 --window-before 5 --window-after 5 \
  --output quant_display.html
```

不传 `--event-time` 时显示完整记录。自定义 CSV 列名可使用 `--time-column` 和
`--wheel-columns FL FR RL RR`；检测器仍会回放显示窗口之前的所有帧，以保证
窗口左边界处的在线基线和状态与完整回放一致。

## 测试

```bash
haar/bin/python -m unittest \
  quant_wheel_blowout_detector.test_detector \
  quant_wheel_blowout_detector.test_display -v
```

## 当前开发回放

- 8 条真实 RR 爆胎：8/8 检出，无提前或错误轮位报警；平均确认延迟
  0.834 秒，最大 0.86 秒。
- 488 条增强测试：408/408 个事件在 2 秒内检出，80/80 个正常窗口无报警；
  平均延迟 0.834 秒，最大 0.992 秒。
- 37 条真实正常道路：8,922,100 帧、24.784 小时，0/37 记录报警。
- 合成回归覆盖四个单轮轮位、共同加速、转弯、轴间同步变化、低速失效和复位。

阈值使用过同源数据开发，包括根据正常道路反例加入轮位指纹稳定率，因此这些
结果不是锁参后的独立盲测。真实正样本只有 RR；正式放行仍需补齐其他轮位、
多轮实车事件，以及按日期和道路隔离的 walk-forward 测试。

机器可读结果见 [`validation_summary.json`](validation_summary.json)。
