# 小波形态爆胎检测器

## 两套纯轮速算法

仓库同时保留两套只读取四轮校正轮速的独立算法：

- `wheel_speed_only_blowout_detector`：双空间物理硬门限，支持平稳工况下 55 帧
  早确认，否则保留 70 帧复核；
- `quant_wheel_blowout_detector`：Hadamard 三因子、在线协方差、匹配滤波和
  CUSUM 风险分，默认 55 帧确认并排除超过 2.5% 的轮滑型物理投影。

两套算法优化前后的统一评估入口为：

```bash
python3 evaluate_speed_algorithms.py --jobs 4
```

逐样本、逐正常道路和汇总结果位于 `speed_algorithm_evaluation/`。这些结果是
同源开发回放，不是锁参后的独立盲测。

## 胎压对角融合四轮检测（推荐低误报方案）

算法资料分为两份：

- 逐帧执行顺序、精确公式、状态机、阈值判定和数值示例见
  [`docs/pressure_fusion_detector_algorithm.md`](docs/pressure_fusion_detector_algorithm.md)；
- 接口治理、安全降级、验证证据和量产准入要求见
  [`docs/pressure_fusion_detector_algorithm_spec.md`](docs/pressure_fusion_detector_algorithm_spec.md)。

`pressure_fusion_detector.py` 面向以下车辆配置：四轮都有轮速，且恰好一组
对角轮有逐轮胎压爆胎信号，但软件预先不知道是 `FL+RR` 还是 `FR+RL`。
输入胎压信号使用三态语义：

- `True`：胎压系统确认该轮爆胎；
- `False`：胎压系统确认该轮正常；
- `None`：该轮没有信号或当前不可用。

检测器根据两个非 `None` 的位置自动识别胎压对角。胎压对角逐轮使用直接
信号；另一对角的两个轮分别运行轮速检测，可同时或先后报警。轮速检测同时
要求两个因果特征成立：

```text
individual_i = log(speed_i) - mean(log(speed_pressure_diagonal))
diagonal     = sum(log(speed_speed_diagonal))
               - sum(log(speed_pressure_diagonal))
```

两者都先减去滚动正常基线。`individual_i` 定位具体车轮；`diagonal` 是四轮
Hadamard 对角残差，一阶转弯左右轮差、共同加减速和前后轴差会在其中抵消。
确认阶段还要求高位持续、同组另一轮没有反向大幅变化、四轮平均速度没有处于
剧烈瞬变。轮速报警约在事件后 0.8 秒输出，换取较低误报；胎压报警立即输出。

```python
from wavelet_shape_blowout_detector import (
    PressureFusionBlowoutDetector,
    PressureFusionFrame,
)

detector = PressureFusionBlowoutDetector()

# 本车当前识别到 FR+RL 有胎压信号，FL+RR 由轮速检测。
result = detector.push(
    PressureFusionFrame.from_sequences(
        t_sec,
        [fl_speed, fr_speed, rl_speed, rr_speed],
        [None, fr_pressure_blowout, rl_pressure_blowout, None],
    )
)
print(result.blowout_alarms)  # FL, FR, RL, RR
print(result.alarm_sources)   # pressure / wheel_speed_confirmed / none
```

如果胎压参考组任一轮已经报爆胎，`speed_detection_available` 会变为 `False`，
检测器暂停对另一对角作仅轮速强判定。这避免使用已爆胎轮作为正常参考后产生
连锁误报。四轮同时发生幅度相同的轮速变化时，相对轮速不可观测；此时只能
输出有胎压信号的轮位，其他轮需要独立胎压、车速、轮端加速度等额外信息。

回归测试：

```bash
python3 -m unittest \
  wavelet_shape_blowout_detector.test_pressure_fusion_detector -v
```

CSV 中两列胎压信号位于 `FR+RL` 时：

```bash
python3 -m wavelet_shape_blowout_detector.pressure_fusion_cli \
  --input wheel_speed.csv --output pressure_fusion_result.csv \
  --pressure-fr-column FR_pressure_blowout \
  --pressure-rl-column RL_pressure_blowout
```

当前离线回放结果：`ly` 的 8/8 条 RR 实车爆胎均检出，无事件前/错误轮位
报警，平均确认延迟 0.828 秒、最大 0.85 秒；`RobustData` 的 37 条明文正常
实路记录共 8,922,100 帧（24.78 小时），分别假设 `FL+RR` 和 `FR+RL` 为
胎压对角，两种布局均为 0/37 记录误报。阈值使用了这批数据分析，不是独立
盲测结果；FL/FR/RL 实车爆胎及多轮爆胎目前只有轮位变换和合成回归测试，
上线前仍需保留新道路、新轮位数据作锁定参数后的独立验证。
逐条回放数值保存在 `pressure_fusion_evaluation_summary.json`。

### 胎压融合算法显示台

显示台用于排查算法进入疑似状态的具体片段。它会完整回放所选记录，把内部
`candidate` 区间按轮位列在左侧，并区分最终确认和被排除的候选；右侧 Plotly
图同步显示四轮校正轮速、逐轮/对角增益、上升沿证据和锁存报警。点击任一疑似
段即可跳到附近窗口，图表支持拖动、滚轮缩放、双击复位和底部范围条。

```bash
haar/bin/python -m wavelet_shape_blowout_detector.serve_pressure_fusion_display \
  --port 8771
```

浏览器访问 `http://127.0.0.1:8771`。首次打开长记录时需要完整扫描一次，结果会
缓存在内存中；单次曲线窗口默认最多 120 秒，可通过 `--max-window-s` 调整。

该算法专门提取真实样本中的三段形态：爆胎轮相对其他轮先加快、约
0.1 秒后小幅回落、随后保持偏快。

## 算法

默认把四轮分为 `FL+RR`、`FR+RL` 两组对角。每个目标轮使用另一组
对角的两轮均值作为参考：

```text
FL、RR reference = mean(FR, RL)
FR、RL reference = mean(FL, RR)
ratio(t) = target(t) / reference(t)
gain(t)  = smoothed_ratio(t) / rolling_baseline(t) - 1
haar(t)  = mean(gain newest 50 ms) - mean(gain previous 50 ms)
```

因此无需预先知道哪一组对角正常：变快的一组产生正向爆胎形态，正常组相对
它只会变慢，不会通过正向上升门限。也可用
`--reference-mode peer_median` 恢复“其余三轮中位数”参考方式。

默认状态机依次要求：

1. `haar >= +0.55%`，且峰值增益位于 `+0.4%`～`+2.0%`；
2. 之后 40～240 ms 内出现 `-0.33%`～`-1.2%` 的下降沿，且轮速
   没有跌回事件前基线以下；
3. 再观察 300 ms，增益中位数至少 `+0.6%`，最后 100 ms 至少
   `+0.85%`，且平台没有超过此前峰值；
4. 三项均满足时锁存对应轮位的爆胎报警。

输入应优先使用齿圈误差校正后的 100 Hz 轮速。算法是因果的，默认确认报警
延迟约 0.5～0.6 秒。Haar 系数负责提取上升和下降沿，低频滑动基线负责确认
后续持续偏快；只看小波细节系数无法表达持续平台。

## Python API

```python
from wavelet_shape_blowout_detector import (
    WaveletShapeBlowoutDetector,
    WheelFrame,
)

detector = WaveletShapeBlowoutDetector()
result = detector.push(WheelFrame.from_sequences(t_sec, four_wheel_speeds))
if result.new_blowouts[3]:
    print("RR blowout", result.estimated_onset_times_s[3])
```

如果某个轮位有独立的“确认正常”信号，可随帧传入。`True` 表示确认正常，
`False` 或 `None` 只表示未知，不会直接判为爆胎：

```python
result = detector.push(
    WheelFrame.from_sequences(
        t_sec,
        four_wheel_speeds,
        normal_signals=[False, False, False, True],  # RR 确认正常
    )
)
```

此时 FL、FR、RL 都优先使用 RR 轮速作为正常锚点，RR 自身不会进入爆胎
状态机。`result.reference_sources` 会说明每个轮当前使用
`peer_median` 还是 `confirmed_normal:RR`。

## CSV 命令行

```bash
.py/bin/python -m wavelet_shape_blowout_detector.cli \
  --input wheel_speed_raw_vs_corrected.csv \
  --output wavelet_shape_result.csv
```

默认轮速列为 `wheel0_corrected_rad_s`～`wheel3_corrected_rad_s`，可用
`--wheel-columns FL FR RL RR` 指定其他列。默认同时运行四个轮位的状态机，
输出独立的 `FL_blowout`、`FR_blowout`、`RL_blowout` 和 `RR_blowout`：

```bash
.py/bin/python -m wavelet_shape_blowout_detector.cli \
  --input input.csv --output output.csv
```

如果只需要检测 RR，可传入 `--target-wheels RR`。

正常信号来自 CSV 时，可以指定对应列。例如 RR 是正常对角参考轮：

```bash
.py/bin/python -m wavelet_shape_blowout_detector.cli \
  --input input.csv --output output.csv \
  --normal-rr-column RR_normal
```

## 多轮检测边界

已知两组对角中有一组正常时，不需要知道具体是哪组。每个轮都使用另一组对角
作为参考；异常对角的一个或两个轮会分别产生正向报警，正常对角只产生负向
相对变化。报警采用逐轮独立状态机，某个轮已经锁存不会停止其他轮继续检测。

如果没有正常信号，三个轮或四个轮同时出现完全相同的轮速变化时，四轮相对
量会互相抵消。若三个轮爆胎且剩余轮有可信的正常信号，该正常轮会成为共同
参考，三个异常轮仍可分别检测。四轮同时爆胎时不存在正常轮，仍需要独立车速、
胎压、轮端加速度或其他直接爆胎信号。

当前阈值来自同一批 8 条 RR 真实爆胎样本，不能视作独立验证结果。双轮检测
目前只有合成回归测试；FL、FR、RL 轮位和长时间正常道路误报率仍需要新数据
验证。

## 测试 Display

单条 CSV 可以生成交互式 HTML：

```bash
python3 -m wavelet_shape_blowout_detector.display \
  --input wheel_speed_raw_vs_corrected.csv \
  --event-time 402.16 \
  --output display.html
```

页面依次显示四轮轮速、相对增益、Haar 系数和四路报警。批量生成 8 条原始
样本及索引页：

```bash
python3 -m wavelet_shape_blowout_detector.build_test_display \
  --manifest /path/to/template_manifest.csv \
  --output-dir display
```

Display 需要 `plotly`，核心检测器本身仍只依赖 Python 标准库。

### 488 条增强测试按需查看

488 条增强样本不预生成静态 HTML。启动查看器后，首页可以按事件/正常类型、
来源事件和 sample_id 筛选，点击样本时才运行当前小波算法：

```bash
haar/bin/python -m wavelet_shape_blowout_detector.serve_augmented_display \
  --dataset-dir /home/zich/haar/augmented_event_dataset_v2 \
  --port 8765
```

浏览器访问 `http://127.0.0.1:8765`。事件样本默认显示事件前后各 5 秒，正常
样本显示完整窗口；最近 12 个页面缓存在内存中。

首页和样本页顶部都可以切换三种视图，选择会在样本链接、上一条/下一条和返回
首页时保持：

- `当前版`：原有上升沿、下降沿、平台状态机；
- `Evidence`：累计上升、回撤和高位持续证据算法；
- `对比`：对同一样本上下显示两种算法的分类、报警轮位、延迟和完整曲线。

Evidence 的对外报警使用低延迟累计上升证据，后续形态确认单独显示。图中的
浅色短柱是快速报警，深色高柱是经过回撤和持续性验证后的确认锁存；确认结果
不会把首报延迟推迟到 0.5 秒之后。

报警区间会以对应轮位颜色铺满图表背景；事件前报警、错误轮位报警和正常样本
报警会额外叠加红色虚线边框及“误报”标识。主页支持全文检索，并可按样本类型、
来源事件和以下评价分类筛选：及时检出、延迟检出、漏报、事件误报、误报后检出、
正常通过、正常误报。

重新生成修复版增强数据：

```bash
/home/zich/py/.py/bin/python -m wavelet_shape_blowout_detector.build_augmented_event_dataset \
  --labels /home/zich/py/wheel_cog_outputs/blowout_manual_labeling_package/labeling_package/event_time_labels.csv \
  --batch-summary /home/zich/py/wheel_cog_outputs/fast_alarm_batch_outputs/fast_alarm_batch_summary.csv \
  --output-dir /home/zich/haar/augmented_event_dataset_v2 \
  --seed 20260713
```

重新计算 488 条样本的小波评价表：

```bash
/home/zich/py/.py/bin/python -m wavelet_shape_blowout_detector.evaluate_augmented \
  --dataset-dir /home/zich/haar/augmented_event_dataset_v2 \
  --output-csv display_488/v2_current_evaluation.csv \
  --output-json display_488/v2_current_evaluation_summary.json \
  --algorithm hard --jobs 12 --overwrite
```

重新计算 Evidence 评价表：

```bash
/home/zich/py/.py/bin/python -m wavelet_shape_blowout_detector.evaluate_augmented \
  --dataset-dir /home/zich/haar/augmented_event_dataset_v2 \
  --output-csv display_488/v2_evidence_evaluation.csv \
  --output-json display_488/v2_evidence_evaluation_summary.json \
  --algorithm evidence --jobs 12 --overwrite
```

修复版数据使用真实正常段的连续四轮相关残差，不再给四轮独立添加高斯白噪声。
当前 488 条相关增强样本中，两种算法均检出 408/408；当前版正常误报 9/80、
平均首报延迟约 131 ms，Evidence 正常误报 10/80、平均延迟约 117 ms。两者都
只有 1 条事件样本带事件前或错误轮位报警。这些仍是由 8 条真实事件派生的压力
测试结果，不是独立实路统计。

## RobustData 实路正常数据

`RobustData` 中带日期的单条 `.txt` 是明文齿时间戳，可以先使用
`~/py/wheel_cog_outputs/process_wheel_cog.py` 做齿圈误差校正，再运行当前版和
Evidence 检测器。以下命令会完成转换、检测并持续写入批量汇总；中断后重跑会
复用已经生成的 CSV 和单条评价：

```bash
/home/zich/py/.py/bin/python -m wavelet_shape_blowout_detector.evaluate_robust_data \
  --input /home/zich/haar/RobustData \
  --output-dir /home/zich/haar/robust_data_results \
  --algorithm both
```

先验证一条时，可直接把 `--input` 指向具体文件。也可加 `--limit 1` 只处理按
路径排序后的第一条。结果入口为
`robust_data_results/robust_evaluation_summary.csv`，每条正常道路记录、每版算法
各占一行。`false_alarm` 统计对外快速报警，`confirmed_false_alarm` 统计通过完整
形态确认的误报；`alarm_wheels` 和各轮首次报警时间用于进一步定位。

4 个 `AllData_*.txt` 的文件头包含 `E-SafeNet/LOCK`，是加密汇总文件，不是可读
文本导出，批处理会明确标为 `locked` 并跳过。对应的带日期单条明文记录已经覆盖
在各子目录中。`#...txt#` 是编辑器临时文件，也会自动忽略。

评价完成后启动按需 Display：

```bash
/home/zich/py/.py/bin/python -m wavelet_shape_blowout_detector.serve_robust_display \
  --results-dir /home/zich/haar/robust_data_results \
  --port 8766
```

浏览器访问 `http://127.0.0.1:8766`。主页支持当前版、Evidence 和对比视图，以及
道路类型、评价状态和文件名筛选。单条记录默认只画首个误报附近的 10 秒，避免
把十几分钟或更长的记录一次性传给浏览器；详情页可以输入其他起止时间，或点击
各轮首次报警时间快速跳转。检测器仍从记录开头运行到所选窗口末端，因此保留了
完整的因果基线和报警状态。

详情页还提供 `齿信号` 视图，直接读取原始 16 位齿时间戳和单条转换目录中的
`learned_tooth_correction_factors.csv`。图中依次显示逐齿到达事件、校正齿周期
残差、累计相位残差、12 齿滑动周期以及每 10 ms 齿数/异常周期。负的周期残差
表示目标轮相对参考对角变快；相位残差持续增大表示累计齿领先。齿视图单次窗口
最多 30 秒，避免向浏览器传输过多逐齿点。

也可以为单条原始记录直接生成静态 HTML：

```bash
/home/zich/py/.py/bin/python -m wavelet_shape_blowout_detector.tooth_display \
  --input /path/to/raw_tooth_timestamps.txt \
  --factors /path/to/learned_tooth_correction_factors.csv \
  --start 179 --end 182 \
  --output tooth_display.html
```
