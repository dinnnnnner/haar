# 纯四轮轮速爆胎检测器

这是一个独立的新算法包。它不读取胎压信号，也不修改或依赖原有检测器；在线
输入只有时间戳和按 `FL, FR, RL, RR` 排列的四轮轮速。

## 核心判据

先对四轮轮速取自然对数。目标轮 `i` 的第一项证据为其相对另一组对角轮的
偏差：

```text
individual_i = log(w_i) - mean(log(w_opposite_diagonal))
```

第二项证据为四轮 Hadamard 对角残差，并按目标轮所在对角取符号：

```text
d = log(w_FL) - log(w_FR) - log(w_RL) + log(w_RR)
diagonal_i = sign_i * d, sign = (+1, -1, -1, +1)
```

两项都减去滚动正常基线。共同加减速在比值中消失；理想的一阶左右转弯差、
前后轴共同差在 `d` 中消失。算法只有在两项同时出现正向边沿，且随后约
0.7 秒持续高于基线时才锁存该轮报警。确认期还会检查同对角另一轮没有强烈
反向变化，以及四轮几何平均速度没有剧烈瞬变。

报警一旦锁存，只能调用 `reset()` 清除。某轮报警后，以该轮为参考的另一组
对角暂停判断，避免污染参考导致连锁误报；同一对角的另一个轮仍可继续检测。

## Python API

```python
from wheel_speed_only_blowout_detector import (
    WheelSpeedBlowoutDetector,
    WheelSpeedFrame,
)

detector = WheelSpeedBlowoutDetector()
result = detector.push(
    WheelSpeedFrame.from_sequences(t_sec, [fl_speed, fr_speed, rl_speed, rr_speed])
)

print(result.new_blowouts)       # 本帧新确认的 FL/FR/RL/RR
print(result.blowout_alarms)     # 锁存报警
print(result.states)             # warming/monitoring/candidate/alarm/...
```

默认参数面向齿圈误差校正后的 100 Hz 轮速。启动后需要 2 秒正常数据建立基线，
平均轮速低于 `20 rad/s` 时不判断。

## CSV CLI

```bash
python3 -m wheel_speed_only_blowout_detector.cli \
  --input augmented_event_dataset_v2/samples/E01_event_000.csv \
  --output blowout_result.csv
```

上面的输入文件在本仓库中真实存在，可直接用于试运行。处理自己的数据时，把
`--input` 改为对应 CSV 的实际路径；不能直接使用不存在的示例文件名
`wheel_speed.csv`。

## 算法控制台

控制台只使用 Python 标准库，无需安装 Plotly。启动后访问
`http://127.0.0.1:8772`：

```bash
python3 -m wheel_speed_only_blowout_detector.serve_console --port 8772
```

主页包含 8 条真实爆胎基线和 37 条正常道路记录，也可以在“打开自己的 CSV”
中填写工作区相对路径或绝对路径。首次打开一条记录时会完整回放并缓存扫描
结果；长记录可能需要数秒。

详情页提供：

- 四轮校正轮速；
- 逐轮增益和带符号的 Hadamard 对角增益；
- 逐轮/对角上升沿及候选门限；
- 四轮候选状态和锁存报警；
- 完整记录内所有候选区间，区分“已排除”和“已确认”；
- 时间窗口切换、前后翻屏、同步鼠标读数和人工事件标记。

主页的“取消耗时统计”会全量扫描清单中的记录，统计候选从进入 `candidate`
到被算法明确排除的时长，并给出平均值、中位数、P95 和最大值。明细默认按
耗时从长到短排列，可按记录类型、轮位筛选或切换排序；点击“跳转查看”会直接
打开该候选前后各 2 秒的证据图。文件结束时仍处于候选状态的未决区间不会被
误计为已取消。首次打开统计页需要全量扫描当前清单，完成后结果会缓存在服务
进程内；两万级明细按每页 200 条输出，避免浏览器一次渲染全部记录。

CSV 使用其他列名时可在启动时指定：

```bash
python3 -m wheel_speed_only_blowout_detector.serve_console \
  --time-column timestamp \
  --wheel-columns FL FR RL RR
```

默认列名为 `time_s` 和 `wheel0_corrected_rad_s` 至
`wheel3_corrected_rad_s`。其他列名可用以下方式指定：

```bash
python3 -m wheel_speed_only_blowout_detector.cli \
  --input input.csv --output output.csv \
  --wheel-columns FL FR RL RR
```

## 可观测性边界

纯相对轮速无法识别四轮完全等幅、同步变化。相邻两轮同时等幅变化时，对角
残差也会抵消；本算法选择不报警，以避免把转弯、轴滑移或制动误判成爆胎。
这类场景需要独立车速、横摆率、纵横向加速度、胎压或轮端振动等额外信号。

阈值必须用目标车型、轮胎、齿圈处理链和道路数据重新锁定。仓库里的合成测试
只验证状态机和空间消扰逻辑，不能替代各轮位的实车爆胎试验。

## 测试

```bash
python3 -m unittest \
  wheel_speed_only_blowout_detector.test_detector \
  wheel_speed_only_blowout_detector.test_console -v
```

## 当前离线回放结果

- 8 条真实 RR 爆胎基线样本：8/8 正确检出，无提前或错误轮位报警；平均确认
  延迟 0.828 秒，最大 0.85 秒。
- 488 条增强测试：407/408 条事件在 2 秒内正确检出，1 条高噪声增强事件
  漏检；80/80 条正常窗口无误报。
- 37 条真实正常道路记录：8,922,100 帧、24.784 小时，0/37 记录误报。

参数曾使用同源数据开发，以上是开发回放而不是锁参后的独立盲测。真实正样本
目前只有 RR 轮位；FL、FR、RL 和多轮爆胎仍需补充实车验证。逐项汇总保存在
[`validation_summary.json`](validation_summary.json)。
