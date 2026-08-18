from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence

from quant_wheel_blowout_detector import QuantBlowoutDetector, QuantFrame
from wheel_speed_only_blowout_detector import (
    WheelSpeedBlowoutDetector,
    WheelSpeedFrame,
)


WHEEL_NAMES = ("FL", "FR", "RL", "RR")
WHEEL_COLORS = ("#2563eb", "#f59e0b", "#16a34a", "#dc2626")
TIMER_WRAP_US = 65_536
DEFAULT_SAMPLE_TIME_S = 0.01
DEFAULT_COG_COUNT = 48


@dataclass(frozen=True)
class RawFrame:
    wheel_timestamps: tuple[tuple[int, ...], ...]
    blowout_signal: bool


@dataclass
class CaseAnalysis:
    input_path: Path
    times: list[float]
    wheel_speeds: list[list[float]]
    blowout_signal: list[bool]
    signal_event_time_s: float | None
    phase_factors: tuple[tuple[float, ...], ...]
    wheel_individual_gains_pct: list[list[float | None]]
    wheel_individual_edges_pct: list[list[float | None]]
    wheel_diagonal_gains_pct: list[list[float | None]]
    wheel_diagonal_edges_pct: list[list[float | None]]
    wheel_candidates: list[list[bool]]
    wheel_alarms: list[list[bool]]
    quant_factor_residuals_pct: list[list[float | None]]
    quant_factor_edges_pct: list[list[float | None]]
    quant_physical_levels_pct: list[list[float | None]]
    quant_physical_edges_pct: list[list[float | None]]
    quant_shock_z_scores: list[list[float | None]]
    quant_level_z_scores: list[list[float | None]]
    quant_shock_isolation: list[list[float | None]]
    quant_level_isolation: list[list[float | None]]
    quant_cusum_scores: list[list[float | None]]
    quant_persistence_scores: list[list[float | None]]
    quant_risk_scores: list[list[float | None]]
    quant_states: list[list[str]]
    quant_leading_wheels: list[int | None]
    quant_leading_margins: list[float]
    quant_candidates: list[list[bool]]
    quant_alarms: list[list[bool]]
    wheel_first_alarms: tuple[float | None, ...]
    quant_first_alarms: tuple[float | None, ...]

    @property
    def duration_s(self) -> float:
        return self.times[-1] - self.times[0]


def _timestamp_row(line: str) -> tuple[int, ...]:
    fields = line.split()
    if not fields:
        return ()
    count = int(fields[0])
    if count < 0 or len(fields) - 1 < count:
        raise ValueError(
            f"齿时间戳行声明 {count} 个值，实际只有 {len(fields) - 1} 个"
        )
    return tuple(int(value) for value in fields[1 : count + 1])


def iter_raw_frames(path: Path) -> Iterable[RawFrame]:
    """Read the five non-empty rows that make up each 0818 frame."""

    in_data = False
    rows: list[str] = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not in_data:
                if stripped.lower() == "marks end":
                    in_data = True
                continue
            if not stripped:
                continue
            rows.append(stripped)
            if len(rows) != 5:
                continue
            wheel_rows = tuple(_timestamp_row(row) for row in rows[:4])
            signal_fields = rows[4].split()
            if not signal_fields:
                raise ValueError(f"帧数据行为空：{path}")
            yield RawFrame(
                wheel_timestamps=wheel_rows,
                blowout_signal=float(signal_fields[-1]) != 0.0,
            )
            rows.clear()
    if not in_data:
        raise ValueError(f"未找到 Marks end：{path}")
    if rows:
        raise ValueError(f"文件结尾存在不完整帧（{len(rows)}/5 行）：{path}")


def _wheel_intervals(frames: Sequence[RawFrame], wheel: int) -> list[int]:
    previous: int | None = None
    intervals: list[int] = []
    for frame in frames:
        for timestamp in frame.wheel_timestamps[wheel]:
            if previous is not None and timestamp != previous:
                intervals.append((timestamp - previous) % TIMER_WRAP_US)
            previous = timestamp
    return intervals


def learn_phase_factors(
    frames: Sequence[RawFrame], cog_count: int = DEFAULT_COG_COUNT
) -> tuple[tuple[float, ...], ...]:
    """Estimate static tooth-pitch correction from complete, smooth revolutions."""

    all_factors: list[tuple[float, ...]] = []
    for wheel in range(4):
        intervals = _wheel_intervals(frames, wheel)
        phase_samples: list[list[float]] = [[] for _ in range(cog_count)]
        for start in range(0, len(intervals) - cog_count + 1, cog_count):
            lap = intervals[start : start + cog_count]
            if (
                min(lap) < 200
                or max(lap) > 15_000
                or max(lap) >= 2.0 * min(lap)
            ):
                continue
            lap_mean = sum(lap) / cog_count
            for phase, interval_us in enumerate(lap):
                phase_samples[phase].append(interval_us / lap_mean)
        factors = [median(samples) if samples else 1.0 for samples in phase_samples]
        factor_mean = sum(factors) / cog_count
        all_factors.append(tuple(value / factor_mean for value in factors))
    return tuple(all_factors)


def corrected_wheel_speeds(
    frames: Sequence[RawFrame],
    phase_factors: Sequence[Sequence[float]],
    cog_count: int = DEFAULT_COG_COUNT,
) -> list[list[float]]:
    previous: list[int | None] = [None] * 4
    phases = [0] * 4
    frames_without_event = [0] * 4
    current_speeds = [0.0] * 4
    output: list[list[float]] = []
    for frame in frames:
        for wheel, timestamps in enumerate(frame.wheel_timestamps):
            frames_without_event[wheel] += 1
            for timestamp in timestamps:
                frames_without_event[wheel] = 0
                prior = previous[wheel]
                previous[wheel] = timestamp
                if prior is None or timestamp == prior:
                    continue
                interval_us = (timestamp - prior) % TIMER_WRAP_US
                factor = phase_factors[wheel][phases[wheel] % cog_count]
                phases[wheel] += 1
                if 200 <= interval_us <= 50_000:
                    corrected_us = interval_us / factor
                    current_speeds[wheel] = (
                        2.0 * math.pi / (cog_count * corrected_us * 1.0e-6)
                    )
            if frames_without_event[wheel] >= 5:
                current_speeds[wheel] = 0.0
        output.append(current_speeds.copy())
    return output


def sustained_signal_onset(
    values: Sequence[bool],
    minimum_frames: int = 20,
    sample_time_s: float = DEFAULT_SAMPLE_TIME_S,
) -> float | None:
    run_start: int | None = None
    for index, value in enumerate(values):
        if value and run_start is None:
            run_start = index
        elif not value:
            run_start = None
        if run_start is not None and index - run_start + 1 >= minimum_frames:
            return run_start * sample_time_s
    return None


def _percent(value: float) -> float | None:
    return value * 100.0 if math.isfinite(value) else None


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def analyze_file(
    input_path: Path,
    *,
    sample_time_s: float = DEFAULT_SAMPLE_TIME_S,
    cog_count: int = DEFAULT_COG_COUNT,
    minimum_signal_frames: int = 20,
) -> CaseAnalysis:
    frames = list(iter_raw_frames(input_path))
    if not frames:
        raise ValueError(f"没有数据帧：{input_path}")
    factors = learn_phase_factors(frames, cog_count)
    frame_speeds = corrected_wheel_speeds(frames, factors, cog_count)
    times = [index * sample_time_s for index in range(len(frames))]
    signals = [frame.blowout_signal for frame in frames]

    wheel_detector = WheelSpeedBlowoutDetector()
    quant_detector = QuantBlowoutDetector()
    wheel_speeds = [[] for _ in range(4)]
    wheel_individual_gains = [[] for _ in range(4)]
    wheel_individual_edges = [[] for _ in range(4)]
    wheel_diagonal_gains = [[] for _ in range(4)]
    wheel_diagonal_edges = [[] for _ in range(4)]
    wheel_candidates = [[] for _ in range(4)]
    wheel_alarms = [[] for _ in range(4)]
    quant_factor_residuals = [[] for _ in range(3)]
    quant_factor_edges = [[] for _ in range(3)]
    quant_physical_levels = [[] for _ in range(4)]
    quant_physical_edges = [[] for _ in range(4)]
    quant_shock_z = [[] for _ in range(4)]
    quant_level_z = [[] for _ in range(4)]
    quant_shock_isolation = [[] for _ in range(4)]
    quant_level_isolation = [[] for _ in range(4)]
    quant_cusum = [[] for _ in range(4)]
    quant_persistence = [[] for _ in range(4)]
    quant_risk_scores = [[] for _ in range(4)]
    quant_states = [[] for _ in range(4)]
    quant_leading_wheels: list[int | None] = []
    quant_leading_margins: list[float] = []
    quant_candidates = [[] for _ in range(4)]
    quant_alarms = [[] for _ in range(4)]
    wheel_first: list[float | None] = [None] * 4
    quant_first: list[float | None] = [None] * 4

    for t_sec, speeds in zip(times, frame_speeds):
        wheel_result = wheel_detector.push(
            WheelSpeedFrame.from_sequences(t_sec, speeds)
        )
        quant_result = quant_detector.push(QuantFrame.from_sequences(t_sec, speeds))
        for factor in range(3):
            quant_factor_residuals[factor].append(
                _percent(quant_result.factor_residuals[factor])
            )
            quant_factor_edges[factor].append(
                _percent(quant_result.factor_edges[factor])
            )
        quant_leading_wheels.append(quant_result.leading_wheel)
        quant_leading_margins.append(quant_result.leading_margin)
        for wheel in range(4):
            wheel_speeds[wheel].append(speeds[wheel])
            wheel_individual_gains[wheel].append(
                _percent(wheel_result.individual_gains[wheel])
            )
            wheel_individual_edges[wheel].append(
                _percent(wheel_result.individual_edges[wheel])
            )
            wheel_diagonal_gains[wheel].append(
                _percent(wheel_result.diagonal_gains[wheel])
            )
            wheel_diagonal_edges[wheel].append(
                _percent(wheel_result.diagonal_edges[wheel])
            )
            wheel_candidates[wheel].append(wheel_result.candidates[wheel])
            wheel_alarms[wheel].append(wheel_result.blowout_alarms[wheel])
            quant_physical_levels[wheel].append(
                _percent(quant_result.physical_levels[wheel])
            )
            quant_physical_edges[wheel].append(
                _percent(quant_result.physical_edges[wheel])
            )
            quant_shock_z[wheel].append(_finite(quant_result.shock_z_scores[wheel]))
            quant_level_z[wheel].append(_finite(quant_result.level_z_scores[wheel]))
            quant_shock_isolation[wheel].append(
                _finite(quant_result.shock_isolation[wheel])
            )
            quant_level_isolation[wheel].append(
                _finite(quant_result.level_isolation[wheel])
            )
            quant_cusum[wheel].append(_finite(quant_result.cusum_scores[wheel]))
            quant_persistence[wheel].append(
                _finite(quant_result.persistence_scores[wheel])
            )
            quant_risk_scores[wheel].append(quant_result.risk_scores[wheel])
            quant_states[wheel].append(quant_result.states[wheel])
            quant_candidates[wheel].append(quant_result.states[wheel] == "candidate")
            quant_alarms[wheel].append(quant_result.blowout_alarms[wheel])
            if wheel_result.new_blowouts[wheel] and wheel_first[wheel] is None:
                wheel_first[wheel] = t_sec
            if quant_result.new_blowouts[wheel] and quant_first[wheel] is None:
                quant_first[wheel] = t_sec

    return CaseAnalysis(
        input_path=input_path,
        times=times,
        wheel_speeds=wheel_speeds,
        blowout_signal=signals,
        signal_event_time_s=sustained_signal_onset(
            signals, minimum_signal_frames, sample_time_s
        ),
        phase_factors=factors,
        wheel_individual_gains_pct=wheel_individual_gains,
        wheel_individual_edges_pct=wheel_individual_edges,
        wheel_diagonal_gains_pct=wheel_diagonal_gains,
        wheel_diagonal_edges_pct=wheel_diagonal_edges,
        wheel_candidates=wheel_candidates,
        wheel_alarms=wheel_alarms,
        quant_factor_residuals_pct=quant_factor_residuals,
        quant_factor_edges_pct=quant_factor_edges,
        quant_physical_levels_pct=quant_physical_levels,
        quant_physical_edges_pct=quant_physical_edges,
        quant_shock_z_scores=quant_shock_z,
        quant_level_z_scores=quant_level_z,
        quant_shock_isolation=quant_shock_isolation,
        quant_level_isolation=quant_level_isolation,
        quant_cusum_scores=quant_cusum,
        quant_persistence_scores=quant_persistence,
        quant_risk_scores=quant_risk_scores,
        quant_states=quant_states,
        quant_leading_wheels=quant_leading_wheels,
        quant_leading_margins=quant_leading_margins,
        quant_candidates=quant_candidates,
        quant_alarms=quant_alarms,
        wheel_first_alarms=tuple(wheel_first),
        quant_first_alarms=tuple(quant_first),
    )


def _compact(values: Sequence[float | None], digits: int = 5) -> list[float | None]:
    return [None if value is None else round(value, digits) for value in values]


def _trace(
    *,
    x: Sequence[float],
    y: Sequence[object],
    name: str,
    color: str,
    row: int,
    dash: str = "solid",
    width: float = 1.2,
    legendgroup: str | None = None,
    showlegend: bool = True,
    hovertemplate: str | None = None,
) -> dict[str, object]:
    axis = "" if row == 1 else str(row)
    return {
        "type": "scattergl",
        "mode": "lines",
        "x": _compact(x, 2),
        "y": y,
        "name": name,
        "xaxis": f"x{axis}",
        "yaxis": f"y{axis}",
        "line": {"color": color, "width": width, "dash": dash},
        "legendgroup": legendgroup or name,
        "showlegend": showlegend,
        "connectgaps": False,
        "hovertemplate": hovertemplate or f"{name}: %{{y:.3f}}<extra></extra>",
    }


def _active(values: Sequence[bool], level: float) -> list[float | None]:
    return [level if value else None for value in values]


def build_plot_payload(data: CaseAnalysis) -> tuple[list[dict[str, object]], dict[str, object]]:
    traces: list[dict[str, object]] = []
    for wheel, (name, color) in enumerate(zip(WHEEL_NAMES, WHEEL_COLORS)):
        traces.append(
            _trace(
                x=data.times,
                y=_compact(data.wheel_speeds[wheel]),
                name=name,
                color=color,
                row=1,
                legendgroup=name,
                hovertemplate=f"{name} 轮速: %{{y:.4f}} rad/s<extra></extra>",
            )
        )
        traces.append(
            _trace(
                x=data.times,
                y=_compact(data.wheel_individual_gains_pct[wheel]),
                name=f"{name} gain",
                color=color,
                row=2,
                legendgroup=name,
                showlegend=False,
            )
        )
        traces.append(
            _trace(
                x=data.times,
                y=_compact(data.wheel_individual_edges_pct[wheel]),
                name=f"{name} edge",
                color=color,
                row=2,
                dash="dot",
                legendgroup=name,
                showlegend=False,
            )
        )
        traces.append(
            _trace(
                x=data.times,
                y=_compact(data.wheel_diagonal_gains_pct[wheel]),
                name=f"{name} diagonal",
                color=color,
                row=3,
                legendgroup=name,
                showlegend=False,
            )
        )
        traces.append(
            _trace(
                x=data.times,
                y=_compact(data.quant_physical_levels_pct[wheel]),
                name=f"{name} physical",
                color=color,
                row=4,
                legendgroup=name,
                showlegend=False,
            )
        )
        traces.append(
            _trace(
                x=data.times,
                y=_compact(data.quant_risk_scores[wheel], 3),
                name=f"{name} risk",
                color=color,
                row=5,
                legendgroup=name,
                showlegend=False,
            )
        )

    traces.append(
        _trace(
            x=data.times,
            y=[5.0 if value else 4.55 for value in data.blowout_signal],
            name="爆胎信号位",
            color="#111827",
            row=6,
            width=2.0,
            hovertemplate="爆胎信号位: %{customdata}<extra></extra>",
        )
    )
    traces[-1]["customdata"] = [int(value) for value in data.blowout_signal]
    for wheel, (name, color) in enumerate(zip(WHEEL_NAMES, WHEEL_COLORS)):
        wheel_level = 3.7 - 0.22 * wheel
        quant_level = 2.3 - 0.22 * wheel
        traces.append(
            _trace(
                x=data.times,
                y=_active(data.wheel_candidates[wheel], wheel_level),
                name=f"wheel_only {name} candidate",
                color=color,
                row=6,
                dash="dot",
                width=3.0,
                showlegend=False,
            )
        )
        traces.append(
            _trace(
                x=data.times,
                y=_active(data.wheel_alarms[wheel], wheel_level),
                name=f"wheel_only {name} alarm",
                color=color,
                row=6,
                width=7.0,
                showlegend=False,
            )
        )
        traces.append(
            _trace(
                x=data.times,
                y=_active(data.quant_candidates[wheel], quant_level),
                name=f"quant {name} candidate",
                color=color,
                row=6,
                dash="dot",
                width=3.0,
                showlegend=False,
            )
        )
        traces.append(
            _trace(
                x=data.times,
                y=_active(data.quant_alarms[wheel], quant_level),
                name=f"quant {name} alarm",
                color=color,
                row=6,
                width=7.0,
                showlegend=False,
            )
        )

    axis_titles = (
        "校正轮速 / rad·s⁻¹",
        "wheel_only 逐轮 gain / edge (%)",
        "wheel_only 对角 gain (%)",
        "quant 物理投影 (%)",
        "quant risk",
        "信号 / 状态",
    )
    layout: dict[str, object] = {
        "template": "plotly_white",
        "height": 1450,
        "hovermode": "x unified",
        "dragmode": "pan",
        "grid": {"rows": 6, "columns": 1, "pattern": "independent", "roworder": "top to bottom"},
        "legend": {"orientation": "h", "x": 0.0, "y": 1.035},
        "margin": {"l": 86, "r": 32, "t": 70, "b": 55},
        "uirevision": data.input_path.name,
    }
    for row, title in enumerate(axis_titles, start=1):
        suffix = "" if row == 1 else str(row)
        layout[f"xaxis{suffix}"] = {
            "matches": "x" if row > 1 else None,
            "showticklabels": row == 6,
            "title": "时间 / s" if row == 6 else None,
            "rangeslider": {"visible": row == 6, "thickness": 0.05},
        }
        layout[f"yaxis{suffix}"] = {"title": title, "automargin": True}
    layout["yaxis6"] = {
        "title": "信号 / 状态",
        "range": [1.15, 5.35],
        "tickmode": "array",
        "tickvals": [5.0, 3.37, 1.97],
        "ticktext": ["爆胎信号", "wheel_only", "quant"],
    }
    if data.signal_event_time_s is not None:
        layout["shapes"] = [
            {
                "type": "line",
                "xref": "x",
                "yref": "paper",
                "x0": data.signal_event_time_s,
                "x1": data.signal_event_time_s,
                "y0": 0,
                "y1": 1,
                "line": {"color": "#ef4444", "width": 1.5, "dash": "dash"},
            }
        ]
        layout["annotations"] = [
            {
                "xref": "x",
                "yref": "paper",
                "x": data.signal_event_time_s,
                "y": 1,
                "text": f"信号持续置 1 · {data.signal_event_time_s:.2f}s",
                "showarrow": True,
                "arrowcolor": "#ef4444",
                "font": {"color": "#b91c1c", "size": 12},
            }
        ]
    return traces, layout


def _alarm_text(values: Sequence[float | None], event: float | None) -> str:
    alarms = []
    for name, alarm in zip(WHEEL_NAMES, values):
        if alarm is None:
            continue
        delay = "" if event is None else f"（{alarm - event:+.2f}s）"
        alarms.append(f"{name} {alarm:.2f}s{delay}")
    return "、".join(alarms) if alarms else "未报警"


def _case_page(data: CaseAnalysis, index_href: str = "index.html") -> str:
    traces, layout = build_plot_payload(data)
    event_text = (
        "未找到持续高电平"
        if data.signal_event_time_s is None
        else f"{data.signal_event_time_s:.2f} s"
    )
    factor_ranges = [
        (max(values) - min(values)) * 100.0 for values in data.phase_factors
    ]
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>0818 双算法回放 · {html.escape(data.input_path.stem)}</title>
<script src="https://cdn.plot.ly/plotly-3.7.0.min.js"></script>
<style>
:root{{--ink:#172033;--muted:#67738a;--line:#dce3ed;--bg:#f4f7fb;--blue:#1d4ed8}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 Inter,system-ui,"Segoe UI",sans-serif}}main{{max-width:1680px;margin:auto;padding:24px}}a{{color:var(--blue);text-decoration:none;font-weight:700}}header{{display:flex;justify-content:space-between;gap:20px;align-items:flex-start}}h1{{margin:3px 0 5px;font-size:28px}}.eyebrow{{color:var(--blue);font-size:11px;font-weight:800;letter-spacing:.14em}}.muted{{color:var(--muted)}}.cards{{display:grid;grid-template-columns:repeat(4,minmax(180px,1fr));gap:12px;margin:18px 0}}.card,.panel,.note{{background:white;border:1px solid var(--line);border-radius:12px}}.card{{padding:15px;border-top:3px solid #94a3b8}}.card strong,.card span{{display:block}}.card strong{{font-size:21px;margin:3px 0}}.card span,.card small{{color:var(--muted)}}.panel{{padding:8px;overflow:hidden}}.note{{padding:13px 16px;margin:12px 0;color:#475569}}#plot{{height:1450px}}@media(max-width:850px){{main{{padding:12px}}.cards{{grid-template-columns:1fr 1fr}}header{{display:block}}}}
</style></head><body><main>
<header><div><div class="eyebrow">0818 · WHEEL ONLY vs QUANT</div><h1>{html.escape(data.input_path.stem)}</h1><div class="muted">{html.escape(str(data.input_path))} · {len(data.times):,} 帧 · {data.duration_s:.2f} s</div></div><a href="{index_href}">← 返回总览</a></header>
<section class="cards"><div class="card"><span>爆胎信号持续置 1</span><strong>{event_text}</strong><small>红色虚线；原始 0/1 全量保留</small></div>
<div class="card"><span>wheel_only 首次报警</span><strong>{html.escape(_alarm_text(data.wheel_first_alarms, data.signal_event_time_s))}</strong><small>括号内为相对信号延迟</small></div>
<div class="card"><span>quant 首次报警</span><strong>{html.escape(_alarm_text(data.quant_first_alarms, data.signal_event_time_s))}</strong><small>括号内为相对信号延迟</small></div>
<div class="card"><span>48 齿相位校正峰峰值</span><strong>{' / '.join(f'{value:.2f}%' for value in factor_ranges)}</strong><small>FL / FR / RL / RR</small></div></section>
<div class="note">点线表示 candidate，粗实线表示锁存报警。末行黑线为每帧最后一个原始信号位；事件时刻采用首段至少连续 20 帧的高电平，因此不会把文件开头的短暂残留高电平当成正式事件。</div>
<section class="panel"><div id="plot"></div></section>
<script>const traces={json.dumps(traces, ensure_ascii=False, separators=(',', ':'))};const layout={json.dumps(layout, ensure_ascii=False, separators=(',', ':'))};Plotly.newPlot('plot',traces,layout,{{responsive:true,scrollZoom:true,displaylogo:false,modeBarButtonsToAdd:['drawline','eraseshape']}});</script>
</main></body></html>"""


def _status(values: Sequence[float | None], event: float | None) -> tuple[str, str]:
    alarms = [(wheel, value) for wheel, value in enumerate(values) if value is not None]
    if not alarms:
        return "miss", "未报警"
    text = _alarm_text(values, event)
    wrong = any(wheel != 3 for wheel, _ in alarms)
    early = event is not None and any(value < event for _, value in alarms)
    return ("warn" if wrong or early else "hit"), text


def _index_page(analyses: Sequence[CaseAnalysis]) -> str:
    rows = []
    for data in analyses:
        event = data.signal_event_time_s
        wheel_class, wheel_text = _status(data.wheel_first_alarms, event)
        quant_class, quant_text = _status(data.quant_first_alarms, event)
        rows.append(
            f"<tr><td><a href='{html.escape(data.input_path.stem)}.html'>{html.escape(data.input_path.stem)}</a></td>"
            f"<td>{len(data.times):,}</td><td>{data.duration_s:.2f}s</td>"
            f"<td>{'—' if event is None else f'{event:.2f}s'}</td>"
            f"<td><span class='badge {wheel_class}'>{html.escape(wheel_text)}</span></td>"
            f"<td><span class='badge {quant_class}'>{html.escape(quant_text)}</span></td>"
            f"<td><a class='button' href='{html.escape(data.input_path.stem)}.html'>打开回放</a></td></tr>"
        )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>0818 爆胎数据双算法 Display</title>
<style>:root{{--ink:#172033;--muted:#67738a;--line:#dce3ed;--bg:#f4f7fb;--blue:#1d4ed8}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 Inter,system-ui,"Segoe UI",sans-serif}}main{{max-width:1320px;margin:auto;padding:32px}}h1{{margin:2px 0 8px;font-size:31px}}.eyebrow{{color:var(--blue);font-size:11px;font-weight:800;letter-spacing:.14em}}.muted{{color:var(--muted)}}.panel,.note{{background:#fff;border:1px solid var(--line);border-radius:12px}}.panel{{padding:18px;margin-top:22px;overflow:auto}}.note{{padding:16px 18px;margin-top:14px;color:#475569}}table{{width:100%;border-collapse:collapse}}th,td{{padding:12px 10px;border-bottom:1px solid #e7ebf1;text-align:left;white-space:nowrap}}th{{color:var(--muted);font-size:12px}}a{{color:var(--blue);text-decoration:none;font-weight:700}}.button{{display:inline-block;background:var(--blue);color:white;border-radius:7px;padding:6px 10px}}.badge{{display:inline-block;border-radius:999px;padding:4px 9px;font-weight:700}}.hit{{background:#dcfce7;color:#166534}}.miss{{background:#f1f5f9;color:#64748b}}.warn{{background:#fef3c7;color:#92400e}}@media(max-width:700px){{main{{padding:16px}}}}</style></head><body><main>
<div class="eyebrow">NEW DATA · 2026-08-18</div><h1>0818 爆胎数据双算法 Display</h1><p class="muted">仅对比 wheel_speed_only 与 quant。爆胎真值直接读取每帧第 5 行的最后一个值。</p>
<section class="panel"><table><thead><tr><th>记录</th><th>帧数</th><th>时长</th><th>信号时刻</th><th>wheel_only</th><th>quant</th><th></th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
<section class="note"><b>数据口径：</b>前四行按 FL / FR / RL / RR 解析齿时间戳，100 Hz 重建轮速；每条记录离线学习 48 齿静态相位误差后，再以当前默认参数完整因果回放两套算法。表中括号是相对持续爆胎信号上升沿的时间差。</section>
</main></body></html>"""


def build_display(input_dir: Path, output_dir: Path) -> list[CaseAnalysis]:
    paths = sorted(input_dir.glob("*.txt"))
    if not paths:
        raise ValueError(f"目录内没有 txt 数据：{input_dir}")
    analyses = [analyze_file(path) for path in paths]
    output_dir.mkdir(parents=True, exist_ok=True)
    for data in analyses:
        (output_dir / f"{data.input_path.stem}.html").write_text(
            _case_page(data), encoding="utf-8"
        )
    (output_dir / "index.html").write_text(
        _index_page(analyses), encoding="utf-8"
    )
    return analyses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the 0818 wheel-only/quant display.")
    parser.add_argument("--input-dir", type=Path, default=Path("0818"))
    parser.add_argument("--output-dir", type=Path, default=Path("0818/display"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyses = build_display(args.input_dir, args.output_dir)
    print(f"wrote {args.output_dir / 'index.html'}")
    for data in analyses:
        print(
            f"{data.input_path.name}: signal={data.signal_event_time_s}, "
            f"wheel_only={data.wheel_first_alarms}, quant={data.quant_first_alarms}"
        )


if __name__ == "__main__":
    main()
