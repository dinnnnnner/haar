from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .pressure_fusion_detector import (
    WHEEL_NAMES,
    PressureFusionBlowoutDetector,
    PressureFusionConfig,
    PressureFusionFrame,
)


WHEEL_COLORS = ("#2563eb", "#f59e0b", "#16a34a", "#dc2626")
PLOT_CONFIG = {
    "responsive": True,
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToAdd": ["drawline", "eraseshape"],
}


@dataclass(frozen=True)
class SuspectInterval:
    wheel_index: int
    start_s: float
    end_s: float
    confirmed: bool
    peak_individual_gain_pct: float | None
    peak_diagonal_gain_pct: float | None

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


@dataclass(frozen=True)
class ScanResult:
    start_s: float
    end_s: float
    frames: int
    suspects: tuple[SuspectInterval, ...]
    first_alarm_times: tuple[float | None, float | None, float | None, float | None]


@dataclass
class WindowData:
    times: list[float]
    wheels: list[list[float]]
    gains: list[list[float | None]]
    edges: list[list[float | None]]
    diagonal_gain: list[float | None]
    diagonal_edge: list[float | None]
    candidates: list[list[bool]]
    alarms: list[list[bool]]


def _finite_percent(value: float) -> float | None:
    return value * 100.0 if math.isfinite(value) else None


def _pressure_values(
    pressure_indices: Sequence[int],
    event_time_s: float | None,
    t_sec: float,
) -> list[bool | None]:
    values: list[bool | None] = [None] * 4
    for index in pressure_indices:
        values[index] = False
    if event_time_s is not None and 3 in pressure_indices and t_sec >= event_time_s:
        values[3] = True
    return values


def scan_csv(
    input_path: Path,
    pressure_indices: Sequence[int],
    cfg: PressureFusionConfig | None = None,
    event_time_s: float | None = None,
) -> ScanResult:
    detector = PressureFusionBlowoutDetector(cfg)
    starts: list[float | None] = [None] * 4
    peak_individual: list[float | None] = [None] * 4
    peak_diagonal: list[float | None] = [None] * 4
    first_alarms: list[float | None] = [None] * 4
    suspects: list[SuspectInterval] = []
    first_time: float | None = None
    last_time: float | None = None
    frames = 0

    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            t_sec = float(row["time_s"])
            wheels = [float(row[f"wheel{i}_corrected_rad_s"]) for i in range(4)]
            result = detector.push(
                PressureFusionFrame.from_sequences(
                    t_sec,
                    wheels,
                    _pressure_values(pressure_indices, event_time_s, t_sec),
                )
            )
            first_time = t_sec if first_time is None else first_time
            last_time = t_sec
            frames += 1
            for index in range(4):
                active = result.candidates[index]
                if active and starts[index] is None:
                    starts[index] = t_sec
                    peak_individual[index] = _finite_percent(
                        result.individual_gains[index]
                    )
                    peak_diagonal[index] = _finite_percent(result.diagonal_gain)
                elif active:
                    individual = _finite_percent(result.individual_gains[index])
                    diagonal = _finite_percent(result.diagonal_gain)
                    if individual is not None:
                        peak_individual[index] = max(
                            peak_individual[index] or individual, individual
                        )
                    if diagonal is not None:
                        peak_diagonal[index] = max(
                            peak_diagonal[index] or diagonal, diagonal
                        )
                elif starts[index] is not None:
                    suspects.append(
                        SuspectInterval(
                            wheel_index=index,
                            start_s=starts[index],
                            end_s=t_sec,
                            confirmed=(
                                result.new_blowouts[index]
                                and result.alarm_sources[index]
                                == "wheel_speed_confirmed"
                            ),
                            peak_individual_gain_pct=peak_individual[index],
                            peak_diagonal_gain_pct=peak_diagonal[index],
                        )
                    )
                    starts[index] = None
                    peak_individual[index] = None
                    peak_diagonal[index] = None
                if result.new_blowouts[index] and first_alarms[index] is None:
                    first_alarms[index] = t_sec

    if first_time is None or last_time is None:
        raise ValueError("CSV 中没有数据")
    for index, start in enumerate(starts):
        if start is not None:
            suspects.append(
                SuspectInterval(
                    index,
                    start,
                    last_time,
                    False,
                    peak_individual[index],
                    peak_diagonal[index],
                )
            )
    suspects.sort(key=lambda item: (item.start_s, item.wheel_index))
    return ScanResult(
        first_time,
        last_time,
        frames,
        tuple(suspects),
        tuple(first_alarms),  # type: ignore[arg-type]
    )


def analyze_window(
    input_path: Path,
    pressure_indices: Sequence[int],
    start_s: float,
    end_s: float,
    cfg: PressureFusionConfig | None = None,
    event_time_s: float | None = None,
) -> WindowData:
    detector = PressureFusionBlowoutDetector(cfg)
    data = WindowData(
        [],
        [[] for _ in range(4)],
        [[] for _ in range(4)],
        [[] for _ in range(4)],
        [],
        [],
        [[] for _ in range(4)],
        [[] for _ in range(4)],
    )
    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            t_sec = float(row["time_s"])
            if t_sec > end_s:
                break
            wheels = [float(row[f"wheel{i}_corrected_rad_s"]) for i in range(4)]
            result = detector.push(
                PressureFusionFrame.from_sequences(
                    t_sec,
                    wheels,
                    _pressure_values(pressure_indices, event_time_s, t_sec),
                )
            )
            if t_sec < start_s:
                continue
            data.times.append(t_sec)
            for index in range(4):
                data.wheels[index].append(wheels[index])
                data.gains[index].append(
                    _finite_percent(result.individual_gains[index])
                )
                data.edges[index].append(
                    _finite_percent(result.individual_edges[index])
                )
                data.candidates[index].append(result.candidates[index])
                data.alarms[index].append(result.blowout_alarms[index])
            data.diagonal_gain.append(_finite_percent(result.diagonal_gain))
            data.diagonal_edge.append(_finite_percent(result.diagonal_edge))
    if not data.times:
        raise ValueError("所选窗口内没有数据")
    return data


def build_figure(
    data: WindowData,
    cfg: PressureFusionConfig,
    suspects: Sequence[SuspectInterval] = (),
    event_time_s: float | None = None,
    title: str | None = None,
) -> go.Figure:
    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        row_heights=(0.38, 0.23, 0.23, 0.16),
        subplot_titles=(
            "四轮校正轮速",
            "逐轮相对增益",
            "上升沿证据（实线=逐轮，虚线=对角）",
            "疑似候选与锁存报警",
        ),
    )
    dt = [b - a for a, b in zip(data.times, data.times[1:]) if b > a]
    bar_width = median(dt) * 1.05 if dt else 0.01
    for index, (name, color) in enumerate(zip(WHEEL_NAMES, WHEEL_COLORS)):
        figure.add_trace(
            go.Scattergl(
                x=data.times,
                y=data.wheels[index],
                name=name,
                line={"color": color, "width": 1.25},
                hovertemplate=f"{name}<br>t=%{{x:.3f}} s<br>轮速=%{{y:.4f}} rad/s<extra></extra>",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scattergl(
                x=data.times,
                y=data.gains[index],
                name=f"{name} gain",
                legendgroup=name,
                showlegend=False,
                line={"color": color, "width": 1.1},
                hovertemplate=f"{name} gain=%{{y:.3f}}%<extra></extra>",
            ),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Scattergl(
                x=data.times,
                y=data.edges[index],
                name=f"{name} edge",
                legendgroup=name,
                showlegend=False,
                line={"color": color, "width": 1.1},
                hovertemplate=f"{name} edge=%{{y:.3f}}%<extra></extra>",
            ),
            row=3,
            col=1,
        )
        figure.add_trace(
            go.Bar(
                x=data.times,
                y=[0.32 if active else 0 for active in data.candidates[index]],
                base=index,
                width=bar_width,
                marker={"color": color, "opacity": 0.32, "line": {"width": 0}},
                name=f"{name} 疑似",
                legendgroup=name,
                showlegend=False,
                hovertemplate=f"{name} 疑似<extra></extra>",
            ),
            row=4,
            col=1,
        )
        figure.add_trace(
            go.Bar(
                x=data.times,
                y=[0.72 if active else 0 for active in data.alarms[index]],
                base=index,
                width=bar_width,
                marker={"color": color, "opacity": 0.88, "line": {"width": 0}},
                name=f"{name} 报警",
                legendgroup=name,
                showlegend=False,
                hovertemplate=f"{name} 报警<extra></extra>",
            ),
            row=4,
            col=1,
        )

    figure.add_trace(
        go.Scattergl(
            x=data.times,
            y=data.diagonal_gain,
            name="diagonal gain",
            line={"color": "#7c3aed", "width": 1.3, "dash": "dash"},
            hovertemplate="diagonal gain=%{y:.3f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scattergl(
            x=data.times,
            y=data.diagonal_edge,
            name="diagonal edge",
            line={"color": "#7c3aed", "width": 1.3, "dash": "dash"},
            hovertemplate="diagonal edge=%{y:.3f}%<extra></extra>",
        ),
        row=3,
        col=1,
    )
    if cfg.min_individual_edge == cfg.min_diagonal_edge:
        figure.add_hline(
            y=cfg.min_individual_edge * 100,
            line_dash="dot",
            line_color="#64748b",
            annotation_text="逐轮/对角候选门限",
            row=3,
            col=1,
        )
    else:
        figure.add_hline(
            y=cfg.min_individual_edge * 100,
            line_dash="dot",
            line_color="#64748b",
            annotation_text="逐轮候选门限",
            row=3,
            col=1,
        )
        figure.add_hline(
            y=cfg.min_diagonal_edge * 100,
            line_dash="dot",
            line_color="#7c3aed",
            annotation_text="对角候选门限",
            row=3,
            col=1,
        )
    for interval in suspects:
        if interval.end_s < data.times[0] or interval.start_s > data.times[-1]:
            continue
        for row in range(1, 5):
            figure.add_vrect(
                x0=interval.start_s,
                x1=interval.end_s,
                fillcolor=WHEEL_COLORS[interval.wheel_index],
                opacity=0.14 if not interval.confirmed else 0.22,
                line_color=WHEEL_COLORS[interval.wheel_index],
                line_width=1 if not interval.confirmed else 2,
                line_dash="dot" if not interval.confirmed else "solid",
                layer="below",
                row=row,
                col=1,
            )
    if event_time_s is not None:
        figure.add_vline(
            x=event_time_s,
            line_width=2,
            line_dash="dash",
            line_color="#7c3aed",
            annotation_text="人工事件",
        )
    figure.update_yaxes(title_text="rad/s", row=1, col=1)
    figure.update_yaxes(title_text="%", row=2, col=1)
    figure.update_yaxes(title_text="%", row=3, col=1)
    figure.update_yaxes(
        tickvals=[index + 0.36 for index in range(4)],
        ticktext=WHEEL_NAMES,
        range=(-0.1, 3.85),
        row=4,
        col=1,
    )
    figure.update_xaxes(
        title_text="时间 / s",
        rangeslider={"visible": True},
        row=4,
        col=1,
    )
    figure.update_layout(
        title=title,
        template="plotly_white",
        height=1050,
        hovermode="x unified",
        dragmode="pan",
        barmode="overlay",
        legend={"orientation": "h", "y": 1.045, "x": 0},
        margin={"l": 70, "r": 35, "t": 105, "b": 55},
        uirevision="pressure-fusion-display",
    )
    return figure
