from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .detector import (
    WHEEL_NAMES,
    QuantBlowoutConfig,
    QuantBlowoutDetector,
    QuantFrame,
)


DEFAULT_WHEEL_COLUMNS = tuple(
    f"wheel{index}_corrected_rad_s" for index in range(4)
)
WHEEL_COLORS = ("#2563eb", "#f59e0b", "#16a34a", "#dc2626")
FACTOR_NAMES = ("左右 s", "前后 a", "对角 d")
FACTOR_COLORS = ("#0891b2", "#7c3aed", "#db2777")
PLOT_CONFIG = {
    "responsive": True,
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToAdd": ["drawline", "eraseshape"],
}


@dataclass
class QuantDisplayData:
    input_path: Path
    times: list[float]
    wheels: list[list[float]]
    factor_residuals_pct: list[list[float]]
    factor_edges_pct: list[list[float]]
    physical_levels_pct: list[list[float]]
    physical_edges_pct: list[list[float]]
    shock_z_scores: list[list[float]]
    level_z_scores: list[list[float]]
    shock_isolation: list[list[float]]
    level_isolation: list[list[float]]
    cusum_scores: list[list[float]]
    persistence_scores: list[list[float]]
    risk_scores: list[list[float]]
    states: list[list[str]]
    alarms: list[list[bool]]
    new_blowouts: list[list[bool]]
    speed_valid: list[bool]
    warmed_up: list[bool]
    leading_wheels: list[int | None]
    leading_margins: list[float]
    first_alarm_times: list[float | None]
    onset_times: list[float | None]


def analyze_csv(
    input_path: Path,
    cfg: QuantBlowoutConfig | None = None,
    time_column: str = "time_s",
    wheel_columns: Sequence[str] = DEFAULT_WHEEL_COLUMNS,
    start_time_s: float | None = None,
    end_time_s: float | None = None,
) -> QuantDisplayData:
    """Replay a CSV causally and retain the requested display window.

    Samples before ``start_time_s`` are still fed through the detector so its
    online baseline and state at the left edge match a full-file replay.
    """
    if len(wheel_columns) != 4:
        raise ValueError("four wheel columns are required")
    if not input_path.is_file():
        raise ValueError(f"input CSV does not exist: {input_path}")

    detector = QuantBlowoutDetector(cfg)
    times: list[float] = []
    wheels = [[] for _ in WHEEL_NAMES]
    factor_residuals = [[] for _ in FACTOR_NAMES]
    factor_edges = [[] for _ in FACTOR_NAMES]
    physical_levels = [[] for _ in WHEEL_NAMES]
    physical_edges = [[] for _ in WHEEL_NAMES]
    shock_z = [[] for _ in WHEEL_NAMES]
    level_z = [[] for _ in WHEEL_NAMES]
    shock_isolation = [[] for _ in WHEEL_NAMES]
    level_isolation = [[] for _ in WHEEL_NAMES]
    cusum = [[] for _ in WHEEL_NAMES]
    persistence = [[] for _ in WHEEL_NAMES]
    risks = [[] for _ in WHEEL_NAMES]
    states = [[] for _ in WHEEL_NAMES]
    alarms = [[] for _ in WHEEL_NAMES]
    new_blowouts = [[] for _ in WHEEL_NAMES]
    speed_valid: list[bool] = []
    warmed_up: list[bool] = []
    leading_wheels: list[int | None] = []
    leading_margins: list[float] = []
    first_alarm_times: list[float | None] = [None] * 4
    onset_times: list[float | None] = [None] * 4

    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {input_path}")
        required = [time_column, *wheel_columns]
        missing = [column for column in required if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"missing CSV columns: {missing}")

        for row in reader:
            t_sec = float(row[time_column])
            if end_time_s is not None and t_sec > end_time_s:
                break
            wheel_values = [float(row[column]) for column in wheel_columns]
            result = detector.push(QuantFrame.from_sequences(t_sec, wheel_values))
            for wheel, is_new in enumerate(result.new_blowouts):
                if is_new and first_alarm_times[wheel] is None:
                    first_alarm_times[wheel] = t_sec
                    onset_times[wheel] = result.estimated_onset_times_s[wheel]
            if start_time_s is not None and t_sec < start_time_s:
                continue

            times.append(t_sec)
            speed_valid.append(result.speed_valid)
            warmed_up.append(result.warmed_up)
            leading_wheels.append(result.leading_wheel)
            leading_margins.append(result.leading_margin)
            for factor in range(3):
                factor_residuals[factor].append(
                    result.factor_residuals[factor] * 100.0
                )
                factor_edges[factor].append(result.factor_edges[factor] * 100.0)
            for wheel in range(4):
                wheels[wheel].append(wheel_values[wheel])
                physical_levels[wheel].append(result.physical_levels[wheel] * 100.0)
                physical_edges[wheel].append(result.physical_edges[wheel] * 100.0)
                shock_z[wheel].append(result.shock_z_scores[wheel])
                level_z[wheel].append(result.level_z_scores[wheel])
                shock_isolation[wheel].append(result.shock_isolation[wheel])
                level_isolation[wheel].append(result.level_isolation[wheel])
                cusum[wheel].append(result.cusum_scores[wheel])
                persistence[wheel].append(result.persistence_scores[wheel])
                risks[wheel].append(result.risk_scores[wheel])
                states[wheel].append(result.states[wheel])
                alarms[wheel].append(result.blowout_alarms[wheel])
                new_blowouts[wheel].append(result.new_blowouts[wheel])

    if not times:
        raise ValueError("display window contains no samples")
    return QuantDisplayData(
        input_path=input_path,
        times=times,
        wheels=wheels,
        factor_residuals_pct=factor_residuals,
        factor_edges_pct=factor_edges,
        physical_levels_pct=physical_levels,
        physical_edges_pct=physical_edges,
        shock_z_scores=shock_z,
        level_z_scores=level_z,
        shock_isolation=shock_isolation,
        level_isolation=level_isolation,
        cusum_scores=cusum,
        persistence_scores=persistence,
        risk_scores=risks,
        states=states,
        alarms=alarms,
        new_blowouts=new_blowouts,
        speed_valid=speed_valid,
        warmed_up=warmed_up,
        leading_wheels=leading_wheels,
        leading_margins=leading_margins,
        first_alarm_times=first_alarm_times,
        onset_times=onset_times,
    )


def _line(
    x: Sequence[float],
    y: Sequence[float],
    name: str,
    color: str,
    *,
    dash: str = "solid",
    showlegend: bool = False,
    legendgroup: str | None = None,
    hovertemplate: str | None = None,
    customdata: object | None = None,
) -> go.Scattergl:
    return go.Scattergl(
        x=x,
        y=y,
        name=name,
        line={"color": color, "width": 1.2, "dash": dash},
        showlegend=showlegend,
        legendgroup=legendgroup,
        hovertemplate=hovertemplate or f"{name}=%{{y:.3f}}<extra></extra>",
        customdata=customdata,
    )


def build_figure(
    data: QuantDisplayData,
    cfg: QuantBlowoutConfig,
    event_time_s: float | None = None,
    title: str | None = None,
) -> go.Figure:
    figure = make_subplots(
        rows=7,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=(0.19, 0.13, 0.16, 0.15, 0.13, 0.15, 0.09),
        subplot_titles=(
            "四轮轮速",
            "Hadamard 因子残差（实线）与瞬时边沿（虚线）",
            "逐轮物理指纹投影（实线=持续偏离，虚线=瞬时边沿）",
            "协方差标准化匹配分（实线=shock，虚线=level）",
            "轮位隔离度（实线=shock，虚线=level）",
            "风险分（hover 同时显示 CUSUM / persistence）",
            "状态（浅色=候选，深色=锁存报警）",
        ),
    )

    for factor, (name, color) in enumerate(zip(FACTOR_NAMES, FACTOR_COLORS)):
        figure.add_trace(
            _line(
                data.times,
                data.factor_residuals_pct[factor],
                name,
                color,
                showlegend=True,
                legendgroup=f"factor-{factor}",
                hovertemplate=f"{name} residual=%{{y:.4f}}%<extra></extra>",
            ),
            row=2,
            col=1,
        )
        figure.add_trace(
            _line(
                data.times,
                data.factor_edges_pct[factor],
                f"{name} edge",
                color,
                dash="dash",
                legendgroup=f"factor-{factor}",
                hovertemplate=f"{name} edge=%{{y:.4f}}%<extra></extra>",
            ),
            row=2,
            col=1,
        )

    dt_values = [
        right - left
        for left, right in zip(data.times, data.times[1:])
        if right > left
    ]
    bar_width = median(dt_values) * 1.05 if dt_values else 0.01
    for wheel, (name, color) in enumerate(zip(WHEEL_NAMES, WHEEL_COLORS)):
        wheel_custom = list(
            zip(
                data.states[wheel],
                data.risk_scores[wheel],
                data.shock_z_scores[wheel],
                data.level_z_scores[wheel],
                data.speed_valid,
                data.warmed_up,
            )
        )
        figure.add_trace(
            _line(
                data.times,
                data.wheels[wheel],
                name,
                color,
                showlegend=True,
                legendgroup=name,
                customdata=wheel_custom,
                hovertemplate=(
                    f"{name}<br>t=%{{x:.3f}} s<br>轮速=%{{y:.4f}} rad/s"
                    "<br>状态=%{customdata[0]}<br>风险=%{customdata[1]:.1f}"
                    "<br>shock z=%{customdata[2]:.2f}<br>level z=%{customdata[3]:.2f}"
                    "<br>速度有效=%{customdata[4]}<br>已预热=%{customdata[5]}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        for row, solid, dashed, solid_label, dashed_label in (
            (3, data.physical_levels_pct[wheel], data.physical_edges_pct[wheel], "level", "edge"),
            (4, data.shock_z_scores[wheel], data.level_z_scores[wheel], "shock z", "level z"),
            (5, data.shock_isolation[wheel], data.level_isolation[wheel], "shock isolation", "level isolation"),
        ):
            figure.add_trace(
                _line(
                    data.times,
                    solid,
                    f"{name} {solid_label}",
                    color,
                    legendgroup=name,
                    hovertemplate=f"{name} {solid_label}=%{{y:.3f}}<extra></extra>",
                ),
                row=row,
                col=1,
            )
            figure.add_trace(
                _line(
                    data.times,
                    dashed,
                    f"{name} {dashed_label}",
                    color,
                    dash="dash",
                    legendgroup=name,
                    hovertemplate=f"{name} {dashed_label}=%{{y:.3f}}<extra></extra>",
                ),
                row=row,
                col=1,
            )

        risk_custom = list(
            zip(
                data.cusum_scores[wheel],
                data.persistence_scores[wheel],
                data.states[wheel],
                data.leading_wheels,
                data.leading_margins,
            )
        )
        figure.add_trace(
            _line(
                data.times,
                data.risk_scores[wheel],
                f"{name} risk",
                color,
                legendgroup=name,
                customdata=risk_custom,
                hovertemplate=(
                    f"{name} risk=%{{y:.1f}}"
                    "<br>CUSUM=%{customdata[0]:.2f}"
                    "<br>persistence=%{customdata[1]:.2f}"
                    "<br>状态=%{customdata[2]}"
                    "<br>领先轮索引=%{customdata[3]}"
                    "<br>领先差=%{customdata[4]:.1f}<extra></extra>"
                ),
            ),
            row=6,
            col=1,
        )
        candidate_height = [
            0.38 if state == "candidate" else 0.0 for state in data.states[wheel]
        ]
        alarm_height = [0.78 if active else 0.0 for active in data.alarms[wheel]]
        figure.add_trace(
            go.Bar(
                x=data.times,
                y=candidate_height,
                base=wheel,
                width=bar_width,
                name=f"{name} candidate",
                marker={"color": color, "opacity": 0.30, "line": {"width": 0}},
                legendgroup=name,
                showlegend=False,
                hovertemplate=f"{name} 候选<extra></extra>",
            ),
            row=7,
            col=1,
        )
        figure.add_trace(
            go.Bar(
                x=data.times,
                y=alarm_height,
                base=wheel,
                width=bar_width,
                name=f"{name} alarm",
                marker={"color": color, "opacity": 0.90, "line": {"width": 0}},
                legendgroup=name,
                showlegend=False,
                hovertemplate=f"{name} 锁存报警<extra></extra>",
            ),
            row=7,
            col=1,
        )

    threshold_specs = (
        (3, cfg.min_physical_edge * 100.0, "edge触发", "#0f766e"),
        (3, cfg.min_physical_persistence * 100.0, "level持续", "#64748b"),
        (4, cfg.shock_trigger_z, "shock触发", "#0f766e"),
        (4, cfg.min_median_level_z, "level确认", "#64748b"),
        (5, cfg.shock_isolation_z, "shock隔离", "#0f766e"),
        (5, cfg.min_level_isolation_z, "level隔离", "#64748b"),
        (6, cfg.min_median_risk, "中位风险", "#64748b"),
        (6, cfg.min_peak_risk, "峰值风险", "#b91c1c"),
    )
    for row, threshold, label, color in threshold_specs:
        figure.add_hline(
            y=threshold,
            line_dash="dot",
            line_color=color,
            annotation_text=label,
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
    for wheel, alarm_time in enumerate(data.first_alarm_times):
        if alarm_time is None or not data.times[0] <= alarm_time <= data.times[-1]:
            continue
        figure.add_vline(
            x=alarm_time,
            line_width=1.5,
            line_dash="dot",
            line_color=WHEEL_COLORS[wheel],
            annotation_text=f"{WHEEL_NAMES[wheel]}报警",
            annotation_position="top right",
        )

    figure.update_yaxes(title_text="rad/s", row=1, col=1)
    figure.update_yaxes(title_text="%", row=2, col=1)
    figure.update_yaxes(title_text="%", row=3, col=1)
    figure.update_yaxes(title_text="z", row=4, col=1)
    figure.update_yaxes(title_text="z差", row=5, col=1)
    figure.update_yaxes(title_text="0–100", range=(-2, 102), row=6, col=1)
    figure.update_yaxes(
        tickvals=[index + 0.38 for index in range(4)],
        ticktext=WHEEL_NAMES,
        range=(-0.1, 3.9),
        row=7,
        col=1,
    )
    figure.update_xaxes(
        title_text="时间 / s",
        rangeslider={"visible": True},
        row=7,
        col=1,
    )
    figure.update_layout(
        title=title or f"四轮轮速量化爆胎检测 — {data.input_path.name}",
        template="plotly_white",
        height=1600,
        hovermode="x unified",
        dragmode="pan",
        barmode="overlay",
        legend={"orientation": "h", "y": 1.035, "x": 0.0},
        margin={"l": 72, "r": 48, "t": 120, "b": 60},
    )
    return figure


def write_display_html(
    data: QuantDisplayData,
    output_path: Path,
    cfg: QuantBlowoutConfig,
    event_time_s: float | None = None,
    title: str | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = build_figure(data, cfg, event_time_s=event_time_s, title=title)
    figure.write_html(
        output_path,
        include_plotlyjs="cdn",
        full_html=True,
        config=PLOT_CONFIG,
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an interactive display for the quantitative detector."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--event-time", type=float)
    parser.add_argument("--window-before", type=float, default=5.0)
    parser.add_argument("--window-after", type=float, default=5.0)
    parser.add_argument("--time-column", default="time_s")
    parser.add_argument("--wheel-columns", nargs=4, default=DEFAULT_WHEEL_COLUMNS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = QuantBlowoutConfig()
    start = None if args.event_time is None else args.event_time - args.window_before
    end = None if args.event_time is None else args.event_time + args.window_after
    data = analyze_csv(
        args.input,
        cfg=cfg,
        time_column=args.time_column,
        wheel_columns=args.wheel_columns,
        start_time_s=start,
        end_time_s=end,
    )
    write_display_html(data, args.output, cfg, event_time_s=args.event_time)
    alarms = [
        f"{name}={time:.3f}s (onset={data.onset_times[index]:.3f}s)"
        if data.onset_times[index] is not None
        else f"{name}={time:.3f}s"
        for index, (name, time) in enumerate(zip(WHEEL_NAMES, data.first_alarm_times))
        if time is not None
    ]
    print(f"wrote {args.output}")
    print("first alarms: " + (", ".join(alarms) if alarms else "none"))


if __name__ == "__main__":
    main()
