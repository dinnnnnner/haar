from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .detector import (
    REFERENCE_MODES,
    WHEEL_NAMES,
    WaveletShapeBlowoutDetector,
    WaveletShapeConfig,
    WheelFrame,
)


DEFAULT_WHEEL_COLUMNS = tuple(
    f"wheel{index}_corrected_rad_s" for index in range(4)
)
WHEEL_COLORS = ("#2563eb", "#f59e0b", "#16a34a", "#dc2626")


@dataclass
class DisplayData:
    input_path: Path
    times: list[float]
    wheels: list[list[float]]
    gains: list[list[float]]
    haar: list[list[float]]
    states: list[list[str]]
    alarms: list[list[bool]]
    new_blowouts: list[list[bool]]
    reference_sources: list[list[str]]
    first_alarm_times: list[float | None]
    onset_times: list[float | None]


def _parse_optional_bool(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in {"", "na", "nan", "none", "unknown"}:
        return None
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"cannot parse normal signal value {value!r}")


def analyze_csv(
    input_path: Path,
    cfg: WaveletShapeConfig | None = None,
    time_column: str = "time_s",
    wheel_columns: Sequence[str] = DEFAULT_WHEEL_COLUMNS,
    normal_columns: Sequence[str | None] = (None, None, None, None),
    start_time_s: float | None = None,
    end_time_s: float | None = None,
) -> DisplayData:
    if len(wheel_columns) != 4 or len(normal_columns) != 4:
        raise ValueError("four wheel columns and four normal columns are required")
    detector = WaveletShapeBlowoutDetector(cfg)
    times: list[float] = []
    wheels = [[] for _ in WHEEL_NAMES]
    gains = [[] for _ in WHEEL_NAMES]
    haar = [[] for _ in WHEEL_NAMES]
    states = [[] for _ in WHEEL_NAMES]
    alarms = [[] for _ in WHEEL_NAMES]
    new_blowouts = [[] for _ in WHEEL_NAMES]
    references = [[] for _ in WHEEL_NAMES]
    first_alarm_times: list[float | None] = [None] * 4
    onset_times: list[float | None] = [None] * 4

    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {input_path}")
        required = [time_column, *wheel_columns]
        required.extend(column for column in normal_columns if column)
        missing = [column for column in required if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"missing CSV columns: {missing}")

        for row in reader:
            t_sec = float(row[time_column])
            if end_time_s is not None and t_sec > end_time_s:
                break
            wheel_values = [float(row[column]) for column in wheel_columns]
            normal_values = [
                None if column is None else _parse_optional_bool(row[column])
                for column in normal_columns
            ]
            result = detector.push(
                WheelFrame.from_sequences(t_sec, wheel_values, normal_values)
            )
            for index, is_new in enumerate(result.new_blowouts):
                if is_new and first_alarm_times[index] is None:
                    first_alarm_times[index] = t_sec
                    onset_times[index] = result.estimated_onset_times_s[index]

            if start_time_s is not None and t_sec < start_time_s:
                continue
            times.append(t_sec)
            for index in range(4):
                wheels[index].append(wheel_values[index])
                gains[index].append(result.normalized_gains[index] * 100.0)
                haar[index].append(result.haar_coefficients[index] * 100.0)
                states[index].append(result.states[index])
                alarms[index].append(result.blowout_alarms[index])
                new_blowouts[index].append(result.new_blowouts[index])
                references[index].append(result.reference_sources[index])

    if not times:
        raise ValueError("display window contains no samples")
    return DisplayData(
        input_path=input_path,
        times=times,
        wheels=wheels,
        gains=gains,
        haar=haar,
        states=states,
        alarms=alarms,
        new_blowouts=new_blowouts,
        reference_sources=references,
        first_alarm_times=first_alarm_times,
        onset_times=onset_times,
    )


def build_figure(
    data: DisplayData,
    cfg: WaveletShapeConfig,
    event_time_s: float | None = None,
    title: str | None = None,
) -> go.Figure:
    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=(0.34, 0.24, 0.24, 0.18),
        subplot_titles=(
            "四轮轮速",
            "相对正常基线增益",
            "因果 Haar 小波系数",
            "逐轮锁存报警",
        ),
    )
    for index, (name, color) in enumerate(zip(WHEEL_NAMES, WHEEL_COLORS)):
        custom = list(zip(data.states[index], data.reference_sources[index]))
        hover = (
            f"{name}<br>t=%{{x:.3f}} s<br>轮速=%{{y:.4f}} rad/s"
            "<br>状态=%{customdata[0]}<br>参考=%{customdata[1]}<extra></extra>"
        )
        figure.add_trace(
            go.Scattergl(
                x=data.times,
                y=data.wheels[index],
                name=name,
                line={"color": color, "width": 1.2},
                customdata=custom,
                hovertemplate=hover,
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scattergl(
                x=data.times,
                y=data.gains[index],
                name=f"{name} gain",
                line={"color": color, "width": 1.1},
                legendgroup=name,
                showlegend=False,
                hovertemplate=f"{name} gain=%{{y:.3f}}%<extra></extra>",
            ),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Scattergl(
                x=data.times,
                y=data.haar[index],
                name=f"{name} Haar",
                line={"color": color, "width": 1.1},
                legendgroup=name,
                showlegend=False,
                hovertemplate=f"{name} Haar=%{{y:.3f}}%<extra></extra>",
            ),
            row=3,
            col=1,
        )
        alarm_y = [index + (0.72 if active else 0.0) for active in data.alarms[index]]
        figure.add_trace(
            go.Scattergl(
                x=data.times,
                y=alarm_y,
                name=f"{name} alarm",
                mode="lines",
                line={"color": color, "width": 2, "shape": "hv"},
                legendgroup=name,
                showlegend=False,
                hovertemplate=f"{name} alarm<extra></extra>",
            ),
            row=4,
            col=1,
        )

    figure.add_hline(
        y=cfg.min_steady_gain * 100.0,
        line_dash="dot",
        line_color="#64748b",
        annotation_text="持续增益门限",
        row=2,
        col=1,
    )
    figure.add_hline(
        y=cfg.min_rise_coefficient * 100.0,
        line_dash="dot",
        line_color="#16a34a",
        annotation_text="上升沿门限",
        row=3,
        col=1,
    )
    figure.add_hline(
        y=-cfg.min_fall_coefficient * 100.0,
        line_dash="dot",
        line_color="#dc2626",
        annotation_text="下降沿门限",
        row=3,
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
    for index, alarm_time in enumerate(data.first_alarm_times):
        if alarm_time is None or alarm_time < data.times[0] or alarm_time > data.times[-1]:
            continue
        figure.add_vline(
            x=alarm_time,
            line_width=1.5,
            line_dash="dot",
            line_color=WHEEL_COLORS[index],
            annotation_text=f"{WHEEL_NAMES[index]}报警",
            annotation_position="top right",
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
    figure.update_xaxes(title_text="时间 / s", row=4, col=1)
    figure.update_layout(
        title=title or f"小波形态爆胎检测 — {data.input_path.name}",
        template="plotly_white",
        height=1050,
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.04, "x": 0.0},
        margin={"l": 70, "r": 40, "t": 110, "b": 60},
    )
    return figure


def write_display_html(
    data: DisplayData,
    output_path: Path,
    cfg: WaveletShapeConfig,
    event_time_s: float | None = None,
    title: str | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = build_figure(data, cfg, event_time_s=event_time_s, title=title)
    figure.write_html(
        output_path,
        include_plotlyjs="cdn",
        full_html=True,
        config={"responsive": True, "scrollZoom": True, "displaylogo": False},
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a wavelet detector display.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--event-time", type=float)
    parser.add_argument("--window-before", type=float, default=5.0)
    parser.add_argument("--window-after", type=float, default=5.0)
    parser.add_argument("--time-column", default="time_s")
    parser.add_argument(
        "--wheel-columns", nargs=4, default=DEFAULT_WHEEL_COLUMNS
    )
    parser.add_argument(
        "--reference-mode", choices=REFERENCE_MODES, default="opposite_diagonal"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = WaveletShapeConfig(reference_mode=args.reference_mode)
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
        f"{name}={time:.3f}s"
        for name, time in zip(WHEEL_NAMES, data.first_alarm_times)
        if time is not None
    ]
    print(f"wrote {args.output}")
    print("first alarms: " + (", ".join(alarms) if alarms else "none"))


if __name__ == "__main__":
    main()
