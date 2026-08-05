from __future__ import annotations

import argparse
import csv
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .detector import OPPOSITE_DIAGONAL_REFERENCES, WHEEL_NAMES
from .display import PLOT_CONFIG, WHEEL_COLORS


TIMER_WRAP_US = 65_536
DEFAULT_SAMPLE_TIME_S = 0.01
DEFAULT_COG_COUNT = 48
DEFAULT_PERIOD_WINDOW_TEETH = 12
DEFAULT_BASELINE_SECONDS = 3.0


@dataclass(frozen=True)
class ToothDisplayData:
    input_path: Path
    factors_path: Path
    start_time_s: float
    end_time_s: float
    frame_times: list[float]
    tooth_times: list[list[float]]
    tooth_phases: list[list[int]]
    raw_periods_us: list[list[float]]
    corrected_periods_us: list[list[float]]
    rolling_periods_us: list[list[float]]
    period_residuals_pct: list[list[float]]
    phase_residuals_teeth: list[list[float]]
    tooth_counts: list[list[int]]
    abnormal_period_counts: list[list[int]]
    baseline_rate_ratios: tuple[float, float, float, float]
    timer_wraps: tuple[int, int, int, int]
    duplicate_timestamps: tuple[int, int, int, int]
    period_window_teeth: int
    baseline_seconds: float

    @property
    def displayed_tooth_events(self) -> int:
        return sum(len(values) for values in self.tooth_times)

    @property
    def abnormal_period_events(self) -> int:
        return sum(sum(values) for values in self.abnormal_period_counts)


def load_tooth_factors(
    path: Path,
    cog_count: int = DEFAULT_COG_COUNT,
) -> tuple[tuple[float, ...], ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < cog_count:
        raise ValueError(
            f"tooth factor file has {len(rows)} phases; expected at least {cog_count}: {path}"
        )
    factors: list[tuple[float, ...]] = []
    for wheel_index in range(4):
        column = f"wheel{wheel_index}_factor"
        if column not in rows[0]:
            raise ValueError(f"missing {column} in tooth factor file: {path}")
        wheel_factors = tuple(float(row[column]) for row in rows[:cog_count])
        if any(not math.isfinite(value) or value <= 0.0 for value in wheel_factors):
            raise ValueError(f"invalid {column} in tooth factor file: {path}")
        factors.append(wheel_factors)
    return tuple(factors)


def _integer_fields(line: str) -> list[int]:
    values: list[int] = []
    for part in line.split():
        try:
            values.append(int(part))
        except ValueError:
            break
    return values


def _timestamps_from_row(line: str) -> list[int]:
    fields = _integer_fields(line)
    if not fields:
        return []
    count = max(0, fields[0])
    if count > len(fields) - 1:
        raise ValueError(
            f"tooth row declares {count} timestamps but contains {len(fields) - 1}"
        )
    return [max(0, min(TIMER_WRAP_US - 1, value)) for value in fields[1 : 1 + count]]


def iter_tooth_frames(path: Path) -> Iterable[tuple[list[int], ...]]:
    """Yield the four wheel timestamp rows from each five-row raw frame."""

    in_data = False
    frame_rows: list[list[int]] = []
    nonempty_row_index = 0
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not in_data:
                if stripped.lower() == "marks end":
                    in_data = True
                continue
            if not stripped:
                continue
            slot = nonempty_row_index % 5
            if slot < 4:
                frame_rows.append(_timestamps_from_row(stripped))
            nonempty_row_index += 1
            if slot == 4:
                if len(frame_rows) != 4:
                    raise ValueError(f"incomplete wheel timestamp frame in {path}")
                yield tuple(frame_rows)
                frame_rows = []
    if not in_data:
        raise ValueError(f"could not find 'Marks end' in {path}")


def _safe_median(values: Sequence[float], fallback: float) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(median(finite)) if finite else fallback


def analyze_tooth_file(
    input_path: Path,
    factors_path: Path,
    start_time_s: float,
    end_time_s: float,
    *,
    sample_time_s: float = DEFAULT_SAMPLE_TIME_S,
    cog_count: int = DEFAULT_COG_COUNT,
    period_window_teeth: int = DEFAULT_PERIOD_WINDOW_TEETH,
    baseline_seconds: float = DEFAULT_BASELINE_SECONDS,
) -> ToothDisplayData:
    if start_time_s < 0.0 or end_time_s <= start_time_s:
        raise ValueError("tooth display time range is invalid")
    if sample_time_s <= 0.0:
        raise ValueError("sample_time_s must be positive")
    if cog_count <= 1:
        raise ValueError("cog_count must exceed one")
    if period_window_teeth < 2:
        raise ValueError("period_window_teeth must be at least two")
    if baseline_seconds <= 0.0:
        raise ValueError("baseline_seconds must be positive")

    input_path = input_path.resolve()
    factors_path = factors_path.resolve()
    factors = load_tooth_factors(factors_path, cog_count)
    previous_timestamps: list[int | None] = [None] * 4
    phase_indices = [0] * 4
    timer_wraps = [0] * 4
    duplicate_timestamps = [0] * 4
    corrected_windows = [deque(maxlen=period_window_teeth) for _ in range(4)]
    raw_quality_windows = [deque(maxlen=cog_count) for _ in range(4)]
    rate_ratio_histories = [
        deque(maxlen=max(10, math.ceil(baseline_seconds / sample_time_s)))
        for _ in range(4)
    ]
    baseline_ratios: tuple[float, float, float, float] | None = None
    phase_residuals = [0.0] * 4
    last_display_time: float | None = None

    frame_times: list[float] = []
    tooth_times: list[list[float]] = [[] for _ in range(4)]
    tooth_phases: list[list[int]] = [[] for _ in range(4)]
    raw_periods: list[list[float]] = [[] for _ in range(4)]
    corrected_periods: list[list[float]] = [[] for _ in range(4)]
    rolling_periods: list[list[float]] = [[] for _ in range(4)]
    period_residuals: list[list[float]] = [[] for _ in range(4)]
    displayed_phase_residuals: list[list[float]] = [[] for _ in range(4)]
    tooth_counts: list[list[int]] = [[] for _ in range(4)]
    abnormal_counts: list[list[int]] = [[] for _ in range(4)]

    clock_origin_counter: int | None = None
    clock_origin_frame = 0
    stop_frame = math.floor(end_time_s / sample_time_s) + 1

    for frame_index, frame in enumerate(iter_tooth_frames(input_path)):
        if frame_index > stop_frame:
            break
        frame_time = frame_index * sample_time_s
        frame_abnormal = [0] * 4
        frame_counts = [0] * 4
        for wheel_index, timestamps in enumerate(frame):
            frame_counts[wheel_index] = len(timestamps)
            for timestamp in timestamps:
                if clock_origin_counter is None:
                    clock_origin_counter = timestamp
                    clock_origin_frame = frame_index
                expected_counter = clock_origin_counter + round(
                    (frame_index - clock_origin_frame) * sample_time_s * 1.0e6
                )
                wrap_number = round(
                    (expected_counter - timestamp) / TIMER_WRAP_US
                )
                unwrapped_counter = timestamp + wrap_number * TIMER_WRAP_US
                event_time = (
                    clock_origin_frame * sample_time_s
                    + (unwrapped_counter - clock_origin_counter) / 1.0e6
                )

                previous = previous_timestamps[wheel_index]
                previous_timestamps[wheel_index] = timestamp
                if previous is None:
                    continue
                if timestamp == previous:
                    if start_time_s <= event_time <= end_time_s:
                        duplicate_timestamps[wheel_index] += 1
                    frame_abnormal[wheel_index] += 1
                    continue
                delta_us = timestamp - previous
                if delta_us < 0:
                    delta_us += TIMER_WRAP_US
                    if start_time_s <= event_time <= end_time_s:
                        timer_wraps[wheel_index] += 1
                phase = phase_indices[wheel_index] % cog_count
                phase_indices[wheel_index] += 1
                corrected_us = delta_us / factors[wheel_index][phase]
                quality_history = raw_quality_windows[wheel_index]
                if len(quality_history) >= 8:
                    typical = float(median(quality_history))
                    if delta_us < 0.5 * typical or delta_us > 1.5 * typical:
                        frame_abnormal[wheel_index] += 1
                quality_history.append(float(delta_us))
                corrected_windows[wheel_index].append(corrected_us)
                if start_time_s <= event_time <= end_time_s:
                    tooth_times[wheel_index].append(event_time)
                    tooth_phases[wheel_index].append(phase)
                    raw_periods[wheel_index].append(float(delta_us))
                    corrected_periods[wheel_index].append(corrected_us)

        estimated_periods = [
            math.nan
            if len(window) < period_window_teeth
            else sum(window) / len(window)
            for window in corrected_windows
        ]
        rates = [
            math.nan if not math.isfinite(value) or value <= 0.0 else 1.0e6 / value
            for value in estimated_periods
        ]
        rate_ratios = [math.nan] * 4
        if all(math.isfinite(value) and value > 0.0 for value in rates):
            for wheel_index, reference_indices in enumerate(
                OPPOSITE_DIAGONAL_REFERENCES
            ):
                reference_rate = sum(rates[index] for index in reference_indices) / 2.0
                rate_ratios[wheel_index] = rates[wheel_index] / reference_rate
                if frame_time < start_time_s:
                    rate_ratio_histories[wheel_index].append(
                        rate_ratios[wheel_index]
                    )

        if frame_time < start_time_s:
            continue
        if frame_time > end_time_s:
            break
        if baseline_ratios is None:
            baseline_ratios = tuple(
                _safe_median(
                    list(rate_ratio_histories[wheel_index]),
                    rate_ratios[wheel_index]
                    if math.isfinite(rate_ratios[wheel_index])
                    else 1.0,
                )
                for wheel_index in range(4)
            )  # type: ignore[assignment]

        dt = 0.0 if last_display_time is None else frame_time - last_display_time
        last_display_time = frame_time
        frame_times.append(frame_time)
        for wheel_index, reference_indices in enumerate(
            OPPOSITE_DIAGONAL_REFERENCES
        ):
            period = estimated_periods[wheel_index]
            rolling_periods[wheel_index].append(period)
            tooth_counts[wheel_index].append(frame_counts[wheel_index])
            abnormal_counts[wheel_index].append(frame_abnormal[wheel_index])
            if not all(math.isfinite(value) and value > 0.0 for value in rates):
                period_residuals[wheel_index].append(math.nan)
                displayed_phase_residuals[wheel_index].append(
                    phase_residuals[wheel_index]
                )
                continue
            reference_rate = sum(rates[index] for index in reference_indices) / 2.0
            predicted_rate = baseline_ratios[wheel_index] * reference_rate
            residual_pct = (predicted_rate / rates[wheel_index] - 1.0) * 100.0
            period_residuals[wheel_index].append(residual_pct)
            phase_residuals[wheel_index] += (
                rates[wheel_index] - predicted_rate
            ) * dt
            displayed_phase_residuals[wheel_index].append(
                phase_residuals[wheel_index]
            )

    if not frame_times:
        raise ValueError("tooth display window contains no frames")
    if baseline_ratios is None:
        baseline_ratios = (1.0, 1.0, 1.0, 1.0)
    return ToothDisplayData(
        input_path=input_path,
        factors_path=factors_path,
        start_time_s=start_time_s,
        end_time_s=end_time_s,
        frame_times=frame_times,
        tooth_times=tooth_times,
        tooth_phases=tooth_phases,
        raw_periods_us=raw_periods,
        corrected_periods_us=corrected_periods,
        rolling_periods_us=rolling_periods,
        period_residuals_pct=period_residuals,
        phase_residuals_teeth=displayed_phase_residuals,
        tooth_counts=tooth_counts,
        abnormal_period_counts=abnormal_counts,
        baseline_rate_ratios=baseline_ratios,
        timer_wraps=tuple(timer_wraps),  # type: ignore[arg-type]
        duplicate_timestamps=tuple(duplicate_timestamps),  # type: ignore[arg-type]
        period_window_teeth=period_window_teeth,
        baseline_seconds=baseline_seconds,
    )


def build_tooth_figure(
    data: ToothDisplayData,
    *,
    title: str | None = None,
    event_time_s: float | None = None,
    alarm_times: Sequence[tuple[str, float, str]] = (),
) -> go.Figure:
    figure = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=(0.15, 0.24, 0.24, 0.22, 0.15),
        subplot_titles=(
            "逐齿到达事件",
            "校正齿周期残差（负值=目标轮相对变快）",
            "累计相位残差",
            f"{data.period_window_teeth}齿滑动平均周期",
            "每10 ms齿数与异常周期标记",
        ),
    )
    for wheel_index, (name, color) in enumerate(zip(WHEEL_NAMES, WHEEL_COLORS)):
        raster_custom = list(
            zip(
                data.tooth_phases[wheel_index],
                data.raw_periods_us[wheel_index],
                data.corrected_periods_us[wheel_index],
            )
        )
        figure.add_trace(
            go.Scattergl(
                x=data.tooth_times[wheel_index],
                y=[wheel_index] * len(data.tooth_times[wheel_index]),
                mode="markers",
                name=name,
                marker={"color": color, "size": 5, "symbol": "line-ns"},
                customdata=raster_custom,
                hovertemplate=(
                    f"{name}<br>t=%{{x:.6f}} s<br>齿位=%{{customdata[0]}}"
                    "<br>原始周期=%{customdata[1]:.1f} us"
                    "<br>校正周期=%{customdata[2]:.1f} us<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scattergl(
                x=data.frame_times,
                y=data.period_residuals_pct[wheel_index],
                mode="lines",
                name=f"{name} period residual",
                line={"color": color, "width": 1.2},
                legendgroup=name,
                showlegend=False,
                hovertemplate=f"{name} 周期残差=%{{y:.3f}}%<extra></extra>",
            ),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Scattergl(
                x=data.frame_times,
                y=data.phase_residuals_teeth[wheel_index],
                mode="lines",
                name=f"{name} phase residual",
                line={"color": color, "width": 1.4},
                legendgroup=name,
                showlegend=False,
                hovertemplate=f"{name} 相位残差=%{{y:.3f}} 齿<extra></extra>",
            ),
            row=3,
            col=1,
        )
        figure.add_trace(
            go.Scattergl(
                x=data.frame_times,
                y=data.rolling_periods_us[wheel_index],
                mode="lines",
                name=f"{name} rolling period",
                line={"color": color, "width": 1.1},
                legendgroup=name,
                showlegend=False,
                hovertemplate=f"{name} 滑动周期=%{{y:.1f}} us<extra></extra>",
            ),
            row=4,
            col=1,
        )
        figure.add_trace(
            go.Scattergl(
                x=data.frame_times,
                y=data.tooth_counts[wheel_index],
                mode="lines",
                name=f"{name} tooth count",
                line={"color": color, "width": 1.0},
                legendgroup=name,
                showlegend=False,
                hovertemplate=f"{name} 齿数=%{{y:.0f}}<extra></extra>",
            ),
            row=5,
            col=1,
        )
        abnormal_x = [
            time
            for time, count in zip(
                data.frame_times, data.abnormal_period_counts[wheel_index]
            )
            if count
        ]
        abnormal_y = [
            count
            for count in data.abnormal_period_counts[wheel_index]
            if count
        ]
        if abnormal_x:
            figure.add_trace(
                go.Scattergl(
                    x=abnormal_x,
                    y=abnormal_y,
                    mode="markers",
                    name=f"{name} abnormal",
                    marker={"color": "#991b1b", "size": 8, "symbol": "x"},
                    legendgroup=name,
                    showlegend=False,
                    hovertemplate=f"{name} 异常周期=%{{y:.0f}}<extra></extra>",
                ),
                row=5,
                col=1,
            )

    figure.update_yaxes(
        tickvals=list(range(4)), ticktext=WHEEL_NAMES, range=(-0.6, 3.6), row=1, col=1
    )
    figure.update_yaxes(title_text="%", row=2, col=1)
    figure.update_yaxes(title_text="齿", row=3, col=1)
    figure.update_yaxes(title_text="us", row=4, col=1)
    figure.update_yaxes(title_text="齿/10ms", row=5, col=1)
    figure.add_hline(y=0.0, line_dash="dot", line_color="#64748b", row=2, col=1)
    figure.add_hline(y=0.0, line_dash="dot", line_color="#64748b", row=3, col=1)
    if event_time_s is not None:
        figure.add_vline(
            x=event_time_s,
            line_width=2,
            line_dash="dash",
            line_color="#7c3aed",
            annotation_text="人工事件",
        )
    for label, alarm_time, color in alarm_times:
        if data.start_time_s <= alarm_time <= data.end_time_s:
            figure.add_vline(
                x=alarm_time,
                line_width=1.5,
                line_dash="dot",
                line_color=color,
                annotation_text=label,
                annotation_position="top right",
            )
    figure.update_xaxes(
        title_text="时间 / s", rangeslider={"visible": True}, row=5, col=1
    )
    figure.update_layout(
        title=title or f"齿事件爆胎特征 — {data.input_path.name}",
        template="plotly_white",
        height=1350,
        hovermode="x unified",
        dragmode="pan",
        legend={"orientation": "h", "y": 1.04, "x": 0.0},
        margin={"l": 75, "r": 40, "t": 110, "b": 60},
    )
    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a raw tooth-event display.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--factors", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=float)
    parser.add_argument("--end", type=float)
    parser.add_argument("--event-time", type=float)
    parser.add_argument("--window-before", type=float, default=1.0)
    parser.add_argument("--window-after", type=float, default=2.0)
    parser.add_argument(
        "--period-window-teeth", type=int, default=DEFAULT_PERIOD_WINDOW_TEETH
    )
    parser.add_argument("--baseline-seconds", type=float, default=DEFAULT_BASELINE_SECONDS)
    args = parser.parse_args()
    if (args.start is None) != (args.end is None):
        parser.error("--start and --end must be supplied together")
    if args.start is None:
        if args.event_time is None:
            args.start, args.end = 0.0, 10.0
        else:
            args.start = max(0.0, args.event_time - args.window_before)
            args.end = args.event_time + args.window_after
    if args.factors is None:
        args.factors = args.input.parent / "learned_tooth_correction_factors.csv"
    return args


def main() -> None:
    args = parse_args()
    data = analyze_tooth_file(
        args.input,
        args.factors,
        args.start,
        args.end,
        period_window_teeth=args.period_window_teeth,
        baseline_seconds=args.baseline_seconds,
    )
    figure = build_tooth_figure(data, event_time_s=args.event_time)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        args.output, include_plotlyjs="cdn", full_html=True, config=PLOT_CONFIG
    )
    print(f"wrote {args.output}")
    print(
        f"displayed_tooth_events={data.displayed_tooth_events}, "
        f"abnormal_period_events={data.abnormal_period_events}"
    )


if __name__ == "__main__":
    main()
