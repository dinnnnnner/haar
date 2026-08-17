from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .detector import (
    WHEEL_COUNT,
    WheelSpeedBlowoutConfig,
    WheelSpeedBlowoutDetector,
    WheelSpeedFrame,
    WheelSpeedResult,
)


DEFAULT_TIME_COLUMN = "time_s"
DEFAULT_WHEEL_COLUMNS = tuple(
    f"wheel{index}_corrected_rad_s" for index in range(WHEEL_COUNT)
)


@dataclass(frozen=True)
class SuspectInterval:
    wheel_index: int
    start_s: float
    end_s: float
    confirmed: bool
    peak_individual_gain_pct: float | None
    peak_diagonal_gain_pct: float | None
    peak_individual_edge_pct: float | None
    peak_diagonal_edge_pct: float | None

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


@dataclass(frozen=True)
class ScanResult:
    start_s: float
    end_s: float
    first_warmed_s: float | None
    frames: int
    valid_frames: int
    suspects: tuple[SuspectInterval, ...]
    first_alarm_times: tuple[float | None, float | None, float | None, float | None]


@dataclass
class WindowData:
    times: list[float]
    wheels: list[list[float]]
    individual_gains: list[list[float | None]]
    diagonal_gains: list[list[float | None]]
    individual_edges: list[list[float | None]]
    diagonal_edges: list[list[float | None]]
    candidates: list[list[bool]]
    alarms: list[list[bool]]
    states: list[list[str]]


def _finite_percent(value: float) -> float | None:
    return value * 100.0 if math.isfinite(value) else None


def _update_peak(current: float | None, value: float) -> float | None:
    converted = _finite_percent(value)
    if converted is None:
        return current
    return converted if current is None else max(current, converted)


def _validate_columns(
    fieldnames: Sequence[str] | None,
    input_path: Path,
    time_column: str,
    wheel_columns: Sequence[str],
) -> None:
    if fieldnames is None:
        raise ValueError(f"CSV 没有表头：{input_path}")
    required = (time_column, *wheel_columns)
    missing = [column for column in required if column not in fieldnames]
    if missing:
        raise ValueError(f"CSV 缺少列：{', '.join(missing)}")


def _frame_from_row(
    row: dict[str, str], time_column: str, wheel_columns: Sequence[str]
) -> WheelSpeedFrame:
    return WheelSpeedFrame.from_sequences(
        float(row[time_column]),
        [float(row[column]) for column in wheel_columns],
    )


def scan_csv(
    input_path: Path,
    cfg: WheelSpeedBlowoutConfig | None = None,
    *,
    time_column: str = DEFAULT_TIME_COLUMN,
    wheel_columns: Sequence[str] = DEFAULT_WHEEL_COLUMNS,
) -> ScanResult:
    """Replay a complete CSV and collect every candidate interval."""

    detector = WheelSpeedBlowoutDetector(cfg)
    starts: list[float | None] = [None] * WHEEL_COUNT
    peak_gains: list[float | None] = [None] * WHEEL_COUNT
    peak_diag_gains: list[float | None] = [None] * WHEEL_COUNT
    peak_edges: list[float | None] = [None] * WHEEL_COUNT
    peak_diag_edges: list[float | None] = [None] * WHEEL_COUNT
    first_alarms: list[float | None] = [None] * WHEEL_COUNT
    suspects: list[SuspectInterval] = []
    first_time: float | None = None
    last_time: float | None = None
    first_warmed: float | None = None
    frames = 0
    valid_frames = 0

    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        _validate_columns(reader.fieldnames, input_path, time_column, wheel_columns)
        for row in reader:
            result = detector.push(_frame_from_row(row, time_column, wheel_columns))
            t_sec = result.t_sec
            first_time = t_sec if first_time is None else first_time
            last_time = t_sec
            frames += 1
            valid_frames += int(result.speed_valid)
            if result.warmed_up and first_warmed is None:
                first_warmed = t_sec
            for wheel in range(WHEEL_COUNT):
                active = result.candidates[wheel]
                if active and starts[wheel] is None:
                    starts[wheel] = t_sec
                if active:
                    peak_gains[wheel] = _update_peak(
                        peak_gains[wheel], result.individual_gains[wheel]
                    )
                    peak_diag_gains[wheel] = _update_peak(
                        peak_diag_gains[wheel], result.diagonal_gains[wheel]
                    )
                    peak_edges[wheel] = _update_peak(
                        peak_edges[wheel], result.individual_edges[wheel]
                    )
                    peak_diag_edges[wheel] = _update_peak(
                        peak_diag_edges[wheel], result.diagonal_edges[wheel]
                    )
                elif starts[wheel] is not None:
                    suspects.append(
                        SuspectInterval(
                            wheel,
                            starts[wheel],
                            t_sec,
                            result.new_blowouts[wheel],
                            peak_gains[wheel],
                            peak_diag_gains[wheel],
                            peak_edges[wheel],
                            peak_diag_edges[wheel],
                        )
                    )
                    starts[wheel] = None
                    peak_gains[wheel] = None
                    peak_diag_gains[wheel] = None
                    peak_edges[wheel] = None
                    peak_diag_edges[wheel] = None
                if result.new_blowouts[wheel] and first_alarms[wheel] is None:
                    first_alarms[wheel] = t_sec

    if first_time is None or last_time is None:
        raise ValueError("CSV 中没有数据")
    for wheel, start in enumerate(starts):
        if start is not None:
            suspects.append(
                SuspectInterval(
                    wheel,
                    start,
                    last_time,
                    False,
                    peak_gains[wheel],
                    peak_diag_gains[wheel],
                    peak_edges[wheel],
                    peak_diag_edges[wheel],
                )
            )
    suspects.sort(key=lambda interval: (interval.start_s, interval.wheel_index))
    return ScanResult(
        first_time,
        last_time,
        first_warmed,
        frames,
        valid_frames,
        tuple(suspects),
        tuple(first_alarms),  # type: ignore[arg-type]
    )


def window_data_from_results(results: Iterable[WheelSpeedResult]) -> WindowData:
    data = WindowData(
        times=[],
        wheels=[[] for _ in range(WHEEL_COUNT)],
        individual_gains=[[] for _ in range(WHEEL_COUNT)],
        diagonal_gains=[[] for _ in range(WHEEL_COUNT)],
        individual_edges=[[] for _ in range(WHEEL_COUNT)],
        diagonal_edges=[[] for _ in range(WHEEL_COUNT)],
        candidates=[[] for _ in range(WHEEL_COUNT)],
        alarms=[[] for _ in range(WHEEL_COUNT)],
        states=[[] for _ in range(WHEEL_COUNT)],
    )
    for result in results:
        data.times.append(result.t_sec)
        for wheel in range(WHEEL_COUNT):
            data.wheels[wheel].append(result.wheels[wheel])
            data.individual_gains[wheel].append(
                _finite_percent(result.individual_gains[wheel])
            )
            data.diagonal_gains[wheel].append(
                _finite_percent(result.diagonal_gains[wheel])
            )
            data.individual_edges[wheel].append(
                _finite_percent(result.individual_edges[wheel])
            )
            data.diagonal_edges[wheel].append(
                _finite_percent(result.diagonal_edges[wheel])
            )
            data.candidates[wheel].append(result.candidates[wheel])
            data.alarms[wheel].append(result.blowout_alarms[wheel])
            data.states[wheel].append(result.states[wheel])
    if not data.times:
        raise ValueError("所选窗口内没有数据")
    return data


def analyze_window(
    input_path: Path,
    start_s: float,
    end_s: float,
    cfg: WheelSpeedBlowoutConfig | None = None,
    *,
    time_column: str = DEFAULT_TIME_COLUMN,
    wheel_columns: Sequence[str] = DEFAULT_WHEEL_COLUMNS,
) -> WindowData:
    """Replay causally from the file start, retaining only the requested window."""

    detector = WheelSpeedBlowoutDetector(cfg)
    results: list[WheelSpeedResult] = []
    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        _validate_columns(reader.fieldnames, input_path, time_column, wheel_columns)
        for row in reader:
            frame = _frame_from_row(row, time_column, wheel_columns)
            if frame.t_sec > end_s:
                break
            result = detector.push(frame)
            if frame.t_sec >= start_s:
                results.append(result)
    return window_data_from_results(results)
