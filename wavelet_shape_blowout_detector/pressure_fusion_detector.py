from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from statistics import median
from typing import Sequence

WHEEL_NAMES = ("FL", "FR", "RL", "RR")
WHEEL_COUNT = 4
DIAGONALS = ((0, 3), (1, 2))


@dataclass(frozen=True)
class PressureFusionConfig:
    """Pressure-assisted detector settings for corrected 100 Hz wheel speed.

    Ratio thresholds are log ratios; 0.006 is approximately 0.6 percent.
    """

    sample_rate_hz: float = 100.0
    min_avg_speed: float = 20.0
    smooth_window: int = 5
    edge_half_window: int = 6
    baseline_window: int = 500
    baseline_min_samples: int = 200
    baseline_refresh_frames: int = 10
    min_individual_edge: float = 0.0058
    min_diagonal_edge: float = 0.0058
    min_individual_peak: float = 0.0070
    min_diagonal_peak: float = 0.0070
    max_individual_peak: float = 0.0250
    max_diagonal_peak: float = 0.0450
    confirm_frames: int = 70
    persistence_tail_frames: int = 40
    min_individual_persistence: float = 0.0055
    min_diagonal_persistence: float = 0.0055
    persistence_floor: float = 0.0035
    min_persistence_fraction: float = 0.75
    min_mate_persistence: float = -0.0035
    max_common_speed_range: float = 0.050
    candidate_drop_limit: float = -0.0040
    clear_after_invalid_frames: int = 50

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0 or self.min_avg_speed <= 0:
            raise ValueError("sample rate and minimum speed must be positive")
        if self.smooth_window <= 0 or self.smooth_window % 2 == 0:
            raise ValueError("smooth_window must be a positive odd integer")
        if self.edge_half_window <= 0:
            raise ValueError("edge_half_window must be positive")
        if not 2 <= self.baseline_min_samples <= self.baseline_window:
            raise ValueError("baseline sample limits are invalid")
        if self.baseline_refresh_frames <= 0:
            raise ValueError("baseline_refresh_frames must be positive")
        if self.confirm_frames <= 0:
            raise ValueError("confirm_frames must be positive")
        if not 1 <= self.persistence_tail_frames <= self.confirm_frames:
            raise ValueError("persistence_tail_frames is outside confirm_frames")
        if not 0 <= self.min_persistence_fraction <= 1:
            raise ValueError("min_persistence_fraction must be in [0, 1]")
        if self.max_individual_peak <= self.min_individual_peak:
            raise ValueError("individual peak limits are invalid")
        if self.max_diagonal_peak <= self.min_diagonal_peak:
            raise ValueError("diagonal peak limits are invalid")


@dataclass(frozen=True)
class PressureFusionFrame:
    t_sec: float
    wheels: tuple[float, float, float, float]
    # True=confirmed blowout, False=confirmed healthy, None=no signal.
    pressure_blowouts: tuple[bool | None, bool | None, bool | None, bool | None] = (
        None,
        None,
        None,
        None,
    )

    @classmethod
    def from_sequences(
        cls,
        t_sec: float,
        wheels: Sequence[float],
        pressure_blowouts: Sequence[bool | None] | None = None,
    ) -> PressureFusionFrame:
        if len(wheels) != WHEEL_COUNT:
            raise ValueError("expected four wheel speeds")
        pressure = [None] * WHEEL_COUNT if pressure_blowouts is None else list(pressure_blowouts)
        if len(pressure) != WHEEL_COUNT:
            raise ValueError("expected four pressure signal entries")
        return cls(
            float(t_sec),
            tuple(float(value) for value in wheels),  # type: ignore[arg-type]
            tuple(None if value is None else bool(value) for value in pressure),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class PressureFusionResult:
    t_sec: float
    wheels: tuple[float, float, float, float]
    pressure_blowouts: tuple[bool | None, bool | None, bool | None, bool | None]
    sensor_diagonal: tuple[str, ...]
    speed_diagonal: tuple[str, ...]
    speed_valid: bool
    speed_detection_available: bool
    individual_gains: tuple[float, float, float, float]
    individual_edges: tuple[float, float, float, float]
    diagonal_gain: float
    diagonal_edge: float
    candidates: tuple[bool, bool, bool, bool]
    new_blowouts: tuple[bool, bool, bool, bool]
    blowout_alarms: tuple[bool, bool, bool, bool]
    alarm_sources: tuple[str, str, str, str]
    estimated_onset_indices: tuple[int | None, int | None, int | None, int | None]
    estimated_onset_times_s: tuple[float | None, float | None, float | None, float | None]


@dataclass
class _Candidate:
    onset_index: int
    onset_time_s: float
    individual_values: list[float]
    diagonal_values: list[float]
    mate_values: list[float]
    common_log_speeds: list[float]
    max_individual: float
    max_diagonal: float


class PressureFusionBlowoutDetector:
    """Fuse one automatically discovered pressure diagonal with wheel speed."""

    def __init__(self, cfg: PressureFusionConfig | None = None):
        self.cfg = cfg or PressureFusionConfig()
        self._frame_index = -1
        self._last_t_sec: float | None = None
        self._sensor_diagonal: tuple[int, int] | None = None
        self._speed_diagonal: tuple[int, int] | None = None
        self._invalid_frames = 0
        self._raw = [deque(maxlen=self.cfg.smooth_window) for _ in range(4)]
        self._smooth = [deque(maxlen=self.cfg.smooth_window) for _ in range(4)]
        self._edges = [deque(maxlen=2 * self.cfg.edge_half_window) for _ in range(4)]
        self._baselines = [deque(maxlen=self.cfg.baseline_window) for _ in range(4)]
        self._baseline_cache: list[float | None] = [None] * 4
        self._diag_raw: deque[float] = deque(maxlen=self.cfg.smooth_window)
        self._diag_smooth: deque[float] = deque(maxlen=self.cfg.smooth_window)
        self._diag_edges: deque[float] = deque(maxlen=2 * self.cfg.edge_half_window)
        self._diag_baseline: deque[float] = deque(maxlen=self.cfg.baseline_window)
        self._diag_baseline_cache: float | None = None
        self._candidates: list[_Candidate | None] = [None] * 4
        self._alarms = [False] * 4
        self._sources = ["none"] * 4
        self._onset_indices: list[int | None] = [None] * 4
        self._onset_times: list[float | None] = [None] * 4
        self._previous_pressure = [False] * 4

    def reset(self) -> None:
        self.__init__(self.cfg)

    def push(self, frame: PressureFusionFrame) -> PressureFusionResult:
        self._validate(frame)
        self._frame_index += 1
        self._last_t_sec = frame.t_sec
        self._discover_diagonal(frame.pressure_blowouts)
        new = [False] * 4
        self._apply_pressure(frame, new)

        wheels = tuple(abs(value) for value in frame.wheels)
        speed_valid = min(wheels) > 1e-9 and sum(wheels) / 4 >= self.cfg.min_avg_speed
        speed_available = speed_valid and self._reference_healthy(frame.pressure_blowouts)
        gains = [math.nan] * 4
        edges = [math.nan] * 4
        diag_gain = math.nan
        diag_edge = math.nan

        if not speed_available:
            self._handle_unavailable(speed_valid)
        else:
            self._invalid_frames = 0
            assert self._sensor_diagonal is not None and self._speed_diagonal is not None
            logs = tuple(math.log(value) for value in wheels)
            common_log_speed = sum(logs) / WHEEL_COUNT
            sensor_mean = sum(logs[i] for i in self._sensor_diagonal) / 2
            diag_value = sum(logs[i] for i in self._speed_diagonal) - sum(
                logs[i] for i in self._sensor_diagonal
            )
            diag_smoothed = self._smooth_value(diag_value, self._diag_raw, self._diag_smooth)
            diag_baseline = self._current_diagonal_baseline()
            if diag_baseline is not None:
                diag_gain = diag_smoothed - diag_baseline
                self._diag_edges.append(diag_gain)
                diag_edge = self._edge(self._diag_edges)

            for wheel in self._speed_diagonal:
                value = logs[wheel] - sensor_mean
                smoothed = self._smooth_value(value, self._raw[wheel], self._smooth[wheel])
                baseline = self._current_baseline(wheel)
                if baseline is None or diag_baseline is None:
                    self._baselines[wheel].append(smoothed)
                    continue
                gains[wheel] = smoothed - baseline
                self._edges[wheel].append(gains[wheel])
                edges[wheel] = self._edge(self._edges[wheel])
            for wheel in self._speed_diagonal:
                if not math.isfinite(gains[wheel]):
                    continue
                mate = self._speed_diagonal[1 - self._speed_diagonal.index(wheel)]
                self._advance(
                    wheel,
                    gains[wheel],
                    edges[wheel],
                    diag_gain,
                    diag_edge,
                    gains[mate],
                    common_log_speed,
                    new,
                )

            if diag_baseline is None:
                self._diag_baseline.append(diag_smoothed)
            elif not any(self._candidates) and not any(self._alarms[i] for i in self._speed_diagonal):
                self._diag_baseline.append(diag_smoothed)
                for wheel in self._speed_diagonal:
                    if self._smooth[wheel]:
                        self._baselines[wheel].append(self._smooth[wheel][-1])

        return PressureFusionResult(
            frame.t_sec,
            frame.wheels,
            frame.pressure_blowouts,
            self._names(self._sensor_diagonal),
            self._names(self._speed_diagonal),
            speed_valid,
            speed_available,
            tuple(gains),  # type: ignore[arg-type]
            tuple(edges),  # type: ignore[arg-type]
            diag_gain,
            diag_edge,
            tuple(value is not None for value in self._candidates),  # type: ignore[arg-type]
            tuple(new),  # type: ignore[arg-type]
            tuple(self._alarms),  # type: ignore[arg-type]
            tuple(self._sources),  # type: ignore[arg-type]
            tuple(self._onset_indices),  # type: ignore[arg-type]
            tuple(self._onset_times),  # type: ignore[arg-type]
        )

    def _advance(
        self,
        wheel: int,
        gain: float,
        edge: float,
        diag_gain: float,
        diag_edge: float,
        mate_gain: float,
        common_log_speed: float,
        new: list[bool],
    ) -> None:
        if self._alarms[wheel]:
            return
        candidate = self._candidates[wheel]
        if candidate is None:
            if edge >= self.cfg.min_individual_edge and diag_edge >= self.cfg.min_diagonal_edge:
                delay = self.cfg.smooth_window - 1 + self.cfg.edge_half_window
                self._candidates[wheel] = _Candidate(
                    max(0, self._frame_index - delay),
                    self._last_t_sec - delay / self.cfg.sample_rate_hz,  # type: ignore[operator]
                    [gain],
                    [diag_gain],
                    [mate_gain],
                    [common_log_speed],
                    gain,
                    diag_gain,
                )
            return
        candidate.individual_values.append(gain)
        candidate.diagonal_values.append(diag_gain)
        candidate.mate_values.append(mate_gain)
        candidate.common_log_speeds.append(common_log_speed)
        candidate.max_individual = max(candidate.max_individual, gain)
        candidate.max_diagonal = max(candidate.max_diagonal, diag_gain)
        if (
            gain < self.cfg.candidate_drop_limit
            or diag_gain < self.cfg.candidate_drop_limit
            or candidate.max_individual > self.cfg.max_individual_peak
            or candidate.max_diagonal > self.cfg.max_diagonal_peak
        ):
            self._candidates[wheel] = None
            return
        if len(candidate.individual_values) < self.cfg.confirm_frames:
            return
        a = candidate.individual_values[-self.cfg.persistence_tail_frames :]
        d = candidate.diagonal_values[-self.cfg.persistence_tail_frames :]
        mate = candidate.mate_values[-self.cfg.persistence_tail_frames :]
        common_range = max(candidate.common_log_speeds) - min(
            candidate.common_log_speeds
        )
        confirmed = (
            candidate.max_individual >= self.cfg.min_individual_peak
            and candidate.max_diagonal >= self.cfg.min_diagonal_peak
            and median(a) >= self.cfg.min_individual_persistence
            and median(d) >= self.cfg.min_diagonal_persistence
            and self._above_fraction(a) >= self.cfg.min_persistence_fraction
            and self._above_fraction(d) >= self.cfg.min_persistence_fraction
            and median(mate) >= self.cfg.min_mate_persistence
            and common_range <= self.cfg.max_common_speed_range
        )
        if confirmed:
            self._alarms[wheel] = True
            self._sources[wheel] = "wheel_speed_confirmed"
            self._onset_indices[wheel] = candidate.onset_index
            self._onset_times[wheel] = candidate.onset_time_s
            new[wheel] = True
        self._candidates[wheel] = None

    def _apply_pressure(self, frame: PressureFusionFrame, new: list[bool]) -> None:
        for wheel, signal in enumerate(frame.pressure_blowouts):
            active = signal is True
            if active and not self._previous_pressure[wheel]:
                new[wheel] = not self._alarms[wheel]
                self._alarms[wheel] = True
                self._sources[wheel] = "pressure"
                self._onset_indices[wheel] = self._frame_index
                self._onset_times[wheel] = frame.t_sec
                self._candidates[wheel] = None
            self._previous_pressure[wheel] = active

    def _discover_diagonal(self, pressure: tuple[bool | None, bool | None, bool | None, bool | None]) -> None:
        available = tuple(i for i, value in enumerate(pressure) if value is not None)
        if not available:
            return
        if available not in DIAGONALS:
            raise ValueError("non-None pressure signals must be exactly one diagonal")
        if self._sensor_diagonal is None:
            self._sensor_diagonal = available  # type: ignore[assignment]
            self._speed_diagonal = DIAGONALS[1 - DIAGONALS.index(available)]
        elif available != self._sensor_diagonal:
            raise ValueError("pressure sensor diagonal changed without reset")

    def _reference_healthy(self, pressure: tuple[bool | None, bool | None, bool | None, bool | None]) -> bool:
        return self._sensor_diagonal is not None and all(
            pressure[i] is False and not self._alarms[i]
            for i in self._sensor_diagonal
        )

    def _handle_unavailable(self, speed_valid: bool) -> None:
        self._invalid_frames = 0 if speed_valid else self._invalid_frames + 1
        self._candidates = [None] * 4
        for window in (*self._edges, self._diag_edges):
            window.clear()
        if self._invalid_frames >= self.cfg.clear_after_invalid_frames:
            for window in (
                *self._raw,
                *self._smooth,
                *self._baselines,
                self._diag_raw,
                self._diag_smooth,
                self._diag_baseline,
            ):
                window.clear()
            self._baseline_cache = [None] * 4
            self._diag_baseline_cache = None

    @staticmethod
    def _smooth_value(value: float, raw: deque[float], smooth: deque[float]) -> float:
        raw.append(value)
        smooth.append(float(median(raw)))
        return sum(smooth) / len(smooth)

    def _current_baseline(self, wheel: int) -> float | None:
        values = self._baselines[wheel]
        if len(values) < self.cfg.baseline_min_samples:
            return None
        if (
            self._baseline_cache[wheel] is None
            or self._frame_index % self.cfg.baseline_refresh_frames == 0
        ):
            self._baseline_cache[wheel] = float(median(values))
        return self._baseline_cache[wheel]

    def _current_diagonal_baseline(self) -> float | None:
        if len(self._diag_baseline) < self.cfg.baseline_min_samples:
            return None
        if (
            self._diag_baseline_cache is None
            or self._frame_index % self.cfg.baseline_refresh_frames == 0
        ):
            self._diag_baseline_cache = float(median(self._diag_baseline))
        return self._diag_baseline_cache

    def _edge(self, values: deque[float]) -> float:
        half = self.cfg.edge_half_window
        if len(values) < 2 * half:
            return math.nan
        data = list(values)
        return sum(data[half:]) / half - sum(data[:half]) / half

    def _above_fraction(self, values: list[float]) -> float:
        return sum(value >= self.cfg.persistence_floor for value in values) / len(values)

    @staticmethod
    def _names(indices: tuple[int, int] | None) -> tuple[str, ...]:
        return () if indices is None else tuple(WHEEL_NAMES[i] for i in indices)

    def _validate(self, frame: PressureFusionFrame) -> None:
        if not math.isfinite(frame.t_sec) or not all(math.isfinite(v) for v in frame.wheels):
            raise ValueError("frame contains a non-finite value")
        if self._last_t_sec is not None and frame.t_sec <= self._last_t_sec:
            raise ValueError("frame times must be strictly increasing")
