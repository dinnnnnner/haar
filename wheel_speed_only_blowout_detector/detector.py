from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from statistics import median
from typing import Iterable, Sequence


WHEEL_NAMES = ("FL", "FR", "RL", "RR")
WHEEL_COUNT = 4
DIAGONALS = ((0, 3), (1, 2))
OPPOSITE_DIAGONAL = ((1, 2), (0, 3), (0, 3), (1, 2))
DIAGONAL_SIGN = (1.0, -1.0, -1.0, 1.0)


@dataclass(frozen=True)
class WheelSpeedBlowoutConfig:
    """Settings for corrected 100 Hz wheel speeds.

    All gain and edge thresholds are natural-log ratios.  For small changes,
    0.006 is approximately 0.6 percent.
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
        if self.sample_rate_hz <= 0.0 or self.min_avg_speed <= 0.0:
            raise ValueError("sample_rate_hz and min_avg_speed must be positive")
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
        if not 0.0 <= self.min_persistence_fraction <= 1.0:
            raise ValueError("min_persistence_fraction must be in [0, 1]")
        if self.max_individual_peak <= self.min_individual_peak:
            raise ValueError("individual peak limits are invalid")
        if self.max_diagonal_peak <= self.min_diagonal_peak:
            raise ValueError("diagonal peak limits are invalid")
        if self.clear_after_invalid_frames <= 0:
            raise ValueError("clear_after_invalid_frames must be positive")


@dataclass(frozen=True)
class WheelSpeedFrame:
    """One timestamped FL, FR, RL, RR wheel-speed sample."""

    t_sec: float
    wheels: tuple[float, float, float, float]

    @classmethod
    def from_sequences(
        cls, t_sec: float, wheels: Sequence[float]
    ) -> WheelSpeedFrame:
        if len(wheels) != WHEEL_COUNT:
            raise ValueError("expected four wheel speeds in FL, FR, RL, RR order")
        return cls(
            t_sec=float(t_sec),
            wheels=tuple(float(value) for value in wheels),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class WheelSpeedResult:
    t_sec: float
    wheels: tuple[float, float, float, float]
    speed_valid: bool
    warmed_up: bool
    detection_available: tuple[bool, bool, bool, bool]
    individual_gains: tuple[float, float, float, float]
    individual_edges: tuple[float, float, float, float]
    diagonal_gains: tuple[float, float, float, float]
    diagonal_edges: tuple[float, float, float, float]
    states: tuple[str, str, str, str]
    candidates: tuple[bool, bool, bool, bool]
    new_blowouts: tuple[bool, bool, bool, bool]
    blowout_alarms: tuple[bool, bool, bool, bool]
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


class WheelSpeedBlowoutDetector:
    """Causal, latched blowout detector with no pressure-sensor input.

    A wheel is accepted only when two independent spatial projections agree:

    * its log speed rises relative to the opposite diagonal; and
    * the signed four-wheel diagonal residual rises in the same direction.

    The second projection cancels common acceleration, axle bias, and the
    first-order left/right component of steering.  A candidate must then stay
    elevated while its diagonal mate does not move strongly in the opposite
    direction and the common vehicle speed remains free of a large transient.
    """

    def __init__(self, cfg: WheelSpeedBlowoutConfig | None = None):
        self.cfg = cfg or WheelSpeedBlowoutConfig()
        self._frame_index = -1
        self._last_t_sec: float | None = None
        self._invalid_frames = 0
        self._raw = [deque(maxlen=self.cfg.smooth_window) for _ in range(4)]
        self._smooth = [deque(maxlen=self.cfg.smooth_window) for _ in range(4)]
        self._edges = [
            deque(maxlen=2 * self.cfg.edge_half_window) for _ in range(4)
        ]
        self._baselines = [deque(maxlen=self.cfg.baseline_window) for _ in range(4)]
        self._baseline_cache: list[float | None] = [None] * 4
        self._diag_raw: deque[float] = deque(maxlen=self.cfg.smooth_window)
        self._diag_smooth: deque[float] = deque(maxlen=self.cfg.smooth_window)
        self._diag_edges: deque[float] = deque(
            maxlen=2 * self.cfg.edge_half_window
        )
        self._diag_baseline: deque[float] = deque(maxlen=self.cfg.baseline_window)
        self._diag_baseline_cache: float | None = None
        self._candidates: list[_Candidate | None] = [None] * 4
        self._alarms = [False] * 4
        self._onset_indices: list[int | None] = [None] * 4
        self._onset_times: list[float | None] = [None] * 4

    def reset(self) -> None:
        """Clear histories, candidates, and latched alarms."""
        self.__init__(self.cfg)

    def push(self, frame: WheelSpeedFrame) -> WheelSpeedResult:
        self._validate(frame)
        self._frame_index += 1
        self._last_t_sec = frame.t_sec
        wheels = tuple(abs(value) for value in frame.wheels)
        speed_valid = min(wheels) > 1.0e-9 and sum(wheels) / 4 >= self.cfg.min_avg_speed
        if not speed_valid:
            self._handle_invalid()
            return self._result(frame, False, [math.nan] * 4, [math.nan] * 4,
                                [math.nan] * 4, [math.nan] * 4, [False] * 4)

        self._invalid_frames = 0
        logs = tuple(math.log(value) for value in wheels)
        common_log_speed = sum(logs) / 4
        individual_raw = tuple(
            logs[wheel]
            - sum(logs[peer] for peer in OPPOSITE_DIAGONAL[wheel]) / 2
            for wheel in range(4)
        )
        diagonal_raw = logs[0] + logs[3] - logs[1] - logs[2]

        individual_smoothed = [
            self._smooth_value(value, self._raw[index], self._smooth[index])
            for index, value in enumerate(individual_raw)
        ]
        diagonal_smoothed = self._smooth_value(
            diagonal_raw, self._diag_raw, self._diag_smooth
        )
        individual_baselines = [self._current_baseline(i) for i in range(4)]
        diagonal_baseline = self._current_diagonal_baseline()
        warmed_up = diagonal_baseline is not None and all(
            value is not None for value in individual_baselines
        )

        gains = [math.nan] * 4
        edges = [math.nan] * 4
        diagonal_gains = [math.nan] * 4
        diagonal_edges = [math.nan] * 4
        new = [False] * 4

        if warmed_up:
            assert diagonal_baseline is not None
            signed_diagonal_delta = diagonal_smoothed - diagonal_baseline
            self._diag_edges.append(signed_diagonal_delta)
            unsigned_diagonal_edge = self._edge(self._diag_edges)
            for wheel in range(4):
                baseline = individual_baselines[wheel]
                assert baseline is not None
                gains[wheel] = individual_smoothed[wheel] - baseline
                self._edges[wheel].append(gains[wheel])
                edges[wheel] = self._edge(self._edges[wheel])
                diagonal_gains[wheel] = DIAGONAL_SIGN[wheel] * signed_diagonal_delta
                diagonal_edges[wheel] = DIAGONAL_SIGN[wheel] * unsigned_diagonal_edge

            availability = self._availability(speed_valid=True)
            for wheel in range(4):
                if not availability[wheel]:
                    self._candidates[wheel] = None
                    continue
                mate = DIAGONALS[0 if wheel in DIAGONALS[0] else 1]
                mate_wheel = mate[1] if mate[0] == wheel else mate[0]
                self._advance(
                    wheel,
                    gains[wheel],
                    edges[wheel],
                    diagonal_gains[wheel],
                    diagonal_edges[wheel],
                    gains[mate_wheel],
                    common_log_speed,
                    new,
                )

        if not warmed_up:
            for wheel in range(4):
                self._baselines[wheel].append(individual_smoothed[wheel])
            self._diag_baseline.append(diagonal_smoothed)
        elif not any(self._candidates) and not any(self._alarms):
            for wheel in range(4):
                self._baselines[wheel].append(individual_smoothed[wheel])
            self._diag_baseline.append(diagonal_smoothed)

        return self._result(
            frame,
            True,
            gains,
            edges,
            diagonal_gains,
            diagonal_edges,
            new,
        )

    def _advance(
        self,
        wheel: int,
        gain: float,
        edge: float,
        diagonal_gain: float,
        diagonal_edge: float,
        mate_gain: float,
        common_log_speed: float,
        new: list[bool],
    ) -> None:
        if self._alarms[wheel] or not all(
            math.isfinite(value)
            for value in (gain, edge, diagonal_gain, diagonal_edge, mate_gain)
        ):
            return
        candidate = self._candidates[wheel]
        if candidate is None:
            if (
                edge >= self.cfg.min_individual_edge
                and diagonal_edge >= self.cfg.min_diagonal_edge
            ):
                delay = self.cfg.smooth_window - 1 + self.cfg.edge_half_window
                onset_time = self._last_t_sec - delay / self.cfg.sample_rate_hz
                self._candidates[wheel] = _Candidate(
                    onset_index=max(0, self._frame_index - delay),
                    onset_time_s=onset_time,  # type: ignore[arg-type]
                    individual_values=[gain],
                    diagonal_values=[diagonal_gain],
                    mate_values=[mate_gain],
                    common_log_speeds=[common_log_speed],
                    max_individual=gain,
                    max_diagonal=diagonal_gain,
                )
            return

        candidate.individual_values.append(gain)
        candidate.diagonal_values.append(diagonal_gain)
        candidate.mate_values.append(mate_gain)
        candidate.common_log_speeds.append(common_log_speed)
        candidate.max_individual = max(candidate.max_individual, gain)
        candidate.max_diagonal = max(candidate.max_diagonal, diagonal_gain)
        if (
            gain < self.cfg.candidate_drop_limit
            or diagonal_gain < self.cfg.candidate_drop_limit
            or candidate.max_individual > self.cfg.max_individual_peak
            or candidate.max_diagonal > self.cfg.max_diagonal_peak
        ):
            self._candidates[wheel] = None
            return
        if len(candidate.individual_values) < self.cfg.confirm_frames:
            return

        tail = self.cfg.persistence_tail_frames
        individual_tail = candidate.individual_values[-tail:]
        diagonal_tail = candidate.diagonal_values[-tail:]
        mate_tail = candidate.mate_values[-tail:]
        common_range = max(candidate.common_log_speeds) - min(
            candidate.common_log_speeds
        )
        confirmed = (
            candidate.max_individual >= self.cfg.min_individual_peak
            and candidate.max_diagonal >= self.cfg.min_diagonal_peak
            and median(individual_tail) >= self.cfg.min_individual_persistence
            and median(diagonal_tail) >= self.cfg.min_diagonal_persistence
            and self._above_fraction(individual_tail)
            >= self.cfg.min_persistence_fraction
            and self._above_fraction(diagonal_tail)
            >= self.cfg.min_persistence_fraction
            and median(mate_tail) >= self.cfg.min_mate_persistence
            and common_range <= self.cfg.max_common_speed_range
        )
        if confirmed:
            self._alarms[wheel] = True
            self._onset_indices[wheel] = candidate.onset_index
            self._onset_times[wheel] = candidate.onset_time_s
            new[wheel] = True
        self._candidates[wheel] = None

    def _availability(self, speed_valid: bool) -> tuple[bool, bool, bool, bool]:
        if not speed_valid:
            return (False, False, False, False)
        return tuple(
            not any(self._alarms[peer] for peer in OPPOSITE_DIAGONAL[wheel])
            for wheel in range(4)
        )  # type: ignore[return-value]

    def _handle_invalid(self) -> None:
        self._invalid_frames += 1
        self._candidates = [None] * 4
        for window in (*self._edges, self._diag_edges):
            window.clear()
        if self._invalid_frames < self.cfg.clear_after_invalid_frames:
            return
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
    def _smooth_value(
        value: float, raw: deque[float], smooth: deque[float]
    ) -> float:
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

    def _result(
        self,
        frame: WheelSpeedFrame,
        speed_valid: bool,
        gains: list[float],
        edges: list[float],
        diagonal_gains: list[float],
        diagonal_edges: list[float],
        new: list[bool],
    ) -> WheelSpeedResult:
        warmed_up = (
            speed_valid
            and self._diag_baseline_cache is not None
            and all(value is not None for value in self._baseline_cache)
        )
        availability = self._availability(speed_valid and warmed_up)
        states = []
        for wheel in range(4):
            if self._alarms[wheel]:
                states.append("alarm")
            elif not speed_valid:
                states.append("invalid")
            elif not warmed_up:
                states.append("warming")
            elif not availability[wheel]:
                states.append("reference_contaminated")
            elif self._candidates[wheel] is not None:
                states.append("candidate")
            else:
                states.append("monitoring")
        return WheelSpeedResult(
            t_sec=frame.t_sec,
            wheels=frame.wheels,
            speed_valid=speed_valid,
            warmed_up=warmed_up,
            detection_available=availability,
            individual_gains=tuple(gains),  # type: ignore[arg-type]
            individual_edges=tuple(edges),  # type: ignore[arg-type]
            diagonal_gains=tuple(diagonal_gains),  # type: ignore[arg-type]
            diagonal_edges=tuple(diagonal_edges),  # type: ignore[arg-type]
            states=tuple(states),  # type: ignore[arg-type]
            candidates=tuple(value is not None for value in self._candidates),  # type: ignore[arg-type]
            new_blowouts=tuple(new),  # type: ignore[arg-type]
            blowout_alarms=tuple(self._alarms),  # type: ignore[arg-type]
            estimated_onset_indices=tuple(self._onset_indices),  # type: ignore[arg-type]
            estimated_onset_times_s=tuple(self._onset_times),  # type: ignore[arg-type]
        )

    def _validate(self, frame: WheelSpeedFrame) -> None:
        if not math.isfinite(frame.t_sec) or not all(
            math.isfinite(value) for value in frame.wheels
        ):
            raise ValueError("frame contains a non-finite value")
        if self._last_t_sec is not None and frame.t_sec <= self._last_t_sec:
            raise ValueError("frame times must be strictly increasing")


def run_detection(
    frames: Iterable[WheelSpeedFrame],
    cfg: WheelSpeedBlowoutConfig | None = None,
) -> list[WheelSpeedResult]:
    detector = WheelSpeedBlowoutDetector(cfg)
    return [detector.push(frame) for frame in frames]
