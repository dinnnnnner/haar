from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from statistics import median
from typing import Iterable, Sequence


WHEEL_NAMES = ("FL", "FR", "RL", "RR")
WHEEL_COUNT = 4
OPPOSITE_DIAGONAL = ((1, 2), (0, 3), (0, 3), (1, 2))
TEMPLATES = (
    (1.0, 1.0, 1.0),
    (-1.0, 1.0, -1.0),
    (1.0, -1.0, -1.0),
    (-1.0, -1.0, 1.0),
)


@dataclass(frozen=True)
class QuantBlowoutConfig:
    """Configuration for the online factor/CUSUM detector at 100 Hz."""

    sample_rate_hz: float = 100.0
    min_avg_speed: float = 20.0
    smooth_window: int = 5
    edge_half_window: int = 6
    warmup_frames: int = 300
    mean_alpha: float = 0.0010
    level_cov_alpha: float = 0.0040
    edge_cov_alpha: float = 0.0120
    covariance_refresh_frames: int = 10
    covariance_shrinkage: float = 0.25
    level_variance_floor: float = 2.5e-8
    edge_variance_floor: float = 1.0e-8

    shock_trigger_z: float = 5.0
    shock_isolation_z: float = 2.0
    min_physical_edge: float = 0.0038
    cusum_decay: float = 0.94
    cusum_drift_z: float = 1.0
    persistence_decay: float = 0.985
    persistence_drift_z: float = 0.5

    confirm_frames: int = 55
    persistence_tail_frames: int = 40
    candidate_timeout_frames: int = 120
    min_physical_peak: float = 0.0060
    max_physical_peak: float = 0.0250
    min_physical_persistence: float = 0.0042
    physical_persistence_floor: float = 0.0028
    min_persistence_fraction: float = 0.75
    min_level_isolation_z: float = 1.0
    level_isolation_floor_z: float = 1.0
    min_isolation_fraction: float = 0.95
    min_median_level_z: float = 1.5
    min_median_risk: float = 55.0
    min_peak_risk: float = 82.0
    max_common_log_range: float = 0.050
    reset_physical_level: float = -0.0025
    reset_below_frames: int = 15
    clear_after_invalid_frames: int = 50

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0 or self.min_avg_speed <= 0:
            raise ValueError("sample_rate_hz and min_avg_speed must be positive")
        if self.smooth_window <= 0 or self.smooth_window % 2 == 0:
            raise ValueError("smooth_window must be a positive odd integer")
        if self.edge_half_window <= 0 or self.warmup_frames < 20:
            raise ValueError("feature windows are invalid")
        for name in ("mean_alpha", "level_cov_alpha", "edge_cov_alpha"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if not 0.0 <= self.covariance_shrinkage <= 1.0:
            raise ValueError("covariance_shrinkage must be in [0, 1]")
        if self.covariance_refresh_frames <= 0:
            raise ValueError("covariance_refresh_frames must be positive")
        if not 1 <= self.persistence_tail_frames <= self.confirm_frames:
            raise ValueError("persistence_tail_frames is outside confirm_frames")
        if self.candidate_timeout_frames < self.confirm_frames:
            raise ValueError("candidate timeout must cover confirmation")
        if self.max_physical_peak <= self.min_physical_peak:
            raise ValueError("physical peak limits are invalid")
        if not 0.0 <= self.min_persistence_fraction <= 1.0:
            raise ValueError("min_persistence_fraction must be in [0, 1]")
        if not 0.0 <= self.min_isolation_fraction <= 1.0:
            raise ValueError("min_isolation_fraction must be in [0, 1]")


@dataclass(frozen=True)
class QuantFrame:
    t_sec: float
    wheels: tuple[float, float, float, float]

    @classmethod
    def from_sequences(cls, t_sec: float, wheels: Sequence[float]) -> QuantFrame:
        if len(wheels) != WHEEL_COUNT:
            raise ValueError("expected four wheel speeds in FL, FR, RL, RR order")
        return cls(
            float(t_sec),
            tuple(float(value) for value in wheels),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class QuantResult:
    t_sec: float
    wheels: tuple[float, float, float, float]
    speed_valid: bool
    warmed_up: bool
    market_factors: tuple[float, float, float]
    factor_residuals: tuple[float, float, float]
    factor_edges: tuple[float, float, float]
    shock_z_scores: tuple[float, float, float, float]
    level_z_scores: tuple[float, float, float, float]
    shock_isolation: tuple[float, float, float, float]
    level_isolation: tuple[float, float, float, float]
    physical_levels: tuple[float, float, float, float]
    physical_edges: tuple[float, float, float, float]
    cusum_scores: tuple[float, float, float, float]
    persistence_scores: tuple[float, float, float, float]
    risk_scores: tuple[float, float, float, float]
    states: tuple[str, str, str, str]
    new_blowouts: tuple[bool, bool, bool, bool]
    blowout_alarms: tuple[bool, bool, bool, bool]
    leading_wheel: int | None
    leading_margin: float
    estimated_onset_indices: tuple[int | None, int | None, int | None, int | None]
    estimated_onset_times_s: tuple[float | None, float | None, float | None, float | None]


class _OnlineMoments:
    def __init__(self, mean_alpha: float, cov_alpha: float) -> None:
        self.mean_alpha = mean_alpha
        self.cov_alpha = cov_alpha
        self.samples = 0
        self.mean = [0.0, 0.0, 0.0]
        self.cov = [[0.0] * 3 for _ in range(3)]

    def update(self, value: Sequence[float]) -> None:
        self.samples += 1
        if self.samples == 1:
            self.mean[:] = value
            return
        mean_alpha = max(self.mean_alpha, 1.0 / self.samples) if self.samples < 100 else self.mean_alpha
        previous_mean = self.mean[:]
        delta = [value[i] - previous_mean[i] for i in range(3)]
        for i in range(3):
            self.mean[i] += mean_alpha * delta[i]
        cov_alpha = max(self.cov_alpha, 1.0 / self.samples) if self.samples < 100 else self.cov_alpha
        for i in range(3):
            for j in range(3):
                innovation = delta[i] * (value[j] - self.mean[j])
                self.cov[i][j] = (1.0 - cov_alpha) * self.cov[i][j] + cov_alpha * innovation

    def clear(self) -> None:
        self.samples = 0
        self.mean[:] = (0.0, 0.0, 0.0)
        for row in self.cov:
            row[:] = (0.0, 0.0, 0.0)


@dataclass
class _WheelState:
    phase: str = "warming"
    cusum: float = 0.0
    persistence: float = 0.0
    risk: float = 0.0
    alarm: bool = False
    onset_index: int | None = None
    onset_time_s: float | None = None
    candidate_frames: int = 0
    below_frames: int = 0
    peak_physical: float = 0.0
    peak_risk: float = 0.0
    physical_history: list[float] = field(default_factory=list)
    level_z_history: list[float] = field(default_factory=list)
    isolation_history: list[float] = field(default_factory=list)
    risk_history: list[float] = field(default_factory=list)
    common_history: list[float] = field(default_factory=list)


def _inverse_covariance(
    covariance: Sequence[Sequence[float]], floor: float, shrinkage: float
) -> tuple[tuple[float, float, float], ...]:
    matrix = [[float(covariance[i][j]) for j in range(3)] for i in range(3)]
    for i in range(3):
        matrix[i][i] = max(matrix[i][i], floor)
        for j in range(3):
            if i != j:
                matrix[i][j] *= 1.0 - shrinkage
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if not math.isfinite(determinant) or abs(determinant) < floor**3 * 1.0e-6:
        return (
            (1.0 / matrix[0][0], 0.0, 0.0),
            (0.0, 1.0 / matrix[1][1], 0.0),
            (0.0, 0.0, 1.0 / matrix[2][2]),
        )
    inverse_det = 1.0 / determinant
    return (
        ((e * i - f * h) * inverse_det, (c * h - b * i) * inverse_det, (b * f - c * e) * inverse_det),
        ((f * g - d * i) * inverse_det, (a * i - c * g) * inverse_det, (c * d - a * f) * inverse_det),
        ((d * h - e * g) * inverse_det, (b * g - a * h) * inverse_det, (a * e - b * d) * inverse_det),
    )


def _matvec(matrix: Sequence[Sequence[float]], value: Sequence[float]) -> list[float]:
    return [sum(matrix[i][j] * value[j] for j in range(3)) for i in range(3)]


def _matched_scores(
    value: Sequence[float], inverse: Sequence[Sequence[float]]
) -> tuple[float, float, float, float]:
    weighted = _matvec(inverse, value)
    scores = []
    for template in TEMPLATES:
        numerator = sum(template[i] * weighted[i] for i in range(3))
        template_weighted = _matvec(inverse, template)
        denominator = math.sqrt(
            max(1.0e-12, sum(template[i] * template_weighted[i] for i in range(3)))
        )
        scores.append(numerator / denominator)
    return tuple(scores)  # type: ignore[return-value]


def _physical_projections(value: Sequence[float]) -> tuple[float, float, float, float]:
    return tuple(
        sum(template[index] * value[index] for index in range(3))
        for template in TEMPLATES
    )  # type: ignore[return-value]


def _isolations(scores: Sequence[float]) -> tuple[float, float, float, float]:
    return tuple(
        scores[wheel] - max(scores[peer] for peer in range(4) if peer != wheel)
        for wheel in range(4)
    )  # type: ignore[return-value]


def _sigmoid(value: float) -> float:
    if value >= 40.0:
        return 1.0
    if value <= -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


class QuantBlowoutDetector:
    """Adaptive factor-neutral matched-filter and CUSUM detector."""

    def __init__(self, cfg: QuantBlowoutConfig | None = None) -> None:
        self.cfg = cfg or QuantBlowoutConfig()
        self._frame_index = -1
        self._last_t_sec: float | None = None
        self._invalid_frames = 0
        self._raw_logs = [deque(maxlen=self.cfg.smooth_window) for _ in range(4)]
        self._smooth_logs = [deque(maxlen=self.cfg.smooth_window) for _ in range(4)]
        self._factor_edges: deque[tuple[float, float, float]] = deque(
            maxlen=2 * self.cfg.edge_half_window
        )
        self._level_model = _OnlineMoments(self.cfg.mean_alpha, self.cfg.level_cov_alpha)
        self._edge_model = _OnlineMoments(self.cfg.mean_alpha, self.cfg.edge_cov_alpha)
        self._level_inverse = _inverse_covariance(
            self._level_model.cov, self.cfg.level_variance_floor, self.cfg.covariance_shrinkage
        )
        self._edge_inverse = _inverse_covariance(
            self._edge_model.cov, self.cfg.edge_variance_floor, self.cfg.covariance_shrinkage
        )
        self._states = [_WheelState() for _ in range(4)]

    def reset(self) -> None:
        self.__init__(self.cfg)

    def push(self, frame: QuantFrame) -> QuantResult:
        self._validate(frame)
        self._frame_index += 1
        self._last_t_sec = frame.t_sec
        wheels = tuple(abs(value) for value in frame.wheels)
        speed_valid = min(wheels) > 1.0e-9 and sum(wheels) / 4 >= self.cfg.min_avg_speed
        if not speed_valid:
            self._handle_invalid()
            return self._empty_result(frame)
        self._invalid_frames = 0

        logs = [math.log(value) for value in wheels]
        smoothed_logs = [self._smooth_log(index, value) for index, value in enumerate(logs)]
        factors = self._factors(smoothed_logs)
        self._factor_edges.append(factors)
        edge = self._edge()
        warmed_up = (
            self._level_model.samples >= self.cfg.warmup_frames
            and self._edge_model.samples >= self.cfg.warmup_frames - 2 * self.cfg.edge_half_window
            and edge is not None
        )

        if not warmed_up:
            self._level_model.update(factors)
            if edge is not None:
                self._edge_model.update(edge)
            for state in self._states:
                if not state.alarm:
                    state.phase = "warming"
            return self._result(
                frame, factors, (math.nan,) * 3, (math.nan,) * 3,
                (math.nan,) * 4, (math.nan,) * 4, (math.nan,) * 4,
                (math.nan,) * 4, (math.nan,) * 4, (math.nan,) * 4,
                [False] * 4, False,
            )

        assert edge is not None
        if self._frame_index % self.cfg.covariance_refresh_frames == 0:
            self._refresh_inverses()
        level_residual = tuple(factors[i] - self._level_model.mean[i] for i in range(3))
        edge_residual = tuple(edge[i] - self._edge_model.mean[i] for i in range(3))
        level_z = _matched_scores(level_residual, self._level_inverse)
        shock_z = _matched_scores(edge_residual, self._edge_inverse)
        level_isolation = _isolations(level_z)
        shock_isolation = _isolations(shock_z)
        physical_level = _physical_projections(level_residual)
        physical_edge = _physical_projections(edge_residual)
        common_log_speed = sum(smoothed_logs) / 4
        new = [False] * 4

        for wheel in range(4):
            if not self._available(wheel):
                self._clear_candidate(wheel)
                continue
            self._advance(
                wheel,
                shock_z[wheel],
                level_z[wheel],
                shock_isolation[wheel],
                level_isolation[wheel],
                physical_edge[wheel],
                physical_level[wheel],
                common_log_speed,
                new,
            )

        if not any(state.phase == "candidate" for state in self._states) and not any(
            state.alarm for state in self._states
        ):
            self._level_model.update(factors)
            self._edge_model.update(edge)

        return self._result(
            frame,
            factors,
            level_residual,
            edge_residual,
            shock_z,
            level_z,
            shock_isolation,
            level_isolation,
            physical_level,
            physical_edge,
            new,
            True,
        )

    def _advance(
        self,
        wheel: int,
        shock_z: float,
        level_z: float,
        shock_isolation: float,
        level_isolation: float,
        physical_edge: float,
        physical_level: float,
        common_log_speed: float,
        new: list[bool],
    ) -> None:
        state = self._states[wheel]
        if state.alarm:
            return
        state.cusum = max(
            0.0,
            self.cfg.cusum_decay * state.cusum
            + max(0.0, shock_z - self.cfg.cusum_drift_z),
        )
        state.persistence = max(
            0.0,
            self.cfg.persistence_decay * state.persistence
            + max(0.0, level_z - self.cfg.persistence_drift_z),
        )
        state.risk = self._risk(
            shock_z,
            level_z,
            shock_isolation,
            level_isolation,
            state.cusum,
            state.persistence,
        )
        if state.phase != "candidate":
            state.phase = "monitoring"
            if (
                shock_z >= self.cfg.shock_trigger_z
                and shock_isolation >= self.cfg.shock_isolation_z
                and physical_edge >= self.cfg.min_physical_edge
            ):
                delay = self.cfg.smooth_window - 1 + self.cfg.edge_half_window
                state.phase = "candidate"
                state.onset_index = max(0, self._frame_index - delay)
                state.onset_time_s = self._last_t_sec - delay / self.cfg.sample_rate_hz  # type: ignore[operator]
                self._start_candidate(state)
            return

        state.candidate_frames += 1
        state.peak_physical = max(state.peak_physical, physical_level)
        state.peak_risk = max(state.peak_risk, state.risk)
        state.physical_history.append(physical_level)
        state.level_z_history.append(level_z)
        state.isolation_history.append(level_isolation)
        state.risk_history.append(state.risk)
        state.common_history.append(common_log_speed)
        state.below_frames = state.below_frames + 1 if physical_level < self.cfg.reset_physical_level else 0
        if (
            state.below_frames >= self.cfg.reset_below_frames
            or state.peak_physical > self.cfg.max_physical_peak
            or state.candidate_frames >= self.cfg.candidate_timeout_frames
        ):
            self._clear_candidate(wheel)
            return
        if state.candidate_frames < self.cfg.confirm_frames:
            return
        tail = self.cfg.persistence_tail_frames
        physical_tail = state.physical_history[-tail:]
        z_tail = state.level_z_history[-tail:]
        isolation_tail = state.isolation_history[-tail:]
        risk_tail = state.risk_history[-tail:]
        common_range = max(state.common_history) - min(state.common_history)
        confirmed = (
            state.peak_physical >= self.cfg.min_physical_peak
            and median(physical_tail) >= self.cfg.min_physical_persistence
            and sum(value >= self.cfg.physical_persistence_floor for value in physical_tail) / len(physical_tail)
            >= self.cfg.min_persistence_fraction
            and median(z_tail) >= self.cfg.min_median_level_z
            and median(isolation_tail) >= self.cfg.min_level_isolation_z
            and sum(
                value >= self.cfg.level_isolation_floor_z
                for value in isolation_tail
            )
            / len(isolation_tail)
            >= self.cfg.min_isolation_fraction
            and median(risk_tail) >= self.cfg.min_median_risk
            and state.peak_risk >= self.cfg.min_peak_risk
            and common_range <= self.cfg.max_common_log_range
        )
        if confirmed:
            state.phase = "alarm"
            state.alarm = True
            new[wheel] = True
        else:
            return

    @staticmethod
    def _start_candidate(state: _WheelState) -> None:
        state.candidate_frames = 0
        state.below_frames = 0
        state.peak_physical = 0.0
        state.peak_risk = state.risk
        state.physical_history.clear()
        state.level_z_history.clear()
        state.isolation_history.clear()
        state.risk_history.clear()
        state.common_history.clear()

    def _clear_candidate(self, wheel: int) -> None:
        state = self._states[wheel]
        if state.alarm:
            return
        state.phase = "monitoring"
        state.candidate_frames = 0
        state.below_frames = 0
        state.physical_history.clear()
        state.level_z_history.clear()
        state.isolation_history.clear()
        state.risk_history.clear()
        state.common_history.clear()

    @staticmethod
    def _risk(
        shock_z: float,
        level_z: float,
        shock_isolation: float,
        level_isolation: float,
        cusum: float,
        persistence: float,
    ) -> float:
        composite = (
            0.12 * shock_z
            + 0.36 * level_z
            + 0.08 * shock_isolation
            + 0.16 * level_isolation
            + 0.10 * min(cusum / 6.0, 8.0)
            + 0.18 * min(persistence / 20.0, 8.0)
        )
        return 100.0 * _sigmoid((composite - 3.0) / 1.15)

    def _available(self, wheel: int) -> bool:
        return not any(self._states[peer].alarm for peer in OPPOSITE_DIAGONAL[wheel])

    def _smooth_log(self, wheel: int, value: float) -> float:
        self._raw_logs[wheel].append(value)
        filtered = float(median(self._raw_logs[wheel]))
        self._smooth_logs[wheel].append(filtered)
        return sum(self._smooth_logs[wheel]) / len(self._smooth_logs[wheel])

    @staticmethod
    def _factors(logs: Sequence[float]) -> tuple[float, float, float]:
        fl, fr, rl, rr = logs
        return (
            (fl - fr + rl - rr) / 4.0,
            (fl + fr - rl - rr) / 4.0,
            (fl - fr - rl + rr) / 4.0,
        )

    def _edge(self) -> tuple[float, float, float] | None:
        half = self.cfg.edge_half_window
        if len(self._factor_edges) < 2 * half:
            return None
        values = list(self._factor_edges)
        return tuple(
            sum(row[factor] for row in values[half:]) / half
            - sum(row[factor] for row in values[:half]) / half
            for factor in range(3)
        )  # type: ignore[return-value]

    def _refresh_inverses(self) -> None:
        self._level_inverse = _inverse_covariance(
            self._level_model.cov,
            self.cfg.level_variance_floor,
            self.cfg.covariance_shrinkage,
        )
        self._edge_inverse = _inverse_covariance(
            self._edge_model.cov,
            self.cfg.edge_variance_floor,
            self.cfg.covariance_shrinkage,
        )

    def _handle_invalid(self) -> None:
        self._invalid_frames += 1
        for wheel in range(4):
            self._clear_candidate(wheel)
        if self._invalid_frames < self.cfg.clear_after_invalid_frames:
            return
        for window in (*self._raw_logs, *self._smooth_logs, self._factor_edges):
            window.clear()
        self._level_model.clear()
        self._edge_model.clear()

    def _empty_result(self, frame: QuantFrame) -> QuantResult:
        nan3 = (math.nan, math.nan, math.nan)
        nan4 = (math.nan, math.nan, math.nan, math.nan)
        return self._result(
            frame, nan3, nan3, nan3, nan4, nan4, nan4, nan4, nan4, nan4,
            [False] * 4, False,
        )

    def _result(
        self,
        frame: QuantFrame,
        factors: Sequence[float],
        factor_residuals: Sequence[float],
        factor_edges: Sequence[float],
        shock_z: Sequence[float],
        level_z: Sequence[float],
        shock_isolation: Sequence[float],
        level_isolation: Sequence[float],
        physical_level: Sequence[float],
        physical_edge: Sequence[float],
        new: list[bool],
        warmed_up: bool,
    ) -> QuantResult:
        risks = tuple(state.risk for state in self._states)
        if warmed_up:
            leading = max(range(4), key=risks.__getitem__)
            runner_up = max(risks[index] for index in range(4) if index != leading)
            leading_margin = risks[leading] - runner_up
        else:
            leading = None
            leading_margin = math.nan
        return QuantResult(
            frame.t_sec,
            frame.wheels,
            all(abs(value) > 1.0e-9 for value in frame.wheels)
            and sum(abs(value) for value in frame.wheels) / 4 >= self.cfg.min_avg_speed,
            warmed_up,
            tuple(factors),  # type: ignore[arg-type]
            tuple(factor_residuals),  # type: ignore[arg-type]
            tuple(factor_edges),  # type: ignore[arg-type]
            tuple(shock_z),  # type: ignore[arg-type]
            tuple(level_z),  # type: ignore[arg-type]
            tuple(shock_isolation),  # type: ignore[arg-type]
            tuple(level_isolation),  # type: ignore[arg-type]
            tuple(physical_level),  # type: ignore[arg-type]
            tuple(physical_edge),  # type: ignore[arg-type]
            tuple(state.cusum for state in self._states),  # type: ignore[arg-type]
            tuple(state.persistence for state in self._states),  # type: ignore[arg-type]
            risks,  # type: ignore[arg-type]
            tuple(state.phase for state in self._states),  # type: ignore[arg-type]
            tuple(new),  # type: ignore[arg-type]
            tuple(state.alarm for state in self._states),  # type: ignore[arg-type]
            leading,
            leading_margin,
            tuple(state.onset_index for state in self._states),  # type: ignore[arg-type]
            tuple(state.onset_time_s for state in self._states),  # type: ignore[arg-type]
        )

    def _validate(self, frame: QuantFrame) -> None:
        if not math.isfinite(frame.t_sec) or not all(math.isfinite(value) for value in frame.wheels):
            raise ValueError("frame contains a non-finite value")
        if self._last_t_sec is not None and frame.t_sec <= self._last_t_sec:
            raise ValueError("frame times must be strictly increasing")


def run_detection(
    frames: Iterable[QuantFrame], cfg: QuantBlowoutConfig | None = None
) -> list[QuantResult]:
    detector = QuantBlowoutDetector(cfg)
    return [detector.push(frame) for frame in frames]
