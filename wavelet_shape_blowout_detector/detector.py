from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from statistics import median
from typing import Iterable, Sequence


WHEEL_COUNT = 4
WHEEL_NAMES = ("FL", "FR", "RL", "RR")
STATE_NAMES = ("warming", "idle", "wait_fall", "confirm", "alarm")
REFERENCE_MODES = ("opposite_diagonal", "peer_median")
OPPOSITE_DIAGONAL_REFERENCES = (
    (1, 2),  # FL uses FR + RL.
    (0, 3),  # FR uses FL + RR.
    (0, 3),  # RL uses FL + RR.
    (1, 2),  # RR uses FR + RL.
)


@dataclass(frozen=True)
class WaveletShapeConfig:
    """Configuration for the causal rise-fall-plateau detector.

    Defaults target the corrected 100 Hz wheel-speed series. Ratios and
    thresholds are dimensionless: 0.0055 means 0.55 percent.
    """

    sample_rate_hz: float = 100.0
    reference_mode: str = "opposite_diagonal"
    smooth_window: int = 5
    haar_half_window: int = 5
    baseline_window: int = 300
    baseline_min_samples: int = 100
    min_avg_speed: float = 20.0
    min_rise_coefficient: float = 0.0055
    min_peak_gain: float = 0.0040
    max_peak_gain: float = 0.0200
    min_fall_coefficient: float = 0.0033
    max_fall_coefficient: float = 0.0120
    min_valley_gain: float = 0.0
    fall_min_frames: int = 4
    fall_max_frames: int = 24
    confirm_frames: int = 30
    min_steady_gain: float = 0.0060
    steady_tail_frames: int = 10
    min_tail_gain: float = 0.0085
    max_steady_above_peak: float = 0.0010
    clear_baseline_after_invalid_frames: int = 50
    target_wheels: tuple[int, ...] = (0, 1, 2, 3)

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive")
        if self.reference_mode not in REFERENCE_MODES:
            raise ValueError(f"reference_mode must be one of {REFERENCE_MODES}")
        if self.smooth_window <= 0 or self.smooth_window % 2 == 0:
            raise ValueError("smooth_window must be a positive odd integer")
        if self.haar_half_window <= 0:
            raise ValueError("haar_half_window must be positive")
        if self.baseline_window < 2:
            raise ValueError("baseline_window must be at least 2")
        if not 2 <= self.baseline_min_samples <= self.baseline_window:
            raise ValueError(
                "baseline_min_samples must be between 2 and baseline_window"
            )
        if self.min_avg_speed <= 0.0:
            raise ValueError("min_avg_speed must be positive")
        if self.min_rise_coefficient <= 0.0:
            raise ValueError("min_rise_coefficient must be positive")
        if self.min_peak_gain < 0.0:
            raise ValueError("min_peak_gain must be non-negative")
        if self.max_peak_gain <= self.min_peak_gain:
            raise ValueError("max_peak_gain must exceed min_peak_gain")
        if self.min_fall_coefficient <= 0.0:
            raise ValueError("min_fall_coefficient must be positive")
        if self.max_fall_coefficient <= self.min_fall_coefficient:
            raise ValueError(
                "max_fall_coefficient must exceed min_fall_coefficient"
            )
        if not 0 <= self.fall_min_frames <= self.fall_max_frames:
            raise ValueError("fall frame limits are invalid")
        if self.confirm_frames <= 0:
            raise ValueError("confirm_frames must be positive")
        if self.min_steady_gain <= 0.0:
            raise ValueError("min_steady_gain must be positive")
        if not 1 <= self.steady_tail_frames <= self.confirm_frames:
            raise ValueError(
                "steady_tail_frames must be between 1 and confirm_frames"
            )
        if self.min_tail_gain <= 0.0:
            raise ValueError("min_tail_gain must be positive")
        if self.max_steady_above_peak < 0.0:
            raise ValueError("max_steady_above_peak must be non-negative")
        if self.clear_baseline_after_invalid_frames <= 0:
            raise ValueError("clear_baseline_after_invalid_frames must be positive")
        if len(set(self.target_wheels)) != len(self.target_wheels):
            raise ValueError("target_wheels must not contain duplicates")
        if any(index < 0 or index >= WHEEL_COUNT for index in self.target_wheels):
            raise ValueError("target_wheels contains an invalid wheel index")


@dataclass(frozen=True)
class WheelFrame:
    t_sec: float
    wheels: tuple[float, float, float, float]
    normal_signals: tuple[bool | None, bool | None, bool | None, bool | None] = (
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
        normal_signals: Sequence[bool | None] | None = None,
    ) -> WheelFrame:
        if len(wheels) != WHEEL_COUNT:
            raise ValueError(f"expected {WHEEL_COUNT} wheel speeds")
        signals = [None] * WHEEL_COUNT if normal_signals is None else list(normal_signals)
        if len(signals) != WHEEL_COUNT:
            raise ValueError(f"expected {WHEEL_COUNT} normal signal entries")
        return cls(
            t_sec=float(t_sec),
            wheels=tuple(float(value) for value in wheels),  # type: ignore[arg-type]
            normal_signals=tuple(
                None if value is None else bool(value) for value in signals
            ),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class WaveletShapeResult:
    t_sec: float
    wheels: tuple[float, float, float, float]
    speed_valid: bool
    normal_signals: tuple[bool | None, bool | None, bool | None, bool | None]
    reference_sources: tuple[str, str, str, str]
    target_peer_ratios: tuple[float, float, float, float]
    normalized_gains: tuple[float, float, float, float]
    haar_coefficients: tuple[float, float, float, float]
    states: tuple[str, str, str, str]
    shape_events: tuple[bool, bool, bool, bool]
    new_blowouts: tuple[bool, bool, bool, bool]
    blowout_alarms: tuple[bool, bool, bool, bool]
    estimated_onset_indices: tuple[int | None, int | None, int | None, int | None]
    estimated_onset_times_s: tuple[float | None, float | None, float | None, float | None]
    rise_coefficients: tuple[float, float, float, float]
    fall_coefficients: tuple[float, float, float, float]
    steady_gains: tuple[float, float, float, float]
    steady_tail_gains: tuple[float, float, float, float]


@dataclass
class _Candidate:
    phase: str
    start_index: int
    onset_index: int
    onset_time_s: float
    rise_coefficient: float
    peak_gain: float
    fall_coefficient: float = 0.0
    fall_index: int | None = None
    confirmation_gains: list[float] | None = None


@dataclass(frozen=True)
class _CompletedEvidence:
    event: bool
    rise_coefficient: float
    fall_coefficient: float
    steady_gain: float
    steady_tail_gain: float


class WaveletShapeBlowoutDetector:
    """Causal detector with a fast alarm and full shape verification.

    For each wheel, the signal is its speed divided by the median speed of the
    other three wheels. A rolling baseline removes static tyre-radius and turn
    offsets. The transition detector is a shift-invariant causal Haar
    coefficient: the mean of the newest half-window minus the preceding one.
    """

    def __init__(self, cfg: WaveletShapeConfig | None = None):
        self.cfg = cfg or WaveletShapeConfig()
        self._frame_index = -1
        self._last_t_sec: float | None = None
        self._invalid_frames = 0
        self._raw_ratio_windows = [
            deque(maxlen=self.cfg.smooth_window) for _ in range(WHEEL_COUNT)
        ]
        self._median_windows = [
            deque(maxlen=self.cfg.smooth_window) for _ in range(WHEEL_COUNT)
        ]
        self._gain_windows = [
            deque(maxlen=2 * self.cfg.haar_half_window)
            for _ in range(WHEEL_COUNT)
        ]
        self._baselines = [
            deque(maxlen=self.cfg.baseline_window) for _ in range(WHEEL_COUNT)
        ]
        history_length = self.cfg.smooth_window + self.cfg.haar_half_window + 2
        self._time_history: deque[tuple[int, float]] = deque(maxlen=history_length)
        self._candidates: list[_Candidate | None] = [None] * WHEEL_COUNT
        self._reference_keys: list[tuple[int, ...] | None] = [None] * WHEEL_COUNT
        self._alarms = [False] * WHEEL_COUNT
        self._onset_indices: list[int | None] = [None] * WHEEL_COUNT
        self._onset_times: list[float | None] = [None] * WHEEL_COUNT

    def reset(self) -> None:
        """Clear signal history and all latched alarms."""
        self.__init__(self.cfg)

    def push(self, frame: WheelFrame) -> WaveletShapeResult:
        self._validate_frame(frame)
        self._frame_index += 1
        self._last_t_sec = frame.t_sec
        self._time_history.append((self._frame_index, frame.t_sec))

        magnitudes = tuple(abs(value) for value in frame.wheels)
        speed_valid = (
            sum(magnitudes) / WHEEL_COUNT >= self.cfg.min_avg_speed
            and sum(value > 1.0e-9 for value in magnitudes) >= 3
        )
        if not speed_valid:
            self._handle_invalid_frame()
            return self._empty_result(frame, speed_valid=False)

        self._invalid_frames = 0
        ratios, reference_keys, reference_sources = self._target_peer_ratios(
            magnitudes, frame.normal_signals
        )
        gains = [math.nan] * WHEEL_COUNT
        coefficients = [math.nan] * WHEEL_COUNT
        shape_events = [False] * WHEEL_COUNT
        new_blowouts = [False] * WHEEL_COUNT
        rise_values = [0.0] * WHEEL_COUNT
        fall_values = [0.0] * WHEEL_COUNT
        steady_values = [math.nan] * WHEEL_COUNT
        steady_tail_values = [math.nan] * WHEEL_COUNT
        target_set = set(self.cfg.target_wheels)

        for wheel_index, ratio in enumerate(ratios):
            if reference_keys[wheel_index] != self._reference_keys[wheel_index]:
                self._reset_wheel_tracking(wheel_index, clear_baseline=True)
                self._reference_keys[wheel_index] = reference_keys[wheel_index]
            smoothed_ratio = self._smooth_ratio(wheel_index, ratio)
            baseline = self._baseline(wheel_index)
            if baseline is None:
                self._baselines[wheel_index].append(smoothed_ratio)
                continue

            gain = smoothed_ratio / baseline - 1.0
            gains[wheel_index] = gain
            gain_window = self._gain_windows[wheel_index]
            gain_window.append(gain)
            coefficient = self._haar_coefficient(gain_window)
            coefficients[wheel_index] = coefficient

            confirmed_normal = frame.normal_signals[wheel_index] is True
            if (
                wheel_index not in target_set
                or (
                    self._alarms[wheel_index]
                    and self._candidates[wheel_index] is None
                )
                or confirmed_normal
            ):
                if confirmed_normal:
                    self._clear_provisional_alarm(wheel_index)
                    self._candidates[wheel_index] = None
                if not self._alarms[wheel_index]:
                    self._baselines[wheel_index].append(smoothed_ratio)
                continue

            was_alarm = self._alarms[wheel_index]
            completed = self._advance_state(
                wheel_index, smoothed_ratio, gain, coefficient
            )
            if not was_alarm and self._alarms[wheel_index]:
                new_blowouts[wheel_index] = True
            candidate = self._candidates[wheel_index]
            if candidate is not None:
                rise_values[wheel_index] = candidate.rise_coefficient
                fall_values[wheel_index] = candidate.fall_coefficient
            if completed is not None:
                rise_values[wheel_index] = completed.rise_coefficient
                fall_values[wheel_index] = completed.fall_coefficient
                steady_values[wheel_index] = completed.steady_gain
                steady_tail_values[wheel_index] = completed.steady_tail_gain
            if completed is not None and completed.event:
                shape_events[wheel_index] = True

        states = tuple(self._state_name(index) for index in range(WHEEL_COUNT))
        return WaveletShapeResult(
            t_sec=frame.t_sec,
            wheels=frame.wheels,
            speed_valid=True,
            normal_signals=frame.normal_signals,
            reference_sources=reference_sources,
            target_peer_ratios=ratios,
            normalized_gains=tuple(gains),  # type: ignore[arg-type]
            haar_coefficients=tuple(coefficients),  # type: ignore[arg-type]
            states=states,  # type: ignore[arg-type]
            shape_events=tuple(shape_events),  # type: ignore[arg-type]
            new_blowouts=tuple(new_blowouts),  # type: ignore[arg-type]
            blowout_alarms=tuple(self._alarms),  # type: ignore[arg-type]
            estimated_onset_indices=tuple(self._onset_indices),  # type: ignore[arg-type]
            estimated_onset_times_s=tuple(self._onset_times),  # type: ignore[arg-type]
            rise_coefficients=tuple(rise_values),  # type: ignore[arg-type]
            fall_coefficients=tuple(fall_values),  # type: ignore[arg-type]
            steady_gains=tuple(steady_values),  # type: ignore[arg-type]
            steady_tail_gains=tuple(steady_tail_values),  # type: ignore[arg-type]
        )

    def _advance_state(
        self,
        wheel_index: int,
        smoothed_ratio: float,
        gain: float,
        coefficient: float,
    ) -> _CompletedEvidence | None:
        candidate = self._candidates[wheel_index]
        if candidate is None:
            if (
                coefficient >= self.cfg.min_rise_coefficient
                and gain >= self.cfg.min_peak_gain
            ):
                onset_index, onset_time = self._estimated_onset()
                self._candidates[wheel_index] = _Candidate(
                    phase="wait_fall",
                    start_index=self._frame_index,
                    onset_index=onset_index,
                    onset_time_s=onset_time,
                    rise_coefficient=coefficient,
                    peak_gain=gain,
                )
                # Raise the externally visible alarm as soon as the causal
                # rise is established.  The candidate remains active so the
                # original fall/plateau checks can either confirm the alarm
                # or automatically retract it as a false positive.
                self._alarms[wheel_index] = True
                self._onset_indices[wheel_index] = onset_index
                self._onset_times[wheel_index] = onset_time
            else:
                self._baselines[wheel_index].append(smoothed_ratio)
            return None

        if candidate.phase == "wait_fall":
            candidate.rise_coefficient = max(
                candidate.rise_coefficient, coefficient
            )
            candidate.peak_gain = max(candidate.peak_gain, gain)
            age = self._frame_index - candidate.start_index
            if (
                candidate.peak_gain > self.cfg.max_peak_gain
                or coefficient < -self.cfg.max_fall_coefficient
                or gain < self.cfg.min_valley_gain
            ):
                self._clear_provisional_alarm(wheel_index)
                self._candidates[wheel_index] = None
                return None
            if (
                age >= self.cfg.fall_min_frames
                and age <= self.cfg.fall_max_frames
                and coefficient <= -self.cfg.min_fall_coefficient
            ):
                candidate.phase = "confirm"
                candidate.fall_coefficient = coefficient
                candidate.fall_index = self._frame_index
                candidate.confirmation_gains = []
            elif age > self.cfg.fall_max_frames:
                self._clear_provisional_alarm(wheel_index)
                self._candidates[wheel_index] = None
                self._baselines[wheel_index].append(smoothed_ratio)
            return None

        if candidate.confirmation_gains is None or candidate.fall_index is None:
            raise RuntimeError("invalid confirm candidate")
        candidate.confirmation_gains.append(gain)
        if len(candidate.confirmation_gains) < self.cfg.confirm_frames:
            return None

        steady_gain = float(median(candidate.confirmation_gains))
        steady_tail_gain = float(
            median(candidate.confirmation_gains[-self.cfg.steady_tail_frames :])
        )
        event = (
            steady_gain >= self.cfg.min_steady_gain
            and steady_tail_gain >= self.cfg.min_tail_gain
            and steady_gain
            <= candidate.peak_gain + self.cfg.max_steady_above_peak
        )
        if event:
            self._alarms[wheel_index] = True
            self._onset_indices[wheel_index] = candidate.onset_index
            self._onset_times[wheel_index] = candidate.onset_time_s
        else:
            self._clear_provisional_alarm(wheel_index)
            self._baselines[wheel_index].append(smoothed_ratio)
        self._candidates[wheel_index] = None
        return _CompletedEvidence(
            event=event,
            rise_coefficient=candidate.rise_coefficient,
            fall_coefficient=candidate.fall_coefficient,
            steady_gain=steady_gain,
            steady_tail_gain=steady_tail_gain,
        )

    def _estimated_onset(self) -> tuple[int, float]:
        delay = self.cfg.smooth_window - 1 + self.cfg.haar_half_window
        estimated_index = max(0, self._frame_index - delay)
        for index, t_sec in self._time_history:
            if index == estimated_index:
                return index, t_sec
        if self._last_t_sec is None:
            raise RuntimeError("onset requested before the first frame")
        estimated_time = self._last_t_sec - delay / self.cfg.sample_rate_hz
        return estimated_index, estimated_time

    def _smooth_ratio(self, wheel_index: int, ratio: float) -> float:
        raw_window = self._raw_ratio_windows[wheel_index]
        raw_window.append(ratio)
        median_value = float(median(raw_window))
        median_window = self._median_windows[wheel_index]
        median_window.append(median_value)
        return sum(median_window) / len(median_window)

    def _baseline(self, wheel_index: int) -> float | None:
        values = self._baselines[wheel_index]
        if len(values) < self.cfg.baseline_min_samples:
            return None
        return float(median(values))

    def _haar_coefficient(self, values: deque[float]) -> float:
        half = self.cfg.haar_half_window
        if len(values) < 2 * half:
            return math.nan
        window = list(values)
        return sum(window[half:]) / half - sum(window[:half]) / half

    def _target_peer_ratios(
        self,
        wheels: tuple[float, float, float, float],
        normal_signals: tuple[
            bool | None, bool | None, bool | None, bool | None
        ],
    ) -> tuple[
        tuple[float, float, float, float],
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]],
        tuple[str, str, str, str],
    ]:
        ratios = []
        reference_keys = []
        reference_sources = []
        confirmed_normal = {
            index for index, signal in enumerate(normal_signals) if signal is True
        }
        for wheel_index, target in enumerate(wheels):
            normal_peers = tuple(sorted(confirmed_normal - {wheel_index}))
            if normal_peers:
                reference_indices = normal_peers
            elif self.cfg.reference_mode == "opposite_diagonal":
                reference_indices = OPPOSITE_DIAGONAL_REFERENCES[wheel_index]
            else:
                reference_indices = tuple(
                    index for index in range(WHEEL_COUNT) if index != wheel_index
                )
            peers = [wheels[index] for index in reference_indices]
            peer_reference = float(median(peers))
            ratios.append(target / peer_reference if peer_reference > 1.0e-9 else math.nan)
            reference_keys.append(reference_indices)
            if normal_peers:
                names = "+".join(WHEEL_NAMES[index] for index in normal_peers)
                reference_sources.append(f"confirmed_normal:{names}")
            elif self.cfg.reference_mode == "opposite_diagonal":
                names = "+".join(
                    WHEEL_NAMES[index] for index in reference_indices
                )
                reference_sources.append(f"opposite_diagonal:{names}")
            else:
                reference_sources.append("peer_median")
        return (
            tuple(ratios),  # type: ignore[return-value]
            tuple(reference_keys),  # type: ignore[return-value]
            tuple(reference_sources),  # type: ignore[return-value]
        )

    def _reset_wheel_tracking(
        self, wheel_index: int, clear_baseline: bool
    ) -> None:
        self._clear_provisional_alarm(wheel_index)
        self._raw_ratio_windows[wheel_index].clear()
        self._median_windows[wheel_index].clear()
        self._gain_windows[wheel_index].clear()
        self._candidates[wheel_index] = None
        if clear_baseline:
            self._baselines[wheel_index].clear()

    def _handle_invalid_frame(self) -> None:
        self._invalid_frames += 1
        for window in (
            self._raw_ratio_windows,
            self._median_windows,
            self._gain_windows,
        ):
            for values in window:
                values.clear()
        for wheel_index in range(WHEEL_COUNT):
            self._clear_provisional_alarm(wheel_index)
        self._candidates = [None] * WHEEL_COUNT
        if self._invalid_frames >= self.cfg.clear_baseline_after_invalid_frames:
            for values in self._baselines:
                values.clear()

    def _state_name(self, wheel_index: int) -> str:
        candidate = self._candidates[wheel_index]
        if candidate is not None:
            return candidate.phase
        if self._alarms[wheel_index]:
            return "alarm"
        if len(self._baselines[wheel_index]) < self.cfg.baseline_min_samples:
            return "warming"
        return "idle"

    def _clear_provisional_alarm(self, wheel_index: int) -> None:
        """Retract an alarm only while its verification candidate is active."""
        if self._candidates[wheel_index] is None:
            return
        self._alarms[wheel_index] = False
        self._onset_indices[wheel_index] = None
        self._onset_times[wheel_index] = None

    def _empty_result(
        self, frame: WheelFrame, speed_valid: bool
    ) -> WaveletShapeResult:
        nan4 = (math.nan, math.nan, math.nan, math.nan)
        false4 = (False, False, False, False)
        return WaveletShapeResult(
            t_sec=frame.t_sec,
            wheels=frame.wheels,
            speed_valid=speed_valid,
            normal_signals=frame.normal_signals,
            reference_sources=("unavailable",) * WHEEL_COUNT,
            target_peer_ratios=nan4,
            normalized_gains=nan4,
            haar_coefficients=nan4,
            states=tuple(  # type: ignore[arg-type]
                self._state_name(index) for index in range(WHEEL_COUNT)
            ),
            shape_events=false4,
            new_blowouts=false4,
            blowout_alarms=tuple(self._alarms),  # type: ignore[arg-type]
            estimated_onset_indices=tuple(self._onset_indices),  # type: ignore[arg-type]
            estimated_onset_times_s=tuple(self._onset_times),  # type: ignore[arg-type]
            rise_coefficients=(0.0, 0.0, 0.0, 0.0),
            fall_coefficients=(0.0, 0.0, 0.0, 0.0),
            steady_gains=nan4,
            steady_tail_gains=nan4,
        )

    def _validate_frame(self, frame: WheelFrame) -> None:
        if not math.isfinite(frame.t_sec):
            raise ValueError("frame time must be finite")
        if self._last_t_sec is not None and frame.t_sec <= self._last_t_sec:
            raise ValueError("frame times must be strictly increasing")
        if len(frame.wheels) != WHEEL_COUNT:
            raise ValueError(f"expected {WHEEL_COUNT} wheel speeds")
        if not all(math.isfinite(value) for value in frame.wheels):
            raise ValueError("wheel speeds must be finite")


def run_detection(
    frames: Iterable[WheelFrame],
    cfg: WaveletShapeConfig | None = None,
) -> list[WaveletShapeResult]:
    detector = WaveletShapeBlowoutDetector(cfg)
    return [detector.push(frame) for frame in frames]
