from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from statistics import median

from .detector import (
    OPPOSITE_DIAGONAL_REFERENCES,
    REFERENCE_MODES,
    WHEEL_COUNT,
    WHEEL_NAMES,
    WheelFrame,
)


@dataclass(frozen=True)
class EvidenceConfig:
    """Configuration for the cumulative rise/pullback/persistence detector."""

    sample_rate_hz: float = 100.0
    reference_mode: str = "opposite_diagonal"
    smooth_window: int = 5
    haar_half_window: int = 5
    slope_half_window: int = 3
    baseline_window: int = 300
    baseline_min_samples: int = 100
    noise_window: int = 200
    min_avg_speed: float = 20.0

    haar_scale: float = 0.0032
    slope_scale: float = 0.0024
    gain_scale: float = 0.0045
    drawdown_scale: float = 0.0028
    min_pullback_gain: float = 0.0015
    elevated_gain: float = 0.0045
    strong_elevated_gain: float = 0.0075

    rise_decay: float = 0.92
    pullback_decay: float = 0.94
    persistence_decay: float = 0.985
    rise_candidate_score: float = 2.2
    min_trigger_haar: float = 0.0012
    pullback_confirm_score: float = 2.8
    persistence_confirm_score: float = 12.0
    min_candidate_gain: float = 0.0030
    max_physical_gain: float = 0.035
    candidate_timeout_frames: int = 180
    reset_below_frames: int = 20
    clear_baseline_after_invalid_frames: int = 50
    target_wheels: tuple[int, ...] = (0, 1, 2, 3)

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive")
        if self.reference_mode not in REFERENCE_MODES:
            raise ValueError(f"reference_mode must be one of {REFERENCE_MODES}")
        if self.smooth_window <= 0 or self.smooth_window % 2 == 0:
            raise ValueError("smooth_window must be a positive odd integer")
        if self.haar_half_window <= 0 or self.slope_half_window <= 0:
            raise ValueError("feature windows must be positive")
        if not 2 <= self.baseline_min_samples <= self.baseline_window:
            raise ValueError("baseline sample limits are invalid")
        if self.noise_window < 10:
            raise ValueError("noise_window must be at least 10")
        if self.min_avg_speed <= 0.0:
            raise ValueError("min_avg_speed must be positive")
        if not 0.0 < self.rise_decay <= 1.0:
            raise ValueError("rise_decay must be in (0, 1]")
        if not 0.0 < self.pullback_decay <= 1.0:
            raise ValueError("pullback_decay must be in (0, 1]")
        if not 0.0 < self.persistence_decay <= 1.0:
            raise ValueError("persistence_decay must be in (0, 1]")
        if self.candidate_timeout_frames <= 0 or self.reset_below_frames <= 0:
            raise ValueError("candidate frame limits must be positive")
        if len(set(self.target_wheels)) != len(self.target_wheels):
            raise ValueError("target_wheels must not contain duplicates")
        if any(index < 0 or index >= WHEEL_COUNT for index in self.target_wheels):
            raise ValueError("target_wheels contains an invalid wheel index")


@dataclass(frozen=True)
class EvidenceResult:
    t_sec: float
    wheels: tuple[float, float, float, float]
    speed_valid: bool
    normal_signals: tuple[bool | None, bool | None, bool | None, bool | None]
    reference_sources: tuple[str, str, str, str]
    target_peer_ratios: tuple[float, float, float, float]
    normalized_gains: tuple[float, float, float, float]
    haar_coefficients: tuple[float, float, float, float]
    short_slopes: tuple[float, float, float, float]
    noise_scales: tuple[float, float, float, float]
    states: tuple[str, str, str, str]
    fast_alarms: tuple[bool, bool, bool, bool]
    confirmed_alarms: tuple[bool, bool, bool, bool]
    new_fast_alarms: tuple[bool, bool, bool, bool]
    new_confirmed_alarms: tuple[bool, bool, bool, bool]
    rise_evidence: tuple[float, float, float, float]
    pullback_evidence: tuple[float, float, float, float]
    persistence_evidence: tuple[float, float, float, float]
    estimated_onset_indices: tuple[int | None, int | None, int | None, int | None]
    estimated_onset_times_s: tuple[float | None, float | None, float | None, float | None]

    @property
    def phases(self) -> tuple[str, str, str, str]:
        return self.states

    @property
    def blowout_alarms(self) -> tuple[bool, bool, bool, bool]:
        """Compatibility output: low-latency cumulative-evidence alarm."""
        return self.fast_alarms

    @property
    def new_blowouts(self) -> tuple[bool, bool, bool, bool]:
        return self.new_fast_alarms

    @property
    def shape_events(self) -> tuple[bool, bool, bool, bool]:
        return self.new_confirmed_alarms


@dataclass
class _WheelEvidence:
    phase: str = "warming"
    rise_evidence: float = 0.0
    max_rise_evidence: float = 0.0
    pullback_evidence: float = 0.0
    persistence_evidence: float = 0.0
    running_low_gain: float = 0.0
    running_peak_gain: float = 0.0
    pullback_seen: bool = False
    fast_alarm: bool = False
    latched_alarm: bool = False
    candidate_frames: int = 0
    below_frames: int = 0
    onset_index: int | None = None
    onset_time_s: float | None = None


def _positive_score(value: float, scale: float) -> float:
    return max(0.0, math.tanh(value / max(scale, 1.0e-9)))


class EvidenceBlowoutDetector:
    """Causal cumulative-evidence detector described in the design document."""

    def __init__(self, cfg: EvidenceConfig | None = None):
        self.cfg = cfg or EvidenceConfig()
        self._frame_index = -1
        self._last_t_sec: float | None = None
        self._invalid_frames = 0
        self._raw_ratio_windows = [
            deque(maxlen=self.cfg.smooth_window) for _ in range(WHEEL_COUNT)
        ]
        self._median_windows = [
            deque(maxlen=self.cfg.smooth_window) for _ in range(WHEEL_COUNT)
        ]
        feature_length = max(
            2 * self.cfg.haar_half_window,
            2 * self.cfg.slope_half_window,
        )
        self._gain_windows = [deque(maxlen=feature_length) for _ in range(WHEEL_COUNT)]
        self._baselines = [
            deque(maxlen=self.cfg.baseline_window) for _ in range(WHEEL_COUNT)
        ]
        self._noise_histories = [
            deque(maxlen=self.cfg.noise_window) for _ in range(WHEEL_COUNT)
        ]
        history_length = self.cfg.smooth_window + self.cfg.haar_half_window + 2
        self._time_history: deque[tuple[int, float]] = deque(maxlen=history_length)
        self._reference_keys: list[tuple[int, ...] | None] = [None] * WHEEL_COUNT
        self._states = [_WheelEvidence() for _ in range(WHEEL_COUNT)]

    def reset(self) -> None:
        self.__init__(self.cfg)

    def push(self, frame: WheelFrame) -> EvidenceResult:
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
            return self._empty_result(frame)

        self._invalid_frames = 0
        ratios, reference_keys, reference_sources = self._target_peer_ratios(
            magnitudes, frame.normal_signals
        )
        gains = [math.nan] * WHEEL_COUNT
        haar = [math.nan] * WHEEL_COUNT
        slopes = [math.nan] * WHEEL_COUNT
        noises = [math.nan] * WHEEL_COUNT
        new_fast = [False] * WHEEL_COUNT
        new_confirmed = [False] * WHEEL_COUNT
        target_set = set(self.cfg.target_wheels)

        for wheel_index, ratio in enumerate(ratios):
            if reference_keys[wheel_index] != self._reference_keys[wheel_index]:
                self._reset_tracking(wheel_index, clear_baseline=True)
                self._reference_keys[wheel_index] = reference_keys[wheel_index]
            smoothed_ratio = self._smooth_ratio(wheel_index, ratio)
            baseline = self._baseline(wheel_index)
            state = self._states[wheel_index]
            if baseline is None:
                self._baselines[wheel_index].append(smoothed_ratio)
                state.phase = "warming"
                continue

            gain = smoothed_ratio / baseline - 1.0
            gains[wheel_index] = gain
            window = self._gain_windows[wheel_index]
            window.append(gain)
            coefficient = self._haar_coefficient(window)
            slope = self._short_slope(window)
            noise = self._noise_scale(wheel_index)
            haar[wheel_index] = coefficient
            slopes[wheel_index] = slope
            noises[wheel_index] = noise

            confirmed_normal = frame.normal_signals[wheel_index] is True
            if (
                wheel_index not in target_set
                or confirmed_normal
                or state.latched_alarm
            ):
                if not state.latched_alarm:
                    self._clear_candidate(wheel_index)
                    self._adapt_normal(wheel_index, smoothed_ratio, gain)
                continue

            was_fast = state.fast_alarm
            was_confirmed = state.latched_alarm
            self._advance_state(wheel_index, gain, coefficient, slope, noise)
            new_fast[wheel_index] = not was_fast and state.fast_alarm
            new_confirmed[wheel_index] = not was_confirmed and state.latched_alarm
            if state.phase == "normal":
                self._adapt_normal(wheel_index, smoothed_ratio, gain)

        return EvidenceResult(
            t_sec=frame.t_sec,
            wheels=frame.wheels,
            speed_valid=True,
            normal_signals=frame.normal_signals,
            reference_sources=reference_sources,
            target_peer_ratios=ratios,
            normalized_gains=tuple(gains),  # type: ignore[arg-type]
            haar_coefficients=tuple(haar),  # type: ignore[arg-type]
            short_slopes=tuple(slopes),  # type: ignore[arg-type]
            noise_scales=tuple(noises),  # type: ignore[arg-type]
            states=tuple(state.phase for state in self._states),  # type: ignore[arg-type]
            fast_alarms=tuple(state.fast_alarm for state in self._states),  # type: ignore[arg-type]
            confirmed_alarms=tuple(state.latched_alarm for state in self._states),  # type: ignore[arg-type]
            new_fast_alarms=tuple(new_fast),  # type: ignore[arg-type]
            new_confirmed_alarms=tuple(new_confirmed),  # type: ignore[arg-type]
            rise_evidence=tuple(state.rise_evidence for state in self._states),  # type: ignore[arg-type]
            pullback_evidence=tuple(state.pullback_evidence for state in self._states),  # type: ignore[arg-type]
            persistence_evidence=tuple(state.persistence_evidence for state in self._states),  # type: ignore[arg-type]
            estimated_onset_indices=tuple(state.onset_index for state in self._states),  # type: ignore[arg-type]
            estimated_onset_times_s=tuple(state.onset_time_s for state in self._states),  # type: ignore[arg-type]
        )

    def _advance_state(
        self,
        wheel_index: int,
        gain: float,
        haar: float,
        slope: float,
        noise: float,
    ) -> None:
        state = self._states[wheel_index]
        haar_scale = max(self.cfg.haar_scale, 4.0 * noise)
        slope_scale = max(self.cfg.slope_scale, 3.0 * noise)
        gain_scale = max(self.cfg.gain_scale, 5.0 * noise)
        haar_score = _positive_score(haar, haar_scale)
        slope_score = _positive_score(slope, slope_scale)
        transient_gate = max(haar_score, slope_score)
        rise_increment = (
            1.10 * haar_score
            + 0.60 * slope_score
            + 0.25 * _positive_score(gain, gain_scale) * transient_gate
            - 0.35
        )
        state.rise_evidence = min(
            20.0,
            max(0.0, self.cfg.rise_decay * state.rise_evidence + rise_increment),
        )
        state.max_rise_evidence = max(state.max_rise_evidence, state.rise_evidence)

        if state.phase in {"warming", "normal"}:
            state.phase = "normal"
            if abs(gain) > self.cfg.max_physical_gain:
                state.rise_evidence = 0.0
                state.max_rise_evidence = 0.0
                return
            if (
                state.rise_evidence >= self.cfg.rise_candidate_score
                and gain >= self.cfg.min_candidate_gain
                and haar >= self.cfg.min_trigger_haar
            ):
                onset_index, onset_time = self._estimated_onset()
                state.phase = "candidate_building"
                state.fast_alarm = True
                state.running_low_gain = min(0.0, gain)
                state.running_peak_gain = gain
                state.candidate_frames = 0
                state.below_frames = 0
                state.onset_index = onset_index
                state.onset_time_s = onset_time
            return

        state.candidate_frames += 1
        state.running_peak_gain = max(state.running_peak_gain, gain)
        drawdown = state.running_peak_gain - gain
        pullback_increment = 0.0
        if drawdown >= self.cfg.min_pullback_gain:
            pullback_increment = (
                0.85 * _positive_score(drawdown, self.cfg.drawdown_scale)
                + 0.75 * _positive_score(-haar, haar_scale)
                + 0.45 * _positive_score(-slope, slope_scale)
            )
        state.pullback_evidence = min(
            20.0,
            self.cfg.pullback_decay * state.pullback_evidence + pullback_increment,
        )
        if state.pullback_evidence >= self.cfg.pullback_confirm_score:
            state.pullback_seen = True
            state.phase = "elevated_verify"

        if state.pullback_seen:
            level_score = _positive_score(
                gain - self.cfg.elevated_gain,
                max(self.cfg.strong_elevated_gain - self.cfg.elevated_gain, noise * 4.0),
            )
            low_level_penalty = _positive_score(
                self.cfg.elevated_gain - gain,
                max(self.cfg.elevated_gain, noise * 4.0),
            )
            state.persistence_evidence = min(
                30.0,
                max(
                    0.0,
                    self.cfg.persistence_decay * state.persistence_evidence
                    + 0.65 * level_score
                    - 0.35 * low_level_penalty,
                ),
            )

        if gain < 0.0:
            state.below_frames += 1
        else:
            state.below_frames = 0

        if (
            state.max_rise_evidence >= self.cfg.rise_candidate_score
            and state.pullback_seen
            and state.persistence_evidence >= self.cfg.persistence_confirm_score
        ):
            state.latched_alarm = True
            state.fast_alarm = True
            state.phase = "latched_alarm"
            return

        if (
            abs(gain) > self.cfg.max_physical_gain
            or state.candidate_frames > self.cfg.candidate_timeout_frames
            or state.below_frames >= self.cfg.reset_below_frames
        ):
            self._clear_candidate(wheel_index)

    def _adapt_normal(self, wheel_index: int, ratio: float, gain: float) -> None:
        self._baselines[wheel_index].append(ratio)
        self._noise_histories[wheel_index].append(gain)

    def _noise_scale(self, wheel_index: int) -> float:
        values = self._noise_histories[wheel_index]
        if len(values) < 10:
            return 0.0
        center = float(median(values))
        return max(1.0e-5, 1.4826 * float(median(abs(v - center) for v in values)))

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
        window = list(values)[-2 * half :]
        return sum(window[half:]) / half - sum(window[:half]) / half

    def _short_slope(self, values: deque[float]) -> float:
        half = self.cfg.slope_half_window
        if len(values) < 2 * half:
            return math.nan
        window = list(values)[-2 * half :]
        return float(median(window[half:]) - median(window[:half]))

    def _estimated_onset(self) -> tuple[int, float]:
        delay = self.cfg.smooth_window - 1 + self.cfg.haar_half_window
        estimated_index = max(0, self._frame_index - delay)
        for index, t_sec in self._time_history:
            if index == estimated_index:
                return index, t_sec
        if self._last_t_sec is None:
            raise RuntimeError("onset requested before first frame")
        return estimated_index, self._last_t_sec - delay / self.cfg.sample_rate_hz

    def _target_peer_ratios(
        self,
        wheels: tuple[float, float, float, float],
        normal_signals: tuple[bool | None, bool | None, bool | None, bool | None],
    ) -> tuple[
        tuple[float, float, float, float],
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]],
        tuple[str, str, str, str],
    ]:
        ratios: list[float] = []
        keys: list[tuple[int, ...]] = []
        sources: list[str] = []
        confirmed = {i for i, value in enumerate(normal_signals) if value is True}
        for wheel_index, target in enumerate(wheels):
            normal_peers = tuple(sorted(confirmed - {wheel_index}))
            if normal_peers:
                refs = normal_peers
            elif self.cfg.reference_mode == "opposite_diagonal":
                refs = OPPOSITE_DIAGONAL_REFERENCES[wheel_index]
            else:
                refs = tuple(i for i in range(WHEEL_COUNT) if i != wheel_index)
            reference = float(median(wheels[i] for i in refs))
            ratios.append(target / reference if reference > 1.0e-9 else math.nan)
            keys.append(refs)
            if normal_peers:
                sources.append(
                    "confirmed_normal:" + "+".join(WHEEL_NAMES[i] for i in refs)
                )
            elif self.cfg.reference_mode == "opposite_diagonal":
                sources.append(
                    "opposite_diagonal:" + "+".join(WHEEL_NAMES[i] for i in refs)
                )
            else:
                sources.append("peer_median")
        return tuple(ratios), tuple(keys), tuple(sources)  # type: ignore[return-value]

    def _clear_candidate(self, wheel_index: int) -> None:
        state = self._states[wheel_index]
        if state.latched_alarm:
            return
        self._states[wheel_index] = _WheelEvidence(phase="normal")

    def _reset_tracking(self, wheel_index: int, clear_baseline: bool) -> None:
        state = self._states[wheel_index]
        latched = state.latched_alarm
        onset_index = state.onset_index
        onset_time = state.onset_time_s
        self._raw_ratio_windows[wheel_index].clear()
        self._median_windows[wheel_index].clear()
        self._gain_windows[wheel_index].clear()
        self._noise_histories[wheel_index].clear()
        if clear_baseline:
            self._baselines[wheel_index].clear()
        self._states[wheel_index] = _WheelEvidence(
            phase="latched_alarm" if latched else "warming",
            fast_alarm=latched,
            latched_alarm=latched,
            onset_index=onset_index,
            onset_time_s=onset_time,
        )

    def _handle_invalid_frame(self) -> None:
        self._invalid_frames += 1
        for windows in (
            self._raw_ratio_windows,
            self._median_windows,
            self._gain_windows,
        ):
            for values in windows:
                values.clear()
        for wheel_index, state in enumerate(self._states):
            if not state.latched_alarm:
                self._clear_candidate(wheel_index)
        if self._invalid_frames >= self.cfg.clear_baseline_after_invalid_frames:
            for values in self._baselines:
                values.clear()
            for wheel_index, state in enumerate(self._states):
                if not state.latched_alarm:
                    state.phase = "warming"

    def _empty_result(self, frame: WheelFrame) -> EvidenceResult:
        nan4 = (math.nan, math.nan, math.nan, math.nan)
        false4 = (False, False, False, False)
        return EvidenceResult(
            t_sec=frame.t_sec,
            wheels=frame.wheels,
            speed_valid=False,
            normal_signals=frame.normal_signals,
            reference_sources=("unavailable",) * WHEEL_COUNT,
            target_peer_ratios=nan4,
            normalized_gains=nan4,
            haar_coefficients=nan4,
            short_slopes=nan4,
            noise_scales=nan4,
            states=tuple(state.phase for state in self._states),  # type: ignore[arg-type]
            fast_alarms=tuple(state.fast_alarm for state in self._states),  # type: ignore[arg-type]
            confirmed_alarms=tuple(state.latched_alarm for state in self._states),  # type: ignore[arg-type]
            new_fast_alarms=false4,
            new_confirmed_alarms=false4,
            rise_evidence=tuple(state.rise_evidence for state in self._states),  # type: ignore[arg-type]
            pullback_evidence=tuple(state.pullback_evidence for state in self._states),  # type: ignore[arg-type]
            persistence_evidence=tuple(state.persistence_evidence for state in self._states),  # type: ignore[arg-type]
            estimated_onset_indices=tuple(state.onset_index for state in self._states),  # type: ignore[arg-type]
            estimated_onset_times_s=tuple(state.onset_time_s for state in self._states),  # type: ignore[arg-type]
        )

    def _validate_frame(self, frame: WheelFrame) -> None:
        if not math.isfinite(frame.t_sec):
            raise ValueError("frame time must be finite")
        if self._last_t_sec is not None and frame.t_sec <= self._last_t_sec:
            raise ValueError("frame times must be strictly increasing")
        if not all(math.isfinite(value) for value in frame.wheels):
            raise ValueError("wheel speeds must be finite")


def run_evidence_detection(
    frames: list[WheelFrame], cfg: EvidenceConfig | None = None
) -> list[EvidenceResult]:
    detector = EvidenceBlowoutDetector(cfg)
    return [detector.push(frame) for frame in frames]
