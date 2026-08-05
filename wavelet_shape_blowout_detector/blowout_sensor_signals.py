from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


WHEEL_NAMES = ("FL", "FR", "RL", "RR")
DIAGONAL_WHEEL_SETS = (frozenset(("FL", "RR")), frozenset(("FR", "RL")))


def validate_sensor_diagonal(sensor_wheels: Sequence[str]) -> tuple[str, str]:
    normalized = tuple(str(wheel).strip().upper() for wheel in sensor_wheels)
    if len(normalized) != 2 or frozenset(normalized) not in DIAGONAL_WHEEL_SETS:
        raise ValueError("sensor wheels must be one diagonal: FL RR or FR RL")
    return normalized  # type: ignore[return-value]


def sensor_column_name(wheel: str) -> str:
    normalized = wheel.strip().upper()
    if normalized not in WHEEL_NAMES:
        raise ValueError(f"unknown wheel {wheel!r}")
    return f"{normalized}_blowout_signal"


def build_delayed_sensor_signals(
    times: np.ndarray,
    wheel_onsets_s: Mapping[str, float],
    sensor_wheels: Sequence[str] = ("FL", "RR"),
    delay_s: float = 0.30,
) -> dict[str, np.ndarray]:
    """Build latched wheel-specific delayed signals for one sensor diagonal."""
    if delay_s < 0.0:
        raise ValueError("sensor delay cannot be negative")
    diagonal = validate_sensor_diagonal(sensor_wheels)
    normalized_onsets = {
        str(wheel).strip().upper(): float(onset)
        for wheel, onset in wheel_onsets_s.items()
    }
    unknown = sorted(set(normalized_onsets) - set(WHEEL_NAMES))
    if unknown:
        raise ValueError(f"unknown onset wheels: {unknown}")

    signals: dict[str, np.ndarray] = {}
    for wheel in diagonal:
        onset = normalized_onsets.get(wheel)
        trigger_time = None if onset is None else onset + delay_s
        tolerance = (
            0.0
            if trigger_time is None
            else max(1.0, abs(trigger_time)) * 1.0e-12
        )
        active = (
            np.zeros(len(times), dtype=np.int8)
            if onset is None
            else (times >= trigger_time - tolerance).astype(np.int8)
        )
        signals[sensor_column_name(wheel)] = active
    return signals
