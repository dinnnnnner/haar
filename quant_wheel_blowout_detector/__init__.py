"""Online quantitative blowout detection from four wheel speeds."""

from .detector import (
    WHEEL_NAMES,
    QuantBlowoutConfig,
    QuantBlowoutDetector,
    QuantFrame,
    QuantResult,
    run_detection,
)

__all__ = [
    "WHEEL_NAMES",
    "QuantBlowoutConfig",
    "QuantBlowoutDetector",
    "QuantFrame",
    "QuantResult",
    "run_detection",
]
