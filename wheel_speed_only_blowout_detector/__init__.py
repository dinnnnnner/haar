"""Blowout detection using only four corrected wheel-speed channels."""

from .detector import (
    WHEEL_NAMES,
    WheelSpeedBlowoutConfig,
    WheelSpeedBlowoutDetector,
    WheelSpeedFrame,
    WheelSpeedResult,
    run_detection,
)

__all__ = [
    "WHEEL_NAMES",
    "WheelSpeedBlowoutConfig",
    "WheelSpeedBlowoutDetector",
    "WheelSpeedFrame",
    "WheelSpeedResult",
    "run_detection",
]
