"""Rise-fall-plateau wavelet wheel-blowout detector."""

from .detector import (
    WaveletShapeBlowoutDetector,
    WaveletShapeConfig,
    WaveletShapeResult,
    WheelFrame,
    run_detection,
)
from .evidence_detector import (
    EvidenceBlowoutDetector,
    EvidenceConfig,
    EvidenceResult,
    run_evidence_detection,
)
from .pressure_fusion_detector import (
    DIAGONALS,
    PressureFusionBlowoutDetector,
    PressureFusionConfig,
    PressureFusionFrame,
    PressureFusionResult,
)

__all__ = [
    "WaveletShapeBlowoutDetector",
    "WaveletShapeConfig",
    "WaveletShapeResult",
    "WheelFrame",
    "run_detection",
    "EvidenceBlowoutDetector",
    "EvidenceConfig",
    "EvidenceResult",
    "run_evidence_detection",
    "DIAGONALS",
    "PressureFusionBlowoutDetector",
    "PressureFusionConfig",
    "PressureFusionFrame",
    "PressureFusionResult",
]
