from __future__ import annotations

import unittest

from .detector import WheelFrame
from .evidence_detector import EvidenceBlowoutDetector, EvidenceConfig


class EvidenceBlowoutDetectorTests(unittest.TestCase):
    def _run_shape(
        self,
        *,
        target: int = 3,
        include_pullback: bool = True,
        persistent: bool = True,
    ):
        detector = EvidenceBlowoutDetector(EvidenceConfig(target_wheels=(target,)))
        results = []
        for index in range(300):
            gain = 0.0
            if 140 <= index < 150:
                gain = 0.012 * (index - 140) / 10.0
            elif 150 <= index < 158:
                gain = (
                    0.012 - 0.010 * (index - 150) / 8.0
                    if include_pullback
                    else 0.012
                )
            elif 158 <= index < 163 and include_pullback:
                gain = 0.002
            elif index >= 158:
                gain = (
                    (0.009 if persistent else 0.0)
                    if include_pullback
                    else 0.012
                )
            wheels = [50.0] * 4
            wheels[target] *= 1.0 + gain
            results.append(
                detector.push(WheelFrame.from_sequences(index * 0.01, wheels))
            )
        return results

    def test_confirms_rise_pullback_and_persistence_for_each_wheel(self):
        for target in range(4):
            with self.subTest(target=target):
                results = self._run_shape(target=target)
                self.assertTrue(any(row.new_fast_alarms[target] for row in results))
                self.assertTrue(any(row.new_confirmed_alarms[target] for row in results))
                self.assertTrue(results[-1].confirmed_alarms[target])

    def test_step_without_pullback_is_not_confirmed(self):
        results = self._run_shape(include_pullback=False)
        self.assertTrue(any(row.fast_alarms[3] for row in results))
        self.assertFalse(any(row.new_confirmed_alarms[3] for row in results))

    def test_pulse_without_high_persistence_is_not_confirmed(self):
        results = self._run_shape(persistent=False)
        self.assertTrue(any(row.fast_alarms[3] for row in results))
        self.assertFalse(any(row.new_confirmed_alarms[3] for row in results))

    def test_confirmed_alarm_is_latched_until_reset(self):
        results = self._run_shape()
        self.assertTrue(results[-1].confirmed_alarms[3])
        detector = EvidenceBlowoutDetector()
        for index, row in enumerate(results):
            detector.push(WheelFrame.from_sequences(index * 0.01, row.wheels))
        for index in range(300, 500):
            result = detector.push(
                WheelFrame.from_sequences(index * 0.01, [50.0] * 4)
            )
        self.assertTrue(result.confirmed_alarms[3])
        detector.reset()
        result = detector.push(WheelFrame.from_sequences(0.0, [50.0] * 4))
        self.assertFalse(any(result.confirmed_alarms))

    def test_common_acceleration_does_not_build_evidence(self):
        detector = EvidenceBlowoutDetector()
        for index in range(300):
            speed = 40.0 + 0.1 * min(index, 200)
            result = detector.push(
                WheelFrame.from_sequences(index * 0.01, [speed] * 4)
            )
        self.assertFalse(any(result.fast_alarms))
        self.assertFalse(any(result.confirmed_alarms))


if __name__ == "__main__":
    unittest.main()
