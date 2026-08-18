from __future__ import annotations

import math
import unittest

from .detector import QuantBlowoutConfig, QuantBlowoutDetector, QuantFrame


class QuantBlowoutDetectorTests(unittest.TestCase):
    def _run(
        self,
        events: dict[int, tuple[int, float]],
        *,
        frames: int = 720,
        turn: bool = True,
    ):
        detector = QuantBlowoutDetector()
        results = []
        for index in range(frames):
            common = 50.0 + min(index, 420) * 0.005
            steering = 0.0015 * max(0, min(index - 350, 80)) if turn else 0.0
            wheels = [
                common + steering,
                common - steering,
                common + steering,
                common - steering,
            ]
            for wheel, (start, amount) in events.items():
                if index >= start:
                    wheels[wheel] *= 1.0 + amount
            results.append(
                detector.push(QuantFrame.from_sequences(index * 0.01, wheels))
            )
        return results

    def test_each_wheel_has_the_correct_factor_fingerprint(self) -> None:
        for target in range(4):
            with self.subTest(target=target):
                results = self._run({target: (480, 0.011)})
                expected = [False] * 4
                expected[target] = True
                self.assertEqual(list(results[-1].blowout_alarms), expected)
                self.assertTrue(any(row.new_blowouts[target] for row in results))

    def test_common_acceleration_and_first_order_turn_are_neutral(self) -> None:
        results = self._run({})
        self.assertFalse(any(any(row.blowout_alarms) for row in results))

    def test_axle_step_has_no_unique_wheel_isolation(self) -> None:
        results = self._run({0: (480, 0.011), 1: (480, 0.011)}, turn=False)
        self.assertFalse(any(results[-1].blowout_alarms))

    def test_oversized_single_wheel_step_is_rejected_as_wheel_slip(self) -> None:
        results = self._run({0: (480, 0.060)}, turn=False)
        self.assertFalse(any(results[-1].blowout_alarms))

    def test_risk_score_leads_and_alarm_latches(self) -> None:
        results = self._run({3: (480, 0.011)})
        event_rows = [row for row in results[480:] if row.warmed_up]
        self.assertTrue(any(row.leading_wheel == 3 for row in event_rows[:30]))
        self.assertGreater(max(row.risk_scores[3] for row in event_rows), 90.0)
        self.assertTrue(results[-1].blowout_alarms[3])

    def test_low_speed_rewarms_without_clearing_alarm(self) -> None:
        detector = QuantBlowoutDetector(
            QuantBlowoutConfig(
                warmup_frames=100,
                confirm_frames=30,
                persistence_tail_frames=20,
                candidate_timeout_frames=60,
                clear_after_invalid_frames=10,
            )
        )
        for index in range(250):
            wheels = [50.0] * 4
            if index >= 150:
                wheels[0] *= 1.012
            result = detector.push(QuantFrame.from_sequences(index * 0.01, wheels))
        self.assertTrue(result.blowout_alarms[0])
        for index in range(250, 265):
            result = detector.push(QuantFrame.from_sequences(index * 0.01, [0.0] * 4))
        self.assertFalse(result.warmed_up)
        self.assertTrue(result.blowout_alarms[0])

    def test_reset_and_input_validation(self) -> None:
        detector = QuantBlowoutDetector()
        detector.push(QuantFrame.from_sequences(0.0, [50.0] * 4))
        with self.assertRaises(ValueError):
            detector.push(QuantFrame.from_sequences(0.0, [50.0] * 4))
        detector.reset()
        with self.assertRaises(ValueError):
            detector.push(QuantFrame.from_sequences(0.0, [50.0, math.nan, 50.0, 50.0]))


if __name__ == "__main__":
    unittest.main()
