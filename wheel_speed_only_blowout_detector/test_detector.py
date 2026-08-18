from __future__ import annotations

import math
import unittest

from .detector import (
    WheelSpeedBlowoutConfig,
    WheelSpeedBlowoutDetector,
    WheelSpeedFrame,
)


class WheelSpeedBlowoutDetectorTests(unittest.TestCase):
    def _run(
        self,
        events: dict[int, tuple[int, float]],
        *,
        frames: int = 460,
        turn: bool = True,
    ):
        cfg = WheelSpeedBlowoutConfig(
            baseline_min_samples=120,
            baseline_window=300,
            confirm_frames=60,
            persistence_tail_frames=35,
            early_confirm_frames=45,
        )
        detector = WheelSpeedBlowoutDetector(cfg)
        results = []
        for index in range(frames):
            common = 50.0 + min(index, 250) * 0.01
            steering = 0.0015 * max(0, min(index - 150, 80)) if turn else 0.0
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
                detector.push(WheelSpeedFrame.from_sequences(index * 0.01, wheels))
            )
        return results

    def test_each_single_wheel_is_detected_and_localized(self):
        for target in range(4):
            with self.subTest(target=target):
                results = self._run({target: (240, 0.011)})
                expected = [False] * 4
                expected[target] = True
                self.assertEqual(list(results[-1].blowout_alarms), expected)
                self.assertTrue(any(row.new_blowouts[target] for row in results))

    def test_two_wheels_on_same_diagonal_can_alarm(self):
        for events, expected in (
            ({0: (240, 0.011), 3: (240, 0.010)}, (True, False, False, True)),
            ({1: (240, 0.011), 2: (240, 0.010)}, (False, True, True, False)),
        ):
            with self.subTest(events=events):
                self.assertEqual(self._run(events)[-1].blowout_alarms, expected)

    def test_sequential_same_diagonal_events_are_detected(self):
        results = self._run({0: (220, 0.011), 3: (330, 0.010)}, frames=520)
        self.assertEqual(results[-1].blowout_alarms, (True, False, False, True))

    def test_common_acceleration_and_first_order_turn_do_not_alarm(self):
        results = self._run({})
        self.assertFalse(any(any(row.blowout_alarms) for row in results))

    def test_axle_step_is_rejected_by_diagonal_evidence(self):
        results = self._run({0: (240, 0.011), 1: (240, 0.011)}, turn=False)
        self.assertFalse(any(results[-1].blowout_alarms))

    def test_reference_contamination_suppresses_opposite_diagonal(self):
        results = self._run({0: (220, 0.011), 1: (330, 0.011)}, frames=520)
        self.assertTrue(results[-1].blowout_alarms[0])
        self.assertFalse(results[-1].blowout_alarms[1])
        self.assertEqual(results[-1].states[1], "reference_contaminated")

    def test_low_speed_clears_warmup_but_not_latched_alarm(self):
        cfg = WheelSpeedBlowoutConfig(
            baseline_min_samples=20,
            baseline_window=50,
            confirm_frames=20,
            persistence_tail_frames=10,
            early_confirm_frames=15,
            clear_after_invalid_frames=10,
        )
        detector = WheelSpeedBlowoutDetector(cfg)
        for index in range(100):
            wheels = [50.0] * 4
            if index >= 50:
                wheels[3] *= 1.012
            result = detector.push(WheelSpeedFrame.from_sequences(index * 0.01, wheels))
        self.assertTrue(result.blowout_alarms[3])
        for index in range(100, 115):
            result = detector.push(WheelSpeedFrame.from_sequences(index * 0.01, [0.0] * 4))
        self.assertFalse(result.warmed_up)
        self.assertTrue(result.blowout_alarms[3])

    def test_early_confirmation_requires_a_quiet_common_speed(self):
        cfg = WheelSpeedBlowoutConfig(
            baseline_min_samples=120,
            baseline_window=300,
        )
        detector = WheelSpeedBlowoutDetector(cfg)
        alarm_index = None
        candidate_start = None
        for index in range(380):
            ramp = 0.0
            if index >= 220:
                ramp = 0.05 * min(index - 220, 40) / 40
            wheels = [50.0 * (1.0 + ramp)] * 4
            if index >= 220:
                wheels[3] *= 1.011
            result = detector.push(
                WheelSpeedFrame.from_sequences(index * 0.01, wheels)
            )
            if result.candidates[3] and candidate_start is None:
                candidate_start = index
            if result.new_blowouts[3]:
                alarm_index = index
        self.assertIsNotNone(candidate_start)
        self.assertIsNotNone(alarm_index)
        self.assertGreaterEqual(alarm_index, candidate_start + cfg.confirm_frames - 1)

    def test_reset_and_input_validation(self):
        detector = WheelSpeedBlowoutDetector(
            WheelSpeedBlowoutConfig(baseline_min_samples=20)
        )
        detector.push(WheelSpeedFrame.from_sequences(0.0, [50.0] * 4))
        with self.assertRaises(ValueError):
            detector.push(WheelSpeedFrame.from_sequences(0.0, [50.0] * 4))
        detector.reset()
        with self.assertRaises(ValueError):
            detector.push(
                WheelSpeedFrame.from_sequences(0.0, [50.0, math.nan, 50.0, 50.0])
            )


if __name__ == "__main__":
    unittest.main()
