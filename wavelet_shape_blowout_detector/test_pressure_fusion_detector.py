from __future__ import annotations

import unittest

from .pressure_fusion_detector import (
    PressureFusionBlowoutDetector,
    PressureFusionConfig,
    PressureFusionFrame,
)


class PressureFusionDetectorTests(unittest.TestCase):
    def _run(
        self,
        sensor_diagonal: tuple[int, int],
        events: dict[int, tuple[int, float]],
        pressure_events: set[int] | None = None,
        frames: int = 420,
    ):
        detector = PressureFusionBlowoutDetector(
            PressureFusionConfig(
                baseline_min_samples=120,
                baseline_window=300,
                confirm_frames=60,
                persistence_tail_frames=35,
            )
        )
        pressure_events = pressure_events or set()
        results = []
        for index in range(frames):
            common = 50.0 + min(index, 250) * 0.01
            # First-order steering has the same left/right effect on both axles.
            turn = 0.0015 * max(0, min(index - 150, 80))
            wheels = [common + turn, common - turn, common + turn, common - turn]
            for wheel, (start, amount) in events.items():
                if index >= start:
                    wheels[wheel] *= 1.0 + amount
            pressure = [None] * 4
            for wheel in sensor_diagonal:
                pressure[wheel] = wheel in pressure_events and index >= events[wheel][0]
            results.append(
                detector.push(
                    PressureFusionFrame.from_sequences(index * 0.01, wheels, pressure)
                )
            )
        return results

    def test_auto_discovers_either_diagonal_without_false_alarm(self):
        cases = (((0, 3), ("FR", "RL")), ((1, 2), ("FL", "RR")))
        for diagonal, expected_speed in cases:
            with self.subTest(diagonal=diagonal):
                results = self._run(diagonal, {})
                self.assertEqual(results[-1].speed_diagonal, expected_speed)
                self.assertFalse(any(any(row.blowout_alarms) for row in results))

    def test_pressure_wheels_are_localized_directly(self):
        for diagonal in ((0, 3), (1, 2)):
            for target in diagonal:
                with self.subTest(diagonal=diagonal, target=target):
                    results = self._run(
                        diagonal, {target: (240, 0.01)}, {target}
                    )
                    expected = [False] * 4
                    expected[target] = True
                    self.assertEqual(list(results[-1].blowout_alarms), expected)
                    self.assertEqual(results[-1].alarm_sources[target], "pressure")

    def test_each_speed_only_wheel_is_localized(self):
        for diagonal in ((0, 3), (1, 2)):
            speed_diagonal = (1, 2) if diagonal == (0, 3) else (0, 3)
            for target in speed_diagonal:
                with self.subTest(diagonal=diagonal, target=target):
                    results = self._run(diagonal, {target: (240, 0.011)})
                    expected = [False] * 4
                    expected[target] = True
                    self.assertEqual(list(results[-1].blowout_alarms), expected)
                    self.assertEqual(
                        results[-1].alarm_sources[target], "wheel_speed_confirmed"
                    )

    def test_two_speed_only_wheels_can_alarm_together(self):
        results = self._run((1, 2), {0: (240, 0.011), 3: (240, 0.010)})
        self.assertEqual(results[-1].blowout_alarms, (True, False, False, True))

    def test_sequential_speed_only_blowouts_are_independent(self):
        results = self._run(
            (1, 2), {0: (220, 0.011), 3: (320, 0.010)}, frames=500
        )
        self.assertEqual(results[-1].blowout_alarms, (True, False, False, True))

    def test_reference_blowout_suspends_speed_only_decisions(self):
        results = self._run(
            (1, 2),
            {1: (220, 0.010), 3: (220, 0.011)},
            {1},
        )
        self.assertTrue(results[-1].blowout_alarms[1])
        self.assertFalse(results[-1].blowout_alarms[3])
        self.assertFalse(results[-1].speed_detection_available)

    def test_latched_reference_alarm_stays_unusable_after_pressure_pulse(self):
        detector = PressureFusionBlowoutDetector(
            PressureFusionConfig(baseline_min_samples=20)
        )
        result = None
        for index in range(80):
            # A one-frame True pulse is enough to latch the pressure alarm.
            pressure = [None, index == 40, False, None]
            result = detector.push(
                PressureFusionFrame.from_sequences(
                    index * 0.01, [50.0] * 4, pressure
                )
            )
        assert result is not None
        self.assertTrue(result.blowout_alarms[1])
        self.assertFalse(result.speed_detection_available)

    def test_invalid_or_changing_layout_is_rejected(self):
        detector = PressureFusionBlowoutDetector()
        with self.assertRaises(ValueError):
            detector.push(
                PressureFusionFrame.from_sequences(
                    0.0, [50.0] * 4, [False, False, None, None]
                )
            )
        detector.reset()
        detector.push(
            PressureFusionFrame.from_sequences(
                0.0, [50.0] * 4, [False, None, None, False]
            )
        )
        with self.assertRaises(ValueError):
            detector.push(
                PressureFusionFrame.from_sequences(
                    0.01, [50.0] * 4, [None, False, False, None]
                )
            )


if __name__ == "__main__":
    unittest.main()
