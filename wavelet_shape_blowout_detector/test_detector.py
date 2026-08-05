from __future__ import annotations

import csv
import unittest
from pathlib import Path

from .detector import (
    WaveletShapeBlowoutDetector,
    WaveletShapeConfig,
    WheelFrame,
)


class WaveletShapeBlowoutDetectorTests(unittest.TestCase):
    def _run_target_shape(
        self,
        target: int = 3,
        persistent: bool = True,
        include_fall: bool = True,
    ):
        detector = WaveletShapeBlowoutDetector(
            WaveletShapeConfig(target_wheels=(target,))
        )
        results = []
        for index in range(260):
            gain = 0.0
            if 140 <= index < 150:
                gain = 0.012 * (index - 140) / 10.0
            elif 150 <= index < 158:
                if include_fall:
                    gain = 0.012 - 0.010 * (index - 150) / 8.0
                else:
                    gain = 0.012
            elif 158 <= index < 163 and include_fall:
                gain = 0.002
            elif index >= 158:
                gain = 0.009 if persistent else 0.0
            wheels = [50.0] * 4
            wheels[target] *= 1.0 + gain
            results.append(
                detector.push(WheelFrame.from_sequences(index * 0.01, wheels))
            )
        return results

    def test_detects_rise_fall_and_persistent_overspeed(self):
        for target in range(4):
            with self.subTest(target=target):
                results = self._run_target_shape(target)
                self.assertTrue(results[-1].blowout_alarms[target])
                first = next(result for result in results if result.new_blowouts[target])
                self.assertLessEqual(
                    first.t_sec - first.estimated_onset_times_s[target], 0.1
                )
                self.assertLessEqual(first.estimated_onset_times_s[target], 1.45)
                verified = next(
                    result for result in results if result.shape_events[target]
                )
                self.assertGreater(verified.t_sec, first.t_sec)
                self.assertEqual(
                    sum(result.new_blowouts[target] for result in results), 1
                )

    def test_rejects_step_without_fall(self):
        results = self._run_target_shape(include_fall=False)
        self.assertTrue(any(result.blowout_alarms[3] for result in results))
        self.assertFalse(results[-1].blowout_alarms[3])
        self.assertFalse(any(result.shape_events[3] for result in results))

    def test_rejects_pulse_without_persistent_overspeed(self):
        results = self._run_target_shape(persistent=False)
        self.assertTrue(any(result.blowout_alarms[3] for result in results))
        self.assertFalse(results[-1].blowout_alarms[3])
        self.assertFalse(any(result.shape_events[3] for result in results))

    def test_common_acceleration_cancels_in_wheel_ratio(self):
        detector = WaveletShapeBlowoutDetector()
        results = []
        for index in range(300):
            common_speed = 40.0 + min(index, 200) * 0.1
            results.append(
                detector.push(
                    WheelFrame.from_sequences(index * 0.01, [common_speed] * 4)
                )
            )
        self.assertFalse(any(any(result.blowout_alarms) for result in results))

    def test_rejects_sensor_spike_even_if_it_lands_on_a_high_plateau(self):
        detector = WaveletShapeBlowoutDetector()
        results = []
        for index in range(260):
            gain = 0.0
            if 140 <= index < 150:
                gain = 0.05 * (index - 140) / 10.0
            elif 150 <= index < 158:
                gain = 0.05 - 0.048 * (index - 150) / 8.0
            elif 158 <= index < 163:
                gain = 0.002
            elif index >= 163:
                gain = 0.009
            wheels = [50.0, 50.0, 50.0, 50.0 * (1.0 + gain)]
            results.append(
                detector.push(WheelFrame.from_sequences(index * 0.01, wheels))
            )
        self.assertFalse(results[-1].blowout_alarms[3])
        self.assertFalse(any(result.shape_events[3] for result in results))

    def test_two_simultaneous_wheels_have_independent_alarm_outputs(self):
        for targets in ((0, 3), (1, 2)):
            with self.subTest(targets=targets):
                detector = WaveletShapeBlowoutDetector()
                results = []
                for index in range(260):
                    gain = 0.0
                    if 140 <= index < 150:
                        gain = 0.012 * (index - 140) / 10.0
                    elif 150 <= index < 158:
                        gain = 0.012 - 0.010 * (index - 150) / 8.0
                    elif 158 <= index < 163:
                        gain = 0.002
                    elif index >= 163:
                        gain = 0.009
                    wheels = [50.0] * 4
                    for target in targets:
                        wheels[target] *= 1.0 + gain
                    results.append(
                        detector.push(
                            WheelFrame.from_sequences(index * 0.01, wheels)
                        )
                    )
                expected = tuple(index in targets for index in range(4))
                self.assertEqual(results[-1].blowout_alarms, expected)

    def test_three_equal_simultaneous_changes_are_not_observable(self):
        detector = WaveletShapeBlowoutDetector()
        results = []
        for index in range(260):
            gain = 0.0
            if 140 <= index < 150:
                gain = 0.012 * (index - 140) / 10.0
            elif 150 <= index < 158:
                gain = 0.012 - 0.010 * (index - 150) / 8.0
            elif 158 <= index < 163:
                gain = 0.002
            elif index >= 163:
                gain = 0.009
            wheels = [50.0 * (1.0 + gain)] * 3 + [50.0]
            results.append(
                detector.push(WheelFrame.from_sequences(index * 0.01, wheels))
            )
        self.assertFalse(any(any(result.blowout_alarms) for result in results))

    def test_confirmed_normal_wheel_anchors_three_blowout_wheels(self):
        detector = WaveletShapeBlowoutDetector()
        results = []
        for index in range(260):
            gain = 0.0
            if 140 <= index < 150:
                gain = 0.012 * (index - 140) / 10.0
            elif 150 <= index < 158:
                gain = 0.012 - 0.010 * (index - 150) / 8.0
            elif 158 <= index < 163:
                gain = 0.002
            elif index >= 163:
                gain = 0.009
            wheels = [50.0 * (1.0 + gain)] * 3 + [50.0]
            results.append(
                detector.push(
                    WheelFrame.from_sequences(
                        index * 0.01,
                        wheels,
                        normal_signals=[None, None, None, True],
                    )
                )
            )
        self.assertEqual(
            results[-1].blowout_alarms,
            (True, True, True, False),
        )
        self.assertEqual(
            results[-1].reference_sources[:3],
            ("confirmed_normal:RR",) * 3,
        )

    def test_all_eight_real_templates_detect_rr(self):
        template_dir = (
            Path(__file__).resolve().parents[1]
            / "wheel_cog_outputs"
            / "real_blowout_templates"
            / "templates"
        )
        for path in sorted(template_dir.glob("E*_RR.csv")):
            with self.subTest(template=path.name):
                detector = WaveletShapeBlowoutDetector(
                    WaveletShapeConfig(target_wheels=(3,))
                )
                results = []
                with path.open(newline="", encoding="utf-8") as handle:
                    for row_number, row in enumerate(csv.DictReader(handle)):
                        t_sec = float(row["time_rel_s"]) + 5.0
                        gain = float(row["normalized_gain_raw"])
                        results.append(
                            detector.push(
                                WheelFrame.from_sequences(
                                    t_sec,
                                    [50.0, 50.0, 50.0, 50.0 * gain],
                                )
                            )
                        )
                        if t_sec > 6.0:
                            break
                self.assertTrue(results[-1].blowout_alarms[3])
                first = next(result for result in results if result.new_blowouts[3])
                self.assertGreaterEqual(first.estimated_onset_times_s[3], 4.9)
                self.assertLessEqual(first.estimated_onset_times_s[3], 5.2)

    def test_validation_and_reset(self):
        with self.assertRaises(ValueError):
            WaveletShapeConfig(smooth_window=4)
        detector = WaveletShapeBlowoutDetector()
        detector.push(WheelFrame.from_sequences(0.0, [50.0] * 4))
        with self.assertRaises(ValueError):
            detector.push(WheelFrame.from_sequences(0.0, [50.0] * 4))
        detector.reset()
        result = detector.push(WheelFrame.from_sequences(0.0, [50.0] * 4))
        self.assertEqual(result.states, ("warming",) * 4)


if __name__ == "__main__":
    unittest.main()
