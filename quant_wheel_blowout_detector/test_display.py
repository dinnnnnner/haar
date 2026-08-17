from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from .detector import QuantBlowoutConfig
from .display import analyze_csv, build_figure, write_display_html


class QuantDisplayTests(unittest.TestCase):
    @staticmethod
    def _write_input(path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["time_s", *(f"wheel{i}_corrected_rad_s" for i in range(4))]
            )
            for index in range(720):
                common = 50.0 + min(index, 420) * 0.005
                steering = 0.0015 * max(0, min(index - 350, 80))
                wheels = [
                    common + steering,
                    common - steering,
                    common + steering,
                    common - steering,
                ]
                if index >= 480:
                    wheels[3] *= 1.011
                writer.writerow([index * 0.01, *wheels])

    def test_replays_full_history_and_builds_interactive_display(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "event.csv"
            output_path = root / "display.html"
            self._write_input(input_path)
            cfg = QuantBlowoutConfig()

            data = analyze_csv(
                input_path,
                cfg=cfg,
                start_time_s=4.5,
                end_time_s=6.2,
            )

            self.assertAlmostEqual(data.times[0], 4.5)
            self.assertAlmostEqual(data.times[-1], 6.2)
            self.assertEqual(data.first_alarm_times[:3], [None, None, None])
            self.assertIsNotNone(data.first_alarm_times[3])
            self.assertTrue(any(state == "candidate" for state in data.states[3]))
            self.assertTrue(data.alarms[3][-1])
            self.assertTrue(all(len(values) == len(data.times) for values in data.wheels))

            figure = build_figure(data, cfg, event_time_s=4.8)
            self.assertEqual(figure.layout.height, 1600)
            self.assertGreaterEqual(len(figure.data), 35)
            write_display_html(data, output_path, cfg, event_time_s=4.8)
            html = output_path.read_text(encoding="utf-8")
            self.assertIn("plotly", html.lower())
            self.assertIn("RR risk", html)

    def test_rejects_missing_columns_and_empty_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "event.csv"
            self._write_input(input_path)
            with self.assertRaisesRegex(ValueError, "missing CSV columns"):
                analyze_csv(input_path, wheel_columns=("FL", "FR", "RL", "RR"))
            with self.assertRaisesRegex(ValueError, "contains no samples"):
                analyze_csv(input_path, start_time_s=99.0)


if __name__ == "__main__":
    unittest.main()
