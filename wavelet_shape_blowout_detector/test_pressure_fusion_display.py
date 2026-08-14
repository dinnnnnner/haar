from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from .pressure_fusion_detector import PressureFusionConfig
from .pressure_fusion_display import analyze_window, build_figure, scan_csv


class PressureFusionDisplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temporary_directory.name) / "wheel_speed.csv"
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["time_s", *(f"wheel{i}_corrected_rad_s" for i in range(4))]
            )
            for index in range(420):
                common = 50.0 + min(index, 250) * 0.01
                wheels = [common] * 4
                if index >= 240:
                    wheels[0] *= 1.011
                writer.writerow([index * 0.01, *wheels])
        self.cfg = PressureFusionConfig(
            baseline_min_samples=120,
            baseline_window=300,
            confirm_frames=60,
            persistence_tail_frames=35,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_scan_finds_and_classifies_candidate_interval(self) -> None:
        scan = scan_csv(self.csv_path, (1, 2), self.cfg)

        confirmed = [item for item in scan.suspects if item.confirmed]
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0].wheel_index, 0)
        self.assertGreater(confirmed[0].duration_s, 0.5)
        self.assertEqual(scan.first_alarm_times[0], confirmed[0].end_s)

    def test_plotly_window_contains_wheel_speed_and_candidate_rows(self) -> None:
        scan = scan_csv(self.csv_path, (1, 2), self.cfg)
        data = analyze_window(self.csv_path, (1, 2), 2.2, 3.3, self.cfg)
        figure = build_figure(data, self.cfg, scan.suspects, title="test")

        self.assertEqual(len(data.wheels), 4)
        self.assertEqual(figure.layout.dragmode, "pan")
        self.assertTrue(figure.layout.xaxis4.rangeslider.visible)
        self.assertTrue(any(trace.name == "FL" for trace in figure.data))
        self.assertTrue(any(trace.name == "FL 疑似" for trace in figure.data))


if __name__ == "__main__":
    unittest.main()
