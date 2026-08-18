from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from .console_data import analyze_window, scan_csv
from .detector import WheelSpeedBlowoutConfig
from .serve_console import ConsoleState


class ConsoleDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.csv_path = self.root / "wheel_speed.csv"
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["time_s", *(f"wheel{i}_corrected_rad_s" for i in range(4))]
            )
            for index in range(430):
                wheels = [50.0] * 4
                if 180 <= index < 192:
                    wheels[0] *= 1.011
                if index >= 240:
                    wheels[3] *= 1.011
                writer.writerow([index * 0.01, *wheels])
        self.cfg = WheelSpeedBlowoutConfig(
            baseline_min_samples=120,
            baseline_window=300,
            confirm_frames=60,
            persistence_tail_frames=35,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_scan_collects_confirmed_candidate(self) -> None:
        scan = scan_csv(self.csv_path, self.cfg)

        confirmed = [item for item in scan.suspects if item.confirmed]
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0].wheel_index, 3)
        self.assertFalse(confirmed[0].cancelled)
        self.assertEqual(scan.first_alarm_times[3], confirmed[0].end_s)
        self.assertIsNotNone(confirmed[0].peak_diagonal_gain_pct)
        cancelled = [item for item in scan.suspects if item.cancelled]
        self.assertEqual(len(cancelled), 1)
        self.assertEqual(cancelled[0].wheel_index, 0)
        self.assertAlmostEqual(cancelled[0].duration_s, 0.59)

    def test_scan_does_not_count_candidate_open_at_eof_as_cancelled(self) -> None:
        unfinished = self.root / "unfinished.csv"
        with unfinished.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["time_s", *(f"wheel{i}_corrected_rad_s" for i in range(4))]
            )
            for index in range(200):
                wheels = [50.0] * 4
                if index >= 180:
                    wheels[0] *= 1.011
                writer.writerow([index * 0.01, *wheels])

        scan = scan_csv(unfinished, self.cfg)

        self.assertEqual(len(scan.suspects), 1)
        self.assertFalse(scan.suspects[0].confirmed)
        self.assertFalse(scan.suspects[0].cancelled)

    def test_window_replay_keeps_four_wheel_evidence(self) -> None:
        data = analyze_window(self.csv_path, 2.2, 3.4, self.cfg)

        self.assertEqual(len(data.wheels), 4)
        self.assertEqual(len(data.individual_edges), 4)
        self.assertEqual(data.times[0], 2.2)
        self.assertTrue(any(data.alarms[3]))

    def test_console_renders_index_case_and_custom_csv(self) -> None:
        event_dir = self.root / "events"
        event_dir.mkdir()
        event_csv = event_dir / "event.csv"
        event_csv.write_bytes(self.csv_path.read_bytes())
        event_manifest = event_dir / "manifest.csv"
        with event_manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "sample_id",
                    "sample_type",
                    "source_event_id",
                    "source_file",
                    "sample_file",
                    "event_time_in_sample_s",
                    "is_augmented",
                ),
            )
            writer.writeheader()
            writer.writerow(
                {
                    "sample_id": "E01_event_000",
                    "sample_type": "event",
                    "source_event_id": "E01",
                    "source_file": "event.csv",
                    "sample_file": "event.csv",
                    "event_time_in_sample_s": "2.40",
                    "is_augmented": "0",
                }
            )
        normal_manifest = self.root / "normal.csv"
        with normal_manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("sample_id", "sample_file", "source_file", "scenario"),
            )
            writer.writeheader()
            writer.writerow(
                {
                    "sample_id": "R001",
                    "sample_file": str(self.csv_path),
                    "source_file": "normal",
                    "scenario": "test road",
                }
            )
        validation = self.root / "validation.json"
        validation.write_text(
            json.dumps(
                {
                    "algorithm": "test",
                    "evaluation_date": "2026-08-17",
                    "real_positive_replay": {
                        "samples": 1,
                        "detected": 1,
                        "mean_confirmation_delay_s": 0.8,
                        "max_confirmation_delay_s": 0.8,
                        "wrong_wheel_or_pre_event_alarms": 0,
                    },
                    "augmented_replay": {
                        "event_samples": 1,
                        "events_detected_within_2s": 1,
                        "event_misses": 0,
                    },
                    "real_normal_road_replay": {
                        "cases": 1,
                        "frames": 430,
                        "duration_hours": 0.001,
                        "false_alarm_cases": 0,
                    },
                }
            ),
            encoding="utf-8",
        )
        state = ConsoleState(
            event_manifest,
            normal_manifest,
            validation,
            self.cfg,
        )

        index_html = state.render_index()
        self.assertIn("纯四轮轮速爆胎算法控制台", index_html)
        self.assertIn("取消耗时统计", index_html)
        case_html = state.render_case("E01", None, None)
        self.assertIn("候选区间", case_html)
        self.assertIn("耗时 ↓", case_html)
        cancelled = state.cancelled_candidates()
        self.assertEqual(len(cancelled), 2)
        self.assertTrue(all(item.duration_s == cancelled[0].duration_s for item in cancelled))
        cancellation_html = state.render_cancellations()
        self.assertIn("误报取消耗时统计", cancellation_html)
        self.assertIn("正常道路 1 个", cancellation_html)
        self.assertIn("跳转查看", cancellation_html)
        self.assertIn("/case/E01?start=", cancellation_html)
        self.assertIn("自定义 CSV", state.render_custom(str(self.csv_path), 2.2, 3.4))


if __name__ == "__main__":
    unittest.main()
