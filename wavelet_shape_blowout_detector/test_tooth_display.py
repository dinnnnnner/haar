from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from .tooth_display import (
    analyze_tooth_file,
    build_tooth_figure,
    iter_tooth_frames,
)


class ToothDisplayTests(unittest.TestCase):
    @staticmethod
    def _write_factors(path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["phase", *(f"wheel{index}_factor" for index in range(4))])
            for phase in range(48):
                writer.writerow([phase, 1.0, 1.0, 1.0, 1.0])

    @staticmethod
    def _wheel_events(end_time_s: float, period_s: float, change_time_s: float | None = None) -> list[float]:
        events: list[float] = []
        time_s = 0.001
        while time_s < end_time_s:
            events.append(time_s)
            step = period_s
            if change_time_s is not None and time_s >= change_time_s:
                step *= 0.99
            time_s += step
        return events

    @classmethod
    def _write_raw(cls, path: Path) -> None:
        end_time_s = 2.0
        wheel_events = [
            cls._wheel_events(end_time_s, 0.001) for _ in range(3)
        ] + [cls._wheel_events(end_time_s, 0.001, change_time_s=1.0)]
        frame_count = int(end_time_s / 0.01)
        with path.open("w", encoding="utf-8") as handle:
            handle.write("Marks start\nMarks end\n")
            for frame_index in range(frame_count):
                left = frame_index * 0.01
                right = left + 0.01
                for events in wheel_events:
                    timestamps = [
                        round(time_s * 1.0e6) % 65_536
                        for time_s in events
                        if left <= time_s < right
                    ]
                    handle.write(
                        " ".join([str(len(timestamps)), *(str(value) for value in timestamps)])
                        + "\n"
                    )
                handle.write("0\n")

    def test_analyzes_relative_period_and_phase_directly_from_teeth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "raw.txt"
            factors_path = root / "learned_tooth_correction_factors.csv"
            self._write_raw(raw_path)
            self._write_factors(factors_path)

            frames = iter_tooth_frames(raw_path)
            self.assertEqual(len(next(iter(frames))), 4)
            data = analyze_tooth_file(
                raw_path,
                factors_path,
                0.5,
                1.8,
                baseline_seconds=0.4,
            )

            rr_after_event = [
                residual
                for time_s, residual in zip(
                    data.frame_times, data.period_residuals_pct[3]
                )
                if time_s >= 1.2
            ]
            self.assertLess(min(rr_after_event), -0.8)
            self.assertGreater(data.phase_residuals_teeth[3][-1], 5.0)
            self.assertGreater(data.displayed_tooth_events, 4_000)
            self.assertTrue(all(len(values) == len(data.frame_times) for values in data.tooth_counts))

            figure = build_tooth_figure(data, event_time_s=1.0)
            self.assertGreaterEqual(len(figure.data), 20)
            self.assertEqual(figure.layout.height, 1350)


if __name__ == "__main__":
    unittest.main()
