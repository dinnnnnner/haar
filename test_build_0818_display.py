from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from build_0818_display import iter_raw_frames, sustained_signal_onset
from evaluate_0818_algorithms import (
    DEFERRED_CASES,
    algorithm_configs,
    evaluate_0818_case,
)
from serve_0818_console import ConsoleState, candidate_intervals


class Build0818DisplayTests(unittest.TestCase):
    def test_reads_last_value_of_each_fifth_row_as_signal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.txt"
            path.write_text(
                "Marks start\nMarks end\n"
                "1 100\n1 110\n1 120\n1 130\n9 8 0\n"
                "1 200\n1 210\n1 220\n1 230\n9 8 1\n",
                encoding="utf-8",
            )
            frames = list(iter_raw_frames(path))
            self.assertEqual(len(frames), 2)
            self.assertFalse(frames[0].blowout_signal)
            self.assertTrue(frames[1].blowout_signal)
            self.assertEqual(frames[1].wheel_timestamps[3], (230,))

    def test_sustained_onset_ignores_short_initial_high(self) -> None:
        values = [True] * 5 + [False] * 10 + [True] * 20
        self.assertEqual(sustained_signal_onset(values, 20, 0.01), 0.15)


class Serve0818ConsoleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.state = ConsoleState(Path(__file__).resolve().parent / "0818")

    def test_index_and_summary_cover_all_new_records(self) -> None:
        page = self.state.render_index()
        self.assertIn("0818 爆胎双算法控制台", page)
        self.assertIn("40kph_RRBlowOut", page)
        self.assertIn("Brk_RRBlowOut", page)
        self.assertEqual(len(self.state.summary()["cases"]), 4)

    def test_detail_supports_algorithm_and_time_window(self) -> None:
        page = self.state.render_case(
            "Acc_RRBlowOut", 60.0, 66.0, "compare"
        )
        self.assertIn("Plotly.newPlot", page)
        self.assertIn("wheel_only：持续证据", page)
        self.assertIn("quant：Hadamard 因子残差", page)
        self.assertIn("quant：轮位隔离度", page)
        self.assertIn("CUSUM", page)
        self.assertIn("id='plot'", page)
        self.assertIn("60.00–66.00s", page)
        wheel_page = self.state.render_case(
            "Acc_RRBlowOut", 60.0, 66.0, "wheel"
        )
        self.assertIn('const MODE="wheel"', wheel_page)
        self.assertNotIn("<canvas", wheel_page)

    def test_candidate_scan_contains_confirmed_acc_rr(self) -> None:
        data = self.state.analyze("Acc_RRBlowOut")
        intervals = candidate_intervals(data)
        self.assertTrue(
            any(
                item.algorithm == "wheel"
                and item.wheel == 3
                and item.confirmed
                for item in intervals
            )
        )


class Evaluate0818AlgorithmsTests(unittest.TestCase):
    def test_optimized_algorithms_use_only_non_deferred_0818_cases(self) -> None:
        configs = algorithm_configs()
        self.assertNotEqual(configs["wheelonly_previous"], configs["wheelonly_0818"])
        self.assertNotEqual(configs["quant_previous"], configs["quant_0818"])

        input_dir = Path(__file__).resolve().parent / "0818"
        paths = [
            path
            for path in sorted(input_dir.glob("*.txt"))
            if path.stem not in DEFERRED_CASES
        ]
        self.assertEqual(
            [path.stem for path in paths],
            ["60kpa_RRBlowOut", "Acc_RRBlowOut", "Brk_RRBlowOut"],
        )
        for path in paths:
            rows = {row["algorithm"]: row for row in evaluate_0818_case(path)}
            for algorithm in ("wheelonly_0818", "quant_0818"):
                self.assertTrue(rows[algorithm]["detected_within_2s"])
                self.assertFalse(rows[algorithm]["pre_event_false_alarm"])
                self.assertFalse(rows[algorithm]["wrong_wheel_false_alarm"])


if __name__ == "__main__":
    unittest.main()
