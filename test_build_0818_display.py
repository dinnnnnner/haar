from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from build_0818_display import (
    analyze_wheel_speed_csv,
    corrected_wheel_speeds,
    iter_corrected_raw_speed_rows,
    iter_raw_frames,
    learn_phase_factors,
    learn_phase_factors_from_file,
    sustained_signal_onset,
)
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

    def test_streaming_raw_correction_matches_in_memory_correction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.txt"
            lines = ["Marks start\n", "Marks end\n"]
            timestamps = [1000, 2000, 3000, 4000]
            for index in range(100):
                for wheel in range(4):
                    timestamps[wheel] = (timestamps[wheel] + 900 + wheel * 10) % 65536
                    lines.append(f"1 {timestamps[wheel]}\n")
                lines.append(f"0 0 {int(index >= 80)}\n")
            path.write_text("".join(lines), encoding="utf-8")

            frames = list(iter_raw_frames(path))
            expected_factors = learn_phase_factors(frames)
            streamed_factors = learn_phase_factors_from_file(path)
            self.assertEqual(streamed_factors, expected_factors)
            expected_speeds = corrected_wheel_speeds(frames, expected_factors)
            streamed = list(iter_corrected_raw_speed_rows(path, streamed_factors))
            self.assertEqual([row[1] for row in streamed], expected_speeds)
            self.assertTrue(streamed[-1][2])

    def test_analyzes_only_requested_csv_window_after_causal_warmup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wheel_speed.csv"
            path.write_text(
                "time_s,wheel0_corrected_rad_s,wheel1_corrected_rad_s,"
                "wheel2_corrected_rad_s,wheel3_corrected_rad_s\n"
                + "".join(
                    f"{index / 100:.2f},10,10,10,10\n" for index in range(21)
                ),
                encoding="utf-8",
            )
            data = analyze_wheel_speed_csv(path, 0.10, 0.15)
            self.assertEqual(data.times, [0.10, 0.11, 0.12, 0.13, 0.14, 0.15])
            self.assertEqual(len(data.wheel_speeds[0]), 6)
            self.assertFalse(any(data.blowout_signal))

    def test_csv_replay_reads_requested_signal_columns_and_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wheel_speed.csv"
            path.write_text(
                "time_s,wheel0_corrected_rad_s,wheel1_corrected_rad_s,"
                "wheel2_corrected_rad_s,wheel3_corrected_rad_s,RR_blowout_signal\n"
                "0.00,10,10,10,10,0\n0.01,10,10,10,9,1\n",
                encoding="utf-8",
            )
            data = analyze_wheel_speed_csv(
                path,
                0.0,
                0.01,
                signal_columns=("RR_blowout_signal",),
                signal_event_time_s=0.01,
            )
            self.assertEqual(data.blowout_signal, [False, True])
            self.assertEqual(data.signal_event_time_s, 0.01)


class Serve0818ConsoleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parent
        cls.state = ConsoleState(root / "0818")
        cls.state_0819 = ConsoleState(
            root / "0818", input_0819_dir=root / "0819"
        )
        cls.robust_state = ConsoleState(
            root / "0818",
            robust_evaluation=(
                root / "speed_algorithm_evaluation" / "robust_evaluation.csv"
            ),
        )
        cls.ly_state = ConsoleState(
            root / "0818",
            ly_manifest=root / "augmented_event_dataset_v2" / "manifest.csv",
        )

    def test_index_and_summary_cover_all_new_records(self) -> None:
        page = self.state.render_index()
        self.assertIn("0818 Quant 爆胎控制台", page)
        self.assertIn("40kph_RRBlowOut", page)
        self.assertIn("Brk_RRBlowOut", page)
        self.assertEqual(len(self.state.summary()["cases"]), 4)

    def test_detail_supports_algorithm_and_time_window(self) -> None:
        page = self.state.render_case(
            "Acc_RRBlowOut", 60.0, 66.0, "quant"
        )
        self.assertIn("Plotly.newPlot", page)
        self.assertIn("quant：Hadamard 因子残差", page)
        self.assertIn("quant：轮位隔离度", page)
        self.assertIn("CUSUM", page)
        self.assertIn("曲线说明", page)
        self.assertIn("quant 因子残差", page)
        self.assertIn("物理投影：实线 level（持续量）/ 虚线 edge（边沿）", page)
        self.assertIn("粗线：锁存报警", page)
        self.assertIn("id='plot'", page)
        self.assertIn("layout.annotations.push", page)
        self.assertIn("<b>${titles[r-1]}</b>", page)
        self.assertIn("60.00–66.00s", page)
        self.assertIn("Plotly 开始", page)
        self.assertIn("全记录信号与候选", page)
        self.assertIn("不受图窗影响", page)
        self.assertIn('const MODE="quant"', page)
        self.assertNotIn("<span>wheel_only</span>", page)
        self.assertNotIn("<canvas", page)

    def test_0819_index_summary_and_detail_cover_all_new_records(self) -> None:
        summary = self.state_0819.summary("0819")
        self.assertEqual(len(summary["cases"]), 5)
        self.assertEqual(sum(case["frames"] for case in summary["cases"]), 34_100)
        self.assertTrue(
            all(case["signal_event_time_s"] is None for case in summary["cases"])
        )
        self.assertTrue(
            all(
                all(value is None for value in case["quant_first_alarms_s"].values())
                for case in summary["cases"]
            )
        )
        page = self.state_0819.render_index("0819")
        self.assertIn("0819 Quant 回放控制台", page)
        self.assertIn("34,100 帧", page)
        self.assertIn("20260819152701", page)
        self.assertIn("20260819_yacc_max", page)
        self.assertIn("0/5", page)
        detail = self.state_0819.render_case(
            "20260819152701", 120.0, 130.0, "quant", "0819"
        )
        self.assertIn("0819 新采数据", detail)
        self.assertIn("原始帧末信号均为 0", detail)
        self.assertIn("120.00–130.00s", detail)
        self.assertIn("dataset=0819", detail)
        self.assertIn('const MODE="quant"', detail)

    def test_0820_index_summary_and_windowed_detail_use_saved_evaluation(self) -> None:
        import json

        root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory) / "0820"
            input_dir.mkdir()
            raw_path = input_dir / "rough road.txt"
            lines = ["Marks start\n", "Marks end\n"]
            timestamps = [1000, 2000, 3000, 4000]
            for _index in range(100):
                for wheel in range(4):
                    timestamps[wheel] += 1000
                    lines.append(f"1 {timestamps[wheel]}\n")
                lines.append("0\n")
            raw_path.write_text("".join(lines), encoding="utf-8")
            stat = raw_path.stat()
            evaluation = Path(directory) / "summary.json"
            evaluation.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "cases": [
                            {
                                "case": raw_path.stem,
                                "input_file": raw_path.name,
                                "input_size": stat.st_size,
                                "frames": 100,
                                "duration_s": 0.99,
                                "signal_event_time_s": None,
                                "quant_first_alarms_s": {
                                    "FL": None,
                                    "FR": None,
                                    "RL": None,
                                    "RR": None,
                                },
                                "candidate_intervals": [],
                                "phase_factors": [[1.0] * 48 for _ in range(4)],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            state = ConsoleState(
                root / "0818",
                input_0820_dir=input_dir,
                evaluation_0820=evaluation,
            )
            page = state.render_index("0820")
            detail = state.render_case(
                raw_path.stem, 0.10, 0.20, "quant", "0820"
            )
        self.assertIn("0820 Quant 回放控制台", page)
        self.assertIn("0/1", page)
        self.assertEqual(len(state.summary("0820")["cases"]), 1)
        self.assertIn("0820 颠簸路数据", detail)
        self.assertIn("0.10–0.20s", detail)
        self.assertIn("dataset=0820", detail)

    def test_robust_index_and_detail_use_current_detectors(self) -> None:
        summary = self.robust_state.summary("robust")
        self.assertEqual(len(summary["cases"]), 37)
        page = self.robust_state.render_index("robust")
        self.assertIn("RobustData Quant 控制台", page)
        self.assertIn("24.78 小时", page)
        self.assertIn("R001", page)
        self.assertNotIn("wheel_only 误报", page)
        detail = self.robust_state.render_case(
            "R001", 0.0, 0.10, "quant", "robust"
        )
        self.assertIn("正常道路真值", detail)
        self.assertIn("dataset=robust", detail)
        self.assertIn("四轮相位校正轮速", detail)
        self.assertIn("quant：风险分", detail)

    def test_robust_sidebar_keeps_candidates_outside_plot_window(self) -> None:
        root = Path(__file__).resolve().parent
        source = (
            root
            / "augmented_event_dataset_v2"
            / "samples"
            / "E01_event_000.csv"
        )
        with tempfile.TemporaryDirectory() as directory:
            evaluation = Path(directory) / "robust_evaluation.csv"
            evaluation.write_text(
                "algorithm,case,csv_path,frames,valid_frames,duration_s,"
                "false_alarm,alarm_wheels,candidate_entries\n"
                f"quant_optimized,Synthetic/E01,{source},5000,5000,49.99,"
                "False,RR,1\n",
                encoding="utf-8",
            )
            state = ConsoleState(root / "0818", robust_evaluation=evaluation)
            page = state.render_case("R001", 0.0, 5.0, "quant", "robust")
        self.assertIn("0.00–5.00s", page)
        self.assertIn("40.20–40.44s", page)
        self.assertIn("RR 40.44s", page)
        self.assertIn("全记录信号与候选", page)

    def test_ly_index_and_detail_show_original_events_with_quant(self) -> None:
        summary = self.ly_state.summary("ly")
        self.assertEqual(len(summary["cases"]), 8)
        self.assertEqual(
            sum(
                case["quant_first_alarms_s"]["RR"] is not None
                for case in summary["cases"]
            ),
            2,
        )
        page = self.ly_state.render_index("ly")
        self.assertIn("LY 实车爆胎 Quant 控制台", page)
        self.assertIn("2/8", page)
        self.assertIn("E01_event_000", page)
        self.assertIn("20260116_yuan_baotai_rr100_45kmh.txt", page)
        detail = self.ly_state.render_case("E01", 39.0, 42.0, "quant", "ly")
        self.assertIn("LY 实车爆胎 · RR", detail)
        self.assertIn("原文件时刻 402.16s", detail)
        self.assertIn("const EVENT=40.0", detail)
        self.assertIn("const MODE=\"quant\"", detail)
        self.assertIn("黑线：原始爆胎信号", detail)
        self.assertIn("爆胎时刻", detail)
        self.assertIn("dataset=ly", detail)

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
            self.assertLessEqual(rows["quant_0818"]["delay_frames"], 46)
            self.assertFalse(rows["quant_0818"]["detected_within_20_frames"])


if __name__ == "__main__":
    unittest.main()
