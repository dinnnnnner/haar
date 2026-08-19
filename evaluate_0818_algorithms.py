from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from build_0818_display import (
    WHEEL_NAMES,
    corrected_wheel_speeds,
    iter_raw_frames,
    learn_phase_factors,
    sustained_signal_onset,
)
from quant_wheel_blowout_detector import (
    QuantBlowoutConfig,
    QuantBlowoutDetector,
    QuantFrame,
)
from wheel_speed_only_blowout_detector import (
    WheelSpeedBlowoutConfig,
    WheelSpeedBlowoutDetector,
    WheelSpeedFrame,
)


WHEEL_COLUMNS = tuple(f"wheel{index}_corrected_rad_s" for index in range(4))
DEFERRED_CASES = ("40kph_RRBlowOut",)
ALGORITHMS = (
    "wheelonly_previous",
    "wheelonly_0818",
    "quant_previous",
    "quant_0818",
)


def algorithm_configs() -> dict[str, WheelSpeedBlowoutConfig | QuantBlowoutConfig]:
    """Return independent previous/default and 0818-tuned configurations."""

    return {
        "wheelonly_previous": WheelSpeedBlowoutConfig(
            min_individual_edge=0.0058,
            min_diagonal_edge=0.0058,
            max_individual_peak=0.0250,
            min_diagonal_persistence=0.0055,
            braking_min_mate_persistence=-0.0035,
            early_max_braking_speed_range=0.035,
            max_braking_speed_range=0.050,
            min_braking_range_fraction=0.0,
        ),
        "wheelonly_0818": WheelSpeedBlowoutConfig(),
        "quant_previous": QuantBlowoutConfig(
            confirm_frames=55,
            persistence_tail_frames=40,
            shock_trigger_z=5.0,
            max_shock_trigger_z=1.0e9,
            shock_isolation_z=2.0,
            min_physical_edge=0.0038,
            min_physical_peak=0.0060,
            min_physical_peak_with_common_motion=0.0060,
            small_peak_max_common_log_range=0.050,
            min_physical_persistence=0.0042,
            max_physical_peak=0.0250,
            min_isolation_fraction=0.95,
            max_peer_physical_median=1.0,
            min_median_risk=55.0,
            max_braking_log_range=0.050,
            min_braking_range_fraction=0.0,
        ),
        "quant_0818": QuantBlowoutConfig(),
    }


def _new_detector(
    name: str, config: WheelSpeedBlowoutConfig | QuantBlowoutConfig
) -> WheelSpeedBlowoutDetector | QuantBlowoutDetector:
    if name.startswith("wheelonly"):
        assert isinstance(config, WheelSpeedBlowoutConfig)
        return WheelSpeedBlowoutDetector(config)
    assert isinstance(config, QuantBlowoutConfig)
    return QuantBlowoutDetector(config)


def _push(
    name: str,
    detector: WheelSpeedBlowoutDetector | QuantBlowoutDetector,
    t_sec: float,
    wheels: Sequence[float],
) -> Any:
    if name.startswith("wheelonly"):
        assert isinstance(detector, WheelSpeedBlowoutDetector)
        return detector.push(WheelSpeedFrame.from_sequences(t_sec, wheels))
    assert isinstance(detector, QuantBlowoutDetector)
    return detector.push(QuantFrame.from_sequences(t_sec, wheels))


def evaluate_0818_case(path: Path, detection_window_s: float = 2.0) -> list[dict[str, Any]]:
    frames = list(iter_raw_frames(path))
    speeds = corrected_wheel_speeds(frames, learn_phase_factors(frames))
    event_time = sustained_signal_onset([frame.blowout_signal for frame in frames])
    if event_time is None:
        raise ValueError(f"没有持续爆胎信号：{path}")

    configs = algorithm_configs()
    detectors = {name: _new_detector(name, config) for name, config in configs.items()}
    first_alarms = {name: [None] * 4 for name in ALGORITHMS}
    for index, wheels in enumerate(speeds):
        t_sec = index / 100.0
        for name, detector in detectors.items():
            result = _push(name, detector, t_sec, wheels)
            for wheel, is_new in enumerate(result.new_blowouts):
                if is_new and first_alarms[name][wheel] is None:
                    first_alarms[name][wheel] = t_sec

    rows = []
    for name in ALGORITHMS:
        alarms = first_alarms[name]
        target_alarm = alarms[3]
        pre_event = any(value is not None and value < event_time for value in alarms)
        wrong_wheel = any(value is not None for value in alarms[:3])
        delay = (
            None
            if target_alarm is None or target_alarm < event_time
            else target_alarm - event_time
        )
        delay_frames = (
            None
            if delay is None
            else int(round(delay * configs[name].sample_rate_hz))
        )
        rows.append(
            {
                "algorithm": name,
                "case": path.stem,
                "event_time_s": event_time,
                "target_wheel": "RR",
                "detected_within_2s": (
                    delay is not None
                    and delay <= detection_window_s
                    and not pre_event
                    and not wrong_wheel
                ),
                "delay_s": delay,
                "delay_frames": delay_frames,
                "detected_within_20_frames": (
                    delay_frames is not None
                    and delay_frames <= 20
                    and not pre_event
                    and not wrong_wheel
                ),
                "pre_event_false_alarm": pre_event,
                "wrong_wheel_false_alarm": wrong_wheel,
                "alarm_wheels": ";".join(
                    WHEEL_NAMES[index]
                    for index, value in enumerate(alarms)
                    if value is not None
                ),
                "first_alarm_times_s": ";".join(
                    f"{WHEEL_NAMES[index]}:{value:.2f}"
                    for index, value in enumerate(alarms)
                    if value is not None
                ),
            }
        )
    return rows


def _evaluate_normal_case(path: Path) -> list[dict[str, Any]]:
    configs = algorithm_configs()
    detectors = {name: _new_detector(name, config) for name, config in configs.items()}
    first_alarms = {name: [None] * 4 for name in ALGORITHMS}
    frames = 0
    first_time: float | None = None
    last_time: float | None = None
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = ("time_s", *WHEEL_COLUMNS)
        missing = [name for name in required if name not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"CSV 缺少列 {missing}: {path}")
        for row in reader:
            t_sec = float(row["time_s"])
            wheels = [float(row[column]) for column in WHEEL_COLUMNS]
            frames += 1
            first_time = t_sec if first_time is None else first_time
            last_time = t_sec
            for name, detector in detectors.items():
                result = _push(name, detector, t_sec, wheels)
                for wheel, is_new in enumerate(result.new_blowouts):
                    if is_new and first_alarms[name][wheel] is None:
                        first_alarms[name][wheel] = t_sec

    try:
        case = str(path.parent.relative_to(path.parents[2]))
    except (IndexError, ValueError):
        case = path.parent.name
    duration = 0.0 if first_time is None or last_time is None else last_time - first_time
    return [
        {
            "algorithm": name,
            "case": case,
            "csv_path": str(path),
            "frames": frames,
            "duration_s": duration,
            "false_alarm": any(value is not None for value in first_alarms[name]),
            "alarm_wheels": ";".join(
                WHEEL_NAMES[index]
                for index, value in enumerate(first_alarms[name])
                if value is not None
            ),
        }
        for name in ALGORITHMS
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(
    event_rows: list[dict[str, Any]], normal_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    configs = algorithm_configs()
    algorithms = {}
    for name in ALGORITHMS:
        positives = [row for row in event_rows if row["algorithm"] == name]
        normals = [row for row in normal_rows if row["algorithm"] == name]
        delays = [
            float(row["delay_s"])
            for row in positives
            if row["detected_within_2s"] and row["delay_s"] is not None
        ]
        delay_frames = [
            int(row["delay_frames"])
            for row in positives
            if row["detected_within_2s"] and row["delay_frames"] is not None
        ]
        algorithms[name] = {
            "config": asdict(configs[name]),
            "0818_positive_replay": {
                "samples": len(positives),
                "detected_within_2s": sum(
                    bool(row["detected_within_2s"]) for row in positives
                ),
                "pre_event_false_alarm_samples": sum(
                    bool(row["pre_event_false_alarm"]) for row in positives
                ),
                "wrong_wheel_false_alarm_samples": sum(
                    bool(row["wrong_wheel_false_alarm"]) for row in positives
                ),
                "mean_delay_s": mean(delays) if delays else None,
                "max_delay_s": max(delays, default=None),
                "mean_delay_frames": mean(delay_frames) if delay_frames else None,
                "max_delay_frames": max(delay_frames, default=None),
                "detected_within_20_frames": sum(
                    bool(row["detected_within_20_frames"])
                    for row in positives
                ),
                "misses": [
                    row["case"] for row in positives if not row["detected_within_2s"]
                ],
            },
            "normal_road_replay": {
                "cases": len(normals),
                "frames": sum(int(row["frames"]) for row in normals),
                "duration_hours": sum(float(row["duration_s"]) for row in normals)
                / 3600.0,
                "false_alarm_cases": sum(bool(row["false_alarm"]) for row in normals),
                "false_alarm_case_names": [
                    row["case"] for row in normals if row["false_alarm"]
                ],
            },
        }
    return {
        "schema_version": 1,
        "optimization_basis": {
            "positive_dataset": "0818",
            "included_cases": sorted({row["case"] for row in event_rows}),
            "deferred_cases": list(DEFERRED_CASES),
            "excluded_positive_datasets": ["ly"],
            "note": "40kph 标注和轮速反馈问题暂不纳入；正常道路仅用于误报回放。",
        },
        "algorithms": algorithms,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="独立评估基于 0818 优化的两套轮速算法")
    parser.add_argument("--input-dir", type=Path, default=Path("0818"))
    parser.add_argument(
        "--robust-dir", type=Path, default=Path("robust_data_results/cases")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("0818_algorithm_evaluation")
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--skip-robust", action="store_true")
    args = parser.parse_args()
    if args.jobs <= 0:
        parser.error("--jobs 必须为正数")
    return args


def main() -> None:
    args = parse_args()
    event_paths = [
        path
        for path in sorted(args.input_dir.glob("*.txt"))
        if path.stem not in DEFERRED_CASES
    ]
    if not event_paths:
        raise ValueError(f"没有可评估的 0818 数据：{args.input_dir}")
    event_rows = [row for path in event_paths for row in evaluate_0818_case(path)]
    event_rows.sort(key=lambda row: (row["algorithm"], row["case"]))
    _write_csv(args.output_dir / "0818_evaluation.csv", event_rows)

    normal_rows: list[dict[str, Any]] = []
    if not args.skip_robust:
        paths = sorted(args.robust_dir.rglob("wheel_speed_raw_vs_corrected.csv"))
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(_evaluate_normal_case, path) for path in paths]
            for completed, future in enumerate(as_completed(futures), 1):
                normal_rows.extend(future.result())
                if completed % 5 == 0 or completed == len(futures):
                    print(f"normal: {completed}/{len(futures)}", flush=True)
        normal_rows.sort(key=lambda row: (row["algorithm"], row["case"]))
        _write_csv(args.output_dir / "normal_road_evaluation.csv", normal_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_summary(event_rows, normal_rows), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
