from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

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


WHEEL_NAMES = ("FL", "FR", "RL", "RR")
WHEEL_COLUMNS = tuple(f"wheel{index}_corrected_rad_s" for index in range(4))
ALGORITHMS = (
    "wheel_speed_only_baseline",
    "wheel_speed_only_optimized",
    "quant_baseline",
    "quant_optimized",
)


def _new_detector(name: str) -> tuple[object, object]:
    if name == "wheel_speed_only_baseline":
        config = WheelSpeedBlowoutConfig(early_confirm_frames=None)
        return WheelSpeedBlowoutDetector(config), config
    if name == "wheel_speed_only_optimized":
        config = WheelSpeedBlowoutConfig()
        return WheelSpeedBlowoutDetector(config), config
    if name == "quant_baseline":
        config = QuantBlowoutConfig(confirm_frames=70, max_physical_peak=1.0)
        return QuantBlowoutDetector(config), config
    if name == "quant_optimized":
        config = QuantBlowoutConfig()
        return QuantBlowoutDetector(config), config
    raise ValueError(f"unsupported algorithm: {name}")


def _target_indices(value: str) -> set[int]:
    normalized = value.replace("+", ";").replace(",", ";")
    names = {item.strip().upper() for item in normalized.split(";") if item.strip()}
    return {index for index, name in enumerate(WHEEL_NAMES) if name in names}


def _read_frame(row: dict[str, str]) -> tuple[float, list[float]]:
    return float(row["time_s"]), [float(row[column]) for column in WHEEL_COLUMNS]


def _push(detector: object, name: str, t_sec: float, wheels: Sequence[float]) -> object:
    if name.startswith("wheel_speed_only"):
        return detector.push(WheelSpeedFrame.from_sequences(t_sec, wheels))  # type: ignore[attr-defined]
    return detector.push(QuantFrame.from_sequences(t_sec, wheels))  # type: ignore[attr-defined]


def _validate_columns(fieldnames: Sequence[str] | None, path: Path) -> None:
    if fieldnames is None:
        raise ValueError(f"CSV has no header: {path}")
    required = ("time_s", *WHEEL_COLUMNS)
    missing = [column for column in required if column not in fieldnames]
    if missing:
        raise ValueError(f"CSV is missing columns {missing}: {path}")


def _evaluate_augmented_one(
    task: tuple[Path, dict[str, str], tuple[str, ...], float]
) -> list[dict[str, Any]]:
    dataset_dir, manifest_row, algorithm_names, detection_window_s = task
    detectors = {name: _new_detector(name)[0] for name in algorithm_names}
    first_alarms = {name: [None] * 4 for name in algorithm_names}
    candidate_entries = {name: 0 for name in algorithm_names}
    previous_states = {name: ["warming"] * 4 for name in algorithm_names}
    sample_path = dataset_dir / manifest_row["sample_file"]

    with sample_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        _validate_columns(reader.fieldnames, sample_path)
        for row in reader:
            t_sec, wheels = _read_frame(row)
            for name, detector in detectors.items():
                result = _push(detector, name, t_sec, wheels)
                for wheel in range(4):
                    if (
                        result.states[wheel] == "candidate"
                        and previous_states[name][wheel] != "candidate"
                    ):
                        candidate_entries[name] += 1
                    if result.new_blowouts[wheel] and first_alarms[name][wheel] is None:
                        first_alarms[name][wheel] = t_sec
                previous_states[name] = list(result.states)

    event_text = manifest_row.get("event_time_in_sample_s", "").strip().lower()
    event_time = None if event_text in {"", "nan", "none"} else float(event_text)
    targets = _target_indices(manifest_row.get("target_wheels", ""))
    output: list[dict[str, Any]] = []
    for name in algorithm_names:
        alarm_pairs = [
            (wheel, alarm_time)
            for wheel, alarm_time in enumerate(first_alarms[name])
            if alarm_time is not None
        ]
        pre_event_false = event_time is not None and any(
            alarm_time < event_time for _, alarm_time in alarm_pairs
        )
        wrong_wheel_false = any(wheel not in targets for wheel, _ in alarm_pairs)
        target_detections = (
            []
            if event_time is None
            else [
                alarm_time
                for wheel, alarm_time in alarm_pairs
                if wheel in targets and alarm_time >= event_time
            ]
        )
        detection_time = min(target_detections, default=None)
        delay_s = None if detection_time is None else detection_time - event_time  # type: ignore[operator]
        sample_type = manifest_row["sample_type"]
        false_alarm = bool(alarm_pairs) if sample_type == "normal" else (
            pre_event_false or wrong_wheel_false
        )
        detected_within_window = (
            sample_type == "event"
            and delay_s is not None
            and delay_s <= detection_window_s
            and not false_alarm
        )
        output.append(
            {
                "algorithm": name,
                "sample_id": manifest_row["sample_id"],
                "sample_type": sample_type,
                "source_event_id": manifest_row["source_event_id"],
                "is_augmented": manifest_row.get("is_augmented", ""),
                "target_wheels": manifest_row.get("target_wheels", ""),
                "event_time_s": event_time,
                "detected_within_window": detected_within_window,
                "delay_s": delay_s,
                "false_alarm": false_alarm,
                "pre_event_false_alarm": pre_event_false,
                "wrong_wheel_false_alarm": wrong_wheel_false,
                "alarm_wheels": ";".join(
                    WHEEL_NAMES[wheel] for wheel, _ in alarm_pairs
                ),
                "first_alarm_times_s": ";".join(
                    f"{WHEEL_NAMES[wheel]}:{alarm_time:.8f}"
                    for wheel, alarm_time in alarm_pairs
                ),
                "candidate_entries": candidate_entries[name],
            }
        )
    return output


def _evaluate_robust_one(
    task: tuple[Path, tuple[str, ...]]
) -> list[dict[str, Any]]:
    path, algorithm_names = task
    detectors = {name: _new_detector(name)[0] for name in algorithm_names}
    first_alarms = {name: [None] * 4 for name in algorithm_names}
    candidate_entries = {name: 0 for name in algorithm_names}
    previous_states = {name: ["warming"] * 4 for name in algorithm_names}
    frames = 0
    valid_frames = {name: 0 for name in algorithm_names}
    first_time: float | None = None
    last_time: float | None = None

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        _validate_columns(reader.fieldnames, path)
        for row in reader:
            t_sec, wheels = _read_frame(row)
            frames += 1
            first_time = t_sec if first_time is None else first_time
            last_time = t_sec
            for name, detector in detectors.items():
                result = _push(detector, name, t_sec, wheels)
                valid_frames[name] += int(result.speed_valid)
                for wheel in range(4):
                    if (
                        result.states[wheel] == "candidate"
                        and previous_states[name][wheel] != "candidate"
                    ):
                        candidate_entries[name] += 1
                    if result.new_blowouts[wheel] and first_alarms[name][wheel] is None:
                        first_alarms[name][wheel] = t_sec
                previous_states[name] = list(result.states)

    duration_s = 0.0 if first_time is None or last_time is None else last_time - first_time
    return [
        {
            "algorithm": name,
            "case": str(path.parent.relative_to(path.parents[2])),
            "csv_path": str(path),
            "frames": frames,
            "valid_frames": valid_frames[name],
            "duration_s": duration_s,
            "false_alarm": any(value is not None for value in first_alarms[name]),
            "alarm_wheels": ";".join(
                WHEEL_NAMES[index]
                for index, value in enumerate(first_alarms[name])
                if value is not None
            ),
            "candidate_entries": candidate_entries[name],
        }
        for name in algorithm_names
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _summarize(
    augmented_rows: list[dict[str, Any]], robust_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema_version": 1,
        "evaluation_scope": {
            "detection_window_s": 2.0,
            "important": (
                "Augmented samples share eight source events and are a development "
                "stress test, not an independent locked-parameter blind test."
            ),
        },
        "algorithms": {},
    }
    for name in ALGORITHMS:
        rows = [row for row in augmented_rows if row["algorithm"] == name]
        events = [row for row in rows if row["sample_type"] == "event"]
        normals = [row for row in rows if row["sample_type"] == "normal"]
        originals = [row for row in events if str(row["is_augmented"]) == "0"]
        robust = [row for row in robust_rows if row["algorithm"] == name]
        delays = [
            float(row["delay_s"])
            for row in events
            if row["detected_within_window"] and row["delay_s"] is not None
        ]
        original_delays = [
            float(row["delay_s"])
            for row in originals
            if row["detected_within_window"] and row["delay_s"] is not None
        ]
        _, config = _new_detector(name)
        summary["algorithms"][name] = {
            "config": asdict(config),
            "real_positive_replay": {
                "samples": len(originals),
                "detected_within_2s": sum(
                    bool(row["detected_within_window"]) for row in originals
                ),
                "false_alarm_samples": sum(bool(row["false_alarm"]) for row in originals),
                "mean_delay_s": mean(original_delays) if original_delays else None,
                "max_delay_s": max(original_delays, default=None),
            },
            "augmented_replay": {
                "samples": len(rows),
                "event_samples": len(events),
                "events_detected_within_2s": sum(
                    bool(row["detected_within_window"]) for row in events
                ),
                "event_misses": [
                    row["sample_id"]
                    for row in events
                    if not row["detected_within_window"]
                ],
                "event_false_alarm_samples": sum(
                    bool(row["false_alarm"]) for row in events
                ),
                "normal_samples": len(normals),
                "normal_false_alarm_samples": sum(
                    bool(row["false_alarm"]) for row in normals
                ),
                "mean_delay_s": mean(delays) if delays else None,
                "p95_delay_s": _percentile(delays, 0.95),
                "max_delay_s": max(delays, default=None),
                "candidate_entries": sum(int(row["candidate_entries"]) for row in rows),
                "per_source": {
                    source: {
                        "event_samples": sum(
                            row["sample_type"] == "event"
                            for row in rows
                            if row["source_event_id"] == source
                        ),
                        "events_detected_within_2s": sum(
                            bool(row["detected_within_window"])
                            for row in rows
                            if row["source_event_id"] == source
                            and row["sample_type"] == "event"
                        ),
                        "normal_false_alarm_samples": sum(
                            bool(row["false_alarm"])
                            for row in rows
                            if row["source_event_id"] == source
                            and row["sample_type"] == "normal"
                        ),
                    }
                    for source in sorted({row["source_event_id"] for row in rows})
                },
            },
            "real_normal_road_replay": {
                "cases": len(robust),
                "frames": sum(int(row["frames"]) for row in robust),
                "duration_hours": sum(float(row["duration_s"]) for row in robust) / 3600.0,
                "false_alarm_cases": sum(bool(row["false_alarm"]) for row in robust),
                "false_alarm_case_names": [
                    row["case"] for row in robust if row["false_alarm"]
                ],
                "candidate_entries": sum(int(row["candidate_entries"]) for row in robust),
            },
        }
    return summary


def _run_parallel(tasks: list[object], worker: object, jobs: int, label: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(worker, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            output.extend(future.result())
            if completed % 10 == 0 or completed == len(futures):
                print(f"{label}: {completed}/{len(futures)}", flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline and optimized wheel-speed-only detectors."
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("augmented_event_dataset_v2")
    )
    parser.add_argument(
        "--robust-dir", type=Path, default=Path("robust_data_results/cases")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("speed_algorithm_evaluation")
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--skip-robust", action="store_true")
    args = parser.parse_args()
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    return args


def main() -> None:
    args = parse_args()
    manifest_path = args.dataset_dir / "manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        manifest = list(csv.DictReader(handle))
    algorithm_names = tuple(ALGORITHMS)
    augmented_tasks = [
        (args.dataset_dir, row, algorithm_names, 2.0) for row in manifest
    ]
    augmented_rows = _run_parallel(
        augmented_tasks, _evaluate_augmented_one, args.jobs, "augmented"
    )
    augmented_rows.sort(key=lambda row: (row["algorithm"], row["sample_id"]))
    _write_csv(args.output_dir / "augmented_evaluation.csv", augmented_rows)

    robust_rows: list[dict[str, Any]] = []
    if not args.skip_robust:
        robust_paths = sorted(args.robust_dir.rglob("wheel_speed_raw_vs_corrected.csv"))
        robust_tasks = [(path, algorithm_names) for path in robust_paths]
        robust_rows = _run_parallel(
            robust_tasks, _evaluate_robust_one, args.jobs, "robust"
        )
        robust_rows.sort(key=lambda row: (row["algorithm"], row["case"]))
        _write_csv(args.output_dir / "robust_evaluation.csv", robust_rows)

    summary = _summarize(augmented_rows, robust_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"summary: {(args.output_dir / 'summary.json').resolve()}")


if __name__ == "__main__":
    main()
