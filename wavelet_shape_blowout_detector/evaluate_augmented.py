from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence

from .detector import REFERENCE_MODES, WHEEL_NAMES, WaveletShapeConfig
from .display import DisplayData, analyze_csv


CLASS_LABELS = {
    "EVENT_OK": "及时检出",
    "EVENT_LATE": "延迟检出",
    "EVENT_MISS": "漏报",
    "EVENT_FALSE_ALARM": "事件误报",
    "EVENT_FALSE_ALARM_AND_DETECTED": "误报后检出",
    "NORMAL_OK": "正常通过",
    "NORMAL_FALSE_ALARM": "正常误报",
}


def target_indices(value: str) -> set[int]:
    normalized = value.replace(";", ",").replace("+", ",")
    names = {item.strip().upper() for item in normalized.split(",") if item.strip()}
    return {index for index, name in enumerate(WHEEL_NAMES) if name in names}


def classify_result(
    row: dict[str, str],
    data: DisplayData,
    detection_window_s: float = 2.0,
) -> dict[str, object]:
    sample_type = row["sample_type"]
    event_text = row.get("event_time_in_sample_s", "").strip().lower()
    event_time = None if event_text in {"", "nan", "none"} else float(event_text)
    targets = target_indices(row.get("target_wheels", ""))
    alarm_wheels = {
        index for index, intervals in enumerate(data.alarm_intervals) if intervals
    }
    false_intervals: list[tuple[float, float]] = []
    false_wheels: set[int] = set()
    pre_event_false = False
    wrong_wheel_false = False
    detection_times: list[tuple[int, float]] = []

    if sample_type == "normal" or event_time is None:
        for index, intervals in enumerate(data.alarm_intervals):
            if intervals:
                false_wheels.add(index)
                false_intervals.extend(intervals)
        classification = "NORMAL_FALSE_ALARM" if false_intervals else "NORMAL_OK"
        delay_s = None
    else:
        for index, intervals in enumerate(data.alarm_intervals):
            for start_s, end_s in intervals:
                is_false = False
                if start_s < event_time:
                    pre_event_false = True
                    is_false = True
                if index not in targets:
                    wrong_wheel_false = True
                    is_false = True
                if is_false:
                    false_wheels.add(index)
                    false_intervals.append((start_s, end_s))
                if index in targets and start_s >= event_time:
                    detection_times.append((index, start_s))
        first_detection = min((time for _, time in detection_times), default=None)
        delay_s = None if first_detection is None else first_detection - event_time
        detected = delay_s is not None
        has_false = bool(false_intervals)
        if has_false and detected:
            classification = "EVENT_FALSE_ALARM_AND_DETECTED"
        elif has_false:
            classification = "EVENT_FALSE_ALARM"
        elif delay_s is None:
            classification = "EVENT_MISS"
        elif delay_s <= detection_window_s:
            classification = "EVENT_OK"
        else:
            classification = "EVENT_LATE"

    detection_wheels = {index for index, _ in detection_times}
    return {
        "sample_id": row["sample_id"],
        "sample_type": sample_type,
        "source_event_id": row["source_event_id"],
        "classification": classification,
        "classification_label": CLASS_LABELS[classification],
        "delay_s": delay_s,
        "alarm_wheels": ",".join(WHEEL_NAMES[index] for index in sorted(alarm_wheels)),
        "detection_wheels": ",".join(
            WHEEL_NAMES[index] for index in sorted(detection_wheels)
        ),
        "false_alarm_wheels": ",".join(
            WHEEL_NAMES[index] for index in sorted(false_wheels)
        ),
        "pre_event_false_alarm": pre_event_false,
        "wrong_wheel_false_alarm": wrong_wheel_false,
        "alarm_events": sum(len(intervals) for intervals in data.alarm_intervals),
        "false_alarm_intervals": false_intervals,
    }


def evaluate_manifest(
    dataset_dir: Path,
    output_csv: Path,
    output_json: Path,
    cfg: WaveletShapeConfig,
    detection_window_s: float = 2.0,
    jobs: int = 1,
    overwrite: bool = False,
    algorithm: str = "hard",
) -> list[dict[str, object]]:
    manifest_path = dataset_dir / "manifest.csv"
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    csv_fields = [
        "sample_id",
        "sample_type",
        "source_event_id",
        "classification",
        "classification_label",
        "delay_s",
        "alarm_wheels",
        "detection_wheels",
        "false_alarm_wheels",
        "pre_event_false_alarm",
        "wrong_wheel_false_alarm",
        "alarm_events",
    ]
    existing: list[dict[str, object]] = []
    if output_csv.exists() and not overwrite:
        with output_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            existing = list(csv.DictReader(handle))
    completed_ids = {str(row["sample_id"]) for row in existing}
    remaining_rows = [
        row for row in manifest if row["sample_id"] not in completed_ids
    ]
    if completed_ids:
        print(f"resuming after {len(completed_ids)}/{len(manifest)}", flush=True)
    tasks = [
        (dataset_dir, row, cfg, detection_window_s, algorithm)
        for row in remaining_rows
    ]
    if jobs == 1:
        iterator = map(_evaluate_one, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=jobs)
        iterator = executor.map(_evaluate_one, tasks, chunksize=4)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not existing or overwrite
    mode = "w" if write_header else "a"
    new_results: list[dict[str, object]] = []
    with output_csv.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        if write_header:
            writer.writeheader()
        for number, evaluation in enumerate(iterator, start=1):
            new_results.append(evaluation)
            writer.writerow({key: evaluation[key] for key in csv_fields})
            handle.flush()
            completed = len(existing) + number
            if completed % 25 == 0 or completed == len(manifest):
                print(f"evaluated {completed}/{len(manifest)}", flush=True)
    if executor is not None:
        executor.shutdown()

    results = [*existing, *new_results]
    if len(results) != len(manifest):
        print(
            f"checkpointed {len(results)}/{len(manifest)}; rerun to continue",
            flush=True,
        )
        return results
    summary = {
        "algorithm": algorithm,
        "samples": len(results),
        "classification_counts": {
            code: sum(row["classification"] == code for row in results)
            for code in CLASS_LABELS
        },
        "important": (
            "Augmented samples are correlated derivatives of eight real events; "
            "these counts are stress-test results, not independent real-road metrics."
        ),
    }
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return results


def _evaluate_one(
    task: tuple[Path, dict[str, str], WaveletShapeConfig, float, str],
) -> dict[str, object]:
    dataset_dir, row, cfg, detection_window_s, algorithm = task
    data = analyze_csv(
        dataset_dir / row["sample_file"], cfg=cfg, algorithm=algorithm
    )
    return classify_result(row, data, detection_window_s)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the 488 wavelet samples.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--detection-window", type=float, default=2.0)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--algorithm", choices=("hard", "evidence"), default="hard"
    )
    parser.add_argument(
        "--reference-mode", choices=REFERENCE_MODES, default="opposite_diagonal"
    )
    args = parser.parse_args()
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    return args


def main() -> None:
    args = parse_args()
    evaluate_manifest(
        args.dataset_dir,
        args.output_csv,
        args.output_json,
        WaveletShapeConfig(reference_mode=args.reference_mode),
        detection_window_s=args.detection_window,
        jobs=args.jobs,
        overwrite=args.overwrite,
        algorithm=args.algorithm,
    )


if __name__ == "__main__":
    main()
