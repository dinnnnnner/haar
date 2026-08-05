from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable

from .detector import (
    WHEEL_NAMES,
    WaveletShapeBlowoutDetector,
    WaveletShapeConfig,
    WheelFrame,
)
from .evidence_detector import EvidenceBlowoutDetector, EvidenceConfig


DEFAULT_CONVERTER_DIRS = (
    Path.home() / "py" / "wheel_cog_outputs",
    Path("/mnt/d/py/wheel_cog_outputs"),
)
WHEEL_COLUMNS = tuple(f"wheel{i}_corrected_rad_s" for i in range(4))
ALGORITHMS = ("hard", "evidence")
EVALUATION_SCHEMA_VERSION = 1


def is_locked(path: Path) -> bool:
    with path.open("rb") as handle:
        head = handle.read(256)
    return b"E-SafeNet" in head and b"LOCK" in head


def discover_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"input does not exist: {input_path}")
    return sorted(
        path
        for path in input_path.rglob("*.txt")
        if path.is_file()
        and not (path.name.startswith("#") and path.name.endswith("#"))
        and not path.name.endswith("~")
    )


def find_converter_dir(explicit: Path | None) -> Path:
    candidates = (explicit,) if explicit is not None else DEFAULT_CONVERTER_DIRS
    for directory in candidates:
        if directory is not None and (directory / "process_wheel_cog.py").is_file():
            return directory
    searched = ", ".join(str(path) for path in candidates if path is not None)
    raise FileNotFoundError(
        "cannot find process_wheel_cog.py; searched: " + searched
    )


def load_converter(converter_dir: Path) -> Callable[[Path, Path], dict[str, Any]]:
    module_path = converter_dir / "process_wheel_cog.py"
    spec = importlib.util.spec_from_file_location("robust_wheel_cog_converter", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load converter: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.process_file


def _float_or_none(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return value


def _new_detector(algorithm: str) -> tuple[object, dict[str, Any]]:
    if algorithm == "hard":
        cfg = WaveletShapeConfig()
        return WaveletShapeBlowoutDetector(cfg), asdict(cfg)
    if algorithm == "evidence":
        cfg = EvidenceConfig()
        return EvidenceBlowoutDetector(cfg), asdict(cfg)
    raise ValueError(f"unsupported algorithm: {algorithm}")


def evaluate_wheel_csv(
    csv_path: Path,
    algorithms: Iterable[str],
) -> list[dict[str, Any]]:
    detectors: dict[str, object] = {}
    configs: dict[str, dict[str, Any]] = {}
    stats: dict[str, dict[str, Any]] = {}
    for algorithm in algorithms:
        detector, config = _new_detector(algorithm)
        detectors[algorithm] = detector
        configs[algorithm] = config
        stats[algorithm] = {
            "previous_alarms": [False] * 4,
            "alarm_frames": [0] * 4,
            "alarm_events": [0] * 4,
            "confirmation_events": [0] * 4,
            "first_alarm_time_s": [None] * 4,
            "first_estimated_onset_time_s": [None] * 4,
        }

    frames = 0
    valid_speed_frames = 0
    first_time_s: float | None = None
    last_time_s: float | None = None
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        required = ("time_s", *WHEEL_COLUMNS)
        missing = [name for name in required if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"missing CSV columns in {csv_path}: {missing}")

        for row in reader:
            t_sec = float(row["time_s"])
            wheel_frame = WheelFrame.from_sequences(
                t_sec, [float(row[column]) for column in WHEEL_COLUMNS]
            )
            frames += 1
            if first_time_s is None:
                first_time_s = t_sec
            last_time_s = t_sec
            frame_is_valid = False
            for algorithm, detector in detectors.items():
                result = detector.push(wheel_frame)  # type: ignore[attr-defined]
                frame_is_valid = frame_is_valid or result.speed_valid
                current = result.blowout_alarms
                state = stats[algorithm]
                for wheel in range(4):
                    if current[wheel]:
                        state["alarm_frames"][wheel] += 1
                    if current[wheel] and not state["previous_alarms"][wheel]:
                        state["alarm_events"][wheel] += 1
                        if state["first_alarm_time_s"][wheel] is None:
                            state["first_alarm_time_s"][wheel] = t_sec
                            state["first_estimated_onset_time_s"][wheel] = (
                                _float_or_none(result.estimated_onset_times_s[wheel])
                            )
                    if result.shape_events[wheel]:
                        state["confirmation_events"][wheel] += 1
                state["previous_alarms"] = list(current)
            if frame_is_valid:
                valid_speed_frames += 1

    results: list[dict[str, Any]] = []
    for algorithm in algorithms:
        state = stats[algorithm]
        alarm_wheels = [
            WHEEL_NAMES[index]
            for index, count in enumerate(state["alarm_events"])
            if count
        ]
        confirmed_alarm_wheels = [
            WHEEL_NAMES[index]
            for index, count in enumerate(state["confirmation_events"])
            if count
        ]
        results.append(
            {
                "algorithm": algorithm,
                "frames": frames,
                "valid_speed_frames": valid_speed_frames,
                "duration_s": (
                    None
                    if first_time_s is None or last_time_s is None
                    else last_time_s - first_time_s
                ),
                "false_alarm": bool(alarm_wheels),
                "alarm_wheels": alarm_wheels,
                "alarm_events": sum(state["alarm_events"]),
                "alarm_frames": sum(state["alarm_frames"]),
                "confirmed_false_alarm": bool(confirmed_alarm_wheels),
                "confirmed_alarm_wheels": confirmed_alarm_wheels,
                "confirmation_events": sum(state["confirmation_events"]),
                "per_wheel": {
                    name: {
                        "alarm_events": state["alarm_events"][index],
                        "alarm_frames": state["alarm_frames"][index],
                        "confirmation_events": state["confirmation_events"][index],
                        "first_alarm_time_s": state["first_alarm_time_s"][index],
                        "first_estimated_onset_time_s": state[
                            "first_estimated_onset_time_s"
                        ][index],
                    }
                    for index, name in enumerate(WHEEL_NAMES)
                },
                "config": configs[algorithm],
            }
        )
    return results


def _case_relative_path(path: Path, input_path: Path) -> Path:
    if input_path.is_file():
        return Path(path.stem)
    relative = path.relative_to(input_path).with_suffix("")
    cleaned_parts = [
        re.sub(r"[^0-9A-Za-z_.-]+", "_", part).strip("._") or "case"
        for part in relative.parts
    ]
    return Path(*cleaned_parts)


def _flat_row(
    input_file: Path,
    case_dir: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "status": "ok",
        "input_file": str(input_file),
        "case_dir": str(case_dir),
        "algorithm": result["algorithm"],
        "frames": result["frames"],
        "valid_speed_frames": result["valid_speed_frames"],
        "duration_s": result["duration_s"],
        "false_alarm": result["false_alarm"],
        "alarm_wheels": ",".join(result["alarm_wheels"]),
        "alarm_events": result["alarm_events"],
        "alarm_frames": result["alarm_frames"],
        "confirmed_false_alarm": result["confirmed_false_alarm"],
        "confirmed_alarm_wheels": ",".join(result["confirmed_alarm_wheels"]),
        "confirmation_events": result["confirmation_events"],
    }
    for name in WHEEL_NAMES:
        wheel = result["per_wheel"][name]
        for field in (
            "alarm_events",
            "alarm_frames",
            "confirmation_events",
            "first_alarm_time_s",
            "first_estimated_onset_time_s",
        ):
            row[f"{name}_{field}"] = wheel[field]
    return row


def _write_batch_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "robust_evaluation_summary.json"
    csv_path = output_dir / "robust_evaluation_summary.csv"
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_batch(args: argparse.Namespace) -> list[dict[str, Any]]:
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    inputs = discover_inputs(input_path)
    if args.limit is not None:
        inputs = inputs[: args.limit]
    if not inputs:
        raise ValueError(f"no .txt inputs found under {input_path}")
    algorithms = ALGORITHMS if args.algorithm == "both" else (args.algorithm,)
    converter_dir = find_converter_dir(args.converter_dir)
    process_file = load_converter(converter_dir)
    print(f"converter: {converter_dir / 'process_wheel_cog.py'}", flush=True)
    print(f"inputs: {len(inputs)}", flush=True)

    rows: list[dict[str, Any]] = []
    for number, raw_path in enumerate(inputs, start=1):
        relative_case = _case_relative_path(raw_path, input_path)
        case_dir = output_dir / "cases" / relative_case
        print(f"[{number}/{len(inputs)}] {raw_path}", flush=True)
        if is_locked(raw_path):
            print("  skipped: E-SafeNet locked", flush=True)
            rows.append(
                {
                    "status": "locked",
                    "input_file": str(raw_path),
                    "case_dir": str(case_dir),
                    "error": "E-SafeNet locked file; use the individual plaintext exports",
                }
            )
            _write_batch_outputs(rows, output_dir)
            continue

        case_summary_path = case_dir / "wavelet_robust_summary.json"
        try:
            if case_summary_path.exists() and not args.overwrite_evaluation:
                cached = json.loads(case_summary_path.read_text(encoding="utf-8"))
                results = cached["results"]
                wanted = {result["algorithm"] for result in results}
                if (
                    cached.get("schema_version") != EVALUATION_SCHEMA_VERSION
                    or not set(algorithms).issubset(wanted)
                ):
                    results = []
            else:
                results = []

            if not results:
                wheel_csv = case_dir / "wheel_speed_raw_vs_corrected.csv"
                if args.overwrite_conversion or not wheel_csv.is_file():
                    print("  converting cog timestamps...", flush=True)
                    conversion = process_file(raw_path, case_dir)
                    wheel_csv = Path(conversion["wheel_speed_csv"])
                else:
                    print("  using cached wheel-speed CSV", flush=True)
                print("  evaluating " + ", ".join(algorithms) + "...", flush=True)
                results = evaluate_wheel_csv(wheel_csv, algorithms)
                case_summary_path.write_text(
                    json.dumps(
                        {
                            "schema_version": EVALUATION_SCHEMA_VERSION,
                            "input_file": str(raw_path),
                            "wheel_speed_csv": str(wheel_csv),
                            "results": results,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            else:
                print("  using cached evaluation", flush=True)
            for result in results:
                if result["algorithm"] in algorithms:
                    rows.append(_flat_row(raw_path, case_dir, result))
        except Exception as exc:
            print(f"  error: {exc!r}", flush=True)
            rows.append(
                {
                    "status": "error",
                    "input_file": str(raw_path),
                    "case_dir": str(case_dir),
                    "error": repr(exc),
                }
            )
        _write_batch_outputs(rows, output_dir)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert RobustData cog timestamps and evaluate normal-road false alarms."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--algorithm", choices=("hard", "evidence", "both"), default="both"
    )
    parser.add_argument(
        "--converter-dir",
        type=Path,
        help=(
            "directory containing process_wheel_cog.py; defaults to ~/py, then D:/py"
        ),
    )
    parser.add_argument(
        "--limit", type=int, help="process only the first N discovered files"
    )
    parser.add_argument("--overwrite-conversion", action="store_true")
    parser.add_argument("--overwrite-evaluation", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def main() -> None:
    args = parse_args()
    rows = run_batch(args)
    ok = sum(row.get("status") == "ok" for row in rows)
    locked = sum(row.get("status") == "locked" for row in rows)
    errors = sum(row.get("status") == "error" for row in rows)
    false_alarms = sum(
        row.get("status") == "ok" and bool(row.get("false_alarm")) for row in rows
    )
    print(
        f"done: ok_rows={ok}, locked_files={locked}, errors={errors}, "
        f"false_alarm_rows={false_alarms}",
        flush=True,
    )
    print(args.output_dir.resolve() / "robust_evaluation_summary.csv", flush=True)


if __name__ == "__main__":
    main()
