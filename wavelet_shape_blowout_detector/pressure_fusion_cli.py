from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from .pressure_fusion_detector import (
    WHEEL_NAMES,
    PressureFusionBlowoutDetector,
    PressureFusionFrame,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pressure-diagonal fusion blowout detection on a CSV."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-column", default="time_s")
    parser.add_argument(
        "--wheel-columns",
        nargs=4,
        default=[f"wheel{i}_corrected_rad_s" for i in range(4)],
        metavar=("FL", "FR", "RL", "RR"),
    )
    for name in WHEEL_NAMES:
        parser.add_argument(f"--pressure-{name.lower()}-column")
    return parser.parse_args()


def _optional_bool(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in {"", "none", "null", "nan"}:
        return None
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid pressure boolean: {value!r}")


def main() -> None:
    args = _parse_args()
    pressure_columns = [
        getattr(args, f"pressure_{name.lower()}_column") for name in WHEEL_NAMES
    ]
    configured = tuple(index for index, value in enumerate(pressure_columns) if value)
    if configured not in ((0, 3), (1, 2)):
        raise SystemExit(
            "pressure columns must be exactly FL+RR or exactly FR+RL"
        )

    detector = PressureFusionBlowoutDetector()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "time_s",
        "sensor_diagonal",
        "speed_diagonal",
        "speed_valid",
        "speed_detection_available",
        "diagonal_gain",
        "diagonal_edge",
    ]
    for name in WHEEL_NAMES:
        fieldnames.extend(
            (
                f"{name}_gain",
                f"{name}_edge",
                f"{name}_candidate",
                f"{name}_new_blowout",
                f"{name}_blowout_alarm",
                f"{name}_alarm_source",
                f"{name}_estimated_onset_s",
            )
        )

    with args.input.open(newline="", encoding="utf-8-sig") as source, args.output.open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError("input CSV has no header")
        required = [args.time_column, *args.wheel_columns]
        required.extend(value for value in pressure_columns if value)
        missing = [value for value in required if value not in reader.fieldnames]
        if missing:
            raise ValueError(f"input CSV is missing columns: {missing}")
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            pressure = [
                None if column is None else _optional_bool(row[column])
                for column in pressure_columns
            ]
            result = detector.push(
                PressureFusionFrame.from_sequences(
                    float(row[args.time_column]),
                    [float(row[column]) for column in args.wheel_columns],
                    pressure,
                )
            )
            output: dict[str, object] = {
                "time_s": result.t_sec,
                "sensor_diagonal": "+".join(result.sensor_diagonal),
                "speed_diagonal": "+".join(result.speed_diagonal),
                "speed_valid": int(result.speed_valid),
                "speed_detection_available": int(result.speed_detection_available),
                "diagonal_gain": result.diagonal_gain,
                "diagonal_edge": result.diagonal_edge,
            }
            for index, name in enumerate(WHEEL_NAMES):
                onset = result.estimated_onset_times_s[index]
                output.update(
                    {
                        f"{name}_gain": result.individual_gains[index],
                        f"{name}_edge": result.individual_edges[index],
                        f"{name}_candidate": int(result.candidates[index]),
                        f"{name}_new_blowout": int(result.new_blowouts[index]),
                        f"{name}_blowout_alarm": int(result.blowout_alarms[index]),
                        f"{name}_alarm_source": result.alarm_sources[index],
                        f"{name}_estimated_onset_s": (
                            "" if onset is None or not math.isfinite(onset) else onset
                        ),
                    }
                )
            writer.writerow(output)


if __name__ == "__main__":
    main()
