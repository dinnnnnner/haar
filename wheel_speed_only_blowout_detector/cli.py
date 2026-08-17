from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from .detector import WHEEL_NAMES, WheelSpeedBlowoutDetector, WheelSpeedFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect blowouts from four wheel-speed channels only."
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(
            f"input CSV does not exist: {args.input.resolve()}\n"
            "Pass the path to an existing CSV after --input; "
            "wheel_speed.csv in the documentation is only an example name."
        )
    detector = WheelSpeedBlowoutDetector()
    fields = ["time_s", "speed_valid", "warmed_up"]
    for name in WHEEL_NAMES:
        fields.extend(
            (
                f"{name}_gain",
                f"{name}_edge",
                f"{name}_diagonal_gain",
                f"{name}_diagonal_edge",
                f"{name}_state",
                f"{name}_candidate",
                f"{name}_new_blowout",
                f"{name}_blowout_alarm",
                f"{name}_estimated_onset_s",
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(newline="", encoding="utf-8-sig") as source, args.output.open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError("input CSV has no header")
        required = [args.time_column, *args.wheel_columns]
        missing = [column for column in required if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"input CSV is missing columns: {missing}")
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            result = detector.push(
                WheelSpeedFrame.from_sequences(
                    float(row[args.time_column]),
                    [float(row[column]) for column in args.wheel_columns],
                )
            )
            output: dict[str, object] = {
                "time_s": result.t_sec,
                "speed_valid": int(result.speed_valid),
                "warmed_up": int(result.warmed_up),
            }
            for wheel, name in enumerate(WHEEL_NAMES):
                onset = result.estimated_onset_times_s[wheel]
                output.update(
                    {
                        f"{name}_gain": result.individual_gains[wheel],
                        f"{name}_edge": result.individual_edges[wheel],
                        f"{name}_diagonal_gain": result.diagonal_gains[wheel],
                        f"{name}_diagonal_edge": result.diagonal_edges[wheel],
                        f"{name}_state": result.states[wheel],
                        f"{name}_candidate": int(result.candidates[wheel]),
                        f"{name}_new_blowout": int(result.new_blowouts[wheel]),
                        f"{name}_blowout_alarm": int(result.blowout_alarms[wheel]),
                        f"{name}_estimated_onset_s": (
                            "" if onset is None or not math.isfinite(onset) else onset
                        ),
                    }
                )
            writer.writerow(output)


if __name__ == "__main__":
    main()
