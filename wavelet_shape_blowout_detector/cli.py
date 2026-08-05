from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from .detector import (
    REFERENCE_MODES,
    WHEEL_NAMES,
    WaveletShapeBlowoutDetector,
    WaveletShapeConfig,
    WheelFrame,
)


DEFAULT_WHEEL_COLUMNS = tuple(
    f"wheel{index}_corrected_rad_s" for index in range(4)
)


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"cannot parse normal signal value {value!r}")


def process_csv(
    input_path: Path,
    output_path: Path,
    cfg: WaveletShapeConfig,
    time_column: str,
    wheel_columns: tuple[str, str, str, str],
    normal_columns: tuple[str | None, str | None, str | None, str | None],
) -> None:
    detector = WaveletShapeBlowoutDetector(cfg)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {input_path}")
        requested_columns = [time_column, *wheel_columns]
        requested_columns.extend(column for column in normal_columns if column)
        missing = [
            column
            for column in requested_columns
            if column not in reader.fieldnames
        ]
        if missing:
            raise ValueError(f"missing CSV columns: {missing}")

        fieldnames = [time_column, *wheel_columns, "speed_valid"]
        for name in WHEEL_NAMES:
            fieldnames.extend(
                [
                    f"{name}_target_peer_ratio",
                    f"{name}_normal_signal",
                    f"{name}_reference_source",
                    f"{name}_normalized_gain",
                    f"{name}_haar_coefficient",
                    f"{name}_state",
                    f"{name}_shape_event",
                    f"{name}_new_blowout",
                    f"{name}_blowout",
                    f"{name}_estimated_onset_index",
                    f"{name}_estimated_onset_time_s",
                    f"{name}_rise_coefficient",
                    f"{name}_fall_coefficient",
                    f"{name}_steady_gain",
                    f"{name}_steady_tail_gain",
                ]
            )

        with output_path.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                normal_signals = [
                    None if column is None else _parse_bool(row[column])
                    for column in normal_columns
                ]
                result = detector.push(
                    WheelFrame.from_sequences(
                        float(row[time_column]),
                        [float(row[column]) for column in wheel_columns],
                        normal_signals,
                    )
                )
                output: dict[str, object] = {
                    time_column: result.t_sec,
                    **{
                        column: result.wheels[index]
                        for index, column in enumerate(wheel_columns)
                    },
                    "speed_valid": int(result.speed_valid),
                }
                for index, name in enumerate(WHEEL_NAMES):
                    values = {
                        f"{name}_target_peer_ratio": result.target_peer_ratios[index],
                        f"{name}_normal_signal": (
                            "" if result.normal_signals[index] is None
                            else int(result.normal_signals[index])
                        ),
                        f"{name}_reference_source": result.reference_sources[index],
                        f"{name}_normalized_gain": result.normalized_gains[index],
                        f"{name}_haar_coefficient": result.haar_coefficients[index],
                        f"{name}_state": result.states[index],
                        f"{name}_shape_event": int(result.shape_events[index]),
                        f"{name}_new_blowout": int(result.new_blowouts[index]),
                        f"{name}_blowout": int(result.blowout_alarms[index]),
                        f"{name}_estimated_onset_index": result.estimated_onset_indices[index],
                        f"{name}_estimated_onset_time_s": result.estimated_onset_times_s[index],
                        f"{name}_rise_coefficient": result.rise_coefficients[index],
                        f"{name}_fall_coefficient": result.fall_coefficients[index],
                        f"{name}_steady_gain": result.steady_gains[index],
                        f"{name}_steady_tail_gain": result.steady_tail_gains[index],
                    }
                    output.update(
                        {
                            key: "" if isinstance(value, float) and math.isnan(value) else value
                            for key, value in values.items()
                        }
                    )
                writer.writerow(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect blowouts from a rise-fall-persistent wheel-speed shape."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-column", default="time_s")
    parser.add_argument(
        "--wheel-columns",
        nargs=4,
        default=DEFAULT_WHEEL_COLUMNS,
        metavar=("FL", "FR", "RL", "RR"),
    )
    parser.add_argument("--sample-rate-hz", type=float, default=100.0)
    parser.add_argument(
        "--reference-mode", choices=REFERENCE_MODES, default="opposite_diagonal"
    )
    parser.add_argument("--min-avg-speed", type=float, default=20.0)
    parser.add_argument("--min-rise-coefficient", type=float, default=0.0055)
    parser.add_argument("--min-fall-coefficient", type=float, default=0.0033)
    parser.add_argument("--min-steady-gain", type=float, default=0.0060)
    for name in WHEEL_NAMES:
        parser.add_argument(f"--normal-{name.lower()}-column")
    parser.add_argument(
        "--target-wheels",
        nargs="+",
        choices=WHEEL_NAMES,
        default=WHEEL_NAMES,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = tuple(WHEEL_NAMES.index(name) for name in args.target_wheels)
    cfg = WaveletShapeConfig(
        sample_rate_hz=args.sample_rate_hz,
        reference_mode=args.reference_mode,
        min_avg_speed=args.min_avg_speed,
        min_rise_coefficient=args.min_rise_coefficient,
        min_fall_coefficient=args.min_fall_coefficient,
        min_steady_gain=args.min_steady_gain,
        target_wheels=targets,
    )
    process_csv(
        args.input,
        args.output,
        cfg,
        args.time_column,
        tuple(args.wheel_columns),  # type: ignore[arg-type]
        tuple(
            getattr(args, f"normal_{name.lower()}_column")
            for name in WHEEL_NAMES
        ),  # type: ignore[arg-type]
    )


if __name__ == "__main__":
    main()
