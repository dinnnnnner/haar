from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from build_0818_display import (
    DEFAULT_SAMPLE_TIME_S,
    WHEEL_NAMES,
    iter_corrected_raw_speed_rows,
    learn_phase_factors_from_file,
)
from quant_wheel_blowout_detector import QuantBlowoutDetector, QuantFrame


WORKSPACE_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = WORKSPACE_ROOT / "0820"
DEFAULT_OUTPUT = WORKSPACE_ROOT / "0820_quant_evaluation" / "summary.json"
SCHEMA_VERSION = 1


def evaluate_case(
    input_path: Path, phase_factors: Sequence[Sequence[float]]
) -> dict[str, Any]:
    detector = QuantBlowoutDetector()
    candidate_starts: list[float | None] = [None] * 4
    first_alarms: list[float | None] = [None] * 4
    intervals: list[dict[str, Any]] = []
    signal_run_start: int | None = None
    signal_event_time_s: float | None = None
    frames = 0
    last_time_s = 0.0
    last_alarms = (False, False, False, False)

    for index, (time_s, speeds, signal) in enumerate(
        iter_corrected_raw_speed_rows(input_path, phase_factors)
    ):
        result = detector.push(QuantFrame.from_sequences(time_s, speeds))
        frames = index + 1
        last_time_s = time_s
        last_alarms = result.blowout_alarms

        if signal:
            if signal_run_start is None:
                signal_run_start = index
            if signal_event_time_s is None and index - signal_run_start + 1 >= 20:
                signal_event_time_s = signal_run_start * DEFAULT_SAMPLE_TIME_S
        else:
            signal_run_start = None

        for wheel in range(4):
            if result.new_blowouts[wheel] and first_alarms[wheel] is None:
                first_alarms[wheel] = time_s
            active = result.states[wheel] == "candidate"
            if active and candidate_starts[wheel] is None:
                candidate_starts[wheel] = time_s
            elif not active and candidate_starts[wheel] is not None:
                intervals.append(
                    {
                        "wheel": wheel,
                        "start_s": candidate_starts[wheel],
                        "end_s": time_s,
                        "confirmed": bool(result.blowout_alarms[wheel]),
                    }
                )
                candidate_starts[wheel] = None

    if not frames:
        raise ValueError(f"没有数据帧：{input_path}")
    for wheel, start_s in enumerate(candidate_starts):
        if start_s is not None:
            intervals.append(
                {
                    "wheel": wheel,
                    "start_s": start_s,
                    "end_s": last_time_s,
                    "confirmed": bool(last_alarms[wheel]),
                }
            )

    return {
        "case": input_path.stem,
        "input_file": input_path.name,
        "input_size": input_path.stat().st_size,
        "frames": frames,
        "duration_s": last_time_s,
        "signal_event_time_s": signal_event_time_s,
        "quant_first_alarms_s": dict(zip(WHEEL_NAMES, first_alarms)),
        "candidate_intervals": sorted(
            intervals, key=lambda item: (item["start_s"], item["wheel"])
        ),
        "phase_factors": [list(row) for row in phase_factors],
    }


def run(input_dir: Path, output: Path) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    paths = sorted(input_dir.glob("*.txt"))
    if not paths:
        raise ValueError(f"目录内没有 0820 txt 数据：{input_dir}")

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "input_dir": str(input_dir),
        "algorithm": "quant",
        "cases": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] {path.name}", flush=True)
        print("  学习 48 齿相位校正...", flush=True)
        phase_factors = learn_phase_factors_from_file(path)
        print("  回放 quant...", flush=True)
        case = evaluate_case(path, phase_factors)
        summary["cases"].append(case)
        output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        alarms = [
            f"{wheel} {value:.2f}s"
            for wheel, value in case["quant_first_alarms_s"].items()
            if value is not None
        ]
        print(
            f"  {case['frames']:,} 帧，quant "
            + ("、".join(alarms) if alarms else "未报警"),
            flush=True,
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="流式评价 0820 原始记录的 quant 结果")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run(args.input_dir, args.output)
    alarms = sum(
        any(value is not None for value in case["quant_first_alarms_s"].values())
        for case in summary["cases"]
    )
    print(
        f"完成：{len(summary['cases'])} 条，报警 {alarms}/{len(summary['cases'])}",
        flush=True,
    )
    print(args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
