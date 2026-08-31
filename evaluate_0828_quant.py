from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import median

from build_0818_display import iter_raw_frames, learn_phase_factors_from_file
from evaluate_0820_quant import SCHEMA_VERSION, evaluate_case


WORKSPACE_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = WORKSPACE_ROOT / "0828" / "20260828爆胎测试"
DEFAULT_OUTPUT = WORKSPACE_ROOT / "0828_quant_evaluation" / "summary.json"


def _raw_start_seconds(path: Path) -> float:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        header = "".join(handle.readline() for _ in range(10))
    match = re.search(r"Start time\s*:\s*\d+:(\d+):(\d+)", header)
    if match is None:
        raise ValueError(f"原始记录缺少 Start time：{path}")
    minute, second = (int(value) for value in match.groups())
    return minute * 60.0 + second


def _pressure_start_seconds(path: Path) -> float:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            match = re.match(r"\s*(\d+)-(\d+)-(\d+)\s*,", line)
            if match is None:
                continue
            minute, second, millisecond = (int(value) for value in match.groups())
            return minute * 60.0 + second + millisecond / 1000.0
    raise ValueError(f"胎压记录缺少时间数据：{path}")


def _repair_stuck_high_event_times(
    input_dir: Path, cases: list[dict[str, object]]
) -> None:
    """Recover a stuck-high event time from the second logger's clock offset."""

    offsets: list[float] = []
    for case in cases:
        event = case.get("signal_event_time_s")
        if not isinstance(event, (int, float)) or event <= 0.0:
            continue
        raw_path = input_dir / str(case["input_file"])
        pressure_path = raw_path.with_suffix(".csv")
        if not pressure_path.is_file():
            continue
        offsets.append(
            (
                _raw_start_seconds(raw_path)
                + float(event)
                - _pressure_start_seconds(pressure_path)
            )
            % 3600.0
        )
    if not offsets:
        return
    clock_offset_s = median(offsets)

    for case in cases:
        if case.get("signal_event_time_s") != 0.0:
            continue
        raw_path = input_dir / str(case["input_file"])
        pressure_path = raw_path.with_suffix(".csv")
        if not pressure_path.is_file():
            continue
        if not all(frame.blowout_signal for frame in iter_raw_frames(raw_path)):
            continue
        raw_to_pressure_s = (
            _raw_start_seconds(raw_path) - _pressure_start_seconds(pressure_path)
        ) % 3600.0
        repaired = (clock_offset_s - raw_to_pressure_s) % 3600.0
        duration = float(case["duration_s"])
        if repaired <= duration:
            case["signal_event_time_s"] = repaired
            case["signal_event_source"] = "pressure_clock_recovered_from_stuck_high"
            case["pressure_clock_offset_s"] = clock_offset_s


def run(input_dir: Path, output: Path) -> dict[str, object]:
    input_dir = input_dir.resolve()
    paths = sorted(input_dir.rglob("*.txt"))
    if not paths:
        raise ValueError(f"目录内没有 0828 txt 数据：{input_dir}")

    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "0828",
        "input_dir": str(input_dir),
        "algorithm": "quant",
        "cases": [],
    }
    cases = summary["cases"]
    assert isinstance(cases, list)
    output.parent.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(paths, start=1):
        relative_path = path.relative_to(input_dir)
        group = relative_path.parent.as_posix()
        case_id = f"{group} / {path.stem}"
        print(f"[{index}/{len(paths)}] {case_id}", flush=True)
        print("  学习 48 齿相位校正...", flush=True)
        phase_factors = learn_phase_factors_from_file(path)
        print("  回放 quant...", flush=True)
        case = evaluate_case(path, phase_factors)
        case["case"] = case_id
        case["input_file"] = relative_path.as_posix()
        cases.append(case)
        output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    _repair_stuck_high_event_times(input_dir, cases)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="流式评价 0828 FL 爆胎原始记录")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run(args.input_dir, args.output)
    cases = summary["cases"]
    assert isinstance(cases, list)
    correct = sum(
        case["quant_first_alarms_s"]["FL"] is not None
        and all(
            case["quant_first_alarms_s"][wheel] is None
            for wheel in ("FR", "RL", "RR")
        )
        for case in cases
    )
    print(f"完成：{len(cases)} 条，FL 正确检出 {correct}/{len(cases)}", flush=True)
    print(args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
