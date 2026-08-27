from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_0818_display import learn_phase_factors_from_file
from evaluate_0820_quant import SCHEMA_VERSION, evaluate_case


WORKSPACE_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = WORKSPACE_ROOT / "0826"
DEFAULT_OUTPUT = WORKSPACE_ROOT / "0826_quant_evaluation" / "summary.json"


def run(input_dir: Path, output: Path) -> dict[str, object]:
    input_dir = input_dir.resolve()
    paths = sorted(input_dir.rglob("*.txt"))
    if not paths:
        raise ValueError(f"目录内没有 0826 txt 数据：{input_dir}")

    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "0826",
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
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="流式评价 0826 FL 爆胎原始记录")
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
