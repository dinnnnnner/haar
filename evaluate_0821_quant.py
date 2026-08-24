from __future__ import annotations

import argparse
from pathlib import Path

from evaluate_0820_quant import run


WORKSPACE_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = WORKSPACE_ROOT / "0821"
DEFAULT_OUTPUT = WORKSPACE_ROOT / "0821_quant_evaluation" / "summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="流式评价 0821 原始记录的 quant 结果")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run(args.input_dir, args.output, dataset_name="0821")
    correct = sum(
        case["quant_first_alarms_s"]["RR"] is not None
        and all(
            case["quant_first_alarms_s"][wheel] is None
            for wheel in ("FL", "FR", "RL")
        )
        for case in summary["cases"]
    )
    print(
        f"完成：{len(summary['cases'])} 条，RR 正确检出 "
        f"{correct}/{len(summary['cases'])}",
        flush=True,
    )
    print(args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
