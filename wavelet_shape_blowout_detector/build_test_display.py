from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

from .detector import REFERENCE_MODES, WHEEL_NAMES, WaveletShapeConfig
from .display import analyze_csv, write_display_html


def build_batch(
    manifest_path: Path,
    output_dir: Path,
    cfg: WaveletShapeConfig,
    window_before_s: float = 5.0,
    window_after_s: float = 5.0,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            event_id = row["source_event_id"]
            event_time = float(row["event_time_s"])
            source_path = Path(row["source_wheel_csv"])
            data = analyze_csv(
                source_path,
                cfg=cfg,
                start_time_s=event_time - window_before_s,
                end_time_s=event_time + window_after_s,
            )
            output_path = output_dir / f"{event_id}.html"
            write_display_html(
                data,
                output_path,
                cfg,
                event_time_s=event_time,
                title=f"{event_id} 小波形态检测 — {source_path.parent.name}",
            )
            alarm_wheels = [
                name
                for name, time in zip(WHEEL_NAMES, data.first_alarm_times)
                if time is not None
            ]
            rr_alarm = data.first_alarm_times[3]
            rows.append(
                {
                    "event_id": event_id,
                    "source": source_path.parent.name,
                    "html": output_path.name,
                    "alarm_wheels": ",".join(alarm_wheels) or "none",
                    "rr_delay_s": None if rr_alarm is None else rr_alarm - event_time,
                }
            )
    _write_index(output_dir / "index.html", rows)
    return rows


def _write_index(path: Path, rows: list[dict[str, object]]) -> None:
    table_rows = []
    for row in rows:
        delay = row["rr_delay_s"]
        delay_text = "—" if delay is None else f"{float(delay):.3f}"
        table_rows.append(
            "<tr>"
            f"<td><a href='{html.escape(str(row['html']))}'>{html.escape(str(row['event_id']))}</a></td>"
            f"<td>{html.escape(str(row['source']))}</td>"
            f"<td>{html.escape(str(row['alarm_wheels']))}</td>"
            f"<td>{delay_text}</td>"
            "</tr>"
        )
    content = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>小波爆胎测试 Display</title>
<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;color:#172033}table{border-collapse:collapse;width:100%}th,td{padding:10px 12px;border-bottom:1px solid #dbe2ea;text-align:left}th{background:#f4f7fa}a{color:#185abd;text-decoration:none}a:hover{text-decoration:underline}</style>
</head><body><h1>小波形态爆胎检测：8 条原始样本</h1>
<p>点击事件查看四轮轮速、相对增益、Haar 系数和逐轮报警。</p>
<table><thead><tr><th>事件</th><th>样本</th><th>报警轮位</th><th>RR 延迟 / s</th></tr></thead><tbody>
""" + "\n".join(table_rows) + """
</tbody></table></body></html>
"""
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the eight-case wavelet display.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("display"))
    parser.add_argument("--window-before", type=float, default=5.0)
    parser.add_argument("--window-after", type=float, default=5.0)
    parser.add_argument(
        "--reference-mode", choices=REFERENCE_MODES, default="opposite_diagonal"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = WaveletShapeConfig(reference_mode=args.reference_mode)
    rows = build_batch(
        args.manifest,
        args.output_dir,
        cfg,
        window_before_s=args.window_before,
        window_after_s=args.window_after,
    )
    print(f"wrote {len(rows)} cases to {args.output_dir}")


if __name__ == "__main__":
    main()
