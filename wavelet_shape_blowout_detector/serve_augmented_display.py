from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

if __package__:
    from .detector import REFERENCE_MODES, WHEEL_NAMES, WaveletShapeConfig
    from .display import PLOT_CONFIG, analyze_csv, build_figure
    from .evaluate_augmented import CLASS_LABELS, classify_result
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from wavelet_shape_blowout_detector.detector import (  # noqa: E402
        REFERENCE_MODES,
        WHEEL_NAMES,
        WaveletShapeConfig,
    )
    from wavelet_shape_blowout_detector.display import (  # noqa: E402
        PLOT_CONFIG,
        analyze_csv,
        build_figure,
    )
    from wavelet_shape_blowout_detector.evaluate_augmented import (  # noqa: E402
        CLASS_LABELS,
        classify_result,
    )


DEFAULT_DATASET_DIR = (
    Path(__file__).resolve().parents[1] / "augmented_event_dataset_v2"
)
DEFAULT_EVALUATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "display_488"
    / "v2_current_evaluation.csv"
)
DEFAULT_EVIDENCE_EVALUATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "display_488"
    / "v2_evidence_evaluation.csv"
)


def _optional_float(value: object) -> float | None:
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "na"}:
        return None
    number = float(text)
    return number if math.isfinite(number) else None


class ViewerState:
    def __init__(
        self,
        dataset_dir: Path,
        cfg: WaveletShapeConfig,
        window_before_s: float = 5.0,
        window_after_s: float = 5.0,
        cache_size: int = 12,
        evaluation_path: Path | None = None,
        evidence_evaluation_path: Path | None = None,
    ) -> None:
        self.dataset_dir = dataset_dir.resolve()
        self.cfg = cfg
        self.window_before_s = window_before_s
        self.window_after_s = window_after_s
        manifest_path = self.dataset_dir / "manifest.csv"
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError(f"manifest contains no samples: {manifest_path}")
        config_path = self.dataset_dir / "dataset_config.json"
        self.dataset_config = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.exists()
            else {}
        )
        self.evaluations = {
            "hard": self._load_evaluations(evaluation_path),
            "evidence": self._load_evaluations(evidence_evaluation_path),
        }
        self.rows_by_id = {row["sample_id"]: row for row in self.rows}
        self.sample_ids = [row["sample_id"] for row in self.rows]
        self.positions = {
            sample_id: index for index, sample_id in enumerate(self.sample_ids)
        }
        self._cached_sample = lru_cache(maxsize=cache_size)(self._render_sample)

    @staticmethod
    def _load_evaluations(path: Path | None) -> dict[str, dict[str, str]]:
        if path is None or not path.exists():
            return {}
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return {item["sample_id"]: item for item in csv.DictReader(handle)}

    def render_index(self, algorithm: str = "hard") -> str:
        if algorithm not in {"hard", "evidence", "compare"}:
            raise ValueError("algorithm must be hard, evidence, or compare")
        event_count = sum(row["sample_type"] == "event" for row in self.rows)
        normal_count = len(self.rows) - event_count
        noise_model = self.dataset_config.get("noise_model", "unknown")
        selected_algorithms = (
            ("hard", "evidence") if algorithm == "compare" else (algorithm,)
        )
        algorithm_names = {"hard": "当前版", "evidence": "Evidence"}
        classification_counts = {
            selected: {
                code: sum(
                    self.evaluations[selected]
                    .get(row["sample_id"], {})
                    .get("classification")
                    == code
                    for row in self.rows
                )
                for code in CLASS_LABELS
            }
            for selected in selected_algorithms
        }
        table_rows = []
        for row in self.rows:
            sample_id = row["sample_id"]
            event_time = _optional_float(row.get("event_time_in_sample_s"))
            event_text = "—" if event_time is None else f"{event_time:.3f}"
            evaluations = [
                (selected, self.evaluations[selected].get(sample_id, {}))
                for selected in selected_algorithms
            ]
            statuses = [
                evaluation.get("classification", "UNASSESSED")
                for _, evaluation in evaluations
            ]

            def joined_value(key: str, formatter=None) -> str:
                parts = []
                for selected, evaluation in evaluations:
                    raw = evaluation.get(key, "")
                    value = formatter(raw) if formatter else (raw or "—")
                    prefix = f"{algorithm_names[selected]}：" if algorithm == "compare" else ""
                    parts.append(f"{prefix}{value}")
                return "<br>".join(html.escape(str(part)) for part in parts)

            classification_parts = []
            for selected, evaluation in evaluations:
                classification = evaluation.get("classification", "UNASSESSED")
                label = evaluation.get("classification_label", "未评估")
                prefix = f"{algorithm_names[selected]} " if algorithm == "compare" else ""
                classification_parts.append(
                    f"<span class='badge status-{html.escape(classification)}'>"
                    f"{html.escape(prefix + label)}</span>"
                )
            classification_html = "<br>".join(classification_parts)
            delay_text = joined_value(
                "delay_s",
                lambda value: (
                    "—"
                    if _optional_float(value) is None
                    else f"{_optional_float(value):+.3f}"
                ),
            )
            alarm_wheels = joined_value("alarm_wheels")
            false_wheels = joined_value("false_alarm_wheels")
            table_rows.append(
                "<tr "
                f"data-type='{html.escape(row['sample_type'])}' "
                f"data-source='{html.escape(row['source_event_id'])}' "
                f"data-status='{html.escape(' '.join(statuses))}'>"
                f"<td><a href='/sample/{quote(sample_id)}?algorithm={quote(algorithm)}'>{html.escape(sample_id)}</a></td>"
                f"<td>{classification_html}</td>"
                f"<td>{html.escape(row['sample_type'])}</td>"
                f"<td>{html.escape(row['source_event_id'])}</td>"
                f"<td>{alarm_wheels}</td>"
                f"<td>{false_wheels}</td>"
                f"<td>{delay_text}</td>"
                f"<td>{event_text}</td>"
                f"<td>{html.escape(row.get('target_wheels', '') or '—')}</td>"
                f"<td>{html.escape(row.get('noise_gain', '') or '—')}</td>"
                f"<td>{html.escape(row.get('dropout_samples', '') or '0')}</td>"
                "</tr>"
            )
        tabs = "".join(
            f"<a class='algorithm-tab {'active' if value == algorithm else ''}' "
            f"href='/?algorithm={value}'>{label}</a>"
            for value, label in (
                ("hard", "当前版"),
                ("evidence", "Evidence"),
                ("compare", "对比"),
            )
        )
        summary_parts = []
        for selected in selected_algorithms:
            prefix = f"{algorithm_names[selected]}：" if algorithm == "compare" else ""
            summary_parts.extend(
                f'<span class="badge status-{code}">{prefix}{label} {classification_counts[selected][code]}</span>'
                for code, label in CLASS_LABELS.items()
                if classification_counts[selected][code]
            )
        return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>小波 488 测试 Display</title>
<style>{_STYLE}</style></head><body>
<main><h1>小波形态爆胎检测：488 条增强测试</h1>
<p class="muted">{len(self.rows)} 条样本：{event_count} 条事件，{normal_count} 条正常。增强模型：{html.escape(str(noise_model))}。先选择算法，再点击样本查看结果。</p>
<div class="algorithm-tabs algorithm-tabs-index">{tabs}</div>
<div class="summary">{''.join(summary_parts)}</div>
<div class="controls"><input id="search" placeholder="搜索样本、分类、轮位、E01…"><select id="kind"><option value="">全部类型</option><option value="event">event</option><option value="normal">normal</option></select><select id="source"><option value="">全部来源</option>{''.join(f'<option>{name}</option>' for name in sorted(set(row['source_event_id'] for row in self.rows)))}</select><select id="status"><option value="">全部分类</option>{''.join(f'<option value="{code}">{label}</option>' for code, label in CLASS_LABELS.items())}</select><span id="count"></span></div>
<table><thead><tr><th>sample_id</th><th>分类</th><th>类型</th><th>来源</th><th>报警轮</th><th>误报轮</th><th>延迟/s</th><th>事件时刻/s</th><th>目标轮</th><th>噪声</th><th>丢帧</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
</main><script>{_FILTER_SCRIPT}</script></body></html>"""

    def render_sample(self, sample_id: str, algorithm: str = "hard") -> str:
        if algorithm not in {"hard", "evidence", "compare"}:
            raise ValueError("algorithm must be hard, evidence, or compare")
        return self._cached_sample(sample_id, algorithm)

    def _render_sample(self, sample_id: str, algorithm: str) -> str:
        row = self.rows_by_id.get(sample_id)
        if row is None:
            raise KeyError(sample_id)
        event_time = _optional_float(row.get("event_time_in_sample_s"))
        if event_time is None:
            start_time = None
            end_time = None
        else:
            start_time = event_time - self.window_before_s
            end_time = event_time + self.window_after_s
        sample_path = self.dataset_dir / row["sample_file"]
        position = self.positions[sample_id]
        query = f"?algorithm={quote(algorithm)}"
        previous_link = (
            ""
            if position == 0
            else f"<a href='/sample/{quote(self.sample_ids[position - 1])}{query}'>← 上一条</a>"
        )
        next_link = (
            ""
            if position + 1 == len(self.sample_ids)
            else f"<a href='/sample/{quote(self.sample_ids[position + 1])}{query}'>下一条 →</a>"
        )
        algorithm_names = {"hard": "当前版", "evidence": "Evidence"}
        selected_algorithms = (
            ("hard", "evidence") if algorithm == "compare" else (algorithm,)
        )
        panels = []
        for panel_index, selected in enumerate(selected_algorithms):
            data = analyze_csv(
                sample_path,
                cfg=self.cfg,
                start_time_s=start_time,
                end_time_s=end_time,
                algorithm=selected,
            )
            evaluation = classify_result(row, data)
            figure = build_figure(
                data,
                self.cfg,
                event_time_s=event_time,
                title=f"{sample_id} — {algorithm_names[selected]}",
                false_alarm_intervals=evaluation["false_alarm_intervals"],
            )
            plot_html = figure.to_html(
                full_html=False,
                include_plotlyjs="cdn" if panel_index == 0 else False,
                config=PLOT_CONFIG,
            )
            alarm_parts = []
            for name, alarm_time in zip(WHEEL_NAMES, data.first_alarm_times):
                if alarm_time is None:
                    continue
                if event_time is None:
                    alarm_parts.append(f"{name}: {alarm_time:.3f}s")
                else:
                    alarm_parts.append(
                        f"{name}: {alarm_time:.3f}s (相对事件 {alarm_time - event_time:+.3f}s)"
                    )
            alarm_text = "；".join(alarm_parts) if alarm_parts else "无报警"
            classification = str(evaluation["classification"])
            classification_label = str(evaluation["classification_label"])
            false_wheels = str(evaluation["false_alarm_wheels"] or "—")
            panels.append(
                f"<section class='algorithm-panel'><h2>{algorithm_names[selected]}　"
                f"<span class='badge status-{html.escape(classification)}'>"
                f"{html.escape(classification_label)}</span></h2>"
                f"<div class='cards'><div><b>类型</b><br>{html.escape(row['sample_type'])}</div>"
                f"<div><b>来源</b><br>{html.escape(row['source_event_id'])}</div>"
                f"<div><b>目标</b><br>{html.escape(row.get('target_wheels', '') or '—')}</div>"
                f"<div><b>检测</b><br>{html.escape(alarm_text)}</div>"
                f"<div><b>误报轮位</b><br>{html.escape(false_wheels)}</div></div>"
                f"{plot_html}</section>"
            )
        tabs = "".join(
            f"<a class='algorithm-tab {'active' if value == algorithm else ''}' "
            f"href='/sample/{quote(sample_id)}?algorithm={value}'>{label}</a>"
            for value, label in (
                ("hard", "当前版"),
                ("evidence", "Evidence"),
                ("compare", "对比"),
            )
        )
        return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(sample_id)}</title><style>{_STYLE}</style></head><body>
<main class="wide"><nav><a href='/?algorithm={quote(algorithm)}'>← 返回488样本</a><span>{previous_link}　{next_link}</span></nav>
<h1>{html.escape(sample_id)}</h1><div class="algorithm-tabs">{tabs}</div>
{''.join(panels)}</main></body></html>"""


_STYLE = """
body{font-family:system-ui,-apple-system,sans-serif;margin:0;color:#172033;background:#f7f9fc}main{max-width:1450px;margin:32px auto;padding:0 20px}.wide{max-width:1500px}h1{margin-bottom:8px}.muted{color:#667085}.summary{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0}.controls{display:flex;gap:10px;align-items:center;margin:18px 0;position:sticky;top:0;background:#f7f9fc;padding:10px 0;z-index:5}input,select{padding:9px 11px;border:1px solid #cbd5e1;border-radius:7px;background:white}input{min-width:280px}table{border-collapse:collapse;width:100%;background:white;border-radius:10px}th,td{padding:9px 11px;border-bottom:1px solid #e5eaf0;text-align:left;font-size:14px;white-space:nowrap}th{background:#eef3f8;position:sticky;top:var(--controls-height,61px);z-index:4}thead th:first-child{border-top-left-radius:10px}thead th:last-child{border-top-right-radius:10px}a{color:#185abd;text-decoration:none}a:hover{text-decoration:underline}nav{display:flex;justify-content:space-between}.cards{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin:18px 0}.cards div{background:white;padding:13px;border:1px solid #e0e6ed;border-radius:9px}.badge{display:inline-block;padding:4px 8px;border-radius:999px;font-size:13px;font-weight:650;background:#e2e8f0;color:#334155}.status-EVENT_OK,.status-NORMAL_OK{background:#dcfce7;color:#166534}.status-EVENT_LATE{background:#fef3c7;color:#92400e}.status-EVENT_MISS,.status-EVENT_FALSE_ALARM,.status-EVENT_FALSE_ALARM_AND_DETECTED,.status-NORMAL_FALSE_ALARM{background:#fee2e2;color:#991b1b}.algorithm-tabs{display:flex;gap:8px;margin:16px 0 22px}.algorithm-tab{padding:9px 16px;border:1px solid #cbd5e1;border-radius:8px;background:white;font-weight:650}.algorithm-tab.active{background:#185abd;color:white;border-color:#185abd}.algorithm-panel{background:white;border:1px solid #e0e6ed;border-radius:12px;padding:18px;margin:0 0 24px}.algorithm-panel h2{margin:0 0 12px}.algorithm-panel .cards div{background:#f8fafc}@media(max-width:800px){.cards{grid-template-columns:1fr 1fr}.controls{flex-wrap:wrap}table{display:block;overflow:auto}th{position:static}}
"""

_FILTER_SCRIPT = """
const rows=[...document.querySelectorAll('tbody tr')],search=document.querySelector('#search'),kind=document.querySelector('#kind'),source=document.querySelector('#source'),status=document.querySelector('#status'),count=document.querySelector('#count');
const controls=document.querySelector('.controls');
function syncStickyOffset(){document.documentElement.style.setProperty('--controls-height',`${controls.offsetHeight}px`);}
new ResizeObserver(syncStickyOffset).observe(controls);syncStickyOffset();
function filter(){const q=search.value.trim().toLowerCase();let n=0;for(const row of rows){const show=(!q||row.textContent.toLowerCase().includes(q))&&(!kind.value||row.dataset.type===kind.value)&&(!source.value||row.dataset.source===source.value)&&(!status.value||row.dataset.status.split(' ').includes(status.value));row.hidden=!show;if(show)n++;}count.textContent=`显示 ${n}/${rows.length}`;}search.addEventListener('input',filter);kind.addEventListener('change',filter);source.addEventListener('change',filter);status.addEventListener('change',filter);filter();
"""


class ViewerHandler(BaseHTTPRequestHandler):
    state: ViewerState

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/":
                algorithm = parse_qs(parsed.query).get("algorithm", ["hard"])[0]
                self._send_html(self.state.render_index(algorithm))
                return
            if path.startswith("/sample/"):
                sample_id = unquote(path.removeprefix("/sample/"))
                algorithm = parse_qs(parsed.query).get("algorithm", ["hard"])[0]
                self._send_html(self.state.render_sample(sample_id, algorithm))
                return
        except KeyError:
            self.send_error(404, "sample not found")
            return
        except Exception as exc:  # Keep one bad sample from stopping the viewer.
            self.send_error(500, str(exc))
            return
        self.send_error(404)

    def _send_html(self, content: str) -> None:
        payload = content.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the 488-case wavelet display.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION_PATH)
    parser.add_argument(
        "--evidence-evaluation",
        type=Path,
        default=DEFAULT_EVIDENCE_EVALUATION_PATH,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--cache-size", type=int, default=12)
    parser.add_argument("--window-before", type=float, default=5.0)
    parser.add_argument("--window-after", type=float, default=5.0)
    parser.add_argument(
        "--reference-mode", choices=REFERENCE_MODES, default="opposite_diagonal"
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.cache_size < 0:
        parser.error("--cache-size cannot be negative")
    if args.window_before < 0.0 or args.window_after < 0.0:
        parser.error("display windows cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    cfg = WaveletShapeConfig(reference_mode=args.reference_mode)
    ViewerHandler.state = ViewerState(
        args.dataset_dir,
        cfg,
        window_before_s=args.window_before,
        window_after_s=args.window_after,
        cache_size=args.cache_size,
        evaluation_path=args.evaluation,
        evidence_evaluation_path=args.evidence_evaluation,
    )
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    print(f"serving {len(ViewerHandler.state.rows)} samples at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
