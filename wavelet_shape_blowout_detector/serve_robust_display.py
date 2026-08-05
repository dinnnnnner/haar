from __future__ import annotations

import argparse
import csv
import html
import math
import sys
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, quote, urlparse

if __package__:
    from .detector import REFERENCE_MODES, WHEEL_NAMES, WaveletShapeConfig
    from .display import PLOT_CONFIG, analyze_csv, build_figure
    from .tooth_display import analyze_tooth_file, build_tooth_figure
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
    from wavelet_shape_blowout_detector.tooth_display import (  # noqa: E402
        analyze_tooth_file,
        build_tooth_figure,
    )


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "robust_data_results"
ALGORITHM_NAMES = {"hard": "当前版", "evidence": "Evidence", "tooth": "齿信号"}


def _optional_float(value: object) -> float | None:
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "na"}:
        return None
    number = float(text)
    return number if math.isfinite(number) else None


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class RobustViewerState:
    def __init__(
        self,
        results_dir: Path,
        cfg: WaveletShapeConfig,
        window_before_s: float = 5.0,
        window_after_s: float = 5.0,
        max_window_s: float = 120.0,
        cache_size: int = 8,
    ) -> None:
        self.results_dir = results_dir.resolve()
        self.cfg = cfg
        self.window_before_s = window_before_s
        self.window_after_s = window_after_s
        self.max_window_s = max_window_s
        self.cache_size = cache_size
        self.summary_path = self.results_dir / "robust_evaluation_summary.csv"
        with self.summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(
                f"evaluation summary contains no rows: {self.summary_path}"
            )
        self.summary_signature = self._summary_signature()

        grouped: dict[str, list[dict[str, str]]] = {}
        order: list[str] = []
        for row in rows:
            input_file = row["input_file"]
            if input_file not in grouped:
                grouped[input_file] = []
                order.append(input_file)
            grouped[input_file].append(row)
        self.case_ids = [f"R{index:03d}" for index in range(1, len(order) + 1)]
        self.cases: dict[str, dict[str, object]] = {}
        for case_id, input_file in zip(self.case_ids, order):
            case_rows = grouped[input_file]
            status = next(
                (row.get("status", "error") for row in case_rows if row.get("status") != "ok"),
                "ok",
            )
            evaluations = {
                row["algorithm"]: row
                for row in case_rows
                if row.get("status") == "ok" and row.get("algorithm")
            }
            self.cases[case_id] = {
                "id": case_id,
                "input_file": Path(input_file),
                "case_dir": Path(case_rows[0].get("case_dir", "")),
                "status": status,
                "error": case_rows[0].get("error", ""),
                "evaluations": evaluations,
            }
        self.positions = {
            case_id: index for index, case_id in enumerate(self.case_ids)
        }
        self._cached_case = lru_cache(maxsize=cache_size)(self._render_case)

    def _summary_signature(self) -> tuple[int, int]:
        stat = self.summary_path.stat()
        return stat.st_mtime_ns, stat.st_size

    def reload_if_changed(self) -> RobustViewerState:
        if self._summary_signature() == self.summary_signature:
            return self
        return type(self)(
            self.results_dir,
            self.cfg,
            window_before_s=self.window_before_s,
            window_after_s=self.window_after_s,
            max_window_s=self.max_window_s,
            cache_size=self.cache_size,
        )

    @staticmethod
    def _evaluation_status(row: dict[str, str] | None) -> tuple[str, str]:
        if row is None:
            return "UNASSESSED", "未评估"
        if _as_bool(row.get("confirmed_false_alarm", "")):
            return "CONFIRMED_FALSE", "确认误报"
        if _as_bool(row.get("false_alarm", "")):
            return "FAST_FALSE", "快速误报"
        return "NORMAL_OK", "正常通过"

    @staticmethod
    def _first_alarm(row: dict[str, str] | None) -> float | None:
        if row is None:
            return None
        values = [
            _optional_float(row.get(f"{name}_first_alarm_time_s"))
            for name in WHEEL_NAMES
        ]
        finite = [value for value in values if value is not None]
        return min(finite) if finite else None

    def _selected_algorithms(self, algorithm: str) -> tuple[str, ...]:
        if algorithm == "compare":
            return ("hard", "evidence")
        if algorithm == "tooth":
            # The index and default window keep using the current detector's
            # evaluation metadata; the detail panel itself is rendered from
            # raw tooth timestamps.
            return ("hard",)
        if algorithm in ALGORITHM_NAMES:
            return (algorithm,)
        raise ValueError("algorithm must be hard, evidence, compare, or tooth")

    def render_index(self, algorithm: str = "hard") -> str:
        selected = self._selected_algorithms(algorithm)
        summary_counts = {"NORMAL_OK": 0, "FAST_FALSE": 0, "CONFIRMED_FALSE": 0}
        table_rows: list[str] = []
        for case_id in self.case_ids:
            case = self.cases[case_id]
            input_file = case["input_file"]
            evaluations = case["evaluations"]
            status = str(case["status"])
            status_codes: list[str] = []
            status_parts: list[str] = []
            wheel_parts: list[str] = []
            time_parts: list[str] = []
            duration: float | None = None
            for selected_algorithm in selected:
                row = evaluations.get(selected_algorithm)  # type: ignore[union-attr]
                code, label = self._evaluation_status(row)
                status_codes.append(code)
                if row is not None:
                    summary_counts[code] += 1
                    duration = duration or _optional_float(row.get("duration_s"))
                prefix = (
                    f"{ALGORITHM_NAMES[selected_algorithm]} "
                    if algorithm == "compare"
                    else ""
                )
                status_parts.append(
                    f"<span class='badge status-{code}'>{html.escape(prefix + label)}</span>"
                )
                wheels = "—" if row is None else (row.get("alarm_wheels") or "—")
                first_alarm = self._first_alarm(row)
                wheel_parts.append(html.escape(prefix + wheels))
                time_parts.append(
                    html.escape(
                        prefix + ("—" if first_alarm is None else f"{first_alarm:.2f}")
                    )
                )
            if status != "ok":
                label = "加密跳过" if status == "locked" else "处理错误"
                status_codes = [status.upper()]
                status_parts = [f"<span class='badge status-{status.upper()}'>{label}</span>"]
            name = input_file.stem  # type: ignore[union-attr]
            group = input_file.parent.name  # type: ignore[union-attr]
            name_html = (
                f"<a href='/case/{quote(case_id)}?algorithm={quote(algorithm)}'>"
                f"{html.escape(name)}</a>"
                if status == "ok"
                else html.escape(name)
            )
            table_rows.append(
                f"<tr data-group='{html.escape(group)}' "
                f"data-status='{html.escape(' '.join(status_codes))}'>"
                f"<td>{case_id}</td><td>{name_html}</td><td>{html.escape(group)}</td>"
                f"<td>{'<br>'.join(status_parts)}</td>"
                f"<td>{'<br>'.join(wheel_parts)}</td>"
                f"<td>{'<br>'.join(time_parts)}</td>"
                f"<td>{'—' if duration is None else f'{duration:.2f}'}</td></tr>"
            )
        tabs = self._tabs("/", algorithm)
        summary = "".join(
            f"<span class='badge status-{code}'>{label} {count}</span>"
            for code, label, count in (
                ("NORMAL_OK", "正常通过", summary_counts["NORMAL_OK"]),
                ("FAST_FALSE", "仅快速误报", summary_counts["FAST_FALSE"]),
                (
                    "CONFIRMED_FALSE",
                    "确认误报",
                    summary_counts["CONFIRMED_FALSE"],
                ),
            )
        )
        groups = sorted(
            {case["input_file"].parent.name for case in self.cases.values()}  # type: ignore[union-attr]
        )
        return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RobustData Display</title><style>{_STYLE}</style></head><body><main>
<h1>RobustData 实路鲁棒性 Display</h1>
<p class="muted">共 {len(self.cases)} 条已记录结果。点击记录后默认定位到首个误报附近；详情页可输入任意时间窗口。</p>
<div class="algorithm-tabs">{tabs}</div><div class="summary">{summary}</div>
<div class="controls"><input id="search" placeholder="搜索文件、轮位、R001…">
<select id="group"><option value="">全部道路</option>{''.join(f'<option>{html.escape(group)}</option>' for group in groups)}</select>
<select id="status"><option value="">全部状态</option><option value="NORMAL_OK">正常通过</option><option value="FAST_FALSE">快速误报</option><option value="CONFIRMED_FALSE">确认误报</option><option value="LOCKED">加密跳过</option><option value="ERROR">处理错误</option></select><span id="count"></span></div>
<table><thead><tr><th>ID</th><th>记录</th><th>道路</th><th>评价</th><th>报警轮</th><th>首报/s</th><th>时长/s</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
</main><script>{_FILTER_SCRIPT}</script></body></html>"""

    def render_case(
        self,
        case_id: str,
        algorithm: str,
        start_time_s: float | None,
        end_time_s: float | None,
    ) -> str:
        self._selected_algorithms(algorithm)
        case = self.cases.get(case_id)
        if case is None:
            raise KeyError(case_id)
        if case["status"] != "ok":
            raise ValueError(f"case is not viewable: {case['status']}")
        evaluations = case["evaluations"]
        selected = self._selected_algorithms(algorithm)
        if start_time_s is None and end_time_s is None:
            first_alarms = [
                self._first_alarm(evaluations.get(name))  # type: ignore[union-attr]
                for name in selected
            ]
            centers = [value for value in first_alarms if value is not None]
            if centers:
                center = min(centers)
                start_time_s = max(0.0, center - self.window_before_s)
                end_time_s = center + self.window_after_s
            else:
                start_time_s = 0.0
                end_time_s = min(30.0, self.max_window_s)
        elif start_time_s is None or end_time_s is None:
            raise ValueError("start and end must be supplied together")
        if start_time_s < 0.0 or end_time_s <= start_time_s:
            raise ValueError("display time range is invalid")
        allowed_max_window = (
            min(self.max_window_s, 30.0) if algorithm == "tooth" else self.max_window_s
        )
        if end_time_s - start_time_s > allowed_max_window:
            raise ValueError(
                f"display window cannot exceed {allowed_max_window:g} seconds"
            )
        return self._cached_case(case_id, algorithm, start_time_s, end_time_s)

    def _render_case(
        self,
        case_id: str,
        algorithm: str,
        start_time_s: float,
        end_time_s: float,
    ) -> str:
        case = self.cases[case_id]
        input_file: Path = case["input_file"]  # type: ignore[assignment]
        case_dir: Path = case["case_dir"]  # type: ignore[assignment]
        evaluations: dict[str, dict[str, str]] = case["evaluations"]  # type: ignore[assignment]
        if algorithm == "tooth":
            factors_path = case_dir / "learned_tooth_correction_factors.csv"
            if not input_file.is_file():
                raise FileNotFoundError(f"raw tooth timestamp file not found: {input_file}")
            if not factors_path.is_file():
                raise FileNotFoundError(f"tooth factor file not found: {factors_path}")
            tooth_data = analyze_tooth_file(
                input_file,
                factors_path,
                start_time_s,
                end_time_s,
            )
            hard_row = evaluations.get("hard")
            alarm_markers: list[tuple[str, float, str]] = []
            if hard_row is not None:
                for wheel_index, name in enumerate(WHEEL_NAMES):
                    alarm_time = _optional_float(
                        hard_row.get(f"{name}_first_alarm_time_s")
                    )
                    if alarm_time is not None:
                        alarm_markers.append(
                            (f"{name}首报", alarm_time, ("#2563eb", "#f59e0b", "#16a34a", "#dc2626")[wheel_index])
                        )
            figure = build_tooth_figure(
                tooth_data,
                title=(
                    f"{input_file.name} — 齿信号 "
                    f"[{start_time_s:.2f}, {end_time_s:.2f}] s"
                ),
                alarm_times=alarm_markers,
            )
            plot_html = figure.to_html(
                full_html=False,
                include_plotlyjs="cdn",
                config=PLOT_CONFIG,
            )
            wraps = " / ".join(
                f"{name} {count}"
                for name, count in zip(WHEEL_NAMES, tooth_data.timer_wraps)
            )
            baselines = " / ".join(
                f"{name} {value:.5f}"
                for name, value in zip(
                    WHEEL_NAMES, tooth_data.baseline_rate_ratios
                )
            )
            alarm_links = self._alarm_links(case_id, algorithm, hard_row)
            panels = [
                "<section class='algorithm-panel'><h2>齿信号事件域</h2>"
                "<div class='cards'>"
                f"<div><b>窗口内齿事件</b><br>{tooth_data.displayed_tooth_events}</div>"
                f"<div><b>异常齿周期</b><br>{tooth_data.abnormal_period_events}</div>"
                f"<div><b>计时器回绕</b><br>{html.escape(wraps)}</div>"
                f"<div><b>正常速率比基线</b><br>{html.escape(baselines)}</div>"
                f"<div><b>首报跳转</b><br>{alarm_links}</div>"
                f"</div>{plot_html}</section>"
            ]
        else:
            panels = []
        wheel_csv = case_dir / "wheel_speed_raw_vs_corrected.csv"
        if algorithm != "tooth" and not wheel_csv.is_file():
            raise FileNotFoundError(f"converted wheel-speed CSV not found: {wheel_csv}")
        selected = () if algorithm == "tooth" else self._selected_algorithms(algorithm)
        for panel_index, selected_algorithm in enumerate(selected):
            data = analyze_csv(
                wheel_csv,
                cfg=self.cfg,
                start_time_s=start_time_s,
                end_time_s=end_time_s,
                algorithm=selected_algorithm,
            )
            false_intervals = [
                interval
                for intervals in data.alarm_intervals
                for interval in intervals
            ]
            figure = build_figure(
                data,
                self.cfg,
                title=(
                    f"{input_file.name} — {ALGORITHM_NAMES[selected_algorithm]} "
                    f"[{start_time_s:.2f}, {end_time_s:.2f}] s"
                ),
                false_alarm_intervals=false_intervals,
            )
            plot_html = figure.to_html(
                full_html=False,
                include_plotlyjs="cdn" if panel_index == 0 else False,
                config=PLOT_CONFIG,
            )
            row = evaluations.get(selected_algorithm)
            code, label = self._evaluation_status(row)
            alarm_links = self._alarm_links(case_id, algorithm, row)
            confirmed_wheels = "—" if row is None else (
                row.get("confirmed_alarm_wheels") or "—"
            )
            panels.append(
                f"<section class='algorithm-panel'><h2>{ALGORITHM_NAMES[selected_algorithm]}　"
                f"<span class='badge status-{code}'>{label}</span></h2>"
                f"<div class='cards'><div><b>快速报警轮</b><br>{html.escape('—' if row is None else (row.get('alarm_wheels') or '—'))}</div>"
                f"<div><b>确认误报轮</b><br>{html.escape(confirmed_wheels)}</div>"
                f"<div><b>快速报警次数</b><br>{html.escape('—' if row is None else (row.get('alarm_events') or '0'))}</div>"
                f"<div><b>确认次数</b><br>{html.escape('—' if row is None else (row.get('confirmation_events') or '0'))}</div>"
                f"<div><b>首报跳转</b><br>{alarm_links}</div></div>{plot_html}</section>"
            )
        position = self.positions[case_id]
        previous_link = self._neighbor_link(position - 1, algorithm, "← 上一条")
        next_link = self._neighbor_link(position + 1, algorithm, "下一条 →")
        tabs = self._tabs(f"/case/{quote(case_id)}", algorithm)
        return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(input_file.name)}</title><style>{_STYLE}</style></head><body><main class="wide">
<nav><a href='/?algorithm={quote(algorithm)}'>← 返回 RobustData</a><span>{previous_link}　{next_link}</span></nav>
<h1>{html.escape(input_file.name)}</h1><p class="muted">{html.escape(str(input_file))}</p>
<div class="algorithm-tabs">{tabs}</div>
<form class="range-form" method="get"><input type="hidden" name="algorithm" value="{html.escape(algorithm)}">
<label>开始/s <input type="number" step="0.01" min="0" name="start" value="{start_time_s:.2f}"></label>
<label>结束/s <input type="number" step="0.01" min="0" name="end" value="{end_time_s:.2f}"></label>
<button type="submit">查看窗口</button><span class="muted">单次最多 {min(self.max_window_s, 30.0) if algorithm == 'tooth' else self.max_window_s:g} 秒</span></form>
{''.join(panels)}</main></body></html>"""

    def _alarm_links(
        self,
        case_id: str,
        algorithm: str,
        row: dict[str, str] | None,
    ) -> str:
        if row is None:
            return "—"
        links: list[str] = []
        for name in WHEEL_NAMES:
            alarm_time = _optional_float(row.get(f"{name}_first_alarm_time_s"))
            if alarm_time is None:
                continue
            start = max(0.0, alarm_time - self.window_before_s)
            end = alarm_time + self.window_after_s
            links.append(
                f"<a href='/case/{quote(case_id)}?algorithm={quote(algorithm)}"
                f"&start={start:.3f}&end={end:.3f}'>{name} {alarm_time:.2f}s</a>"
            )
        return "；".join(links) if links else "—"

    def _neighbor_link(self, position: int, algorithm: str, label: str) -> str:
        if not 0 <= position < len(self.case_ids):
            return ""
        case_id = self.case_ids[position]
        if self.cases[case_id]["status"] != "ok":
            return ""
        return (
            f"<a href='/case/{quote(case_id)}?algorithm={quote(algorithm)}'>"
            f"{label}</a>"
        )

    @staticmethod
    def _tabs(base_path: str, algorithm: str) -> str:
        return "".join(
            f"<a class='algorithm-tab {'active' if value == algorithm else ''}' "
            f"href='{base_path}?algorithm={value}'>{label}</a>"
            for value, label in (
                ("hard", "当前版"),
                ("evidence", "Evidence"),
                ("compare", "对比"),
                ("tooth", "齿信号"),
            )
        )


_STYLE = """
body{font-family:system-ui,-apple-system,sans-serif;margin:0;color:#172033;background:#f7f9fc}main{max-width:1450px;margin:32px auto;padding:0 20px}.wide{max-width:1500px}h1{margin-bottom:8px}.muted{color:#667085}.summary{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0}.controls{display:flex;gap:10px;align-items:center;margin:18px 0;position:sticky;top:0;background:#f7f9fc;padding:10px 0;z-index:5}input,select,button{padding:9px 11px;border:1px solid #cbd5e1;border-radius:7px;background:white}button{background:#185abd;color:white;border-color:#185abd;cursor:pointer}.controls input{min-width:280px}table{border-collapse:collapse;width:100%;background:white;border-radius:10px}th,td{padding:9px 11px;border-bottom:1px solid #e5eaf0;text-align:left;font-size:14px;white-space:nowrap}th{background:#eef3f8;position:sticky;top:var(--controls-height,61px);z-index:4}a{color:#185abd;text-decoration:none}a:hover{text-decoration:underline}nav{display:flex;justify-content:space-between}.cards{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin:18px 0}.cards div{background:#f8fafc;padding:13px;border:1px solid #e0e6ed;border-radius:9px}.badge{display:inline-block;padding:4px 8px;border-radius:999px;font-size:13px;font-weight:650;background:#e2e8f0;color:#334155}.status-NORMAL_OK{background:#dcfce7;color:#166534}.status-FAST_FALSE{background:#fef3c7;color:#92400e}.status-CONFIRMED_FALSE,.status-ERROR{background:#fee2e2;color:#991b1b}.status-LOCKED{background:#e2e8f0;color:#334155}.algorithm-tabs{display:flex;gap:8px;margin:16px 0 22px}.algorithm-tab{padding:9px 16px;border:1px solid #cbd5e1;border-radius:8px;background:white;font-weight:650}.algorithm-tab.active{background:#185abd;color:white;border-color:#185abd}.algorithm-panel{background:white;border:1px solid #e0e6ed;border-radius:12px;padding:18px;margin:0 0 24px}.algorithm-panel h2{margin:0 0 12px}.range-form{display:flex;gap:12px;align-items:center;flex-wrap:wrap;background:white;border:1px solid #e0e6ed;border-radius:10px;padding:12px;margin:0 0 18px}.range-form input{width:100px}@media(max-width:800px){.cards{grid-template-columns:1fr 1fr}.controls{flex-wrap:wrap}table{display:block;overflow:auto}th{position:static}}
"""

_FILTER_SCRIPT = """
const rows=[...document.querySelectorAll('tbody tr')],search=document.querySelector('#search'),group=document.querySelector('#group'),status=document.querySelector('#status'),count=document.querySelector('#count'),controls=document.querySelector('.controls');
function sync(){document.documentElement.style.setProperty('--controls-height',`${controls.offsetHeight}px`)}new ResizeObserver(sync).observe(controls);sync();
function filter(){const q=search.value.trim().toLowerCase();let n=0;for(const row of rows){const show=(!q||row.textContent.toLowerCase().includes(q))&&(!group.value||row.dataset.group===group.value)&&(!status.value||row.dataset.status.split(' ').includes(status.value));row.hidden=!show;if(show)n++}count.textContent=`显示 ${n}/${rows.length}`};search.addEventListener('input',filter);group.addEventListener('change',filter);status.addEventListener('change',filter);filter();
"""


class RobustViewerHandler(BaseHTTPRequestHandler):
    state: RobustViewerState
    state_lock = Lock()

    @classmethod
    def _current_state(cls) -> RobustViewerState:
        with cls.state_lock:
            previous = cls.state
            try:
                current = previous.reload_if_changed()
            except Exception as exc:
                # The evaluator rewrites the CSV in place. Keep serving the last
                # complete snapshot if a request arrives during that short window.
                print(f"could not reload RobustData summary yet: {exc}", flush=True)
                return previous
            if current is not previous:
                cls.state = current
                print(
                    f"reloaded {len(current.cases)} RobustData cases",
                    flush=True,
                )
            return current

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        algorithm = query.get("algorithm", ["hard"])[0]
        state = self._current_state()
        try:
            if parsed.path == "/":
                self._send_html(state.render_index(algorithm))
                return
            if parsed.path.startswith("/case/"):
                case_id = parsed.path.removeprefix("/case/")
                start = _optional_float(query.get("start", [""])[0])
                end = _optional_float(query.get("end", [""])[0])
                self._send_html(
                    state.render_case(case_id, algorithm, start, end)
                )
                return
        except KeyError:
            self.send_error(404, "case not found")
            return
        except Exception as exc:
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
    parser = argparse.ArgumentParser(description="Serve the RobustData display.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--cache-size", type=int, default=8)
    parser.add_argument("--window-before", type=float, default=5.0)
    parser.add_argument("--window-after", type=float, default=5.0)
    parser.add_argument("--max-window", type=float, default=120.0)
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
    if args.max_window <= 0.0:
        parser.error("--max-window must be positive")
    return args


def main() -> None:
    args = parse_args()
    RobustViewerHandler.state = RobustViewerState(
        args.results_dir,
        WaveletShapeConfig(reference_mode=args.reference_mode),
        window_before_s=args.window_before,
        window_after_s=args.window_after,
        max_window_s=args.max_window,
        cache_size=args.cache_size,
    )
    server = ThreadingHTTPServer((args.host, args.port), RobustViewerHandler)
    print(
        f"serving {len(RobustViewerHandler.state.cases)} RobustData cases from "
        f"{RobustViewerHandler.state.results_dir} at http://{args.host}:{args.port}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
