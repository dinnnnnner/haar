from __future__ import annotations

import argparse
import html
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from plotly.offline import get_plotlyjs

from .pressure_fusion_detector import WHEEL_NAMES, PressureFusionConfig
from .pressure_fusion_display import (
    PLOT_CONFIG,
    ScanResult,
    analyze_window,
    build_figure,
    scan_csv,
)
from .serve_pressure_fusion_acceptance import (
    DEFAULT_BATCH_SUMMARY,
    DEFAULT_REPORT,
    DEFAULT_ROBUST_MANIFEST,
    LAYOUTS,
    AcceptanceState,
)


class DisplayState(AcceptanceState):
    def __init__(
        self,
        report_path: Path,
        robust_manifest: Path,
        batch_summary: Path,
        cfg: PressureFusionConfig,
        max_window_s: float = 120.0,
        cache_size: int = 12,
    ) -> None:
        super().__init__(
            report_path,
            robust_manifest,
            batch_summary,
            cfg,
            max_window_s=max_window_s,
            cache_size=cache_size,
        )
        self._cached_scan = lru_cache(maxsize=cache_size)(self._scan_case)
        self._cached_display = lru_cache(maxsize=cache_size)(self._render_case)

    def render_index(self, layout: str = "FR_RL") -> str:
        self._validate_layout(layout)
        rows = []
        for case_id in self.case_ids:
            case = self.cases[case_id]
            rows.append(
                f"<tr data-type='{case['sample_type']}' "
                f"data-scenario='{html.escape(str(case['scenario']))}'>"
                f"<td><a href='/case/{quote(case_id)}?layout={layout}'>{case_id}</a></td>"
                f"<td><a href='/case/{quote(case_id)}?layout={layout}'>{html.escape(str(case['name']))}</a></td>"
                f"<td>{html.escape(str(case['scenario']))}</td>"
                f"<td>{'爆胎' if case['sample_type'] == 'event' else '正常道路'}</td>"
                f"<td>{'RR' if case['sample_type'] == 'event' else '—'}</td>"
                "<td><span class='scan-hint'>打开后扫描</span></td>"
                f"<td><a class='mini-button' href='/case/{quote(case_id)}?layout={layout}'>看疑似段</a></td></tr>"
            )
        return _page(
            "胎压融合算法显示台",
            f"""
<header><div><p class='eyebrow'>PRESSURE × WHEEL SPEED</p><h1>胎压融合算法显示台</h1>
<p class='muted'>逐条回放算法内部疑似候选，查看四轮轮速与判定证据。</p></div>
<div class='legend'><span><i class='candidate'></i>疑似候选</span><span><i class='confirmed'></i>确认报警</span></div></header>
{self._layout_tabs("/", layout)}
<section class='cards intro-cards'>
  <div class='card'><span>记录</span><strong>{len(self.case_ids)}</strong><small>爆胎 + 正常道路</small></div>
  <div class='card'><span>胎压对角</span><strong>{layout.replace('_', '+')}</strong><small>另一对角由轮速检测</small></div>
  <div class='card'><span>疑似定义</span><strong>candidate</strong><small>逐轮与对角上升沿同时过门限</small></div>
  <div class='card'><span>图表交互</span><strong>Plotly</strong><small>拖动 · 滚轮缩放 · 范围条</small></div>
</section>
<section class='panel'><div class='controls'>
  <input id='search' placeholder='搜索 ID、文件、道路…'>
  <select id='type'><option value=''>全部类型</option><option value='event'>爆胎</option><option value='normal'>正常道路</option></select>
  <select id='scenario'><option value=''>全部场景</option>{self._scenario_options()}</select>
  <span id='count'></span>
</div><div class='table-wrap'><table><thead><tr><th>ID</th><th>记录</th><th>场景</th><th>类型</th><th>目标轮</th><th>疑似段</th><th></th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div></section>
<section class='notice'><b>查看方式：</b>首次打开一条记录时会完整回放并缓存疑似段；长记录可能需要数秒。曲线窗口仍从记录开头运行算法，以保留因果基线。</section>
<script>{_FILTER_SCRIPT}</script>
""",
        )

    def render_case(
        self,
        case_id: str,
        layout: str,
        start_s: float | None,
        end_s: float | None,
    ) -> str:
        self._validate_layout(layout)
        if case_id not in self.cases:
            raise KeyError(case_id)
        scan = self._cached_scan(case_id, layout)
        event_time = self.cases[case_id]["event_time_s"]
        if start_s is None and end_s is None:
            if scan.suspects:
                focus_start = scan.suspects[0].start_s
                focus_end = scan.suspects[0].end_s
                start_s = max(scan.start_s, focus_start - 2.0)
                end_s = min(scan.end_s, focus_end + 2.0)
            elif event_time is not None:
                start_s = max(scan.start_s, float(event_time) - 2.0)
                end_s = min(scan.end_s, float(event_time) + 3.0)
            else:
                start_s = scan.start_s
                end_s = min(scan.end_s, scan.start_s + 30.0)
        if start_s is None or end_s is None:
            raise ValueError("start 和 end 必须同时提供")
        start_s = max(scan.start_s, start_s)
        end_s = min(scan.end_s, end_s)
        if end_s <= start_s:
            raise ValueError("时间窗口无效")
        if end_s - start_s > self.max_window_s:
            raise ValueError(f"单次窗口不能超过 {self.max_window_s:g} 秒")
        return self._cached_display(
            case_id,
            layout,
            round(start_s, 3),
            round(end_s, 3),
        )

    def _scan_case(self, case_id: str, layout: str) -> ScanResult:
        case = self.cases[case_id]
        csv_path: Path = case["csv_path"]  # type: ignore[assignment]
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        event_time = case["event_time_s"]
        return scan_csv(
            csv_path,
            LAYOUTS[layout],
            self.cfg,
            None if event_time is None else float(event_time),
        )

    def _render_case(
        self, case_id: str, layout: str, start_s: float, end_s: float
    ) -> str:
        case = self.cases[case_id]
        scan = self._cached_scan(case_id, layout)
        csv_path: Path = case["csv_path"]  # type: ignore[assignment]
        event_time = case["event_time_s"]
        data = analyze_window(
            csv_path,
            LAYOUTS[layout],
            start_s,
            end_s,
            self.cfg,
            None if event_time is None else float(event_time),
        )
        figure = build_figure(
            data,
            self.cfg,
            scan.suspects,
            None if event_time is None else float(event_time),
            f"{case_id} · {case['name']}",
        )
        plot_html = figure.to_html(
            full_html=False,
            include_plotlyjs=False,
            config=PLOT_CONFIG,
        )
        confirmed = sum(interval.confirmed for interval in scan.suspects)
        suspect_rows = []
        for index, interval in enumerate(scan.suspects, start=1):
            focus_start = max(scan.start_s, interval.start_s - 2.0)
            focus_end = min(scan.end_s, interval.end_s + 2.0)
            selected = interval.end_s >= start_s and interval.start_s <= end_s
            status = "已确认" if interval.confirmed else "已排除"
            peak_individual = (
                "—"
                if interval.peak_individual_gain_pct is None
                else f"{interval.peak_individual_gain_pct:.3f}%"
            )
            peak_diagonal = (
                "—"
                if interval.peak_diagonal_gain_pct is None
                else f"{interval.peak_diagonal_gain_pct:.3f}%"
            )
            suspect_rows.append(
                f"<a class='suspect {'selected' if selected else ''}' "
                f"href='/case/{quote(case_id)}?layout={layout}&start={focus_start:.3f}&end={focus_end:.3f}#plot'>"
                f"<span class='suspect-head'><b>#{index} · {WHEEL_NAMES[interval.wheel_index]}</b>"
                f"<em class='{'ok' if interval.confirmed else ''}'>{status}</em></span>"
                f"<span>{interval.start_s:.3f}–{interval.end_s:.3f}s · {interval.duration_s:.3f}s</span>"
                f"<small>逐轮峰值 {peak_individual}　对角峰值 {peak_diagonal}</small></a>"
            )
        if not suspect_rows:
            suspect_rows.append(
                "<div class='empty'><b>没有疑似候选</b><span>本记录未进入 candidate 状态。</span></div>"
            )
        position = self.case_ids.index(case_id)
        previous_link = (
            ""
            if position == 0
            else f"<a class='button secondary' href='/case/{quote(self.case_ids[position - 1])}?layout={layout}'>← 上一条</a>"
        )
        next_link = (
            ""
            if position + 1 == len(self.case_ids)
            else f"<a class='button secondary' href='/case/{quote(self.case_ids[position + 1])}?layout={layout}'>下一条 →</a>"
        )
        width = end_s - start_s
        previous_start = max(scan.start_s, start_s - width)
        previous_end = previous_start + width
        next_end = min(scan.end_s, end_s + width)
        next_start = next_end - width
        return _page(
            str(case["name"]),
            f"""
<nav class='top-nav'><a href='/?layout={layout}'>← 返回显示台</a><span>{previous_link}{next_link}</span></nav>
<header class='case-header'><div><p class='eyebrow'>{case_id} · {html.escape(str(case['scenario']))}</p>
<h1>{html.escape(str(case['name']))}</h1><p class='path'>{html.escape(str(csv_path))}</p></div></header>
{self._layout_tabs(f"/case/{quote(case_id)}", layout)}
<section class='cards'>
  <div class='card'><span>疑似候选</span><strong>{len(scan.suspects)}</strong><small>完整记录扫描</small></div>
  <div class='card'><span>最终确认</span><strong>{confirmed}</strong><small>其余候选被算法排除</small></div>
  <div class='card'><span>记录时长</span><strong>{scan.end_s - scan.start_s:.1f}s</strong><small>{scan.frames:,} 帧</small></div>
  <div class='card'><span>当前窗口</span><strong>{start_s:.2f}–{end_s:.2f}s</strong><small>{end_s - start_s:.2f} 秒</small></div>
</section>
<form class='range panel' method='get'><input type='hidden' name='layout' value='{layout}'>
  <label>开始/s <input type='number' step='0.01' name='start' value='{start_s:.2f}'></label>
  <label>结束/s <input type='number' step='0.01' name='end' value='{end_s:.2f}'></label>
  <button>查看窗口</button>
  <a class='mini-button' href='/case/{quote(case_id)}?layout={layout}&start={previous_start:.3f}&end={previous_end:.3f}#plot'>← 前一屏</a>
  <a class='mini-button' href='/case/{quote(case_id)}?layout={layout}&start={next_start:.3f}&end={next_end:.3f}#plot'>后一屏 →</a>
  <span class='muted'>最多 {self.max_window_s:g} 秒</span>
</form>
<div class='workbench'>
  <aside><div class='aside-title'><h2>疑似部分</h2><span>点选后跳转</span></div>{''.join(suspect_rows)}</aside>
  <section id='plot' class='plot-panel'>{plot_html}</section>
</div>
<section class='notice'><b>图例：</b>浅色虚线背景表示进入疑似候选后被排除；较深实线背景表示最终确认。轮速、增益和上升沿共用时间轴，双击复位，滚轮缩放，拖动平移。</section>
""",
            include_plotly=True,
        )


class DisplayHandler(BaseHTTPRequestHandler):
    state: DisplayState
    plotly_js = get_plotlyjs().encode("utf-8")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/assets/plotly.min.js":
                self.send_response(200)
                self.send_header("Content-Type", "text/javascript; charset=utf-8")
                self.send_header("Cache-Control", "public, max-age=86400")
                self.send_header("Content-Length", str(len(self.plotly_js)))
                self.end_headers()
                self.wfile.write(self.plotly_js)
                return
            if parsed.path == "/":
                self._send(
                    self.state.render_index(query.get("layout", ["FR_RL"])[0])
                )
                return
            if parsed.path.startswith("/case/"):
                case_id = unquote(parsed.path.removeprefix("/case/"))
                start = query.get("start", [None])[0]
                end = query.get("end", [None])[0]
                self._send(
                    self.state.render_case(
                        case_id,
                        query.get("layout", ["FR_RL"])[0],
                        None if start is None else float(start),
                        None if end is None else float(end),
                    )
                )
                return
            self.send_error(404)
        except (KeyError, ValueError, FileNotFoundError) as error:
            self.send_error(400, str(error))

    def _send(self, body: str) -> None:
        payload = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return


def _page(title: str, body: str, include_plotly: bool = False) -> str:
    script = "<script src='/assets/plotly.min.js'></script>" if include_plotly else ""
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title>
<style>{_STYLE}</style>{script}</head><body><main>{body}</main></body></html>"""


_FILTER_SCRIPT = """
const rows=[...document.querySelectorAll('tbody tr')],search=document.querySelector('#search'),type=document.querySelector('#type'),scenario=document.querySelector('#scenario'),count=document.querySelector('#count');
function apply(){const q=search.value.toLowerCase();let visible=0;rows.forEach(row=>{const ok=(!q||row.innerText.toLowerCase().includes(q))&&(!type.value||row.dataset.type===type.value)&&(!scenario.value||row.dataset.scenario===scenario.value);row.hidden=!ok;if(ok)visible++;});count.textContent=`显示 ${visible}/${rows.length}`;}[search,type,scenario].forEach(x=>x.addEventListener('input',apply));apply();
"""


_STYLE = f"""
:root{{--ink:#14202e;--muted:#687386;--line:#dce3eb;--blue:#1d4ed8;--navy:#102a43;--bg:#f3f6f9;--green:#087f5b}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 Inter,system-ui,-apple-system,"Segoe UI",sans-serif}}main{{max-width:1600px;margin:auto;padding:26px}}
header{{display:flex;justify-content:space-between;align-items:flex-start;gap:24px;margin:6px 0 18px}}h1{{font-size:29px;line-height:1.2;margin:3px 0 7px}}h2{{margin:0;font-size:17px}}.eyebrow{{color:#1d4ed8;font-size:12px;font-weight:800;letter-spacing:.12em;margin:0}}.muted,.path{{color:var(--muted)}}.path{{margin:0;max-width:1000px;word-break:break-all;font-size:12px}}
a{{color:var(--blue);text-decoration:none}}.button,button{{border:0;border-radius:7px;background:var(--blue);color:#fff;padding:9px 14px;cursor:pointer;display:inline-block}}.secondary{{background:#64748b}}.mini-button{{display:inline-block;border-radius:6px;background:#e0e7ff;color:#3730a3;padding:6px 10px;font-weight:650}}
.top-nav{{display:flex;justify-content:space-between;align-items:center;margin-bottom:20px}}.top-nav>span{{display:flex;gap:8px}}.top-nav .button{{padding:7px 11px}}.case-header{{margin-bottom:4px}}
.legend{{display:flex;gap:18px;background:#fff;border:1px solid var(--line);border-radius:9px;padding:12px 15px}}.legend span{{display:flex;align-items:center;gap:7px;color:var(--muted)}}.legend i{{width:22px;height:12px;border-radius:3px;background:#2563eb33;border:1px dotted #2563eb}}.legend i.confirmed{{background:#dc262644;border:2px solid #dc2626}}
.tabs{{display:flex;gap:8px;margin:15px 0}}.tabs a{{padding:8px 13px;border-radius:7px;background:#fff;border:1px solid var(--line);color:var(--ink)}}.tabs a.active{{background:var(--navy);border-color:var(--navy);color:#fff}}
.cards{{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:11px;margin:14px 0}}.card,.panel,.plot-panel,.notice,aside{{background:#fff;border:1px solid var(--line);border-radius:10px}}.card{{padding:14px 16px}}.card span,.card small{{display:block;color:var(--muted)}}.card strong{{display:block;font-size:22px;margin:2px 0}}.intro-cards .card:nth-child(3) strong,.intro-cards .card:nth-child(4) strong{{font-size:19px}}
.panel,.notice{{padding:16px;margin:14px 0}}.notice{{background:#fffbeb;border-color:#f5d98b}}.controls,.range{{display:flex;align-items:center;gap:10px;flex-wrap:wrap}}input,select{{background:#fff;border:1px solid #cbd5e1;border-radius:6px;padding:9px 10px}}.controls input{{min-width:270px}}
.table-wrap{{overflow:auto}}table{{width:100%;border-collapse:collapse;margin-top:12px}}th,td{{text-align:left;padding:9px 10px;border-bottom:1px solid #e8edf2;white-space:nowrap}}th{{background:#f8fafc;position:sticky;top:0}}td a{{font-weight:650}}.scan-hint{{color:var(--muted)}}
.workbench{{display:grid;grid-template-columns:290px minmax(0,1fr);gap:14px;align-items:start}}aside{{padding:12px;position:sticky;top:12px;max-height:calc(100vh - 24px);overflow:auto}}.aside-title{{display:flex;align-items:baseline;justify-content:space-between;padding:3px 5px 10px}}.aside-title span{{font-size:12px;color:var(--muted)}}
.suspect{{display:block;color:var(--ink);border:1px solid #e3e8ef;border-left:4px solid #94a3b8;border-radius:7px;padding:9px 10px;margin:0 0 8px;background:#fff}}.suspect:hover,.suspect.selected{{border-color:#93b4e8;background:#eff6ff}}.suspect-head{{display:flex;justify-content:space-between;align-items:center}}.suspect span,.suspect small{{display:block}}.suspect small{{color:var(--muted);margin-top:3px}}.suspect em{{font-style:normal;font-size:11px;background:#f1f5f9;color:#64748b;padding:2px 6px;border-radius:999px}}.suspect em.ok{{background:#dcfce7;color:#166534}}.empty{{padding:24px 8px;text-align:center;color:var(--muted)}}.empty b,.empty span{{display:block}}
.plot-panel{{min-width:0;padding:2px 6px;overflow:hidden}}.range label{{display:flex;align-items:center;gap:5px}}.range input{{width:105px}}
@media(max-width:900px){{main{{padding:15px}}.cards{{grid-template-columns:1fr 1fr}}.workbench{{grid-template-columns:1fr}}aside{{position:static;max-height:330px}}header{{display:block}}.legend{{margin-top:12px;width:max-content}}}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Pressure-fusion algorithm display console")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--robust-manifest", type=Path, default=DEFAULT_ROBUST_MANIFEST)
    parser.add_argument("--batch-summary", type=Path, default=DEFAULT_BATCH_SUMMARY)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument("--max-window-s", type=float, default=120.0)
    args = parser.parse_args()
    state = DisplayState(
        args.report,
        args.robust_manifest,
        args.batch_summary,
        PressureFusionConfig(),
        max_window_s=args.max_window_s,
    )
    DisplayHandler.state = state
    server = ThreadingHTTPServer((args.host, args.port), DisplayHandler)
    print(f"Pressure fusion algorithm display: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
