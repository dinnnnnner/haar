from __future__ import annotations

import argparse
import csv
import html
import json
import math
from dataclasses import asdict
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Sequence
from urllib.parse import parse_qs, quote, unquote, urlencode, urlparse

from .console_data import (
    DEFAULT_TIME_COLUMN,
    DEFAULT_WHEEL_COLUMNS,
    ScanResult,
    SuspectInterval,
    analyze_window,
    scan_csv,
)
from .detector import WHEEL_NAMES, WheelSpeedBlowoutConfig


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENT_MANIFEST = WORKSPACE_ROOT / "augmented_event_dataset_v2" / "manifest.csv"
DEFAULT_NORMAL_MANIFEST = WORKSPACE_ROOT / "robust_fast_dataset" / "manifest.csv"
DEFAULT_VALIDATION = (
    WORKSPACE_ROOT / "wheel_speed_only_blowout_detector" / "validation_summary.json"
)
WHEEL_COLORS = ("#2563eb", "#ea8a00", "#16a34a", "#dc2626")


class ConsoleState:
    def __init__(
        self,
        event_manifest: Path,
        normal_manifest: Path,
        validation_path: Path,
        cfg: WheelSpeedBlowoutConfig,
        *,
        max_window_s: float = 120.0,
        cache_size: int = 12,
        time_column: str = DEFAULT_TIME_COLUMN,
        wheel_columns: Sequence[str] = DEFAULT_WHEEL_COLUMNS,
    ) -> None:
        self.event_manifest = event_manifest.resolve()
        self.normal_manifest = normal_manifest.resolve()
        self.validation_path = validation_path.resolve()
        self.validation = json.loads(self.validation_path.read_text(encoding="utf-8"))
        self.cfg = cfg
        self.max_window_s = max_window_s
        self.time_column = time_column
        self.wheel_columns = tuple(wheel_columns)
        self.cases: dict[str, dict[str, object]] = {}
        self.case_ids: list[str] = []
        self._load_event_cases()
        self._load_normal_cases()
        self._cached_scan = lru_cache(maxsize=cache_size)(self._scan_path)
        self._cached_detail = lru_cache(maxsize=cache_size)(self._render_case_cached)

    def _load_event_cases(self) -> None:
        dataset_root = self.event_manifest.parent
        with self.event_manifest.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("sample_type") != "event" or row.get("is_augmented") != "0":
                    continue
                case_id = row["source_event_id"]
                self.case_ids.append(case_id)
                self.cases[case_id] = {
                    "id": case_id,
                    "sample_type": "event",
                    "scenario": "实车爆胎基线",
                    "name": row["source_file"],
                    "csv_path": (dataset_root / row["sample_file"]).resolve(),
                    "event_time_s": float(row["event_time_in_sample_s"]),
                    "target_wheels": ("RR",),
                }

    def _load_normal_cases(self) -> None:
        result_files = list(
            (WORKSPACE_ROOT / "robust_data_results" / "cases").rglob(
                "wheel_speed_raw_vs_corrected.csv"
            )
        )
        with self.normal_manifest.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                case_id = row["sample_id"]
                csv_path = Path(row["sample_file"])
                if not csv_path.is_file():
                    matches = [
                        path
                        for path in result_files
                        if path.parent.name.startswith(row["source_file"])
                    ]
                    if matches:
                        csv_path = matches[0]
                summary_path = csv_path.parent / "summary.json"
                summary: dict[str, object] = {}
                if summary_path.is_file():
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                self.case_ids.append(case_id)
                self.cases[case_id] = {
                    "id": case_id,
                    "sample_type": "normal",
                    "scenario": row.get("scenario") or "正常道路",
                    "name": row.get("source_file") or case_id,
                    "csv_path": csv_path.resolve(),
                    "event_time_s": None,
                    "target_wheels": (),
                    "frames": int(summary.get("frames", 0)),
                    "duration_s": float(summary.get("duration_s", 0.0)),
                }

    def render_index(self) -> str:
        rows: list[str] = []
        for case_id in self.case_ids:
            case = self.cases[case_id]
            is_event = case["sample_type"] == "event"
            rows.append(
                f"<tr data-type='{case['sample_type']}' "
                f"data-scenario='{html.escape(str(case['scenario']))}'>"
                f"<td><a href='/case/{quote(case_id)}'>{case_id}</a></td>"
                f"<td><a href='/case/{quote(case_id)}'>{html.escape(str(case['name']))}</a></td>"
                f"<td>{html.escape(str(case['scenario']))}</td>"
                f"<td>{'爆胎' if is_event else '正常道路'}</td>"
                f"<td>{'+'.join(case['target_wheels']) or '—'}</td>"
                f"<td><span class='badge {'event' if is_event else 'normal'}'>"
                f"{'正确检出' if is_event else '全量无误报'}</span></td>"
                f"<td><a class='mini-button' href='/case/{quote(case_id)}'>运行并查看</a></td></tr>"
            )
        positive = self.validation["real_positive_replay"]
        augmented = self.validation["augmented_replay"]
        normal = self.validation["real_normal_road_replay"]
        return _page(
            "纯四轮轮速爆胎算法控制台",
            f"""
<header><div><p class='eyebrow'>FOUR-WHEEL SPEED · NO TPMS</p>
<h1>纯四轮轮速爆胎算法控制台</h1>
<p class='muted'>完整因果回放 · 双空间证据 · 候选区间 · 四轮独立锁存</p></div>
<nav><a class='button secondary' href='/report'>算法与验证</a><a class='button' href='/validation.json'>下载验证 JSON</a></nav></header>
<section class='cards'>
  <div class='card accent'><span>真实爆胎检出</span><strong>{positive['detected']}/{positive['samples']}</strong><small>平均延迟 {positive['mean_confirmation_delay_s']:.3f}s</small></div>
  <div class='card'><span>增强事件</span><strong>{augmented['events_detected_within_2s']}/{augmented['event_samples']}</strong><small>2 秒内正确轮位</small></div>
  <div class='card'><span>正常道路误报</span><strong>{normal['false_alarm_cases']}/{normal['cases']}</strong><small>{normal['duration_hours']:.3f} 小时</small></div>
  <div class='card'><span>正常道路帧数</span><strong>{normal['frames']:,}</strong><small>四轮 100 Hz 回放</small></div>
</section>
<section class='panel open-file'><div><h2>打开自己的 CSV</h2><p class='muted'>输入工作区相对路径或绝对路径；页面会从文件开头运行算法以建立因果基线。</p></div>
<form action='/file' method='get'><input name='path' required placeholder='例如 data/wheel_speed.csv'><button>打开并分析</button></form></section>
<section class='panel'><div class='controls'>
  <input id='search' placeholder='搜索 ID、文件或道路场景…'>
  <select id='type'><option value=''>全部类型</option><option value='event'>爆胎</option><option value='normal'>正常道路</option></select>
  <select id='scenario'><option value=''>全部场景</option>{self._scenario_options()}</select>
  <span id='count'></span>
</div><div class='table-wrap'><table><thead><tr><th>ID</th><th>记录</th><th>场景</th><th>类型</th><th>目标轮</th><th>回放结论</th><th>操作</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div></section>
<section class='notice'><b>使用边界：</b>真实正样本目前仅覆盖 RR；相邻两轮或四轮完全等幅同步变化不从相对轮速强判，以控制转弯、轴滑移和制动误报。</section>
<script>{_FILTER_SCRIPT}</script>
""",
        )

    def render_report(self) -> str:
        positive = self.validation["real_positive_replay"]
        augmented = self.validation["augmented_replay"]
        normal = self.validation["real_normal_road_replay"]
        config_rows = "".join(
            f"<tr><td>{html.escape(name)}</td><td>{html.escape(str(value))}</td></tr>"
            for name, value in asdict(self.cfg).items()
        )
        return _page(
            "纯轮速算法与验证",
            f"""
<nav class='top-nav'><a href='/'>← 返回控制台</a></nav>
<header><div><p class='eyebrow'>ALGORITHM & VALIDATION</p><h1>纯轮速算法与验证</h1>
<p class='muted'>{html.escape(str(self.validation['algorithm']))} · {self.validation['evaluation_date']}</p></div>
<nav><a class='button' href='/validation.json'>下载 JSON</a><button onclick='window.print()'>打印 / 导出 PDF</button></nav></header>
<section class='panel report'><h2>判定逻辑</h2><div class='formula'>
individualᵢ = log(wᵢ) − mean(log(w<sub>另一对角</sub>))<br>
diagonal = log(w<sub>FL</sub>) − log(w<sub>FR</sub>) − log(w<sub>RL</sub>) + log(w<sub>RR</sub>)
</div><ol><li>两项证据分别减去滚动正常基线。</li><li>逐轮边沿和按轮位取符号的对角边沿必须同时为正。</li><li>候选持续约 0.7 秒后检查高位占比、同对角伙伴和共同车速瞬变。</li><li>确认后锁存；参考已报警轮的另一对角暂停强判，防止连锁误报。</li></ol></section>
<section class='panel report'><h2>开发回放</h2><div class='cards compact'>
<div class='card'><span>真实 RR 爆胎</span><strong>{positive['detected']}/{positive['samples']}</strong><small>错误轮位/提前报警 {positive['wrong_wheel_or_pre_event_alarms']}</small></div>
<div class='card'><span>平均/最大延迟</span><strong>{positive['mean_confirmation_delay_s']:.3f}/{positive['max_confirmation_delay_s']:.2f}s</strong></div>
<div class='card'><span>增强事件</span><strong>{augmented['events_detected_within_2s']}/{augmented['event_samples']}</strong><small>漏检 {augmented['event_misses']}</small></div>
<div class='card'><span>正常道路</span><strong>0/{normal['cases']}</strong><small>{normal['frames']:,} 帧</small></div></div>
<div class='notice danger'><b>不是独立盲测：</b>参数使用过同源数据开发；FL、FR、RL 真实爆胎和多轮实车场景仍需补齐。</div></section>
<details class='panel report'><summary>当前参数</summary><div class='table-wrap'><table><thead><tr><th>参数</th><th>值</th></tr></thead><tbody>{config_rows}</tbody></table></div></details>
""",
        )

    def render_case(
        self, case_id: str, start_s: float | None, end_s: float | None
    ) -> str:
        if case_id not in self.cases:
            raise KeyError(case_id)
        return self._cached_detail(case_id, start_s, end_s)

    def _render_case_cached(
        self, case_id: str, start_s: float | None, end_s: float | None
    ) -> str:
        case = self.cases[case_id]
        return self._render_detail(case, start_s, end_s, f"/case/{quote(case_id)}")

    def render_custom(
        self, path_value: str, start_s: float | None, end_s: float | None
    ) -> str:
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = WORKSPACE_ROOT / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() != ".csv":
            raise ValueError("控制台只支持 CSV 文件")
        case = {
            "id": "CUSTOM",
            "sample_type": "custom",
            "scenario": "自定义 CSV",
            "name": path.name,
            "csv_path": path,
            "event_time_s": None,
            "target_wheels": (),
        }
        base_url = "/file?" + urlencode({"path": str(path)})
        return self._render_detail(case, start_s, end_s, base_url)

    def _render_detail(
        self,
        case: dict[str, object],
        start_s: float | None,
        end_s: float | None,
        base_url: str,
    ) -> str:
        csv_path: Path = case["csv_path"]  # type: ignore[assignment]
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        scan = self._cached_scan(str(csv_path))
        event = case["event_time_s"]
        if start_s is None and end_s is None:
            if event is not None:
                start_s = max(scan.start_s, float(event) - 2.0)
                end_s = min(scan.end_s, float(event) + 3.0)
            elif scan.suspects:
                start_s = max(scan.start_s, scan.suspects[0].start_s - 2.0)
                end_s = min(scan.end_s, scan.suspects[0].end_s + 2.0)
            else:
                start_s = scan.first_warmed_s or scan.start_s
                end_s = min(scan.end_s, start_s + 30.0)
        if start_s is None or end_s is None or end_s <= start_s:
            raise ValueError("start/end 时间窗口无效")
        start_s = max(scan.start_s, start_s)
        end_s = min(scan.end_s, end_s)
        if end_s - start_s > self.max_window_s:
            raise ValueError(f"单次窗口不能超过 {self.max_window_s:g} 秒")
        data = analyze_window(
            csv_path,
            start_s,
            end_s,
            self.cfg,
            time_column=self.time_column,
            wheel_columns=self.wheel_columns,
        )
        payload = asdict(data)
        payload["suspects"] = [asdict(interval) for interval in scan.suspects]
        plot_data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        marker = "null" if event is None else str(event)
        confirmed = sum(interval.confirmed for interval in scan.suspects)
        first_alarm = ", ".join(
            f"{WHEEL_NAMES[index]} {value:.2f}s"
            for index, value in enumerate(scan.first_alarm_times)
            if value is not None
        ) or "—"
        suspect_rows = self._suspect_rows(scan, base_url, start_s, end_s)
        separator = "&" if "?" in base_url else "?"
        width = end_s - start_s
        previous_start = max(scan.start_s, start_s - width)
        previous_end = min(scan.end_s, previous_start + width)
        next_end = min(scan.end_s, end_s + width)
        next_start = max(scan.start_s, next_end - width)
        return _page(
            str(case["name"]),
            f"""
<nav class='top-nav'><a href='/'>← 返回控制台</a><span>{self._case_navigation(case)}</span></nav>
<header class='case-header'><div><p class='eyebrow'>{html.escape(str(case['id']))} · {html.escape(str(case['scenario']))}</p>
<h1>{html.escape(str(case['name']))}</h1><p class='path'>{html.escape(str(csv_path))}</p></div>
<div class='legend'>{''.join(f"<span><i style='background:{color}'></i>{name}</span>" for name, color in zip(WHEEL_NAMES, WHEEL_COLORS))}</div></header>
<section class='cards compact'>
  <div class='card'><span>完整记录候选</span><strong>{len(scan.suspects)}</strong><small>含已排除候选</small></div>
  <div class='card'><span>最终确认</span><strong>{confirmed}</strong><small>首报 {first_alarm}</small></div>
  <div class='card'><span>记录规模</span><strong>{scan.frames:,}</strong><small>{scan.end_s - scan.start_s:.1f}s · 有效 {scan.valid_frames:,} 帧</small></div>
  <div class='card'><span>当前窗口</span><strong>{start_s:.2f}–{end_s:.2f}s</strong><small>{end_s - start_s:.2f} 秒</small></div>
</section>
<form class='range panel' method='get' action='{html.escape(base_url.split('?')[0])}'>
  {self._hidden_path(base_url)}
  <label>开始/s <input type='number' step='0.01' name='start' value='{start_s:.2f}'></label>
  <label>结束/s <input type='number' step='0.01' name='end' value='{end_s:.2f}'></label>
  <button>查看窗口</button>
  <a class='mini-button' href='{base_url}{separator}start={previous_start:.3f}&end={previous_end:.3f}#plot'>← 前一屏</a>
  <a class='mini-button' href='{base_url}{separator}start={next_start:.3f}&end={next_end:.3f}#plot'>后一屏 →</a>
  <span class='muted'>算法始终从记录开头运行；单窗口最多 {self.max_window_s:g} 秒。</span>
</form>
<div id='readout' class='readout'>移动鼠标到曲线上查看同一时刻的四轮证据</div>
<div class='workbench'>
  <aside><div class='aside-title'><h2>候选区间</h2><span>点击聚焦</span></div>{suspect_rows}</aside>
  <section id='plot' class='charts'>
    <div class='plot-panel'><canvas id='wheelPlot' height='300'></canvas></div>
    <div class='plot-panel'><canvas id='gainPlot' height='300'></canvas></div>
    <div class='plot-panel'><canvas id='edgePlot' height='300'></canvas></div>
    <div class='plot-panel'><canvas id='alarmPlot' height='300'></canvas></div>
  </section>
</div>
<section class='notice'><b>图例：</b>彩色背景为算法进入候选的区间，实边框表示最终确认；逐轮证据为实线、对角证据为虚线。边沿图中的水平虚线是候选门限。</section>
<script>const D={plot_data};const COLORS={json.dumps(WHEEL_COLORS)};const N={json.dumps(WHEEL_NAMES)};const eventTime={marker};const CFG={json.dumps(asdict(self.cfg))};{_CHART_SCRIPT}</script>
""",
        )

    def _scan_path(self, path_text: str) -> ScanResult:
        return scan_csv(
            Path(path_text),
            self.cfg,
            time_column=self.time_column,
            wheel_columns=self.wheel_columns,
        )

    def _suspect_rows(
        self,
        scan: ScanResult,
        base_url: str,
        start_s: float,
        end_s: float,
    ) -> str:
        separator = "&" if "?" in base_url else "?"
        rows: list[str] = []
        for number, interval in enumerate(scan.suspects, start=1):
            focus_start = max(scan.start_s, interval.start_s - 2.0)
            focus_end = min(scan.end_s, interval.end_s + 2.0)
            selected = interval.end_s >= start_s and interval.start_s <= end_s
            rows.append(
                f"<a class='suspect {'selected' if selected else ''}' "
                f"style='border-left-color:{WHEEL_COLORS[interval.wheel_index]}' "
                f"href='{base_url}{separator}start={focus_start:.3f}&end={focus_end:.3f}#plot'>"
                f"<span class='suspect-head'><b>#{number} · {WHEEL_NAMES[interval.wheel_index]}</b>"
                f"<em class='{'ok' if interval.confirmed else ''}'>{'已确认' if interval.confirmed else '已排除'}</em></span>"
                f"<span>{interval.start_s:.3f}–{interval.end_s:.3f}s · {interval.duration_s:.3f}s</span>"
                f"<small>逐轮峰值 {self._pct(interval.peak_individual_gain_pct)}　对角峰值 {self._pct(interval.peak_diagonal_gain_pct)}</small></a>"
            )
        if not rows:
            return "<div class='empty'><b>没有候选</b><span>完整记录从未进入 candidate 状态。</span></div>"
        return "".join(rows)

    def _case_navigation(self, case: dict[str, object]) -> str:
        case_id = str(case["id"])
        if case_id not in self.case_ids:
            return ""
        position = self.case_ids.index(case_id)
        previous = (
            ""
            if position == 0
            else f"<a class='button secondary' href='/case/{quote(self.case_ids[position - 1])}'>← 上一条</a>"
        )
        following = (
            ""
            if position + 1 == len(self.case_ids)
            else f"<a class='button secondary' href='/case/{quote(self.case_ids[position + 1])}'>下一条 →</a>"
        )
        return previous + following

    @staticmethod
    def _hidden_path(base_url: str) -> str:
        parsed = urlparse(base_url)
        path = parse_qs(parsed.query).get("path", [None])[0]
        return "" if path is None else f"<input type='hidden' name='path' value='{html.escape(path)}'>"

    @staticmethod
    def _pct(value: float | None) -> str:
        return "—" if value is None else f"{value:.3f}%"

    def _scenario_options(self) -> str:
        scenarios = sorted({str(case["scenario"]) for case in self.cases.values()})
        return "".join(
            f"<option value='{html.escape(value)}'>{html.escape(value)}</option>"
            for value in scenarios
        )


class ConsoleHandler(BaseHTTPRequestHandler):
    state: ConsoleState

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            if parsed.path == "/":
                self._send_html(self.state.render_index())
                return
            if parsed.path == "/report":
                self._send_html(self.state.render_report())
                return
            if parsed.path == "/validation.json":
                self._send_json(self.state.validation)
                return
            if parsed.path.startswith("/case/"):
                case_id = unquote(parsed.path.removeprefix("/case/"))
                self._send_html(
                    self.state.render_case(
                        case_id,
                        self._optional_float(query, "start"),
                        self._optional_float(query, "end"),
                    )
                )
                return
            if parsed.path == "/file":
                path_value = query.get("path", [""])[0]
                if not path_value:
                    raise ValueError("缺少 CSV 路径")
                self._send_html(
                    self.state.render_custom(
                        path_value,
                        self._optional_float(query, "start"),
                        self._optional_float(query, "end"),
                    )
                )
                return
            self.send_error(404)
        except (KeyError, ValueError, FileNotFoundError, OSError) as error:
            self.send_error(400, str(error))

    @staticmethod
    def _optional_float(query: dict[str, list[str]], name: str) -> float | None:
        value = query.get(name, [None])[0]
        return None if value is None else float(value)

    def _send_html(self, body: str) -> None:
        payload = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, value: object) -> None:
        payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return


def _page(title: str, body: str) -> str:
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{html.escape(title)}</title><style>{_STYLE}</style></head><body><main>{body}</main></body></html>"""


_FILTER_SCRIPT = """
const rows=[...document.querySelectorAll('tbody tr')],search=document.querySelector('#search'),type=document.querySelector('#type'),scenario=document.querySelector('#scenario'),count=document.querySelector('#count');
function apply(){const q=search.value.toLowerCase();let visible=0;rows.forEach(row=>{const ok=(!q||row.innerText.toLowerCase().includes(q))&&(!type.value||row.dataset.type===type.value)&&(!scenario.value||row.dataset.scenario===scenario.value);row.hidden=!ok;if(ok)visible++;});count.textContent=`显示 ${visible}/${rows.length}`;}[search,type,scenario].forEach(item=>item.addEventListener('input',apply));apply();
"""


_CHART_SCRIPT = r"""
let hoverIndex=null,raf=null;
function finiteValues(series){const out=[];series.forEach(row=>row.forEach(v=>{if(v!==null&&Number.isFinite(v))out.push(v);}));return out;}
function nice(v){if(!Number.isFinite(v))return '—';const a=Math.abs(v);return a>=100?v.toFixed(0):a>=10?v.toFixed(1):a>=1?v.toFixed(2):v.toFixed(3);}
function drawChart(id,title,series,labels,colors,unit,opts={}){
 const canvas=document.getElementById(id),ratio=window.devicePixelRatio||1,width=Math.max(640,canvas.clientWidth||1100),height=300;
 canvas.width=width*ratio;canvas.height=height*ratio;canvas.style.height=height+'px';const c=canvas.getContext('2d');c.scale(ratio,ratio);
 const m={l:68,r:22,t:55,b:40},pw=width-m.l-m.r,ph=height-m.t-m.b,x0=D.times[0],x1=D.times[D.times.length-1];
 let ys=finiteValues(series),y0=opts.yMin??Math.min(...ys),y1=opts.yMax??Math.max(...ys);if(!Number.isFinite(y0)||!Number.isFinite(y1)){y0=0;y1=1;}if(y1-y0<1e-9){y0-=.5;y1+=.5;}const pad=(y1-y0)*.08;y0-=opts.noPad?0:pad;y1+=opts.noPad?0:pad;
 const X=x=>m.l+(x-x0)/Math.max(1e-9,x1-x0)*pw,Y=y=>m.t+(y1-y)/(y1-y0)*ph;
 c.fillStyle='#fff';c.fillRect(0,0,width,height);
 D.suspects.forEach(s=>{if(s.end_s<x0||s.start_s>x1)return;const xa=X(Math.max(x0,s.start_s)),xb=X(Math.min(x1,s.end_s));c.fillStyle=COLORS[s.wheel_index]+(s.confirmed?'2d':'18');c.fillRect(xa,m.t,Math.max(1,xb-xa),ph);c.strokeStyle=COLORS[s.wheel_index]+(s.confirmed?'cc':'66');c.setLineDash(s.confirmed?[]:[4,3]);c.strokeRect(xa,m.t,Math.max(1,xb-xa),ph);c.setLineDash([]);});
 c.font='600 15px system-ui';c.fillStyle='#16202a';c.fillText(title,m.l,24);c.font='12px system-ui';
 for(let k=0;k<=4;k++){const y=m.t+ph*k/4,v=y1-(y1-y0)*k/4;c.strokeStyle='#e5e7eb';c.lineWidth=1;c.beginPath();c.moveTo(m.l,y);c.lineTo(width-m.r,y);c.stroke();c.fillStyle='#64748b';c.textAlign='right';c.fillText(nice(v),m.l-8,y+4);}
 c.textAlign='center';for(let k=0;k<=5;k++){const x=m.l+pw*k/5,v=x0+(x1-x0)*k/5;c.fillStyle='#64748b';c.fillText(v.toFixed(2),x,height-14);}c.save();c.translate(15,m.t+ph/2);c.rotate(-Math.PI/2);c.fillText(unit,0,0);c.restore();
 (opts.hLines||[]).forEach(h=>{const y=Y(h.value);if(y<m.t||y>m.t+ph)return;c.strokeStyle=h.color;c.setLineDash([5,4]);c.beginPath();c.moveTo(m.l,y);c.lineTo(width-m.r,y);c.stroke();c.setLineDash([]);c.fillStyle=h.color;c.textAlign='right';c.fillText(h.label,width-m.r-4,y-4);});
 if(eventTime!==null&&eventTime>=x0&&eventTime<=x1){const x=X(eventTime);c.strokeStyle='#7c3aed';c.setLineDash([6,4]);c.lineWidth=2;c.beginPath();c.moveTo(x,m.t);c.lineTo(x,m.t+ph);c.stroke();c.setLineDash([]);c.fillStyle='#7c3aed';c.textAlign='left';c.fillText('人工事件',x+4,m.t+12);}
 const stride=Math.max(1,Math.ceil(D.times.length/5000));series.forEach((row,j)=>{c.strokeStyle=colors[j];c.lineWidth=opts.widths?opts.widths[j]:1.4;c.setLineDash(opts.dashes?opts.dashes[j]:[]);c.beginPath();let drawing=false;for(let i=0;i<row.length;i+=stride){const v=row[i];if(v===null||!Number.isFinite(v)){drawing=false;continue;}const x=X(D.times[i]),y=Y(v);if(!drawing){c.moveTo(x,y);drawing=true;}else c.lineTo(x,y);}c.stroke();c.setLineDash([]);});
 let lx=m.l;c.textAlign='left';labels.forEach((label,j)=>{if(lx>width-130)return;c.strokeStyle=colors[j];c.setLineDash(opts.dashes?opts.dashes[j]:[]);c.lineWidth=2;c.beginPath();c.moveTo(lx,38);c.lineTo(lx+14,38);c.stroke();c.setLineDash([]);c.fillStyle='#374151';c.fillText(label,lx+19,42);lx+=c.measureText(label).width+48;});
 if(hoverIndex!==null&&hoverIndex<D.times.length){const x=X(D.times[hoverIndex]);c.strokeStyle='#0f172a88';c.lineWidth=1;c.setLineDash([2,2]);c.beginPath();c.moveTo(x,m.t);c.lineTo(x,m.t+ph);c.stroke();c.setLineDash([]);}
 canvas.onmousemove=e=>{const rect=canvas.getBoundingClientRect(),x=e.clientX-rect.left;hoverIndex=Math.max(0,Math.min(D.times.length-1,Math.round((x-m.l)/pw*(D.times.length-1))));schedule();};canvas.onmouseleave=()=>{hoverIndex=null;schedule();};
}
function renderAll(){
 const diag=[D.diagonal_gains[0],D.diagonal_gains[1]],diagEdge=[D.diagonal_edges[0],D.diagonal_edges[1]],dash4=[[],[],[],[]],dash2=[[6,4],[6,4]];
 drawChart('wheelPlot','四轮校正轮速',D.wheels,N,COLORS,'rad/s');
 drawChart('gainPlot','持续证据：逐轮增益（实线）与对角增益（虚线）',[...D.individual_gains,...diag],[...N.map(n=>n+' individual'),'D FL+RR','D FR+RL'],[...COLORS,'#7c3aed','#0f766e'],'%',{dashes:[...dash4,...dash2],hLines:[{value:CFG.min_individual_persistence*100,color:'#64748b',label:'持续门限'}]});
 drawChart('edgePlot','触发证据：逐轮边沿（实线）与对角边沿（虚线）',[...D.individual_edges,...diagEdge],[...N.map(n=>n+' edge'),'D FL+RR','D FR+RL'],[...COLORS,'#7c3aed','#0f766e'],'%',{dashes:[...dash4,...dash2],hLines:[{value:CFG.min_individual_edge*100,color:'#475569',label:'候选门限'}]});
 const rows=[],labels=[],colors=[],widths=[];N.forEach((n,i)=>{rows.push(D.candidates[i].map(v=>v?i+1:null));labels.push(n+' candidate');colors.push(COLORS[i]+'88');widths.push(3);rows.push(D.alarms[i].map(v=>v?i+1:null));labels.push(n+' alarm');colors.push(COLORS[i]);widths.push(7);});
 drawChart('alarmPlot','四轮候选与锁存报警',rows,labels,colors,'轮位',{yMin:.5,yMax:4.5,noPad:true,widths});
 const readout=document.getElementById('readout');if(hoverIndex===null){readout.textContent='移动鼠标到曲线上查看同一时刻的四轮证据';}else{const i=hoverIndex,items=N.map((n,w)=>`${n} 轮速 ${nice(D.wheels[w][i])} · 增益 ${nice(D.individual_gains[w][i])}% · 对角 ${nice(D.diagonal_gains[w][i])}% · ${D.states[w][i]}`);readout.innerHTML=`<b>t=${D.times[i].toFixed(3)}s</b><span>${items.join('</span><span>')}</span>`;}
}
function schedule(){if(raf)return;raf=requestAnimationFrame(()=>{raf=null;renderAll();});}renderAll();let timer;window.addEventListener('resize',()=>{clearTimeout(timer);timer=setTimeout(renderAll,120);});
"""


_STYLE = """
:root{--ink:#17212b;--muted:#657286;--line:#dde4ec;--blue:#1d4ed8;--green:#15803d;--bg:#f4f7fa}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 Inter,system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1560px;margin:auto;padding:28px}a{text-decoration:none}header{display:flex;justify-content:space-between;gap:22px;align-items:flex-start;margin-bottom:18px}h1{margin:0 0 5px;font-size:29px;letter-spacing:-.02em}h2{margin:0 0 5px}.eyebrow{margin:0 0 6px;color:#1d4ed8;font-size:11px;font-weight:800;letter-spacing:.14em}.muted,.path{color:var(--muted)}.path{max-width:950px;word-break:break-all}nav,.top-nav span{display:flex;gap:8px;flex-wrap:wrap}.top-nav{justify-content:space-between;margin-bottom:16px}.top-nav>a{color:var(--blue);font-weight:700}.button,button{display:inline-block;border:0;border-radius:8px;background:var(--blue);color:white;padding:9px 14px;cursor:pointer;font:inherit}.secondary{background:#64748b}.cards{display:grid;grid-template-columns:repeat(4,minmax(160px,1fr));gap:12px;margin:16px 0}.cards.compact{grid-template-columns:repeat(4,minmax(140px,1fr))}.card,.panel,.plot-panel,.notice,.readout,details,aside{background:#fff;border:1px solid var(--line);border-radius:11px}.card{padding:16px}.card.accent{border-top:3px solid var(--blue)}.card span,.card small{display:block;color:var(--muted)}.card strong{display:block;font-size:24px;margin:3px 0}.panel,.notice,details{padding:18px;margin:14px 0}.open-file{display:flex;align-items:center;justify-content:space-between;gap:20px}.open-file form{display:flex;gap:8px;min-width:min(600px,55%)}.open-file input{flex:1}.controls,.range{display:flex;gap:10px;align-items:center;flex-wrap:wrap}input,select{border:1px solid #c8d2df;border-radius:7px;padding:9px 10px;background:white;font:inherit}.controls input{min-width:280px}.range label{display:flex;align-items:center;gap:5px}.range input{width:105px}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;margin-top:12px}th,td{padding:9px 10px;border-bottom:1px solid #e8edf2;text-align:left;white-space:nowrap}th{background:#f8fafc;position:sticky;top:0}td a{color:var(--blue);font-weight:650}.mini-button{display:inline-block;background:#e4e9ff;color:#3730a3!important;border-radius:7px;padding:5px 9px}.badge{display:inline-block;border-radius:999px;padding:3px 9px;font-weight:700}.badge.event{background:#dbeafe;color:#1d4ed8}.badge.normal{background:#dcfce7;color:#166534}.notice{background:#fffbeb;border-color:#fde68a}.danger{background:#fff1f2;border-color:#fecdd3}.formula{font:15px/2 ui-monospace,SFMono-Regular,Consolas,monospace;background:#f8fafc;border:1px solid var(--line);padding:14px;border-radius:8px}.report{max-width:1120px}.legend{display:grid;grid-template-columns:repeat(2,auto);gap:5px 16px;background:white;border:1px solid var(--line);border-radius:9px;padding:10px 13px}.legend span{display:flex;align-items:center;gap:7px}.legend i{width:13px;height:3px;border-radius:2px}.workbench{display:grid;grid-template-columns:300px minmax(0,1fr);gap:14px;align-items:start}aside{padding:12px;position:sticky;top:12px;max-height:calc(100vh - 24px);overflow:auto}.aside-title{display:flex;justify-content:space-between;align-items:baseline;padding:3px 5px 10px}.aside-title span{font-size:12px;color:var(--muted)}.suspect{display:block;color:var(--ink);border:1px solid #e3e8ef;border-left:4px solid #94a3b8;border-radius:8px;padding:9px 10px;margin:0 0 8px;background:#fff}.suspect:hover,.suspect.selected{border-color:#93b4e8;background:#eff6ff}.suspect-head{display:flex!important;justify-content:space-between;align-items:center}.suspect span,.suspect small{display:block}.suspect small{color:var(--muted);margin-top:3px}.suspect em{font-style:normal;font-size:11px;background:#f1f5f9;color:#64748b;padding:2px 6px;border-radius:999px}.suspect em.ok{background:#dcfce7;color:#166534}.empty{padding:26px 8px;text-align:center;color:var(--muted)}.empty b,.empty span{display:block}.charts{min-width:0}.plot-panel{padding:4px 9px;margin-bottom:12px;overflow:hidden}.readout{position:sticky;top:0;z-index:3;margin:12px 0;padding:10px 14px;box-shadow:0 3px 12px #0f172a0d;display:flex;gap:16px;flex-wrap:wrap}.readout span{color:#475569}details summary{font-size:17px;font-weight:700;cursor:pointer}@media(max-width:900px){main{padding:15px}.cards,.cards.compact{grid-template-columns:1fr 1fr}.workbench{grid-template-columns:1fr}aside{position:static;max-height:340px}header,.open-file{display:block}.open-file form{min-width:100%;margin-top:12px}.legend{margin-top:12px;width:max-content}}@media print{body{background:#fff}nav,.controls,.range,.open-file,aside{display:none}.panel,.card,.notice{break-inside:avoid}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Four-wheel-speed blowout console")
    parser.add_argument("--event-manifest", type=Path, default=DEFAULT_EVENT_MANIFEST)
    parser.add_argument("--normal-manifest", type=Path, default=DEFAULT_NORMAL_MANIFEST)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--time-column", default=DEFAULT_TIME_COLUMN)
    parser.add_argument("--wheel-columns", nargs=4, default=DEFAULT_WHEEL_COLUMNS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8772)
    parser.add_argument("--max-window-s", type=float, default=120.0)
    args = parser.parse_args()
    if args.max_window_s <= 0:
        parser.error("--max-window-s must be positive")
    state = ConsoleState(
        args.event_manifest,
        args.normal_manifest,
        args.validation,
        WheelSpeedBlowoutConfig(),
        max_window_s=args.max_window_s,
        time_column=args.time_column,
        wheel_columns=args.wheel_columns,
    )
    ConsoleHandler.state = state
    server = ThreadingHTTPServer((args.host, args.port), ConsoleHandler)
    print(f"Wheel-speed-only blowout console: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
