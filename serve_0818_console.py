from __future__ import annotations

import argparse
import bisect
import html
import json
from dataclasses import dataclass
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Sequence
from urllib.parse import parse_qs, quote, unquote, urlencode, urlparse

from build_0818_display import CaseAnalysis, WHEEL_NAMES, analyze_file


WORKSPACE_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = WORKSPACE_ROOT / "0818"
WHEEL_COLORS = ("#2563eb", "#ea8a00", "#16a34a", "#dc2626")
ALGORITHMS = {"compare": "双算法对比", "wheel": "wheel_only", "quant": "quant"}


@dataclass(frozen=True)
class CandidateInterval:
    algorithm: str
    wheel: int
    start_s: float
    end_s: float
    confirmed: bool

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def candidate_intervals(data: CaseAnalysis) -> tuple[CandidateInterval, ...]:
    intervals: list[CandidateInterval] = []
    for algorithm, candidates, alarms in (
        ("wheel", data.wheel_candidates, data.wheel_alarms),
        ("quant", data.quant_candidates, data.quant_alarms),
    ):
        for wheel in range(4):
            start: int | None = None
            for index, active in enumerate(candidates[wheel]):
                if active and start is None:
                    start = index
                if not active and start is not None:
                    intervals.append(
                        CandidateInterval(
                            algorithm,
                            wheel,
                            data.times[start],
                            data.times[index],
                            any(alarms[wheel][start : index + 1]),
                        )
                    )
                    start = None
            if start is not None:
                intervals.append(
                    CandidateInterval(
                        algorithm,
                        wheel,
                        data.times[start],
                        data.times[-1],
                        any(alarms[wheel][start:]),
                    )
                )
    return tuple(
        sorted(intervals, key=lambda item: (item.start_s, item.algorithm, item.wheel))
    )


def _alarm_text(values: Sequence[float | None], event: float | None) -> str:
    parts = []
    for wheel, alarm in enumerate(values):
        if alarm is None:
            continue
        delay = "" if event is None else f" / {alarm - event:+.2f}s"
        parts.append(f"{WHEEL_NAMES[wheel]} {alarm:.2f}s{delay}")
    return "、".join(parts) if parts else "未报警"


class ConsoleState:
    def __init__(self, input_dir: Path, *, max_window_s: float = 120.0) -> None:
        self.input_dir = input_dir.resolve()
        self.max_window_s = max_window_s
        paths = sorted(self.input_dir.glob("*.txt"))
        if not paths:
            raise ValueError(f"目录内没有 txt 数据：{self.input_dir}")
        self.case_ids = [path.stem for path in paths]
        self.paths = {path.stem: path for path in paths}
        self._analysis = lru_cache(maxsize=len(paths))(self._analyze)

    def _analyze(self, case_id: str) -> CaseAnalysis:
        if case_id not in self.paths:
            raise KeyError(case_id)
        return analyze_file(self.paths[case_id])

    def analyze(self, case_id: str) -> CaseAnalysis:
        return self._analysis(case_id)

    def render_index(self) -> str:
        analyses = [self.analyze(case_id) for case_id in self.case_ids]
        wheel_hits = sum(
            any(value is not None for value in data.wheel_first_alarms)
            for data in analyses
        )
        quant_hits = sum(
            any(value is not None for value in data.quant_first_alarms)
            for data in analyses
        )
        rows = []
        for data in analyses:
            case_id = data.input_path.stem
            event = data.signal_event_time_s
            rows.append(
                f"<tr><td><a href='/case/{quote(case_id)}'>{html.escape(case_id)}</a>"
                f"<small class='cell-sub'>{html.escape(data.input_path.name)}</small></td>"
                f"<td>{len(data.times):,}</td><td>{data.duration_s:.2f}s</td>"
                f"<td>{'—' if event is None else f'{event:.2f}s'}</td>"
                f"<td>{html.escape(_alarm_text(data.wheel_first_alarms, event))}</td>"
                f"<td>{html.escape(_alarm_text(data.quant_first_alarms, event))}</td>"
                f"<td><a class='mini-button' href='/case/{quote(case_id)}'>运行并查看</a></td></tr>"
            )
        return _page(
            "0818 爆胎双算法控制台",
            f"""
<header><div><p class='eyebrow'>0818 · WHEEL SPEED REPLAY</p><h1>0818 爆胎双算法控制台</h1>
<p class='muted'>原始齿信号重建 · wheel_only / quant · 每帧末值真值 · 完整因果回放</p></div>
<nav><a class='button' href='/summary.json'>下载回放摘要</a></nav></header>
<section class='cards'>
 <div class='card accent'><span>新采记录</span><strong>{len(analyses)}</strong><small>{sum(len(data.times) for data in analyses):,} 帧</small></div>
 <div class='card'><span>wheel_only 报警</span><strong>{wheel_hits}/{len(analyses)}</strong><small>当前默认参数</small></div>
 <div class='card'><span>quant 报警</span><strong>{quant_hits}/{len(analyses)}</strong><small>当前默认参数</small></div>
 <div class='card'><span>爆胎轮位</span><strong>RR</strong><small>信号位来自每帧末值</small></div>
</section>
<section class='panel'><div class='controls'><input id='search' placeholder='搜索记录…'><span id='count'></span></div>
<div class='table-wrap'><table><thead><tr><th>记录</th><th>帧数</th><th>时长</th><th>信号时刻</th><th>wheel_only</th><th>quant</th><th>操作</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div></section>
<section class='notice'><b>事件口径：</b>曲线逐帧显示原始 0/1 信号；红色事件线取第一段连续至少 20 帧的高电平，避免 Brk 文件开头 5 帧残留高电平干扰定位。</section>
<script>{_FILTER_SCRIPT}</script>
""",
        )

    def summary(self) -> dict[str, object]:
        cases = []
        for case_id in self.case_ids:
            data = self.analyze(case_id)
            cases.append(
                {
                    "case": case_id,
                    "frames": len(data.times),
                    "duration_s": data.duration_s,
                    "signal_event_time_s": data.signal_event_time_s,
                    "wheel_only_first_alarms_s": dict(
                        zip(WHEEL_NAMES, data.wheel_first_alarms)
                    ),
                    "quant_first_alarms_s": dict(
                        zip(WHEEL_NAMES, data.quant_first_alarms)
                    ),
                }
            )
        return {"input_dir": str(self.input_dir), "cases": cases}

    def render_case(
        self,
        case_id: str,
        start_s: float | None,
        end_s: float | None,
        algorithm: str,
    ) -> str:
        if algorithm not in ALGORITHMS:
            raise ValueError("algorithm 必须是 compare、wheel 或 quant")
        data = self.analyze(case_id)
        event = data.signal_event_time_s
        if start_s is None and end_s is None:
            focus = event if event is not None else 0.0
            start_s = max(0.0, focus - 3.0)
            end_s = min(data.times[-1], focus + 15.0)
        if start_s is None or end_s is None or end_s <= start_s:
            raise ValueError("start/end 时间窗口无效")
        start_s = max(data.times[0], start_s)
        end_s = min(data.times[-1], end_s)
        if end_s - start_s > self.max_window_s:
            raise ValueError(f"单次窗口不能超过 {self.max_window_s:g} 秒")
        left = bisect.bisect_left(data.times, start_s)
        right = bisect.bisect_right(data.times, end_s)
        if left >= right:
            raise ValueError("所选窗口没有数据")
        payload = self._window_payload(data, left, right)
        intervals = candidate_intervals(data)
        sidebar = self._candidate_sidebar(
            case_id, intervals, algorithm, start_s, end_s, event
        )
        options = "".join(
            f"<option value='{value}'{' selected' if algorithm == value else ''}>{label}</option>"
            for value, label in ALGORITHMS.items()
        )
        navigation = self._case_navigation(case_id)
        event_text = "—" if event is None else f"{event:.2f}s"
        return _page(
            f"0818 回放 · {case_id}",
            f"""
<nav class='top-nav'><a href='/'>← 返回控制台</a><span>{navigation}</span></nav>
<header><div><p class='eyebrow'>CAUSAL REPLAY · {html.escape(ALGORITHMS[algorithm])}</p><h1>{html.escape(case_id)}</h1>
<p class='path'>{html.escape(str(data.input_path))}</p></div></header>
<section class='cards compact'>
 <div class='card accent'><span>持续爆胎信号</span><strong>{event_text}</strong><small>原始信号仍逐帧显示</small></div>
 <div class='card'><span>wheel_only</span><strong>{html.escape(_alarm_text(data.wheel_first_alarms, event))}</strong></div>
 <div class='card'><span>quant</span><strong>{html.escape(_alarm_text(data.quant_first_alarms, event))}</strong></div>
 <div class='card'><span>当前窗口</span><strong>{start_s:.2f}–{end_s:.2f}s</strong><small>{right-left:,} 帧</small></div>
</section>
<form class='range panel' method='get' action='/case/{quote(case_id)}'>
 <label>算法 <select name='algorithm'>{options}</select></label>
 <label>开始 <input name='start' type='number' step='0.01' value='{start_s:.2f}'></label>
 <label>结束 <input name='end' type='number' step='0.01' value='{end_s:.2f}'></label>
 <button>查看窗口</button><span class='muted'>单次最多 {self.max_window_s:g} 秒</span>
</form>
<div class='workbench'><aside>{sidebar}</aside><div class='charts'>
<div class='readout' id='readout'>在 Plotly 图上悬停，查看同一时刻的四轮证据</div>
<section class='plot-panel'><div id='plot'></div></section></div></div>
<script>const D={json.dumps(payload, ensure_ascii=False, separators=(',', ':'))};const MODE={json.dumps(algorithm)};const EVENT={json.dumps(event)};const N={json.dumps(WHEEL_NAMES)};const COLORS={json.dumps(WHEEL_COLORS)};{_CHART_SCRIPT}</script>
""",
        )

    @staticmethod
    def _window_payload(
        data: CaseAnalysis, left: int, right: int
    ) -> dict[str, object]:
        def cut(rows: Sequence[Sequence[object]]) -> list[list[object]]:
            return [list(row[left:right]) for row in rows]

        return {
            "times": data.times[left:right],
            "wheels": cut(data.wheel_speeds),
            "signal": [int(value) for value in data.blowout_signal[left:right]],
            "wheel_gains": cut(data.wheel_individual_gains_pct),
            "wheel_edges": cut(data.wheel_individual_edges_pct),
            "wheel_diagonal": cut(data.wheel_diagonal_gains_pct),
            "wheel_diagonal_edges": cut(data.wheel_diagonal_edges_pct),
            "wheel_candidates": cut(data.wheel_candidates),
            "wheel_alarms": cut(data.wheel_alarms),
            "quant_factor_residuals": cut(data.quant_factor_residuals_pct),
            "quant_factor_edges": cut(data.quant_factor_edges_pct),
            "quant_physical": cut(data.quant_physical_levels_pct),
            "quant_physical_edges": cut(data.quant_physical_edges_pct),
            "quant_shock_z": cut(data.quant_shock_z_scores),
            "quant_level_z": cut(data.quant_level_z_scores),
            "quant_shock_isolation": cut(data.quant_shock_isolation),
            "quant_level_isolation": cut(data.quant_level_isolation),
            "quant_cusum": cut(data.quant_cusum_scores),
            "quant_persistence": cut(data.quant_persistence_scores),
            "quant_risk": cut(data.quant_risk_scores),
            "quant_states": cut(data.quant_states),
            "quant_leading_wheels": data.quant_leading_wheels[left:right],
            "quant_leading_margins": data.quant_leading_margins[left:right],
            "quant_candidates": cut(data.quant_candidates),
            "quant_alarms": cut(data.quant_alarms),
            "wheel_first_alarms": list(data.wheel_first_alarms),
            "quant_first_alarms": list(data.quant_first_alarms),
        }

    def _candidate_sidebar(
        self,
        case_id: str,
        intervals: Sequence[CandidateInterval],
        algorithm: str,
        start_s: float,
        end_s: float,
        event: float | None,
    ) -> str:
        visible = [
            item
            for item in intervals
            if algorithm == "compare" or item.algorithm == algorithm
        ]
        rows = []
        if event is not None:
            target = self._window_url(
                case_id, max(0.0, event - 3), event + 15, algorithm
            )
            rows.append(
                f"<a class='suspect event-link' href='{target}'><span class='suspect-head'><b>爆胎信号</b><em class='signal'>真值</em></span>"
                f"<span>{event:.2f}s 持续置 1</span><small>跳转到事件窗口</small></a>"
            )
        for item in visible:
            target = self._window_url(
                case_id,
                max(0.0, item.start_s - 2.0),
                min(self.analyze(case_id).times[-1], item.end_s + 3.0),
                algorithm,
            )
            selected = not (item.end_s < start_s or item.start_s > end_s)
            status = "已确认" if item.confirmed else "已排除"
            rows.append(
                f"<a class='suspect{' selected' if selected else ''}' href='{target}' style='border-left-color:{WHEEL_COLORS[item.wheel]}'>"
                f"<span class='suspect-head'><b>{html.escape(ALGORITHMS[item.algorithm])} · {WHEEL_NAMES[item.wheel]}</b>"
                f"<em class='{'ok' if item.confirmed else ''}'>{status}</em></span>"
                f"<span>{item.start_s:.2f}–{item.end_s:.2f}s</span><small>候选持续 {item.duration_s:.2f}s</small></a>"
            )
        if not rows:
            rows.append(
                "<div class='empty'><b>没有候选</b><span>所选算法未进入 candidate。</span></div>"
            )
        return (
            f"<div class='aside-title'><b>信号与候选</b><span>{len(visible)} 个候选</span></div>"
            + "".join(rows)
        )

    @staticmethod
    def _window_url(case_id: str, start: float, end: float, algorithm: str) -> str:
        return f"/case/{quote(case_id)}?" + urlencode(
            {
                "start": f"{start:.2f}",
                "end": f"{end:.2f}",
                "algorithm": algorithm,
            }
        )

    def _case_navigation(self, case_id: str) -> str:
        position = self.case_ids.index(case_id)
        previous = (
            ""
            if position == 0
            else f"<a class='button secondary' href='/case/{quote(self.case_ids[position-1])}'>← 上一条</a>"
        )
        following = (
            ""
            if position + 1 == len(self.case_ids)
            else f"<a class='button secondary' href='/case/{quote(self.case_ids[position+1])}'>下一条 →</a>"
        )
        return previous + following


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
            if parsed.path == "/summary.json":
                self._send_json(self.state.summary())
                return
            if parsed.path.startswith("/case/"):
                case_id = unquote(parsed.path.removeprefix("/case/"))
                self._send_html(
                    self.state.render_case(
                        case_id,
                        self._optional_float(query, "start"),
                        self._optional_float(query, "end"),
                        query.get("algorithm", ["compare"])[0],
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
<meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title>
<script src='https://cdn.plot.ly/plotly-3.7.0.min.js'></script>
<style>{_STYLE}</style></head><body><main>{body}</main></body></html>"""


_FILTER_SCRIPT = """
const rows=[...document.querySelectorAll('tbody tr')],search=document.querySelector('#search'),count=document.querySelector('#count');
function apply(){const q=search.value.toLowerCase();let visible=0;rows.forEach(row=>{const ok=!q||row.innerText.toLowerCase().includes(q);row.hidden=!ok;if(ok)visible++;});count.textContent=`显示 ${visible}/${rows.length}`;}search.addEventListener('input',apply);apply();
"""


_CHART_SCRIPT = r"""
const traces=[],titles=['四轮相位校正轮速 / rad·s⁻¹'],FACTOR_NAMES=['左右 s','前后 a','对角 d'],FACTOR_COLORS=['#0891b2','#7c3aed','#db2777'];
let wheelGainRow=null,wheelEdgeRow=null,wheelStateRow=null,quantFactorRow=null,quantPhysicalRow=null,quantZRow=null,quantIsolationRow=null,quantRiskRow=null,quantStateRow=null;
if(MODE!=='quant'){wheelGainRow=titles.push('wheel_only：持续证据（逐轮实线 / 对角虚线）(%)');wheelEdgeRow=titles.push('wheel_only：触发证据（逐轮实线 / 对角虚线）(%)');wheelStateRow=titles.push('wheel_only：候选与锁存报警')}
if(MODE!=='wheel'){quantFactorRow=titles.push('quant：Hadamard 因子残差（实线）/ 瞬时边沿（虚线）');quantPhysicalRow=titles.push('quant：物理指纹投影（实线=level / 虚线=edge）');quantZRow=titles.push('quant：协方差标准化匹配分（实线=shock / 虚线=level）');quantIsolationRow=titles.push('quant：轮位隔离度（实线=shock / 虚线=level）');quantRiskRow=titles.push('quant：风险分（hover 含 CUSUM / persistence）');quantStateRow=titles.push('quant：候选与锁存报警')}
const rowCount=titles.length;
function axes(row){const suffix=row===1?'':String(row);return {xaxis:'x'+suffix,yaxis:'y'+suffix}}
function add(row,y,name,color,opts={}){traces.push({type:'scattergl',mode:'lines',x:D.times,y,name,...axes(row),legendgroup:opts.legendgroup||name,showlegend:opts.showlegend??false,connectgaps:false,line:{color,width:opts.width||1.25,dash:opts.dash||'solid'},customdata:opts.customdata,hovertemplate:opts.hovertemplate||`${name}: %{y:.3f}<extra></extra>`})}
N.forEach((name,w)=>add(1,D.wheels[w],name,COLORS[w],{showlegend:true,legendgroup:name,hovertemplate:`${name} 轮速: %{y:.4f} rad/s<extra></extra>`}));
if(wheelGainRow!==null){
 N.forEach((name,w)=>add(wheelGainRow,D.wheel_gains[w],name+' individual',COLORS[w],{legendgroup:name}));
 add(wheelGainRow,D.wheel_diagonal[0],'D FL+RR','#7c3aed',{dash:'dash'});add(wheelGainRow,D.wheel_diagonal[1],'D FR+RL','#0f766e',{dash:'dash'});
 N.forEach((name,w)=>add(wheelEdgeRow,D.wheel_edges[w],name+' edge',COLORS[w],{legendgroup:name}));
 add(wheelEdgeRow,D.wheel_diagonal_edges[0],'D FL+RR edge','#7c3aed',{dash:'dash'});add(wheelEdgeRow,D.wheel_diagonal_edges[1],'D FR+RL edge','#0f766e',{dash:'dash'});
}
if(quantFactorRow!==null){
 FACTOR_NAMES.forEach((name,f)=>{add(quantFactorRow,D.quant_factor_residuals[f],name,FACTOR_COLORS[f],{showlegend:true,legendgroup:'factor-'+f,hovertemplate:`${name} residual: %{y:.4f}%<extra></extra>`});add(quantFactorRow,D.quant_factor_edges[f],name+' edge',FACTOR_COLORS[f],{dash:'dash',legendgroup:'factor-'+f,hovertemplate:`${name} edge: %{y:.4f}%<extra></extra>`})});
 N.forEach((name,w)=>{add(quantPhysicalRow,D.quant_physical[w],name+' level',COLORS[w],{legendgroup:name});add(quantPhysicalRow,D.quant_physical_edges[w],name+' edge',COLORS[w],{dash:'dash',legendgroup:name});add(quantZRow,D.quant_shock_z[w],name+' shock z',COLORS[w],{legendgroup:name});add(quantZRow,D.quant_level_z[w],name+' level z',COLORS[w],{dash:'dash',legendgroup:name});add(quantIsolationRow,D.quant_shock_isolation[w],name+' shock isolation',COLORS[w],{legendgroup:name});add(quantIsolationRow,D.quant_level_isolation[w],name+' level isolation',COLORS[w],{dash:'dash',legendgroup:name});const custom=D.times.map((_,i)=>[D.quant_cusum[w][i],D.quant_persistence[w][i],D.quant_states[w][i],D.quant_leading_wheels[i],D.quant_leading_margins[i]]);add(quantRiskRow,D.quant_risk[w],name+' risk',COLORS[w],{legendgroup:name,customdata:custom,hovertemplate:`${name} risk: %{y:.1f}<br>CUSUM=%{customdata[0]:.2f}<br>persistence=%{customdata[1]:.2f}<br>状态=%{customdata[2]}<br>领先轮=%{customdata[3]}<br>领先差=%{customdata[4]:.1f}<extra></extra>`})});
}
function addState(row,candidates,alarms,prefix){add(row,D.signal.map(v=>v?4.82:4.55),'爆胎信号位','#111827',{width:2.2,customdata:D.signal,hovertemplate:'爆胎信号位: %{customdata}<extra></extra>'});N.forEach((name,w)=>{const level=w+.38;add(row,candidates[w].map(v=>v?level:null),prefix+' '+name+' candidate',COLORS[w]+'88',{width:3});add(row,alarms[w].map(v=>v?level:null),prefix+' '+name+' alarm',COLORS[w],{width:7})})}
if(wheelStateRow!==null)addState(wheelStateRow,D.wheel_candidates,D.wheel_alarms,'W');if(quantStateRow!==null)addState(quantStateRow,D.quant_candidates,D.quant_alarms,'Q');
const layout={template:'plotly_white',height:Math.max(1000,rowCount*250),hovermode:'x unified',dragmode:'pan',grid:{rows:rowCount,columns:1,pattern:'independent',roworder:'top to bottom'},legend:{orientation:'h',x:0,y:1.022},margin:{l:82,r:38,t:82,b:62},uirevision:MODE,barmode:'overlay',shapes:[],annotations:[]};
for(let r=1;r<=rowCount;r++){const suffix=r===1?'':String(r);layout['xaxis'+suffix]={matches:r===1?undefined:'x',showticklabels:r===rowCount,title:r===rowCount?'时间 / s':undefined,rangeslider:{visible:r===rowCount,thickness:.045}};layout['yaxis'+suffix]={title:titles[r-1],automargin:true}}
for(const stateRow of [wheelStateRow,quantStateRow])if(stateRow!==null){const suffix=stateRow===1?'':String(stateRow);layout['yaxis'+suffix]={title:'轮位 / 信号',range:[-.1,5.08],tickmode:'array',tickvals:[.38,1.38,2.38,3.38,4.82],ticktext:['FL','FR','RL','RR','爆胎信号']}}
if(quantRiskRow!==null){const suffix=quantRiskRow===1?'':String(quantRiskRow);layout['yaxis'+suffix]={title:titles[quantRiskRow-1],range:[-2,102],automargin:true}}
if(EVENT!==null){layout.shapes.push({type:'line',xref:'x',yref:'paper',x0:EVENT,x1:EVENT,y0:0,y1:1,line:{color:'#dc2626',width:1.5,dash:'dash'}});layout.annotations.push({xref:'x',yref:'paper',x:EVENT,y:1,text:'爆胎信号',showarrow:true,arrowcolor:'#dc2626',font:{color:'#b91c1c',size:12}})}
function hline(targetRow,value,color,label){if(targetRow===null)return;const suffix=targetRow===1?'':String(targetRow);layout.shapes.push({type:'line',xref:'x'+suffix,yref:'y'+suffix,x0:D.times[0],x1:D.times[D.times.length-1],y0:value,y1:value,line:{color,width:1,dash:'dot'}});layout.annotations.push({xref:'x'+suffix,yref:'y'+suffix,x:D.times[D.times.length-1],y:value,text:label,showarrow:false,xanchor:'right',yanchor:'bottom',font:{color,size:10}})}
hline(wheelGainRow,.55,'#64748b','持续门限');hline(wheelEdgeRow,.58,'#475569','候选门限');hline(quantPhysicalRow,.38,'#0f766e','edge触发');hline(quantPhysicalRow,.42,'#64748b','level持续');hline(quantZRow,5,'#0f766e','shock触发');hline(quantZRow,1.5,'#64748b','level确认');hline(quantIsolationRow,2,'#0f766e','shock隔离');hline(quantIsolationRow,1,'#64748b','level隔离');hline(quantRiskRow,55,'#64748b','中位风险');hline(quantRiskRow,82,'#b91c1c','峰值风险');
for(const [times,color,prefix] of [[D.wheel_first_alarms,'#1d4ed8','W'],[D.quant_first_alarms,'#7c3aed','Q']])times.forEach((time,w)=>{if(time!==null&&time>=D.times[0]&&time<=D.times[D.times.length-1]){layout.shapes.push({type:'line',xref:'x',yref:'paper',x0:time,x1:time,y0:0,y1:1,line:{color,width:1,dash:'dot'}});layout.annotations.push({xref:'x',yref:'paper',x:time,y:.97,text:`${prefix} ${N[w]}报警`,showarrow:false,textangle:-90,font:{color,size:10}})}});
const plot=document.getElementById('plot'),config={responsive:true,scrollZoom:true,displaylogo:false,modeBarButtonsToAdd:['drawline','eraseshape']};
Plotly.newPlot(plot,traces,layout,config).then(()=>{plot.on('plotly_hover',event=>{const x=event.points[0].x,i=D.times.indexOf(x);if(i<0)return;const parts=N.map((name,w)=>`${name} ${D.wheels[w][i].toFixed(3)}rad/s · W增益 ${D.wheel_gains[w][i]===null?'—':D.wheel_gains[w][i].toFixed(3)}% · Q风险 ${D.quant_risk[w][i]===null?'—':D.quant_risk[w][i].toFixed(2)}`);document.getElementById('readout').innerHTML=`<b>t=${Number(x).toFixed(2)}s · 信号=${D.signal[i]}</b><span>${parts.join('</span><span>')}</span>`});plot.on('plotly_unhover',()=>{document.getElementById('readout').textContent='在 Plotly 图上悬停，查看同一时刻的四轮证据'})});
"""


_STYLE = """
:root{--ink:#17212b;--muted:#657286;--line:#dde4ec;--blue:#1d4ed8;--bg:#f4f7fa}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 Inter,system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1560px;margin:auto;padding:28px}a{text-decoration:none;color:var(--blue);font-weight:700}header{display:flex;justify-content:space-between;gap:22px;align-items:flex-start;margin-bottom:18px}h1{margin:0 0 5px;font-size:29px}.eyebrow{margin:0 0 6px;color:var(--blue);font-size:11px;font-weight:800;letter-spacing:.14em}.muted,.path{color:var(--muted)}.path{word-break:break-all}.top-nav{display:flex;justify-content:space-between;margin-bottom:16px}.top-nav span,nav{display:flex;gap:8px}.button,button{display:inline-block;border:0;border-radius:8px;background:var(--blue);color:white;padding:9px 14px;cursor:pointer;font:inherit}.secondary{background:#64748b}.cards{display:grid;grid-template-columns:repeat(4,minmax(160px,1fr));gap:12px;margin:16px 0}.card,.panel,.plot-panel,.notice,.readout,aside{background:#fff;border:1px solid var(--line);border-radius:11px}.card{padding:16px}.card.accent{border-top:3px solid var(--blue)}.card span,.card small{display:block;color:var(--muted)}.card strong{display:block;font-size:21px;margin:3px 0}.panel,.notice{padding:18px;margin:14px 0}.controls,.range{display:flex;gap:10px;align-items:center;flex-wrap:wrap}input,select{border:1px solid #c8d2df;border-radius:7px;padding:9px 10px;background:white;font:inherit}.controls input{min-width:300px}.range label{display:flex;align-items:center;gap:6px}.range input{width:110px}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;margin-top:12px}th,td{padding:10px;border-bottom:1px solid #e8edf2;text-align:left;white-space:nowrap}th{background:#f8fafc}.mini-button{display:inline-block;background:#e4e9ff;color:#3730a3;border-radius:7px;padding:5px 9px}.cell-sub{display:block;color:var(--muted);font-weight:400}.notice{background:#fffbeb;border-color:#fde68a}.workbench{display:grid;grid-template-columns:310px minmax(0,1fr);gap:14px;align-items:start}aside{padding:12px;position:sticky;top:12px;max-height:calc(100vh - 24px);overflow:auto}.aside-title{display:flex;justify-content:space-between;padding:4px 5px 11px}.aside-title span{color:var(--muted);font-size:12px}.suspect{display:block;color:var(--ink);border:1px solid #e3e8ef;border-left:4px solid #94a3b8;border-radius:8px;padding:9px 10px;margin-bottom:8px;background:#fff}.suspect:hover,.suspect.selected{border-color:#93b4e8;background:#eff6ff}.suspect-head{display:flex;justify-content:space-between}.suspect span,.suspect small{display:block}.suspect small{color:var(--muted)}.suspect em{font-style:normal;font-size:11px;background:#f1f5f9;color:#64748b;border-radius:999px;padding:2px 6px}.suspect em.ok{background:#dcfce7;color:#166534}.suspect em.signal{background:#fee2e2;color:#b91c1c}.event-link{border-left-color:#dc2626}.empty{text-align:center;color:var(--muted);padding:25px 4px}.charts{min-width:0}.plot-panel{padding:4px 9px;margin-bottom:12px;overflow:hidden}.plot-panel #plot{width:100%;min-height:950px}.readout{position:sticky;top:0;z-index:3;padding:10px 14px;margin-bottom:12px;box-shadow:0 3px 12px #0f172a0d;display:flex;gap:15px;flex-wrap:wrap}.readout span{color:#475569}@media(max-width:900px){main{padding:15px}.cards{grid-template-columns:1fr 1fr}.workbench{grid-template-columns:1fr}aside{position:static;max-height:360px}header{display:block}.controls input{min-width:100%}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve the 0818 wheel-only/quant console."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8773)
    parser.add_argument("--max-window-s", type=float, default=120.0)
    args = parser.parse_args()
    if args.max_window_s <= 0:
        parser.error("--max-window-s must be positive")
    state = ConsoleState(args.input_dir, max_window_s=args.max_window_s)
    ConsoleHandler.state = state
    server = ThreadingHTTPServer((args.host, args.port), ConsoleHandler)
    print(f"0818 wheel-only/quant console: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
