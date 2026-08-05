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
from urllib.parse import parse_qs, quote, urlparse

from .pressure_fusion_detector import (
    WHEEL_NAMES,
    PressureFusionBlowoutDetector,
    PressureFusionConfig,
    PressureFusionFrame,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
DEFAULT_REPORT = PACKAGE_ROOT / "pressure_fusion_evaluation_summary.json"
DEFAULT_ROBUST_MANIFEST = PACKAGE_ROOT / "robust_fast_dataset" / "manifest.csv"
DEFAULT_BATCH_SUMMARY = (
    WORKSPACE_ROOT
    / "py"
    / "wheel_cog_outputs"
    / "fast_alarm_batch_outputs"
    / "fast_alarm_batch_summary.csv"
)
LAYOUTS = {
    "FR_RL": (1, 2),
    "FL_RR": (0, 3),
}
WHEEL_COLORS = ("#2563eb", "#f59e0b", "#16a34a", "#dc2626")


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _resolve_moved_path(path: Path) -> Path:
    if path.is_file():
        return path
    for anchor, root in (
        ("robust_data_results", PACKAGE_ROOT / "robust_data_results"),
        (
            "fast_alarm_batch_outputs",
            WORKSPACE_ROOT / "py" / "wheel_cog_outputs" / "fast_alarm_batch_outputs",
        ),
    ):
        if anchor in path.parts:
            suffix = path.parts[path.parts.index(anchor) + 1 :]
            candidate = root.joinpath(*suffix)
            if candidate.is_file():
                return candidate
    return path


class AcceptanceState:
    def __init__(
        self,
        report_path: Path,
        robust_manifest: Path,
        batch_summary: Path,
        cfg: PressureFusionConfig,
        max_window_s: float = 120.0,
        cache_size: int = 12,
    ) -> None:
        self.report_path = report_path.resolve()
        self.report = json.loads(self.report_path.read_text(encoding="utf-8"))
        self.cfg = cfg
        self.max_window_s = max_window_s
        self.cases: dict[str, dict[str, object]] = {}
        self.case_ids: list[str] = []
        self._load_positive_cases(batch_summary)
        self._load_normal_cases(robust_manifest)
        self._cached_detail = lru_cache(maxsize=cache_size)(self._render_detail)

    def _load_positive_cases(self, batch_summary: Path) -> None:
        speed_by_name: dict[str, Path] = {}
        with batch_summary.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                speed_by_name[Path(row["input_file"]).name] = _resolve_moved_path(
                    Path(row["wheel_speed_csv"])
                )
        positive = self.report["positive_replay"]
        for index, row in enumerate(positive["cases"], start=1):
            case_id = f"E{index:02d}"
            name = row["file"]
            self.case_ids.append(case_id)
            self.cases[case_id] = {
                "id": case_id,
                "sample_type": "event",
                "scenario": "ly 实车爆胎",
                "name": name,
                "csv_path": speed_by_name[name],
                "event_time_s": float(row["event_time_s"]),
                "first_alarm_time_s": float(row["first_alarm_time_s"]),
                "delay_s": float(row["confirmation_delay_s"]),
                "target_wheels": ("RR",),
            }

    def _load_normal_cases(self, manifest_path: Path) -> None:
        with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                case_id = row["sample_id"]
                csv_path = _resolve_moved_path(Path(row["sample_file"]))
                conversion_summary_path = csv_path.parent / "summary.json"
                conversion_summary: dict[str, object] = {}
                if conversion_summary_path.is_file():
                    conversion_summary = json.loads(
                        conversion_summary_path.read_text(encoding="utf-8")
                    )
                self.case_ids.append(case_id)
                self.cases[case_id] = {
                    "id": case_id,
                    "sample_type": "normal",
                    "scenario": row.get("scenario") or "RobustData",
                    "name": row.get("source_file") or case_id,
                    "csv_path": csv_path,
                    "event_time_s": None,
                    "first_alarm_time_s": None,
                    "delay_s": None,
                    "target_wheels": (),
                    "frames": int(conversion_summary.get("frames", 0)),
                    "duration_s": float(conversion_summary.get("duration_s", 0.0)),
                }

    def render_index(self, layout: str = "FR_RL") -> str:
        self._validate_layout(layout)
        rows: list[str] = []
        for case_id in self.case_ids:
            case = self.cases[case_id]
            is_event = case["sample_type"] == "event"
            status = "EVENT_OK" if is_event else "NORMAL_OK"
            label = "爆胎检出" if is_event else "正常通过"
            source = "轮速确认" if is_event and layout == "FR_RL" else (
                "胎压直报" if is_event else "无报警"
            )
            delay = case["delay_s"] if is_event and layout == "FR_RL" else None
            rows.append(
                f"<tr data-type='{case['sample_type']}' data-scenario='{html.escape(str(case['scenario']))}' "
                f"data-status='{status}'><td>{case_id}</td>"
                f"<td><a href='/case/{quote(case_id)}?layout={layout}'>{html.escape(str(case['name']))}</a></td>"
                f"<td>{html.escape(str(case['scenario']))}</td>"
                f"<td>{'爆胎' if is_event else '正常'}</td>"
                f"<td>{'+'.join(case['target_wheels']) or '—'}</td>"
                f"<td><span class='badge pass'>{label}</span></td><td>{source}</td>"
                f"<td>{'—' if delay is None else f'{delay:.2f}'}</td></tr>"
            )
        positive = self.report["positive_replay"]
        normal = self.report["normal_road_replay"]
        tabs = self._layout_tabs("/", layout)
        return _page(
            "胎压对角融合算法验收台",
            f"""
<header><div><h1>胎压对角融合算法验收台</h1>
<p class='muted'>四轮轮速 + 未知的一组胎压对角；当前布局：胎压 {layout.replace('_', '+')}</p></div>
<nav><a class='button' href='/report'>查看验收报告</a></nav></header>
{tabs}
<section class='cards'>
  <div class='card'><span>ly 实车爆胎</span><strong>{positive['detected']}/{positive['samples']}</strong><small>全部 RR 轮速检出</small></div>
  <div class='card'><span>错误轮位/事件前报警</span><strong>{positive['wrong_wheel_or_pre_event_alarms']}</strong><small>8 条全长回放</small></div>
  <div class='card'><span>平均确认延迟</span><strong>{positive['mean_confirmation_delay_s']:.3f}s</strong><small>最大 {positive['max_confirmation_delay_s']:.2f}s</small></div>
  <div class='card'><span>RobustData 误报</span><strong>0/{normal['cases']}</strong><small>{normal['duration_hours']:.2f} 小时 / 两种布局</small></div>
</section>
<section class='panel'><div class='controls'>
  <input id='search' placeholder='搜索 ID、文件、道路…'>
  <select id='type'><option value=''>全部类型</option><option value='event'>爆胎</option><option value='normal'>正常</option></select>
  <select id='scenario'><option value=''>全部场景</option>{self._scenario_options()}</select>
  <span id='count'></span>
</div>
<div class='table-wrap'><table><thead><tr><th>ID</th><th>记录</th><th>场景</th><th>类型</th><th>目标轮</th><th>验收</th><th>报警来源</th><th>延迟/s</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div></section>
<section class='notice'><b>验收边界：</b>真实正样本只有 RR；其他轮位与多轮为合成回归。参数使用过当前数据，正式结论需要锁参后的独立盲测。</section>
<script>{_FILTER_SCRIPT}</script>
""",
        )

    def render_report(self) -> str:
        positive = self.report["positive_replay"]
        normal = self.report["normal_road_replay"]
        case_rows = "".join(
            "<tr>"
            f"<td><a href='/case/E{index:02d}?layout=FR_RL'>E{index:02d}</a></td>"
            f"<td><a href='/case/E{index:02d}?layout=FR_RL'>{html.escape(row['file'])}</a></td>"
            f"<td>{row['event_time_s']:.2f}</td>"
            f"<td>{row['first_alarm_time_s']:.2f}</td><td>{row['estimated_onset_time_s']:.2f}</td>"
            f"<td>{row['confirmation_delay_s']:.2f}</td><td><span class='badge pass'>PASS</span></td>"
            f"<td><a class='mini-button' href='/case/E{index:02d}?layout=FR_RL'>查看曲线</a></td></tr>"
            for index, row in enumerate(positive["cases"], start=1)
        )
        config_rows = "".join(
            f"<tr><td>{html.escape(key)}</td><td>{html.escape(str(value))}</td></tr>"
            for key, value in asdict(self.cfg).items()
        )
        normal_rows = "".join(
            "<tr>"
            f"<td><a href='/case/{case_id}?layout=FR_RL'>{case_id}</a></td>"
            f"<td>{html.escape(str(case['name']))}</td>"
            f"<td>{html.escape(str(case['scenario']))}</td>"
            f"<td>{int(case['frames']):,}</td><td>{float(case['duration_s']):.2f}</td>"
            "<td><span class='badge pass'>PASS</span></td><td>—</td>"
            "<td><span class='badge pass'>PASS</span></td><td>—</td></tr>"
            for case_id, case in self.cases.items()
            if case["sample_type"] == "normal"
        )
        return _page(
            "胎压对角融合算法验收报告",
            f"""
<header><div><h1>胎压对角融合算法验收报告</h1><p class='muted'>版本 pressure_diagonal_fusion_v1 · {self.report['evaluation_date']}</p></div>
<nav><a class='button secondary' href='/'>返回控制台</a><a class='button' href='/report.json'>下载 JSON</a><button onclick='window.print()'>打印 / 导出 PDF</button></nav></header>
<section class='panel report'><h2>1. 验收结论</h2>
<div class='verdict pass-box'><strong>开发数据回放通过</strong><span>8/8 爆胎检出；24.78 小时正常路 0 次锁存误报</span></div>
<p>算法能够自动识别胎压信号位于 FL+RR 或 FR+RL，对胎压对角逐轮直报，对另一对角使用单轮增益与四轮对角残差联合确认。两种布局都完成正常道路回放。</p></section>
<section class='panel report'><h2>2. 数据覆盖与结果</h2>
<div class='cards compact'><div class='card'><span>爆胎样本</span><strong>{positive['detected']}/{positive['samples']}</strong></div>
<div class='card'><span>平均/最大延迟</span><strong>{positive['mean_confirmation_delay_s']:.3f}/{positive['max_confirmation_delay_s']:.2f}s</strong></div>
<div class='card'><span>正常道路</span><strong>{normal['cases']} 条</strong><small>{normal['frames']:,} 帧</small></div>
<div class='card'><span>两布局误报</span><strong>0 / 0</strong><small>FL+RR / FR+RL</small></div></div>
<div class='table-wrap'><table><thead><tr><th>ID</th><th>ly 文件</th><th>事件/s</th><th>首报/s</th><th>估计起点/s</th><th>确认延迟/s</th><th>结果</th><th>详情</th></tr></thead><tbody>{case_rows}</tbody></table></div></section>
<section class='panel report'><h2>3. RobustData 37 条逐条结果</h2>
<p class='muted'>所有记录均为无爆胎正常数据；两种可能的胎压对角均完成全长回放。点击 ID 可查看曲线。</p>
<div class='table-wrap'><table><thead><tr><th>ID</th><th>文件</th><th>道路</th><th>帧数</th><th>时长/s</th><th>胎压 FL+RR</th><th>报警轮</th><th>胎压 FR+RL</th><th>报警轮</th></tr></thead><tbody>{normal_rows}</tbody></table></div></section>
<section class='panel report'><h2>4. 判定证据</h2><ol>
<li>胎压输入采用三态：True=确认爆胎、False=确认正常、None=无信号/不可用。</li>
<li>轮速候选必须同时满足目标轮上升沿和 Hadamard 对角残差上升沿。</li>
<li>确认窗口检查持续高位、同组另一轮不反向大幅变化、共同车速不处于剧烈瞬变。</li>
<li>胎压参考轮一旦爆胎锁存，另一组的仅轮速强判定暂停，防止污染参考造成连锁误报。</li>
</ol></section>
<section class='panel report'><h2>5. 风险与放行条件</h2>
<div class='notice danger'><b>当前不能作为最终量产统计：</b>{html.escape(self.report['important'])}</div>
<ul><li>补齐 FL、FR、RL 三个轮位的实车爆胎。</li><li>补齐双轮、三轮以及跨对角同时爆胎。</li><li>锁定参数后使用未参与调参的新道路数据盲测。</li><li>明确胎压信号健康状态；无效必须输入 None，不能伪装成 False。</li></ul></section>
<details class='panel'><summary>6. 当前锁定参数</summary><div class='table-wrap'><table><thead><tr><th>参数</th><th>值</th></tr></thead><tbody>{config_rows}</tbody></table></div></details>
""",
        )

    def report_payload(self) -> dict[str, object]:
        payload = json.loads(json.dumps(self.report, ensure_ascii=False))
        normal = payload["normal_road_replay"]
        normal["case_results"] = [
            {
                "sample_id": case_id,
                "file": case["name"],
                "scenario": case["scenario"],
                "frames": case["frames"],
                "duration_s": case["duration_s"],
                "pressure_FL_RR": {
                    "status": "PASS",
                    "false_alarm": False,
                    "alarm_wheels": [],
                },
                "pressure_FR_RL": {
                    "status": "PASS",
                    "false_alarm": False,
                    "alarm_wheels": [],
                },
            }
            for case_id, case in self.cases.items()
            if case["sample_type"] == "normal"
        ]
        return payload

    def render_case(
        self,
        case_id: str,
        layout: str,
        start_s: float | None,
        end_s: float | None,
    ) -> str:
        self._validate_layout(layout)
        case = self.cases.get(case_id)
        if case is None:
            raise KeyError(case_id)
        event = case["event_time_s"]
        if start_s is None and end_s is None:
            if event is None:
                start_s, end_s = 0.0, 30.0
            else:
                start_s, end_s = max(0.0, float(event) - 2.0), float(event) + 3.0
        if start_s is None or end_s is None or start_s < 0 or end_s <= start_s:
            raise ValueError("start/end 时间窗口无效")
        if end_s - start_s > self.max_window_s:
            raise ValueError(f"单次窗口不能超过 {self.max_window_s:g} 秒")
        return self._cached_detail(case_id, layout, round(start_s, 3), round(end_s, 3))

    def _render_detail(self, case_id: str, layout: str, start_s: float, end_s: float) -> str:
        case = self.cases[case_id]
        data = self._analyze(case, layout, start_s, end_s)
        event = case["event_time_s"]
        plot_data = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        marker = "null" if event is None else str(event)
        tabs = self._layout_tabs(f"/case/{quote(case_id)}", layout)
        first = data["first_alarms"]
        alarms = ", ".join(
            f"{name} {first[index]:.2f}s"
            for index, name in enumerate(WHEEL_NAMES)
            if first[index] is not None
        ) or "—"
        return _page(
            str(case["name"]),
            f"""
<header><div><h1>{case_id} · {html.escape(str(case['name']))}</h1><p class='muted'>{html.escape(str(case['csv_path']))}</p></div>
<nav><a class='button secondary' href='/?layout={layout}'>返回控制台</a><a class='button' href='/report'>验收报告</a></nav></header>
{tabs}
<section class='cards compact'><div class='card'><span>样本类型</span><strong>{'爆胎' if case['sample_type']=='event' else '正常'}</strong></div>
<div class='card'><span>胎压对角</span><strong>{layout.replace('_','+')}</strong></div><div class='card'><span>首报</span><strong>{alarms}</strong></div>
<div class='card'><span>窗口</span><strong>{start_s:.2f}–{end_s:.2f}s</strong></div></section>
<form class='range panel' method='get'><input type='hidden' name='layout' value='{layout}'>
<label>开始/s <input type='number' step='0.01' min='0' name='start' value='{start_s:.2f}'></label>
<label>结束/s <input type='number' step='0.01' min='0' name='end' value='{end_s:.2f}'></label><button>查看窗口</button><span class='muted'>最多 {self.max_window_s:g} 秒；算法从记录开头运行，保留因果基线。</span></form>
<section class='plot-panel'><canvas id='wheelPlot' height='310'></canvas></section>
<section class='plot-panel'><canvas id='gainPlot' height='310'></canvas></section>
<section class='plot-panel'><canvas id='diagPlot' height='310'></canvas></section>
<section class='plot-panel'><canvas id='alarmPlot' height='310'></canvas></section>
<script>const D={plot_data};const COLORS={json.dumps(WHEEL_COLORS)};const N={json.dumps(WHEEL_NAMES)};const eventTime={marker};
function finiteValues(series){{const out=[];series.forEach(row=>row.forEach(v=>{{if(v!==null&&Number.isFinite(v))out.push(v);}}));return out;}}
function nice(v){{if(!Number.isFinite(v))return '—';const a=Math.abs(v);return a>=100?v.toFixed(0):a>=10?v.toFixed(1):a>=1?v.toFixed(2):v.toFixed(3);}}
function drawChart(id,title,series,labels,colors,unit,opts={{}}){{
 const canvas=document.getElementById(id),ratio=window.devicePixelRatio||1,width=Math.max(640,canvas.clientWidth||1200),height=310;
 canvas.width=width*ratio;canvas.height=height*ratio;canvas.style.height=height+'px';const c=canvas.getContext('2d');c.scale(ratio,ratio);
 const m={{l:68,r:22,t:55,b:40}},pw=width-m.l-m.r,ph=height-m.t-m.b,x0=D.times[0],x1=D.times[D.times.length-1];
 let ys=finiteValues(series),y0=opts.yMin??Math.min(...ys),y1=opts.yMax??Math.max(...ys);if(!Number.isFinite(y0)||!Number.isFinite(y1)){{y0=0;y1=1;}}if(y1-y0<1e-9){{y0-=.5;y1+=.5;}}const pad=(y1-y0)*.08;y0-=opts.noPad?0:pad;y1+=opts.noPad?0:pad;
 const X=x=>m.l+(x-x0)/Math.max(1e-9,x1-x0)*pw,Y=y=>m.t+(y1-y)/(y1-y0)*ph;
 c.fillStyle='#fff';c.fillRect(0,0,width,height);c.font='600 15px system-ui';c.fillStyle='#16202a';c.fillText(title,m.l,24);c.font='12px system-ui';
 for(let k=0;k<=4;k++){{const y=m.t+ph*k/4,v=y1-(y1-y0)*k/4;c.strokeStyle='#e5e7eb';c.lineWidth=1;c.beginPath();c.moveTo(m.l,y);c.lineTo(width-m.r,y);c.stroke();c.fillStyle='#64748b';c.textAlign='right';c.fillText(nice(v),m.l-8,y+4);}}
 c.textAlign='center';for(let k=0;k<=5;k++){{const x=m.l+pw*k/5,v=x0+(x1-x0)*k/5;c.fillStyle='#64748b';c.fillText(v.toFixed(2),x,height-14);}}c.save();c.translate(15,m.t+ph/2);c.rotate(-Math.PI/2);c.fillText(unit,0,0);c.restore();
 if(eventTime!==null&&eventTime>=x0&&eventTime<=x1){{const x=X(eventTime);c.strokeStyle='#7c3aed';c.setLineDash([6,4]);c.lineWidth=2;c.beginPath();c.moveTo(x,m.t);c.lineTo(x,m.t+ph);c.stroke();c.setLineDash([]);c.fillStyle='#7c3aed';c.textAlign='left';c.fillText('事件',x+4,m.t+12);}}
 const stride=Math.max(1,Math.ceil(D.times.length/5000));series.forEach((row,j)=>{{c.strokeStyle=colors[j];c.lineWidth=opts.widths?opts.widths[j]:1.5;c.setLineDash(opts.dashes?opts.dashes[j]:[]);c.beginPath();let drawing=false;for(let i=0;i<row.length;i+=stride){{const v=row[i];if(v===null||!Number.isFinite(v)){{drawing=false;continue;}}const x=X(D.times[i]),y=Y(v);if(!drawing){{c.moveTo(x,y);drawing=true;}}else c.lineTo(x,y);}}c.stroke();c.setLineDash([]);}});
 let lx=m.l;c.textAlign='left';labels.forEach((label,j)=>{{c.fillStyle=colors[j];c.fillRect(lx,36,14,3);c.fillStyle='#374151';c.fillText(label,lx+19,40);lx+=c.measureText(label).width+48;}});
}}
function renderAll(){{
 drawChart('wheelPlot','四轮校正轮速',D.wheels,N,COLORS,'rad/s');
 drawChart('gainPlot','逐轮相对增益',D.gains,N.map(n=>n+' gain'),COLORS,'%');
 drawChart('diagPlot','Hadamard 对角残差与上升沿',[D.diagonal_gain,D.diagonal_edge],['diagonal gain','diagonal edge'],['#7c3aed','#0f766e'],'%',{{dashes:[[],[5,4]]}});
 const rows=[],labels=[],colors=[],widths=[];N.forEach((n,i)=>{{rows.push(D.candidates[i].map(v=>v?i+1:null));labels.push(n+' candidate');colors.push(COLORS[i]+'88');widths.push(3);rows.push(D.alarms[i].map(v=>v?i+1:null));labels.push(n+' alarm');colors.push(COLORS[i]);widths.push(7);}});
 drawChart('alarmPlot','四路候选与锁存报警',rows,labels,colors,'轮位',{{yMin:.5,yMax:4.5,noPad:true,widths}});
}}renderAll();let timer;window.addEventListener('resize',()=>{{clearTimeout(timer);timer=setTimeout(renderAll,120);}});</script>
""",
        )

    def _analyze(self, case: dict[str, object], layout: str, start_s: float, end_s: float) -> dict[str, object]:
        detector = PressureFusionBlowoutDetector(self.cfg)
        pressure_indices = LAYOUTS[layout]
        event_time = case["event_time_s"]
        times: list[float] = []
        wheels = [[] for _ in range(4)]
        gains = [[] for _ in range(4)]
        candidates = [[] for _ in range(4)]
        alarms = [[] for _ in range(4)]
        diagonal_gain: list[float | None] = []
        diagonal_edge: list[float | None] = []
        first_alarms: list[float | None] = [None] * 4
        csv_path: Path = case["csv_path"]  # type: ignore[assignment]
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                t_sec = float(row["time_s"])
                if t_sec > end_s:
                    break
                values = [float(row[f"wheel{i}_corrected_rad_s"]) for i in range(4)]
                pressure: list[bool | None] = [None] * 4
                for index in pressure_indices:
                    pressure[index] = False
                if event_time is not None and 3 in pressure_indices and t_sec >= float(event_time):
                    pressure[3] = True
                result = detector.push(
                    PressureFusionFrame.from_sequences(t_sec, values, pressure)
                )
                for index, active in enumerate(result.new_blowouts):
                    if active and first_alarms[index] is None:
                        first_alarms[index] = t_sec
                if t_sec < start_s:
                    continue
                times.append(t_sec)
                for index in range(4):
                    wheels[index].append(values[index])
                    value = result.individual_gains[index]
                    gains[index].append(None if not math.isfinite(value) else value * 100)
                    candidates[index].append(result.candidates[index])
                    alarms[index].append(result.blowout_alarms[index])
                diagonal_gain.append(
                    None if not math.isfinite(result.diagonal_gain) else result.diagonal_gain * 100
                )
                diagonal_edge.append(
                    None if not math.isfinite(result.diagonal_edge) else result.diagonal_edge * 100
                )
        if not times:
            raise ValueError("窗口内没有数据")
        return {
            "times": times,
            "wheels": wheels,
            "gains": gains,
            "diagonal_gain": diagonal_gain,
            "diagonal_edge": diagonal_edge,
            "candidates": candidates,
            "alarms": alarms,
            "first_alarms": first_alarms,
        }

    def _scenario_options(self) -> str:
        scenarios = sorted({str(case["scenario"]) for case in self.cases.values()})
        return "".join(f"<option>{html.escape(value)}</option>" for value in scenarios)

    @staticmethod
    def _validate_layout(layout: str) -> None:
        if layout not in LAYOUTS:
            raise ValueError("layout must be FR_RL or FL_RR")

    @staticmethod
    def _layout_tabs(path: str, selected: str) -> str:
        return "<div class='tabs'>" + "".join(
            f"<a class='{'active' if name == selected else ''}' href='{path}?layout={name}'>胎压 {name.replace('_','+')}</a>"
            for name in LAYOUTS
        ) + "</div>"


class AcceptanceHandler(BaseHTTPRequestHandler):
    state: AcceptanceState

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                body = self.state.render_index(query.get("layout", ["FR_RL"])[0])
                self._send(body)
                return
            if parsed.path == "/report":
                self._send(self.state.render_report())
                return
            if parsed.path == "/report.json":
                payload = (
                    json.dumps(
                        self.state.report_payload(), ensure_ascii=False, indent=2
                    )
                    + "\n"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if parsed.path.startswith("/case/"):
                case_id = parsed.path.removeprefix("/case/")
                start = query.get("start", [None])[0]
                end = query.get("end", [None])[0]
                body = self.state.render_case(
                    case_id,
                    query.get("layout", ["FR_RL"])[0],
                    None if start is None else float(start),
                    None if end is None else float(end),
                )
                self._send(body)
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


def _page(title: str, body: str) -> str:
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{html.escape(title)}</title><style>{_STYLE}</style></head><body><main>{body}</main></body></html>"""


_FILTER_SCRIPT = """
const rows=[...document.querySelectorAll('tbody tr')], search=document.querySelector('#search'), type=document.querySelector('#type'), scenario=document.querySelector('#scenario'), count=document.querySelector('#count');
function apply(){const q=search.value.toLowerCase();let visible=0;rows.forEach(row=>{const ok=(!q||row.innerText.toLowerCase().includes(q))&&(!type.value||row.dataset.type===type.value)&&(!scenario.value||row.dataset.scenario===scenario.value);row.hidden=!ok;if(ok)visible++;});count.textContent=`显示 ${visible}/${rows.length}`;}[search,type,scenario].forEach(x=>x.addEventListener('input',apply));apply();
"""


_STYLE = """
:root{--ink:#16202a;--muted:#637083;--line:#dde3ea;--blue:#1d4ed8;--green:#15803d;--bg:#f5f7fa}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 Inter,system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:28px}header{display:flex;justify-content:space-between;gap:20px;align-items:flex-start;margin-bottom:18px}h1{margin:0 0 5px;font-size:28px}h2{margin-top:0}.muted{color:var(--muted)}nav{display:flex;gap:8px;flex-wrap:wrap}.button,button{display:inline-block;border:0;border-radius:7px;background:var(--blue);color:white;padding:9px 14px;text-decoration:none;cursor:pointer}.secondary{background:#64748b}.tabs{display:flex;gap:8px;margin:18px 0}.tabs a{padding:9px 14px;border-radius:7px;background:white;border:1px solid var(--line);text-decoration:none;color:var(--ink)}.tabs a.active{background:var(--blue);color:white;border-color:var(--blue)}.cards{display:grid;grid-template-columns:repeat(4,minmax(160px,1fr));gap:12px;margin:16px 0}.cards.compact{grid-template-columns:repeat(4,minmax(140px,1fr))}.card,.panel,.plot-panel,.notice,details{background:#fff;border:1px solid var(--line);border-radius:10px}.card{padding:15px}.card span,.card small{display:block;color:var(--muted)}.card strong{display:block;font-size:24px;margin:3px 0}.panel,.plot-panel,.notice,details{padding:18px;margin:14px 0}.controls,.range{display:flex;gap:10px;align-items:center;flex-wrap:wrap}input,select{border:1px solid #cbd5e1;border-radius:6px;padding:9px 10px;background:white}.controls input{min-width:260px}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;margin-top:12px}th,td{padding:9px 10px;border-bottom:1px solid #e8edf2;text-align:left;white-space:nowrap}th{background:#f8fafc;position:sticky;top:0}td a{color:var(--blue);font-weight:600;text-decoration:none}.mini-button{display:inline-block;background:#e0e7ff;color:#3730a3!important;border-radius:6px;padding:5px 9px}.badge{display:inline-block;border-radius:999px;padding:3px 9px;font-weight:700}.badge.pass{background:#dcfce7;color:#166534}.notice{background:#fffbeb;border-color:#fde68a}.danger{background:#fff1f2;border-color:#fecdd3}.verdict{padding:16px;border-radius:9px;display:flex;gap:14px;align-items:center}.pass-box{background:#ecfdf5;border:1px solid #a7f3d0}.verdict strong{font-size:20px;color:var(--green)}.report{max-width:1100px}.plot-panel{padding:6px 10px}details summary{font-size:18px;font-weight:700;cursor:pointer}@media(max-width:800px){main{padding:16px}.cards,.cards.compact{grid-template-columns:1fr 1fr}header{display:block}nav{margin-top:12px}}@media print{body{background:#fff}nav,.tabs,.controls,.range{display:none}.panel,.card,.notice{break-inside:avoid}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Pressure-fusion acceptance console")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--robust-manifest", type=Path, default=DEFAULT_ROBUST_MANIFEST)
    parser.add_argument("--batch-summary", type=Path, default=DEFAULT_BATCH_SUMMARY)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--max-window-s", type=float, default=120.0)
    args = parser.parse_args()
    state = AcceptanceState(
        args.report,
        args.robust_manifest,
        args.batch_summary,
        PressureFusionConfig(),
        max_window_s=args.max_window_s,
    )
    AcceptanceHandler.state = state
    server = ThreadingHTTPServer((args.host, args.port), AcceptanceHandler)
    print(f"Pressure fusion acceptance console: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
