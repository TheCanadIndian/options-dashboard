"""Static site pages for GitHub Pages, written to docs/."""

from __future__ import annotations

import json
import subprocess
from datetime import date, datetime, timedelta
from html import escape

import numpy as np
from typing import Any

from . import config, earnings, portfolio, regime, scorecard, structure, universe
from .config import HOME

OUT = HOME / "docs" / "index.html"

FACTOR_LABELS = {
    "trend": "Strength of the market read", "liquidity": "Liquidity", "breakeven": "Breakeven vs expected move",
    "iv_value": "Option price vs volatility", "gamma": "Gamma backdrop", "theta": "Low time decay",
    "delta": "Delta near target", "target": "Breakeven inside structural target",
}
# Factor weights in force before the gamma factor existed; older positions do not store their own.
LEGACY_WEIGHTS = {"trend": 30, "liquidity": 20, "breakeven": 15, "iv_value": 15, "theta": 10, "delta": 10}

CSS = """
:root{color-scheme:dark;--page:#000;--surface:#0b0b0b;--head:#140e03;--ink:#e9e9e6;--ink2:#b3b3ad;--muted:#7d7d78;
--amber:#ffa028;--amber2:#ffbe5c;--grid:#1c1c1b;--axis:#3a3a38;--border:#272725;--series:#ffa028;--good:#2fd36b;
--bad:#ff5252;--track:#232321;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;
--s7:#9085e9;--s8:#e66767;--q-lead:#199e70;--q-weak:#c98500;--q-lag:#e66767;--q-imp:#3987e5;
--mono:"IBM Plex Mono",ui-monospace,"SFMono-Regular",Consolas,"Liberation Mono",monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:13px/1.45 var(--mono);font-variant-numeric:tabular-nums}
a{color:var(--amber2)}
main{max-width:1240px;margin:0 auto;padding:10px 14px 28px}
.topbar{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;background:#0d0d0c;
border-bottom:1px solid var(--border);padding:6px 14px;font-size:12px}
.brand{color:#000;background:var(--amber);font-weight:700;padding:2px 8px;letter-spacing:.08em}
.topbar .clock{color:var(--ink2)}.topbar .clock b{color:var(--amber)}
.tape{overflow:hidden;white-space:nowrap;border-bottom:1px solid var(--border);background:#050505;font-size:12px}
.tape .roll{display:inline-block;padding:5px 0;animation:roll 60s linear infinite}
.tape span.q{margin:0 18px}.tape .sym{color:var(--amber);font-weight:600;margin-right:6px}
@keyframes roll{from{transform:translateX(0)}to{transform:translateX(-50%)}}
@media (prefers-reduced-motion:reduce){.tape{overflow-x:auto}.tape .roll{animation:none}}
nav{display:flex;flex-wrap:wrap;gap:4px;margin:10px 0 12px}
main>*{min-width:0}
nav a{padding:7px 12px;border:1px solid var(--border);color:var(--ink2);text-decoration:none;font-size:12px;
letter-spacing:.06em;background:#0d0d0c}
nav a b{color:var(--amber);margin-right:6px}
nav a[aria-current]{background:var(--amber);color:#000;border-color:var(--amber)}nav a[aria-current] b{color:#000}
h1{font-size:15px;margin:0;color:var(--amber);text-transform:uppercase;letter-spacing:.1em}
h2{font-size:12px;margin:22px 0 8px;color:var(--amber);text-transform:uppercase;letter-spacing:.12em;
background:var(--head);border-left:3px solid var(--amber);padding:5px 8px}
h3{font-size:13px}h4{color:var(--amber2)}
.sub{color:var(--muted);margin:3px 0 0;font-size:12px}.muted{color:var(--muted)}.small{font-size:11.5px}
.notice{border:1px solid var(--border);padding:8px 10px;margin:12px 0 0;color:var(--ink2);font-size:12px;background:var(--surface)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:6px;margin-top:10px}
.tile,.card,.panel{background:var(--surface);border:1px solid var(--border)}
.tile{padding:8px 10px}.tile .k{color:var(--amber);font-size:10.5px;text-transform:uppercase;letter-spacing:.08em}
.tile .v{font-size:20px;font-weight:600;margin:2px 0}.tile .d{font-size:11.5px;color:var(--muted)}
.up{color:var(--good)}.down{color:var(--bad)}
.panel{padding:10px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,440px),1fr));gap:8px}
.card{padding:10px}
.card header{display:flex;justify-content:space-between;gap:10px;align-items:baseline;flex-wrap:wrap;
border-bottom:1px solid var(--border);padding-bottom:6px}
.card h3{margin:0;font-size:13px;color:var(--ink);text-transform:uppercase;letter-spacing:.04em}
.pl{font-size:13px;font-weight:600;white-space:nowrap}
.tags{display:flex;gap:4px;flex-wrap:wrap;margin:6px 0 0}
.tag{border:1px solid #4a3410;color:var(--amber2);padding:0 6px;font-size:11px;text-transform:uppercase}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(105px,1fr));gap:6px 10px;margin:8px 0 0}
.facts div span{display:block;color:var(--amber);font-size:10.5px;text-transform:uppercase}.facts div b{font-weight:600}
.range{margin:16px 0 4px}.bar{position:relative;height:4px;background:var(--track)}
.bar i{position:absolute;top:-4px;width:2px;height:12px;background:var(--muted)}
.bar u{position:absolute;top:-5px;width:14px;height:14px;margin-left:-7px;background:var(--amber);border:2px solid var(--surface)}
.ends{display:flex;justify-content:space-between;font-size:11px;color:var(--ink2);margin-top:6px}
.review{margin:10px 0 0;padding:6px 8px;border-left:3px solid var(--amber);background:#0f0b04;font-size:12px;color:var(--ink2)}
.review b{color:var(--ink)}
details{margin-top:8px;border-top:1px solid var(--border);padding-top:4px}
summary{cursor:pointer;font-weight:600;padding:10px 0;min-height:24px;color:var(--amber2);text-transform:uppercase;
font-size:11.5px;letter-spacing:.06em}
details h4{margin:10px 0 4px;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.05em}
details ul{margin:0;padding-left:16px}details li{margin:2px 0}
ul.plain{list-style:none;padding:0}
.pts{display:grid;grid-template-columns:minmax(120px,1.3fr) 1fr auto;gap:4px 10px;align-items:center;font-size:12px}
.pts .b{height:4px;background:var(--track);overflow:hidden}.pts .b span{display:block;height:100%;background:var(--amber)}
.pts .n{color:var(--ink2)}
table{width:100%;border-collapse:collapse;font-size:12px}
th,td{text-align:left;padding:4px 8px;border-bottom:1px solid #161615;vertical-align:top}
th{color:var(--amber);font-weight:600;font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;
border-bottom:1px solid #3a2a0c;background:#0e0b05}
tbody tr:nth-child(even){background:#070707}tbody tr:hover{background:#17130a}
td.num{white-space:nowrap}
.scroll{overflow-x:auto}
td a{color:var(--amber2);text-decoration:none}td a:hover{text-decoration:underline}
.sortable th{cursor:pointer;white-space:nowrap;user-select:none}
.sortable th[data-dir=asc]::after{content:" ▲"}.sortable th[data-dir=desc]::after{content:" ▼"}
.sortable td{white-space:nowrap}
.filters{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin-bottom:8px;font-size:12px;color:var(--ink2)}
.filters input[type=checkbox]{width:20px;height:20px;vertical-align:middle;margin:0 6px 0 0;accent-color:var(--amber)}
.filters label{display:inline-flex;align-items:center;min-height:36px}
.filters select{font:inherit;padding:6px 8px;background:#0d0d0c;color:var(--ink);border:1px solid var(--border)}
.card:target{outline:1px solid var(--amber)}
.dbar{position:relative;height:6px;background:var(--track)}
.dbar span{position:absolute;top:0;height:6px}.dbar i{position:absolute;left:50%;top:-3px;width:1px;height:12px;background:var(--axis)}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:11.5px;color:var(--ink2);margin-bottom:6px}
.legend i{display:inline-block;width:9px;height:9px;margin-right:6px}
.linechart{position:relative}.linechart svg{display:block;width:100%;height:auto}
svg text{font-family:var(--mono)}
.tip{position:absolute;pointer-events:none;background:#0d0d0c;border:1px solid var(--amber);padding:4px 8px;
font-size:11.5px;white-space:nowrap;display:none}
.statusbar{display:flex;flex-wrap:wrap;gap:4px 18px;margin-top:24px;padding:6px 10px;background:#0d0d0c;
border:1px solid var(--border);font-size:11.5px;color:var(--ink2)}
.statusbar b{color:var(--amber);font-weight:600;margin-right:4px}
footer{margin-top:8px;color:var(--muted);font-size:11px}
.rg-toggle{display:flex;gap:0;margin:0 0 10px}
.rg-toggle input{position:absolute;opacity:0;pointer-events:none}
.rg-toggle label{padding:7px 14px;min-height:32px;border:1px solid var(--border);background:#0d0d0c;color:var(--ink2);
cursor:pointer;font-size:12px;letter-spacing:.06em;text-transform:uppercase}
.rg-toggle input:checked+label{background:var(--amber);border-color:var(--amber);color:#000;font-weight:600}
.rg-toggle input:focus-visible+label{outline:2px solid var(--amber2);outline-offset:2px}
.rg:has(.rg-toggle) .rg-pane{display:none}
.rg:has(#rg-sec:checked) .rg-sec,.rg:has(#rg-mag7:checked) .rg-mag7{display:block}
.grid2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;align-items:start}
.grid2>section>h2:first-child{margin-top:22px}
@media (max-width:1000px){.grid2{grid-template-columns:minmax(0,1fr)}}
.mob{display:none}
@media (max-width:640px){
.desk{display:none}.mob{display:block}.lb-sum{display:none}
.tiles{grid-template-columns:repeat(2,minmax(0,1fr))}.tile .v{font-size:17px}
main{padding:8px 10px 28px}nav a{padding:8px 9px}}
"""

METHOD_SHORT = {"auction": "Auction", "gamma": "Dealer flows", "wyckoff": "Wyckoff", "vpa": "VPA",
                "avwap": "AVWAPs", "vol": "Vol surface", "sector": "Sector", "trend": "Trend",
                "seasonal": "Seasonal"}

TABLE_JS = """
document.querySelectorAll('table.sortable').forEach(function(table){
table.querySelectorAll('th').forEach(function(th,i){th.tabIndex=0;function sort(){
var body=table.tBodies[0],dir=th.dataset.dir==='desc'?'asc':'desc';
table.querySelectorAll('th').forEach(function(o){delete o.dataset.dir});th.dataset.dir=dir;
Array.from(body.rows).sort(function(a,b){var x=a.cells[i].dataset.v,y=b.cells[i].dataset.v,
nx=parseFloat(x),ny=parseFloat(y),c=(isNaN(nx)||isNaN(ny))?x.localeCompare(y):nx-ny;
return dir==='asc'?c:-c}).forEach(function(r){body.appendChild(r)})}
th.addEventListener('click',sort);th.addEventListener('keydown',function(e){if(e.key==='Enter')sort()})})});
(function(){var t=document.getElementById('f-ticker'),b=document.getElementById('f-buy'),
n=document.getElementById('f-count');if(!t)return;var rows=document.querySelectorAll('#flagged tbody tr');
function apply(){var shown=0;rows.forEach(function(r){var ok=(!t.value||r.dataset.t===t.value)&&
(!b.checked||r.dataset.b==='1');r.style.display=ok?'':'none';shown+=ok});
n.textContent='Showing '+shown+' of '+rows.length}
t.addEventListener('change',apply);b.addEventListener('change',apply);apply()})();
"""

CHART_JS = """
document.querySelectorAll('.linechart').forEach(function(el){var d=JSON.parse(el.dataset.points),
svg=el.querySelector('svg'),line=svg.querySelector('.xh'),dot=svg.querySelector('.xd'),tip=el.querySelector('.tip');
function move(e){var r=svg.getBoundingClientRect(),cx=(e.touches?e.touches[0].clientX:e.clientX)-r.left,
vw=svg.viewBox.baseVal.width,vx=cx/r.width*vw,best=0,gap=1e9;for(var i=0;i<d.length;i++){var g=Math.abs(d[i][0]-vx);if(g<gap){gap=g;best=i}}
var p=d[best];line.setAttribute('x1',p[0]);line.setAttribute('x2',p[0]);dot.setAttribute('cx',p[0]);
dot.setAttribute('cy',p[1]);line.style.display=dot.style.display='';tip.style.display='block';
tip.innerHTML='<b>'+p[3]+'</b><br>'+p[2];var x=p[0]/vw*r.width;
tip.style.left=Math.min(Math.max(x-tip.offsetWidth/2,0),r.width-tip.offsetWidth)+'px';
tip.style.top=Math.max(p[1]/260*r.height-tip.offsetHeight-12,0)+'px'}
function out(){line.style.display=dot.style.display='none';tip.style.display='none'}
svg.addEventListener('mousemove',move);svg.addEventListener('touchstart',move,{passive:true});
svg.addEventListener('touchmove',move,{passive:true});svg.addEventListener('mouseleave',out)});
"""


def _money(x: float, cents: bool = True) -> str:
    return f"${x:,.2f}" if cents else f"${x:,.0f}"


def _signed(x: float, cents: bool = True) -> str:
    body = f"${abs(x):,.2f}" if cents else f"${abs(x):,.0f}"
    return ("+" if x >= 0 else "−") + body


def _delta(x: float, pct: float | None = None) -> str:
    """Signed dollar change with an arrow, so direction never relies on colour alone."""
    text = f"{'▲' if x >= 0 else '▼'} {_signed(x)}"
    if pct is not None:
        text += f" ({pct:+.1%})"
    return f'<span class="{"up" if x >= 0 else "down"}">{text}</span>'


def _when(iso: str) -> str:
    t = datetime.fromisoformat(iso)
    return f"{t:%b} {t.day}, {t:%I:%M %p}".replace(", 0", ", ")


def _day(iso: str) -> str:
    t = datetime.fromisoformat(iso)
    return f"{t:%b} {t.day}"


def _both(desktop: str, phone: str) -> str:
    """Two drawings of the same chart; CSS shows the phone one on narrow screens."""
    return f'<div class="desk">{desktop}</div><div class="mob">{phone}</div>'


def _thin(ticks: list, most: int) -> list:
    return ticks[::max(1, -(-len(ticks) // most))]


def _line_chart(values: list[float], hover_labels: list[str], hover_values: list[str], ticks: list[tuple[int, str]],
                axis_format, baseline: float, baseline_label: str, aria: str, pad_floor: float,
                width: int = 800) -> str:
    """One-series line chart with a dashed reference line and a hover crosshair (see CHART_JS)."""
    W, H, L, R, T, B = width, 260, 56, 12, 12, 28
    lo, hi = min(values + [baseline]), max(values + [baseline])
    pad = max((hi - lo) * 0.15, pad_floor)
    lo, hi = lo - pad, hi + pad

    def x(i: int) -> float:
        return L + (W - L - R) * i / (len(values) - 1)

    def y(v: float) -> float:
        return T + (H - T - B) * (1 - (v - lo) / (hi - lo))

    grid = "".join(
        f'<line x1="{L}" x2="{W - R}" y1="{y(lo + (hi - lo) * k / 4):.1f}" y2="{y(lo + (hi - lo) * k / 4):.1f}" stroke="var(--grid)"/>'
        f'<text x="{L - 8}" y="{y(lo + (hi - lo) * k / 4) + 4:.1f}" text-anchor="end">{axis_format(lo + (hi - lo) * k / 4)}</text>'
        for k in range(5))
    tick_html = "".join(
        f'<text x="{x(i):.1f}" y="{H - 8}" text-anchor="{"start" if n == 0 else "middle"}">{escape(text)}</text>'
        for n, (i, text) in enumerate(ticks))
    path = " ".join(f"{'M' if i == 0 else 'L'}{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values))
    data = [[round(x(i), 1), round(y(v), 1), hover_values[i], hover_labels[i]] for i, v in enumerate(values)]
    base = y(baseline)
    return (
        f'<div class="linechart" data-points="{escape(json.dumps(data))}">'
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{escape(aria)}" font-size="13" fill="var(--muted)">'
        f'{grid}{tick_html}'
        f'<line x1="{L}" x2="{W - R}" y1="{base:.1f}" y2="{base:.1f}" stroke="var(--axis)" stroke-dasharray="4 4"/>'
        f'<text x="{W - R}" y="{base - 5:.1f}" text-anchor="end">{escape(baseline_label)}</text>'
        f'<path d="{path}" fill="none" stroke="var(--series)" stroke-width="2" stroke-linejoin="round"/>'
        f'<line class="xh" y1="{T}" y2="{H - B}" stroke="var(--axis)" style="display:none"/>'
        f'<circle class="xd" r="5" fill="var(--series)" stroke="var(--surface)" stroke-width="2" style="display:none"/>'
        f'</svg><div class="tip"></div></div>'
    )


def _chart(points: list[list], start_cash: float) -> str:
    """Account value at each scan, against the starting cash."""
    if len(points) < 2:
        return '<p class="muted">The equity chart appears after the second scan.</p>'
    ticks, seen = [], set()
    for i, (t, _) in enumerate(points):  # one label per trading day, at that day's first scan
        if t[:10] not in seen:
            seen.add(t[:10])
            ticks.append((i, _day(t)))
    values = [v for _, v in points]

    def draw(width: int, most: int) -> str:
        return _line_chart(values, [_when(t) for t, _ in points], [_money(v) for v in values], _thin(ticks, most),
                           lambda v: _money(v, False), start_cash, f"Starting {_money(start_cash, False)}",
                           "Account value over time", start_cash * 0.005, width)

    return _both(draw(800, 8), draw(420, 4))


def _lines(reasons: list[str]) -> str:
    marks = {"+": "▲", "-": "▼", "!": "⚠", "=": "•"}
    return "".join(f"<li>{marks[r[0]]} {escape(r[2:])}</li>" for r in reasons)


def _methods_html(methods: list[dict[str, Any]]) -> str:
    """Each method's score and the observations behind it."""
    return "".join(
        f'<h4>{escape(m["name"])}: {m["score"]:+.0f}</h4><ul class="plain">{_lines(m["reasons"])}</ul>'
        for m in methods
    )


def _reasoning(pos: dict[str, Any], opened: bool) -> str:
    if "methods" in pos:
        read = _methods_html(pos["methods"])
    else:  # positions opened before the market-structure methods were added
        read = f'<ul class="plain">{_lines(pos["trend_reasons"])}</ul>'
    notes = "".join(f"<li>{escape(n)}</li>" for n in pos["contract_notes"])
    weights = pos.get("weights", LEGACY_WEIGHTS)
    rows = "".join(
        f'<div>{FACTOR_LABELS[k]}</div><div class="b"><span style="width:{pos["points"][k] / w * 100:.0f}%"></span></div>'
        f'<div class="n">{pos["points"][k]:.0f} / {w}</div>'
        for k, w in weights.items()
    )
    side = "call" if pos["type"] == "call" else "put"
    return (
        f'<details{" open" if opened else ""}><summary>Why this trade</summary>'
        f'<h4>The market read was {pos["bias"]} (score {pos["trend"]:+.0f} on a scale of -100 to +100), '
        f'so the scanner looked for a {side}</h4>{read}'
        f'<h4>Why this contract</h4><ul>{notes}</ul>'
        f'<h4>Score {pos["score"]:.0f} out of 100</h4><div class="pts">{rows}</div>'
        f'<h4>Exit plan set at entry</h4><ul>'
        f'<li>Take profit at {_money(pos["target_price"])} per share or higher</li>'
        f'<li>Stop loss at {_money(pos["stop_price"])} or lower</li>'
        f'<li>Sell on {_day(pos["time_exit"])} if neither has been hit</li></ul></details>'
    )


def _review(pos: dict[str, Any]) -> str:
    """The most recent review (opening or end of day), if the position has had one."""
    if not pos.get("reviews"):
        return ""
    last = pos["reviews"][-1]
    changed = f' Changed since entry: {escape("; ".join(last["changed"]))}.' if last["changed"] else ""
    label = escape(last.get("label", "End-of-day review"))
    return (f'<p class="review"><b>{label}, {_when(last["time"])}: {escape(last["verdict"])}.</b> '
            f'{escape(last["note"])}{changed}</p>')


def _tags(pos: dict[str, Any]) -> str:
    tags = [f"Score {pos['score']:.0f}", pos["bias"].capitalize()]
    if pos["earnings_before_expiry"]:
        tags.append("⚠ Earnings before expiry")
    return '<div class="tags">' + "".join(f'<span class="tag">{t}</span>' for t in tags) + "</div>"


def _open_card(pos: dict[str, Any], commission: float) -> str:
    value = pos["last_bid"] * 100
    pnl = value - commission - pos["cost"]
    span = pos["target_price"] - pos["stop_price"]
    at = min(max((pos["last_bid"] - pos["stop_price"]) / span, 0), 1) * 100
    entry_at = (pos["entry_price"] - pos["stop_price"]) / span * 100
    dte = (date.fromisoformat(pos["expiration"]) - date.today()).days
    stock = pos["last_spot"] / pos["entry_spot"] - 1
    return (
        f'<article class="card"><header><h3>{escape(pos["label"])}</h3>'
        f'<div class="pl">{_delta(pnl, pnl / pos["cost"])}</div></header>{_tags(pos)}'
        f'<div class="range"><div class="bar"><i style="left:{entry_at:.1f}%" title="Entry"></i>'
        f'<u style="left:{at:.1f}%" title="Current bid"></u></div>'
        f'<div class="ends"><span>Stop {_money(pos["stop_price"])}</span>'
        f'<span>Bought {_money(pos["entry_price"])} · now {_money(pos["last_bid"])}</span>'
        f'<span>Target {_money(pos["target_price"])}</span></div></div>'
        f'<div class="facts">'
        f'<div><span>Paid</span><b>{_money(pos["cost"])}</b></div>'
        f'<div><span>Worth now</span><b>{_money(value)}</b></div>'
        f'<div><span>Bought</span><b>{_when(pos["opened"])}</b></div>'
        f'<div><span>Days to expiry</span><b>{dte}</b></div>'
        f'<div><span>{escape(pos["ticker"])} then / now</span><b>{_money(pos["entry_spot"])} / {_money(pos["last_spot"])} ({stock:+.1%})</b></div>'
        f'<div><span>Best / worst bid</span><b>{_money(pos["high_bid"])} / {_money(pos["low_bid"])}</b></div>'
        f'</div>{_review(pos)}{_reasoning(pos, False)}</article>'
    )


def _closed_card(pos: dict[str, Any]) -> str:
    stock = pos["exit_spot"] / pos["entry_spot"] - 1
    return (
        f'<article class="card"><header><h3>{escape(pos["label"])}</h3>'
        f'<div class="pl">{_delta(pos["pnl"], pos["ret"])}</div></header>'
        f'<div class="tags"><span class="tag">{escape(pos["exit_reason"])}</span>'
        f'<span class="tag">Score {pos["score"]:.0f}</span></div>'
        f'<div class="facts">'
        f'<div><span>Bought</span><b>{_money(pos["entry_price"])} · {_when(pos["opened"])}</b></div>'
        f'<div><span>Sold</span><b>{_money(pos["exit_price"])} · {_when(pos["exit_time"])}</b></div>'
        f'<div><span>{escape(pos["ticker"])} moved</span><b>{stock:+.1%}</b></div>'
        f'</div>{_review(pos)}{_reasoning(pos, False)}</article>'
    )


def render(state: dict[str, Any], cfg: dict[str, Any]) -> str:
    start, closed, positions = state["start_cash"], state["closed"], state["positions"]
    total = portfolio.equity(state)
    invested = total - state["cash"]
    realised = sum(c["pnl"] for c in closed)
    wins = sum(c["pnl"] > 0 for c in closed)
    tiles = [
        ("Portfolio value", _money(total), _delta(total - start, total / start - 1) + " since start"),
        ("Cash", _money(state["cash"]), f"{state['cash'] / total:.0%} of the portfolio"),
        ("In open positions", _money(invested), f"{len(positions)} of {cfg['sim_max_positions']} slots used"),
        ("Closed trades", str(len(closed)),
         (f"{wins} won, {len(closed) - wins} lost · {_delta(realised)}" if closed else "None yet")),
    ]
    tile_html = "".join(
        f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div><div class="d">{d}</div></div>'
        for k, v, d in tiles
    )
    open_html = (
        '<div class="cards">' + "".join(_open_card(p, cfg["sim_commission"]) for p in positions) + "</div>"
        if positions else '<p class="muted">No open positions. The scanner buys when a contract scores '
                          f'{cfg["sim_min_score"]} or more and there is cash for it.</p>'
    )
    closed_html = (
        '<div class="cards">' + "".join(_closed_card(c) for c in reversed(closed)) + "</div>"
        if closed else '<p class="muted">Nothing has been sold yet.</p>'
    )
    log_rows = "".join(
        f'<tr><td class="num">{_when(e["time"])}</td><td>{e["action"]}</td><td>{escape(e["label"])}</td>'
        f'<td class="num">{_money(e["price"])}</td><td>{escape(e["note"])}</td></tr>'
        for e in reversed(state["log"][-60:])
    )
    rules = [
        f"Starts with {_money(start, False)} in cash. Nothing here is real money.",
        "Direction comes from a combined read of auction and market profile (value area, point of control), "
        "dealer positioning (gamma walls and flip, vanna, charm, and customer delta exposure estimated from "
        "classified buy and sell flow), Wyckoff structure, volume price analysis, and trend. Bullish reads "
        "look for calls, bearish reads for puts, and mixed reads are skipped.",
        f"Buys one contract when the scanner scores it {cfg['sim_min_score']} or more out of 100, it costs no more "
        f"than {cfg['risk_per_trade_pct']:g}% of the account, and there is cash for it. At most "
        f"{cfg['sim_max_positions']} positions, one per stock.",
        f"Sells when the bid is {cfg['sim_target_pct']:g}% above the purchase price (target), "
        f"{cfg['sim_stop_pct']:g}% below it (stop), or {cfg['sim_exit_dte']} days before expiry.",
        f"Only trades with the broad market: the average read of {' and '.join(cfg['sim_regime_tickers'])}. When it "
        f"is beyond ±{cfg['sim_regime_threshold']}, a trade against it needs a read of at least "
        f"±{cfg['sim_counter_trend_min']} on its own stock. At most {cfg['sim_max_same_direction']} positions bet the same way.",
        f"Buys only after {cfg['sim_first_buy_time']} New York time, when spreads have settled, only when the "
        f"bid/ask spread is {cfg['sim_max_entry_spread']:g}% or less, and at most {cfg['sim_max_buys_per_scan']} per "
        f"scan and {cfg['sim_max_buys_per_day']} per day.",
        f"Twice a day ({' and '.join(cfg['sim_review_times'])} New York time) every position at least "
        f"{cfg['sim_review_min_days']} trading days old is re-checked against the current read. It is sold only if "
        "the read has reversed against it; a read that has merely gone neutral leaves it to its stop and target.",
        f"Buys fill at the ask and sells at the bid, with {_money(cfg['sim_commission'])} commission each way.",
        "Prices come from Yahoo Finance, delayed about 15 minutes, and are checked every 15 minutes while the "
        "market is open. A real stop order could fill at a different price.",
        f"After a sale, the same stock is not bought again for {cfg['sim_reentry_days']} days.",
    ]
    body = f"""<div class="tiles">{tile_html}</div>
<h2>Portfolio value</h2><div class="panel">{_chart(state["equity"], start)}</div>
<h2>Open positions</h2>{open_html}
<h2>Closed trades</h2>{closed_html}
<h2>Activity</h2><div class="panel scroll"><table><thead><tr><th>Time (ET)</th><th>Action</th><th>Contract</th>
<th>Price</th><th>Note</th></tr></thead><tbody>{log_rows or '<tr><td colspan="5" class="muted">No activity yet.</td></tr>'}</tbody></table></div>
<h2>How the simulation trades</h2><div class="panel"><ul>{"".join(f"<li>{escape(r)}</li>" for r in rules)}</ul></div>"""
    sub = (f"Simulated trades from the options scanner. Started {_when(state['started'])} ET · "
           f"last updated {_when(state['updated'])} ET")
    return _page("Portfolio", sub, body, CHART_JS + TABLE_JS)


NAV = (("Home", "HOME", "index.html"), ("Portfolio", "PORT", "portfolio.html"), ("Strategies", "STRAT", "strategies.html"),
       ("Scanner", "SCAN", "scanner.html"), ("Earnings", "EARN", "earnings.html"), ("Learning", "LEARN", "learning.html"))
TAPE_SYMBOLS = ["SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "HYG", "IBIT", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
                "XLP", "XLU", "XLB", "XLC", "XLRE"]
_chrome: dict[str, str] = {"tape": "", "status": "", "clock": ""}


def set_chrome(result: dict[str, Any] | None, state: dict[str, Any], cfg: dict[str, Any]) -> None:
    """Ticker tape, top-bar clock and status bar shared by every page, from the latest scan."""
    if result is None:
        return
    quotes = []
    model = result.get("regime_model") or {}
    if model.get("vix"):
        last, prev = model["vix"]
        quotes.append(("VIX", last, last / prev - 1 if prev else 0.0))
    for sym in TAPE_SYMBOLS:
        info = result["tickers"].get(sym)
        if not info or info.get("indicators") is None or len(info["indicators"]) < 2:
            continue
        closes = info["indicators"]["Close"]
        quotes.append((sym, float(info["spot"]), float(info["spot"]) / float(closes.iloc[-2]) - 1))
    items = "".join(
        f'<span class="q"><span class="sym">{escape(s)}</span>{p:,.2f} '
        f'<span class="{"up" if c >= 0 else "down"}">{"▲" if c >= 0 else "▼"}{c:+.2%}</span></span>'
        for s, p, c in quotes)
    _chrome["tape"] = f'<div class="tape" aria-label="Market quotes"><div class="roll">{items}{items}</div></div>' if items else ""
    scanned = result["scanned_at"]
    _chrome["clock"] = f'LAST SCAN <b>{scanned:%m/%d %H:%M}</b> ET · 15 MIN DELAY'
    regime = portfolio.market_regime(result, cfg)
    total = portfolio.equity(state)
    parts = []
    if model:
        parts.append(f'<span><b>REGIME</b>{escape(model["state"].upper())} {model["score"]:+.0f}</span>')
    if regime["score"] is not None:
        parts.append(f'<span><b>MKT DIR</b>{escape(regime["label"].upper())} {regime["score"]:+.0f}</span>')
    parts.append(f'<span><b>CORE ACCT</b>{_money(total)} '
                 f'<span class="{"up" if total >= state["start_cash"] else "down"}">{total / state["start_cash"] - 1:+.1%}</span></span>')
    parts.append(f'<span><b>OPEN</b>{len(state["positions"])}</span>')
    parts.append(f'<span><b>FLAGGED</b>{len(result["contracts"])}</span>')
    _chrome["status"] = f'<div class="statusbar">{"".join(parts)}</div>'


def _page(active: str, sub: str, body: str, script: str) -> str:
    nav = "".join(
        f'<a href="{href}"{" aria-current=\"page\"" if name == active else ""}><b>{n}</b>{code}</a>'
        for n, (name, code, href) in enumerate(NAV, 1)
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex">
<meta http-equiv="refresh" content="300">
<meta name="theme-color" content="#000000">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&display=swap" rel="stylesheet">
<title>Options {active}</title><style>{CSS}</style></head>
<body>
<div class="topbar"><span class="brand">OPTIONS TERMINAL</span><span class="clock">{_chrome["clock"]}</span></div>
{_chrome["tape"]}
<main>
<nav aria-label="Pages">{nav}</nav>
<h1>{TITLES[active]}</h1>
<p class="sub">{sub}</p>
{body}
{_chrome["status"]}
<footer>Simulation for education only. Not financial advice, and not a record of real trades.</footer>
</main><script>{script}</script></body></html>"""


def _buy_status(row: Any, state: dict[str, Any], cfg: dict[str, Any], resting: set[str],
                info: dict[str, Any], regime: dict[str, Any],
                limits: tuple[int, int, str | None] | None = None) -> tuple[str, bool]:
    """Why a flagged contract is or is not in the portfolio, and whether it is buy grade."""
    held = {p["symbol"] for p in state["positions"]}
    held_tickers = {p["ticker"] for p in state["positions"]}
    grade = row["score"] >= cfg["sim_min_score"] and not row["stale"]
    if row["contractSymbol"] in held:
        return "In portfolio", grade
    if row["stale"]:
        return "No live quote", False
    if not grade:
        return f"Below {cfg['sim_min_score']}", False
    if row["ticker"] in held_tickers:
        return f"Buy grade · holding {row['ticker']}", True
    if row["ticker"] in resting:
        return "Buy grade · cooling off", True
    max_positions, max_same_way, _ = limits or (cfg["sim_max_positions"], cfg["sim_max_same_direction"], None)
    if len(state["positions"]) >= max_positions:
        return "Buy grade · slots full", True
    blocked = portfolio.entry_block(row, info, regime, cfg)
    if blocked:
        return f"Buy grade · {blocked}", True
    if sum(1 for p in state["positions"] if p["type"] == row["type"]) >= max_same_way:
        return f"Buy grade · enough {row['type']}s held", True
    if row["ask"] * 100 + cfg["sim_commission"] > state["cash"]:
        return "Buy grade · no cash", True
    return "Buy grade", True


def _cell(value: Any, text: str, numeric: bool = True) -> str:
    return f'<td{" class=\"num\"" if numeric else ""} data-v="{escape(str(value))}">{text}</td>'


def _surface_table(surf: dict[str, Any] | None) -> str:
    """Implied volatility by expiry and delta, shaded light to dark as volatility rises."""
    if not surf or not surf.get("grid"):
        return ""
    cols = [("p10", "10Δ put"), ("p25", "25Δ put"), ("atm", "ATM"), ("c25", "25Δ call"), ("c10", "10Δ call")]
    values = [r[k] for r in surf["grid"] for k, _ in cols if r.get(k) == r.get(k) and r.get(k) is not None]
    lo, hi = (min(values), max(values)) if values else (0, 1)

    def cell(v: float | None) -> str:
        if v is None or v != v:
            return '<td class="num muted">n/a</td>'
        shade = 0.08 + 0.5 * ((v - lo) / (hi - lo) if hi > lo else 0.5)
        return (f'<td class="num" style="background:color-mix(in srgb,var(--series) {shade * 100:.0f}%,'
                f'transparent)">{v:.0%}</td>')

    rows = "".join(
        f'<tr><td>{_day(r["expiration"])}</td><td class="num">{r["dte"]}</td>{"".join(cell(r.get(k)) for k, _ in cols)}</tr>'
        for r in surf["grid"]
    )
    facts = []
    if surf.get("rr25") == surf.get("rr25") and surf.get("rr25") is not None:
        facts.append(f"30-day 25-delta risk reversal {surf['rr25'] * 100:+.1f} vol points")
    if surf.get("term_slope") == surf.get("term_slope") and surf.get("term_slope") is not None:
        facts.append(f"60-day against front-month IV {surf['term_slope']:+.0%}")
    return (
        '<h4>Volatility surface (implied volatility by delta)</h4><div class="scroll"><table><thead><tr>'
        '<th>Expiry</th><th>Days</th>' + "".join(f"<th>{label}</th>" for _, label in cols)
        + f'</tr></thead><tbody>{rows}</tbody></table></div>'
        + (f'<p class="muted small">{escape("; ".join(facts))}. Darker cells are higher volatility.</p>' if facts else "")
    )


QUADRANTS = {  # name: (CSS colour token, corner of the graph)
    "leading": ("--q-lead", "top right"), "weakening": ("--q-weak", "bottom right"),
    "lagging": ("--q-lag", "bottom left"), "improving": ("--q-imp", "top left"),
}


def _rotation_graph(rot: dict[str, Any], mag7: dict[str, Any] | None = None) -> str:
    """Relative rotation graph of the sectors, with a toggle to the same graph for the Mag 7 stocks."""
    views = [(key, label, what, r) for key, label, what, r in
             (("sec", "Sectors", "sector ETFs", rot), ("mag7", "Mag 7", "Magnificent 7 stocks", mag7 or {}))
             if any(v.get("tail") for v in r.values())]
    if not views:
        return ""
    legend = "".join(
        f'<span><i style="background:var({QUADRANTS[q][0]})"></i>{q.capitalize()}</span>'
        for q in ("leading", "weakening", "lagging", "improving"))
    panes = []
    for key, label, what, r in views:
        tails = {name: v["tail"] for name, v in r.items() if v.get("tail")}
        unit = "sector" if key == "sec" else "stock"
        panes.append(
            f'<div class="rg-pane rg-{key}">'
            f'{_both(_rotation_svg(r, tails, (760, 560, 64, 24, 24, 56), key + "d", what), _rotation_svg(r, tails, (420, 440, 44, 12, 18, 44), key + "m", what))}'
            f'<p class="muted small">Each line is a {unit}&#39;s last {len(next(iter(tails.values()))) - 1} weeks against '
            f'SPY; the large dot is now, labelled with its ticker. They rotate clockwise: leading, then weakening, lagging '
            f'and improving, then back to leading. Hover over (or tap) a dot for its values.</p></div>')
    toggle = ""
    if len(views) > 1:
        toggle = "".join(
            f'<input type="radio" name="rg" id="rg-{key}" class="rg-{key}-in"{" checked" if i == 0 else ""}>'
            f'<label for="rg-{key}">{label}</label>' for i, (key, label, _, _) in enumerate(views))
        toggle = f'<div class="rg-toggle" role="radiogroup" aria-label="Rotation graph">{toggle}</div>'
    return f'<div class="panel rg">{toggle}<div class="legend">{legend}</div>{"".join(panes)}</div>'


def _rotation_svg(rot: dict[str, Any], tails: dict[str, list], size: tuple, prefix: str,
                  what: str = "sector ETFs") -> str:
    """One drawing of the rotation graph at the given (width, height, left, right, top, bottom)."""
    W, H, L, R, T, B = size
    narrow = W < 600
    xs = [p["ratio"] for t in tails.values() for p in t]
    ys = [p["momentum"] for t in tails.values() for p in t]
    xspan = max(max(abs(v - 100) for v in xs) * 1.15, 1.0)
    yspan = max(max(abs(v - 100) for v in ys) * 1.15, 1.0)

    def x(v: float) -> float:
        return L + (W - L - R) * (v - (100 - xspan)) / (2 * xspan)

    def y(v: float) -> float:
        return T + (H - T - B) * (1 - (v - (100 - yspan)) / (2 * yspan))

    cx, cy = x(100), y(100)
    parts = []
    regions = {"leading": (cx, T, W - R - cx, cy - T), "weakening": (cx, cy, W - R - cx, H - B - cy),
               "lagging": (L, cy, cx - L, H - B - cy), "improving": (L, T, cx - L, cy - T)}
    for name, (rx, ry, rw, rh) in regions.items():
        token = QUADRANTS[name][0]
        parts.append(f'<rect x="{rx:.1f}" y="{ry:.1f}" width="{rw:.1f}" height="{rh:.1f}" fill="var({token})" opacity=".07"/>')
        tx = rx + 10 if name in ("lagging", "improving") else rx + rw - 10
        ty = ry + 20 if name in ("leading", "improving") else ry + rh - 10
        anchor = "start" if name in ("lagging", "improving") else "end"
        parts.append(f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="{anchor}" font-size="13" font-weight="600" '
                     f'fill="var(--ink2)">{name.capitalize()}</text>')
    for k in range(-2, 3):
        gx, gy = x(100 + xspan * k / 2.5), y(100 + yspan * k / 2.5)
        parts.append(f'<line x1="{gx:.1f}" x2="{gx:.1f}" y1="{T}" y2="{H - B}" stroke="var(--grid)"/>'
                     f'<text x="{gx:.1f}" y="{H - B + 16}" text-anchor="middle">{100 + xspan * k / 2.5:.1f}</text>')
        parts.append(f'<line x1="{L}" x2="{W - R}" y1="{gy:.1f}" y2="{gy:.1f}" stroke="var(--grid)"/>'
                     f'<text x="{L - 8}" y="{gy + 4:.1f}" text-anchor="end">{100 + yspan * k / 2.5:.1f}</text>')
    parts.append(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{T}" y2="{H - B}" stroke="var(--axis)" stroke-width="1.5"/>'
                 f'<line x1="{L}" x2="{W - R}" y1="{cy:.1f}" y2="{cy:.1f}" stroke="var(--axis)" stroke-width="1.5"/>')
    x_title = "RS ratio (above 100 is stronger)" if narrow else "RS ratio: strength against SPY (above 100 is stronger)"
    y_title = "RS momentum" if narrow else "RS momentum (above 100 is rising)"
    parts.append(f'<text x="{(L + W - R) / 2:.1f}" y="{H - 10}" text-anchor="middle" fill="var(--ink2)">{x_title}</text>'
                 f'<text transform="translate(12 {(T + H - B) / 2:.1f}) rotate(-90)" text-anchor="middle" '
                 f'fill="var(--ink2)">{y_title}</text>')

    markers = "".join(
        f'<marker id="arrow-{prefix}-{q}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="var({token})"/></marker>'
        for q, (token, _) in QUADRANTS.items())
    parts.insert(0, f"<defs>{markers}</defs>")
    heads = []
    for name, tail in sorted(tails.items(), key=lambda kv: rot[kv[0]]["rank"], reverse=True):
        r = rot[name]
        token = QUADRANTS[r["quadrant"]][0]
        coords = [(x(p["ratio"]), y(p["momentum"])) for p in tail]
        n = len(coords) - 1
        for i in range(n):  # older weeks fade, the latest segment carries the direction arrow
            (x1, y1), (x2, y2) = coords[i], coords[i + 1]
            arrow = ""
            if i == n - 1:  # stop at the head dot's edge so the arrowhead is visible
                length = max(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5, 1e-6)
                if length > 12:
                    x2, y2 = x2 - (x2 - x1) / length * 9, y2 - (y2 - y1) / length * 9
                    arrow = f' marker-end="url(#arrow-{prefix}-{r["quadrant"]})"'
            parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="var({token})" '
                         f'stroke-width="2" stroke-linecap="round" opacity="{0.25 + 0.55 * (i + 1) / n:.2f}"{arrow}/>')
        for i, p in enumerate(tail):
            head = i == n
            tip = f"{r['etf']} {name}, week of {p['week']}: RS ratio {p['ratio']:.2f}, momentum {p['momentum']:.2f}"
            if head:
                tip += f" ({r['quadrant']})"
            parts.append(
                f'<circle cx="{coords[i][0]:.1f}" cy="{coords[i][1]:.1f}" r="{7 if head else 2.5}" fill="var({token})" '
                f'stroke="var(--surface)" stroke-width="2" opacity="{1 if head else 0.25 + 0.5 * i / n:.2f}">'
                f'<title>{escape(tip)}</title></circle>')
        heads.append((coords[-1], r["etf"]))

    # Labels: try right, left, above and below each head; keep the first spot that clears every
    # label already placed and every head dot.
    placed: list[tuple[float, float, float, float]] = []
    dots = [c for c, _ in heads]
    for (hx, hy), etf in heads:
        width = 8 * len(etf) + 4
        for dx, dy, anchor in ((11, 4, "start"), (-11, 4, "end"), (-width / 2, -12, "start"), (-width / 2, 21, "start"),
                               (11, -10, "start"), (11, 17, "start"), (-11, -10, "end"), (-11, 17, "end")):
            left = hx + dx - (width if anchor == "end" else 0)
            box = (left, hy + dy - 11, left + width, hy + dy + 3)
            clear = all(box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3] for b in placed)
            clear = clear and all(not (box[0] - 6 < cx < box[2] + 6 and box[1] - 6 < cy < box[3] + 6)
                                  for cx, cy in dots if (cx, cy) != (hx, hy))
            if clear:
                break
        placed.append(box)
        parts.append(f'<text x="{hx + dx:.1f}" y="{hy + dy:.1f}" text-anchor="{anchor}" font-size="12" font-weight="600" '
                     f'fill="var(--ink)" stroke="var(--surface)" stroke-width="3" paint-order="stroke">{escape(etf)}</text>')

    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Relative rotation graph of the {what}" '
            f'font-size="11" fill="var(--muted)" style="width:100%;height:auto;display:block">{"".join(parts)}</svg>')


def _rotation_table(rot: dict[str, Any]) -> str:
    """Sectors ranked by one-month strength against SPY, with their rotation quadrant."""
    if not rot:
        return '<p class="muted">Sector data unavailable on this scan.</p>'
    marks = {"leading": "▲ Leading", "improving": "↗ Improving", "weakening": "↘ Weakening", "lagging": "▼ Lagging"}
    def pct(value: float) -> str:
        return _cell(value, f"{value:+.1%}")

    def level(value: float) -> str:
        return _cell(value, f"{value:.1f}")

    rows = "".join(
        f'<tr>{_cell(r["rank"], str(r["rank"]))}{_cell(name, escape(name), False)}{_cell(r["etf"], r["etf"], False)}'
        f'{_cell(r["quadrant"], marks[r["quadrant"]], False)}{level(r["ratio"])}{level(r["rs_momentum"])}'
        f'{pct(r["rs_1m"])}{pct(r["rs_3m"])}{pct(r["return_1m"])}</tr>'
        for name, r in sorted(rot.items(), key=lambda kv: kv[1]["rank"])
    )
    return ('<div class="panel scroll"><table class="sortable"><thead><tr><th>Rank</th><th>Sector</th><th>ETF</th>'
            '<th>Quadrant</th><th>RS ratio</th><th>RS momentum</th><th>vs SPY, 1 month</th><th>vs SPY, 3 months</th>'
            '<th>Return, 1 month</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>'
            '<p class="muted small">Leading: stronger than SPY and gaining. Improving: weaker but gaining. Weakening: '
            'stronger but losing ground. Lagging: weaker and losing ground. RS ratio and momentum are the '
            'graph&#39;s coordinates (100 is the market).</p>')


def render_scanner(result: dict[str, Any], state: dict[str, Any], cfg: dict[str, Any]) -> str:
    contracts, tickers = result["contracts"], result["tickers"]
    regime = portfolio.market_regime(result, cfg)
    limits = portfolio.exposure_limits(result, cfg)
    cutoff = (datetime.now() - timedelta(days=cfg["sim_reentry_days"])).isoformat()
    resting = {c["ticker"] for c in state["closed"] if c["exit_time"] > cutoff}

    rows, buy_grade = [], 0
    for _, r in contracts.iterrows():
        status, grade = _buy_status(r, state, cfg, resting, tickers[r["ticker"]], regime, limits)
        buy_grade += grade
        spread = "n/a" if r["spread_pct"] != r["spread_pct"] else f"{r['spread_pct']:.1f}%"
        ratio = "n/a" if r["iv_hv"] != r["iv_hv"] else f"{r['iv_hv']:.2f}"
        warn = " ⚠" if r["earnings_before_expiry"] else ""
        rows.append(
            f'<tr data-t="{escape(r["ticker"])}" data-b="{int(grade)}">'
            + _cell(r["score"], f"{r['score']:.0f}")
            + _cell(r["ticker"], f'<a href="#read-{escape(r["ticker"])}">{escape(r["ticker"])}</a>', False)
            + _cell(r["type"], r["type"].capitalize(), False)
            + _cell(r["strike"], f"${r['strike']:g}")
            + _cell(r["moneyness"], f"{r['moneyness']:+.1%}")
            + _cell(r["expiration"], _day(r["expiration"]) + warn)
            + _cell(r["cost"], _money(r["cost"], False))
            + _cell(status, escape(status), False)
            + _cell(r["dte"], str(int(r["dte"])))
            + _cell(r["delta"], f"{r['delta']:+.2f}")
            + _cell(abs(r["theta"]), f"−${abs(r['theta']) * 100:.2f}")
            + _cell(r["iv"], f"{r['iv']:.0%}")
            + _cell(r["iv_hv"], ratio)
            + _cell(r["breakeven_move"], f"{r['breakeven_move']:.1%}")
            + _cell(r["pop"], f"{r['pop']:.0%}")
            + _cell(r["spread_pct"], spread)
            + _cell(r["openInterest"], f"{int(r['openInterest']):,}") + "</tr>"
        )
    heads = ["Score", "Ticker", "Type", "Strike", "OTM", "Expiry", "Cost", "Status", "Days", "Delta", "Theta/day", "IV",
             "IV/HV", "Breakeven move", "Chance of profit", "Spread", "Open interest"]
    options = "".join(f"<option>{escape(t)}</option>" for t in sorted(contracts["ticker"].unique()))
    table = (
        '<div class="filters"><label>Ticker <select id="f-ticker"><option value="">All</option>'
        f'{options}</select></label><label><input type="checkbox" id="f-buy"> Buy grade only</label>'
        '<span class="muted small" id="f-count"></span></div>'
        '<div class="panel scroll"><table class="sortable" id="flagged"><thead><tr>'
        + "".join(f"<th>{h}</th>" for h in heads) + f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
        '<p class="muted small">Select a column heading to sort. ⚠ marks an expiry that falls after the next '
        'earnings date. OTM is how far the stock must move to reach the strike (negative means in the money). '
        'Cost is the ask price for one contract. Buy grade means the score is high enough to buy; '
        'the note after it says why the portfolio has not.</p>'
        if rows else '<p class="muted">No contracts passed the filters on the last scan.</p>'
    )

    keys = list(METHOD_SHORT)
    read_rows, cards = [], []
    for t, info in sorted(tickers.items(), key=lambda kv: -kv[1]["trend"]):
        scores = {m["key"]: m["score"] for m in info["methods"]}
        lv = info["levels"]
        arrow = {"bullish": "▲ Bullish", "bearish": "▼ Bearish", "neutral": "• Neutral"}[info["bias"]]

        def level(name: str) -> str:
            value = lv.get(name)
            return _cell(value if value is not None else "", "n/a" if value is None else f"${value:,.2f}")

        read_rows.append(
            "<tr>" + _cell(t, f'<a href="#read-{escape(t)}">{escape(t)}</a>', False)
            + _cell(info["spot"], _money(info["spot"]))
            + _cell(info["trend"], f"{info['trend']:+.0f}") + _cell(info["bias"], arrow, False)
            + "".join(_cell(scores.get(k, ""), f"{scores[k]:+.0f}" if k in scores else "n/a") for k in keys)
            + _cell(lv.get("phase", ""), escape(lv.get("phase", "n/a")), False)
            + level("put_wall") + level("flip") + level("call_wall") + level("val") + level("poc") + level("vah")
            + _cell(len(info["contracts"]), str(len(info["contracts"]))) + "</tr>"
        )
        cards.append(
            f'<article class="card" id="read-{escape(t)}"><header><h3>{escape(t)} · {_money(info["spot"])}</h3>'
            f'<div class="pl">{arrow} {info["trend"]:+.0f}</div></header>'
            f'<details><summary>Reasoning</summary>{_methods_html(info["methods"])}'
            f'{_surface_table(info.get("surface"))}</details></article>'
        )
    read_heads = ["Ticker", "Price", "Read", "Bias", *METHOD_SHORT.values(), "Wyckoff phase", "Put wall",
                  "Gamma flip", "Call wall", "Value low", "Point of control", "Value high", "Flagged"]

    tiles = [
        ("Contracts flagged", str(len(contracts)), f"from {contracts['ticker'].nunique()} of {len(tickers)} tickers"),
        ("Buy grade", str(buy_grade), f"score {cfg['sim_min_score']} or more with a live quote"),
        ("Top score", f"{contracts['score'].max():.0f}" if len(contracts) else "None", "out of 100"),
        ("Max cost per contract", _money(cfg["account_size"] * cfg["risk_per_trade_pct"] / 100, False),
         f"{cfg['risk_per_trade_pct']:g}% of the account"),
    ]
    tile_html = "".join(
        f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div><div class="d">{d}</div></div>'
        for k, v, d in tiles
    )
    errors = (
        '<p class="notice">Could not scan: ' + escape(", ".join(result["errors"])) + "</p>" if result["errors"] else ""
    )
    body = f"""<div class="tiles">{tile_html}</div>{errors}
<h2>All flagged contracts</h2>{table}
<h2>Market read by ticker</h2>
<div class="panel scroll"><table class="sortable"><thead><tr>{"".join(f"<th>{h}</th>" for h in read_heads)}</tr></thead>
<tbody>{"".join(read_rows)}</tbody></table></div>
<p class="muted small">Read and method scores run from −100 (bearish) to +100 (bullish). A ticker needs a read of
at least ±{cfg['min_trend_strength']} before any contract is flagged.</p>
<h2>Sector rotation</h2>{_rotation_graph(result.get("rotation") or {}, result.get("mag7_rotation"))}{_rotation_table(result.get("rotation") or {})}
<h2>Reasoning by ticker</h2><div class="cards">{"".join(cards)}</div>"""
    built = universe.load()
    origin = (f" Universe from {escape(built['source'])}, built {_when(built['built'])} ET."
              if built.get("source") else "")
    sub = (f"Every contract that passed the filters on the last scan, whether or not the portfolio bought it, "
           f"across {len(tickers)} liquid stocks and ETFs. Last scan {_when(result['scanned_at'].isoformat())} ET."
           f"{origin}")
    return _page("Scanner", sub, body, TABLE_JS)


TITLES = {"Home": "Options Dashboard", "Portfolio": "Options Paper Portfolio", "Strategies": "Strategy Race",
          "Scanner": "Options Scanner",
          "Earnings": "Earnings Outlook", "Learning": "What's Working"}
LEANS = {"bullish": "▲ Bullish", "bearish": "▼ Bearish", "neutral": "• Neutral"}


def _pct(x: float | None, digits: int = 1, signed: bool = False) -> str:
    if x is None:
        return "n/a"
    return f"{x:+.{digits}%}" if signed else f"{x:.{digits}%}"


def _price(x: float | None) -> str:
    return "n/a" if x is None else f"{'-' if x < 0 else ''}${abs(x):.2f}"


def _play_text(play: dict[str, Any]) -> str:
    legs = " + ".join(f"${leg['strike']:g} {leg['type']}" for leg in play["legs"])
    return legs if play["kind"] == "OTM" else f"{play['kind']}: {legs}"


def _research_note(note: dict[str, Any]) -> str:
    points = "".join(f"<li>{escape(x)}</li>" for x in note.get("points", []))
    sources = "".join(f'<li><a href="{escape(s["url"])}">{escape(s["title"])}</a></li>' for s in note.get("sources", []))
    return (
        f'<div class="review"><b>Claude&#39;s research note ({_day(note["written"])}): {escape(note["view"])}.</b> '
        f'{escape(note["summary"])}{f"<ul>{points}</ul>" if points else ""}'
        f'{f"<details><summary>Sources</summary><ul>{sources}</ul></details>" if sources else ""}</div>'
    )


def _plays_table(o: dict[str, Any]) -> str:
    if not o["plays"]:
        if o["implied_move"] is None:
            return '<p class="muted small">No option expiry falls close enough after the report to isolate it.</p>'
        return '<p class="muted small">No out-of-the-money contract meets the liquidity limits.</p>'
    rows = "".join(
        f'<tr><td>{escape(_play_text(p))}</td><td class="num">{_money(p["cost"], False)}</td>'
        f'<td class="num">{" / ".join(f"{leg['delta']:+.2f}" for leg in p["legs"])}</td>'
        f'<td class="num">{_pct(p["breakeven_move"], signed=True)}</td>'
        f'<td class="num">{p["wins"]} of {p["tests"]}</td>'
        f'<td class="num">{"n/a" if p["avg_result"] is None else _signed(p["avg_result"], False)}</td></tr>'
        for p in o["plays"]
    )
    edge = any((p["avg_result"] or 0) > 0 for p in o["plays"])
    return (
        f'<h4>Out-of-the-money plays, {_day(o["expiry"])} expiry</h4>'
        '<div class="scroll"><table><thead><tr><th>Position</th><th>Cost</th><th>Delta</th><th>Breakeven</th>'
        f'<th>Paid off in past reports</th><th>Average result</th></tr></thead><tbody>{rows}</tbody></table></div>'
        '<p class="muted small">Tested by applying each past report&#39;s move to today&#39;s price and holding to '
        f'expiry (intrinsic value only).{"" if edge else " None of these would have made money on average."}</p>'
    )


def _history_table(o: dict[str, Any]) -> str:
    if not o["history"]:
        return ""
    rows = "".join(
        f'<tr><td>{_day(h["date"])}</td><td class="num">{_price(h["estimate"])}</td>'
        f'<td class="num">{_price(h["actual"])}</td>'
        f'<td class="num">{"n/a" if h["surprise"] is None else f"{h['surprise']:+.1f}%"}</td>'
        f'<td class="num">{_pct(h["move"], signed=True)}</td></tr>'
        for h in reversed(o["history"])
    )
    return ('<h4>Past reports</h4><div class="scroll"><table><thead><tr><th>Date</th><th>EPS estimate</th>'
            f'<th>Reported</th><th>Surprise</th><th>Stock reaction</th></tr></thead><tbody>{rows}</tbody></table></div>')


def _earnings_card(o: dict[str, Any], note: dict[str, Any] | None) -> str:
    factors = "".join(
        f'<li>{"▲" if f["points"] > 0 else "▼" if f["points"] < 0 else "•"} <b>{escape(f["name"])} '
        f'({f["points"]:+.0f})</b>: {escape(f["text"])}</li>'
        for f in o["factors"]
    )
    pricing = (f'<h4>Options pricing: {escape(o["pricing"]["verdict"])}</h4>'
               f'<p class="small">{escape(o["pricing"]["text"])}, on the {_day(o["expiry"])} expiry.</p>'
               if o["pricing"] else "")
    return (
        f'<article class="card" id="earn-{escape(o["ticker"])}"><header><h3>{escape(o["ticker"])} · '
        f'{_day(o["date"])}, {escape(o["timing"])}</h3><div class="pl">{LEANS[o["lean"]]} {o["score"]:+.0f}</div></header>'
        '<div class="facts">'
        f'<div><span>Chance of beating EPS</span><b>{_pct(o["beat_probability"], 0)}</b></div>'
        f'<div><span>EPS estimate</span><b>{_price(o["eps_estimate"])}</b></div>'
        f'<div><span>Implied move</span><b>{_pct(o["implied_move"])}</b></div>'
        f'<div><span>Typical move</span><b>{_pct(o["historical_move"])}</b></div></div>'
        f'{_research_note(note) if note else ""}'
        f'<details><summary>Model, options and history</summary>'
        f'<h4>Model factors: {o["score"]:+.0f} in total</h4><ul class="plain">{factors}</ul>'
        f'{pricing}{_plays_table(o)}{_history_table(o)}</details></article>'
    )


def render_earnings(result: dict[str, Any], cfg: dict[str, Any]) -> str:
    outlooks = sorted((r["earnings_outlook"] for r in result["tickers"].values() if r.get("earnings_outlook")),
                      key=lambda o: (o["date"], o["ticker"]))
    research = earnings.notes()
    rows = []
    for o in outlooks:
        best = o["plays"][0] if o["plays"] else None
        note = research.get(o["ticker"])
        rows.append(
            "<tr>" + _cell(o["date"], _day(o["date"])) + _cell(o["timing"], escape(o["timing"]), False)
            + _cell(o["ticker"], f'<a href="#earn-{escape(o["ticker"])}">{escape(o["ticker"])}</a>', False)
            + _cell(o["score"], f'{LEANS[o["lean"]][0]} {o["score"]:+.0f}')
            + _cell(o["beat_probability"] or "", _pct(o["beat_probability"], 0))
            + _cell(o["implied_move"] or "", _pct(o["implied_move"]))
            + _cell(o["historical_move"] or "", _pct(o["historical_move"]))
            + _cell(o["pricing"]["ratio"] if o["pricing"] else "",
                    escape(o["pricing"]["verdict"]) if o["pricing"] else "n/a", False)
            + _cell(best["avg_result"] if best else "", escape(_play_text(best)) if best else "none", False)
            + _cell(note["view"] if note else "", escape(note["view"]) if note else "—", False) + "</tr>"
        )
    heads = ["Date", "Timing", "Ticker", "Model lean", "Beat chance", "Implied move", "Typical move", "Options",
             "Best tested play", "Research view"]
    table = (
        '<div class="panel scroll"><table class="sortable"><thead><tr>' + "".join(f"<th>{h}</th>" for h in heads)
        + f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
        if rows else f'<p class="muted">No stock in the universe reports in the next {cfg["earnings_window_days"]} days.</p>'
    )
    cards = "".join(_earnings_card(o, research.get(o["ticker"])) for o in outlooks)
    method = [
        "The model lean (−100 to +100) adds up: the record against EPS estimates over the last 8 reports, the "
        "90-day change in the consensus estimate and recent revisions, expected revenue growth, the margin trend, "
        "how the stock has reacted to past reports, a large run-up or sell-off into the date, and the current "
        "market read. ±20 or more counts as a lean.",
        "Beat chance starts from the stock's own beat rate, pulled toward the roughly 70% norm for large companies, "
        "and is nudged by estimate revisions. Beating estimates does not mean the stock rises.",
        "Implied move is the at-the-money straddle on the first expiry after the report, divided by the price. "
        "Typical move is the average close-to-close reaction over the last 8 reports. Options are cheap below "
        "0.85x the typical move and rich above 1.2x. Rich options usually lose value after the report "
        "(volatility crush) even when the stock moves.",
        "Plays are out-of-the-money contracts on that expiry in the lean's direction (both sides when there is no "
        "lean, plus a strangle when options are also cheap), at any price. Each is tested against "
        "this stock's last 8 reactions, which is a small sample.",
        "Research notes are written by Claude from public sources on the date shown and are not refreshed "
        "automatically.",
    ]
    body = f"""<h2>Upcoming reports</h2>{table}
<h2>By company</h2><div class="cards">{cards}</div>
<h2>How the outlook works</h2><div class="panel"><ul>{"".join(f"<li>{escape(m)}</li>" for m in method)}</ul></div>"""
    sub = (f"Stocks in the scan universe reporting in the next {cfg['earnings_window_days']} days. "
           f"Last scan {_when(result['scanned_at'].isoformat())} ET.")
    return _page("Earnings", sub, body, TABLE_JS)


FACTOR_NAMES = {
    "trend": "Strength of the market read", "liquidity": "Liquidity", "breakeven": "Breakeven vs expected move",
    "iv_value": "Option price vs volatility", "gamma": "Gamma backdrop", "theta": "Low time decay",
    "target": "Breakeven inside structural target",
}


def _dollars(x: float | None) -> str:
    return "n/a" if x is None else _signed(x, False)


def _edge_rows(table: dict[str, Any], names: dict[str, str], pct: bool) -> str:
    def weight(w: float) -> str:
        return f"{w:.0%}" if pct else f"{w:g}"

    rows = []
    for k, v in table.items():
        changed = abs(v["active"] - v["base"]) > 1e-9
        t = "n/a" if v.get("t") is None else f"{v['t']:+.1f}"
        rows.append(
            f'<tr><td>{escape(names.get(k, k))}</td><td class="num">{weight(v["base"])}</td>'
            f'<td class="num">{weight(v["active"])}{" (learned)" if changed else ""}</td>'
            f'<td class="num">{_dollars(v.get("edge"))}</td><td class="num">{t}</td>'
            f'<td class="num">{v.get("days", 0)}</td></tr>')
    return "".join(rows)


def _summary_rows(items: list[tuple[str, dict[str, Any]]]) -> str:
    return "".join(
        f'<tr><td>{escape(label)}</td><td class="num">{s["trades"]}</td><td class="num">{_dollars(s["avg"])}</td>'
        f'<td class="num">{_pct(s["win_rate"], 0)}</td><td class="num">{_dollars(s["total"])}</td></tr>'
        for label, s in items)


def render_learning(card: dict[str, Any], cfg: dict[str, Any]) -> str:
    if not card:
        body = ('<p class="muted">The journal starts filling at the next scan in market hours. The first contracts '
                'are graded five trading days after they are flagged.</p>')
        return _page("Learning", "What has made money, and how the scanner adjusts to it.", body, TABLE_JS)
    c, tuning = card["collected"], card["tuning"]
    tiles = [
        ("Contracts graded", f"{c['scanner']:,}", f"over {c['sessions']} sessions"),
        ("Being followed", f"{c['following']:,}", "graded after 5 trading days"),
        ("Earnings plays graded", f"{c['earnings_plays']:,}", "held through the report"),
        ("Learning", "On" if tuning["unlocked"] else "Locked", f"{tuning['days']} of 20 sessions needed"
         if not tuning["unlocked"] else "weekly, profit-tested"),
    ]
    tile_html = "".join(f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div><div class="d">{d}</div></div>'
                        for k, v, d in tiles)
    heads = '<tr><th>Signal</th><th>Original weight</th><th>Weight in use</th><th>Profit edge per trade</th><th>t</th><th>Days</th></tr>'
    summary_heads = '<tr><th>Group</th><th>Trades</th><th>Average profit</th><th>Win rate</th><th>Total</th></tr>'

    picks = card["picks"]
    active_total = sum(p["active"] for p in picks)
    control_total = sum(p["control"] for p in picks)
    picks_text = (f"Each day, the top five contracts (one per stock) under the weights in use made "
                  f"{_dollars(active_total)} in total over {len(picks)} sessions, against {_dollars(control_total)} "
                  f"for the original weights." if picks else "No graded sessions yet.")

    exits = "".join(
        f'<tr><td class="num">+{e["target"]:.0%}</td><td class="num">-{e["stop"]:.0%}</td><td class="num">{e["trades"]}</td>'
        f'<td class="num">{_dollars(e["avg"])}</td><td class="num">{_pct(e["win_rate"], 0)}</td>'
        f'<td>{"in use" if e["current"] else ""}</td></tr>' for e in card["exits"])
    best = max((e for e in card["exits"] if e["avg"] is not None), key=lambda e: e["avg"], default=None)

    earn = card["earnings"]
    earn_rows = "".join(
        f'<tr><td>{_day(r["date"])}</td><td>{escape(r["ticker"])}</td><td>{escape(r["lean"])} ({r["score"]:+.0f})</td>'
        f'<td class="num">{_pct(r["beat_probability"], 0)}</td><td>{"beat" if r["beat"] else "missed"}</td>'
        f'<td class="num">{_pct(r["move"], signed=True)}</td><td class="num">{_pct(r["implied"])}</td></tr>'
        for r in earn["resolved"][:30])
    log_rows = "".join(f'<tr><td>{_day(e["date"])}</td><td>{escape(e["change"])}</td><td>{escape(e["reason"])}</td></tr>'
                       for e in reversed(card["log"]))

    body = f"""<div class="tiles">{tile_html}</div>
<p class="notice">{escape(tuning["message"] or "")}</p>
<h2>Learned picks against the original weights</h2><div class="panel"><p>{escape(picks_text)}</p></div>
<h2>Factors</h2><div class="panel scroll"><table><thead>{heads}</thead><tbody>{_edge_rows(card["factors"], FACTOR_NAMES, False)}</tbody></table></div>
<p class="muted small">Profit edge: each day, the average profit of the contracts this factor rated in its top fifth minus its
bottom fifth, averaged over days. t above 2 or below -2 is unlikely to be chance. Profit is for one contract bought at the ask
and sold at the bid at the target, the stop or after five trading days, less commission.</p>
<h2>Market read methods</h2><div class="panel scroll"><table><thead>{heads}</thead><tbody>{_edge_rows(card["methods"], structure.METHOD_NAMES, True)}</tbody></table></div>
<h2>Profit by score</h2><div class="panel scroll"><table><thead>{summary_heads}</thead><tbody>{_summary_rows([(b["band"], b) for b in card["bands"]])}</tbody></table></div>
<h2>Profit by moneyness</h2><div class="panel scroll"><table><thead>{summary_heads}</thead><tbody>{_summary_rows(sorted(card["moneyness"].items()))}</tbody></table></div>
<h2>Exit rules</h2><div class="panel scroll"><table><thead><tr><th>Target</th><th>Stop</th><th>Trades</th><th>Average profit</th><th>Win rate</th><th></th></tr></thead><tbody>{exits}</tbody></table></div>
<p class="muted small">{escape(f"Best so far: +{best['target']:.0%} target with a -{best['stop']:.0%} stop. " if best else "")}Exit rules are reported, not changed automatically.</p>
<h2>Earnings</h2><div class="panel"><p>Plays held through the report: {card["earnings_plays"]["trades"]} graded, average {_dollars(card["earnings_plays"]["avg"])}, win rate {_pct(card["earnings_plays"]["win_rate"], 0)}.
Leans called the reaction's direction {_pct(earn["lean_hit_rate"], 0)} of the time ({earn["leaning"]} reports).
Beat-probability error (Brier score, 0 is perfect, 0.25 is a coin flip): {"n/a" if earn["brier"] is None else f"{earn['brier']:.3f}"}. {earn["pending"]} predictions waiting for their report.</p>
{f'<div class="scroll"><table><thead><tr><th>Date</th><th>Ticker</th><th>Lean</th><th>Beat chance</th><th>Result</th><th>Reaction</th><th>Implied</th></tr></thead><tbody>{earn_rows}</tbody></table></div>' if earn_rows else ''}</div>
<h2>Change log</h2><div class="panel scroll"><table><thead><tr><th>Date</th><th>Change</th><th>Why</th></tr></thead><tbody>{log_rows or '<tr><td colspan="3" class="muted">No weekly checks yet.</td></tr>'}</tbody></table></div>
<h2>How learning works</h2><div class="panel"><ul>
<li>Every flagged contract is journaled and followed for five trading days, then graded on dollars of profit as the portfolio would have traded it.</li>
<li>Learning stays locked until 20 sessions and 1,000 graded contracts exist. Then, once a week, signals with a clear profit edge get up to 20% more weight and those that lose money up to 20% less.</li>
<li>A change is adopted only if, on the latest five sessions (data it was not fitted to), its daily top-five picks made more money than the weights in use.</li>
<li>The original weights run alongside as a control. If the learned weights make less for two weekly checks running, they revert.</li>
<li>Overlapping contracts on the same stock and day are not independent, so sample sizes overstate certainty; the minimums and small steps guard against learning noise.</li></ul></div>"""
    sub = f"What has made money, graded in dollars per contract, and how the scanner adjusts. Updated {escape(card['built'].replace('T', ' '))} ET."
    return _page("Learning", sub, body, TABLE_JS)


def _promising_rows(rows: list[tuple[Any, str]], tickers: dict[str, Any]) -> str:
    out = []
    for r, status in rows:
        info = tickers[r["ticker"]]
        expiry = datetime.strptime(r["expiration"], "%Y-%m-%d")
        target = (f'{escape(r["target_label"])} ${r["target_price"]:.2f}' if isinstance(r.get("target_label"), str)
                  else "none")
        out.append(
            "<tr>" + _cell(r["score"], f'{r["score"]:.0f}')
            + _cell(r["ticker"], f'<a href="scanner.html#read-{escape(r["ticker"])}">{escape(r["ticker"])}</a> '
                                 f'${r["strike"]:g} {r["type"]} {expiry:%b} {expiry.day}', False)
            + _cell(r["cost"], _money(r["cost"], False)) + _cell(r["moneyness"], f'{r["moneyness"]:+.1%}')
            + _cell(r["delta"], f'{r["delta"]:+.2f}') + _cell(r["breakeven_move"], f'{r["breakeven_move"]:.1%}')
            + _cell(r.get("target_label") or "", target, False)
            + _cell(info["trend"], f'{LEANS[info["bias"]][0]} {info["trend"]:+.0f}')
            + _cell(status, escape(status), False) + "</tr>")
    return "".join(out)


def _regime_section(model: dict[str, Any] | None, limits: tuple[int, int, str | None]) -> str:
    """Market regime: risk score and state, its components, warnings with their record, the odds
    in each state since 2008, and the sectors it favours."""
    if not model:
        return '<p class="muted">The market regime could not be read on this scan.</p>'
    odds = model["odds"]
    state_tiles = [
        ("Risk regime", escape(model["state"].capitalize()), f"risk score {model['score']:+.0f}, "
                                                            f"{model['change']:+.0f} over 10 sessions"),
        ("Odds of a 5%+ drop within a month", _pct(odds.get("drop_odds"), 0),
         f"usually {model['base_drop']:.0%} (all days since {model['since'][:4]})"),
        ("SPY's average next month", _pct(odds.get("spy_return"), 2, signed=True),
         f"up {_pct(odds.get('spy_up'), 0)} of the time in this state"),
        ("Account exposure", f"{limits[0]} positions max", escape(limits[2]) if limits[2] else "full: risk is normal"),
    ]
    tiles = "".join(f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div><div class="d">{d}</div></div>'
                    for k, v, d in state_tiles)

    path = model.get("path") or []
    chart = ""
    if len(path) > 2:
        months, seen = [], set()
        for i, (d, _) in enumerate(path):
            if d[:7] not in seen:
                seen.add(d[:7])
                months.append((i, datetime.fromisoformat(d).strftime("%b")))
        def draw(width: int, most: int) -> str:
            return _line_chart([v for _, v in path], [_day(d) for d, _ in path], [f"{v:+.0f}" for _, v in path],
                               _thin(months, most), lambda v: f"{v:+.0f}", 0.0, "Neutral",
                               "Risk score over the past year", 5.0, width)

        chart = _both(draw(800, 12), draw(420, 6))

    rows = []
    for key, value in model["components"].items():
        width = min(abs(value) / 2.5, 1) * 50
        left = 50 if value >= 0 else 50 - width
        colour = "var(--series)" if value >= 0 else "var(--bad)"
        rows.append(
            f'<tr><td>{escape(regime.COMPONENTS.get(key, key))}</td><td class="num">{"▲" if value >= 0 else "▼"} {value:+.2f}</td>'
            f'<td style="width:40%"><div class="dbar"><span style="left:{left:.0f}%;width:{width:.0f}%;'
            f'background:{colour}"></span><i></i></div></td></tr>')
    components = ('<div class="panel scroll"><table><thead><tr><th>Component</th><th>Reading</th><th>Risk-off ← → Risk-on</th>'
                  f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
    warnings = ('<div class="panel"><ul class="plain">' + "".join(f"<li>⚠ {escape(w)}</li>" for w in model["warnings"])
                + "</ul></div>" if model["warnings"] else "")

    def sector_list(items):
        if not items:
            return '<p class="muted small">None right now.</p>'
        return "<ul>" + "".join(f'<li><b>{escape(i["sector"])} ({i["etf"]})</b> · {escape(i["quadrant"])}</li>'
                                for i in items) + "</ul>"

    return f"""<div class="tiles">{tiles}</div>
<div class="panel" style="margin-top:12px"><h4 style="margin:0 0 6px">Risk score, past year</h4>{chart}</div>
{warnings}{components}
<div class="cards" style="margin-top:12px"><div class="card"><h3>Favoured sectors</h3>{sector_list(model["favoured"])}</div>
<div class="card"><h3>Sectors to avoid</h3>{sector_list(model["avoid"])}</div></div>"""


def render_home(result: dict[str, Any], state: dict[str, Any], cfg: dict[str, Any]) -> str:
    start, positions, closed = state["start_cash"], state["positions"], state["closed"]
    total = portfolio.equity(state)
    values = [v for _, v in state["equity"]] or [start]
    peak, drawdown = values[0], 0.0
    for v in values:
        peak = max(peak, v)
        drawdown = min(drawdown, v / peak - 1)
    wins = sum(c["pnl"] > 0 for c in closed)
    realised = sum(c["pnl"] for c in closed)
    regime = portfolio.market_regime(result, cfg)
    calls = sum(p["type"] == "call" for p in positions)
    regime_text = ("unknown" if regime["score"] is None else
                   f'{LEANS.get(regime["label"], regime["label"])} {regime["score"]:+.0f}')
    tiles = [
        ("Account value", _money(total), _delta(total - start, total / start - 1) + " since the start"),
        ("Cash", _money(state["cash"]), f"{state['cash'] / total:.0%} of the account"),
        ("Open positions", f"{len(positions)} of {cfg['sim_max_positions']}", f"{calls} calls, {len(positions) - calls} puts"),
        ("Closed trades", f"{wins} won, {len(closed) - wins} lost" if closed else "None yet",
         _delta(realised) + " realised" if closed else "since the start"),
        ("Largest drop from peak", f"{drawdown:.1%}", f"peak {_money(max(values))}"),
        ("Market direction", regime_text, f"average read of {' and '.join(cfg['sim_regime_tickers'])}"),
    ]
    tile_html = "".join(f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div><div class="d">{d}</div></div>'
                        for k, v, d in tiles)

    pos_rows = "".join(
        f'<tr><td>{escape(p["label"])}</td><td class="num">{_money(p["entry_price"])}</td>'
        f'<td class="num">{_money(p["last_bid"])}</td>'
        f'<td class="num">{_delta((p["last_bid"] - p["entry_price"]) * 100, p["last_bid"] / p["entry_price"] - 1)}</td>'
        f'<td class="num">{int(np.busday_count(p["opened"][:10], date.today()))}</td>'
        f'<td class="num">{_money(p["stop_price"])}</td><td class="num">{_money(p["target_price"])}</td></tr>'
        for p in positions)
    pos_table = (
        '<div class="panel scroll"><table><thead><tr><th>Position</th><th>Paid</th><th>Bid now</th><th>P/L</th>'
        f'<th>Days held</th><th>Stop</th><th>Target</th></tr></thead><tbody>{pos_rows}</tbody></table></div>'
        if positions else '<p class="muted">No open positions.</p>')

    contracts, tickers = result["contracts"], result["tickers"]
    cutoff = (datetime.now() - timedelta(days=cfg["sim_reentry_days"])).isoformat()
    resting = {c["ticker"] for c in closed if c["exit_time"] > cutoff}
    live = contracts[~contracts["stale"]] if len(contracts) else contracts
    best = live.drop_duplicates("ticker") if len(live) else live
    limits = portfolio.exposure_limits(result, cfg)
    buyable = [(r, _buy_status(r, state, cfg, resting, tickers[r["ticker"]], regime, limits)[0])
               for _, r in best.iterrows() if portfolio.entry_block(r, tickers[r["ticker"]], regime, cfg) is None][:8]
    anywhere = [(r, _buy_status(r, state, cfg, resting, tickers[r["ticker"]], regime, limits)[0])
                for _, r in best.head(8).iterrows()]
    heads = ("<tr><th>Score</th><th>Contract</th><th>Cost</th><th>OTM</th><th>Delta</th><th>Breakeven move</th>"
             "<th>Structural target</th><th>Market read</th><th>Status</th></tr>")

    def table(rows):
        if not rows:
            return '<p class="muted">Nothing qualifies on this scan.</p>'
        return (f'<div class="panel scroll"><table class="sortable"><thead>{heads}</thead>'
                f'<tbody>{_promising_rows(rows, tickers)}</tbody></table></div>')

    body = f"""<div class="tiles">{tile_html}</div>
<div class="grid2"><section><h2>Sector rotation</h2>{_rotation_graph(result.get("rotation") or {}, result.get("mag7_rotation"))}</section>
<section><h2>Market regime</h2>{_regime_section(result.get("regime_model"), portfolio.exposure_limits(result, cfg))}</section></div>
<div class="grid2"><section><h2>Account value</h2><div class="panel">{_chart(state["equity"], start)}</div></section>
<section><h2>Open positions</h2>{pos_table}
<p class="small"><a href="portfolio.html">Full portfolio, with the reasoning and reviews for every trade</a></p></section></div>
<h2>Promising contracts the account could buy</h2>{table(buyable)}
<p class="muted small">The best contract per stock that passes the account&#39;s entry rules: within the
{_money(config.max_premium(cfg), False)} per-trade limit, a bid/ask spread of {cfg['sim_max_entry_spread']:g}% or less, and
not against the market regime unless its own read is at least ±{cfg['sim_counter_trend_min']}. Status says whether the
account would buy it now or what it is waiting for.</p>
<h2>Top setups at any price</h2>{table(anywhere)}
<p class="muted small">The highest-scoring contract per stock, whatever it costs. <a href="scanner.html">Every flagged
contract</a> is on the Scanner page.</p>"""
    sub = (f"Account health, the most promising contracts and where money is rotating. Last scan "
           f"{_when(result['scanned_at'].isoformat())} ET.")
    return _page("Home", sub, body, CHART_JS + TABLE_JS)


MULTI_JS = """
document.querySelectorAll('.multichart').forEach(function(el){var d=JSON.parse(el.dataset.chart),
svg=el.querySelector('svg'),line=svg.querySelector('.xh'),tip=el.querySelector('.tip');
function move(e){var r=svg.getBoundingClientRect(),cx=(e.touches?e.touches[0].clientX:e.clientX)-r.left,
vx=cx/r.width*d.w,best=0,gap=1e9;for(var i=0;i<d.x.length;i++){var g=Math.abs(d.x[i]-vx);if(g<gap){gap=g;best=i}}
line.setAttribute('x1',d.x[best]);line.setAttribute('x2',d.x[best]);line.style.display='';
tip.innerHTML='<b>'+d.labels[best]+'</b>'+d.rows[best];tip.style.display='block';var x=d.x[best]/d.w*r.width;
tip.style.left=Math.min(Math.max(x+12,0),r.width-tip.offsetWidth)+'px';tip.style.top='8px'}
function out(){line.style.display='none';tip.style.display='none'}
svg.addEventListener('mousemove',move);svg.addEventListener('touchstart',move,{passive:true});
svg.addEventListener('touchmove',move,{passive:true});svg.addEventListener('mouseleave',out)});
"""
SERIES_TOKENS = [f"--s{i}" for i in range(1, 9)]


def _multi_chart(series: list[tuple[str, list[list]]], start_cash: float, width: int = 800) -> str:
    """Every account's value over time on one axis, coloured in a fixed order, with a legend."""
    times = sorted({t for _, pts in series for t, _ in pts})
    if len(times) < 2:
        return '<p class="muted">The chart appears after the second scan.</p>'
    W, H, L, R, T, B = width, 300, 60, 12, 12, 28
    values = [v for _, pts in series for _, v in pts] + [start_cash]
    lo, hi = min(values), max(values)
    pad = max((hi - lo) * 0.12, start_cash * 0.005)
    lo, hi = lo - pad, hi + pad
    index = {t: i for i, t in enumerate(times)}

    def x(i: int) -> float:
        return L + (W - L - R) * i / (len(times) - 1)

    def y(v: float) -> float:
        return T + (H - T - B) * (1 - (v - lo) / (hi - lo))

    parts = []
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        parts.append(f'<line x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="var(--grid)"/>'
                     f'<text x="{L - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{_money(v, False)}</text>')
    seen, ticks = set(), []
    for t in times:
        if t[:10] not in seen:
            seen.add(t[:10])
            ticks.append((index[t], _day(t)))
    for n, (i, text) in enumerate(_thin(ticks, 8 if width > 600 else 4)):
        parts.append(f'<text x="{x(i):.1f}" y="{H - 8}" text-anchor="{"start" if n == 0 else "middle"}">{text}</text>')
    base = y(start_cash)
    parts.append(f'<line x1="{L}" x2="{W - R}" y1="{base:.1f}" y2="{base:.1f}" stroke="var(--axis)" stroke-dasharray="4 4"/>')
    for n, (name, pts) in enumerate(series):
        d = " ".join(f"{'M' if j == 0 else 'L'}{x(index[t]):.1f},{y(v):.1f}" for j, (t, v) in enumerate(pts))
        parts.append(f'<path d="{d}" fill="none" stroke="var({SERIES_TOKENS[n % 8]})" stroke-width="2" '
                     f'stroke-linejoin="round"><title>{escape(name)}</title></path>')
    parts.append(f'<line class="xh" y1="{T}" y2="{H - B}" stroke="var(--axis)" style="display:none"/>')

    lookup = [{t: v for t, v in pts} for _, pts in series]
    rows = []
    for t in times:
        here = sorted(((lookup[n].get(t), name, n) for n, (name, _) in enumerate(series) if t in lookup[n]),
                      key=lambda r: -r[0])
        rows.append("".join(f'<br><i style="display:inline-block;width:8px;height:8px;border-radius:2px;'
                            f'background:var({SERIES_TOKENS[n % 8]});margin-right:5px"></i>{escape(name)} {_money(v)}'
                            for v, name, n in here))
    chart = {"w": W, "x": [round(x(i), 1) for i in range(len(times))], "labels": [_when(t) for t in times], "rows": rows}
    legend = "".join(f'<span><i style="background:var({SERIES_TOKENS[n % 8]})"></i>{escape(name)}</span>'
                     for n, (name, _) in enumerate(series))
    return (f'<div class="legend">{legend}</div><div class="multichart linechart" '
            f'data-chart="{escape(json.dumps(chart))}"><svg viewBox="0 0 {W} {H}" role="img" '
            f'aria-label="Each account&#39;s value over time" font-size="13" fill="var(--muted)">{"".join(parts)}</svg>'
            f'<div class="tip"></div></div>')


def _account_stats(state: dict[str, Any]) -> dict[str, Any]:
    total = portfolio.equity(state)
    closed = state["closed"]
    values = [v for _, v in state["equity"]] or [state["start_cash"]]
    peak, drop = values[0], 0.0
    for v in values:
        peak = max(peak, v)
        drop = min(drop, v / peak - 1)
    wins = sum(c["pnl"] > 0 for c in closed)
    return {"value": total, "return": total / state["start_cash"] - 1, "realised": sum(c["pnl"] for c in closed),
            "closed": len(closed), "wins": wins, "win_rate": wins / len(closed) if closed else None,
            "avg": sum(c["pnl"] for c in closed) / len(closed) if closed else None, "open": len(state["positions"]),
            "drop": drop, "started": state["started"]}


def render_strategies(cfg: dict[str, Any]) -> str:
    accounts = portfolio.accounts(cfg)
    states = {a["id"]: portfolio.load(portfolio.account_cfg(cfg, a["id"]), a["id"]) for a in accounts}
    stats = {k: _account_stats(s) for k, s in states.items()}
    ranked = sorted(accounts, key=lambda a: -stats[a["id"]]["value"])

    rows = "".join(
        "<tr>" + _cell(n, str(n)) + _cell(a["name"], f'<a href="#acct-{a["id"]}"><b>{escape(a["name"])}</b></a>'
                                       f'<br><span class="muted small lb-sum">{escape(a["summary"])}</span>', False)
        + _cell(st["return"], f'<span class="{"up" if st["return"] >= 0 else "down"}">{"▲" if st["return"] >= 0 else "▼"} '
                              f'{st["return"]:+.1%}</span>')
        + _cell(st["value"], _money(st["value"]))
        + _cell(st["realised"], _signed(st["realised"]) if st["closed"] else "–")
        + _cell(st["closed"], f'{st["wins"]}–{st["closed"] - st["wins"]}' if st["closed"] else "0")
        + _cell(st["win_rate"] or -1, _pct(st["win_rate"], 0)) + _cell(st["avg"] or 0, _signed(st["avg"]) if st["avg"] is not None else "–")
        + _cell(st["open"], str(st["open"])) + _cell(st["drop"], f'{st["drop"]:.1%}')
        + _cell(st["started"], _day(st["started"])) + "</tr>"
        for n, (a, st) in enumerate(((a, stats[a["id"]]) for a in ranked), 1))
    table = ('<div class="panel scroll"><table class="sortable"><thead><tr><th>Rank</th><th>Strategy</th><th>Return</th>'
             '<th>Value</th><th>Realised P/L</th><th>Won–lost</th><th>Win rate</th><th>Average closed trade</th>'
             f'<th>Open</th><th>Largest drop</th><th>Started</th></tr></thead><tbody>{rows}</tbody></table></div>')

    def group_chart(group: str) -> str:
        members = [a for a in accounts if a.get("group") in (group, "both")]
        series = [(a["name"], states[a["id"]]["equity"]) for a in members]
        return _both(_multi_chart(series, float(cfg["account_size"]), 800),
                     _multi_chart(series, float(cfg["account_size"]), 420))

    chart = (f'<h4 style="margin:0 0 8px">Rule changes</h4>{group_chart("rules")}'
             f'<h4 style="margin:24px 0 8px">Scoring weights</h4>{group_chart("weights")}')

    def describe(key: str, value: Any) -> str:
        if key == "method_weights":
            return "Market read weights: " + ", ".join(f"{METHOD_SHORT.get(k, k)} {v:.0%}" for k, v in value.items())
        if key == "factor_weights":
            return "Contract score weights: " + ", ".join(f"{FACTOR_NAMES.get(k, k).lower()} {v}" for k, v in value.items())
        return f"{key.replace('sim_', '').replace('_', ' ')}: {value}"

    cards = []
    for a in accounts:
        s, st = states[a["id"]], stats[a["id"]]
        rules = "".join(f"<li>{escape(describe(k, v))}</li>" for k, v in a["rules"].items()) \
            or "<li>The shared settings, unchanged</li>"
        open_rows = "".join(
            f'<li>{escape(p["label"])}: paid {_money(p["entry_price"])}, bid {_money(p["last_bid"])} '
            f'{_delta((p["last_bid"] - p["entry_price"]) * 100, p["last_bid"] / p["entry_price"] - 1)}</li>'
            for p in s["positions"]) or '<li class="muted">None</li>'
        closed_rows = "".join(
            f'<li>{escape(c["label"])}: {_delta(c["pnl"], c["ret"])} · {escape(c["exit_reason"])}</li>'
            for c in reversed(s["closed"][-6:])) or '<li class="muted">None yet</li>'
        cards.append(
            f'<article class="card" id="acct-{a["id"]}"><header><h3>{escape(a["name"])}</h3>'
            f'<div class="pl">{_delta(st["value"] - s["start_cash"], st["return"])}</div></header>'
            f'<p class="small muted">{escape(a["summary"])}</p>'
            f'<details><summary>Rules, positions and trades</summary><h4>Rule changes from the shared settings</h4>'
            f'<ul>{rules}</ul><h4>Open positions</h4><ul>{open_rows}</ul><h4>Latest closed trades</h4>'
            f'<ul>{closed_rows}</ul></details></article>')

    body = f"""<h2>Leaderboard</h2>{table}
<h2>Value over time</h2><div class="panel">{chart}</div>
<h2>Each strategy</h2><div class="cards">{"".join(cards)}</div>
<p class="muted small">Every account starts with {_money(float(cfg["account_size"]), False)} and trades the same scans at the
same moments, so differences come from the rules. Every account only buys to open and sells to close.</p>"""
    return _page("Strategies", "Several simulated accounts, each trying different rules on the same scans.", body,
                 TABLE_JS + MULTI_JS)


def build(state: dict[str, Any], cfg: dict[str, Any], result: dict[str, Any] | None = None) -> None:
    OUT.parent.mkdir(exist_ok=True)
    set_chrome(result, state, cfg)
    OUT.with_name("portfolio.html").write_text(render(state, cfg), encoding="utf-8")
    OUT.with_name("strategies.html").write_text(render_strategies(cfg), encoding="utf-8")
    if result is not None:
        OUT.write_text(render_home(result, state, cfg), encoding="utf-8")
        OUT.with_name("scanner.html").write_text(render_scanner(result, state, cfg), encoding="utf-8")
        OUT.with_name("earnings.html").write_text(render_earnings(result, cfg), encoding="utf-8")
    OUT.with_name("learning.html").write_text(render_learning(scorecard.load(), cfg), encoding="utf-8")


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=HOME, capture_output=True, text=True, timeout=120,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def publish() -> str | None:
    """Commit and push the pages. Returns an error message, or None on success."""
    page = "docs"
    if _git("rev-parse", "--is-inside-work-tree").returncode != 0:
        return "not a git repository"
    _git("add", page)
    if _git("diff", "--cached", "--quiet", "--", page).returncode == 0:
        return None  # nothing changed
    commit = _git("commit", "-m", f"Update site {datetime.now():%Y-%m-%d %H:%M}", "--", page)
    if commit.returncode != 0:
        return f"commit failed: {commit.stderr.strip() or commit.stdout.strip()}"
    push = _git("push")
    return None if push.returncode == 0 else f"push failed: {push.stderr.strip()}"
