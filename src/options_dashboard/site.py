"""Static portfolio page for GitHub Pages, written to docs/index.html."""

from __future__ import annotations

import json
import subprocess
from datetime import date, datetime, timedelta
from html import escape
from typing import Any

from . import earnings, portfolio, scorecard, structure, universe
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
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--series:#2a78d6;--good:#006300;--bad:#d03b3b;
--track:#e1e0d9}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;
--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
--series:#3987e5;--good:#0ca30c;--bad:#e66767;--track:#383835}}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1080px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:24px;margin:0}h2{font-size:17px;margin:36px 0 12px}
.sub{color:var(--ink2);margin:4px 0 0}.muted{color:var(--muted)}.small{font-size:13px}
.notice{border:1px solid var(--border);border-radius:8px;padding:10px 12px;margin:16px 0 0;color:var(--ink2);font-size:13px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:20px}
.tile,.card,.panel{background:var(--surface);border:1px solid var(--border);border-radius:10px}
.tile{padding:12px 14px}.tile .k{color:var(--ink2);font-size:13px}.tile .v{font-size:24px;font-weight:600}
.tile .d{font-size:13px;color:var(--ink2)}
.up{color:var(--good)}.down{color:var(--bad)}
.panel{padding:14px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,440px),1fr));gap:14px}
.card{padding:14px}
.card header{display:flex;justify-content:space-between;gap:12px;align-items:baseline;flex-wrap:wrap}
.card h3{margin:0;font-size:17px}.pl{font-size:17px;font-weight:600;white-space:nowrap}
.tags{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0 0}
.tag{border:1px solid var(--border);border-radius:999px;padding:1px 9px;font-size:12px;color:var(--ink2)}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(105px,1fr));gap:8px 12px;margin:12px 0 0}
.facts div span{display:block;color:var(--ink2);font-size:12px}.facts div b{font-weight:600}
.range{margin:22px 0 4px}.bar{position:relative;height:6px;border-radius:3px;background:var(--track)}
.bar i{position:absolute;top:-4px;width:2px;height:14px;background:var(--muted)}
.bar u{position:absolute;top:-5px;width:16px;height:16px;margin-left:-8px;border-radius:50%;background:var(--series);
border:2px solid var(--surface)}
.ends{display:flex;justify-content:space-between;font-size:12px;color:var(--ink2);margin-top:8px}
.review{margin:12px 0 0;padding:8px 10px;border-left:3px solid var(--series);background:var(--page);
border-radius:4px;font-size:13px;color:var(--ink2)}.review b{color:var(--ink)}
details{margin-top:12px;border-top:1px solid var(--border);padding-top:10px}
summary{cursor:pointer;font-weight:600}
details h4{margin:12px 0 4px;font-size:13px;color:var(--ink2);font-weight:600}
details ul{margin:0;padding-left:18px}details li{margin:3px 0}
ul.plain{list-style:none;padding:0}
.pts{display:grid;grid-template-columns:minmax(120px,1.3fr) 1fr auto;gap:4px 10px;align-items:center;font-size:13px}
.pts .b{height:6px;border-radius:3px;background:var(--track);overflow:hidden}
.pts .b span{display:block;height:100%;background:var(--series);border-radius:3px}
.pts .n{font-variant-numeric:tabular-nums;color:var(--ink2)}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid);vertical-align:top}
th{color:var(--ink2);font-weight:600}td.num{font-variant-numeric:tabular-nums;white-space:nowrap}
.scroll{overflow-x:auto}
nav{display:flex;gap:6px;margin-bottom:16px}
nav a{padding:5px 14px;border:1px solid var(--border);border-radius:999px;color:var(--ink2);text-decoration:none;font-size:14px}
nav a[aria-current]{background:var(--surface);color:var(--ink);font-weight:600}
td a{color:inherit}
.sortable th{cursor:pointer;white-space:nowrap;user-select:none}
.sortable th[data-dir=asc]::after{content:" ▲"}.sortable th[data-dir=desc]::after{content:" ▼"}
.sortable td{white-space:nowrap}
.filters{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin-bottom:10px;font-size:14px;color:var(--ink2)}
.filters select{font:inherit;padding:3px 6px;background:var(--surface);color:var(--ink);border:1px solid var(--border);border-radius:6px}
.card:target{outline:2px solid var(--series)}
#chart{position:relative}#chart svg{display:block;width:100%;height:auto}
#tip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--border);border-radius:6px;
padding:4px 8px;font-size:12px;white-space:nowrap;display:none;box-shadow:0 2px 8px rgba(0,0,0,.15)}
footer{margin-top:40px;color:var(--muted);font-size:12px}
"""

METHOD_SHORT = {"auction": "Auction", "gamma": "Dealer flows", "wyckoff": "Wyckoff", "vpa": "VPA",
                "vol": "Vol surface", "trend": "Trend"}

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
(function(){var el=document.getElementById('chart');if(!el)return;var d=JSON.parse(el.dataset.points),
svg=el.querySelector('svg'),line=svg.querySelector('#xh'),dot=svg.querySelector('#xd'),tip=document.getElementById('tip');
function move(e){var r=svg.getBoundingClientRect(),cx=(e.touches?e.touches[0].clientX:e.clientX)-r.left,
vx=cx/r.width*800,best=0,gap=1e9;for(var i=0;i<d.length;i++){var g=Math.abs(d[i][0]-vx);if(g<gap){gap=g;best=i}}
var p=d[best];line.setAttribute('x1',p[0]);line.setAttribute('x2',p[0]);dot.setAttribute('cx',p[0]);
dot.setAttribute('cy',p[1]);line.style.display=dot.style.display='';tip.style.display='block';
tip.innerHTML='<b>'+p[3]+'</b><br>'+p[2];var x=p[0]/800*r.width;
tip.style.left=Math.min(Math.max(x-tip.offsetWidth/2,0),r.width-tip.offsetWidth)+'px';
tip.style.top=Math.max(p[1]/260*r.height-tip.offsetHeight-12,0)+'px'}
function out(){line.style.display=dot.style.display='none';tip.style.display='none'}
svg.addEventListener('mousemove',move);svg.addEventListener('touchstart',move,{passive:true});
svg.addEventListener('touchmove',move,{passive:true});svg.addEventListener('mouseleave',out)})();
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


def _chart(points: list[list], start_cash: float) -> str:
    if len(points) < 2:
        return '<p class="muted">The equity chart appears after the second scan.</p>'
    W, H, L, R, T, B = 800, 260, 56, 12, 12, 28
    values = [v for _, v in points] + [start_cash]
    lo, hi = min(values), max(values)
    pad = max((hi - lo) * 0.15, start_cash * 0.005)
    lo, hi = lo - pad, hi + pad

    def x(i: int) -> float:
        return L + (W - L - R) * i / (len(points) - 1)

    def y(v: float) -> float:
        return T + (H - T - B) * (1 - (v - lo) / (hi - lo))

    grid = []
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        grid.append(
            f'<line x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="var(--grid)"/>'
            f'<text x="{L - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{_money(v, False)}</text>'
        )
    # One x label per trading day, placed at that day's first scan.
    labels, seen = [], set()
    for i, (t, _) in enumerate(points):
        day = t[:10]
        if day not in seen:
            seen.add(day)
            labels.append((x(i), _day(t)))
    step = max(1, len(labels) // 8)
    ticks = "".join(
        f'<text x="{px:.1f}" y="{H - 8}" text-anchor="{"start" if n == 0 else "middle"}">{text}</text>'
        for n, (px, text) in enumerate(labels) if n % step == 0
    )
    path = " ".join(f"{'M' if i == 0 else 'L'}{x(i):.1f},{y(v):.1f}" for i, (_, v) in enumerate(points))
    data = [[round(x(i), 1), round(y(v), 1), _money(v), _when(t)] for i, (t, v) in enumerate(points)]
    base = y(start_cash)
    return (
        f'<div id="chart" data-points="{escape(json.dumps(data))}">'
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Portfolio value over time" '
        f'font-size="11" fill="var(--muted)">{"".join(grid)}{ticks}'
        f'<line x1="{L}" x2="{W - R}" y1="{base:.1f}" y2="{base:.1f}" stroke="var(--axis)" stroke-dasharray="4 4"/>'
        f'<text x="{W - R}" y="{base - 5:.1f}" text-anchor="end">Starting {_money(start_cash, False)}</text>'
        f'<path d="{path}" fill="none" stroke="var(--series)" stroke-width="2" stroke-linejoin="round"/>'
        f'<line id="xh" y1="{T}" y2="{H - B}" stroke="var(--axis)" style="display:none"/>'
        f'<circle id="xd" r="5" fill="var(--series)" stroke="var(--surface)" stroke-width="2" style="display:none"/>'
        f'</svg><div id="tip"></div></div>'
    )


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
    return _page("Portfolio", sub, body, CHART_JS)


def _page(active: str, sub: str, body: str, script: str) -> str:
    nav = "".join(
        f'<a href="{href}"{" aria-current=\"page\"" if name == active else ""}>{name}</a>'
        for name, href in (("Portfolio", "index.html"), ("Scanner", "scanner.html"), ("Earnings", "earnings.html"),
                           ("Learning", "learning.html"))
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex">
<meta http-equiv="refresh" content="300">
<title>Options {active}</title><style>{CSS}</style></head>
<body><main>
<nav>{nav}</nav>
<h1>{TITLES[active]}</h1>
<p class="sub">{sub}</p>
{body}
<footer>Simulation for education only. Not financial advice, and not a record of real trades.</footer>
</main><script>{script}</script></body></html>"""


def _buy_status(row: Any, state: dict[str, Any], cfg: dict[str, Any], resting: set[str],
                info: dict[str, Any], regime: dict[str, Any]) -> tuple[str, bool]:
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
    if len(state["positions"]) >= cfg["sim_max_positions"]:
        return "Buy grade · slots full", True
    blocked = portfolio.entry_block(row, info, regime, cfg)
    if blocked:
        return f"Buy grade · {blocked}", True
    if sum(1 for p in state["positions"] if p["type"] == row["type"]) >= cfg["sim_max_same_direction"]:
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


def render_scanner(result: dict[str, Any], state: dict[str, Any], cfg: dict[str, Any]) -> str:
    contracts, tickers = result["contracts"], result["tickers"]
    regime = portfolio.market_regime(result, cfg)
    cutoff = (datetime.now() - timedelta(days=cfg["sim_reentry_days"])).isoformat()
    resting = {c["ticker"] for c in state["closed"] if c["exit_time"] > cutoff}

    rows, buy_grade = [], 0
    for _, r in contracts.iterrows():
        status, grade = _buy_status(r, state, cfg, resting, tickers[r["ticker"]], regime)
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
<h2>Reasoning by ticker</h2><div class="cards">{"".join(cards)}</div>"""
    built = universe.load()
    origin = (f" Universe from {escape(built['source'])}, built {_when(built['built'])} ET."
              if built.get("source") else "")
    sub = (f"Every contract that passed the filters on the last scan, whether or not the portfolio bought it, "
           f"across {len(tickers)} liquid stocks and ETFs. Last scan {_when(result['scanned_at'].isoformat())} ET."
           f"{origin}")
    return _page("Scanner", sub, body, TABLE_JS)


TITLES = {"Portfolio": "Options Paper Portfolio", "Scanner": "Options Scanner", "Earnings": "Earnings Outlook",
          "Learning": "What's Working"}
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


def build(state: dict[str, Any], cfg: dict[str, Any], result: dict[str, Any] | None = None) -> None:
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(render(state, cfg), encoding="utf-8")
    if result is not None:
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
