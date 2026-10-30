"""0DTE lab: a forward test of the end-of-day momentum trade on SPX (and SPY), at real prices.

The rule, from the backtest (Dec 2023 - Oct 2026, hourly bars, modelled prices): at 3:30 PM New
York, if the index is above every high from 9:30 to 2:30 and above the open, buy the nearest
out-of-the-money call whose ask is within the cap; below every low and below the open, a put.
Sell at 3x the cost, otherwise hold to the close and settle at intrinsic value. Buy to open only.

The backtest made money in every year but was sensitive to what the options really cost in the
last half hour, which a model cannot settle. So this lab buys the real contract at its real ask:
the 3:45 scan's Cboe chain is 15 minutes delayed, so it shows the market at about 3:30, which is
the rule's own entry time. The target is checked against the next snapshot (about 3:45), and
everything left is settled at the official close by a later run.

Several variants run side by side so evidence builds faster (see VARIANTS). One contract each.

    python -m options_dashboard.zerodte            # enter at 3:45-4:00, otherwise manage and settle
    python -m options_dashboard.zerodte --report   # print the results so far
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, time as clock
from typing import Any

import pandas as pd
import yfinance as yf

from . import data
from .config import HOME, STATE_DIR

FILE = HOME / "paper" / "zerodte.json"
VARIANTS = [
    {"id": "spx150", "name": "SPX, up to $150", "ticker": "SPX", "cap": 1.50, "target": 3.0, "slack": 0.0},
    {"id": "spx80", "name": "SPX, up to $80", "ticker": "SPX", "cap": 0.80, "target": 3.0, "slack": 0.0},
    {"id": "spx80_loose", "name": "SPX, up to $80, loose trigger", "ticker": "SPX", "cap": 0.80, "target": 3.0,
     "slack": 0.001},
    {"id": "spx150_gamma", "name": "SPX, up to $150, dealers short gamma only", "ticker": "SPX", "cap": 1.50,
     "target": 3.0, "slack": 0.0, "short_gamma": True},
    {"id": "spy15", "name": "SPY, up to $15", "ticker": "SPY", "cap": 0.15, "target": 3.0, "slack": 0.0},
    # The day's direction call (daycall.py): bought at the open on confident days, 2x or the close.
    {"id": "day_spx150", "name": "Day call: SPX, up to $150", "ticker": "SPX", "cap": 1.50, "target": 2.0,
     "strategy": "daycall"},
    {"id": "day_spy40", "name": "Day call: SPY, up to $40", "ticker": "SPY", "cap": 0.40, "target": 2.0,
     "strategy": "daycall"},
]
CALL_FROM, CALL_UNTIL = clock(9, 30), clock(10, 0)  # make the call once the open is known
BUY_FROM = clock(9, 45)  # the 9:45 scan's delayed chain shows the market at about 9:30
SOURCES = {"SPX": ("^GSPC", "_SPX"), "SPY": ("SPY", "SPY")}  # Yahoo bars, Cboe chain
FEES = {"SPX": 1.25, "SPY": 0.65}  # per contract each way (SPX carries an exchange fee)
ENTRY_FROM, ENTRY_UNTIL = clock(15, 45), clock(16, 0)


def load() -> dict[str, Any]:
    try:
        return json.loads(FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"started": datetime.now().isoformat(timespec="minutes"), "trades": [], "days": {}}


def _save(state: dict[str, Any]) -> None:
    FILE.parent.mkdir(parents=True, exist_ok=True)
    FILE.write_text(json.dumps(state, indent=1, default=float), encoding="utf-8")


def _bars(ticker: str) -> pd.DataFrame:
    h = data.call(yf.Ticker(SOURCES[ticker][0]).history, period="1d", interval="5m")
    h.index = h.index.tz_convert("America/New_York")
    return h


def signal(bars: pd.DataFrame, slack: float) -> tuple[bool, float, str] | None:
    """(call?, price at 3:30, reason) when the rule fires, else None."""
    day = bars.index[0]
    upto = bars[bars.index < day.replace(hour=15, minute=30)]
    rng = bars[bars.index < day.replace(hour=14, minute=30)]
    if len(rng) < 50 or len(upto) < 70:
        return None
    price, opening = float(upto["Close"].iloc[-1]), float(bars["Open"].iloc[0])
    hi, lo = float(rng["High"].max()), float(rng["Low"].min())
    if price > hi * (1 - slack) and price > opening:
        return True, price, f"{price:,.2f} at 3:30, above the 9:30-2:30 high ({hi:,.2f}) and the open ({opening:,.2f})"
    if price < lo * (1 + slack) and price < opening:
        return False, price, f"{price:,.2f} at 3:30, below the 9:30-2:30 low ({lo:,.2f}) and the open ({opening:,.2f})"
    return None


def _spy_short_gamma() -> bool | None:
    """Whether SPY's dealers were net short gamma on the last scan (flow-signed), if known."""
    try:
        last = json.loads((STATE_DIR / "last_scan.json").read_text(encoding="utf-8"))
        g = last.get("spy_gamma") or {}
        return None if g.get("net_gex") is None else g["net_gex"] < 0
    except (OSError, json.JSONDecodeError):
        return None


MIN_ASK = 0.05  # cheaper than this is a lottery ticket with no real bid, as in the backtest


def _today_chain(ticker: str) -> pd.DataFrame:
    """Today's expiring contracts with a real offer. For SPX only the SPXW weeklies, which settle at
    the 4 PM close; the monthly SPX contracts settle at the next morning's open."""
    chain, _ = data.cboe_chain(SOURCES[ticker][1])
    today = datetime.now(data.NEW_YORK).date().isoformat()
    keep = (chain["expiration"] == today) & (chain["ask"] >= MIN_ASK)
    if ticker == "SPX":
        keep &= chain["contractSymbol"].str.startswith("SPXW")
    return chain[keep]


def enter(state: dict[str, Any]) -> list[str]:
    """At the 3:45 scan: check the rule for each variant and buy at the real ask."""
    today = datetime.now(data.NEW_YORK).date().isoformat()
    if today in state["days"]:
        return []
    notes, record = [], {}
    bars, chains = {}, {}
    gamma_short = _spy_short_gamma()
    for v in VARIANTS:
        if v.get("strategy", "momentum") != "momentum":
            continue
        t = v["ticker"]
        try:
            bars.setdefault(t, _bars(t))
            sig = signal(bars[t], v["slack"])
        except Exception as exc:
            record[v["id"]] = f"no data ({type(exc).__name__})"
            continue
        if sig is None:
            record[v["id"]] = "no signal"
            continue
        if v.get("short_gamma") and gamma_short is not True:
            record[v["id"]] = "signal, but dealers not short gamma" if gamma_short is False else "signal, gamma unknown"
            continue
        call, price, why = sig
        try:
            chains.setdefault(t, _today_chain(t))
        except Exception as exc:
            record[v["id"]] = f"no chain ({type(exc).__name__})"
            continue
        side = chains[t][(chains[t]["type"] == ("call" if call else "put"))
                         & ((chains[t]["strike"] > price) if call else (chains[t]["strike"] < price))
                         & (chains[t]["ask"] <= v["cap"])]
        if side.empty:
            record[v["id"]] = "signal, but nothing within the cap"
            continue
        pick = side.iloc[(side["strike"] - price).abs().argsort()].iloc[0]
        trade = {"variant": v["id"], "date": today, "ticker": t, "type": "call" if call else "put",
                 "strike": float(pick["strike"]), "symbol": pick["contractSymbol"], "entry": float(pick["ask"]),
                 "entry_bid": float(pick["bid"]), "spot": price, "why": why, "status": "open",
                 "opened": datetime.now().isoformat(timespec="minutes")}
        state["trades"].append(trade)
        record[v["id"]] = "bought"
        notes.append(f"{v['name']}: bought {t} {trade['strike']:g} {trade['type']} at ${trade['entry']:.2f}")
    state["days"][today] = record
    return notes


def make_call(state: dict[str, Any]) -> list[str]:
    """At the open: the day's direction call for SPX and SPY, posted to Discord at once (the edge, such
    as it is, only exists right at the open)."""
    from . import daycall
    today = datetime.now(data.NEW_YORK).date().isoformat()
    calls = state.setdefault("calls", {})
    if today in calls:
        return []
    c = daycall.call()
    try:
        last = json.loads((STATE_DIR / "last_scan.json").read_text(encoding="utf-8"))
        spy_read = last.get("spy_read") or {}
    except (OSError, json.JSONDecodeError):
        spy_read = {}
    calls[today] = {k: {**v, "drivers": v["drivers"]} for k, v in c.items()}
    calls[today]["dashboard_spy"] = spy_read
    v = c["SPY"]  # one market-wide call (measured on SPY's real open), applied to both tickers
    lines = []
    if v["confident"]:
        lines.append(f"**Market {'▲ up' if v['direction'] == 'up' else '▼ down'} "
                     f"{max(v['p_up'], 1 - v['p_up']):.0%}** · {daycall.describe(v)}")
        side = "call" if v["direction"] == "up" else "put"
        for name in ("SPX", "SPY"):
            var = next(x for x in VARIANTS if x.get("strategy") == "daycall" and x["ticker"] == name)
            k = daycall.estimate_strike(c[name], name, var["cap"])
            lines.append(f"→ **{name}**: buy the {'SPXW' if name == 'SPX' else 'SPY'} 0DTE {side}, nearest strike out of the "
                         f"money with an ask up to ${var['cap']:.2f}" + (f" (about {k:,.0f})" if k else "") +
                         f"; sell at {var['target']:.0f}x or hold to the close")
    else:
        lines.append(f"**No confident call: stand aside.** {daycall.describe(v)}")
    if spy_read.get("bias"):
        lines.append(f"Dashboard's full SPY read: {spy_read['bias']} ({spy_read['trend']:+.0f}), a second opinion")
    record = calls_scorecard(state)
    if record["confident"]:
        lines.append(f"Confident calls so far: {record['confident_right']} of {record['confident']} right")
    lines.append("Paper-tested at real prices in the 0DTE lab before it should be trusted; research, not advice.")
    _post_discord("🎯 Today's direction call", "\n".join(lines))
    return [f"Day call: SPX {c['SPX']['p_up']:.0%} up, SPY {c['SPY']['p_up']:.0%} up"]


def _post_discord(title: str, text: str) -> None:
    import os
    import requests
    url = os.environ.get("DISCORD_WEBHOOK")
    if not url:
        return
    try:
        requests.post(url, json={"username": "Options Terminal",
                                 "embeds": [{"title": title, "description": text[:4000], "color": 0xFFA028}]}, timeout=15)
    except requests.RequestException:
        pass


def enter_calls(state: dict[str, Any]) -> list[str]:
    """From the 9:45 scan: buy the confident calls' contracts at the real ask (about 9:30 prices)."""
    today = datetime.now(data.NEW_YORK).date().isoformat()
    day = state.get("calls", {}).get(today)
    if not day or day.get("entered"):
        return []
    notes = []
    for v in (x for x in VARIANTS if x.get("strategy") == "daycall"):
        c = day.get(v["ticker"])
        if not c or not c["confident"]:
            continue
        call_ = c["direction"] == "up"
        try:
            chain = _today_chain(v["ticker"])
            # The price in the same delayed snapshot as the quotes; the index's official open is stale.
            ref = float(data.cboe_chain(SOURCES[v["ticker"]][1])[1].get("price") or c["open"])
        except Exception:
            continue
        side = chain[(chain["type"] == ("call" if call_ else "put"))
                     & ((chain["strike"] > ref) if call_ else (chain["strike"] < ref))
                     & (chain["ask"] <= v["cap"])]
        if side.empty:
            continue
        pick = side.iloc[(side["strike"] - ref).abs().argsort()].iloc[0]
        state["trades"].append({"variant": v["id"], "date": today, "ticker": v["ticker"], "type": "call" if call_ else "put",
                                "strike": float(pick["strike"]), "symbol": pick["contractSymbol"],
                                "entry": float(pick["ask"]), "entry_bid": float(pick["bid"]), "spot": ref,
                                "why": f"day call {c['p_up']:.0%} up", "status": "open",
                                "opened": datetime.now().isoformat(timespec="minutes")})
        notes.append(f"{v['name']}: bought {v['ticker']} {pick['strike']:g} at ${pick['ask']:.2f}")
    day["entered"] = True
    return notes


def calls_scorecard(state: dict[str, Any]) -> dict[str, Any]:
    """How the day calls have done: all calls, confident calls, and the dashboard's SPY read."""
    out = {"days": 0, "right": 0, "confident": 0, "confident_right": 0, "dashboard": 0, "dashboard_right": 0}
    for day in state.get("calls", {}).values():
        for name in ("SPY",):  # the call is measured on SPY's real open (the index's open is stale)
            c = day.get(name) or {}
            if "outcome" not in c:
                continue
            right = c["direction"] == c["outcome"]
            out["days"] += 1
            out["right"] += right
            if c["confident"]:
                out["confident"] += 1
                out["confident_right"] += right
        d = day.get("dashboard_spy") or {}
        if d.get("bias") in ("bullish", "bearish") and "outcome" in (day.get("SPY") or {}):
            out["dashboard"] += 1
            out["dashboard_right"] += (d["bias"] == "bullish") == (day["SPY"]["outcome"] == "up")
    return out


def score_calls(state: dict[str, Any]) -> None:
    """After the close: did each index close above its open?"""
    for d, day in state.get("calls", {}).items():
        for name in ("SPX", "SPY"):
            c = day.get(name)
            if not c or "outcome" in c:
                continue
            try:
                daily = data.call(yf.Ticker(SOURCES[name][0]).history, period="1mo", interval="1d")
                daily.index = daily.index.tz_convert("America/New_York").date
                close = float(daily.loc[pd.Timestamp(d).date(), "Close"])
            except Exception:
                continue
            if pd.Timestamp(d).date() == datetime.now(data.NEW_YORK).date() and datetime.now(data.NEW_YORK).time() < clock(16, 5):
                continue
            c.update(close=close, outcome="up" if close > c["open"] else "down")


def manage(state: dict[str, Any]) -> list[str]:
    """Take 3x targets on the next snapshot; settle the rest at the official close."""
    notes = []
    now = datetime.now(data.NEW_YORK)
    open_trades = [t for t in state["trades"] if t["status"] == "open"]
    if not open_trades:
        return notes
    chains: dict[str, pd.DataFrame] = {}
    for t in open_trades:
        v = next(x for x in VARIANTS if x["id"] == t["variant"])
        if t["date"] == now.date().isoformat() and now.time() < clock(16, 30):
            try:  # the snapshot after entry (about 3:45 with the delay): did it reach the target?
                chains.setdefault(t["ticker"], _today_chain(t["ticker"]))
                q = chains[t["ticker"]][chains[t["ticker"]]["contractSymbol"] == t["symbol"]]
            except Exception:
                q = pd.DataFrame()
            if not q.empty and float(q["bid"].iloc[0]) >= t["entry"] * v["target"]:
                bid = float(q["bid"].iloc[0])
                t.update(status="closed", exit=bid, how=f"target hit (bid ${bid:.2f})",
                         pnl=round((bid - t["entry"]) * 100 - 2 * FEES[t["ticker"]], 2))
                notes.append(f"{v['name']}: target hit, +${t['pnl']:.0f}")
            continue
        try:  # settle at the official close once the day's bar exists
            daily = data.call(yf.Ticker(SOURCES[t["ticker"]][0]).history, period="1mo", interval="1d")
            daily.index = daily.index.tz_convert("America/New_York").date
            close = float(daily.loc[pd.Timestamp(t["date"]).date(), "Close"])
        except Exception:
            continue
        value = max(close - t["strike"], 0) if t["type"] == "call" else max(t["strike"] - close, 0)
        fees = FEES[t["ticker"]] * (2 if value > 0 else 1)
        t.update(status="closed", exit=round(value, 2), close=close,
                 how=f"settled at the close ({close:,.2f})", pnl=round((value - t["entry"]) * 100 - fees, 2))
        notes.append(f"{v['name']}: settled, {'+' if t['pnl'] >= 0 else '-'}${abs(t['pnl']):.0f}")
    return notes


def summary(state: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Per variant: trades, win rate, average and total profit, days the rule was checked."""
    state = state or load()
    out = []
    for v in VARIANTS:
        closed = [t for t in state["trades"] if t["variant"] == v["id"] and t["status"] == "closed"]
        pnls = [t["pnl"] for t in closed]
        out.append({**v, "trades": len(closed), "open": sum(1 for t in state["trades"]
                                                             if t["variant"] == v["id"] and t["status"] == "open"),
                    "wins": sum(p > 0 for p in pnls), "total": sum(pnls), "avg": sum(pnls) / len(pnls) if pnls else None,
                    "checked": sum(1 for d in state["days"].values() if v["id"] in d)})
    return out


def run() -> list[str]:
    state = load()
    now = datetime.now(data.NEW_YORK)
    notes = []
    if now.weekday() < 5 and data.market_open():
        if CALL_FROM <= now.time() < CALL_UNTIL:
            try:
                notes += make_call(state)
            except Exception as exc:  # no bars yet at 9:30 sharp: the 9:45 run tries again
                notes.append(f"Day call not made yet ({type(exc).__name__})")
            if now.time() >= BUY_FROM:
                notes += enter_calls(state)
        if ENTRY_FROM <= now.time() < ENTRY_UNTIL:
            notes += enter(state)
    notes += manage(state)
    score_calls(state)
    _save(state)
    return notes


if __name__ == "__main__":
    if "--report" in sys.argv:
        for s in summary():
            avg = "n/a" if s["avg"] is None else f"${s['avg']:+.2f}"
            print(f"{s['name']:45} {s['trades']} closed, {s['open']} open, {s['wins']} won, avg {avg}, total ${s['total']:+.2f}")
    else:
        for line in run() or ["0DTE lab: nothing to do"]:
            print(line)
