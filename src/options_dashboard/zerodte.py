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
    python -m options_dashboard.zerodte --report   # each variant's paper portfolio
    python -m options_dashboard.zerodte --review   # the daily reviews and what they add up to
    python -m options_dashboard.zerodte --observations   # test the logged gamma, flow and breadth
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
    {"id": "day_spx150_active", "name": "Day call: SPX, up to $150, skipping quiet days", "ticker": "SPX", "cap": 1.50,
     "target": 2.0, "strategy": "daycall", "skip_quiet": True},
    # Trend days: a strong first hour that closes near its extreme, traded in its direction from about
    # 10:30 prices. The plain version lost money in the backtest; the gamma filter (SPY dealers net
    # short gamma, so hedging chases the move) is the user's framework and is tested here, live.
    {"id": "trend_spx150_gamma", "name": "Trend day: SPX, up to $150, dealers short gamma", "ticker": "SPX",
     "cap": 1.50, "target": 2.0, "strategy": "trend", "short_gamma": True},
    {"id": "trend_spx150", "name": "Trend day: SPX, up to $150, no gamma filter (control)", "ticker": "SPX",
     "cap": 1.50, "target": 2.0, "strategy": "trend"},
    {"id": "trend_spy40_gamma", "name": "Trend day: SPY, up to $40, dealers short gamma", "ticker": "SPY",
     "cap": 0.40, "target": 2.0, "strategy": "trend", "short_gamma": True},
    # Market profile setups at 10:30 (daycall.profile_1030): the 80% rule and a reclaimed gap down,
    # sold when the underlying reaches the setup's level (the far side of value), otherwise at the close.
    {"id": "profile_spx150", "name": "Market profile: SPX, up to $150", "ticker": "SPX", "cap": 1.50,
     "strategy": "profile"},
    {"id": "profile_spy40", "name": "Market profile: SPY, up to $40", "ticker": "SPY", "cap": 0.40,
     "strategy": "profile"},
    # Near the money (added Oct 31, 2026): $150 at the open buys an SPX strike about 1% away, and replayed
    # confident days travelled that far their way only 13% of the time (0.4%: 57%). The same $150 buys an
    # SPY strike a dollar or two out, inside the moves that actually happen.
    {"id": "day_spy150", "name": "Day call: SPY, up to $150, near the money", "ticker": "SPY", "cap": 1.50,
     "target": 2.0, "strategy": "daycall"},
    {"id": "trend_spy150", "name": "Trend day: SPY, up to $150, near the money", "ticker": "SPY", "cap": 1.50,
     "target": 2.0, "strategy": "trend"},
    {"id": "profile_spy150", "name": "Market profile: SPY, up to $150, near the money", "ticker": "SPY", "cap": 1.50,
     "strategy": "profile"},
]
FAMILIES = {"daycall": "Day call", "trend": "Trend day", "profile": "Market profile", "momentum": "End-of-day momentum"}
ENTRY_AT = {"daycall": (9, 30), "trend": (10, 30), "profile": (10, 30), "momentum": (15, 30)}  # what the quotes show
START_BALANCE = 1000.0  # each variant's paper portfolio
# Replayed day calls (walk-forward, Nov 2023 - Oct 2026): when the first hour ran 0.25% or more against a
# confident call, the day closed the call's way only 27% of the time (33 days; 73% when it ran with the
# call), so day-call contracts are sold at 10:30 then: the call has been proved wrong.
INVALIDATE = 0.0025
UPDATE_FROM, UPDATE_UNTIL = clock(10, 30), clock(10, 45)  # once the first hour is complete
TREND_FROM, TREND_UNTIL = clock(10, 45), clock(11, 0)  # the 10:45 scan's delayed chain shows about 10:30
TREND_MOVE, TREND_EDGE = 0.0025, 0.25  # first hour at least 0.25%, closing in its top or bottom quarter
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


def _family(variant_id: str) -> str:
    return next(x for x in VARIANTS if x["id"] == variant_id).get("strategy", "momentum")


def _cash(state: dict[str, Any], variant_id: str) -> float:
    """A variant's portfolio cash: the start, plus closed profits, less what open trades cost."""
    cash = START_BALANCE
    for t in state["trades"]:
        if t["variant"] == variant_id:
            cash += t["pnl"] if t["status"] == "closed" else -(t["entry"] * 100 + FEES[t["ticker"]])
    return cash


def _buy(state: dict[str, Any], v: dict[str, Any], pick: pd.Series, ref: float, why: str, **extra: Any) -> str | None:
    """Open one contract at the real ask, if the variant's portfolio can pay for it."""
    cost = float(pick["ask"]) * 100 + FEES[v["ticker"]]
    if cost > _cash(state, v["id"]):
        return f"{v['name']}: skipped, the portfolio can't cover ${cost:.0f}"
    kind = "call" if pick["type"] == "call" else "put"
    state["trades"].append({"variant": v["id"], "date": datetime.now(data.NEW_YORK).date().isoformat(),
                            "ticker": v["ticker"], "type": kind, "strike": float(pick["strike"]),
                            "symbol": pick["contractSymbol"], "entry": float(pick["ask"]), "entry_bid": float(pick["bid"]),
                            "spot": ref, "why": why, "status": "open", **extra,
                            "opened": datetime.now().isoformat(timespec="minutes")})
    return f"{v['name']}: bought {v['ticker']} {pick['strike']:g} {kind} at ${pick['ask']:.2f}"


def _close_on(ticker: str, d: str, cache: dict[str, pd.DataFrame]) -> float | None:
    """The official close of `ticker` ("SPX", "SPY") on date `d`, or None while Yahoo has no good bar
    (right after the close its daily bar can come back blank)."""
    if ticker not in cache:
        try:
            daily = data.call(yf.Ticker(SOURCES[ticker][0]).history, period="1mo", interval="1d", auto_adjust=False)
            daily.index = daily.index.tz_convert("America/New_York").date
            cache[ticker] = daily
        except Exception:
            return None
    daily, day = cache[ticker], pd.Timestamp(d).date()
    if day not in daily.index or pd.isna(daily.loc[day, "Close"]):
        return None
    return float(daily.loc[day, "Close"])


def _heal(state: dict[str, Any]) -> None:
    """Undo anything settled or scored on a blank close, so it is redone on a good one."""
    def blank(x: Any) -> bool:
        return isinstance(x, float) and x != x
    for t in state["trades"]:
        if t["status"] == "closed" and (blank(t.get("pnl")) or blank(t.get("exit"))):
            for k in ("exit", "close", "how", "pnl"):
                t.pop(k, None)
            t["status"] = "open"
    for day in state.get("calls", {}).values():
        for k in ("SPX", "SPY", "update"):
            if isinstance(day.get(k), dict) and blank(day[k].get("close")):
                day[k].pop("close", None)
                day[k].pop("outcome", None)
    for o in state.get("observe", {}).values():
        if blank(o.get("close")):
            for k in ("open", "high", "low", "close"):
                o.pop(k, None)


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
        note = _buy(state, v, pick, price, why)
        record[v["id"]] = "skipped, portfolio too small" if "skipped" in note else "bought"
        notes.append(note)
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
    if v["confident"] and v.get("range") == "quiet":
        lines.append(f"**Market leans {v['direction']}, but a quiet day is forecast: stand aside** "
                     f"(out-of-the-money 0DTE options need movement) · {daycall.describe(v)}")
    elif v["confident"]:
        lines.append(f"**Market {'▲ up' if v['direction'] == 'up' else '▼ down'} "
                     f"{max(v['p_up'], 1 - v['p_up']):.0%}** · {daycall.describe(v)}")
        side = "call" if v["direction"] == "up" else "put"
        for var in (x for x in VARIANTS if x.get("strategy") == "daycall" and not x.get("skip_quiet")):
            name = var["ticker"]
            k = daycall.estimate_strike(c[name], name, var["cap"])
            away = abs(k / c[name]["open"] - 1) if k else None
            lines.append(f"→ **{name}**: buy the {'SPXW' if name == 'SPX' else 'SPY'} 0DTE {side}, nearest strike out of the "
                         f"money with an ask up to ${var['cap']:.2f}" + (f" (about {k:,.0f}, {away:.2%} away" + (
                             f", {away / v['expected_move']:.1f}x the expected move" if v.get("expected_move") else "")
                             + ")" if k else "")
                         + f"; sell at {var['target']:.0f}x or hold to the close")
        lines.append("Sold at 10:30 if the first hour runs 0.25% or more against the call (such days closed the "
                     "call's way only 27% of the time in a two-year replay).")
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
        if not c or not c["confident"] or (v.get("skip_quiet") and c.get("range") == "quiet"):
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
        notes.append(_buy(state, v, pick, ref, f"day call {c['p_up']:.0%} up"))
    day["entered"] = True
    return notes


def invalidate_calls(state: dict[str, Any]) -> list[str]:
    """From the 10:45 scan (about 10:30 prices): sell today's day-call contracts whose first hour ran
    INVALIDATE or more against them, a structural stop (see INVALIDATE). Zero bids are left to settle."""
    today = datetime.now(data.NEW_YORK).date().isoformat()
    u = (state.get("calls", {}).get(today) or {}).get("update")
    if not u:
        return []
    notes, chains = [], {}
    for t in state["trades"]:
        if t["date"] != today or t["status"] != "open" or _family(t["variant"]) != "daycall" or t.get("stop_checked"):
            continue
        t["stop_checked"] = True
        against = -u["first_hour"]["h1_ret"] * (1 if t["type"] == "call" else -1)
        if against < INVALIDATE:
            continue
        try:
            chains.setdefault(t["ticker"], data.cboe_chain(SOURCES[t["ticker"]][1])[0])
            q = chains[t["ticker"]][chains[t["ticker"]]["contractSymbol"] == t["symbol"]]
            bid = float(q["bid"].iloc[0])
        except Exception:
            continue
        if bid <= 0:
            continue
        t.update(status="closed", exit=bid, stopped=True,
                 how=f"sold at 10:30 (bid ${bid:.2f}): the first hour ran {against:.2%} against the call",
                 pnl=round((bid - t["entry"]) * 100 - 2 * FEES[t["ticker"]], 2))
        notes.append(f"{next(x['name'] for x in VARIANTS if x['id'] == t['variant'])}: stopped at 10:30, "
                     f"{'+' if t['pnl'] >= 0 else '-'}${abs(t['pnl']):.0f}")
    if notes:
        _post_discord("🛑 Day-call stop (paper)", "The first hour proved the morning call wrong:\n" + "\n".join(notes))
    return notes


def make_update(state: dict[str, Any]) -> list[str]:
    """At 10:30: the update call from the first hour, posted to Discord and kept for scoring."""
    from . import daycall
    today = datetime.now(data.NEW_YORK).date().isoformat()
    day = state.setdefault("calls", {}).setdefault(today, {})
    if "update" in day:
        return []
    u = daycall.update_1030()
    day["update"] = u
    head = (f"**SPY {'▲ up' if u['direction'] == 'up' else '▼ down'} {max(u['p_up'], 1 - u['p_up']):.0%} from here**"
            if u["confident"] else "**No confident update: stand aside**")
    fh = u["first_hour"]
    trend = abs(fh["h1_ret"]) >= TREND_MOVE and (fh["h1_loc"] >= 1 - TREND_EDGE if fh["h1_ret"] > 0
                                                  else fh["h1_loc"] <= TREND_EDGE)
    gamma = _spy_short_gamma()
    lines = [head, daycall.describe_update(u)]
    if trend:
        lines.append(f"Trend-day setup: a strong first hour closing near its {'high' if fh['h1_ret'] > 0 else 'low'}; "
                     + ("dealers are short gamma, so hedging should chase it" if gamma else
                        "dealers are not short gamma, so the move has less fuel" if gamma is False else "gamma unknown"))
    try:
        pr = daycall.profile_1030()
        day["profile"] = pr
        lines.append("Market profile: " + daycall.describe_profile(pr))
    except Exception:
        pass
    lines.append("Paper-tested in the 0DTE lab; research, not advice.")
    _post_discord("⏱ 10:30 update", "\n".join(lines))
    return [f"10:30 update: {u['p_up']:.0%} up from {u['price']:.2f}"]


def _nearest_otm(ticker: str, up: bool, cap: float) -> tuple[pd.Series, float] | None:
    """The nearest out-of-the-money 0DTE contract within the cap, and the underlying it was quoted against."""
    try:
        chain = _today_chain(ticker)
        ref = float(data.cboe_chain(SOURCES[ticker][1])[1].get("price"))
    except Exception:
        return None
    side = chain[(chain["type"] == ("call" if up else "put"))
                 & ((chain["strike"] > ref) if up else (chain["strike"] < ref)) & (chain["ask"] <= cap)]
    if side.empty:
        return None
    return side.iloc[(side["strike"] - ref).abs().argsort()].iloc[0], ref


def enter_profile(state: dict[str, Any]) -> list[str]:
    """From the 10:45 scan: when a market profile setup is live, buy each profile variant's contract."""
    today = datetime.now(data.NEW_YORK).date().isoformat()
    day = state.get("calls", {}).get(today) or {}
    pr = day.get("profile") or {}
    s = pr.get("setup")
    if not s or day.get("profile_entered"):
        return []
    day["profile_entered"] = True
    up = s["direction"] == "up"
    notes = []
    for v in (x for x in VARIANTS if x.get("strategy") == "profile"):
        got = _nearest_otm(v["ticker"], up, v["cap"])
        if not got:
            continue
        pick, ref = got
        # The level is set on SPY; scale it to the index for SPX.
        level = ref * s["target"] / pr["price"] if s.get("target") else None
        note = _buy(state, v, pick, ref, f"{s['name']}: {s['why']}", target_level=level)
        notes.append(note + ("" if "skipped" in note else f", sell at {level:,.2f}" if level else ", hold to the close"))
    if notes:
        _post_discord(f"📊 Market profile trade (paper): {s['name']}", "\n".join(notes))
    return notes


def enter_trend(state: dict[str, Any]) -> list[str]:
    """From the 10:45 scan: on a trend-day setup, buy each trend variant's contract at the real ask."""
    today = datetime.now(data.NEW_YORK).date().isoformat()
    day = state.get("calls", {}).get(today) or {}
    u = day.get("update")
    if not u or day.get("trend_entered"):
        return []
    day["trend_entered"] = True
    fh = u["first_hour"]
    up = fh["h1_ret"] > 0
    if not (abs(fh["h1_ret"]) >= TREND_MOVE and (fh["h1_loc"] >= 1 - TREND_EDGE if up else fh["h1_loc"] <= TREND_EDGE)):
        return []
    gamma = _spy_short_gamma()
    notes = []
    for v in (x for x in VARIANTS if x.get("strategy") == "trend"):
        if v.get("short_gamma") and gamma is not True:
            continue
        got = _nearest_otm(v["ticker"], up, v["cap"])
        if not got:
            continue
        pick, ref = got
        notes.append(_buy(state, v, pick, ref,
                          f"trend day: first hour {fh['h1_ret']:+.2%}, gamma {'short' if gamma else 'not short'}"))
    if notes:
        _post_discord("📈 Trend-day trades (paper)", "\n".join(notes))
    return notes


def observe(state: dict[str, Any]) -> None:
    """Log each scan's SPY dealer positioning, customer flow, method scores and breadth through the
    day, so they can later be tested against how the day ended (no free history exists for them)."""
    try:
        last = json.loads((STATE_DIR / "last_scan.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    try:  # the scan stamps its machine's local time (UTC on GitHub's runners)
        stamp = pd.Timestamp(last["time"]).tz_localize(datetime.now().astimezone().tzinfo).tz_convert(data.NEW_YORK)
    except (KeyError, ValueError, TypeError):
        return
    if stamp.date() != datetime.now(data.NEW_YORK).date():
        return
    day = state.setdefault("observe", {}).setdefault(stamp.date().isoformat(), {"scans": {}})
    day["scans"].setdefault(stamp.strftime("%H:%M"), {k: last.get(k) for k in (
        "spy_read", "spy_gamma", "spy_methods", "breadth")})


def observations_report(state: dict[str, Any] | None = None, at: str = "10:30") -> list[str]:
    """Test the logged factors: from the scan nearest `at` each day, does the rest of the day follow
    the dealers' gamma, customer delta, breadth and the dashboard's SPY read, and how far does it go?"""
    state = state or load()
    rows = []
    for d, o in sorted(state.get("observe", {}).items()):
        if "close" not in o or not o["scans"]:
            continue
        t = min(o["scans"], key=lambda s: abs(pd.Timestamp(f"{d} {s}") - pd.Timestamp(f"{d} {at}")))
        s = o["scans"][t]
        spot = (s.get("spy_read") or {}).get("spot")
        if not spot:
            continue
        g, b = s.get("spy_gamma") or {}, s.get("breadth") or {}
        rows.append({"rest": o["close"] / spot - 1, "short_gamma": None if g.get("net_gex") is None else g["net_gex"] < 0,
                     "customer_long": None if g.get("customer_dex") is None else g["customer_dex"] > 0,
                     "breadth_up": None if b.get("up") is None else b["up"] > 0.5,
                     "dashboard_bull": (s.get("spy_read") or {}).get("trend", 0) > 0})
    if not rows:
        return ["No completed observation days yet"]
    df = pd.DataFrame(rows)
    out = [f"{len(df)} days, scan nearest {at}; rest of day up {(df['rest'] > 0).mean():.0%}, "
           f"average size {df['rest'].abs().mean():.2%}"]
    for col, yes, no in (("short_gamma", "dealers short gamma", "dealers long gamma"),
                         ("customer_long", "customers net long delta", "customers net short delta"),
                         ("breadth_up", "most tickers up", "most tickers down"),
                         ("dashboard_bull", "dashboard reads SPY up", "dashboard reads SPY down")):
        for flag, name in ((True, yes), (False, no)):
            g = df[df[col] == flag]
            if len(g):
                out.append(f"  {name:28} {len(g):3d} days: rest of day up {(g['rest'] > 0).mean():.0%}, "
                           f"average size {g['rest'].abs().mean():.2%}")
    return out


def calls_scorecard(state: dict[str, Any]) -> dict[str, Any]:
    """How the day calls have done: all calls, confident calls, and the dashboard's SPY read."""
    out = {"days": 0, "right": 0, "confident": 0, "confident_right": 0, "dashboard": 0, "dashboard_right": 0,
           "update": 0, "update_right": 0, "update_confident": 0, "update_confident_right": 0,
           "up_calls": 0, "up_right": 0, "down_calls": 0, "down_right": 0, "range": {}, "profile": {}}
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
                side = "up" if c["direction"] == "up" else "down"
                out[f"{side}_calls"] += 1
                out[f"{side}_right"] += right
            if c.get("range") and c.get("close") and c.get("open"):
                b = out["range"].setdefault(c["range"], {"days": 0, "move": 0.0})
                b["days"] += 1
                b["move"] += abs(c["close"] / c["open"] - 1)
        s = (day.get("profile") or {}).get("setup") or {}
        if "outcome" in s:
            b = out["profile"].setdefault(s["name"], {"days": 0, "right": 0, "targets": 0, "reached": 0})
            b["days"] += 1
            b["right"] += s["direction"] == s["outcome"]
            if s.get("reached") is not None:
                b["targets"] += 1
                b["reached"] += s["reached"]
        u = day.get("update") or {}
        if "outcome" in u:
            r = u["direction"] == u["outcome"]
            out["update"] += 1
            out["update_right"] += r
            if u["confident"]:
                out["update_confident"] += 1
                out["update_confident_right"] += r
        d = day.get("dashboard_spy") or {}
        if d.get("bias") in ("bullish", "bearish") and "outcome" in (day.get("SPY") or {}):
            out["dashboard"] += 1
            out["dashboard_right"] += (d["bias"] == "bullish") == (day["SPY"]["outcome"] == "up")
    return out


def score_calls(state: dict[str, Any]) -> None:
    """After the close: did each index close above its open, and SPY above its 10:30 price?"""
    now = datetime.now(data.NEW_YORK)
    closes: dict[str, pd.DataFrame] = {}
    pending = [d for d, o in state.get("observe", {}).items() if "close" not in o
               and not (pd.Timestamp(d).date() == now.date() and now.time() < clock(16, 5))]
    if pending:
        try:
            daily = data.call(yf.Ticker("SPY").history, period="1mo", interval="1d", auto_adjust=False)
            daily.index = daily.index.tz_convert("America/New_York").date
        except Exception:
            daily = None
        for d in pending if daily is not None else []:
            if pd.Timestamp(d).date() in daily.index and not pd.isna(daily.loc[pd.Timestamp(d).date(), "Close"]):
                bar = daily.loc[pd.Timestamp(d).date()]
                state["observe"][d].update(open=float(bar["Open"]), high=float(bar["High"]),
                                           low=float(bar["Low"]), close=float(bar["Close"]))
    for d, day in state.get("calls", {}).items():
        pr = day.get("profile") or {}
        s = pr.get("setup")
        if s and "outcome" not in s and not (pd.Timestamp(d).date() == now.date() and now.time() < clock(16, 5)):
            try:
                m5 = data.call(yf.Ticker("SPY").history, period="60d", interval="5m")
                m5.index = m5.index.tz_convert("America/New_York")
                rest = m5[(m5.index.date == pd.Timestamp(d).date()) & (m5.index.hour * 60 + m5.index.minute >= 630)
                          & (m5.index.hour < 16)]
                if len(rest):
                    up = s["direction"] == "up"
                    close = float(rest["Close"].iloc[-1])
                    s.update(close=close, outcome="up" if close > pr["price"] else "down",
                             reached=None if not s.get("target") else bool(
                                 rest["High"].max() >= s["target"] if up else rest["Low"].min() <= s["target"]))
            except Exception:
                pass
        if pd.Timestamp(d).date() == now.date() and now.time() < clock(16, 5):
            continue
        u = day.get("update")
        if u and "outcome" not in u and (close := _close_on("SPY", d, closes)) is not None:
            u.update(close=close, outcome="up" if close > u["price"] else "down")
        for name in ("SPX", "SPY"):
            c = day.get(name)
            if c and "outcome" not in c and (close := _close_on(name, d, closes)) is not None:
                c.update(close=close, outcome="up" if close > c["open"] else "down")


def manage(state: dict[str, Any]) -> list[str]:
    """Take 3x targets on the next snapshot; settle the rest at the official close."""
    notes = []
    now = datetime.now(data.NEW_YORK)
    open_trades = [t for t in state["trades"] if t["status"] == "open"]
    if not open_trades:
        return notes
    chains: dict[str, tuple[pd.DataFrame, dict]] = {}
    closes: dict[str, pd.DataFrame] = {}
    for t in open_trades:
        v = next(x for x in VARIANTS if x["id"] == t["variant"])
        if t["date"] == now.date().isoformat() and now.time() < clock(16, 30):
            try:  # each later snapshot (15 minutes delayed): did it reach the target?
                chains.setdefault(t["ticker"], data.cboe_chain(SOURCES[t["ticker"]][1]))
                chain, quote = chains[t["ticker"]]
                q = chain[chain["contractSymbol"] == t["symbol"]]
            except Exception:
                q, quote = pd.DataFrame(), {}
            if q.empty:
                continue
            bid, spot = float(q["bid"].iloc[0]), quote.get("price")
            if t.get("target_level"):  # profile trades: the underlying reached the setup's level
                hit = spot is not None and (spot >= t["target_level"] if t["type"] == "call" else spot <= t["target_level"])
            else:
                hit = bool(v.get("target")) and bid >= t["entry"] * v["target"]
            if hit and bid > 0:
                t.update(status="closed", exit=bid, how=f"target hit (bid ${bid:.2f})",
                         pnl=round((bid - t["entry"]) * 100 - 2 * FEES[t["ticker"]], 2))
                notes.append(f"{v['name']}: target hit, +${t['pnl']:.0f}")
            continue
        close = _close_on(t["ticker"], t["date"], closes)  # settle once the day's bar is good
        if close is None:
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


def portfolios(state: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Each variant as a paper portfolio of START_BALANCE: balance, value over time (one point per
    trading day), return, largest drop from a high, and the win-loss record."""
    state = state or load()
    start = state.get("started", "")[:10] or datetime.now(data.NEW_YORK).date().isoformat()
    days = sorted({t["date"] for t in state["trades"]} | {start})
    out = []
    for v in VARIANTS:
        mine = [t for t in state["trades"] if t["variant"] == v["id"]]
        closed = [t for t in mine if t["status"] == "closed"]
        curve, value, peak, drop = [[f"{start}T09:30", START_BALANCE]], START_BALANCE, START_BALANCE, 0.0
        for d in days:
            value += sum(t["pnl"] for t in closed if t["date"] == d)
            curve.append([f"{d}T16:00", round(value, 2)])
            peak = max(peak, value)
            drop = max(drop, 1 - value / peak)
        pnls = [t["pnl"] for t in closed]
        out.append({"id": v["id"], "name": v["name"], "family": v.get("strategy", "momentum"),
                    "balance": round(START_BALANCE + sum(pnls), 2), "return": sum(pnls) / START_BALANCE,
                    "trades": len(closed), "open": len(mine) - len(closed), "wins": sum(p > 0 for p in pnls),
                    "best": max(pnls, default=None), "worst": min(pnls, default=None), "drop": drop, "curve": curve})
    return out


def _intraday(ticker: str, d: str, cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if ticker not in cache:
        h = data.call(yf.Ticker(SOURCES[ticker][0]).history, period="60d", interval="5m")
        h.index = h.index.tz_convert("America/New_York")
        cache[ticker] = h[(h.index.hour * 60 + h.index.minute >= 570) & (h.index.hour < 16)]
    h = cache[ticker]
    return h[h.index.date == pd.Timestamp(d).date()]


def _held(t: dict[str, Any], close: float) -> float:
    """What a trade would have made held to the close."""
    value = max(close - t["strike"], 0) if t["type"] == "call" else max(t["strike"] - close, 0)
    return round((value - t["entry"]) * 100 - FEES[t["ticker"]] * (2 if value > 0 else 1), 2)


def review_day(state: dict[str, Any], d: str) -> dict[str, Any]:
    """After the close: what the market did against the forecasts, how each call did, why each trade
    won or lost (wrong way, right way but short of the strike, target, stop), and the lessons."""
    bars: dict[str, pd.DataFrame] = {}
    spy = _intraday("SPY", d, bars)
    day = state.get("calls", {}).get(d) or {}
    c, u, pr = day.get("SPY") or {}, day.get("update") or {}, (day.get("profile") or {}).get("setup") or {}
    o, close = float(spy["Open"].iloc[0]), float(spy["Close"].iloc[-1])
    first = spy[spy.index.hour * 60 + spy.index.minute < 630]
    h1 = float(first["Close"].iloc[-1]) / o - 1
    market = {"open": o, "high": float(spy["High"].max()), "low": float(spy["Low"].min()), "close": close,
              "move": close / o - 1, "range": float(spy["High"].max() / spy["Low"].min() - 1), "h1": h1}
    if c.get("expected_move"):
        market.update(expected=c["expected_move"], range_label=c.get("range"),
                      ratio=abs(market["move"]) / c["expected_move"])
    if u.get("price") and u.get("afternoon"):
        market.update(rest=close / u["price"] - 1, rest_expected=u["afternoon"]["expected"],
                      rest_size=u["afternoon"]["size"])

    calls = []
    if c.get("outcome"):
        calls.append({"name": "Day call (at the open)", "said": f"{c['direction']} {max(c['p_up'], 1 - c['p_up']):.0%}"
                      + ("" if c["confident"] else ", not confident"), "right": c["direction"] == c["outcome"],
                      "confident": c["confident"]})
    if u.get("outcome"):
        calls.append({"name": "10:30 update", "said": f"{u['direction']} {max(u['p_up'], 1 - u['p_up']):.0%}"
                      + ("" if u["confident"] else ", not confident"), "right": u["direction"] == u["outcome"],
                      "confident": u["confident"]})
    if pr.get("outcome"):
        calls.append({"name": f"Market profile: {pr['name']}", "said": pr["direction"],
                      "right": pr["direction"] == pr["outcome"], "confident": True,
                      "note": None if pr.get("reached") is None else ("reached its level" if pr["reached"] else "missed its level")})
    dash = day.get("dashboard_spy") or {}
    if dash.get("bias") in ("bullish", "bearish"):
        calls.append({"name": "Dashboard's SPY read", "said": dash["bias"], "confident": True,
                      "right": (dash["bias"] == "bullish") == (market["move"] > 0)})

    trades = []
    for t in (t for t in state["trades"] if t["date"] == d and t["status"] == "closed"):
        fam, sign = _family(t["variant"]), 1 if t["type"] == "call" else -1
        b = _intraday(t["ticker"], d, bars)
        hh, mm = ENTRY_AT[fam]
        after = b[b.index.hour * 60 + b.index.minute >= hh * 60 + mm]
        last = float(b["Close"].iloc[-1])
        dist = (t["strike"] / t["spot"] - 1) * sign
        best = (float(after["High"].max()) / t["spot"] - 1) if sign > 0 else (1 - float(after["Low"].min()) / t["spot"])
        went = (last / t["spot"] - 1) * sign
        held = _held(t, t.get("close") or last)
        if t.get("stopped"):
            verdict = (f"Stopped at 10:30 when the first hour proved the call wrong; held to the close it would have "
                       f"made {_usd(held)}, so the stop {'saved' if t['pnl'] > held else 'cost'} ${abs(t['pnl'] - held):.0f}")
        elif t["how"].startswith("target"):
            verdict = f"Won: target hit ({t['ticker']} travelled up to {best:.2%} its way)"
        elif went <= 0:
            verdict = (f"Wrong direction: {t['ticker']} finished {abs(went):.2%} the other way"
                       + (f" (it did reach the strike first, {dist:.2%} away)" if best >= dist else ""))
        elif went < dist:
            verdict = (f"Right direction, not far enough: the strike was {dist:.2%} away; {t['ticker']} finished "
                       f"{went:.2%} its way (best {best:.2%})")
        elif t["pnl"] <= 0:
            verdict = f"Past the strike, but not by enough to pay back the ${t['entry']:.2f} premium"
        else:
            verdict = f"Won at the close, {went:.2%} its way against a strike {dist:.2%} out"
        trades.append({"variant": t["variant"], "name": next(x["name"] for x in VARIANTS if x["id"] == t["variant"]),
                       "family": fam, "contract": f"{t['ticker']} {t['strike']:g} {t['type']}", "entry": t["entry"],
                       "exit": t.get("exit"), "pnl": t["pnl"], "dist": dist, "best": best, "went": went,
                       "reached": best >= dist, "held": held, "stopped": bool(t.get("stopped")), "verdict": verdict})

    lessons = []
    if c.get("confident") and c.get("outcome"):
        against = -h1 * (1 if c["direction"] == "up" else -1)
        if against >= INVALIDATE:
            lessons.append(f"The first hour ({h1:+.2%}) overruled the morning call. In two years of replayed calls, days like "
                           f"this closed the call's way only 27% of the time, so day-call contracts are now sold at 10:30.")
        elif against <= -INVALIDATE:
            lessons.append(f"The first hour ({h1:+.2%}) confirmed the morning call; replayed, such days closed the call's way 73%.")
    far = {tr["contract"]: tr for tr in trades if tr["family"] == "daycall" and market.get("expected")
           and tr["dist"] > 1.5 * market["expected"]}
    if far:
        lessons.append("Day-call strikes were " + ", ".join(
            f"{tr['dist']:.2%} away ({k}, {tr['dist'] / market['expected']:.1f}x the forecast move)" for k, tr in far.items())
            + f" against a forecast move of {market['expected']:.2%}. Replayed confident days travelled 1% their way only "
              "13% of the time and 0.4% about 57%: the near-the-money SPY variant tests the fix.")
    if market.get("ratio") is not None:
        if market["ratio"] > 1.5:
            lessons.append(f"A bigger day than forecast: SPY moved {abs(market['move']):.2%} open to close against "
                           f"{market['expected']:.2%} expected ({market['ratio']:.1f}x).")
        elif market["ratio"] < 0.5:
            lessons.append(f"A quieter day than forecast: SPY moved {abs(market['move']):.2%} against "
                           f"{market['expected']:.2%} expected.")
    if market.get("rest_expected"):
        r = abs(market["rest"]) / market["rest_expected"]
        if r > 1.5 or r < 0.5:
            lessons.append(f"After 10:30 SPY moved {abs(market['rest']):.2%}, against {market['rest_expected']:.2%} for a "
                           f"'{market['rest_size']}' first hour ({r:.1f}x).")
    for tr in trades:
        if tr["family"] == "trend":
            lessons.append(f"Trend day ({tr['contract']}): {'worked' if tr['pnl'] > 0 else 'failed'}, the first hour's "
                           f"direction {'carried' if tr['went'] > 0 else 'reversed'} ({tr['went']:+.2%} after 10:30).")
    missed = [tr for tr in trades if tr["reached"] and tr["pnl"] <= 0 and not tr["stopped"]]
    for tr in missed:
        lessons.append(f"{tr['name']} touched its strike ({tr['best']:.2%} its way at best) but still lost: "
                       f"a sale on the way, not a hold to the close, would have paid.")
    return {"date": d, "market": market, "calls": calls, "trades": trades, "lessons": lessons,
            "pnl": round(sum(tr["pnl"] for tr in trades), 2)}


def _usd(x: float) -> str:
    return f"{'+' if x >= 0 else '-'}${abs(x):,.0f}"


def describe_review(r: dict[str, Any]) -> str:
    m = r["market"]
    lines = [f"**SPY {m['move']:+.2%}** open to close (range {m['range']:.2%}, first hour {m['h1']:+.2%})"
             + (f"; forecast {m['expected']:.2%}, actual {m['ratio']:.1f}x" if "ratio" in m else "")]
    for c in r["calls"]:
        mark = ("✅" if c["right"] else "❌") if c["confident"] else "➖"  # an unconfident call claimed nothing
        lines.append(f"{mark} {c['name']}: {c['said']}, the day went {'its way' if c['right'] else 'the other way'}"
                     + (f", {c['note']}" if c.get("note") else ""))
    for t in r["trades"]:
        lines.append(f"• {t['name']} ({t['contract']} at ${t['entry']:.2f}): **{_usd(t['pnl'])}**. {t['verdict']}")
    if r["lessons"]:
        lines.append("**Lessons**")
        lines += [f"→ {x}" for x in r["lessons"]]
    lines.append(f"Lab total today: {_usd(r['pnl'])}")
    return "\n".join(lines)


def review(state: dict[str, Any]) -> list[str]:
    """Review each finished trading day once: every trade closed and the calls scored."""
    now = datetime.now(data.NEW_YORK)
    reviews = state.setdefault("reviews", {})
    notes = []
    days = sorted({t["date"] for t in state["trades"]} | set(state.get("calls", {})))
    for d in days:
        if d in reviews or (pd.Timestamp(d).date() == now.date() and now.time() < clock(16, 5)):
            continue
        if any(t["date"] == d and t["status"] == "open" for t in state["trades"]):
            continue
        c = (state.get("calls", {}).get(d) or {}).get("SPY") or {}
        if c and "outcome" not in c:
            continue
        try:
            r = review_day(state, d)
        except Exception as exc:
            notes.append(f"Review of {d} not made ({type(exc).__name__})")
            continue
        reviews[d] = r  # shown on the site; the user keeps breakdowns of results out of Discord
        notes.append(f"Reviewed {d}: {_usd(r['pnl'])}")
    return notes


def lessons_scorecard(state: dict[str, Any] | None = None) -> dict[str, Any]:
    """What the reviews add up to: do the fixes work, and how well are direction and size forecast?"""
    state = state or load()
    rs = list(state.get("reviews", {}).values())
    out: dict[str, Any] = {"days": len(rs), "families": {}, "overruled": 0, "overruled_right": 0, "stops": 0,
                           "stop_saved": 0.0, "ratios": [], "rest_ratios": []}
    for r in rs:
        m = r["market"]
        if m.get("ratio") is not None:
            out["ratios"].append(m["ratio"])
        if m.get("rest_expected"):
            out["rest_ratios"].append(abs(m["rest"]) / m["rest_expected"])
        day = (state.get("calls", {}).get(r["date"]) or {}).get("SPY") or {}
        if day.get("confident") and day.get("outcome") and -m["h1"] * (1 if day["direction"] == "up" else -1) >= INVALIDATE:
            out["overruled"] += 1
            out["overruled_right"] += day["direction"] == day["outcome"]
        for t in r["trades"]:
            f = out["families"].setdefault(t["family"], {"trades": 0, "pnl": 0.0, "wins": 0, "reached": 0, "dist": 0.0,
                                                         "best": 0.0, "wrong_way": 0, "short": 0})
            f["trades"] += 1
            f["pnl"] += t["pnl"]
            f["wins"] += t["pnl"] > 0
            f["reached"] += t["reached"]
            f["dist"] += t["dist"]
            f["best"] += t["best"]
            f["wrong_way"] += t["went"] <= 0 and t["pnl"] <= 0
            f["short"] += 0 < t["went"] < t["dist"] and t["pnl"] <= 0
            if t["stopped"]:
                out["stops"] += 1
                out["stop_saved"] += t["pnl"] - t["held"]
    return out


def describe_scorecard(s: dict[str, Any]) -> list[str]:
    """Plain sentences from lessons_scorecard."""
    import statistics
    if not s["days"]:
        return ["No days reviewed yet."]
    out = [f"{s['days']} day{'s' if s['days'] > 1 else ''} reviewed."]
    for fam, f in s["families"].items():
        n = f["trades"]
        out.append(f"{FAMILIES[fam]}: {n} trade{'s' if n > 1 else ''}, {f['wins']} won, {_usd(f['pnl'])}. Strikes averaged "
                   f"{f['dist'] / n:.2%} away and the market's best run their way {f['best'] / n:.2%}; reached the strike "
                   f"{f['reached']} of {n}. Losses: wrong way {f['wrong_way']}, right way but short of the strike {f['short']}.")
    if s["overruled"]:
        out.append(f"Morning calls overruled by the first hour: {s['overruled']}, of which {s['overruled_right']} still closed "
                   f"the call's way (27% in the replay).")
    if s["stops"]:
        out.append(f"10:30 stops: {s['stops']}, {'saving' if s['stop_saved'] >= 0 else 'costing'} "
                   f"${abs(s['stop_saved']):.0f} against holding.")
    if s["ratios"]:
        out.append(f"Size forecast: the day's actual move ran {statistics.median(s['ratios']):.1f}x the morning forecast "
                   f"(median; 1.15x in the two-year replay)"
                   + (f", and the afternoon {statistics.median(s['rest_ratios']):.1f}x the 10:30 one" if s["rest_ratios"] else "")
                   + ".")
    return out


def run() -> list[str]:
    state = load()
    _heal(state)
    now = datetime.now(data.NEW_YORK)
    notes = []
    if now.weekday() < 5 and data.market_open():
        observe(state)
        if CALL_FROM <= now.time() < CALL_UNTIL:
            try:
                notes += make_call(state)
            except Exception as exc:  # no bars yet at 9:30 sharp: the 9:45 run tries again
                notes.append(f"Day call not made yet ({type(exc).__name__})")
            if now.time() >= BUY_FROM:
                notes += enter_calls(state)
        if UPDATE_FROM <= now.time() < UPDATE_UNTIL:
            try:
                notes += make_update(state)
            except Exception as exc:
                notes.append(f"10:30 update not made ({type(exc).__name__})")
        if TREND_FROM <= now.time() < TREND_UNTIL:
            notes += invalidate_calls(state)
            notes += enter_trend(state)
            notes += enter_profile(state)
        if ENTRY_FROM <= now.time() < ENTRY_UNTIL:
            notes += enter(state)
    notes += manage(state)
    score_calls(state)
    notes += review(state)
    _save(state)
    return notes


if __name__ == "__main__":
    if "--observations" in sys.argv:
        print("\n".join(observations_report()))
    elif "--review" in sys.argv:
        for d, r in sorted(load().get("reviews", {}).items()):
            print(f"== {d}\n{describe_review(r)}\n")
        print("\n".join(describe_scorecard(lessons_scorecard())))
    elif "--report" in sys.argv:
        for p in portfolios():
            print(f"{p['name']:50} ${p['balance']:>8,.2f} ({p['return']:+.1%}), {p['trades']} closed, {p['open']} open, "
                  f"{p['wins']} won, largest drop {p['drop']:.0%}")
    else:
        for line in run() or ["0DTE lab: nothing to do"]:
            print(line)
