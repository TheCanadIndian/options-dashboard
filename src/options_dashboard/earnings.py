"""Earnings outlook: a rule-based prediction for upcoming reports, and OTM option plays.

For a stock reporting soon this combines its record against estimates, analyst
estimate revisions, revenue and margin trends, how the stock has reacted to past
reports, the run-up into the date and the current market read into a lean from
-100 to +100. It then compares the move options are pricing in with the moves the
stock has actually made, and tests out-of-the-money contracts against that history.

This is a screen built from public data, not a forecast with a measured track record.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from . import data, greeks
from .config import HOME, STATE_DIR

CACHE = STATE_DIR / "earnings"
NOTES_FILE = HOME / "research" / "earnings_notes.json"
HISTORY_REPORTS = 8  # reports used for beat rate and reaction history
BASE_BEAT_RATE = 0.70  # roughly how often large US companies beat consensus EPS


# ---------------------------------------------------------------- data

def _num(x: Any) -> float | None:
    try:
        x = float(x)
        return None if np.isnan(x) else x
    except (TypeError, ValueError):
        return None


def _fetch(ticker: str) -> dict[str, Any]:
    tk = yf.Ticker(ticker)
    out: dict[str, Any] = {}
    dates = data.call(tk.get_earnings_dates, limit=24)
    events = []
    for ts, row in dates.iterrows():
        ts = pd.Timestamp(ts).tz_convert("America/New_York")
        events.append({
            "date": ts.date().isoformat(),
            "timing": "before open" if ts.hour < 12 else "after close",
            "estimate": _num(row.get("EPS Estimate")), "actual": _num(row.get("Reported EPS")),
            "surprise": _num(row.get("Surprise(%)")),
        })
    out["events"] = sorted(events, key=lambda e: e["date"])

    def frame_row(frame, period="0q"):
        try:
            return {k: _num(v) for k, v in frame.loc[period].items()}
        except Exception:
            return {}

    out["eps_estimate"] = frame_row(data.call(lambda: tk.earnings_estimate))
    out["revenue_estimate"] = frame_row(data.call(lambda: tk.revenue_estimate))
    out["eps_trend"] = frame_row(data.call(lambda: tk.eps_trend))
    out["eps_revisions"] = frame_row(data.call(lambda: tk.eps_revisions))
    try:
        q = data.call(lambda: tk.quarterly_income_stmt)
        rows = {}
        for name in ("Total Revenue", "Operating Income", "Net Income"):
            if name in q.index:
                rows[name] = {c.date().isoformat(): _num(v) for c, v in q.loc[name].items()}
        out["quarters"] = rows
    except Exception:
        out["quarters"] = {}
    return out


def fundamentals(ticker: str) -> dict[str, Any]:
    """Estimates, revisions, report history and quarterly results, fetched once a day."""
    path = CACHE / f"{ticker}.json"
    today = date.today().isoformat()
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("fetched") == today:
            return cached
    except (OSError, json.JSONDecodeError):
        pass
    found = {"fetched": today, **_fetch(ticker)}
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(found), encoding="utf-8")
    return found


def notes() -> dict[str, Any]:
    """Research notes written by hand (by Claude in a session), keyed by ticker."""
    try:
        return json.loads(NOTES_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


# ---------------------------------------------------------------- history

def reactions(events: list[dict[str, Any]], daily: pd.DataFrame) -> list[dict[str, Any]]:
    """Close-to-close stock move across each past report.

    Before-open reports: prior close to that day's close. After-close reports: that
    day's close to the next day's close.
    """
    closes = daily["Close"]
    days = [d.date() for d in closes.index]
    out = []
    for e in events:
        if e["actual"] is None:
            continue
        day = date.fromisoformat(e["date"])
        if day not in days:
            continue
        i = days.index(day)
        before, after = (i - 1, i) if e["timing"] == "before open" else (i, i + 1)
        if before < 0 or after >= len(days):
            continue
        out.append({**e, "move": float(closes.iloc[after] / closes.iloc[before] - 1)})
    return out[-HISTORY_REPORTS:]


# ---------------------------------------------------------------- outlook

def _dollars(x: float) -> str:
    return f"{'-' if x < 0 else ''}${abs(x):.2f}"


def _factor(points: float, text: str, name: str, cap: float) -> dict[str, Any]:
    return {"name": name, "points": float(np.clip(points, -cap, cap)), "text": text}


def _quarter_change(series: dict[str, float | None]) -> tuple[float | None, float | None]:
    """Latest quarter against the same quarter a year earlier: (growth, latest value)."""
    points = sorted((d, v) for d, v in series.items() if v is not None)
    if len(points) < 5:
        return None, points[-1][1] if points else None
    latest, year_ago = points[-1][1], points[-5][1]
    return (latest / year_ago - 1 if year_ago else None), latest


def outlook(ticker: str, report: date, timing: str | None, daily: pd.DataFrame, read: float,
            chain: pd.DataFrame | None, spot: float, rate: float, cfg: dict[str, Any]) -> dict[str, Any]:
    fund = fundamentals(ticker)
    upcoming = next((e for e in fund["events"] if e["date"] == report.isoformat()), None)
    timing = timing or (upcoming or {}).get("timing") or "unknown"
    past = reactions(fund["events"], daily)
    factors = []

    reported = [e for e in fund["events"] if e["actual"] is not None and e["surprise"] is not None]
    reported = reported[-HISTORY_REPORTS:]
    beats = sum(e["surprise"] > 0 for e in reported)
    if reported:
        rate_ = beats / len(reported)
        # Median of capped surprises: a near-zero estimate turns a small miss into a huge percentage.
        avg_surprise = float(np.median([np.clip(e["surprise"], -50, 50) for e in reported]))
        factors.append(_factor(
            (rate_ - BASE_BEAT_RATE) / (1 - BASE_BEAT_RATE) * 20 + np.clip(avg_surprise / 10, -1, 1) * 5,
            f"Beat EPS estimates in {beats} of the last {len(reported)} reports, median surprise {avg_surprise:+.1f}%",
            "Beat record", 25))

    trend, revisions = fund["eps_trend"], fund["eps_revisions"]
    revision = None
    if trend.get("current") and trend.get("90daysAgo"):
        # Relative to the size of the old estimate, so a loss estimate getting deeper reads as a cut.
        revision = (trend["current"] - trend["90daysAgo"]) / abs(trend["90daysAgo"])
        up, down = int(revisions.get("upLast30days") or 0), int(revisions.get("downLast30days") or 0)
        factors.append(_factor(
            revision / 0.05 * 15 + np.clip(up - down, -5, 5),
            f"Consensus EPS for the quarter {'up' if revision >= 0 else 'down'} {abs(revision):.1%} over 90 days "
            f"({_dollars(trend['90daysAgo'])} to {_dollars(trend['current'])}); {up} upward and {down} downward "
            f"revisions in the last 30 days", "Estimate revisions", 20))

    growth, _ = _quarter_change(fund["quarters"].get("Total Revenue", {}))
    expected = fund["revenue_estimate"].get("growth")
    # Growth beyond +/-200% comes from a tiny or restated base and says nothing useful.
    growth = growth if growth is not None and abs(growth) <= 2 else None
    expected = expected if expected is not None and abs(expected) <= 2 else None
    if growth is not None or expected is not None:
        g = expected if expected is not None else growth
        parts = []
        if growth is not None:
            parts.append(f"last quarter's revenue {growth:+.1%} year over year")
        if expected is not None:
            parts.append(f"analysts expect {expected:+.1%} this quarter")
        factors.append(_factor(np.clip(g / 0.10, -1, 1.5) * 8, "; ".join(parts).capitalize(), "Revenue growth", 12))

    margin_rows = fund["quarters"].get("Operating Income") or fund["quarters"].get("Net Income") or {}
    revenue_rows = fund["quarters"].get("Total Revenue", {})
    shared = sorted(d for d in margin_rows if margin_rows[d] is not None and revenue_rows.get(d))
    if len(shared) >= 5:
        now_m = margin_rows[shared[-1]] / revenue_rows[shared[-1]]
        then_m = margin_rows[shared[-5]] / revenue_rows[shared[-5]]
        label = "Operating" if fund["quarters"].get("Operating Income") else "Net"
        factors.append(_factor((now_m - then_m) / 0.02 * 5,
                               f"{label} margin {now_m:.1%} last quarter against {then_m:.1%} a year earlier",
                               "Margin trend", 10))

    hist_move = float(np.mean([abs(p["move"]) for p in past])) if past else None
    if past:
        ups = sum(p["move"] > 0 for p in past)
        mean = float(np.mean([p["move"] for p in past]))
        beat_fell = sum(1 for p in past if (p["surprise"] or 0) > 0 and p["move"] < 0)
        text = (f"Stock rose after {ups} of the last {len(past)} reports, average reaction {mean:+.1%}, "
                f"typical size {hist_move:.1%}")
        if beat_fell:
            text += f"; it fell after {beat_fell} beats"
        factors.append(_factor(mean / max(hist_move, 1e-6) * 12, text, "Reaction history", 12))

    if hist_move and len(daily) > 21:
        run = float(daily["Close"].iloc[-1] / daily["Close"].iloc[-21] - 1)
        if abs(run) >= 2 * hist_move:
            factors.append(_factor(
                -10 if run > 0 else 10,
                f"{'Up' if run > 0 else 'Down'} {abs(run):.1%} over 20 days into the report, about "
                f"{abs(run) / hist_move:.0f}x a typical reaction: {'expectations look stretched' if run > 0 else 'a lot of bad news may be priced in'}",
                "Run-up", 10))

    factors.append(_factor(read * 0.15, f"Current market read {read:+.0f} (auction, dealer flows, Wyckoff, VPA, trend)",
                           "Market read", 15))

    score = float(np.clip(sum(f["points"] for f in factors), -100, 100))
    lean = "bullish" if score >= 20 else "bearish" if score <= -20 else "neutral"
    beat_prob = None
    if reported:
        beat_prob = (beats + BASE_BEAT_RATE * 4) / (len(reported) + 4) + float(np.clip(revision or 0, -0.1, 0.1))
        beat_prob = float(np.clip(beat_prob, 0.05, 0.95))

    out = {
        "ticker": ticker, "date": report.isoformat(), "timing": timing,
        "days": (report - date.today()).days, "score": score, "lean": lean, "beat_probability": beat_prob,
        "eps_estimate": fund["eps_estimate"].get("avg"), "eps_year_ago": fund["eps_estimate"].get("yearAgoEps"),
        "analysts": fund["eps_estimate"].get("numberOfAnalysts"),
        "revenue_estimate": fund["revenue_estimate"].get("avg"),
        "factors": factors, "history": past, "historical_move": hist_move,
        "implied_move": None, "straddle_move": None, "pricing": None, "expiry": None, "plays": [], "spot": spot,
    }
    if chain is not None and not chain.empty:
        vol = float(daily["hv20"].iloc[-1]) if "hv20" in daily and pd.notna(daily["hv20"].iloc[-1]) else None
        _options(out, chain, spot, rate, cfg, report, timing, past, vol)
    return out


# ---------------------------------------------------------------- options

def _payoff(kind: str, strike: float, spot: float, move: float) -> float:
    after = spot * (1 + move)
    return max(0.0, after - strike) if kind == "call" else max(0.0, strike - after)


def _test(legs: list[dict[str, Any]], spot: float, moves: list[float], cost: float) -> dict[str, Any]:
    """How the position would have paid at expiry after each past reaction (intrinsic value only)."""
    results = [sum(_payoff(l["type"], l["strike"], spot, m) for l in legs) * 100 - cost for m in moves]
    return {"wins": sum(r > 0 for r in results), "tests": len(results),
            "avg_result": float(np.mean(results)) if results else None}


def _short_day(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{d:%b} {d.day}"


def _atm_iv(chain: pd.DataFrame, expiry: str, spot: float) -> float | None:
    side = chain[(chain["expiration"] == expiry) & (chain["impliedVolatility"] > 0.03)
                 & (chain["impliedVolatility"] < 5)]
    if side.empty:
        return None
    nearest = side.assign(gap=(side["strike"] - spot).abs()).nsmallest(4, "gap")  # two strikes, both types
    return float(nearest["impliedVolatility"].median())


def _event_move(chain: pd.DataFrame, spot: float, expiry: str, reaction_day: date,
                realised_vol: float | None) -> tuple[float | None, str]:
    """Expected absolute move priced for the report alone, and how it was separated.

    The expiry after the report carries the event plus ordinary days. With an expiry before the
    report, its IV stands for an ordinary day (the term-structure method):
        event variance = IV_after^2 x T_after - IV_before^2 x (T_after - one day)
    Otherwise 20-day realised volatility stands in for the ordinary days. The expected absolute
    move of a normal distribution is about 0.8 standard deviations.
    """
    days_after = int(np.busday_count(date.today(), date.fromisoformat(expiry))) + 1
    iv_after = _atm_iv(chain, expiry, spot)
    if not iv_after:
        return None, "no implied volatility"
    before = sorted(e for e in chain["expiration"].unique()
                    if date.fromisoformat(e) < reaction_day and np.busday_count(date.today(), date.fromisoformat(e)) >= 2)
    iv_before = _atm_iv(chain, before[-1], spot) if before else None
    if iv_before:
        base, method = iv_before, f"the {_short_day(before[-1])} expiry's {iv_before:.0%} IV for ordinary days"
    elif realised_vol:
        base, method = realised_vol, f"20-day realised volatility ({realised_vol:.0%}) for ordinary days"
    else:
        return None, "nothing to compare against"
    variance = iv_after**2 * days_after / 252 - base**2 * (days_after - 1) / 252
    if variance <= 0:
        return None, method  # the expiry shows no extra volatility for the report
    return float(0.8 * np.sqrt(variance)), method


def _options(out: dict[str, Any], chain: pd.DataFrame, spot: float, rate: float, cfg: dict[str, Any],
             report: date, timing: str, past: list[dict[str, Any]], realised_vol: float | None) -> None:
    reaction_day = report + timedelta(days=1 if timing == "after close" else 0)
    expiries = sorted(e for e in chain["expiration"].unique() if date.fromisoformat(e) >= reaction_day)
    if not expiries:
        return
    expiry = expiries[0]
    if (date.fromisoformat(expiry) - reaction_day).days > 10:
        return  # no expiry close enough after the report to isolate the event
    ex = chain[(chain["expiration"] == expiry) & (chain["bid"] > 0) & (chain["ask"] > 0)].copy()
    if ex.empty:
        return
    ex["mid"] = (ex["bid"] + ex["ask"]) / 2
    ex["spread"] = (ex["ask"] - ex["bid"]) / ex["mid"]
    strike = float(ex.iloc[(ex["strike"] - spot).abs().argsort()].iloc[0]["strike"])
    atm = ex[ex["strike"] == strike]
    if set(atm["type"]) != {"call", "put"}:
        return
    straddle = float(atm["mid"].sum()) / spot
    implied, method = _event_move(chain, spot, expiry, reaction_day, realised_vol)
    out.update(implied_move=implied, straddle_move=straddle, expiry=expiry)
    hist = out["historical_move"]
    if hist and implied:
        ratio = implied / hist
        out["pricing"] = {
            "ratio": ratio,
            "verdict": "cheap" if ratio < 0.85 else "rich" if ratio > 1.2 else "fair",
            "text": (f"Options price about a {implied:.1%} move for the report itself against a typical "
                     f"{hist:.1%} for this stock ({ratio:.2f}x). The {_short_day(expiry)} straddle costs "
                     f"{straddle:.1%} of the price, which also covers ordinary movement on the other days; "
                     f"the report's share was separated using {method}"),
        }

    budget = cfg["account_size"] * cfg["risk_per_trade_pct"] / 100
    usable = ex[(ex["openInterest"] >= cfg["earnings_min_open_interest"]) & (ex["spread"] <= cfg["earnings_max_spread"])]
    moves = [p["move"] for p in past]
    T = max((date.fromisoformat(expiry) - date.today()).days, 0.5) / 365
    plays = []

    def describe(row) -> dict[str, Any]:
        d = float(greeks.greeks(spot, row["strike"], T, rate, max(row["impliedVolatility"], 0.05), 0.0,
                                row["type"] == "call")["delta"])
        return {"type": row["type"], "strike": float(row["strike"]), "bid": float(row["bid"]),
                "ask": float(row["ask"]), "delta": d, "symbol": row["contractSymbol"]}

    # Directional OTM: strikes from a quarter to one-and-a-quarter expected moves away (the report's
    # implied move, or the whole straddle when the report's share could not be separated).
    reach = implied or straddle
    sides = {"bullish": ["call"], "bearish": ["put"], "neutral": ["call", "put"]}[out["lean"]]
    for side in sides:
        sign = 1 if side == "call" else -1
        pool = usable[(usable["type"] == side)
                      & (sign * (usable["strike"] - spot) >= 0.25 * reach * spot)
                      & (sign * (usable["strike"] - spot) <= 1.25 * reach * spot)
                      & (usable["ask"] * 100 <= budget)]
        for _, row in pool.iterrows():
            leg = describe(row)
            cost = row["ask"] * 100
            breakeven = row["strike"] + sign * row["ask"]
            plays.append({
                "kind": "OTM", "legs": [leg], "cost": float(cost),
                "breakeven_move": float(breakeven / spot - 1),
                **_test([leg], spot, moves, cost),
            })

    # Strangle when there is no lean but options are cheap against history.
    if out["lean"] == "neutral" and out["pricing"] and out["pricing"]["verdict"] == "cheap":
        calls = usable[(usable["type"] == "call") & (usable["strike"] > spot)].sort_values("strike")
        puts = usable[(usable["type"] == "put") & (usable["strike"] < spot)].sort_values("strike", ascending=False)
        for (_, c), (_, p) in zip(calls.head(3).iterrows(), puts.head(3).iterrows()):
            cost = (c["ask"] + p["ask"]) * 100
            if cost > budget:
                continue
            legs = [describe(c), describe(p)]
            plays.append({
                "kind": "Strangle", "legs": legs, "cost": float(cost),
                "breakeven_move": float((c["strike"] + c["ask"] + p["ask"]) / spot - 1),
                **_test(legs, spot, moves, cost),
            })

    ranked = sorted(plays, key=lambda p: (p["avg_result"] if p["avg_result"] is not None else -1e9), reverse=True)
    out["plays"] = ranked[: cfg["earnings_max_plays"]]
