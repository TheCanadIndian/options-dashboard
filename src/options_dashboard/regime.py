"""Market regime: risk-on or risk-off, which way it is moving, early warnings of a change, and
which sectors have done best in regimes like the current one.

Seven daily components, each a z-score against its own trailing year (capped at +/-2.5) and
signed so positive means risk-on:

    trend          SPY against its 200-day average, and the 50-day average's slope
    volatility     VIX against VIX3M (an inverted curve is stress) and the VIX's level
    credit         high yield (HYG) against Treasuries (IEF), 20-day change
    appetite       consumer discretionary against staples, small caps against SPY, 20-day change
    defensives     utilities, staples and health care against tech, discretionary, financials
                   and industrials, 20-day change (inverted)
    safe havens    gold and long Treasuries against SPY, 20-day change (inverted)
    breadth        live only: share of the scan universe above its 50-day average

The risk score is their average, scaled to -100..+100. Its level and 10-day change give one of
four states (like the sector rotation graph): risk-on strengthening, risk-on fading, risk-off
deepening, risk-off easing.

The same score is rebuilt for every day since 2008, so each state carries its historical record:
how often SPY fell 5% or more within the next month, SPY's average return over it, and how each
sector did against SPY. Those statistics describe what followed similar readings in the past;
they are not a forecast with a guarantee.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from . import data, sectors
from .config import STATE_DIR

DIR = STATE_DIR / "regime"
TICKERS = ["SPY", "IWM", "HYG", "IEF", "XLY", "XLP", "XLU", "XLV", "XLK", "XLF", "XLI", "GLD", "TLT", "^VIX", "^VIX3M",
           *sectors.SECTOR_ETFS.values()]
COMPONENTS = {
    "trend": "Trend (SPY vs its 200-day average, 50-day slope)",
    "volatility": "Volatility (VIX level and VIX/VIX3M curve)",
    "credit": "Credit (high yield vs Treasuries)",
    "appetite": "Risk appetite (discretionary vs staples, small caps vs SPY)",
    "defensives": "Defensive rotation (defensive vs cyclical sectors)",
    "havens": "Safe havens (gold and Treasuries vs SPY)",
    "breadth": "Breadth (share of stocks above their 50-day average)",
}
STATES = {
    ("on", "up"): "risk-on strengthening", ("on", "down"): "risk-on fading",
    ("off", "down"): "risk-off deepening", ("off", "up"): "risk-off easing",
}
HORIZON = 21  # trading days ahead for the historical statistics
DRAWDOWN = 0.05
MIN_EDGE = 0.002  # a sector must have beaten (or trailed) SPY by 0.2% a month in a state to count


# ---------------------------------------------------------------- data

def _closes() -> pd.DataFrame:
    """Daily closes since 2006 for every input, refreshed weekly and topped up with recent prices."""
    path = DIR / "closes.pkl"
    week = date.today().strftime("%G-W%V")
    frame = None
    try:
        cached = pd.read_pickle(path)
        if cached["week"] == week:
            frame = cached["frame"]
    except Exception:
        pass
    if frame is None:
        cols = {}
        for t in TICKERS:
            h = data.call(yf.Ticker(t).history, period="20y", interval="1d", auto_adjust=True, raise_errors=True)
            cols[t] = h["Close"].tz_localize(None) if h.index.tz is not None else h["Close"]
        frame = pd.DataFrame(cols)
        frame.index = frame.index.normalize()
        DIR.mkdir(parents=True, exist_ok=True)
        pd.to_pickle({"week": week, "frame": frame}, path)
    # Recent closes (cached hourly) take precedence, so today's reading uses today's prices.
    for t in TICKERS:
        try:
            recent = data.history(t)["Close"]
        except Exception:
            continue
        recent.index = (recent.index.tz_localize(None) if recent.index.tz is not None else recent.index).normalize()
        frame = recent.to_frame(t).combine_first(frame)
    return frame.sort_index().ffill()


def _z(series: pd.Series, window: int = 252) -> pd.Series:
    mean = series.rolling(window, min_periods=120).mean()
    sd = series.rolling(window, min_periods=120).std()
    return ((series - mean) / sd).clip(-2.5, 2.5)


def components(px: pd.DataFrame) -> pd.DataFrame:
    """Each component's daily z-score, positive = risk-on."""
    out = pd.DataFrame(index=px.index)
    spy = px["SPY"]
    out["trend"] = (_z(spy / spy.rolling(200).mean() - 1) + _z(spy.rolling(50).mean().pct_change(20))) / 2
    curve = px["^VIX"] / px["^VIX3M"]
    out["volatility"] = (-_z(curve) - _z(px["^VIX"])) / 2
    out["credit"] = _z((px["HYG"] / px["IEF"]).pct_change(20))
    out["appetite"] = (_z((px["XLY"] / px["XLP"]).pct_change(20)) + _z((px["IWM"] / spy).pct_change(20))) / 2
    defensive = px[["XLU", "XLP", "XLV"]].pct_change(20).mean(axis=1)
    cyclical = px[["XLK", "XLY", "XLF", "XLI"]].pct_change(20).mean(axis=1)
    out["defensives"] = -_z(defensive - cyclical)
    havens = px[["GLD", "TLT"]].pct_change(20).mean(axis=1)
    out["havens"] = -_z(havens - spy.pct_change(20))
    return out


def _state(score: float, change: float) -> str:
    return STATES[("on" if score >= 0 else "off", "up" if change >= 0 else "down")]


# ---------------------------------------------------------------- history

def history_stats(px: pd.DataFrame, comps: pd.DataFrame) -> dict[str, Any]:
    """What followed each state since 2008: SPY's drawdown odds and return, and each sector against SPY."""
    score = comps.drop(columns=["breadth"], errors="ignore").mean(axis=1) * 40
    change = score - score.shift(10)
    frame = pd.DataFrame({"score": score, "change": change}).dropna()
    frame = frame[frame.index >= "2008-01-01"]
    frame["state"] = [_state(s, c) for s, c in zip(frame["score"], frame["change"])]
    spy = px["SPY"]
    future_min = spy[::-1].rolling(HORIZON, min_periods=1).min()[::-1].shift(-1)
    frame["drop"] = (future_min / spy - 1).reindex(frame.index) <= -DRAWDOWN
    frame["spy_fwd"] = (spy.shift(-HORIZON) / spy - 1).reindex(frame.index)
    frame = frame.dropna(subset=["spy_fwd"])
    base_drop = float(frame["drop"].mean())
    stats = {"since": str(frame.index[0].date()), "days": len(frame), "base_drop": base_drop,
             "base_return": float(frame["spy_fwd"].mean()), "states": {}, "signals": {}}

    # How each warning has done in the past (breadth has no history, so it is not tested).
    curve = (px["^VIX"] / px["^VIX3M"]).reindex(frame.index)
    spy20 = (spy / spy.shift(20) - 1).reindex(frame.index)
    credit = px["HYG"] / px["IEF"]
    credit20 = (credit / credit.shift(20) - 1).reindex(frame.index)
    crossed = np.sign(frame["score"]) != np.sign(frame["score"].shift(5))
    tests = {
        "vix_inverted": curve >= 1.0,
        "fading_fast": (frame["score"] >= 0) & (frame["change"] <= -25),
        "easing_fast": (frame["score"] < 0) & (frame["change"] >= 25),
        "credit_divergence": (spy20 > 0.01) & (credit20 < -0.005),
        "crossed_below": crossed & (frame["score"] < 0),
        "crossed_above": crossed & (frame["score"] >= 0),
    }
    for name, mask in tests.items():
        rows = frame[mask.fillna(False)]
        if len(rows) >= 20:
            stats["signals"][name] = {"days": len(rows), "drop_odds": float(rows["drop"].mean()),
                                      "spy_return": float(rows["spy_fwd"].mean()),
                                      "spy_up": float((rows["spy_fwd"] > 0).mean())}
    etf_fwd = {name: (px[etf].shift(-HORIZON) / px[etf] - 1).reindex(frame.index) - frame["spy_fwd"]
               for name, etf in sectors.SECTOR_ETFS.items() if etf in px}
    for state, rows in frame.groupby("state"):
        idx = rows.index
        rel = {name: float(s.loc[idx].mean()) for name, s in etf_fwd.items() if s.loc[idx].notna().sum() > 100}
        nxt = frame["state"].shift(-HORIZON).loc[idx]
        stats["states"][state] = {
            "days": len(rows), "drop_odds": float(rows["drop"].mean()), "spy_return": float(rows["spy_fwd"].mean()),
            "spy_up": float((rows["spy_fwd"] > 0).mean()),
            "next_state": {k: float(v) for k, v in nxt.value_counts(normalize=True).items()},
            "sectors": dict(sorted(rel.items(), key=lambda kv: -kv[1])),
        }
    return stats


# ---------------------------------------------------------------- reading

def read(breadth: float | None = None, breadth_change: float | None = None) -> dict[str, Any]:
    """Today's regime: score, state, components, warnings, historical odds and favoured sectors."""
    px = _closes()
    comps = components(px)
    hist = history_stats(px, comps)
    latest = comps.iloc[-1].to_dict()
    parts = {k: float(v) for k, v in latest.items() if pd.notna(v)}
    # The score uses only components with history, so it matches the record behind the odds.
    # Breadth (no free history) is shown and feeds the warnings.
    base = comps.mean(axis=1) * 40
    score = float(base.iloc[-1])
    change = score - float(base.iloc[-11]) if len(base) > 11 else 0.0
    if breadth is not None:
        parts["breadth"] = float(np.clip((breadth - 0.5) / 0.15, -2.5, 2.5))
    state = _state(score, change)
    spy = px["SPY"]

    warnings = []

    def record(name: str) -> str:
        s = hist["signals"].get(name)
        if not s:
            return ""
        return (f" Since {hist['since'][:4]} ({s['days']} days like this) SPY fell {DRAWDOWN:.0%}+ within a month "
                f"{s['drop_odds']:.0%} of the time (usually {hist['base_drop']:.0%}), averaging {s['spy_return']:+.1%}.")

    crossed = np.sign(float(base.iloc[-6])) != np.sign(score) if len(base) > 6 else False
    if crossed:
        warnings.append(f"The risk score crossed {'above' if score >= 0 else 'below'} zero in the last week."
                        + record("crossed_above" if score >= 0 else "crossed_below"))
    if score >= 0 and change <= -25:
        warnings.append(f"Risk-on is fading fast: the score fell {abs(change):.0f} points in 10 sessions." + record("fading_fast"))
    if score < 0 and change >= 25:
        warnings.append(f"Risk-off is easing fast: the score rose {change:.0f} points in 10 sessions." + record("easing_fast"))
    curve = float(px["^VIX"].iloc[-1] / px["^VIX3M"].iloc[-1])
    if curve >= 1.0:
        warnings.append(f"VIX is above VIX3M ({curve:.2f}), an inverted volatility curve." + record("vix_inverted"))
    spy20 = float(spy.iloc[-1] / spy.iloc[-21] - 1)
    credit20 = float((px["HYG"] / px["IEF"]).iloc[-1] / (px["HYG"] / px["IEF"]).iloc[-21] - 1)
    if spy20 > 0.01 and credit20 < -0.005:
        warnings.append(f"Credit is not confirming: SPY is up {spy20:.1%} in a month while high yield lags Treasuries "
                        f"by {abs(credit20):.1%}." + record("credit_divergence"))
    near_high = float(spy.iloc[-1] / spy.iloc[-252:].max()) >= 0.97
    if breadth is not None and near_high and (breadth < 0.5 or (breadth_change or 0) <= -0.10):
        warnings.append(f"Narrowing breadth near highs: SPY is within 3% of its 52-week high but only {breadth:.0%} "
                        "of stocks are above their 50-day average")

    odds = hist["states"].get(state, {})
    rot = sectors.rotation()
    favoured, avoid = [], []
    for name, edge in odds.get("sectors", {}).items():
        quad = rot.get(name, {}).get("quadrant")
        if edge >= MIN_EDGE and quad in ("leading", "improving"):
            favoured.append({"sector": name, "etf": sectors.SECTOR_ETFS[name], "edge": edge, "quadrant": quad})
        elif edge <= -MIN_EDGE and quad in ("lagging", "weakening"):
            avoid.append({"sector": name, "etf": sectors.SECTOR_ETFS[name], "edge": edge, "quadrant": quad})
    avoid.sort(key=lambda s: s["edge"])

    path = base.iloc[-252:]
    return {
        "score": score, "change": change, "state": state, "components": parts, "warnings": warnings,
        "odds": odds, "base_drop": hist["base_drop"], "base_return": hist["base_return"], "since": hist["since"],
        "signals": hist["signals"], "all_states": {k: {kk: vv for kk, vv in v.items() if kk != "sectors"}
                                                   for k, v in hist["states"].items()},
        "history_days": hist["days"], "favoured": favoured[:4], "avoid": avoid[:4],
        "breadth": breadth, "breadth_change": breadth_change,
        "path": [[str(d.date()), float(v)] for d, v in path.items() if pd.notna(v)],
    }


def last_breadth() -> tuple[float | None, float | None]:
    """The most recent stored breadth reading and its five-session change."""
    try:
        stored = json.loads((DIR / "breadth.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    values = [v for _, v in sorted(stored.items())]
    if not values:
        return None, None
    return values[-1], (values[-1] - values[-6]) if len(values) >= 6 else None


def breadth_now(tickers: dict[str, Any]) -> tuple[float | None, float | None]:
    """Share of scanned stocks above their 50-day average, and its change over five sessions."""
    flags = []
    for info in tickers.values():
        ind = info.get("indicators")
        if ind is not None and len(ind) and pd.notna(ind["ema50"].iloc[-1]):
            flags.append(float(ind["Close"].iloc[-1]) > float(ind["ema50"].iloc[-1]))
    if len(flags) < 30:
        return None, None
    value = float(np.mean(flags))
    path = DIR / "breadth.json"
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        stored = {}
    stored[date.today().isoformat()] = value
    stored = dict(sorted(stored.items())[-60:])
    DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stored), encoding="utf-8")
    earlier = [v for d, v in sorted(stored.items()) if d < date.today().isoformat()]
    return value, (value - earlier[-5]) if len(earlier) >= 5 else None
