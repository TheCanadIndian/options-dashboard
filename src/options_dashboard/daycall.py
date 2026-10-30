"""The day's direction call for SPX and SPY, made at the open.

A logistic regression on what is known when the market opens: the overnight gap, yesterday's move
and where it closed in its range, the 5-day move, distance from the 20- and 50-day averages (in
daily ranges), VIX1D against VIX and VIX's overnight change, Monday, Friday and the turn of the
month. Target: does the index close above its open? Retrained every call on all days since VIX1D
began (April 2023).

Walk-forward backtest (Dec 2024 - Oct 2026): on all days it is a coin flip (51-52%). On confident
SPX days (probability 60%+ either way, about one day in five) it was right 54% of the time, and a
$150 0DTE contract bought at the open made about +$23 a trade at a 2x target; that edge vanished if
real options cost 15% more than modelled or if spreads at the open were wide, and buying later
(10:00-10:30) lost money. SPY showed no edge. So the call is made at the open and scored daily at
real prices (zerodte.py) before it should be trusted.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from . import data

FEATURES = ["gap", "prev_ret", "prev_loc", "ret5", "dist20", "dist50", "vix_ratio", "vix_chg", "monday", "friday",
            "month_turn"]
CONFIDENT = 0.10  # |probability - 0.5| at or above this is a confident call
SOURCES = {"SPX": "^GSPC", "SPY": "SPY"}


def _daily(sym: str) -> pd.DataFrame:
    d = data.call(yf.Ticker(sym).history, period="5y", auto_adjust=False)
    d.index = pd.to_datetime(d.index.tz_convert("America/New_York").date)
    return d


def _row(d: pd.DataFrame, i: int, opening: float, v1_open: float, v_open: float, v_prev: float) -> dict[str, float]:
    """Features for day i of `d` (only days before i, plus today's opens)."""
    prev, pc = d.iloc[i - 1], float(d["Close"].iloc[i - 1])
    close = d["Close"]
    span = float((d["High"] - d["Low"]).iloc[i - 14:i].mean())
    day = d.index[i] if i < len(d) else pd.Timestamp(date.today())
    return {
        "gap": opening / pc - 1,
        "prev_ret": pc / float(close.iloc[i - 2]) - 1,
        "prev_loc": (pc - prev["Low"]) / (prev["High"] - prev["Low"]) if prev["High"] > prev["Low"] else 0.5,
        "ret5": pc / float(close.iloc[i - 6]) - 1,
        "dist20": (pc - float(close.iloc[i - 20:i].mean())) / span,
        "dist50": (pc - float(close.iloc[i - 50:i].mean())) / span,
        "vix_ratio": v1_open / v_open,
        "vix_chg": v_open / v_prev - 1,
        "monday": float(day.weekday() == 0), "friday": float(day.weekday() == 4),
        "month_turn": float(day.day <= 3 or day.day >= 27),
    }


def _fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0, steps: int = 3000, lr: float = 0.1) -> np.ndarray:
    w = np.zeros(X.shape[1] + 1)
    Xb = np.c_[np.ones(len(X)), X]
    for _ in range(steps):
        p = 1 / (1 + np.exp(-Xb @ w))
        w -= lr * (Xb.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]] / len(y))
    return w


def _opens_today() -> tuple[dict[str, float], float, float]:
    """Today's opening prints for each index and for VIX1D and VIX, from 5-minute bars."""
    def first_open(sym: str) -> float:
        h = data.call(yf.Ticker(sym).history, period="1d", interval="5m")
        h.index = h.index.tz_convert("America/New_York")
        h = h[h.index.date == datetime.now(data.NEW_YORK).date()]
        if h.empty:
            raise ValueError(f"{sym}: no bars yet today")
        return float(h["Open"].iloc[0])

    return {k: first_open(s) for k, s in SOURCES.items()}, first_open("^VIX1D"), first_open("^VIX")


def call(asof: date | None = None) -> dict[str, Any]:
    """Today's call for each index: probability of closing above the open, and whether it is confident.
    `asof` replays a past day (its opens taken from daily bars), for testing."""
    v1 = _daily("^VIX1D")
    v = _daily("^VIX")
    for x in (v1, v):
        x["Open"] = x["Open"].where(x["Open"] > 0, x["Close"])
    if asof is None:
        opens, v1_today, v_today = _opens_today()
        today = pd.Timestamp(datetime.now(data.NEW_YORK).date())
    else:
        today = pd.Timestamp(asof)
        opens = {k: float(_daily(s_).loc[today, "Open"]) for k, s_ in SOURCES.items()}
        v1_today, v_today = float(v1.loc[today, "Open"]), float(v.loc[today, "Open"])
    out = {}
    for name, sym in SOURCES.items():
        d = _daily(sym)
        d = d[d.index < today]
        rows, labels = [], []
        for i in range(60, len(d)):
            day = d.index[i]
            if day not in v1.index or day not in v.index:
                continue
            j = v.index.get_loc(day)
            rows.append(_row(d, i, float(d["Open"].iloc[i]), float(v1.loc[day, "Open"]), float(v.loc[day, "Open"]),
                             float(v["Close"].iloc[j - 1])))
            labels.append(float(d["Close"].iloc[i] > d["Open"].iloc[i]))
        X = pd.DataFrame(rows)[FEATURES]
        mu, sd = X.mean(), X.std().replace(0, 1)
        w = _fit(((X - mu) / sd).to_numpy(), np.array(labels))
        today_row = _row(pd.concat([d, pd.DataFrame(index=[today], columns=d.columns)]), len(d), opens[name],
                         v1_today, v_today, float(v["Close"][v.index < today].iloc[-1]))
        x = ((pd.DataFrame([today_row])[FEATURES] - mu) / sd).to_numpy()
        p = float(1 / (1 + np.exp(-(np.r_[1.0, x[0]] @ w))))
        calendar = {"monday", "friday", "month_turn"}
        drivers = sorted(((f, float(w[k + 1] * x[0][k])) for k, f in enumerate(FEATURES)
                          if not (f in calendar and today_row[f] == 0)), key=lambda t: -abs(t[1]))[:3]
        out[name] = {"p_up": p, "direction": "up" if p >= 0.5 else "down", "confident": abs(p - 0.5) >= CONFIDENT,
                     "open": opens[name], "drivers": drivers, "trained_days": len(labels), "vix1d": v1_today}
    return out


def estimate_strike(c: dict[str, Any], ticker: str, cap: float) -> float | None:
    """Rough strike a capped 0DTE contract buys at the open (Black-Scholes on VIX1D with a skew), for
    the alert sent before real quotes are in; the lab buys the real contract a few minutes later."""
    from math import erf, log, sqrt

    def ncdf(z: float) -> float:
        return 0.5 * (1 + erf(z / sqrt(2)))

    S, iv, call_ = c["open"], c["vix1d"] / 100, c["direction"] == "up"
    step = 5.0 if ticker == "SPX" else 1.0
    T = 390 / (252 * 390)
    k = (np.ceil(S / step) * step) if call_ else (np.floor(S / step) * step)
    for _ in range(300):
        x = log(k / S) / (iv * sqrt(T))
        vol = iv * (1 - 0.10 * x) if x < 0 else iv * (1 - 0.03 * x)
        sd = max(vol, iv * 0.6) * sqrt(T)
        d1 = (log(S / k) + 0.5 * sd * sd) / sd
        price = S * ncdf(d1) - k * ncdf(d1 - sd) if call_ else k * ncdf(sd - d1) - S * ncdf(-d1)
        if price <= cap:
            return float(k)
        k = k + step if call_ else k - step
    return None


DRIVER_TEXT = {"gap": "the overnight gap", "prev_ret": "yesterday's move", "prev_loc": "where yesterday closed in its range",
               "ret5": "the 5-day move", "dist20": "distance from the 20-day average", "dist50": "distance from the 50-day average",
               "vix_ratio": "VIX1D against VIX", "vix_chg": "VIX's overnight change", "monday": "Monday",
               "friday": "Friday", "month_turn": "the turn of the month"}


def describe(c: dict[str, Any]) -> str:
    lean = ", ".join(f"{DRIVER_TEXT[f]} ({'up' if v > 0 else 'down'})" for f, v in c["drivers"] if abs(v) > 0.02)
    return f"{c['p_up']:.0%} chance of closing above the open" + (f"; leaning on {lean}" if lean else "")
