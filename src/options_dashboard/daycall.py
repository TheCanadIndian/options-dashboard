"""The day's direction call, made at the open: will the market close above where it opened?

Measured on SPY, whose open is a real traded price. The S&P 500 index's official 9:30 "open" is
computed before many of its stocks have traded, so it captures only about 75% of the overnight move
and then catches up; a model trained on the index's open learns that catch-up, which nobody can
trade (SPX options price off the true level from the first second). The call applies to both SPY
and SPX contracts.

Model: a blend of a regularised logistic regression and gradient-boosted trees on what is known at
the open (the gap, yesterday's and recent moves, distance from the 20/50/200-day averages, 2-day
RSI, VIX and its overnight change, VIX9D against VIX, VIX against VIX3M, yesterday's moves in
high-yield credit, Treasuries, the 10-year yield, the dollar, small caps, the Nasdaq, equal weight
against cap weight and tech against utilities, and the calendar), trained on every day since 2011.

Honest record (yearly walk-forward 2016-2026): right 53.5% of days, 54.9% on confident days, with
an average move in the called direction of about +3 basis points, far less than an out-of-the-money
0DTE option costs. Confident calls were right only 49% of the time in 2025 and 47% in 2026. Each
call carries the model's last-12-months record so its reliability is never overstated; the lab
(zerodte.py) scores every call at real prices.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from . import data

CONFIDENT = 0.10  # |probability - 0.5| at or above this is a confident call
SYMS = ["SPY", "^VIX", "^VIX9D", "^VIX3M", "HYG", "TLT", "^TNX", "DX-Y.NYB", "IWM", "QQQ", "RSP", "XLK", "XLU"]
OPEN_NOW = ["SPY", "^VIX", "^VIX9D", "^VIX1D", "^GSPC"]  # today's opening prints that feed the call or the strike


def _daily(sym: str) -> pd.DataFrame:
    d = data.call(yf.Ticker(sym).history, period="max", auto_adjust=False)
    d.index = pd.to_datetime(d.index.tz_convert("America/New_York").date)
    d["Open"] = d["Open"].where(d["Open"] > 0, d["Close"])  # a few index opens are recorded as zero
    return d[d.index >= "2007-01-01"]


def _features(px: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per day: everything known at that day's open, plus the label (SPY closed above its open)."""
    s = px["SPY"]
    c, o, h, l = s["Close"], s["Open"], s["High"], s["Low"]
    pc, rng = c.shift(1), h - l
    atr = rng.rolling(14).mean().shift(1)
    f = pd.DataFrame(index=s.index)
    f["gap"] = o / pc - 1
    f["gap_atr"] = (o - pc) / atr
    f["prev_ret"] = pc / c.shift(2) - 1
    f["prev2_ret"] = c.shift(2) / c.shift(3) - 1
    f["prev_loc"] = ((pc - l.shift(1)) / rng.shift(1)).clip(0, 1)
    f["prev_range"] = rng.shift(1) / atr
    f["prev_body"] = (pc - o.shift(1)) / rng.shift(1)
    for n in (5, 10, 20):
        f[f"ret{n}"] = pc / c.shift(n + 1) - 1
    for n in (20, 50, 200):
        f[f"dist{n}"] = (pc - c.rolling(n).mean().shift(1)) / atr
    chg = c.diff()
    up, dn = chg.clip(lower=0).ewm(alpha=1 / 2).mean(), (-chg.clip(upper=0)).ewm(alpha=1 / 2).mean()
    f["rsi2"] = (100 - 100 / (1 + up / dn.replace(0, np.nan))).shift(1)
    vix, v9, v3 = px["^VIX"], px["^VIX9D"], px["^VIX3M"]
    f["vix"] = vix["Open"]
    f["vix_chg"] = vix["Open"] / vix["Close"].shift(1) - 1
    f["vix9_ratio"] = v9["Open"] / vix["Open"]
    f["vix_term"] = (vix["Close"] / v3["Close"]).shift(1)
    f["vix_ret5"] = (vix["Close"] / vix["Close"].shift(5) - 1).shift(1)

    def ret(sym: str, n: int = 1) -> pd.Series:
        x = px[sym]["Close"]
        return (x / x.shift(n) - 1).shift(1)

    f["hyg"], f["hyg5"], f["tlt"] = ret("HYG"), ret("HYG", 5), ret("TLT")
    f["tnx"] = px["^TNX"]["Close"].diff().shift(1)
    f["dxy"] = ret("DX-Y.NYB")
    f["iwm_spy"] = ret("IWM") - ret("SPY")
    f["qqq_spy"] = ret("QQQ") - ret("SPY")
    f["rsp_spy5"] = ret("RSP", 5) - ret("SPY", 5)
    f["xlk_xlu5"] = ret("XLK", 5) - ret("XLU", 5)
    idx = f.index
    for k, name in enumerate(("mon", "tue", "wed", "thu", "fri")):
        f[name] = (idx.weekday == k).astype(float)
    f["month_turn"] = ((idx.day <= 3) | (idx.day >= 27)).astype(float)
    f["opex"] = ((idx.weekday == 4) & (idx.day >= 15) & (idx.day <= 21)).astype(float)
    f["y"] = (c > o).astype(float)
    return f


def _fit(train: pd.DataFrame, cols: list[str]):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(train[cols])
    X = sc.transform(train[cols])
    lin = LogisticRegression(C=0.05, max_iter=2000).fit(X, train["y"])
    trees = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=250, min_samples_leaf=80,
                                           l2_regularization=1.0, random_state=0).fit(X, train["y"])

    def predict(rows: pd.DataFrame) -> np.ndarray:
        Z = sc.transform(rows[cols])
        return (lin.predict_proba(Z)[:, 1] + trees.predict_proba(Z)[:, 1]) / 2

    return predict, lin, sc


def _opens_today() -> dict[str, float]:
    out = {}
    for sym in OPEN_NOW:
        h = data.call(yf.Ticker(sym).history, period="1d", interval="5m")
        h.index = h.index.tz_convert("America/New_York")
        h = h[h.index.date == datetime.now(data.NEW_YORK).date()]
        if h.empty:
            raise ValueError(f"{sym}: no bars yet today")
        out[sym] = float(h["Open"].iloc[0])
    return out


def call(asof: date | None = None) -> dict[str, Any]:
    """Today's call: the chance SPY closes above its open, applied to SPY and SPX contracts.
    `asof` replays a past day (its opens taken from daily bars), for testing."""
    px = {s: _daily(s) for s in SYMS}
    today = pd.Timestamp(asof or datetime.now(data.NEW_YORK).date())
    if asof is None:
        opens = _opens_today()
    else:
        opens = {s: float(_daily(s).loc[today, "Open"]) for s in OPEN_NOW}
    for sym in ("SPY", "^VIX", "^VIX9D"):  # today's row: only its open is known
        d = px[sym][px[sym].index < today]
        row = pd.DataFrame({"Open": [opens[sym]], "High": [np.nan], "Low": [np.nan], "Close": [np.nan]}, index=[today])
        px[sym] = pd.concat([d, row])
    for sym in SYMS:
        if sym not in ("SPY", "^VIX", "^VIX9D"):
            px[sym] = px[sym][px[sym].index < today].reindex(px["SPY"].index, method="ffill")
    f = _features(px)
    cols = [c for c in f.columns if c != "y"]
    history = f[f.index < today].dropna()
    today_row = f.loc[[today], cols]
    if today_row.isna().any(axis=None):
        today_row = today_row.fillna(history[cols].iloc[-1])
    predict, lin, sc = _fit(history, cols)
    p = float(predict(today_row)[0])

    # The model's own recent record: trained on everything before the last 12 months, judged on them.
    cut = today - pd.DateOffset(months=12)
    old, recent = history[history.index < cut], history[history.index >= cut]
    rec_predict, _, _ = _fit(old, cols)
    rp = rec_predict(recent)
    confident = np.abs(rp - 0.5) >= CONFIDENT
    record = {"days": int(confident.sum()),
              "right": float(((rp[confident] > 0.5) == (recent["y"].to_numpy()[confident] == 1)).mean())
              if confident.any() else None}

    z = sc.transform(today_row[cols])[0]
    calendar = {"mon", "tue", "wed", "thu", "fri", "month_turn", "opex"}
    contrib = [(c_, float(lin.coef_[0][k] * z[k])) for k, c_ in enumerate(cols)
               if not (c_ in calendar and today_row[c_].iloc[0] == 0)]
    drivers = sorted(contrib, key=lambda t: -abs(t[1]))[:3]
    base = {"p_up": p, "direction": "up" if p >= 0.5 else "down", "confident": abs(p - 0.5) >= CONFIDENT,
            "drivers": drivers, "trained_days": len(history), "record": record, "vix1d": opens.get("^VIX1D")}
    return {"SPY": {**base, "open": opens["SPY"]}, "SPX": {**base, "open": opens["^GSPC"]}}


DRIVER_TEXT = {"gap": "the overnight gap", "gap_atr": "the gap's size", "prev_ret": "yesterday's move",
               "prev2_ret": "the day before's move", "prev_loc": "where yesterday closed in its range",
               "prev_range": "yesterday's range", "prev_body": "yesterday's open-to-close", "ret5": "the 5-day move",
               "ret10": "the 10-day move", "ret20": "the 20-day move", "dist20": "distance from the 20-day average",
               "dist50": "distance from the 50-day average", "dist200": "distance from the 200-day average",
               "rsi2": "2-day RSI", "vix": "VIX's level", "vix_chg": "VIX's overnight change",
               "vix9_ratio": "VIX9D against VIX", "vix_term": "VIX against VIX3M", "vix_ret5": "VIX's 5-day change",
               "hyg": "high-yield credit yesterday", "hyg5": "high-yield credit this week", "tlt": "Treasuries yesterday",
               "tnx": "the 10-year yield", "dxy": "the dollar", "iwm_spy": "small caps against SPY",
               "qqq_spy": "the Nasdaq against SPY", "rsp_spy5": "breadth (equal weight against SPY)",
               "xlk_xlu5": "tech against utilities", "mon": "Monday", "tue": "Tuesday", "wed": "Wednesday",
               "thu": "Thursday", "fri": "Friday", "month_turn": "the turn of the month", "opex": "options expiration"}


def describe(c: dict[str, Any]) -> str:
    lean = ", ".join(f"{DRIVER_TEXT.get(f, f)} ({'up' if v > 0 else 'down'})" for f, v in c["drivers"] if abs(v) > 0.02)
    rec = c.get("record") or {}
    track = (f". Its confident calls over the last 12 months were right {rec['right']:.0%} of the time "
             f"({rec['days']} days)" if rec.get("right") is not None else "")
    return f"{c['p_up']:.0%} chance SPY closes above its open" + (f"; leaning on {lean}" if lean else "") + track


def estimate_strike(c: dict[str, Any], ticker: str, cap: float) -> float | None:
    """Rough strike a capped 0DTE contract buys at the open (Black-Scholes on VIX1D with a skew), for
    the alert sent before real quotes are in; the lab buys the real contract a few minutes later."""
    from math import erf, log, sqrt

    def ncdf(z: float) -> float:
        return 0.5 * (1 + erf(z / sqrt(2)))

    if not c.get("vix1d"):
        return None
    S, iv, call_ = c["open"], c["vix1d"] / 100, c["direction"] == "up"
    step = 5.0 if ticker == "SPX" else 1.0
    T = 390 / (252 * 390)
    k = (np.ceil(S / step) * step) if call_ else (np.floor(S / step) * step)
    for _ in range(300):
        x = log(k / S) / (iv * sqrt(T))
        vol = max(iv * (1 - 0.10 * x) if x < 0 else iv * (1 - 0.03 * x), iv * 0.6)
        sd = vol * sqrt(T)
        d1 = (log(S / k) + 0.5 * sd * sd) / sd
        price = S * ncdf(d1) - k * ncdf(d1 - sd) if call_ else k * ncdf(sd - d1) - S * ncdf(-d1)
        if price <= cap:
            return float(k)
        k = k + step if call_ else k - step
    return None
