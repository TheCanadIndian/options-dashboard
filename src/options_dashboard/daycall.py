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

It also uses the regime model's components and score as of the previous close, and trains only on
the last five years: markets drift, and a model trained on all history since 2011 kept learning old
lessons (its confident calls were right 45% of the time in 2025 and 41% in 2026; with a five-year
window, 61% and 63%).

Honest record (yearly walk-forward 2016-2026, five-year window): about 53% of all days right and
55.8% of confident days, with an average move in the called direction of about +3 basis points, far
less than an out-of-the-money 0DTE option costs. Each call carries the model's last-12-months record
so its reliability is never overstated; the lab (zerodte.py) scores every call at real prices.

Direction skill is small and lopsided: up calls beat the market's upward drift by about 2 points
(56% right against 54% of days closing up), and the rarer confident down calls about 10 (56% against
46%). How much the market will move is far more predictable than which way (volatility clusters):
the range forecast (quiet / normal / wide, from a ridge regression and trees on the same signals) put
quiet days at an average move of 0.32% with 42% of them under 0.2%, and wide days at 0.88% with 46%
over 0.75% (2016-2026 walk-forward). Out-of-the-money 0DTE options need movement, so quiet days stand
aside whatever the direction.

update_1030() is a second call once the first hour is in: will SPY close above its 10:30 price? It
uses only the first hour's shape (its move, where it closed in its range, its range against the
usual day, whether it filled the gap), which in a two-year monthly walk-forward was the most
informative signal found (59.8% of confident days, +5.4 basis points), on a small sample.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from . import data

CONFIDENT = 0.10  # |probability - 0.5| at or above this is a confident call
WINDOW_YEARS = 5  # train on the last five years only
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
    f["move"] = c / o - 1
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


def _regime_features(today: pd.Timestamp, index: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.Series]:
    """The regime model's components, score and 10-day change as of each previous close, aligned to
    `index`, and the values for `today` (as of the last close before it)."""
    from . import regime
    comps = regime.components(regime._closes())
    comps.index = pd.to_datetime(pd.Index(comps.index).date)
    comps["reg_score"] = comps.mean(axis=1) * 40
    comps["reg_change"] = comps["reg_score"] - comps["reg_score"].shift(10)
    comps = comps[comps.index < today].add_prefix("rg_")
    return comps.shift(1).reindex(index, method="ffill"), comps.iloc[-1]


def _range(train: pd.DataFrame, today_row: pd.DataFrame, cols: list[str]) -> tuple[float, str]:
    """Expected size of SPY's open-to-close move today, and quiet / normal / wide against the
    training period's own forecasts (bottom third, middle, top third)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    move = np.log((train["move"]).abs() + 1e-4)
    sc = StandardScaler().fit(train[cols])
    X = sc.transform(train[cols])
    ms = [Ridge(alpha=10.0).fit(X, move), HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.03, max_iter=250, min_samples_leaf=80, random_state=0).fit(X, move)]
    fit_sizes = np.exp(np.mean([m.predict(X) for m in ms], axis=0))
    today = float(np.exp(np.mean([m.predict(sc.transform(today_row[cols])) for m in ms])))
    quiet, wide = np.quantile(fit_sizes, [1 / 3, 2 / 3])
    return today, ("quiet" if today <= quiet else "wide" if today >= wide else "normal")


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
    reg_hist, reg_today = _regime_features(today, f.index)
    f = f.join(reg_hist)
    f.loc[today, reg_today.index] = reg_today.to_numpy()
    cols = [c for c in f.columns if c not in ("y", "move")]
    start = today - pd.DateOffset(years=WINDOW_YEARS)
    history = f[(f.index < today)].dropna(subset=cols + ["y", "move"])
    today_row = f.loc[[today], cols]
    if today_row.isna().any(axis=None):
        today_row = today_row.fillna(history[cols].iloc[-1])
    predict, lin, sc = _fit(history[history.index >= start], cols)
    p = float(predict(today_row)[0])
    expected, label = _range(history[history.index >= start], today_row, cols)

    # The model's own recent record: trained on the five years before the last 12 months, judged on them.
    cut = today - pd.DateOffset(months=12)
    old = history[(history.index < cut) & (history.index >= cut - pd.DateOffset(years=WINDOW_YEARS))]
    recent = history[history.index >= cut]
    rec_predict, _, _ = _fit(old, cols)
    rp = rec_predict(recent)
    confident = np.abs(rp - 0.5) >= CONFIDENT
    record = {"days": int(confident.sum()),
              "right": float(((rp[confident] > 0.5) == (recent["y"].to_numpy()[confident] == 1)).mean())
              if confident.any() else None}

    z = sc.transform(today_row[cols])[0]
    calendar = {"mon", "tue", "wed", "thu", "fri", "month_turn", "opex"}
    history = history[history.index >= start]
    contrib = [(c_, float(lin.coef_[0][k] * z[k])) for k, c_ in enumerate(cols)
               if not (c_ in calendar and today_row[c_].iloc[0] == 0)]
    drivers = sorted(contrib, key=lambda t: -abs(t[1]))[:3]
    base = {"p_up": p, "direction": "up" if p >= 0.5 else "down", "confident": abs(p - 0.5) >= CONFIDENT,
            "drivers": drivers, "trained_days": len(history), "record": record, "vix1d": opens.get("^VIX1D"),
            "expected_move": expected, "range": label}
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
               "thu": "Thursday", "fri": "Friday", "month_turn": "the turn of the month", "opex": "options expiration",
               "rg_trend": "the regime's trend", "rg_volatility": "the regime's volatility reading",
               "rg_credit": "the regime's credit reading", "rg_appetite": "risk appetite",
               "rg_defensives": "defensive rotation", "rg_havens": "safe havens", "rg_reg_score": "the risk score",
               "rg_reg_change": "the risk score's 10-day change"}


def describe(c: dict[str, Any]) -> str:
    lean = ", ".join(f"{DRIVER_TEXT.get(f, f)} ({'up' if v > 0 else 'down'})" for f, v in c["drivers"] if abs(v) > 0.02)
    rec = c.get("record") or {}
    track = (f". Its confident calls over the last 12 months were right {rec['right']:.0%} of the time "
             f"({rec['days']} days)" if rec.get("right") is not None else "")
    size = (f" Expected move {c['expected_move']:.2%} ({c['range']} day)" if c.get("expected_move") else "")
    return (f"{c['p_up']:.0%} chance SPY closes above its open" + (f"; leaning on {lean}" if lean else "") + "." + size
            + track)


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


# ---------------------------------------------------------------- the 10:30 update

FIRST_HOUR = ["h1_ret", "h1_loc", "h1_range", "gap_filled"]


def _first_hour(g: pd.DataFrame, prev_close: float, atr: float) -> dict[str, float]:
    """The first hour's shape from intraday bars covering 9:30-10:30."""
    hi, lo = float(g["High"].max()), float(g["Low"].min())
    o, c = float(g["Open"].iloc[0]), float(g["Close"].iloc[-1])
    gap = o / prev_close - 1
    return {"h1_ret": c / o - 1, "h1_loc": (c - lo) / (hi - lo) if hi > lo else 0.5, "h1_range": (hi - lo) / atr,
            "gap_filled": float(lo <= prev_close if gap > 0 else hi >= prev_close), "price": c}


def update_1030(asof: date | None = None) -> dict[str, Any]:
    """Once the first hour is in: the chance SPY closes above its 10:30 price, from the first hour's
    shape, trained on the last two years of hourly bars, with its own recent record."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    daily = _daily("SPY")
    atr_s = (daily["High"] - daily["Low"]).rolling(14).mean().shift(1)
    h = data.call(yf.Ticker("SPY").history, period="730d", interval="60m")
    h.index = h.index.tz_convert("America/New_York")
    today = asof or datetime.now(data.NEW_YORK).date()
    rows = []
    for d, g in h.groupby(h.index.date):
        ts = pd.Timestamp(d)
        if d >= today or len(g) < 7 or ts not in atr_s.index or pd.isna(atr_s.loc[ts]):
            continue
        prev = float(daily["Close"][daily.index < ts].iloc[-1])
        row = _first_hour(g.iloc[:1], prev, float(atr_s.loc[ts]))
        row["y"] = float(g["Close"].iloc[-1] > row["price"])
        row["rest_abs"] = abs(float(g["Close"].iloc[-1]) / row["price"] - 1)
        rows.append(row)
    hist = pd.DataFrame(rows)
    # How wide the first hour was against the last 20 first hours: the clearest sign of how far the
    # rest of the day will travel (narrow first hours led to afternoons about half the size of wide ones).
    hist["rel"] = hist["h1_range"] / hist["h1_range"].rolling(20).mean().shift(1)

    def fit(train: pd.DataFrame):
        sc = StandardScaler().fit(train[FIRST_HOUR])
        X = sc.transform(train[FIRST_HOUR])
        ms = [LogisticRegression(C=0.05, max_iter=2000).fit(X, train["y"]),
              HistGradientBoostingClassifier(max_depth=2, learning_rate=0.03, max_iter=150, min_samples_leaf=40,
                                             random_state=0).fit(X, train["y"])]
        return lambda df: np.mean([m.predict_proba(sc.transform(df[FIRST_HOUR]))[:, 1] for m in ms], axis=0)

    # Today's first hour from 5-minute bars (9:30 to 10:30).
    m5 = data.call(yf.Ticker("SPY").history, period="1d" if asof is None else "60d", interval="5m")
    m5.index = m5.index.tz_convert("America/New_York")
    m5 = m5[(m5.index.date == today)]
    first = m5[m5.index < m5.index[0].replace(hour=10, minute=30)]
    if len(first) < 12:
        raise ValueError("the first hour is not complete yet")
    ts = pd.Timestamp(today)
    row = _first_hour(first, float(daily["Close"][daily.index < ts].iloc[-1]), float(atr_s.dropna().iloc[-1]))
    p = float(fit(hist)(pd.DataFrame([row]))[0])

    # Recent record: refit monthly over the last ~120 days, as it would have been made.
    preds, ys = [], []
    for start in range(len(hist) - 120, len(hist), 21):
        model = fit(hist.iloc[:start])
        chunk = hist.iloc[start:start + 21]
        preds += list(model(chunk))
        ys += list(chunk["y"])
    preds, ys = np.array(preds), np.array(ys)
    k = np.abs(preds - 0.5) >= CONFIDENT
    record = {"days": int(k.sum()), "right": float(((preds[k] > 0.5) == (ys[k] == 1)).mean()) if k.any() else None}

    def size(rel: float) -> str:
        return "quiet" if rel < 0.7 else "busy" if rel > 1.5 else "normal"
    rel = row["h1_range"] / float(hist["h1_range"].tail(20).mean())
    past = hist.dropna(subset=["rel"])
    same = past[past["rel"].map(size) == size(rel)]
    afternoon = {"size": size(rel), "rel": rel, "expected": float(same["rest_abs"].mean()),
                 "typical": float(past["rest_abs"].mean()), "days": len(same)}
    return {"p_up": p, "direction": "up" if p >= 0.5 else "down", "confident": abs(p - 0.5) >= CONFIDENT,
            "price": row["price"], "first_hour": {k_: row[k_] for k_ in FIRST_HOUR}, "record": record,
            "afternoon": afternoon}


def describe_update(u: dict[str, Any]) -> str:
    fh = u["first_hour"]
    shape = (f"first hour {fh['h1_ret']:+.2%}, closed {fh['h1_loc']:.0%} of the way up its range, range "
             f"{fh['h1_range']:.1f}x a usual day's, gap {'filled' if fh['gap_filled'] else 'not filled'}")
    rec = u.get("record") or {}
    track = (f". Its confident calls over the last ~6 months were right {rec['right']:.0%} of the time ({rec['days']} days)"
             if rec.get("right") is not None else "")
    a = u.get("afternoon")
    rest = (f". Rest of the day: {a['size']} (first hour {a['rel']:.1f}x the usual width; such days moved "
            f"{a['expected']:.2%} from 10:30 to the close on average, against {a['typical']:.2%} on all days)" if a else "")
    return f"{u['p_up']:.0%} chance SPY closes above {u['price']:.2f} (its 10:30 price); {shape}{rest}{track}"


# ---------------------------------------------------------------- market profile at 10:30
# Tested on SPY, 2024-2026 (700 days, hourly volume profiles, as Yahoo keeps 30-minute bars only 60 days):
#   80% rule: opened outside yesterday's value area and back inside it by 10:30, it reached the far side
#   of value 66% of the time (90 days), about 13 points more than ordinary days the same distance away.
#   It did not keep going to the close, so the far side of value is the exit, not the close.
#   Gap down reclaimed: opened below yesterday's low and traded back into its range within the first
#   hour; SPY closed above its 10:30 price 69% of the time (39 days, both years).
# Downside breaks rarely followed through in that mostly rising market; that may be the market, not a rule.
PROFILE_RECORD = {"80% rule": "reached the far side of value on 66% of 90 days (2024-26)",
                  "gap down reclaimed": "closed above its 10:30 price on 69% of 39 days (2024-26)"}


def _profile(g: pd.DataFrame, tpo: bool = False) -> dict[str, float]:
    """Profile of one session. Volume profile: each bar's volume spread evenly over its high-low.
    Time profile (tpo=True, 30-minute bars): each period marks every price it traded at once, the
    classic market profile. The value area holds 70% around the point of control."""
    lo, hi = float(g["Low"].min()), float(g["High"].max())
    step = float(g["Close"].iloc[0]) * 0.0002
    grid = np.arange(lo, hi + step, step)
    vol = np.zeros(len(grid))
    for b_lo, b_hi, v in zip(g["Low"], g["High"], g["Volume"]):
        k = (grid >= b_lo - 1e-9) & (grid <= b_hi + 1e-9)
        if k.any():
            vol[k] += 1.0 if tpo else v / k.sum()
    poc = int(vol.argmax())
    a = z = poc
    total = vol[poc]
    while total < 0.7 * vol.sum():
        up = vol[z + 1] if z + 1 < len(vol) else -1
        dn = vol[a - 1] if a > 0 else -1
        if up >= dn:
            z, total = z + 1, total + up
        else:
            a, total = a - 1, total + dn
    return {"poc": float(grid[poc]), "val": float(grid[a]), "vah": float(grid[z]), "high": hi, "low": lo,
            "close": float(g["Close"].iloc[-1])}


def profile_1030(asof: date | None = None) -> dict[str, Any]:
    """Where SPY opened against yesterday's profile, and whether a tested setup is live at 10:30."""
    m5 = data.call(yf.Ticker("SPY").history, period="5d" if asof is None else "60d", interval="5m")
    m5.index = m5.index.tz_convert("America/New_York")
    m5 = m5[(m5.index.hour * 60 + m5.index.minute >= 570) & (m5.index.hour < 16)]
    today = asof or datetime.now(data.NEW_YORK).date()
    before = sorted(d for d in set(m5.index.date) if d < today)
    first = m5[(m5.index.date == today) & (m5.index.hour * 60 + m5.index.minute < 630)]
    if not before or len(first) < 12:
        raise ValueError("yesterday's session or today's first hour is missing")
    # The classic 30-minute time profile (TPO). On the 58 days 5-minute bars allow, it chose the same
    # setup as the hourly volume profile behind the two-year study on 95% of days.
    y_bars = m5[m5.index.date == before[-1]]
    y = _profile(y_bars.resample("30min", origin="start").agg(
        {"High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna(), tpo=True)
    o, p = float(first["Open"].iloc[0]), float(first["Close"].iloc[-1])
    h1_hi, h1_lo = float(first["High"].max()), float(first["Low"].min())
    loc = ("above range" if o > y["high"] else "above value" if o > y["vah"] else "below range" if o < y["low"]
           else "below value" if o < y["val"] else "inside value")
    inside = y["val"] < p < y["vah"]
    setup = None
    if loc.startswith("above") and inside:
        setup = {"name": "80% rule", "direction": "down", "target": y["val"],
                 "why": "opened above yesterday's value and fell back inside it: rotate to the value low"}
    elif loc.startswith("below") and inside:
        setup = {"name": "80% rule", "direction": "up", "target": y["vah"],
                 "why": "opened below yesterday's value and rose back inside it: rotate to the value high"}
    elif loc == "below range" and h1_hi >= y["low"]:
        setup = {"name": "gap down reclaimed", "direction": "up", "target": y["val"] if y["val"] > p else None,
                 "why": "gapped below yesterday's low and traded back into its range in the first hour"}
    return {"date": today.isoformat(), "open": o, "price": p, "location": loc, "yesterday": y, "setup": setup}


def describe_profile(pr: dict[str, Any]) -> str:
    y = pr["yesterday"]
    text = (f"Opened {pr['location']} of yesterday's profile (value {y['val']:.2f}-{y['vah']:.2f}, "
            f"point of control {y['poc']:.2f})")
    s = pr.get("setup")
    if s:
        target = f", target {s['target']:.2f}" if s.get("target") else ", hold to the close"
        text += (f". **{s['name']}, {'▲ up' if s['direction'] == 'up' else '▼ down'}**: {s['why']}{target}. "
                 f"Tested: {PROFILE_RECORD[s['name']]}")
    elif pr["location"] == "inside value":
        text += f". Inside value: yesterday's point of control was touched on 78% of such days"
    return text
