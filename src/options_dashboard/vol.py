"""Implied volatility surface, skew and term structure from one option chain.

The surface is read by delta rather than strike so expiries and stocks compare directly:
for each expiry, implied volatility at the 10- and 25-delta puts, at the money, and the
25- and 10-delta calls, using out-of-the-money options on each side. From it:

    risk reversal (25-delta call IV - 25-delta put IV) at 30 days, judged against the stock's
    own history: rising means upside is in demand, falling means protection is
    put skew / call skew: each 25-delta wing's premium over at-the-money IV
    term structure: at-the-money IV out at ~60 days against the front (~7-21 days);
    an inverted curve (front above back) means near-term stress or an event

A daily snapshot per ticker builds the history behind IV rank and "unusual for this stock".
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from . import greeks
from .config import STATE_DIR

DIR = STATE_DIR / "vol"
POINTS = [("p10", -0.10), ("p25", -0.25), ("atm", 0.0), ("c25", 0.25), ("c10", 0.10)]
LABELS = {"p10": "10Δ put", "p25": "25Δ put", "atm": "ATM", "c25": "25Δ call", "c10": "10Δ call"}
MIN_HISTORY = 20  # sessions before IV rank and skew history are used
MAX_SPREAD = 0.40  # bid/ask spread as a share of the mid beyond which a quote is too loose to read
MAX_IV_RATIO = 2.0  # IV more than this multiple away from its expiry's median is treated as a bad quote


def _wing(side: pd.DataFrame, target: float) -> float:
    """IV at |delta| = target on one side of the smile, interpolated, or NaN outside the quotes."""
    side = side.sort_values("absdelta")
    d, v = side["absdelta"].to_numpy(), side["iv"].to_numpy()
    if len(d) < 2 or not d[0] <= target <= d[-1]:
        return float("nan")
    return float(np.interp(target, d, v))


def surface(chain: pd.DataFrame, spot: float, rate: float, max_dte: int = 120) -> dict[str, Any] | None:
    """The delta-by-expiry surface, 30-day skew metrics, term structure and smile residuals."""
    mid = (chain["bid"] + chain["ask"]) / 2
    df = chain[(chain["impliedVolatility"] > 0.03) & (chain["impliedVolatility"] < 5)
               & chain["dte"].between(3, max_dte) & (chain["bid"] > 0) & (chain["ask"] > chain["bid"])
               & ((chain["ask"] - chain["bid"]) / mid <= MAX_SPREAD)].copy()
    if df.empty:
        return None
    df["iv"] = df["impliedVolatility"].astype(float)
    # Stale or one-off quotes on far strikes produce absurd IVs; keep each expiry within a band of its median.
    typical = df.groupby("expiration")["iv"].transform("median")
    df = df[(df["iv"] <= typical * MAX_IV_RATIO) & (df["iv"] >= typical / MAX_IV_RATIO)]
    if df.empty:
        return None
    T = np.maximum(df["dte"].to_numpy(dtype=float), 0.5) / 365
    is_call = (df["type"] == "call").to_numpy()
    df["absdelta"] = np.abs(greeks.greeks(spot, df["strike"].to_numpy(dtype=float), T, rate,
                                          df["iv"].to_numpy(), 0.0, is_call)["delta"])
    df["logk"] = np.log(df["strike"] / spot)

    rows, residual = [], {}
    for expiry, ex in df.groupby("expiration"):
        otm = ex[((ex["type"] == "call") & (ex["strike"] >= spot)) | ((ex["type"] == "put") & (ex["strike"] <= spot))]
        if len(otm) < 4:
            continue
        nearest = ex.iloc[(ex["strike"] - spot).abs().argsort()[:4]]
        row = {"expiration": expiry, "dte": int(ex["dte"].iloc[0]), "atm": float(nearest["iv"].median())}
        for key, target in POINTS:
            if key != "atm":
                kind = "put" if target < 0 else "call"
                row[key] = _wing(otm[otm["type"] == kind], abs(target))
        rows.append(row)
        # Smile fit across out-of-the-money strikes: how far each contract sits from its neighbours.
        if len(otm) >= 5:
            fit = np.polyfit(otm["logk"], otm["iv"], 2, w=np.sqrt(otm["openInterest"] + 1))
            fitted = np.polyval(fit, ex["logk"])
            residual.update(dict(zip(ex["contractSymbol"], ex["iv"] - fitted)))
    if len(rows) < 2:
        return None
    grid = pd.DataFrame(rows).sort_values("dte").reset_index(drop=True)

    def at(days: int, key: str) -> float:
        ok = grid[["dte", key]].dropna()
        if ok.empty:
            return float("nan")
        return float(np.interp(days, ok["dte"], ok[key]))

    atm30, p25, c25 = at(30, "atm"), at(30, "p25"), at(30, "c25")
    front = grid[grid["dte"].between(5, 21)]
    back = grid[grid["dte"].between(45, 90)]
    term = (float(back["atm"].median()) / float(front["atm"].median()) - 1) if len(front) and len(back) else float("nan")
    return {
        "grid": grid.to_dict("records"),
        "atm30": atm30, "rr25": c25 - p25, "put_skew": p25 - atm30, "call_skew": c25 - atm30,
        "term_slope": term, "front_expiry": front["expiration"].iloc[0] if len(front) else None,
        "residuals": residual,
    }


def _clean(x: float) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def record(ticker: str, surf: dict[str, Any], iv30: float | None) -> dict[str, Any]:
    """Store today's reading (latest scan wins) and return IV rank and skew history stats."""
    path = DIR / f"{ticker}.json"
    try:
        history = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        history = {}
    level = iv30 / 100 if iv30 else surf["atm30"]  # Cboe's 30-day IV is quoted in percent
    rr_ratio = surf["rr25"] / surf["atm30"] if surf["atm30"] else float("nan")
    history[date.today().isoformat()] = {"iv30": _clean(level), "rr": _clean(rr_ratio)}
    history = dict(sorted(history.items())[-260:])
    DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history), encoding="utf-8")

    past = [v for d, v in history.items() if d < date.today().isoformat()]
    ivs = [v["iv30"] for v in past if v["iv30"] is not None]
    rrs = [v["rr"] for v in past if v["rr"] is not None]
    stats = {"sessions": len(past), "iv30": _clean(level), "rr_ratio": _clean(rr_ratio),
             "iv_rank": None, "iv_percentile": None, "rr_z": None,
             "rr_change": (rr_ratio - rrs[-1]) if rrs and not np.isnan(rr_ratio) else None}
    if len(ivs) >= MIN_HISTORY and level:
        lo, hi = min(ivs), max(ivs)
        stats["iv_rank"] = (level - lo) / (hi - lo) if hi > lo else 0.5
        stats["iv_percentile"] = sum(v < level for v in ivs) / len(ivs)
    if len(rrs) >= MIN_HISTORY and not np.isnan(rr_ratio):
        sd = float(np.std(rrs))
        stats["rr_z"] = (rr_ratio - float(np.mean(rrs))) / sd if sd > 0 else 0.0
    return stats


def read(surf: dict[str, Any], stats: dict[str, Any], earnings_soon: bool) -> dict[str, Any]:
    """Direction score and reasons from skew and term structure."""
    score, reasons = 0.0, []
    atm, rr = surf["atm30"], surf["rr25"]
    ratio = rr / atm if atm else float("nan")
    if not np.isnan(ratio):
        # Every stock has its own normal skew (indexes carry steep put skew, high-volatility names a
        # flat smile), so skew is only judged against the stock's own history.
        shape = f"25-delta risk reversal {rr * 100:+.1f} vol points ({ratio:+.0%} of ATM IV)"
        if stats["rr_z"] is not None:
            z = stats["rr_z"]
            pts = float(np.clip(z * 15, -30, 30))
            basis = f"{abs(z):.1f} standard deviations {'above' if z > 0 else 'below'} its usual level"
        elif stats.get("rr_change") is not None:
            change = stats["rr_change"]
            pts = float(np.clip(change / 0.05 * 10, -10, 10))
            basis = f"{'flatter' if change > 0 else 'steeper'} than the last session by {abs(change):.0%} of ATM IV"
        else:
            pts, basis = 0.0, None
        score += pts
        if basis is None:
            reasons.append(f"= {shape}; needs another session of history before it can be judged for this stock")
        elif pts >= 3:
            reasons.append(f"+ Skew moving toward upside: {shape}, {basis}; call demand is rising relative to puts")
        elif pts <= -3:
            reasons.append(f"- Skew moving toward protection: {shape}, {basis}; put demand is rising relative to calls")
        else:
            reasons.append(f"= Skew steady: {shape}, {basis}")
    if not np.isnan(surf["put_skew"]) and not np.isnan(surf["call_skew"]):
        reasons.append(f"= 30-day wings: 25-delta puts {surf['put_skew'] * 100:+.1f} and 25-delta calls "
                       f"{surf['call_skew'] * 100:+.1f} vol points against {atm:.0%} at the money")

    term = surf["term_slope"]
    if not np.isnan(term):
        if term < -0.05 and not earnings_soon:
            score -= 20
            reasons.append(f"- Term structure inverted: front-month IV {abs(term):.0%} above 60-day IV with no "
                           "earnings to explain it, a sign of near-term stress")
        elif term < -0.05:
            reasons.append(f"= Term structure inverted by {abs(term):.0%}, explained by the upcoming earnings report")
        elif term > 0.05:
            reasons.append(f"= Term structure in contango: 60-day IV {term:+.0%} against the front, a calm market")
        else:
            reasons.append(f"= Term structure flat: 60-day IV {term:+.0%} against the front")

    if stats["iv_rank"] is not None:
        reasons.append(f"= IV rank {stats['iv_rank']:.0%} (30-day IV {stats['iv30']:.0%}, percentile "
                       f"{stats['iv_percentile']:.0%} over {stats['sessions']} sessions): options are "
                       f"{'cheap' if stats['iv_rank'] < 0.3 else 'expensive' if stats['iv_rank'] > 0.7 else 'mid-range'} "
                       "against their own history")
    else:
        reasons.append(f"= IV rank needs {MIN_HISTORY} sessions of history ({stats['sessions']} so far)")

    return {"score": float(np.clip(score, -100, 100)), "reasons": reasons,
            "atm30": _clean(atm), "rr25": _clean(rr), "put_skew": _clean(surf["put_skew"]),
            "call_skew": _clean(surf["call_skew"]), "term_slope": _clean(term),
            "iv_rank": stats["iv_rank"], "iv30": stats["iv30"]}
