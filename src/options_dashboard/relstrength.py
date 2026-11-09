"""Relative strength rating, in the style of IBD's: 40% of the 3-month return plus 20% each of the 6-, 9- and
12-month returns, ranked 1-99 across the S&P 500. Built from the research job's weekly closes.

What the history says (weekly snapshots of today's S&P 500, 2008-2026; Nov 2026 study):
- Over 1-3 weeks the rating tells you almost nothing (buckets differ by under 0.4% against SPY), so it is not a
  swing-trade signal.
- Over 3 months the top-rated beat SPY, but mostly because they are volatile: rising markets reward volatile stocks
  of any rating. Among the wildest third, rated 80+ beat SPY by +3.6% a quarter in rising markets (median +1.5%,
  54% beat it) against +2.0% for equally wild stocks rated lower; positive in 15 of 18 years. Calmer top-rated
  stocks did nothing (+0.3%). That pairing is the "promising for upside" list.
- No downside rule held: the lowest-rated lagged SPY no more often than anything else in rising markets, and beat
  it (+4.5% a quarter) after the market had fallen. Near 52-week lows, below their averages, or leaders losing
  their rating: none lagged. So the weakest list is shown as context, not as put candidates.
- The test used today's members only: stocks that collapsed and left the index are missing, which flatters the
  low-rated. Treat the numbers as rough; the shape (volatility explains most of it, no downside edge) is clear.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from . import data

WEEKS = {13: 0.4, 26: 0.2, 39: 0.2, 52: 0.2}  # weeks back: weight (3, 6, 9, 12 months)
UPSIDE_MIN = 80  # rating
WEAK_MAX = 20
EVIDENCE = {
    "upside": "Rated 80+ and among the most volatile third: in rising markets these beat SPY by 3.6% over the next "
              "3 months on average (the typical one by 1.5%; 54% beat it), positive in 15 of 18 years since 2008. "
              "Equally volatile stocks with lower ratings beat it by 2.0%, so the rating adds about 1.6 points a quarter.",
    "calm": "Top-rated calmer stocks did no better than SPY (+0.3%), so they are left off the list.",
    "falling": "After SPY has fallen over six months, momentum usually stumbles: the lowest-rated stocks bounced hardest "
               "(+4.5% a quarter against SPY). The volatile top-rated still held up (+3.0%), but with less to spare.",
    "downside": "No downside rule held up. The lowest-rated lagged SPY no more often than any other stock in rising markets, "
                "and beat it after sell-offs; near 52-week lows, below their averages, or leaders losing their rating: "
                "none of them reliably kept falling. So nothing here is a put candidate.",
    "swings": "Over 1 to 3 weeks the rating made almost no difference (under 0.4% against SPY), so it does not feed the "
              "options scanner.",
    "caveat": "Tested on today's S&P 500 members, weekly from 2008: companies that collapsed and left the index are missing, "
              "which flatters the low-rated. One stock beating SPY 54% of the time is close to a coin flip: it is a tilt "
              "across many names, not a reason to buy one.",
}


def _score(close: pd.DataFrame, back: int = 0) -> pd.Series:
    """The weighted return as of `back` weeks ago, one value per ticker."""
    end = close.iloc[len(close) - 1 - back]
    total = pd.Series(0.0, index=close.columns)
    for weeks, w in WEEKS.items():
        start = close.iloc[len(close) - 1 - back - weeks]
        total = total + w * (end / start - 1)
    return total


def _rate(score: pd.Series) -> pd.Series:
    s = score.dropna()
    return (s.rank(pct=True) * 98 + 1).round().astype(int)


def ratings(members: pd.DataFrame, prices: dict[str, pd.DataFrame], spy: pd.DataFrame | None) -> dict[str, Any]:
    """Every S&P 500 member's rating with context, the upside list and the weakest list."""
    close = pd.DataFrame({t: df["Close"] for t, df in prices.items() if len(df) >= 58}).sort_index().ffill()
    if close.empty or len(close) < 58:
        return {}
    now, before = _rate(_score(close)), _rate(_score(close, 4))
    weekly = close.pct_change().iloc[-13:]
    vol = weekly.std().rank(pct=True)
    avg10, avg40 = close.iloc[-10:].mean(), close.iloc[-40:].mean()
    hi52 = close.iloc[-52:].max()
    last = close.iloc[-1]
    spy_c = spy["Close"] if spy is not None else None
    ret = lambda w: last / close.iloc[-1 - w] - 1
    spy_ret = (lambda w: float(spy_c.iloc[-1] / spy_c.iloc[-1 - w] - 1)) if spy_c is not None and len(spy_c) > 53 else None
    info = members.set_index("symbol")
    rows = []
    for t, r in now.items():
        v = float(vol.get(t, np.nan))
        rows.append({
            "ticker": t, "name": str(info["name"].get(t, t)), "sector": str(info["sector"].get(t, "")),
            "rating": int(r), "change": int(r - before[t]) if t in before.index else None,
            "price": float(last[t]),
            "vs_spy_3m": float(ret(13)[t] - spy_ret(13)) if spy_ret else None,
            "vs_spy_12m": float(ret(52)[t] - spy_ret(52)) if spy_ret else None,
            "ret_3m": float(ret(13)[t]), "ret_12m": float(ret(52)[t]),
            "volatility": None if np.isnan(v) else ("Wild" if v >= 2 / 3 else "Middle" if v >= 1 / 3 else "Calm"),
            "from_high": float(last[t] / hi52[t] - 1),
            "above_10w": bool(last[t] > avg10[t]), "above_40w": bool(last[t] > avg40[t]),
        })
    rows.sort(key=lambda r: -r["rating"])
    upside = [r["ticker"] for r in rows if r["rating"] >= UPSIDE_MIN and r["volatility"] == "Wild"]
    weakest = [r["ticker"] for r in sorted(rows, key=lambda r: r["rating"])
               if r["rating"] < WEAK_MAX and not r["above_10w"] and not r["above_40w"]]
    spy_6m = spy_ret(26) if spy_ret else None
    return {"built": datetime.now(data.NEW_YORK).isoformat(timespec="minutes"), "as_of": str(close.index[-1].date()),
            "rows": rows, "upside": upside, "weakest": weakest, "spy_6m": spy_6m,
            "rising": None if spy_6m is None else spy_6m > 0, "evidence": EVIDENCE}
