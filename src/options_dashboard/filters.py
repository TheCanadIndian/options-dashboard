"""Filter tests: candidate entry filters and exits, replayed on every journaled out-of-the-money contract and
compared with every flagged contract, week by week, until one has enough data to adopt.

The user (Nov 2026): be aggressive with changes until something is profitable, keep buying out-of-the-money
contracts, but only change a rule once a new filter has enough data. "Enough" is fixed here, before looking:

  - at least MIN_CONTRACTS graded contracts the filter would have kept, over at least MIN_WEEKS entry weeks;
  - ahead of all flagged contracts (average dollars per contract) in at least 3 of every 4 entry weeks, including
    the latest one;
  - ahead by at least MIN_EDGE dollars per contract overall.

A filter that meets it is marked "Ready"; one that trails in most weeks with enough data is "Not working".
Grades are each contract's longest completed holding period (5, 10 or 15 trading days), as on the rest of
the Learning page. Contracts on the same stock and day overlap, so treat "Ready" as necessary, not proof.
Stops are never loosened in any candidate (the user's rule): exit candidates only change the target or tighten.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Any, Callable

import numpy as np

from . import data, journal, learning

MIN_CONTRACTS = 300
MIN_WEEKS = 4
MIN_EDGE = 2.0  # dollars per contract


def _rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Out-of-the-money scanner contracts within the per-trade limit, at their longest completed grade."""
    budget = cfg["account_size"] * cfg["risk_per_trade_pct"] / 100
    out = []
    for e in journal.outcomes() + list(journal.open_entries().values()):
        if e.get("source") != "scanner" or (e.get("moneyness") or 0) <= 0 or e["ask"] * 100 > budget:
            continue
        done = [h for h in learning.HORIZONS if learning.reached(e, h)]
        if done:
            out.append({**e, "horizon": max(done)})
    return out


def _day_rank(rows: list[dict[str, Any]], value: Callable[[dict], float | None]) -> dict[int, float]:
    """Each row's percentile (0-1) of `value` among the same day's contracts."""
    by_day = defaultdict(list)
    for i, r in enumerate(rows):
        v = value(r)
        if v is not None:
            by_day[r["entry_date"]].append((v, i))
    pct = {}
    for pairs in by_day.values():
        pairs.sort()
        for k, (_, i) in enumerate(pairs):
            pct[i] = k / max(len(pairs) - 1, 1)
    return pct


def _spy_trend() -> Callable[[str], float | None]:
    """SPY's 20-day change as of the last close on or before a date."""
    from bisect import bisect_right
    try:
        close = data.history("SPY")["Close"]
        days = [d.date().isoformat() if hasattr(d, "date") else str(d)[:10] for d in close.index]
        values = (close / close.shift(20) - 1).tolist()
    except Exception:
        return lambda d: None

    def at(d: str) -> float | None:
        k = bisect_right(days, d) - 1
        return None if k < 0 or values[k] != values[k] else values[k]
    return at


def candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The filters under test. Each keeps a row or not; exits are handled separately."""
    gamma = _day_rank(rows, lambda r: r["factors"].get("gamma"))
    breakeven = _day_rank(rows, lambda r: r["factors"].get("breakeven"))
    iv = _day_rank(rows, lambda r: r["factors"].get("iv_value"))
    spy = _spy_trend()
    trend = {i: spy(r["entry_date"]) for i, r in enumerate(rows)}
    with_mkt = lambda i, r: trend[i] is not None and (trend[i] > 0) == (r["type"] == "call")
    return [
        {"id": "score60", "name": "Score 60 or more", "keep": lambda i, r: (r["score"] or 0) >= 60},
        {"id": "score65", "name": "Score 65 or more", "keep": lambda i, r: (r["score"] or 0) >= 65},
        {"id": "score70", "name": "Score 70 or more (what Core buys)", "keep": lambda i, r: (r["score"] or 0) >= 70},
        {"id": "read40", "name": "Market read at least ±40", "keep": lambda i, r: abs(r.get("read") or 0) >= 40},
        {"id": "gamma_top", "name": "Gamma backdrop in the day's top fifth", "keep": lambda i, r: gamma.get(i, 0) >= 0.8},
        {"id": "breakeven_ok", "name": "Breakeven not in the day's bottom fifth", "keep": lambda i, r: breakeven.get(i, 0) > 0.2},
        {"id": "iv_middle", "name": "Option price vs volatility in the middle 60%", "keep": lambda i, r: 0.2 <= iv.get(i, -1) <= 0.8},
        {"id": "otm12", "name": "At least 12% out of the money", "keep": lambda i, r: (r["moneyness"] or 0) >= 0.12},
        {"id": "otm_near", "name": "Under 8% out of the money", "keep": lambda i, r: (r["moneyness"] or 0) < 0.08},
        {"id": "dte30_60", "name": "30 to 60 days to expiry", "keep": lambda i, r: 30 <= r["dte"] <= 60},
        {"id": "otm8", "name": "At least 8% out of the money", "keep": lambda i, r: (r["moneyness"] or 0) >= 0.08},
        {"id": "score60_gamma", "name": "Score 60+ and gamma backdrop in the day's top fifth",
         "keep": lambda i, r: (r["score"] or 0) >= 60 and gamma.get(i, 0) >= 0.8},
        {"id": "core_plus_gamma", "name": "Score 70+, or 60-69 with gamma in the day's top fifth",
         "keep": lambda i, r: (r["score"] or 0) >= 70 or ((r["score"] or 0) >= 60 and gamma.get(i, 0) >= 0.8)},
        # Direction filters bet on one side of the market: Ready only once they cover a week SPY rose and one it fell.
        {"id": "with_spy", "name": "With SPY's 20-day trend (calls when it is up, puts when down)", "keep": with_mkt,
         "direction": True},
        {"id": "puts60", "name": "Puts only, score 60+", "keep": lambda i, r: r["type"] == "put" and (r["score"] or 0) >= 60,
         "direction": True},
        {"id": "calls60", "name": "Calls only, score 60+", "keep": lambda i, r: r["type"] == "call" and (r["score"] or 0) >= 60,
         "direction": True},
    ]


def _week(d: str) -> str:
    y, w, _ = date.fromisoformat(d).isocalendar()
    return f"{y}-W{w:02d}"


def _spy_weeks() -> dict[str, float]:
    """SPY's change in each ISO week (Friday close to Friday close)."""
    try:
        close = data.history("SPY")["Close"]
        weekly = close.resample("W-FRI").last().dropna().pct_change().dropna()
        return {_week(d.date().isoformat()): float(v) for d, v in weekly.items()}
    except Exception:
        return {}


def _verdict(n: int, weeks: list[float], edge: float | None, both_ways: bool = True) -> str:
    if n < MIN_CONTRACTS or len(weeks) < MIN_WEEKS:
        return "Collecting"
    ahead = sum(w > 0 for w in weeks)
    if ahead >= 0.75 * len(weeks) and weeks[-1] > 0 and (edge or 0) >= MIN_EDGE:
        return "Ready" if both_ways else "Collecting"  # a one-sided bet needs a rising and a falling week
    if ahead <= 0.25 * len(weeks):
        return "Not working"
    return "Mixed"


def _compare(pnl_new: dict[int, float], pnl_base: dict[int, float], rows: list[dict[str, Any]],
             spy_weeks: dict[str, float] | None = None) -> dict[str, Any]:
    """The candidate against every flagged contract: overall and per entry week. With `spy_weeks`, the
    candidate also needs entry weeks when SPY rose and when it fell before it can be Ready."""
    by_week_new, by_week_base = defaultdict(list), defaultdict(list)
    for i, p in pnl_base.items():
        by_week_base[_week(rows[i]["entry_date"])].append(p)
    for i, p in pnl_new.items():
        by_week_new[_week(rows[i]["entry_date"])].append(p)
    weeks = sorted(w for w in by_week_new if by_week_base.get(w))
    diffs = [float(np.mean(by_week_new[w]) - np.mean(by_week_base[w])) for w in weeks]
    vals = list(pnl_new.values())
    avg = float(np.mean(vals)) if vals else None
    base = float(np.mean(list(pnl_base.values()))) if pnl_base else None
    edge = None if avg is None or base is None else avg - base
    both = True
    if spy_weeks is not None:
        moves = [spy_weeks.get(w) for w in weeks]
        both = any(m is not None and m > 0 for m in moves) and any(m is not None and m < 0 for m in moves)
    return {"contracts": len(vals), "sessions": len({rows[i]["entry_date"] for i in pnl_new}), "avg": avg,
            "win_rate": float(np.mean([v > 0 for v in vals])) if vals else None, "edge": edge,
            "weeks": len(weeks), "weeks_ahead": sum(d > 0 for d in diffs), "latest_week": diffs[-1] if diffs else None,
            "status": _verdict(len(vals), diffs, edge, both)}


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    rows = _rows(cfg)
    tgt, stp, fee = cfg["sim_target_pct"] / 100, cfg["sim_stop_pct"] / 100, cfg["sim_commission"]

    def pnls(target: float, stop: float, keep: Callable[[int, dict], bool] | None = None) -> dict[int, float]:
        out = {}
        for i, r in enumerate(rows):
            if keep is not None and not keep(i, r):
                continue
            p = learning.profit(r, target, stop, fee, r["horizon"])
            if p is not None:
                out[i] = p
        return out

    base = pnls(tgt, stp)
    spy_weeks = _spy_weeks()
    filt = [{"id": c["id"], "name": c["name"],
             **_compare(pnls(tgt, stp, c["keep"]), base, rows, spy_weeks if c.get("direction") else None)}
            for c in candidates(rows)]
    exits = []
    for t in (0.75, 1.0, 1.5, 2.0):
        for s in sorted({stp, 0.30, 0.25}):
            if s > stp or (abs(t - tgt) < 1e-9 and abs(s - stp) < 1e-9):
                continue  # never a looser stop than the one in use
            exits.append({"id": f"exit_{int(t * 100)}_{int(s * 100)}", "name": f"Target +{t:.0%}, stop -{s:.0%}",
                          **_compare(pnls(t, s), base, rows)})
    return {"baseline": {"name": f"every out-of-the-money contract the scanner flagged, target +{tgt:.0%}, stop -{stp:.0%}",
                         "contracts": len(base), "avg": float(np.mean(list(base.values()))) if base else None,
                         "win_rate": float(np.mean([v > 0 for v in base.values()])) if base else None},
            "filters": filt, "exits": exits,
            "bar": {"contracts": MIN_CONTRACTS, "weeks": MIN_WEEKS, "edge": MIN_EDGE}}
