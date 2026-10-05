"""Profit grading and self-adjusting weights.

Every journaled contract is graded on dollars of profit for one contract, traded the way the
portfolio trades: bought at the ask, sold at the bid when it reaches the target or the stop,
otherwise at the last bid on the fifth trading day, with commission both ways. Earnings plays
are held through the report and sold at the close of the reaction day.

Once enough contracts have matured (MIN_DAYS sessions and MIN_RECORDS contracts) the factor
and method weights are re-tuned once a week:

  1. On everything except the latest five sessions, each factor and method gets a profit edge:
     per day, the average profit of the contracts it rated in the top fifth minus those in the
     bottom fifth, then a t-statistic across days.
  2. Weights with a clear edge (|t| >= 1) move by at most 20% in its direction.
  3. The proposal is replayed on the latest five sessions, which it was not fitted to: each day
     it picks the top five contracts (one per stock), as the portfolio would. It is adopted only
     if those picks made more money than the picks of the weights in use.
  4. The original weights stay as a control. If the weights in use make less than the control's
     picks in two weekly checks running, they revert to the original.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from typing import Any

import numpy as np

from .config import HOME

FILE = HOME / "journal" / "learned.json"
MIN_DAYS = 20  # sessions of matured contracts before anything is tuned
MIN_RECORDS = 1000
HOLDOUT_DAYS = 5
MAX_STEP = 0.20  # largest change to one weight in one week
PICKS_PER_DAY = 5
REVERT_AFTER = 2  # weekly checks behind the control before reverting


def _state() -> dict[str, Any]:
    try:
        return json.loads(FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"factors": None, "methods": None, "last_tuned": None, "behind_control": 0, "log": []}


def _save(state: dict[str, Any]) -> None:
    FILE.parent.mkdir(parents=True, exist_ok=True)
    FILE.write_text(json.dumps(state, indent=1), encoding="utf-8")


def active(kind: str, base: dict[str, float]) -> dict[str, float]:
    """The weights in use for `kind` ("factors" or "methods"): learned ones, or `base`."""
    learned = _state().get(kind)
    return dict(learned) if learned and set(learned) == set(base) else dict(base)


def log() -> list[dict[str, Any]]:
    return _state()["log"]


# ---------------------------------------------------------------- profit

def profit(entry: dict[str, Any], target: float, stop: float, commission: float) -> float | None:
    """Dollars made on one contract, or None when there are no marks to judge it by."""
    ask = entry["ask"]
    days = {int(k): v for k, v in entry["days"].items()}
    if not days or ask <= 0:
        return None
    if entry.get("source") == "earnings":
        exit_ = _earnings_exit(entry, days)
    else:
        exit_ = None
        up, down = ask * (1 + target), ask * (1 - stop)
        for d in sorted(days):
            m = days[d]
            if m["lo"] <= down:  # checked first: when both happen in a day the order is unknown, assume the worse
                exit_ = m["hi"] if m["hi"] < down else down  # gapped through the stop: take the best bid seen
                break
            if m["hi"] >= up:
                exit_ = m["lo"] if m["lo"] > up else up
                break
        if exit_ is None:
            exit_ = days[max(days)]["close"]
    if exit_ is None:
        return None
    return exit_ * 100 - ask * 100 - 2 * commission


def _earnings_exit(entry: dict[str, Any], days: dict[int, dict]) -> float | None:
    """Close of the reaction day: the report day for before-open reports, the next day otherwise."""
    report = date.fromisoformat(entry["report_date"])
    reaction = int(np.busday_count(date.fromisoformat(entry["entry_date"]), report))
    reaction += 0 if entry.get("timing") == "before open" else 1
    if reaction in days:
        return days[reaction]["close"]
    later = [d for d in days if d >= reaction]
    return days[min(later)]["close"] if later else None


def graded(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Profit for every journaled contract the portfolio could have bought.

    The scanner flags contracts at any price, but dollar profit on a $2,000 contract would swamp
    the $100 ones, and the account cannot buy them, so only contracts within the per-trade limit
    are graded.
    """
    budget = cfg["account_size"] * cfg["risk_per_trade_pct"] / 100
    out = []
    for r in rows:
        if r["ask"] * 100 > budget:
            continue
        p = profit(r, cfg["sim_target_pct"] / 100, cfg["sim_stop_pct"] / 100, cfg["sim_commission"])
        if p is not None:
            out.append({**r, "pnl": p})
    return out


# ---------------------------------------------------------------- edges and picks

def edge(rows: list[dict[str, Any]], value) -> dict[str, Any]:
    """Profit edge of a signal: per day, top-fifth minus bottom-fifth average profit; t across days."""
    by_day = defaultdict(list)
    for r in rows:
        v = value(r)
        if v is not None:
            by_day[r["entry_date"]].append((v, r["pnl"]))
    spreads = []
    for pairs in by_day.values():
        if len(pairs) < 10:
            continue
        pairs.sort(key=lambda p: p[0])
        fifth = max(len(pairs) // 5, 1)
        spreads.append(np.mean([p for _, p in pairs[-fifth:]]) - np.mean([p for _, p in pairs[:fifth]]))
    if not spreads:
        return {"edge": None, "t": None, "days": 0}
    mean = float(np.mean(spreads))
    sd = float(np.std(spreads, ddof=1)) if len(spreads) > 1 else 0.0
    return {"edge": mean, "t": mean / sd * np.sqrt(len(spreads)) if sd > 0 else None, "days": len(spreads)}


def _composite(methods: dict[str, float], weights: dict[str, float]) -> float:
    used = {k: v for k, v in methods.items() if k in weights}
    total = sum(weights[k] for k in used)
    return sum(weights[k] * v for k, v in used.items()) / total if total else 0.0


def rescore(row: dict[str, Any], factors: dict[str, float], methods: dict[str, float],
            min_strength: float, full_conviction: float) -> float | None:
    """The contract's score under other weights, or None if its direction would not have qualified."""
    comp = _composite(row["methods"], methods)
    want = 1 if row["type"] == "call" else -1
    if comp * want < min_strength:
        return None
    parts = dict(row["factors"])
    parts["trend"] = min(abs(comp) / full_conviction, 1.0)
    return sum(factors[k] * (parts.get(k) or 0.0) for k in factors)


def picks_profit(rows: list[dict[str, Any]], factors: dict[str, float], methods: dict[str, float],
                 cfg: dict[str, Any], full_conviction: float) -> float:
    """Profit of the top five contracts per day (one per stock) chosen under the given weights."""
    by_day = defaultdict(list)
    for r in rows:
        s = rescore(r, factors, methods, cfg["min_trend_strength"], full_conviction)
        if s is not None:
            by_day[r["entry_date"]].append((s, r))
    total = 0.0
    for scored in by_day.values():
        seen = set()
        for _, r in sorted(scored, key=lambda x: -x[0]):
            if r["ticker"] in seen:
                continue
            seen.add(r["ticker"])
            total += r["pnl"]
            if len(seen) == PICKS_PER_DAY:
                break
    return total


# ---------------------------------------------------------------- tuning

def _propose(weights: dict[str, float], edges: dict[str, dict], floor: float, total: float) -> dict[str, float]:
    new = {}
    for k, w in weights.items():
        t = edges.get(k, {}).get("t")
        step = float(np.clip(t / 10, -MAX_STEP, MAX_STEP)) if t is not None and abs(t) >= 1 else 0.0
        new[k] = max(w * (1 + step), floor)
    scale = total / sum(new.values())
    return {k: round(v * scale, 4) for k, v in new.items()}


def tune(rows: list[dict[str, Any]], base_factors: dict[str, float], base_methods: dict[str, float],
         cfg: dict[str, Any], full_conviction: float) -> dict[str, Any]:
    """Weekly check and, when unlocked, adjustment. Returns what happened, for the scorecard."""
    state = _state()
    scanner_rows = [r for r in rows if r.get("source") == "scanner" and r.get("factors") and r.get("methods")]
    days = sorted({r["entry_date"] for r in scanner_rows})
    status = {"days": len(days), "records": len(scanner_rows), "unlocked": False, "message": None}
    if len(days) < MIN_DAYS or len(scanner_rows) < MIN_RECORDS:
        status["message"] = (f"Learning unlocks at {MIN_DAYS} sessions and {MIN_RECORDS} graded contracts "
                             f"({len(days)} and {len(scanner_rows)} so far); until then the original weights are used.")
        return status
    status["unlocked"] = True
    week = date.today().strftime("%G-W%V")
    if state.get("last_tuned") == week:
        status["message"] = f"Already tuned this week ({week})."
        return status

    holdout_days = set(days[-HOLDOUT_DAYS:])
    train = [r for r in scanner_rows if r["entry_date"] not in holdout_days]
    holdout = [r for r in scanner_rows if r["entry_date"] in holdout_days]
    cur_f, cur_m = active("factors", base_factors), active("methods", base_methods)
    today = date.today().isoformat()

    # Revert if the learned weights keep losing to the original ones.
    learned = cur_f != base_factors or cur_m != base_methods
    if learned:
        mine = picks_profit(holdout, cur_f, cur_m, cfg, full_conviction)
        control = picks_profit(holdout, base_factors, base_methods, cfg, full_conviction)
        state["behind_control"] = state.get("behind_control", 0) + 1 if mine < control else 0
        if state["behind_control"] >= REVERT_AFTER:
            state.update(factors=None, methods=None, behind_control=0, last_tuned=week)
            state["log"].append({"date": today, "change": "reverted to the original weights",
                                 "reason": f"behind the original weights' picks for {REVERT_AFTER} weeks running "
                                           f"(latest week ${mine:,.0f} against ${control:,.0f})"})
            _save(state)
            status["message"] = "Reverted to the original weights."
            return status

    f_edges = {k: edge(train, lambda r, k=k: r["factors"].get(k)) for k in base_factors}
    m_edges = {k: edge(train, lambda r, k=k: (r["methods"].get(k) or 0) * (1 if r["type"] == "call" else -1)
                       if k in r["methods"] else None) for k in base_methods}
    new_f = _propose(cur_f, f_edges, floor=1.0, total=100.0)
    new_m = _propose(cur_m, m_edges, floor=0.03, total=1.0)
    before = picks_profit(holdout, cur_f, cur_m, cfg, full_conviction)
    after = picks_profit(holdout, new_f, new_m, cfg, full_conviction)
    state["last_tuned"] = week
    if after > before and (new_f != cur_f or new_m != cur_m):
        state.update(factors=new_f, methods=new_m)
        state["log"].append({
            "date": today, "change": "weights adjusted",
            "reason": f"on the latest {HOLDOUT_DAYS} sessions, which the change was not fitted to, its picks made "
                      f"${after:,.0f} against ${before:,.0f} for the weights in use",
            "factors": new_f, "methods": new_m,
        })
        status["message"] = "Weights adjusted."
    else:
        state["log"].append({"date": today, "change": "no change",
                             "reason": f"the proposed weights' picks made ${after:,.0f} on the latest {HOLDOUT_DAYS} "
                                       f"sessions against ${before:,.0f}, so the current weights stay"})
        status["message"] = "No change this week."
    _save(state)
    return status
