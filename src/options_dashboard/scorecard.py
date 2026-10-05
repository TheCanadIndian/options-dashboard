"""Daily scorecard: what has made money, graded in dollars per contract.

Rebuilt once a day from the journal (see journal.py), and runs the weekly tuning step in
learning.py. The result is saved for the site's Learning page.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, datetime
from typing import Any

import numpy as np

from . import data, journal, learning, structure
from .config import HOME

FILE = HOME / "journal" / "scorecard.json"
TARGETS = [0.30, 0.50, 0.75, 1.00]
STOPS = [0.25, 0.40, 0.50]
BANDS = [(0, 60), (60, 70), (70, 80), (80, 101)]


def load() -> dict[str, Any]:
    try:
        return json.loads(FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def due() -> bool:
    """Once a day, from the 15:45 scan on (or the next scan if that was missed)."""
    built = load().get("built", "")
    now = datetime.now(data.NEW_YORK)
    return built[:10] != now.date().isoformat() and now.hour * 60 + now.minute >= 15 * 60 + 45


def _summary(pnls: list[float]) -> dict[str, Any]:
    if not pnls:
        return {"trades": 0, "avg": None, "win_rate": None, "total": None}
    return {"trades": len(pnls), "avg": float(np.mean(pnls)), "win_rate": float(np.mean([p > 0 for p in pnls])),
            "total": float(np.sum(pnls))}


def _earnings_results(cfg: dict[str, Any]) -> dict[str, Any]:
    """How the earnings leans did: direction of the reaction, and beat-probability calibration."""
    from . import earnings  # local: earnings imports the scanner stack

    resolved, hits, brier = [], 0, []
    for p in journal.earnings_predictions().values():
        if p["date"] >= date.today().isoformat():
            continue
        try:
            fund = earnings.fundamentals(p["ticker"])
            event = next((e for e in fund["events"] if e["date"] == p["date"] and e["actual"] is not None), None)
            reaction = next((r for r in earnings.reactions(fund["events"], data.history(p["ticker"]))
                             if r["date"] == p["date"]), None)
        except Exception:
            continue
        if not event or not reaction:
            continue
        beat = (event["surprise"] or 0) > 0
        row = {"ticker": p["ticker"], "date": p["date"], "lean": p["lean"], "score": p["score"],
               "move": reaction["move"], "beat": beat, "surprise": event["surprise"],
               "implied": p["implied_move"], "beat_probability": p["beat_probability"]}
        if p["lean"] != "neutral":
            hits += (reaction["move"] > 0) == (p["lean"] == "bullish")
        if p["beat_probability"] is not None:
            brier.append((p["beat_probability"] - beat) ** 2)
        resolved.append(row)
    leaning = [r for r in resolved if r["lean"] != "neutral"]
    return {
        "resolved": sorted(resolved, key=lambda r: r["date"], reverse=True),
        "lean_hit_rate": hits / len(leaning) if leaning else None, "leaning": len(leaning),
        "brier": float(np.mean(brier)) if brier else None,
        "pending": sum(1 for p in journal.earnings_predictions().values() if p["date"] >= date.today().isoformat()),
    }


def build(cfg: dict[str, Any], base_factors: dict[str, float], full_conviction: float) -> dict[str, Any]:
    rows = learning.graded(journal.outcomes(), cfg)
    scanner_rows = [r for r in rows if r["source"] == "scanner"]
    earnings_rows = [r for r in rows if r["source"] == "earnings"]
    base_methods = structure.METHOD_WEIGHTS
    tuning = learning.tune(rows, base_factors, base_methods, cfg, full_conviction)
    act_f, act_m = learning.active("factors", base_factors), learning.active("methods", base_methods)

    factors = {k: {"base": base_factors[k], "active": act_f[k],
                   **learning.edge(scanner_rows, lambda r, k=k: r["factors"].get(k))} for k in base_factors}
    methods = {k: {"base": base_methods[k], "active": act_m[k],
                   **learning.edge(scanner_rows, lambda r, k=k: (r["methods"][k] * (1 if r["type"] == "call" else -1))
                                   if k in r.get("methods", {}) else None)} for k in base_methods}

    bands = []
    for lo, hi in BANDS:
        pnls = [r["pnl"] for r in scanner_rows if r["score"] is not None and lo <= r["score"] < hi]
        bands.append({"band": f"{lo}+" if hi > 100 else f"{lo}-{hi - 1}", **_summary(pnls)})
    otm = defaultdict(list)
    for r in scanner_rows:
        m = r.get("moneyness")
        if m is not None:
            otm["In the money" if m < -0.005 else "At the money" if m <= 0.005 else
                "OTM up to 5%" if m <= 0.05 else "OTM over 5%"].append(r["pnl"])

    exits = []
    budget = cfg["account_size"] * cfg["risk_per_trade_pct"] / 100
    raw = [r for r in journal.outcomes() if r["source"] == "scanner" and r["ask"] * 100 <= budget]
    for t in TARGETS:
        for s in STOPS:
            pnls = [p for p in (learning.profit(r, t, s, cfg["sim_commission"]) for r in raw) if p is not None]
            exits.append({"target": t, "stop": s, **_summary(pnls),
                          "current": abs(t - cfg["sim_target_pct"] / 100) < 1e-9 and abs(s - cfg["sim_stop_pct"] / 100) < 1e-9})

    by_day = defaultdict(list)
    for r in scanner_rows:
        by_day[r["entry_date"]].append(r)
    picks = []
    for day in sorted(by_day):
        picks.append({"date": day,
                      "active": learning.picks_profit(by_day[day], act_f, act_m, cfg, full_conviction),
                      "control": learning.picks_profit(by_day[day], base_factors, base_methods, cfg, full_conviction)})

    card = {
        "built": datetime.now(data.NEW_YORK).isoformat(timespec="minutes"),
        "collected": {"graded": len(rows), "scanner": len(scanner_rows), "earnings_plays": len(earnings_rows),
                      "following": len(journal.open_entries()), "sessions": len(by_day)},
        "tuning": tuning, "factors": factors, "methods": methods, "bands": bands,
        "moneyness": {k: _summary(v) for k, v in otm.items()},
        "exits": exits, "picks": picks,
        "earnings_plays": _summary([r["pnl"] for r in earnings_rows]),
        "earnings": _earnings_results(cfg), "log": learning.log(),
    }
    FILE.parent.mkdir(parents=True, exist_ok=True)
    FILE.write_text(json.dumps(card, indent=1), encoding="utf-8")
    return card
