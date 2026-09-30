"""Simulated portfolio: buys alert-grade picks and sells on target, stop or time limit.

Buys fill at the ask and sells at the bid, with a commission each way, and exits
are only checked when a scan runs. A price that gaps through a stop between scans
is sold at whatever the bid is then, as it would be in a real account.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from . import config, data
from .config import HOME
from .scanner import WEIGHTS

FILE = HOME / "paper" / "portfolio.json"


def load(cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cash = float(cfg["account_size"])
        return {
            "start_cash": cash, "cash": cash,
            "started": datetime.now().isoformat(timespec="seconds"),
            "updated": None, "positions": [], "closed": [], "equity": [], "log": [],
        }


def _save(state: dict[str, Any]) -> None:
    FILE.parent.mkdir(exist_ok=True)
    FILE.write_text(json.dumps(state, indent=1), encoding="utf-8")


def equity(state: dict[str, Any]) -> float:
    return state["cash"] + sum(p["last_bid"] * 100 for p in state["positions"])


def _contract_notes(row: pd.Series, cfg: dict[str, Any]) -> list[str]:
    """Plain-language reasons this particular contract was chosen."""
    ticker, call = row["ticker"], row["type"] == "call"
    expiry = datetime.strptime(row["expiration"], "%Y-%m-%d").strftime("%b %d").replace(" 0", " ")
    ratio = row["iv_hv"]
    if pd.isna(ratio):
        value = f"Implied volatility is {row['iv']:.0%}."
    else:
        verdict = "cheap" if ratio < 1 else "fairly priced" if ratio < 1.25 else "expensive"
        value = (f"Implied volatility is {row['iv']:.0%} against {row['iv'] / ratio:.0%} realised over "
                 f"the last 20 days, so the option looks {verdict} (ratio {ratio:.2f}).")
    notes = [
        f"Costs ${row['cost']:.0f}, which is the most this trade can lose and fits the "
        f"${config.max_premium(cfg):.0f} per-trade limit.",
        f"Delta {row['delta']:+.2f}: the contract gains about ${abs(row['delta']) * 100:.0f} for each $1 "
        f"{ticker} {'rises' if call else 'falls'}.",
        f"{ticker} must be {'above' if call else 'below'} ${row['breakeven']:.2f} on {expiry} to profit at "
        f"expiry, a {row['breakeven_move']:.1%} move. Options are pricing in a move of about "
        f"{row['expected_move']:.1%} either way.",
        value,
        f"Time decay costs about ${abs(row['theta']) * 100:.2f} a day ({row['theta_pct']:.1%} of the premium).",
        f"Bid/ask spread is {row['spread_pct']:.1f}% with {int(row['openInterest']):,} contracts of open "
        f"interest, so it can be sold without giving much away.",
        f"Model chance of finishing past breakeven at expiry is {row['pop']:.0%}. The plan is to sell "
        f"earlier, at the target or the stop.",
    ]
    if row["earnings_before_expiry"]:
        notes.append("Risk: earnings fall before expiry, and implied volatility usually drops sharply afterwards.")
    return notes


def _close(state: dict[str, Any], pos: dict[str, Any], reason: str, now: str, cfg: dict[str, Any]) -> None:
    proceeds = pos["last_bid"] * 100 - cfg["sim_commission"]
    state["cash"] += proceeds
    pos.update(
        exit_price=pos["last_bid"], exit_time=now, exit_reason=reason, exit_spot=pos["last_spot"],
        pnl=round(proceeds - pos["cost"], 2), ret=proceeds / pos["cost"] - 1,
    )
    state["positions"].remove(pos)
    state["closed"].append(pos)
    state["log"].append({
        "time": now, "action": "SELL", "symbol": pos["symbol"], "label": pos["label"],
        "price": pos["last_bid"], "note": f"{reason}, P/L ${pos['pnl']:+.2f} ({pos['ret']:+.0%})",
    })


def _manage(state: dict[str, Any], now: str, cfg: dict[str, Any]) -> None:
    """Re-price open positions and sell any that hit the stop, target or time limit."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for pos in state["positions"]:
        groups.setdefault((pos["ticker"], pos["expiration"]), []).append(pos)

    for (ticker, expiration), group in groups.items():
        try:
            quotes = data.quotes(ticker, expiration)
            spot = data.spot_price(ticker, group[0]["last_spot"])
        except Exception:
            quotes, spot = None, None
        for pos in group:
            live = False
            if quotes is not None and pos["symbol"] in quotes.index:
                bid = float(quotes.at[pos["symbol"], "bid"])
                ask = float(quotes.at[pos["symbol"], "ask"])
                if bid > 0 and ask > 0:
                    live = True
                    pos.update(last_bid=bid, last_ask=ask, last_spot=spot, last_time=now,
                               high_bid=max(pos["high_bid"], bid), low_bid=min(pos["low_bid"], bid))
                    pos["marks"].append([now, bid])
            dte = (date.fromisoformat(expiration) - date.today()).days
            if dte < 0:
                _close(state, pos, "Expired", now, cfg)
            elif not live:
                continue  # never sell on a stale quote
            elif pos["last_bid"] <= pos["stop_price"]:
                _close(state, pos, "Stop hit", now, cfg)
            elif pos["last_bid"] >= pos["target_price"]:
                _close(state, pos, "Target hit", now, cfg)
            elif dte <= cfg["sim_exit_dte"]:
                _close(state, pos, "Time limit", now, cfg)


def _buy(state: dict[str, Any], result: dict[str, Any], now: str, cfg: dict[str, Any]) -> None:
    contracts = result["contracts"]
    if contracts.empty:
        return
    held = {p["ticker"] for p in state["positions"]}
    cutoff = (datetime.fromisoformat(now) - timedelta(days=cfg["sim_reentry_days"])).isoformat()
    resting = {c["ticker"] for c in state["closed"] if c["exit_time"] > cutoff}
    picks = contracts[(contracts["score"] >= cfg["sim_min_score"]) & ~contracts["stale"]]

    for _, row in picks.iterrows():  # already sorted best first
        if len(state["positions"]) >= cfg["sim_max_positions"]:
            break
        ticker = row["ticker"]
        cost = float(row["ask"]) * 100 + cfg["sim_commission"]
        if ticker in held or ticker in resting or cost > state["cash"]:
            continue
        entry = float(row["ask"])
        info = result["tickers"][ticker]
        expiry = datetime.strptime(row["expiration"], "%Y-%m-%d")
        pos = {
            "symbol": row["contractSymbol"], "ticker": ticker, "type": row["type"],
            "strike": float(row["strike"]), "expiration": row["expiration"],
            "label": f"{ticker} ${row['strike']:g} {row['type']} {expiry:%b} {expiry.day}",
            "opened": now, "entry_price": entry, "cost": round(cost, 2),
            "entry_spot": float(row["spot"]),
            "target_price": round(entry * (1 + cfg["sim_target_pct"] / 100), 2),
            "stop_price": round(entry * (1 - cfg["sim_stop_pct"] / 100), 2),
            "time_exit": (expiry - timedelta(days=cfg["sim_exit_dte"])).strftime("%Y-%m-%d"),
            "last_bid": float(row["bid"]), "last_ask": entry, "last_spot": float(row["spot"]),
            "high_bid": float(row["bid"]), "low_bid": float(row["bid"]), "last_time": now,
            "marks": [[now, float(row["bid"])]],
            "score": float(row["score"]), "trend": float(row["trend"]), "bias": row["bias"],
            "points": {k: float(row[f"pts_{k}"]) for k in WEIGHTS}, "weights": dict(WEIGHTS),
            "methods": info["methods"], "levels": info["levels"],
            "trend_reasons": info["reasons"], "contract_notes": _contract_notes(row, cfg),
            "delta": float(row["delta"]), "theta": float(row["theta"]), "iv": float(row["iv"]),
            "earnings_before_expiry": bool(row["earnings_before_expiry"]),
        }
        state["cash"] -= cost
        state["positions"].append(pos)
        held.add(ticker)
        state["log"].append({
            "time": now, "action": "BUY", "symbol": pos["symbol"], "label": pos["label"],
            "price": entry, "note": f"score {row['score']:.0f}, {row['bias']} trend",
        })


def update(result: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply one scan to the portfolio: manage exits, then look for new buys."""
    state = load(cfg)
    now = datetime.now().isoformat(timespec="seconds")
    _manage(state, now, cfg)
    _buy(state, result, now, cfg)
    state["updated"] = now
    state["equity"].append([now, round(equity(state), 2)])
    _save(state)
    return state
