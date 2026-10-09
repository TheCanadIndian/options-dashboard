"""Which side holds the open interest: customer buy and sell flow, classified from quotes.

Free data has no record of who bought and who sold. Each scan (every 15 minutes in market
hours) sees every contract's cumulative volume, its last trade and its bid and ask. Volume
added since the previous scan is classified by where the last trade printed:

    within the top quarter of the spread    -> customer buy (lifting the offer)
    within the bottom quarter of the spread -> customer sell (hitting the bid)
    in between                              -> unclassified

Over the last few sessions that gives each contract a buy and sell tally. The customer net
position is estimated as (buys - sells), capped at the open interest, so the estimated
customer share is between -1 (all short) and +1 (all long). Open interest that no tracked
flow explains is treated as unknown, contributing nothing, rather than assumed to be a buy.

Coverage (classified volume as a share of open interest) says how much of the open interest
this explains. It starts near zero and builds as sessions accumulate.

Each scan also keeps every near-dated contract's implied volatility. The last reading of a
session becomes the close the next session's changes are measured from, so each strike's own
IV move (skew steepening or flattening, not just the 30-day average) can drive the vanna read.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .config import STATE_DIR

DIR = STATE_DIR / "flow"
SESSIONS = 10  # sessions of classified flow kept per ticker
EDGE = 0.25  # share of the spread at each end that counts as hitting the bid or lifting the offer
IV_MAX_DTE = 60  # implied volatility is kept for contracts expiring within this many days


def _load(ticker: str) -> dict[str, Any]:
    try:
        return json.loads((DIR / f"{ticker}.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"seen": {}, "days": {}}


def record(ticker: str, chain: pd.DataFrame) -> None:
    """Classify the volume traded since the last scan today and add it to today's tally."""
    state = _load(ticker)
    today = date.today().isoformat()
    if state.get("seen_day") != today:
        if state.get("iv_last"):  # the previous session's final readings become the close
            state["iv_close"], state["iv_close_day"] = state["iv_last"], state.get("seen_day")
        state["seen"], state["seen_day"], state["iv_last"] = {}, today, {}
    quoted = chain[(chain["openInterest"] > 0) & (chain["bid"] > 0) & (chain["dte"] <= IV_MAX_DTE)
                   & chain["impliedVolatility"].between(0.03, 5)]
    state.setdefault("iv_last", {}).update(
        {s: round(float(v), 4) for s, v in zip(quoted["contractSymbol"], quoted["impliedVolatility"])})
    tally = state["days"].setdefault(today, {})

    traded = chain[(chain["volume"] > 0) & (chain["bid"] > 0) & (chain["ask"] > chain["bid"])]
    for symbol, vol, last, bid, ask in zip(traded["contractSymbol"], traded["volume"], traded["lastPrice"],
                                           traded["bid"], traded["ask"]):
        new = vol - state["seen"].get(symbol, 0.0)
        state["seen"][symbol] = float(vol)
        if new <= 0:
            continue
        width = ask - bid
        buy, sell = tally.get(symbol, [0.0, 0.0])
        if last >= ask - EDGE * width:
            buy += new
        elif last <= bid + EDGE * width:
            sell += new
        else:
            continue
        tally[symbol] = [buy, sell]

    for old in sorted(state["days"])[:-SESSIONS]:
        del state["days"][old]
    DIR.mkdir(parents=True, exist_ok=True)
    (DIR / f"{ticker}.json").write_text(json.dumps(state), encoding="utf-8")


def iv_changes(ticker: str, chain: pd.DataFrame) -> tuple[np.ndarray, str | None]:
    """Each contract's implied volatility change in vol points since the prior session's close,
    aligned with `chain` (NaN where either reading is missing), and the day of that close."""
    state = _load(ticker)
    close = state.get("iv_close") or {}
    if not close:
        return np.full(len(chain), np.nan), None
    then = np.array([close.get(s, np.nan) for s in chain["contractSymbol"]], dtype=float)
    now = chain["impliedVolatility"].to_numpy(dtype=float)
    now = np.where((now > 0.03) & (now < 5) & (chain["bid"].to_numpy(dtype=float) > 0), now, np.nan)
    return np.clip(100 * (now - then), -IV_MOVE_CAP, IV_MOVE_CAP), state.get("iv_close_day")


IV_MOVE_CAP = 25.0  # vol points; a larger one-day move in one strike is more likely a bad quote


def customer_positions(ticker: str, chain: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimated customer net position per contract (signed contracts, capped at open interest).

    Returns the positions aligned with `chain` and a summary: coverage, sessions, buy and sell totals.
    """
    state = _load(ticker)
    buys: dict[str, float] = {}
    sells: dict[str, float] = {}
    for tally in state["days"].values():
        for symbol, (b, s) in tally.items():
            buys[symbol] = buys.get(symbol, 0.0) + b
            sells[symbol] = sells.get(symbol, 0.0) + s
    symbols = chain["contractSymbol"]
    oi = chain["openInterest"].to_numpy(dtype=float)
    net = np.array([buys.get(s, 0.0) - sells.get(s, 0.0) for s in symbols])
    positions = np.clip(net, -oi, oi)
    classified = float(sum(buys.values()) + sum(sells.values()))
    total_oi = float(oi.sum())
    summary = {
        "sessions": len(state["days"]),
        "coverage": min(classified / total_oi, 1.0) if total_oi else 0.0,
        "buys": float(sum(buys.values())), "sells": float(sum(sells.values())),
    }
    return positions, summary
