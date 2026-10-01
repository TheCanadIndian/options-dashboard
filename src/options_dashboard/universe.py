"""The daily scan universe: liquid US stocks and ETFs whose options trade tightly.

Built once per trading day in two passes. Yahoo's stock screener supplies every US
stock above a volume, price and size floor (sorted by volume), plus a fixed list of
option-heavy ETFs. Each candidate's options are then checked on the expiry nearest
30 days out: open interest near the money, and the bid/ask spread at the money.
The most liquid pass the cut, and the watchlist is always included.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time as clock
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import yfinance as yf
from yfinance import EquityQuery

from . import data
from .config import STATE_DIR

FILE = STATE_DIR / "universe.json"
NEW_YORK = ZoneInfo("America/New_York")

# Screeners cover stocks only; these ETFs carry some of the deepest options markets.
ETFS = [
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "XLI", "XLU", "XLP", "XLY", "XLB", "XBI",
    "SMH", "KRE", "GLD", "SLV", "GDX", "GDXJ", "TLT", "HYG", "EEM", "EFA", "FXI", "KWEB", "USO", "UNG",
    "ARKK", "IBIT", "TQQQ", "SQQQ", "SOXL", "TNA",
]


def _screen(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """US stocks above the volume, price and market-cap floors, most traded first."""
    query = EquityQuery("and", [
        EquityQuery("eq", ["region", "us"]),
        EquityQuery("gt", ["avgdailyvol3m", cfg["universe_min_avg_volume"]]),
        EquityQuery("gt", ["intradayprice", cfg["universe_min_price"]]),
        EquityQuery("gt", ["intradaymarketcap", cfg["universe_min_market_cap"]]),
    ])
    found: dict[str, dict[str, Any]] = {}
    while len(found) < cfg["universe_candidates"]:
        res = data.call(yf.screen, query, offset=len(found), size=250, sortField="avgdailyvol3m", sortAsc=False)
        quotes = res.get("quotes", [])
        for q in quotes:
            earnings = q.get("earningsTimestamp")
            found[q["symbol"]] = {
                "name": q.get("shortName") or q.get("longName") or q["symbol"],
                "price": q.get("regularMarketPrice"),
                "dividend_yield": (q.get("dividendYield") or 0) / 100,
                "earnings": datetime.fromtimestamp(earnings).date().isoformat() if earnings else None,
            }
        if not quotes or len(found) >= res.get("total", 0):
            break
    return dict(list(found.items())[: cfg["universe_candidates"]])


def _liquidity(ticker: str, price: float | None) -> dict[str, Any] | None:
    """Near-the-money open interest and at-the-money spread on the expiry nearest 30 days."""
    tk = yf.Ticker(ticker)
    today = date.today()
    expiries = [(abs((date.fromisoformat(e) - today).days - 30), e) for e in data.call(lambda: tk.options)]
    expiries = [(gap, e) for gap, e in expiries if gap <= 25]
    if not expiries:
        return None
    expiry = min(expiries)[1]
    chain = data.call(tk.option_chain, expiry)
    spot = price or float(chain.underlying["regularMarketPrice"])
    oi, spreads, quoted = 0.0, [], False
    for side in (chain.calls, chain.puts):
        near = side[side["strike"].between(spot * 0.9, spot * 1.1)]
        oi += float(near["openInterest"].fillna(0).sum())
        atm = side.assign(gap=(side["strike"] - spot).abs()).nsmallest(2, "gap")
        live = atm[(atm["bid"] > 0) & (atm["ask"] > 0)]
        quoted |= not live.empty
        mid = (live["bid"] + live["ask"]) / 2
        spreads += list(((live["ask"] - live["bid"]) / mid * 100).round(2))
    return {"oi": oi, "spread": float(np.median(spreads)) if spreads else None, "quoted": quoted,
            "price": spot, "expiry": expiry}


def _market_settled() -> bool:
    """Options quotes are reliable from 15 minutes after the open until the close."""
    now = datetime.now(NEW_YORK)
    return now.weekday() < 5 and clock(9, 45) <= now.time() <= clock(16, 0)


def load() -> dict[str, Any]:
    try:
        return json.loads(FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def build(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    """Return today's universe, rebuilding it when it is stale.

    A universe built without live quotes (market closed) is rebuilt once quotes are
    live, so the spread check always runs on real bids and asks when it can.
    """
    cached = load()
    today = date.today().isoformat()
    live = _market_settled()
    if not force and cached and (cached.get("date") == today and (cached.get("live") or not live)):
        return cached
    if not force and cached and not live:
        return cached  # market closed: yesterday's list is the best available

    candidates = _screen(cfg)
    for etf in ETFS:
        candidates.setdefault(etf, {"name": etf, "price": None, "dividend_yield": None, "earnings": None})

    def check(item):
        symbol, meta = item
        try:
            return symbol, _liquidity(symbol, meta["price"])
        except Exception:
            return symbol, None

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = dict(pool.map(check, candidates.items()))

    passed = []
    for symbol, liq in results.items():
        if not liq or liq["oi"] < cfg["universe_min_option_oi"]:
            continue
        if live and (liq["spread"] is None or liq["spread"] > cfg["universe_max_spread_pct"]):
            continue
        passed.append((liq["oi"], symbol))
    keep = [s for _, s in sorted(passed, reverse=True)[: cfg["universe_size"]]]

    universe = {
        "date": today, "live": live, "built": datetime.now().isoformat(timespec="seconds"),
        "screened": len(candidates), "checked": sum(r is not None for r in results.values()),
        "tickers": {s: {**candidates[s], **results[s]} for s in keep},
    }
    STATE_DIR.mkdir(exist_ok=True)
    FILE.write_text(json.dumps(universe, indent=1), encoding="utf-8")
    return universe


def tickers(cfg: dict[str, Any]) -> list[str]:
    """Watchlist first, then the liquid universe when it is enabled."""
    names = list(cfg["watchlist"])
    if cfg["use_universe"]:
        try:
            universe = build(cfg)
        except Exception:
            universe = load()  # screener unavailable: fall back to the last good list
        names += [t for t in universe.get("tickers", {}) if t not in names]
    return names


def meta(ticker: str) -> dict[str, Any]:
    """Screener details for a ticker (earnings date, dividend yield), when known."""
    return load().get("tickers", {}).get(ticker, {})
