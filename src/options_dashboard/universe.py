"""The daily scan universe: stocks and ETFs whose options trade heavily and tightly.

Built once per day in two passes.

1. Candidates. OCC publishes the previous session's options volume for every US
   underlying. Names are ranked by contracts traded and kept if Nasdaq lists them as
   a stock (above the price and market-cap floors) or an ETF, which drops index
   options such as SPX and VIX. If OCC or Nasdaq is unavailable, Yahoo's stock
   screener (sorted by share volume) plus a fixed ETF list is used instead.
2. Liquidity. Each candidate's full option chain comes from Cboe's delayed quotes in
   one request. It needs enough open interest within 10% of the money across expiries
   7 to 60 days out, and, during market hours, a tight median at-the-money bid/ask
   spread on the expiry in that window with the most open interest.

The watchlist is always scanned as well.
"""

from __future__ import annotations

import io
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time as clock, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from yfinance import EquityQuery

from . import data
from .config import STATE_DIR

FILE = STATE_DIR / "universe.json"
NEW_YORK = ZoneInfo("America/New_York")
HEADERS = data.CBOE_HEADERS  # Nasdaq and OCC refuse requests without a browser user agent

YAHOO_MAX_CANDIDATES = 600  # the Yahoo fallback has no options-volume floor, so it needs a cap

# Used only by the Yahoo fallback, which covers stocks alone.
ETFS = [
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "XLI", "XLU", "XLP", "XLY", "XLB", "XBI",
    "SMH", "KRE", "GLD", "SLV", "GDX", "GDXJ", "TLT", "HYG", "EEM", "EFA", "FXI", "KWEB", "USO", "UNG",
    "ARKK", "IBIT", "TQQQ", "SQQQ", "SOXL", "TNA",
]


# ---------------------------------------------------------------- candidates

def _number(text: Any) -> float:
    try:
        return float(str(text).replace("$", "").replace(",", ""))
    except ValueError:
        return 0.0


def _nasdaq(kind: str) -> list[dict[str, Any]]:
    r = requests.get(f"https://api.nasdaq.com/api/screener/{kind}", params={"tableonly": "true", "download": "true"},
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    body = r.json()["data"]
    return body["rows"] if "rows" in body else body["data"]["rows"]


def _occ_volume() -> tuple[pd.Series, str]:
    """Contracts traded per underlying in the most recent session OCC has published."""
    day = date.today()
    for _ in range(7):
        if day.weekday() < 5:
            params = {"reportDate": day.strftime("%Y%m%d"), "format": "csv", "volumeQueryType": "O",
                      "symbolType": "ALL", "symbol": "", "reportType": "D", "accountType": "ALL",
                      "productKind": "ALL", "porc": "BOTH"}
            r = requests.get("https://marketdata.theocc.com/volume-query", params=params, headers=HEADERS, timeout=60)
            if r.ok and r.text.startswith("quantity"):
                df = pd.read_csv(io.StringIO(r.text))
                if len(df):
                    # Every contract is counted once for each side of the trade.
                    return df.groupby("underlying")["quantity"].sum() / 2, day.isoformat()
        day -= timedelta(days=1)
    raise RuntimeError("OCC has no options volume for the last week")


def _candidates_occ(cfg: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], str]:
    volume, session = _occ_volume()
    stocks = {
        row["symbol"]: row for row in _nasdaq("stocks")
        if _number(row["lastsale"]) >= cfg["universe_min_price"]
        and _number(row["marketCap"]) >= cfg["universe_min_market_cap"]
    }
    etfs = {row["symbol"]: row for row in _nasdaq("etf")}
    found = {}
    for symbol, contracts in volume.sort_values(ascending=False).items():
        limit = cfg["universe_candidates"]
        if contracts < cfg["universe_min_option_volume"] or (limit and len(found) >= limit):
            break
        kind = "stock" if symbol in stocks else "etf" if symbol in etfs else None
        if kind:  # anything else is an index or an unlisted product
            row = stocks.get(symbol) or etfs[symbol]
            found[symbol] = {"name": row.get("name") or row.get("companyName") or symbol, "kind": kind,
                             "option_volume": float(contracts)}
    return found, f"OCC options volume for {session}, filtered with Nasdaq listings"


def _candidates_yahoo(cfg: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], str]:
    query = EquityQuery("and", [
        EquityQuery("eq", ["region", "us"]),
        EquityQuery("gt", ["avgdailyvol3m", cfg["universe_min_avg_volume"]]),
        EquityQuery("gt", ["intradayprice", cfg["universe_min_price"]]),
        EquityQuery("gt", ["intradaymarketcap", cfg["universe_min_market_cap"]]),
    ])
    found: dict[str, dict[str, Any]] = {}
    limit = cfg["universe_candidates"] or YAHOO_MAX_CANDIDATES
    while len(found) < limit:
        res = data.call(yf.screen, query, offset=len(found), size=250, sortField="avgdailyvol3m", sortAsc=False)
        quotes = res.get("quotes", [])
        for q in quotes:
            found[q["symbol"]] = {"name": q.get("shortName") or q["symbol"], "kind": "stock"}
        if not quotes or len(found) >= res.get("total", 0):
            break
    found = dict(list(found.items())[:limit])
    for etf in ETFS:
        found.setdefault(etf, {"name": etf, "kind": "etf"})
    return found, "Yahoo stock screener (share volume) plus a fixed ETF list"


# ---------------------------------------------------------------- liquidity

def _liquidity(symbol: str) -> dict[str, Any] | None:
    """Near-the-money open interest 7 to 60 days out, and the at-the-money spread on the
    expiry in that window with the most open interest (usually the monthly)."""
    chain, quote = data.cboe_chain(symbol)  # Cboe only: the Yahoo fallback would cost a request per expiry
    spot = quote["price"]
    if not spot or chain.empty:
        return None
    days = (pd.to_datetime(chain["expiration"]) - pd.Timestamp(date.today())).dt.days
    near = chain[days.between(7, 60) & chain["strike"].between(spot * 0.9, spot * 1.1)]
    if near.empty:
        return None
    by_expiry = near.groupby("expiration")["openInterest"].sum()
    expiry = str(by_expiry.idxmax())
    spreads, widths = [], []
    for kind in ("call", "put"):
        side = chain[(chain["expiration"] == expiry) & (chain["type"] == kind)]
        atm = side.assign(gap=(side["strike"] - spot).abs()).nsmallest(2, "gap")
        quoted = atm[(atm["bid"] > 0) & (atm["ask"] > 0)]
        mid = (quoted["bid"] + quoted["ask"]) / 2
        spreads += list(((quoted["ask"] - quoted["bid"]) / mid * 100).round(2))
        widths += list((quoted["ask"] - quoted["bid"]).round(2))
    return {"oi": float(by_expiry.sum()), "spread": float(np.median(spreads)) if spreads else None,
            "width": float(np.median(widths)) if widths else None,
            "price": float(spot), "expiry": expiry}


# ---------------------------------------------------------------- build

def load() -> dict[str, Any]:
    try:
        return json.loads(FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _market_settled() -> bool:
    """Option quotes are representative from an hour after the open until the close.

    Opening spreads are still wide (a 9:51 build dropped AAPL and AMZN), and after the
    close Cboe still shows quotes but market makers widen them.
    """
    now = datetime.now(NEW_YORK)
    return now.weekday() < 5 and clock(10, 30) <= now.time() <= clock(16, 0)


def _tight(liq: dict[str, Any], cfg: dict[str, Any]) -> bool:
    """A tight market: a small spread in percent, or a penny-wide one on a cheap option."""
    if liq["spread"] is None:
        return False
    return liq["spread"] <= cfg["universe_max_spread_pct"] or (liq.get("width") or 1) <= cfg["universe_max_spread_width"]


def build(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    """Return today's universe, rebuilding it once per day.

    Outside market hours the spread check is skipped and the list is marked provisional;
    it is rebuilt with the spread check at the first scan once quotes are representative.
    """
    cached = load()
    today = date.today().isoformat()
    live = _market_settled()
    if not force and cached.get("date") == today and (cached.get("live") or not live):
        return cached
    if not force and cached and not live:
        return cached  # market closed: the last list stands until quotes are representative

    try:
        candidates, source = _candidates_occ(cfg)
    except Exception as exc:
        candidates, source = _candidates_yahoo(cfg)
        source += f" (OCC or Nasdaq unavailable: {type(exc).__name__})"

    def check(symbol: str):
        try:
            return symbol, _liquidity(symbol)
        except Exception:
            return symbol, None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = dict(pool.map(check, candidates))

    passed = [
        (liq["oi"], symbol) for symbol, liq in results.items()
        if liq and liq["oi"] >= cfg["universe_min_option_oi"]
        and (not live or _tight(liq, cfg))
    ]
    keep = [s for _, s in sorted(passed, reverse=True)[: cfg["universe_size"]]]
    universe = {
        "date": today, "live": live, "built": datetime.now().isoformat(timespec="seconds"), "source": source,
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
            universe = load()  # every source unavailable: fall back to the last good list
        names += [t for t in universe.get("tickers", {}) if t not in names]
    return names


def meta(ticker: str) -> dict[str, Any]:
    """What the universe build learned about a ticker (kind, option volume), when known."""
    return load().get("tickers", {}).get(ticker, {})
