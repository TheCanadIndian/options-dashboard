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
import re
import threading
import time
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
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130 Safari/537.36"}

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
        if contracts < cfg["universe_min_option_volume"] or len(found) >= cfg["universe_candidates"]:
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
    while len(found) < cfg["universe_candidates"]:
        res = data.call(yf.screen, query, offset=len(found), size=250, sortField="avgdailyvol3m", sortAsc=False)
        quotes = res.get("quotes", [])
        for q in quotes:
            found[q["symbol"]] = {"name": q.get("shortName") or q["symbol"], "kind": "stock"}
        if not quotes or len(found) >= res.get("total", 0):
            break
    found = dict(list(found.items())[: cfg["universe_candidates"]])
    for etf in ETFS:
        found.setdefault(etf, {"name": etf, "kind": "etf"})
    return found, "Yahoo stock screener (share volume) plus a fixed ETF list"


# ---------------------------------------------------------------- liquidity

_OPTION = re.compile(r"^(?P<root>.+?)(?P<exp>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")
_cboe_lock = threading.Lock()
_cboe_next = 0.0
CBOE_PER_SECOND = 2.0  # four a second in bursts drew 429s


def _cboe_slot(backoff: float = 0.0) -> None:
    global _cboe_next
    with _cboe_lock:
        now = time.monotonic()
        if backoff:
            _cboe_next = max(_cboe_next, now + backoff)
        wait = max(0.0, _cboe_next - now)
        _cboe_next = max(now, _cboe_next) + 1.0 / CBOE_PER_SECOND
    time.sleep(wait)


def cboe_chain(symbol: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Every listed contract for a symbol from Cboe's delayed quotes, plus the underlying's quote."""
    for attempt in range(4):
        _cboe_slot()
        r = requests.get(f"https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json",
                         headers=HEADERS, timeout=30)
        if r.status_code != 429 or attempt == 3:
            break
        _cboe_slot(backoff=10.0 * 2**attempt)
    r.raise_for_status()
    body = r.json()["data"]
    df = pd.DataFrame(body["options"])
    parts = df["option"].str.extract(_OPTION)
    df["expiration"] = pd.to_datetime(parts["exp"], format="%y%m%d").dt.strftime("%Y-%m-%d")
    df["type"] = parts["cp"].map({"C": "call", "P": "put"})
    df["strike"] = parts["strike"].astype(float) / 1000
    quote = {k: body.get(k) for k in ("current_price", "close", "iv30", "iv30_change", "last_trade_time")}
    return df.dropna(subset=["type"]), quote


def _liquidity(symbol: str) -> dict[str, Any] | None:
    """Near-the-money open interest 7 to 60 days out, and the at-the-money spread on the
    expiry in that window with the most open interest (usually the monthly)."""
    chain, quote = cboe_chain(symbol)
    spot = quote["current_price"] or quote["close"]
    if not spot or chain.empty:
        return None
    days = (pd.to_datetime(chain["expiration"]) - pd.Timestamp(date.today())).dt.days
    near = chain[days.between(7, 60) & chain["strike"].between(spot * 0.9, spot * 1.1)]
    if near.empty:
        return None
    by_expiry = near.groupby("expiration")["open_interest"].sum()
    expiry = str(by_expiry.idxmax())
    spreads = []
    for kind in ("call", "put"):
        side = chain[(chain["expiration"] == expiry) & (chain["type"] == kind)]
        atm = side.assign(gap=(side["strike"] - spot).abs()).nsmallest(2, "gap")
        quoted = atm[(atm["bid"] > 0) & (atm["ask"] > 0)]
        mid = (quoted["bid"] + quoted["ask"]) / 2
        spreads += list(((quoted["ask"] - quoted["bid"]) / mid * 100).round(2))
    return {"oi": float(by_expiry.sum()), "spread": float(np.median(spreads)) if spreads else None,
            "price": float(spot), "expiry": expiry}


# ---------------------------------------------------------------- build

def load() -> dict[str, Any]:
    try:
        return json.loads(FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _market_settled() -> bool:
    """Option quotes are representative from 15 minutes after the open until the close.

    Outside that window Cboe still shows quotes, but market makers widen them.
    """
    now = datetime.now(NEW_YORK)
    return now.weekday() < 5 and clock(9, 45) <= now.time() <= clock(16, 0)


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
        and (not live or (liq["spread"] is not None and liq["spread"] <= cfg["universe_max_spread_pct"]))
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
