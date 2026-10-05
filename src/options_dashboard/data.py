"""Free market data, delayed roughly 15 minutes.

Option chains come from Cboe's delayed quotes (one request per ticker, all expiries),
falling back to Yahoo. Price bars, rates, dividends and earnings dates come from Yahoo.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from datetime import date, datetime, time as clock
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

from .config import STATE_DIR

# ETFs have no earnings calendar and yfinance logs a 404 for each; callers handle the None.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# Yahoo throttles bursts, so every request goes through one shared pacer.
REQUESTS_PER_SECOND = 3.0
_pace_lock = threading.Lock()
_next_slot = 0.0


def _wait_for_slot(backoff: float = 0.0) -> None:
    global _next_slot
    with _pace_lock:
        now = time.monotonic()
        if backoff:  # rate limited: hold every thread back, not just this one
            _next_slot = max(_next_slot, now + backoff)
        wait = max(0.0, _next_slot - now)
        _next_slot = max(now, _next_slot) + 1.0 / REQUESTS_PER_SECOND
    if wait:
        time.sleep(wait)


def call(fn, *args, **kwargs):
    """Make one Yahoo request, paced, retrying with backoff when rate limited."""
    for attempt in range(4):
        _wait_for_slot()
        try:
            return fn(*args, **kwargs)
        except YFRateLimitError:
            if attempt == 3:
                raise
            _wait_for_slot(backoff=15.0 * 2**attempt)


HISTORY_MAX_AGE = 3600  # seconds; callers refresh today's bar from intraday data in between


def history(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Daily bars, cached for up to an hour so large scans stay within Yahoo's request limits."""
    path = STATE_DIR / "history" / f"{ticker}.pkl"
    try:
        if time.time() - path.stat().st_mtime < HISTORY_MAX_AGE:
            return pd.read_pickle(path)
    except (OSError, ValueError, EOFError):
        pass
    df = call(yf.Ticker(ticker).history, period=period, interval="1d", auto_adjust=True, raise_errors=True)
    if df.empty or len(df) < 60:
        raise ValueError(f"{ticker}: not enough price history")
    df = df.dropna(subset=["Close"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)
    return df


INTRADAY_MAX_AGE = 20 * 60  # seconds: 30-minute bars, so every other 15-minute scan can reuse them


def intraday(ticker: str) -> pd.DataFrame:
    """A month of 30-minute regular-session bars, for the volume profile (cached for 20 minutes)."""
    path = STATE_DIR / "intraday" / f"{ticker}.pkl"
    try:
        if time.time() - path.stat().st_mtime < INTRADAY_MAX_AGE:
            return pd.read_pickle(path)
    except (OSError, ValueError, EOFError):
        pass
    df = call(yf.Ticker(ticker).history, period="1mo", interval="30m", auto_adjust=True, prepost=False,
              raise_errors=True).dropna(subset=["Close"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)
    return df


NEW_YORK = ZoneInfo("America/New_York")


def market_open(now: datetime | None = None) -> bool:
    """Regular US session, Monday to Friday. Exchange holidays are not checked."""
    now = now or datetime.now(NEW_YORK)
    return now.weekday() < 5 and clock(9, 30) <= now.time() <= clock(16, 0)


# ---------------------------------------------------------------- option chains (Cboe)

CBOE_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130 Safari/537.36"}
CBOE_PER_SECOND = 2.0  # four a second in bursts drew 429s
_OPTION = re.compile(r"^(?P<root>.+?)(?P<exp>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")
_cboe_lock = threading.Lock()
_cboe_next = 0.0


def _cboe_slot(backoff: float = 0.0) -> None:
    global _cboe_next
    with _cboe_lock:
        now = time.monotonic()
        if backoff:
            _cboe_next = max(_cboe_next, now + backoff)
        wait = max(0.0, _cboe_next - now)
        _cboe_next = max(now, _cboe_next) + 1.0 / CBOE_PER_SECOND
    time.sleep(wait)


CHAIN_MEMO_SECONDS = 300  # a scan and the position updates after it share one fetch per ticker
_chain_memo: dict[str, tuple[float, pd.DataFrame, dict[str, Any]]] = {}


def cboe_chain(ticker: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Every listed contract from Cboe's delayed quotes (one request), plus the underlying's quote."""
    memo = _chain_memo.get(ticker)
    if memo and time.monotonic() - memo[0] < CHAIN_MEMO_SECONDS:
        return memo[1].copy(), dict(memo[2])
    chain, quote = _fetch_cboe(ticker)
    _chain_memo[ticker] = (time.monotonic(), chain, quote)
    return chain.copy(), dict(quote)


def cached_quotes(ticker: str) -> pd.DataFrame | None:
    """Bid and ask by contract symbol from a chain fetched in the last few minutes, without a request."""
    memo = _chain_memo.get(ticker)
    if not memo or time.monotonic() - memo[0] >= CHAIN_MEMO_SECONDS:
        return None
    return memo[1].set_index("contractSymbol")[["bid", "ask"]]


def _fetch_cboe(ticker: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    for attempt in range(4):
        _cboe_slot()
        r = requests.get(f"https://cdn.cboe.com/api/global/delayed_quotes/options/{ticker}.json",
                         headers=CBOE_HEADERS, timeout=30)
        if r.status_code != 429 or attempt == 3:
            break
        _cboe_slot(backoff=10.0 * 2**attempt)
    r.raise_for_status()
    body = r.json()["data"]
    raw = pd.DataFrame(body["options"])
    parts = raw["option"].str.extract(_OPTION)
    df = pd.DataFrame({
        "contractSymbol": raw["option"],
        "type": parts["cp"].map({"C": "call", "P": "put"}),
        "strike": parts["strike"].astype(float) / 1000,
        "expiration": pd.to_datetime(parts["exp"], format="%y%m%d").dt.strftime("%Y-%m-%d"),
        "bid": raw["bid"], "ask": raw["ask"], "lastPrice": raw["last_trade_price"],
        "volume": raw["volume"], "openInterest": raw["open_interest"], "impliedVolatility": raw["iv"],
    }).dropna(subset=["type"])
    quote = {
        "price": body.get("current_price") or body.get("close"),
        "iv30": body.get("iv30"), "iv30_change": body.get("iv30_change"), "source": "Cboe",
    }
    return df, quote


def option_snapshot(ticker: str, max_dte: int = 60) -> tuple[pd.DataFrame, dict[str, Any]]:
    """All contracts expiring within `max_dte` days, with a `dte` column, and the underlying quote.

    Cboe first; if it cannot serve the ticker, Yahoo (one request per expiry, no IV change).
    """
    try:
        chain, quote = cboe_chain(ticker)
    except Exception:
        chain = option_chain(ticker, 0, max_dte)
        quote = {"price": None, "iv30": None, "iv30_change": None, "source": "Yahoo"}
        if chain.empty:
            return chain, quote
    chain["dte"] = (pd.to_datetime(chain["expiration"]) - pd.Timestamp(date.today())).dt.days
    chain = chain[chain["dte"].between(0, max_dte)].copy()
    for col in ("bid", "ask", "lastPrice", "volume", "openInterest", "impliedVolatility"):
        chain[col] = pd.to_numeric(chain[col], errors="coerce").fillna(0.0)
    return chain.reset_index(drop=True), quote


def option_quotes(ticker: str) -> tuple[pd.DataFrame, float | None]:
    """Current bid/ask for every contract on a ticker, indexed by contract symbol, and the stock price."""
    chain, quote = option_snapshot(ticker, max_dte=400)
    return chain.set_index("contractSymbol"), quote["price"]


def spot_price(ticker: str, fallback: float) -> float:
    try:
        last = float(call(lambda: yf.Ticker(ticker).fast_info["lastPrice"]))
        return last if last > 0 else fallback
    except Exception:
        return fallback


def risk_free_rate(fallback: float) -> float:
    """13-week T-bill yield as a decimal."""
    try:
        irx = call(yf.Ticker("^IRX").history, period="5d", raise_errors=True)["Close"].dropna()
        rate = float(irx.iloc[-1]) / 100.0
        return rate if 0 < rate < 0.2 else fallback
    except Exception:
        return fallback


def _once_a_day(ticker: str, name: str, fetch):
    """Return a per-ticker value fetched at most once per day. `fetch` must return JSON-safe data."""
    path = STATE_DIR / "daily" / f"{ticker}.json"
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        stored = {}
    today = date.today().isoformat()
    if name in stored and stored[name][0] == today:
        return stored[name][1]
    value = fetch()
    stored[name] = [today, value]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stored), encoding="utf-8")
    return value


def dividend_yield(ticker: str) -> float:
    return _once_a_day(ticker, "dividend_yield", lambda: _dividend_yield(ticker))


def next_earnings(ticker: str) -> date | None:
    found = _once_a_day(ticker, "earnings", lambda: (lambda d: d.isoformat() if d else None)(_next_earnings(ticker)))
    return date.fromisoformat(found) if found else None


def _dividend_yield(ticker: str) -> float:
    try:
        tk = yf.Ticker(ticker)
        divs = call(lambda: tk.dividends)
        if divs.empty:
            return 0.0
        cutoff = divs.index.max() - pd.Timedelta(days=365)
        price = float(call(lambda: tk.fast_info["lastPrice"]))
        return float(divs[divs.index > cutoff].sum()) / price if price > 0 else 0.0
    except Exception:
        return 0.0


def _next_earnings(ticker: str) -> date | None:
    try:
        cal =call(lambda: yf.Ticker(ticker).calendar)
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        upcoming = [d for d in dates or [] if d >= date.today()]
        return min(upcoming) if upcoming else None
    except Exception:
        return None


def option_chain(ticker: str, min_dte: int, max_dte: int, max_expiries: int | None = None) -> pd.DataFrame:
    """Calls and puts expiring inside the DTE window, one row per contract.

    With `max_expiries`, only that many expiries are fetched, spread evenly across the window.
    """
    tk = yf.Ticker(ticker)
    today = date.today()
    eligible = []
    for exp in call(lambda: tk.options):
        dte = (datetime.strptime(exp, "%Y-%m-%d").date() - today).days
        if min_dte <= dte <= max_dte:
            eligible.append((exp, dte))
    if max_expiries and len(eligible) > max_expiries:
        picks = sorted({round(i * (len(eligible) - 1) / (max_expiries - 1)) for i in range(max_expiries)})
        eligible = [eligible[i] for i in picks]
    frames = []
    for exp, dte in eligible:
        chain = call(tk.option_chain, exp)
        for kind, side in (("call", chain.calls), ("put", chain.puts)):
            if side.empty:
                continue
            side = side.copy()
            side["type"], side["expiration"], side["dte"] = kind, exp, dte
            frames.append(side)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    for col in ("bid", "ask", "lastPrice", "volume", "openInterest", "impliedVolatility"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df
