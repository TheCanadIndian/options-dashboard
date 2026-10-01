"""Free market data from Yahoo Finance (delayed roughly 15 minutes)."""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import date, datetime

import pandas as pd
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


def intraday(ticker: str) -> pd.DataFrame:
    """A month of 30-minute regular-session bars, for the volume profile."""
    df = call(yf.Ticker(ticker).history, period="1mo", interval="30m", auto_adjust=True, prepost=False,
              raise_errors=True)
    return df.dropna(subset=["Close"])


def open_interest(ticker: str, max_dte: int) -> pd.DataFrame:
    """Open interest for every near-dated contract. It only changes overnight, so it is cached per day."""
    path = STATE_DIR / "oi" / f"{ticker}.pkl"
    today = date.today().isoformat()
    if path.exists():
        try:
            cached = pd.read_pickle(path)
            if cached["date"] == today:
                return cached["chain"]
        except Exception:
            pass  # unreadable cache: refetch
    chain = option_chain(ticker, 0, max_dte)
    if not chain.empty:
        chain = chain[["type", "strike", "expiration", "openInterest", "impliedVolatility"]]
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle({"date": today, "chain": chain}, path)
    return chain


def iv_change(ticker: str, atm_iv: float) -> float | None:
    """Record today's first at-the-money IV and return its change from the previous session.

    None until there is an earlier reading from the last week to compare against.
    """
    path = STATE_DIR / "iv" / f"{ticker}.json"
    try:
        history = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        history = {}
    today = date.today().isoformat()
    if today not in history:
        history[today] = atm_iv
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(sorted(history.items())[-30:])), encoding="utf-8")
    earlier = [d for d in history if d < today and (date.today() - date.fromisoformat(d)).days <= 7]
    return history[today] - history[max(earlier)] if earlier else None


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


def quotes(ticker: str, expiration: str) -> pd.DataFrame:
    """Current bid/ask for one expiry, indexed by contract symbol."""
    tk = yf.Ticker(ticker)
    call(lambda: tk.options)  # the expiry list is its own request
    chain = call(tk.option_chain, expiration)
    df = pd.concat([chain.calls, chain.puts], ignore_index=True)
    for col in ("bid", "ask"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df.set_index("contractSymbol")


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
