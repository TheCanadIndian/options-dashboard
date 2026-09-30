"""Free market data from Yahoo Finance (delayed roughly 15 minutes)."""

from __future__ import annotations

import logging
from datetime import date, datetime

import pandas as pd
import yfinance as yf

# ETFs have no earnings calendar and yfinance logs a 404 for each; callers handle the None.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


def history(ticker: str, period: str = "2y") -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    if df.empty or len(df) < 60:
        raise ValueError(f"{ticker}: not enough price history")
    return df.dropna(subset=["Close"])


def spot_price(ticker: str, fallback: float) -> float:
    try:
        last = float(yf.Ticker(ticker).fast_info["lastPrice"])
        return last if last > 0 else fallback
    except Exception:
        return fallback


def risk_free_rate(fallback: float) -> float:
    """13-week T-bill yield as a decimal."""
    try:
        irx = yf.Ticker("^IRX").history(period="5d")["Close"].dropna()
        rate = float(irx.iloc[-1]) / 100.0
        return rate if 0 < rate < 0.2 else fallback
    except Exception:
        return fallback


def dividend_yield(ticker: str) -> float:
    try:
        divs = yf.Ticker(ticker).dividends
        if divs.empty:
            return 0.0
        cutoff = divs.index.max() - pd.Timedelta(days=365)
        price = float(yf.Ticker(ticker).fast_info["lastPrice"])
        return float(divs[divs.index > cutoff].sum()) / price if price > 0 else 0.0
    except Exception:
        return 0.0


def next_earnings(ticker: str) -> date | None:
    try:
        cal = yf.Ticker(ticker).calendar
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        upcoming = [d for d in dates or [] if d >= date.today()]
        return min(upcoming) if upcoming else None
    except Exception:
        return None


def quotes(ticker: str, expiration: str) -> pd.DataFrame:
    """Current bid/ask for one expiry, indexed by contract symbol."""
    chain = yf.Ticker(ticker).option_chain(expiration)
    df = pd.concat([chain.calls, chain.puts], ignore_index=True)
    for col in ("bid", "ask"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df.set_index("contractSymbol")


def option_chain(ticker: str, min_dte: int, max_dte: int) -> pd.DataFrame:
    """All calls and puts expiring inside the DTE window, one row per contract."""
    tk = yf.Ticker(ticker)
    today = date.today()
    frames = []
    for exp in tk.options:
        dte = (datetime.strptime(exp, "%Y-%m-%d").date() - today).days
        if not min_dte <= dte <= max_dte:
            continue
        chain = tk.option_chain(exp)
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
