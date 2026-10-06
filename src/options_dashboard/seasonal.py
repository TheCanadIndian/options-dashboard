"""Seasonality: how a stock has done from this point in the calendar in past years.

From up to 15 years of daily closes: for each past year, the return over the next 20 trading
days starting from the first session on or after today's month and day, and the return over
the whole current calendar month. Recomputed once a week (the answer barely changes daily).
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from . import data
from .config import STATE_DIR

DIR = STATE_DIR / "seasonal"
FORWARD_DAYS = 20
MIN_YEARS = 7  # fewer years than this is too thin to call a seasonal pattern


def _compute(ticker: str, today: date) -> dict[str, Any] | None:
    hist = data.call(yf.Ticker(ticker).history, period="15y", interval="1d", auto_adjust=True, raise_errors=True)
    closes = hist["Close"].dropna()
    if len(closes) < 500:
        return None
    days = closes.index.tz_localize(None).normalize() if closes.index.tz is not None else closes.index
    closes.index = days
    forward, month = [], []
    for year in range(today.year - 15, today.year):
        try:
            start = pd.Timestamp(year, today.month, min(today.day, 28))
        except ValueError:
            continue
        i = closes.index.searchsorted(start)
        if 0 < i and i + FORWARD_DAYS < len(closes):
            forward.append(float(closes.iloc[i + FORWARD_DAYS] / closes.iloc[i] - 1))
        in_month = closes[(closes.index.year == year) & (closes.index.month == today.month)]
        before = closes[closes.index < pd.Timestamp(year, today.month, 1)]
        if len(in_month) > 10 and len(before):
            month.append(float(in_month.iloc[-1] / before.iloc[-1] - 1))
    if len(forward) < MIN_YEARS:
        return None
    return {
        "years": len(forward), "forward_mean": float(np.mean(forward)), "forward_median": float(np.median(forward)),
        "forward_hit": float(np.mean([r > 0 for r in forward])),
        "month_years": len(month), "month_mean": float(np.mean(month)) if month else None,
        "month_hit": float(np.mean([r > 0 for r in month])) if month else None,
    }


def stats(ticker: str) -> dict[str, Any] | None:
    """Seasonal statistics for today's date, cached for the week."""
    path = DIR / f"{ticker}.json"
    today = date.today()
    week = today.strftime("%G-W%V")
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("week") == week:
            return cached["stats"]
    except (OSError, json.JSONDecodeError):
        pass
    found = _compute(ticker, today)
    DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"week": week, "stats": found}), encoding="utf-8")
    return found


def read(s: dict[str, Any] | None) -> dict[str, Any] | None:
    """Direction score: a consistent seasonal tendency counts, a coin flip does not."""
    if not s:
        return None
    month_name = date.today().strftime("%B")
    hit, mean = s["forward_hit"], s["forward_mean"]
    score = 0.0
    if abs(mean) >= 0.005:
        score = float(np.clip((hit - 0.5) / 0.2 * 25, -40, 40))
        if (score > 0) != (mean > 0):
            score = 0.0  # mostly up but losing on average (or the reverse): no clear pattern
    rose = round(hit * s["years"])
    line = (f"rose in {rose} of the last {s['years']} years over the next {FORWARD_DAYS} trading days from this date "
            f"(average {mean:+.1%}, median {s['forward_median']:+.1%})")
    if score >= 5:
        reasons = [f"+ Seasonally strong: {line}"]
    elif score <= -5:
        reasons = [f"- Seasonally weak: {line}"]
    else:
        reasons = [f"= No clear seasonal pattern: {line}"]
    if s.get("month_mean") is not None:
        up = round(s["month_hit"] * s["month_years"])
        reasons.append(f"= {month_name} has averaged {s['month_mean']:+.1%} (up in {up} of {s['month_years']} years)")
    return {"score": score, "reasons": reasons, "seasonal_hit": hit, "seasonal_mean": mean}
