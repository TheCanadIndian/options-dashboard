"""Anchored VWAPs: volume-weighted average price since the quarters and major volatility events.

Anchors:
    this quarter's first session, and last quarter's
    the most recent volatility event in the last six months, and the largest one
A volatility event is a session whose move was at least 2.5 times the stock's typical daily
move (20-day standard deviation) or whose volume was at least 2.5 times its 20-day average;
earnings gaps and news shocks show up here. Each AVWAP is the average price paid since the
anchor, so it marks where the holders since that point break even: support when price holds
above it, resistance when below. Built from daily bars (typical price x volume).
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

EVENT_MOVE = 2.5  # times the 20-day standard deviation of daily returns
EVENT_VOLUME = 2.5  # times 20-day average volume
EVENT_LOOKBACK = 126  # sessions, about six months
WEIGHTS = {"avwap_quarter": 30, "avwap_event": 30, "avwap_prev_quarter": 20, "avwap_major": 20}
LABELS = {"avwap_quarter": "this quarter's AVWAP", "avwap_prev_quarter": "last quarter's AVWAP",
          "avwap_event": "AVWAP from the latest volatility event", "avwap_major": "AVWAP from the largest volatility event"}


def _vwap_from(daily: pd.DataFrame, start: pd.Timestamp) -> pd.Series:
    d = daily[daily.index >= start]
    typical = (d["High"] + d["Low"] + d["Close"]) / 3
    return (typical * d["Volume"]).cumsum() / d["Volume"].cumsum().replace(0, np.nan)


def anchors(daily: pd.DataFrame) -> dict[str, tuple[pd.Timestamp, str]]:
    """Anchor session and a description for each AVWAP."""
    idx = daily.index
    today = idx[-1]
    q_start_month = 3 * ((today.month - 1) // 3) + 1
    q_start = pd.Timestamp(today.year, q_start_month, 1, tz=idx.tz)
    prev = q_start - pd.DateOffset(months=3)
    found: dict[str, tuple[pd.Timestamp, str]] = {}
    for key, start in (("avwap_quarter", q_start), ("avwap_prev_quarter", prev)):
        after = idx[idx >= start]
        if len(after):
            found[key] = (after[0], f"Q{(after[0].month - 1) // 3 + 1} start")

    ret = daily["Close"].pct_change()
    size = (ret.abs() / ret.rolling(20).std().shift()).fillna(0)
    volume = (daily["Volume"] / daily["Volume"].rolling(20).mean().shift()).fillna(0)
    recent = daily.index[-EVENT_LOOKBACK:-1]  # an event needs at least one session after it
    events = [d for d in recent if size[d] >= EVENT_MOVE or volume[d] >= EVENT_VOLUME]
    if events:
        def describe(d):
            return f"{d:%b} {d.day} ({ret[d]:+.1%} on {volume[d]:.1f}x volume)"
        found["avwap_event"] = (events[-1], describe(events[-1]))
        biggest = max(events, key=lambda d: size[d] * max(volume[d], 1))
        if biggest != events[-1]:
            found["avwap_major"] = (biggest, describe(biggest))
    return found


def read(daily: pd.DataFrame, spot: float) -> dict[str, Any] | None:
    """Whether price is holding above or below each anchored VWAP, and recent crosses."""
    if len(daily) < 60:
        return None
    score, reasons, levels = 0.0, [], {}
    for key, (start, label) in anchors(daily).items():
        line = _vwap_from(daily, start).dropna()
        if line.empty:
            continue
        value = float(line.iloc[-1])
        levels[key] = value
        above = spot >= value
        gap = spot / value - 1
        weight = WEIGHTS[key]
        score += weight if above else -weight
        crossed = ""
        closes = daily["Close"].loc[line.index].iloc[-4:-1]
        if len(closes) and ((closes < line.iloc[-4:-1]).all() if above else (closes > line.iloc[-4:-1]).all()):
            crossed = " after reclaiming it this week" if above else " after losing it this week"
            score += 10 if above else -10
        reasons.append(f"{'+' if above else '-'} {'Above' if above else 'Below'} {LABELS[key]} "
                       f"(${value:.2f}, anchored {label}) by {abs(gap):.1%}{crossed}")
    if not levels:
        return None
    return {"score": float(np.clip(score, -100, 100)), "reasons": reasons, **levels}
