"""Technical indicators on daily bars and a single trend score per ticker."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close, high, low = out["Close"], out["High"], out["Low"]

    out["ema20"] = close.ewm(span=20, adjust=False).mean()
    out["ema50"] = close.ewm(span=50, adjust=False).mean()
    out["sma200"] = close.rolling(200).mean()

    # Wilder RSI
    change = close.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    out["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    out["rsi"] = out["rsi"].fillna(100.0).where(change.notna().cumsum() >= 14)

    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    out["macd"] = macd
    out["macd_signal"] = macd.ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    prev_close = close.shift()
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    out["atr"] = true_range.ewm(alpha=1 / 14, adjust=False).mean()

    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    out["bb_upper"], out["bb_lower"] = mid + 2 * std, mid - 2 * std

    out["vol_ratio"] = out["Volume"] / out["Volume"].rolling(20).mean()
    out["hv20"] = np.log(close / prev_close).rolling(20).std() * np.sqrt(252)
    return out


def snapshot(ind: pd.DataFrame) -> dict:
    """Latest readings plus a trend score from -100 (bearish) to +100 (bullish)."""
    last = ind.iloc[-1]
    close = float(last["Close"])
    score = 0.0
    reasons: list[str] = []

    def vote(condition: bool, weight: float, bull: str, bear: str) -> None:
        nonlocal score
        score += weight if condition else -weight
        reasons.append(("+ " + bull) if condition else ("- " + bear))

    vote(close > last["ema20"], 15, "Price above 20 EMA", "Price below 20 EMA")
    vote(last["ema20"] > last["ema50"], 20, "20 EMA above 50 EMA", "20 EMA below 50 EMA")
    if pd.notna(last["sma200"]):
        vote(close > last["sma200"], 20, "Above 200-day average", "Below 200-day average")
    vote(last["macd_hist"] > 0, 15, "MACD above signal", "MACD below signal")
    if len(ind) > 4:
        vote(
            last["macd_hist"] > ind["macd_hist"].iloc[-4],
            10,
            "MACD momentum rising",
            "MACD momentum falling",
        )
    rsi = float(last["rsi"])
    if rsi >= 55:
        score += 10
        reasons.append(f"+ RSI {rsi:.0f} shows buying strength")
    elif rsi <= 45:
        score -= 10
        reasons.append(f"- RSI {rsi:.0f} shows selling pressure")
    if len(ind) > 21:
        ret20 = close / float(ind["Close"].iloc[-21]) - 1
        vote(ret20 > 0, 10, f"Up {ret20:.1%} over 20 days", f"Down {abs(ret20):.1%} over 20 days")

    # A stretched move is a worse entry in the direction of the trend.
    if rsi > 75 and score > 0:
        score -= 15
        reasons.append(f"! RSI {rsi:.0f} is overbought, late entry risk")
    elif rsi < 25 and score < 0:
        score += 15
        reasons.append(f"! RSI {rsi:.0f} is oversold, bounce risk")

    return {
        "close": close,
        "trend": float(np.clip(score, -100, 100)),
        "rsi": rsi,
        "atr_pct": float(last["atr"] / close),
        "hv20": float(last["hv20"]),
        "vol_ratio": float(last["vol_ratio"]),
        "reasons": reasons,
    }


def bias(trend: float, min_strength: float) -> str:
    if trend >= min_strength:
        return "bullish"
    if trend <= -min_strength:
        return "bearish"
    return "neutral"
