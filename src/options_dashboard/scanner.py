"""Combine trend, greeks and liquidity into a ranked list of affordable contracts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from . import config, data, greeks, structure, ta

WEIGHTS = {
    "trend": 30,  # strength of the composite market read in the contract's direction
    "liquidity": 15,  # tight spread and real open interest
    "breakeven": 15,  # move needed to break even vs the move the market expects
    "iv_value": 15,  # implied vol relative to recent realised vol
    "gamma": 10,  # negative dealer gamma and room before the wall
    "theta": 10,  # share of the premium lost per day
    "delta": 5,  # closeness to the target delta
}

GAMMA_MAX_DTE = 45  # expiries this close carry most of the gamma
FULL_CONVICTION = 70  # composite score that earns all of the direction points

COLUMNS = [
    "ticker", "type", "strike", "expiration", "dte", "score", "cost", "mid", "bid", "ask",
    "spread_pct", "delta", "gamma", "theta", "vega", "iv", "iv_hv", "pop", "breakeven",
    "breakeven_move", "expected_move", "theta_pct", "leverage", "openInterest", "volume",
    "spot", "trend", "bias", "stale", "earnings_before_expiry", "contractSymbol",
    *(f"pts_{k}" for k in WEIGHTS),
]


def _scale(x, best, worst):
    """Map x linearly onto 1 (at `best`) .. 0 (at `worst`)."""
    return np.clip((np.asarray(x, dtype=float) - worst) / (best - worst), 0.0, 1.0)


def _read_structure(ticker: str, ind: pd.DataFrame, snap: dict, spot: float, rate: float) -> dict:
    """Run every method; one that lacks data or fails returns None and is left out."""
    def attempt(read):
        try:
            return read()
        except Exception:
            return None

    atr = float(ind["atr"].iloc[-1])
    return {
        "auction": attempt(lambda: structure.auction(data.intraday(ticker), atr)),
        "gamma": attempt(lambda: structure.gamma(data.open_interest(ticker, GAMMA_MAX_DTE), spot, rate)),
        "wyckoff": attempt(lambda: structure.wyckoff(ind)),
        "vpa": attempt(lambda: structure.vpa(ind)),
        "trend": {"score": snap["trend"], "reasons": snap["reasons"]},
    }


def scan_ticker(ticker: str, cfg: dict[str, Any], rate: float) -> dict[str, Any]:
    ind = ta.add_indicators(data.history(ticker))
    snap = ta.snapshot(ind)
    spot = data.spot_price(ticker, snap["close"])
    reads = _read_structure(ticker, ind, snap, spot, rate)
    score = structure.composite(reads)
    direction = ta.bias(score, cfg["min_trend_strength"])
    levels = {k: v for p in reads.values() if p for k, v in p.items() if k not in ("score", "reasons")}
    result = {
        "ticker": ticker, "spot": spot, "bias": direction, "indicators": ind, **snap,
        "trend": score, "levels": levels,
        "methods": [
            {"key": k, "name": structure.METHOD_NAMES[k], "score": p["score"], "reasons": p["reasons"]}
            for k, p in reads.items() if p
        ],
        "reasons": [
            f"{r[0]} {structure.METHOD_NAMES[k]}: {r[2:]}" for k, p in reads.items() if p for r in p["reasons"]
        ],
    }
    result["contracts"] = pd.DataFrame(columns=COLUMNS)
    if direction == "neutral":
        return result

    chain = data.option_chain(ticker, cfg["min_dte"], cfg["max_dte"])
    if chain.empty:
        return result
    df = chain[chain["type"] == ("call" if direction == "bullish" else "put")].copy()

    # Quotes are zeroed outside market hours; fall back to the last trade and flag it.
    quoted = (df["bid"] > 0) & (df["ask"] > 0)
    df["stale"] = ~quoted
    df["mid"] = np.where(quoted, (df["bid"] + df["ask"]) / 2, df["lastPrice"])
    df["spread_pct"] = np.where(quoted, (df["ask"] - df["bid"]) / df["mid"] * 100, np.nan)
    df["cost"] = np.where(quoted, df["ask"], df["lastPrice"]) * 100

    df = df[
        (df["mid"] > 0.05)
        & (df["cost"] <= config.max_premium(cfg))
        & (df["strike"].between(spot * 0.7, spot * 1.3))
        & (df["openInterest"] >= cfg["min_open_interest"])
        & (df["volume"] >= cfg["min_volume"])
        & (df["stale"] | (df["spread_pct"] <= cfg["max_spread_pct"]))
    ].copy()
    if df.empty:
        return result

    q = data.dividend_yield(ticker)
    is_call = (df["type"] == "call").to_numpy()
    T = np.maximum(df["dte"].to_numpy(dtype=float), 0.5) / 365.0
    K = df["strike"].to_numpy(dtype=float)
    mid = df["mid"].to_numpy(dtype=float)

    solved = np.array(
        [greeks.implied_vol(p, spot, k, t, rate, q, c) for p, k, t, c in zip(mid, K, T, is_call)]
    )
    yahoo_iv = df["impliedVolatility"].to_numpy(dtype=float)
    iv = np.where(np.isnan(solved), yahoo_iv, solved)
    keep = iv > 0.03
    df, is_call, T, K, mid, iv = df[keep].copy(), is_call[keep], T[keep], K[keep], mid[keep], iv[keep]
    if df.empty:
        return result

    for name, values in greeks.greeks(spot, K, T, rate, iv, q, is_call).items():
        df[name] = values
    df["iv"] = iv
    df = df[df["delta"].abs().between(cfg["min_delta"], cfg["max_delta"])].copy()
    if df.empty:
        return result

    is_call = (df["type"] == "call").to_numpy()
    T = np.maximum(df["dte"].to_numpy(dtype=float), 0.5) / 365.0
    iv = df["iv"].to_numpy()
    df["breakeven"] = np.where(is_call, df["strike"] + df["mid"], df["strike"] - df["mid"])
    above = greeks.prob_above(spot, df["breakeven"].to_numpy(), T, rate, iv, q)
    df["pop"] = np.where(is_call, above, 1 - above)
    df["breakeven_move"] = (df["breakeven"] / spot - 1).abs()
    df["expected_move"] = iv * np.sqrt(T)
    df["theta_pct"] = df["theta"].abs() / df["mid"]
    df["leverage"] = df["delta"].abs() * spot / df["mid"]
    df["iv_hv"] = iv / snap["hv20"] if snap["hv20"] > 0 else np.nan

    spread = _scale(df["spread_pct"].fillna(cfg["max_spread_pct"] / 2), 0, cfg["max_spread_pct"])
    parts = {
        "trend": np.full(len(df), min(abs(score) / FULL_CONVICTION, 1.0)),
        "gamma": structure.gamma_fit(reads["gamma"], is_call, df["breakeven"].to_numpy(), spot),
        "liquidity": 0.6 * spread + 0.4 * _scale(np.log10(df["openInterest"] + 1), 3.5, 1.5),
        "breakeven": _scale(df["breakeven_move"] / df["expected_move"], 0.4, 1.5),
        "iv_value": _scale(df["iv_hv"].fillna(1.25), 0.9, 1.6),
        "theta": _scale(df["theta_pct"], 0.005, 0.04),
        "delta": _scale((df["delta"].abs() - cfg["target_delta"]).abs(), 0.0, 0.25),
    }
    for k in WEIGHTS:
        df[f"pts_{k}"] = np.round(WEIGHTS[k] * parts[k], 1)
    df["score"] = sum(WEIGHTS[k] * parts[k] for k in WEIGHTS).round(1)

    earnings = data.next_earnings(ticker)
    df["earnings_before_expiry"] = (
        pd.to_datetime(df["expiration"]).dt.date >= earnings if earnings else False
    )
    df["ticker"], df["spot"], df["trend"], df["bias"] = ticker, spot, score, direction
    result["earnings"] = earnings
    result["contracts"] = df[COLUMNS].sort_values("score", ascending=False).reset_index(drop=True)
    return result


def scan(cfg: dict[str, Any]) -> dict[str, Any]:
    """Scan the whole watchlist. Returns ranked contracts, per-ticker detail and errors."""
    rate = data.risk_free_rate(cfg["fallback_risk_free_rate"])
    tickers, errors = {}, {}

    def work(ticker: str):
        try:
            return ticker, scan_ticker(ticker, cfg, rate), None
        except Exception as exc:  # one bad ticker must not sink the scan
            return ticker, None, f"{type(exc).__name__}: {exc}"

    with ThreadPoolExecutor(max_workers=4) as pool:
        for ticker, res, err in pool.map(work, cfg["watchlist"]):
            if err:
                errors[ticker] = err
            else:
                tickers[ticker] = res

    frames = [r["contracts"] for r in tickers.values() if not r["contracts"].empty]
    contracts = (
        pd.concat(frames, ignore_index=True).sort_values("score", ascending=False)
        if frames
        else pd.DataFrame(columns=COLUMNS)
    )
    return {
        "contracts": contracts.reset_index(drop=True),
        "tickers": tickers,
        "errors": errors,
        "rate": rate,
        "scanned_at": datetime.now(),
    }


def best_per_ticker(contracts: pd.DataFrame) -> pd.DataFrame:
    """Highest scoring contract for each underlying."""
    return contracts.drop_duplicates("ticker", keep="first").reset_index(drop=True)
