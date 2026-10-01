"""Combine trend, greeks and liquidity into a ranked list of affordable contracts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from . import config, data, earnings, greeks, structure, ta, universe

WEIGHTS = {
    "trend": 30,  # strength of the composite market read in the contract's direction
    "liquidity": 15,  # tight spread and real open interest
    "breakeven": 15,  # move needed to break even vs the move the market expects
    "iv_value": 10,  # implied vol relative to recent realised vol
    "gamma": 10,  # negative dealer gamma and room before the wall
    "theta": 10,  # share of the premium lost per day
    "target": 10,  # breakeven inside the next structural level in the trade's direction
}

# Levels a move can run to, by name. A target must be at least one ATR away to count.
TARGET_LEVELS = {
    "vah": "value area high", "val": "value area low", "poc": "point of control",
    "call_wall": "call wall", "put_wall": "put wall", "flip": "gamma flip",
    "range_high": "Wyckoff range high", "range_low": "Wyckoff range low",
}

GAMMA_MAX_DTE = 45  # expiries this close carry most of the gamma
IV_CHANGE_MIN = 0.5  # vol points: a smaller move in 30-day IV is noise for the vanna read
FULL_CONVICTION = 70  # composite score that earns all of the direction points

COLUMNS = [
    "ticker", "type", "strike", "expiration", "dte", "score", "cost", "mid", "bid", "ask",
    "spread_pct", "delta", "gamma", "theta", "vega", "iv", "iv_hv", "pop", "breakeven",
    "breakeven_move", "expected_move", "theta_pct", "leverage", "openInterest", "volume",
    "spot", "trend", "bias", "stale", "earnings_before_expiry", "contractSymbol", "moneyness",
    "target_price", "target_label", "target_move",
    *(f"pts_{k}" for k in WEIGHTS),
]


def _scale(x, best, worst):
    """Map x linearly onto 1 (at `best`) .. 0 (at `worst`)."""
    return np.clip((np.asarray(x, dtype=float) - worst) / (best - worst), 0.0, 1.0)


def structural_target(levels: dict[str, Any], spot: float, atr: float, bullish: bool) -> tuple[float, str] | None:
    """The nearest level at least one ATR away in the trade's direction, with its name."""
    found = []
    for key, name in TARGET_LEVELS.items():
        value = levels.get(key)
        if value is None:
            continue
        gap = value - spot if bullish else spot - value
        if gap >= atr:
            found.append((gap, float(value), name))
    if not found:
        return None
    _, value, name = min(found)
    return value, name


def _read_structure(ind: pd.DataFrame, bars: pd.DataFrame | None, snap: dict, spot: float, rate: float,
                    chain: pd.DataFrame | None, quote: dict[str, Any]) -> dict:
    """Run every method; one that lacks data or fails returns None and is left out."""
    def attempt(read):
        try:
            return read()
        except Exception:
            return None

    def dealer_exposure():
        if chain is None or chain.empty:
            return None
        adv_dollars = float(ind["Volume"].rolling(20).mean().iloc[-1]) * spot
        # Vanna needs the direction of implied volatility: Cboe's change in 30-day IV today,
        # or when that is missing, the price move (volatility usually falls as stocks rise).
        change = quote.get("iv30_change")
        if change is not None and abs(change) >= IV_CHANGE_MIN:
            trend, note = (1 if change > 0 else -1), f"30-day IV {change:+.1f} points today"
        else:
            week = spot / float(ind["Close"].iloc[-6]) - 1
            trend = 0 if abs(week) < 0.01 else (-1 if week > 0 else 1)
            note = f"inferred from the stock's {week:+.1%} move this week"
        return structure.gamma(chain[chain["dte"] <= GAMMA_MAX_DTE], spot, rate, adv_dollars, trend, note)

    atr = float(ind["atr"].iloc[-1])
    return {
        "auction": attempt(lambda: structure.auction(bars, atr)) if bars is not None else None,
        "gamma": attempt(dealer_exposure),
        "wyckoff": attempt(lambda: structure.wyckoff(ind)),
        "vpa": attempt(lambda: structure.vpa(ind)),
        "trend": {"score": snap["trend"], "reasons": snap["reasons"]},
    }


def scan_ticker(ticker: str, cfg: dict[str, Any], rate: float) -> dict[str, Any]:
    daily = data.history(ticker)
    try:
        bars = data.intraday(ticker)
    except Exception:
        bars = None
    try:  # one chain serves dealer exposure and contract selection
        chain, quote = data.option_snapshot(ticker, max_dte=max(cfg["max_dte"], GAMMA_MAX_DTE))
    except Exception:
        chain, quote = None, {}
    spot = quote.get("price") or None  # the price the option quotes were taken against
    if bars is not None and len(bars):
        # Daily bars are cached, so bring today's bar up to date from the session so far.
        spot = spot or float(bars["Close"].iloc[-1])
        session = bars[bars.index.date == bars.index[-1].date()]
        if daily.index[-1].date() == session.index[-1].date():
            daily = daily.astype({"Volume": float})
            last = daily.index[-1]
            daily.loc[last, ["Open", "High", "Low", "Close", "Volume"]] = [
                float(session["Open"].iloc[0]), float(session["High"].max()), float(session["Low"].min()),
                spot, float(session["Volume"].sum()),
            ]
    ind = ta.add_indicators(daily)
    snap = ta.snapshot(ind)
    spot = spot or snap["close"]
    reads = _read_structure(ind, bars, snap, spot, rate, chain, quote)
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
    result["over_budget"] = False
    result["chain_source"] = quote.get("source")
    is_etf = universe.meta(ticker).get("kind") == "etf"
    report = None if is_etf else data.next_earnings(ticker)
    result["earnings"], result["earnings_outlook"] = report, None
    if report and 0 <= (report - date.today()).days <= cfg["earnings_window_days"]:
        try:
            result["earnings_outlook"] = earnings.outlook(ticker, report, None, ind, score, chain, spot, rate, cfg)
        except Exception as exc:  # one ticker's missing financials must not sink its scan
            result["earnings_error"] = f"{type(exc).__name__}: {exc}"
    if direction == "neutral" or chain is None or chain.empty:
        return result
    side = "call" if direction == "bullish" else "put"
    df = chain[(chain["type"] == side) & chain["dte"].between(cfg["min_dte"], cfg["max_dte"])].copy()

    # A contract without a two-sided quote falls back to its last trade and is flagged.
    quoted = (df["bid"] > 0) & (df["ask"] > 0)
    df["stale"] = ~quoted
    df["mid"] = np.where(quoted, (df["bid"] + df["ask"]) / 2, df["lastPrice"])
    df["spread_pct"] = np.where(quoted, (df["ask"] - df["bid"]) / df["mid"] * 100, np.nan)
    df["cost"] = np.where(quoted, df["ask"], df["lastPrice"]) * 100

    df = df[
        (df["mid"] > 0.05)
        & (df["strike"].between(spot * 0.7, spot * 1.3))
        & (df["openInterest"] >= cfg["min_open_interest"])
        & (df["volume"] >= cfg["min_volume"])
        & (df["stale"] | (df["spread_pct"] <= cfg["max_spread_pct"]))
    ]
    affordable = df[df["cost"] <= config.max_premium(cfg)].copy()
    result["over_budget"] = not df.empty and affordable.empty  # tradeable contracts exist, none fit
    df = affordable
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
    listed_iv = df["impliedVolatility"].to_numpy(dtype=float)  # the data source's own IV
    iv = np.where(np.isnan(solved), listed_iv, solved)
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
    # Positive moneyness is out of the money: how far the stock must move to reach the strike.
    df["moneyness"] = np.where(is_call, df["strike"] / spot - 1, 1 - df["strike"] / spot)
    target = structural_target(levels, spot, snap["atr_pct"] * spot, direction == "bullish")
    if target:
        df["target_price"], df["target_label"] = target
        df["target_move"] = abs(target[0] / spot - 1)
        # Full points when the breakeven sits inside the target, none when it needs twice the distance.
        reach = _scale(df["breakeven_move"] / df["target_move"], 1.0, 2.0)
    else:
        df["target_price"], df["target_label"], df["target_move"] = np.nan, None, np.nan
        reach = np.full(len(df), 0.5)  # no level to aim at: neither reward nor penalise

    spread = _scale(df["spread_pct"].fillna(cfg["max_spread_pct"] / 2), 0, cfg["max_spread_pct"])
    parts = {
        "trend": np.full(len(df), min(abs(score) / FULL_CONVICTION, 1.0)),
        "gamma": structure.gamma_fit(reads["gamma"], is_call, df["breakeven"].to_numpy(), spot),
        "liquidity": 0.6 * spread + 0.4 * _scale(np.log10(df["openInterest"] + 1), 3.5, 1.5),
        "breakeven": _scale(df["breakeven_move"] / df["expected_move"], 0.4, 1.5),
        "iv_value": _scale(df["iv_hv"].fillna(1.25), 0.9, 1.6),
        "theta": _scale(df["theta_pct"], 0.005, 0.04),
        "target": reach,
    }
    for k in WEIGHTS:
        df[f"pts_{k}"] = np.round(WEIGHTS[k] * parts[k], 1)
    df["score"] = sum(WEIGHTS[k] * parts[k] for k in WEIGHTS).round(1)

    df["earnings_before_expiry"] = (
        pd.to_datetime(df["expiration"]).dt.date >= report if report else False
    )
    df["ticker"], df["spot"], df["trend"], df["bias"] = ticker, spot, score, direction
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

    with ThreadPoolExecutor(max_workers=8) as pool:
        for ticker, res, err in pool.map(work, universe.tickers(cfg)):
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
