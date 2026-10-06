"""User settings: defaults merged with config.json in the project folder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HOME = Path.cwd()
CONFIG_PATH = HOME / "config.json"
STATE_DIR = HOME / ".state"

DEFAULTS: dict[str, Any] = {
    # Liquid underlyings, weighted toward lower share prices so premiums stay small.
    "watchlist": [
        "SPY", "QQQ", "IWM", "AAPL", "AMD", "NVDA", "TSLA", "PLTR",
        "SOFI", "F", "BAC", "INTC", "HOOD", "UBER", "PFE", "T", "XLF", "SLV",
    ],
    # Liquid universe, rebuilt each trading day (see universe.py); the watchlist is always scanned too
    "use_universe": True,
    "universe_size": None,  # no cap: every name that passes the liquidity tests (a number caps it)
    "universe_candidates": None,  # no cap on names checked (a number checks only the most traded)
    "universe_min_option_volume": 5_000,  # option contracts traded in the last session (OCC)
    "universe_min_avg_volume": 2_000_000,  # shares a day, 3-month average (Yahoo fallback only)
    "universe_min_price": 5.0,
    "universe_min_market_cap": 2_000_000_000,
    "universe_min_option_oi": 10000,  # contracts within 10% of the money, all expiries 7-60 days out
    "universe_max_spread_pct": 10.0,  # median at-the-money bid/ask spread
    "universe_max_spread_width": 0.05,  # or this many dollars wide, for cheap options
    "account_size": 1000.0,
    "risk_per_trade_pct": 15.0,  # max premium per contract as % of account
    "min_dte": 21,
    "max_dte": 60,
    "min_delta": 0.15,  # low enough to include out-of-the-money strikes
    "max_delta": 0.70,
    "min_open_interest": 100,
    "min_volume": 10,
    "max_spread_pct": 15.0,
    "min_trend_strength": 25,  # |composite score| needed before a ticker gets a direction
    "fallback_risk_free_rate": 0.04,
    "alert_min_score": 70,
    "alert_cooldown_hours": 24,
    "alert_max_per_scan": 5,
    # Earnings outlook (see earnings.py)
    "earnings_window_days": 21,  # analyse reports this many days ahead
    "earnings_min_open_interest": 50,
    "earnings_max_spread": 0.25,  # bid/ask spread as a share of the mid
    "earnings_max_plays": 4,
    "paper_min_score": 50,  # picks at or above this are logged as paper trades
    # Simulated portfolio (starts with account_size in cash)
    "sim_min_score": 70,  # buy picks at or above this score
    "sim_target_pct": 50.0,  # sell when the bid is this far above the entry price
    "sim_stop_pct": 40.0,  # sell when the bid is this far below the entry price
    "sim_exit_dte": 7,  # sell with this many days left, before time decay accelerates
    # New York times to re-check every open position against the current read: after the open
    # (spreads are widest in the first 15 minutes) and before the close
    "sim_review_times": ["09:45", "15:30"],
    "sim_max_positions": 5,
    # Market regime: the average combined read of these tickers. Beyond +/- the threshold, trades
    # against it need a read at least this strong on their own stock.
    "sim_regime_tickers": ["SPY", "QQQ"],
    "sim_regime_threshold": 15,
    "sim_counter_trend_min": 50,
    "sim_max_same_direction": 3,  # most positions betting the same way
    "sim_review_min_days": 3,  # trading days before a review can close a new position (stops still apply)
    "sim_max_entry_spread": 8.0,  # % bid/ask spread allowed when buying
    "sim_first_buy_time": "10:30",  # New York time: no buys while opening spreads are wide
    "sim_max_buys_per_scan": 1,
    "sim_max_buys_per_day": 2,
    "sim_commission": 0.65,  # per contract, each way
    "sim_reentry_days": 2,  # wait this long before re-buying a ticker after selling it
    "site_push": True,  # commit and push docs/ to GitHub after each scan
    "ntfy_topic": "",  # push notifications via ntfy.sh; empty = off
    "ntfy_server": "https://ntfy.sh",
    "discord_webhook": "",  # empty = off
}


def load() -> dict[str, Any]:
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(f"Could not read {CONFIG_PATH}: {exc}") from exc
    cfg["watchlist"] = clean_watchlist(cfg["watchlist"])
    return cfg


def save(cfg: dict[str, Any]) -> None:
    out = {k: cfg[k] for k in DEFAULTS if k in cfg}
    CONFIG_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")


def clean_watchlist(raw: Any) -> list[str]:
    if isinstance(raw, str):
        raw = raw.replace(",", " ").split()
    seen: list[str] = []
    for t in raw:
        t = str(t).strip().upper()
        if t and t not in seen:
            seen.append(t)
    return seen


def max_premium(cfg: dict[str, Any]) -> float:
    """Most you would pay for one contract, in dollars."""
    return float(cfg["account_size"]) * float(cfg["risk_per_trade_pct"]) / 100.0
