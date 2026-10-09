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
    "max_dte": 90,  # scanner window; each simulated account narrows it with sim_min_dte / sim_max_dte
    "short_min_dte": 0,  # short-dated pool, for the weeklies and gamma day-trade accounts only
    "short_max_dte": 14,
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
    "sim_min_dte": 21,  # expiries this account buys
    "sim_max_dte": 60,
    "sim_min_moneyness": None,  # e.g. 0.02 buys only contracts at least 2% out of the money
    "sim_use_regime": True,  # market direction filter and regime position limits
    "sim_skip_avoided_sectors": False,  # skip sectors the market regime says to avoid
    # Market regime: the average combined read of these tickers. Beyond +/- the threshold, trades
    # against it need a read at least this strong on their own stock.
    "sim_regime_tickers": ["SPY", "QQQ"],
    "sim_regime_threshold": 15,
    "sim_counter_trend_min": 50,
    "sim_max_same_direction": 3,  # most positions betting the same way
    # When the market regime's historical odds of a 5% drop within a month reach this level, or VIX
    # is above VIX3M, the account holds fewer positions.
    "sim_risk_drop_odds": 0.20,
    "sim_max_positions_risky": 3,
    "sim_max_same_direction_risky": 2,
    "sim_review_min_days": 3,  # trading days before a review can close a new position (stops still apply)
    "sim_max_entry_spread": 8.0,  # % bid/ask spread allowed when buying
    "sim_first_buy_time": "10:30",  # New York time: no buys while opening spreads are wide
    "sim_max_buys_per_scan": 1,
    "sim_max_buys_per_day": 2,
    "sim_commission": 0.65,  # per contract, each way
    "sim_reentry_days": 2,  # wait this long before re-buying a ticker after selling it
    "sim_pool": "swing",  # "swing" buys from the scanner's picks, "short" from the 0-14 day pool
    "sim_flatten_time": None,  # New York time to sell everything still open (day trading), e.g. "15:45"
    "sim_last_buy_time": None,  # New York time after which no new buys are made
    "sim_require_short_gamma": False,  # only buy when dealers are net short gamma near the flip or a wall
    "sim_gamma_distance": 0.02,  # how near the flip or a wall counts, as a share of the price
    "site_push": True,  # commit and push docs/ to GitHub after each scan
    # Simulated accounts, each starting with account_size and trading the same scans under its own
    # rules. All of them only buy to open (long calls and puts) and sell to close. "core" is the main
    # account shown on the Portfolio page; the rest only change the settings listed in "rules".
    # "group" decides which chart an account appears in on the Strategies page. Rules may set
    # "factor_weights" (contract score, out of 100) and "method_weights" (market read, summing to 1);
    # each contract is then re-scored from its raw readings with those weights.
    "accounts": [
        {"id": "core", "name": "Core", "summary": "The current rules", "group": "both", "rules": {}},
        {"id": "long_dated", "group": "rules", "name": "Longer-dated", "summary": "45 to 90 days to expiry, for slower time decay",
         "rules": {"sim_min_dte": 45, "sim_max_dte": 90}},
        {"id": "wide_exits", "group": "rules", "name": "Let winners run", "summary": "Exits at +100% or -50%",
         "rules": {"sim_target_pct": 100.0, "sim_stop_pct": 50.0}},
        {"id": "quick_exits", "group": "rules", "name": "Quick exits", "summary": "Exits at +30% or -25%",
         "rules": {"sim_target_pct": 30.0, "sim_stop_pct": 25.0}},
        {"id": "high_conviction", "group": "rules", "name": "High conviction", "summary": "Only buys at a score of 80 or more",
         "rules": {"sim_min_score": 80}},
        {"id": "otm", "group": "rules", "name": "Out of the money", "summary": "Only contracts at least 2% out of the money",
         "rules": {"sim_min_moneyness": 0.02}},
        {"id": "no_regime", "group": "rules", "name": "No market filter", "summary": "Ignores market direction and regime limits",
         "rules": {"sim_use_regime": False}},
        {"id": "skip_weak_sectors", "group": "rules", "name": "Skip weak sectors", "summary": "Never buys in sectors the regime says to avoid",
         "rules": {"sim_skip_avoided_sectors": True}},
        {"id": "weeklies", "group": "short", "name": "Weeklies",
         "summary": "7 to 14 days to expiry; exits at +40% or -30%, sells with 2 days left, reviewed after 1 day",
         "rules": {"sim_pool": "short", "sim_min_dte": 7, "sim_max_dte": 14, "sim_target_pct": 40.0,
                   "sim_stop_pct": 30.0, "sim_exit_dte": 2, "sim_review_min_days": 1,
                   "factor_weights": {"trend": 30, "liquidity": 15, "breakeven": 15, "iv_value": 10, "gamma": 15,
                                      "theta": 5, "target": 10}, "sim_min_score": 60}},
        {"id": "gamma_day", "group": "short", "name": "Gamma day trades",
         "summary": "0 to 7 days; only when dealers are short gamma within 2% of the flip or a wall; "
                    "exits at +30% or -25% and sells everything by 15:45",
         "rules": {"sim_pool": "short", "sim_min_dte": 0, "sim_max_dte": 7, "sim_target_pct": 30.0,
                   "sim_stop_pct": 25.0, "sim_exit_dte": -1, "sim_review_min_days": 0,
                   "sim_flatten_time": "15:45", "sim_last_buy_time": "14:30", "sim_require_short_gamma": True,
                   "sim_reentry_days": 0, "sim_max_entry_spread": 10.0,
                   "factor_weights": {"trend": 30, "liquidity": 20, "breakeven": 15, "iv_value": 5, "gamma": 25,
                                      "theta": 0, "target": 5}, "sim_min_score": 60}},
        {"id": "w_flows", "name": "Dealer-flow weighted", "group": "weights",
         "summary": "Market read led by dealer gamma, vanna, charm and DEX (35%)",
         "rules": {"method_weights": {"auction": 0.15, "gamma": 0.35, "wyckoff": 0.08, "vpa": 0.08, "avwap": 0.10,
                                      "vol": 0.12, "sector": 0.05, "trend": 0.04, "seasonal": 0.03}}},
        {"id": "w_structure", "name": "Price-structure weighted", "group": "weights",
         "summary": "Market read led by auction, Wyckoff, VPA and AVWAPs",
         "rules": {"method_weights": {"auction": 0.28, "gamma": 0.10, "wyckoff": 0.22, "vpa": 0.15, "avwap": 0.15,
                                      "vol": 0.04, "sector": 0.03, "trend": 0.02, "seasonal": 0.01}}},
        {"id": "w_momentum", "name": "Momentum and rotation weighted", "group": "weights",
         "summary": "Market read led by trend, AVWAPs and sector rotation",
         "rules": {"method_weights": {"auction": 0.12, "gamma": 0.08, "wyckoff": 0.08, "vpa": 0.08, "avwap": 0.18,
                                      "vol": 0.04, "sector": 0.17, "trend": 0.20, "seasonal": 0.05}}},
        {"id": "w_cheap_vol", "name": "Cheap-volatility weighted", "group": "weights",
         "summary": "Contract score led by option price against volatility, time decay and breakeven",
         "rules": {"factor_weights": {"trend": 20, "liquidity": 10, "breakeven": 15, "iv_value": 25, "gamma": 10,
                                      "theta": 15, "target": 5}}},
        {"id": "w_read_heavy", "name": "Market-read weighted", "group": "weights",
         "summary": "Contract score led by the strength of the market read (45%); buys at 60+ since its scale runs lower",
         "rules": {"factor_weights": {"trend": 45, "liquidity": 15, "breakeven": 10, "iv_value": 10, "gamma": 5,
                                      "theta": 10, "target": 5}, "sim_min_score": 60}},
    ],
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
