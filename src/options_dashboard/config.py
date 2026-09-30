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
    "account_size": 1000.0,
    "risk_per_trade_pct": 15.0,  # max premium per contract as % of account
    "min_dte": 21,
    "max_dte": 60,
    "min_delta": 0.30,
    "max_delta": 0.70,
    "target_delta": 0.50,
    "min_open_interest": 100,
    "min_volume": 10,
    "max_spread_pct": 15.0,
    "min_trend_strength": 30,  # |trend score| needed before a ticker gets a direction
    "fallback_risk_free_rate": 0.04,
    "alert_min_score": 70,
    "alert_cooldown_hours": 24,
    "alert_max_per_scan": 5,
    "paper_min_score": 50,  # picks at or above this are logged as paper trades
    # Simulated portfolio (starts with account_size in cash)
    "sim_min_score": 70,  # buy picks at or above this score
    "sim_target_pct": 50.0,  # sell when the bid is this far above the entry price
    "sim_stop_pct": 40.0,  # sell when the bid is this far below the entry price
    "sim_exit_dte": 7,  # sell with this many days left, before time decay accelerates
    "sim_max_positions": 5,
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
