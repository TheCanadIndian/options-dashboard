"""Push alerts to a phone (ntfy.sh) or Discord, without repeating the same contract."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests

from .config import STATE_DIR

STATE_FILE = STATE_DIR / "alerts.json"


def enabled(cfg: dict[str, Any]) -> bool:
    return bool(cfg["ntfy_topic"] or cfg["discord_webhook"])


def describe(row: pd.Series) -> tuple[str, str]:
    title = (
        f"{row['ticker']} ${row['strike']:g} {row['type']} {row['expiration']} "
        f"- score {row['score']:.0f}"
    )
    lines = [
        f"{row['bias'].capitalize()} trend ({row['trend']:+.0f}), stock at ${row['spot']:.2f}",
        f"Cost ${row['cost']:.0f} | delta {row['delta']:.2f} | theta -${abs(row['theta']) * 100:.2f}/day",
        f"IV {row['iv']:.0%} | breakeven ${row['breakeven']:.2f} ({row['breakeven_move']:.1%} away)",
        f"Chance of profit at expiry ~{row['pop']:.0%} | {row['dte']} days left",
    ]
    if row["earnings_before_expiry"]:
        lines.append("Earnings fall before expiry")
    return title, "\n".join(lines)


def send(cfg: dict[str, Any], title: str, body: str) -> list[str]:
    """Deliver to every configured channel. Returns a list of error messages."""
    errors = []
    if cfg["ntfy_topic"]:
        try:
            requests.post(
                f"{cfg['ntfy_server'].rstrip('/')}/{cfg['ntfy_topic']}",
                data=body.encode("utf-8"),
                headers={"Title": title.encode("ascii", "replace").decode(), "Tags": "chart_with_upwards_trend"},
                timeout=10,
            ).raise_for_status()
        except requests.RequestException as exc:
            errors.append(f"ntfy: {exc}")
    if cfg["discord_webhook"]:
        try:
            requests.post(
                cfg["discord_webhook"], json={"content": f"**{title}**\n{body}"}, timeout=10
            ).raise_for_status()
        except requests.RequestException as exc:
            errors.append(f"discord: {exc}")
    return errors


def _load_state() -> dict[str, str]:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def notify(contracts: pd.DataFrame, cfg: dict[str, Any]) -> tuple[int, list[str]]:
    """Alert on fresh, live-quoted contracts above the score threshold.

    Returns (number sent, errors).
    """
    if not enabled(cfg) or contracts.empty:
        return 0, []
    now = datetime.now()
    cooldown = timedelta(hours=cfg["alert_cooldown_hours"])
    state = {
        k: v for k, v in _load_state().items() if now - datetime.fromisoformat(v) < cooldown
    }

    picks = contracts[(contracts["score"] >= cfg["alert_min_score"]) & ~contracts["stale"]]
    # One alert per ticker and direction per cooldown, for its best contract only.
    picks = picks.drop_duplicates("ticker")
    keys = picks["ticker"] + ":" + picks["type"]
    picks = picks[~keys.isin(state.keys())].head(cfg["alert_max_per_scan"])

    sent, errors = 0, []
    for _, row in picks.iterrows():
        failed = send(cfg, *describe(row))
        if failed:
            errors.extend(failed)
        else:
            state[f"{row['ticker']}:{row['type']}"] = now.isoformat()
            sent += 1

    STATE_DIR.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return sent, errors
