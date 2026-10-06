"""Simulated portfolio: buys alert-grade picks and sells on target, stop, time limit
or a failed opening or end-of-day review.

Buys fill at the ask and sells at the bid, with a commission each way, and exits
are only checked when a scan runs. A price that gaps through a stop between scans
is sold at whatever the bid is then, as it would be in a real account.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time as clock, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import config, data, learning
from .config import HOME
from .scanner import WEIGHTS

FILE = HOME / "paper" / "portfolio.json"  # the core account
ACCOUNTS_DIR = HOME / "paper" / "accounts"  # every other account
NEW_YORK = ZoneInfo("America/New_York")


def accounts(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return cfg["accounts"]


def account_cfg(cfg: dict[str, Any], account: str) -> dict[str, Any]:
    """Settings for one account: the shared settings with its own rules on top."""
    spec = next((a for a in cfg["accounts"] if a["id"] == account), None)
    return {**cfg, **(spec["rules"] if spec else {})}


def _path(account: str):
    return FILE if account == "core" else ACCOUNTS_DIR / f"{account}.json"


def load(cfg: dict[str, Any], account: str = "core") -> dict[str, Any]:
    try:
        return json.loads(_path(account).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cash = float(cfg["account_size"])
        return {
            "start_cash": cash, "cash": cash,
            "started": datetime.now().isoformat(timespec="seconds"),
            "updated": None, "positions": [], "closed": [], "equity": [], "log": [],
        }


def _save(state: dict[str, Any], account: str = "core") -> None:
    path = _path(account)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=1), encoding="utf-8")


def equity(state: dict[str, Any]) -> float:
    return state["cash"] + sum(p["last_bid"] * 100 for p in state["positions"])


def _target_note(row: pd.Series, ticker: str, call: bool) -> str:
    """Where the strike sits, and whether the breakeven is inside the next structural level."""
    away = row["moneyness"]
    where = (f"{away:.1%} out of the money" if away > 0.005 else
             f"{abs(away):.1%} in the money" if away < -0.005 else "at the money")
    if pd.isna(row["target_price"]):
        return f"The ${row['strike']:g} strike is {where}; no structural level is far enough away to aim at."
    inside = row["breakeven_move"] <= row["target_move"]
    return (f"The ${row['strike']:g} strike is {where}. The next target is the {row['target_label']} at "
            f"${row['target_price']:.2f}, {row['target_move']:.1%} away, and breakeven "
            f"{'sits inside it' if inside else 'needs a move beyond it'}.")


def _contract_notes(row: pd.Series, cfg: dict[str, Any]) -> list[str]:
    """Plain-language reasons this particular contract was chosen."""
    ticker, call = row["ticker"], row["type"] == "call"
    expiry = datetime.strptime(row["expiration"], "%Y-%m-%d").strftime("%b %d").replace(" 0", " ")
    ratio = row["iv_hv"]
    if pd.isna(ratio):
        value = f"Implied volatility is {row['iv']:.0%}."
    else:
        verdict = "cheap" if ratio < 1 else "fairly priced" if ratio < 1.25 else "expensive"
        value = (f"Implied volatility is {row['iv']:.0%} against {row['iv'] / ratio:.0%} realised over "
                 f"the last 20 days, so the option looks {verdict} (ratio {ratio:.2f}).")
    notes = [
        f"Costs ${row['cost']:.0f}, which is the most this trade can lose and fits the "
        f"${config.max_premium(cfg):.0f} per-trade limit.",
        f"Delta {row['delta']:+.2f}: the contract gains about ${abs(row['delta']) * 100:.0f} for each $1 "
        f"{ticker} {'rises' if call else 'falls'}.",
        f"{ticker} must be {'above' if call else 'below'} ${row['breakeven']:.2f} on {expiry} to profit at "
        f"expiry, a {row['breakeven_move']:.1%} move. Options are pricing in a move of about "
        f"{row['expected_move']:.1%} either way.",
        value,
        f"Time decay costs about ${abs(row['theta']) * 100:.2f} a day ({row['theta_pct']:.1%} of the premium).",
        f"Bid/ask spread is {row['spread_pct']:.1f}% with {int(row['openInterest']):,} contracts of open "
        f"interest, so it can be sold without giving much away.",
        _target_note(row, ticker, call),
        f"Model chance of finishing past breakeven at expiry is {row['pop']:.0%}. The plan is to sell "
        f"earlier, at the target or the stop.",
    ]
    if row["earnings_before_expiry"]:
        notes.append("Risk: earnings fall before expiry, and implied volatility usually drops sharply afterwards.")
    return notes


def _close(state: dict[str, Any], pos: dict[str, Any], reason: str, now: str, cfg: dict[str, Any]) -> None:
    proceeds = pos["last_bid"] * 100 - cfg["sim_commission"]
    state["cash"] += proceeds
    pos.update(
        exit_price=pos["last_bid"], exit_time=now, exit_reason=reason, exit_spot=pos["last_spot"],
        pnl=round(proceeds - pos["cost"], 2), ret=proceeds / pos["cost"] - 1,
    )
    state["positions"].remove(pos)
    state["closed"].append(pos)
    state["log"].append({
        "time": now, "action": "SELL", "symbol": pos["symbol"], "label": pos["label"],
        "price": pos["last_bid"], "note": f"{reason}, P/L ${pos['pnl']:+.2f} ({pos['ret']:+.0%})",
    })


def _manage(state: dict[str, Any], now: str, cfg: dict[str, Any]) -> None:
    """Re-price open positions and sell any that hit the stop, target or time limit.

    Only called during market hours: after the close quotes widen enough to fake a stop.
    """
    groups: dict[str, list[dict]] = {}
    for pos in state["positions"]:
        groups.setdefault(pos["ticker"], []).append(pos)

    for ticker, group in groups.items():
        try:
            quotes, spot = data.option_quotes(ticker)
        except Exception:
            quotes, spot = None, None
        spot = spot or group[0]["last_spot"]
        for pos in group:
            expiration = pos["expiration"]
            live = False
            if quotes is not None and pos["symbol"] in quotes.index:
                bid = float(quotes.at[pos["symbol"], "bid"])
                ask = float(quotes.at[pos["symbol"], "ask"])
                if bid > 0 and ask > 0:
                    live = True
                    pos.update(last_bid=bid, last_ask=ask, last_spot=spot, last_time=now,
                               high_bid=max(pos["high_bid"], bid), low_bid=min(pos["low_bid"], bid))
                    pos["marks"].append([now, bid])
            dte = (date.fromisoformat(expiration) - date.today()).days
            if dte < 0:
                _close(state, pos, "Expired", now, cfg)
            elif not live:
                continue  # never sell on a stale quote
            elif pos["last_bid"] <= pos["stop_price"]:
                _close(state, pos, "Stop hit", now, cfg)
            elif pos["last_bid"] >= pos["target_price"]:
                _close(state, pos, "Target hit", now, cfg)
            elif dte <= cfg["sim_exit_dte"]:
                _close(state, pos, "Time limit", now, cfg)




def _review_label(slot: str) -> str:
    return "Opening review" if slot < "12:00" else "End-of-day review"


def _review_due(state: dict[str, Any], cfg: dict[str, Any]) -> str | None:
    """The review slot due at this scan, if any.

    Each slot runs once per trading day at the first scan on or after its time. If several are
    outstanding (the PC was off at the earlier one), only the latest runs and the rest are skipped.
    """
    now = datetime.now(NEW_YORK)
    if now.weekday() >= 5 or now.time() > clock(16, 0):
        return None
    today = now.date().isoformat()
    done = state.setdefault("reviews_done", {})
    if "last_review" in state:  # state saved before there were several review times
        done.setdefault("15:30", state.pop("last_review"))
    due = [slot for slot in sorted(cfg["sim_review_times"])
           if clock(*map(int, slot.split(":"))) <= now.time() and done.get(slot) != today]
    if not due:
        return None
    for slot in due:
        done[slot] = today
    return due[-1]


def _review(state: dict[str, Any], result: dict[str, Any], now: str, cfg: dict[str, Any],
            label: str = "End-of-day review") -> None:
    """Check each open position against the current market read and sell the ones that no longer hold up.

    Hold while the read points the trade's way or has only drifted to neutral; sell only when it
    has reversed. Positions younger than sim_review_min_days trading days are left to their stops,
    since each early exit pays the bid/ask spread again.
    """
    for pos in list(state["positions"]):
        info = result["tickers"].get(pos["ticker"])
        if info is None:
            continue  # ticker failed to scan or left the watchlist; try again at the next review
        if np.busday_count(pos["opened"][:10], now[:10]) < cfg["sim_review_min_days"]:
            continue
        want = "bullish" if pos["type"] == "call" else "bearish"
        read, bias = info["trend"], info["bias"]
        ret = pos["last_bid"] / pos["entry_price"] - 1
        was = {m["key"]: m["score"] for m in pos.get("methods", [])}
        changed = [
            f"{m['name']} {was[m['key']]:+.0f} to {m['score']:+.0f}"
            for m in info["methods"]
            if m["key"] in was and (m["score"] * was[m["key"]] < 0 or abs(m["score"] - was[m["key"]]) >= 40)
        ]
        # Positions opened before the combined read existed were scored on a different scale.
        entry = f" (it was {pos['trend']:+.0f} at entry)" if was else ""
        summary = f"the read is {bias} at {read:+.0f}{entry}"
        sell = None
        if bias == want:
            verdict, note = "Hold", f"Still viable: {summary}."
        elif bias == "neutral":
            verdict = "Hold"
            note = (f"The read has weakened: {summary}. It has not reversed, so the trade stays open "
                    f"({ret:+.0%}) under its stop and target.")
        else:
            verdict, sell = "Sell", f"{label}: read reversed"
            note = f"No longer viable: {summary}, against the {pos['type']}."

        pos.setdefault("reviews", []).append(
            {"time": now, "label": label, "verdict": verdict, "read": read, "note": note, "changed": changed}
        )
        state["log"].append({
            "time": now, "action": "REVIEW", "symbol": pos["symbol"], "label": pos["label"],
            "price": pos["last_bid"], "note": f"{label}: {verdict}. {note}",
        })
        if sell:
            _close(state, pos, sell, now, cfg)


def market_regime(result: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """The broad market's direction: the average combined read of the regime tickers."""
    reads = {t: result["tickers"][t]["trend"] for t in cfg["sim_regime_tickers"] if t in result["tickers"]}
    if not reads:
        return {"score": None, "label": "unknown", "reads": {}}
    score = float(np.mean(list(reads.values())))
    label = ("bullish" if score >= cfg["sim_regime_threshold"] else
             "bearish" if score <= -cfg["sim_regime_threshold"] else "neutral")
    return {"score": score, "label": label, "reads": reads}


def entry_block(row, info: dict[str, Any], regime: dict[str, Any], cfg: dict[str, Any],
                avoided: set[str] | None = None) -> str | None:
    """Why the account would not buy this contract on its own merits, or None if it could."""
    if row["cost"] > config.max_premium(cfg):
        return "over budget"
    if not (row["spread_pct"] <= cfg["sim_max_entry_spread"]):  # also catches a missing spread
        return "spread too wide"
    if not cfg["sim_min_dte"] <= row["dte"] <= cfg["sim_max_dte"]:
        return "outside its expiry range"
    if cfg.get("sim_min_moneyness") is not None and row["moneyness"] < cfg["sim_min_moneyness"]:
        return "not far enough out of the money"
    if cfg.get("sim_skip_avoided_sectors") and avoided and info.get("levels", {}).get("sector") in avoided:
        return "out-of-favour sector"
    if cfg.get("sim_use_regime", True):
        against = ((regime["label"] == "bullish" and row["type"] == "put")
                   or (regime["label"] == "bearish" and row["type"] == "call"))
        if against and abs(info["trend"]) < cfg["sim_counter_trend_min"]:
            return "against the market"
    return None


def avoided_sectors(result: dict[str, Any]) -> set[str]:
    return {s["sector"] for s in (result.get("regime_model") or {}).get("avoid", [])}


def exposure_limits(result: dict[str, Any], cfg: dict[str, Any]) -> tuple[int, int, str | None]:
    """Most positions in total and in one direction, and why they are reduced (None if they are not)."""
    if not cfg.get("sim_use_regime", True):
        return cfg["sim_max_positions"], cfg["sim_max_same_direction"], None
    model = result.get("regime_model") or {}
    odds = (model.get("odds") or {}).get("drop_odds")
    inverted = any("VIX is above VIX3M" in w for w in model.get("warnings", []))
    if inverted or (odds is not None and odds >= cfg["sim_risk_drop_odds"]):
        reason = ("VIX is above VIX3M" if inverted else
                  f"{model['state']} has seen a 5%+ drop within a month {odds:.0%} of the time")
        return cfg["sim_max_positions_risky"], cfg["sim_max_same_direction_risky"], reason
    return cfg["sim_max_positions"], cfg["sim_max_same_direction"], None


def buys_today(state: dict[str, Any], today: str) -> int:
    return sum(1 for e in state["log"] if e["action"] == "BUY" and e["time"][:10] == today)


def _buy(state: dict[str, Any], result: dict[str, Any], now: str, cfg: dict[str, Any]) -> None:
    contracts = result["contracts"]
    if contracts.empty:
        return
    ny = datetime.now(NEW_YORK)
    if ny.time() < clock(*map(int, cfg["sim_first_buy_time"].split(":"))):
        return  # opening spreads are wide
    room_today = cfg["sim_max_buys_per_day"] - buys_today(state, now[:10])
    allowed = min(cfg["sim_max_buys_per_scan"], room_today)
    if allowed <= 0:
        return
    regime = market_regime(result, cfg)
    state["regime"] = {**regime, "time": now}
    max_positions, max_same_way, _ = exposure_limits(result, cfg)
    held = {p["ticker"] for p in state["positions"]}
    cutoff = (datetime.fromisoformat(now) - timedelta(days=cfg["sim_reentry_days"])).isoformat()
    resting = {c["ticker"] for c in state["closed"] if c["exit_time"] > cutoff}
    picks = contracts[(contracts["score"] >= cfg["sim_min_score"]) & ~contracts["stale"]]
    avoided = avoided_sectors(result)
    bought = 0

    for _, row in picks.iterrows():  # already sorted best first
        if len(state["positions"]) >= max_positions or bought >= allowed:
            break
        ticker = row["ticker"]
        cost = float(row["ask"]) * 100 + cfg["sim_commission"]
        if entry_block(row, result["tickers"][ticker], regime, cfg, avoided):
            continue
        same_way = sum(1 for p in state["positions"] if p["type"] == row["type"])
        if same_way >= max_same_way:
            continue
        if ticker in held or ticker in resting or cost > state["cash"]:
            continue
        bought += 1
        entry = float(row["ask"])
        info = result["tickers"][ticker]
        expiry = datetime.strptime(row["expiration"], "%Y-%m-%d")
        pos = {
            "symbol": row["contractSymbol"], "ticker": ticker, "type": row["type"],
            "strike": float(row["strike"]), "expiration": row["expiration"],
            "label": f"{ticker} ${row['strike']:g} {row['type']} {expiry:%b} {expiry.day}",
            "opened": now, "entry_price": entry, "cost": round(cost, 2),
            "entry_spot": float(row["spot"]),
            "target_price": round(entry * (1 + cfg["sim_target_pct"] / 100), 2),
            "stop_price": round(entry * (1 - cfg["sim_stop_pct"] / 100), 2),
            "time_exit": (expiry - timedelta(days=cfg["sim_exit_dte"])).strftime("%Y-%m-%d"),
            "last_bid": float(row["bid"]), "last_ask": entry, "last_spot": float(row["spot"]),
            "high_bid": float(row["bid"]), "low_bid": float(row["bid"]), "last_time": now,
            "marks": [[now, float(row["bid"])]],
            "score": float(row["score"]), "trend": float(row["trend"]), "bias": row["bias"],
            "points": {k: float(row[f"pts_{k}"]) for k in WEIGHTS}, "weights": learning.active("factors", WEIGHTS),
            "methods": info["methods"], "levels": info["levels"],
            "trend_reasons": info["reasons"], "contract_notes": _contract_notes(row, cfg),
            "delta": float(row["delta"]), "theta": float(row["theta"]), "iv": float(row["iv"]),
            "earnings_before_expiry": bool(row["earnings_before_expiry"]),
        }
        state["cash"] -= cost
        state["positions"].append(pos)
        held.add(ticker)
        state["log"].append({
            "time": now, "action": "BUY", "symbol": pos["symbol"], "label": pos["label"],
            "price": entry, "note": f"score {row['score']:.0f}, {row['bias']} trend",
        })


def run_all(result: dict[str, Any], cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Apply one scan to every account. Returns each account's state by id."""
    return {a["id"]: update(result, account_cfg(cfg, a["id"]), a["id"]) for a in accounts(cfg)}


def update(result: dict[str, Any], cfg: dict[str, Any], account: str = "core") -> dict[str, Any]:
    """Apply one scan to one account: manage exits, then look for new buys. `cfg` is that account's settings."""
    state = load(cfg, account)
    if not data.market_open():
        return state  # option quotes outside the session are too wide to trade or value against
    now = datetime.now().isoformat(timespec="seconds")
    _manage(state, now, cfg)
    slot = _review_due(state, cfg)
    if slot:
        _review(state, result, now, cfg, _review_label(slot))
    _buy(state, result, now, cfg)
    state["updated"] = now
    state["equity"].append([now, round(equity(state), 2)])
    _save(state, account)
    return state
