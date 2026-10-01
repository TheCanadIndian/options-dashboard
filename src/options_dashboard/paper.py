"""Paper-trading log: record each pick as a simulated trade and track how it does.

Entries fill at the ask and are valued at the bid, so results include the spread
you would really pay.
"""

from __future__ import annotations

import csv
import json
from datetime import date, datetime
from typing import Any

import pandas as pd

from . import data
from .config import HOME

PAPER_DIR = HOME / "paper"
TRADES_FILE = PAPER_DIR / "trades.json"
MARKS_FILE = PAPER_DIR / "marks.csv"

ENTRY_FIELDS = ["score", "trend", "delta", "theta", "vega", "iv", "iv_hv", "pop", "breakeven", "dte"]


def load() -> list[dict[str, Any]]:
    try:
        return json.loads(TRADES_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []


def _save(trades: list[dict[str, Any]]) -> None:
    PAPER_DIR.mkdir(exist_ok=True)
    TRADES_FILE.write_text(json.dumps(trades, indent=2), encoding="utf-8")


def _mark(trades: list[dict[str, Any]], now: str) -> int:
    """Re-price open trades from live quotes. Returns how many were updated."""
    open_trades = [t for t in trades if t["status"] == "open"]
    groups: dict[str, list[dict]] = {}
    for t in open_trades:
        if date.fromisoformat(t["expiration"]) < date.today():
            t["status"] = "expired"
            continue
        groups.setdefault(t["ticker"], []).append(t)
    if not data.market_open():
        return 0  # after-hours quotes are too wide to mark against

    marked, rows = 0, []
    for ticker, group in groups.items():
        try:
            quotes, spot = data.option_quotes(ticker)
        except Exception:
            continue  # try again next scan
        spot = spot or group[0]["last_spot"]
        for t in group:
            if t["symbol"] not in quotes.index:
                continue
            bid, ask = float(quotes.at[t["symbol"], "bid"]), float(quotes.at[t["symbol"], "ask"])
            if bid <= 0 or ask <= 0:
                continue
            t.update(last_bid=bid, last_ask=ask, last_spot=spot, last_time=now,
                     high_bid=max(t["high_bid"], bid), low_bid=min(t["low_bid"], bid))
            rows.append([now, t["symbol"], bid, ask, spot])
            marked += 1

    if rows:
        PAPER_DIR.mkdir(exist_ok=True)
        new_file = not MARKS_FILE.exists()
        with MARKS_FILE.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if new_file:
                writer.writerow(["time", "symbol", "bid", "ask", "spot"])
            writer.writerows(rows)
    return marked


def _open_new(trades: list[dict[str, Any]], contracts: pd.DataFrame, cfg: dict[str, Any], now: str) -> int:
    """Open one paper trade per ticker and direction for the best live-quoted pick."""
    if contracts.empty:
        return 0
    taken = {(t["ticker"], t["type"]) for t in trades}
    picks = contracts[(contracts["score"] >= cfg["paper_min_score"]) & ~contracts["stale"]]
    opened = 0
    for _, row in picks.drop_duplicates("ticker").iterrows():
        if (row["ticker"], row["type"]) in taken:
            continue
        trade = {
            "symbol": row["contractSymbol"], "ticker": row["ticker"], "type": row["type"],
            "strike": float(row["strike"]), "expiration": row["expiration"], "opened": now,
            "entry_price": float(row["ask"]), "entry_spot": float(row["spot"]),
            "last_bid": float(row["bid"]), "last_ask": float(row["ask"]),
            "high_bid": float(row["bid"]), "low_bid": float(row["bid"]),
            "last_spot": float(row["spot"]), "last_time": now, "status": "open",
        }
        trade.update({f: float(row[f]) for f in ENTRY_FIELDS})
        trades.append(trade)
        opened += 1
    return opened


def update(result: dict[str, Any], cfg: dict[str, Any]) -> tuple[int, int]:
    """Mark open trades and open new ones from a scan. Returns (opened, marked)."""
    trades = load()
    now = datetime.now().isoformat(timespec="seconds")
    marked = _mark(trades, now)
    opened = _open_new(trades, result["contracts"], cfg, now) if data.market_open() else 0
    _save(trades)
    return opened, marked


def results(trades: list[dict[str, Any]] | None = None) -> pd.DataFrame:
    """One row per paper trade with profit and loss at the latest bid."""
    df = pd.DataFrame(load() if trades is None else trades)
    if df.empty:
        return df
    df["cost"] = df["entry_price"] * 100
    df["value"] = df["last_bid"] * 100
    df["pnl"] = df["value"] - df["cost"]
    df["return"] = df["last_bid"] / df["entry_price"] - 1
    df["best"] = df["high_bid"] / df["entry_price"] - 1
    df["worst"] = df["low_bid"] / df["entry_price"] - 1
    df["stock_move"] = df["last_spot"] / df["entry_spot"] - 1
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def summary(df: pd.DataFrame, alert_min_score: float) -> pd.DataFrame:
    """Performance grouped by score band, to show whether higher scores did better."""
    if df.empty:
        return df
    band = df["score"].ge(alert_min_score).map(
        {True: f"Alert grade ({alert_min_score:.0f}+)", False: f"Below {alert_min_score:.0f}"}
    )
    out = df.groupby(band).agg(
        trades=("pnl", "size"),
        winners=("pnl", lambda s: int((s > 0).sum())),
        cost=("cost", "sum"),
        pnl=("pnl", "sum"),
        avg_return=("return", "mean"),
        median_return=("return", "median"),
    )
    out.loc["All"] = [
        len(df), int((df["pnl"] > 0).sum()), df["cost"].sum(), df["pnl"].sum(),
        df["return"].mean(), df["return"].median(),
    ]
    out["return_on_cost"] = out["pnl"] / out["cost"]
    return out.rename_axis("group").reset_index()
