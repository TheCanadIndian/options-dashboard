"""Journal of everything the scanner flags, and what each contract went on to do.

Each scan in market hours records:
  - every flagged contract the first time it appears each day, with the raw 0-1 reading on
    each scoring factor (so it can be re-scored under any weights later), the score under the
    weights in use and under the original control weights, and its ticker's method scores
  - the out-of-the-money plays on the Earnings page, as contracts to follow the same way
  - each ticker's method scores for the day, and each earnings prediction until its report

Each journaled contract is then followed for five trading days: the highest, lowest and last
bid seen each day. After that it is moved to a monthly outcome file, where the scorecard
grades it on dollars of profit (see scorecard.py).
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

import numpy as np

from . import data
from .config import HOME

DIR = HOME / "journal"
OPEN_FILE = DIR / "open.json"
EARNINGS_FILE = DIR / "earnings.json"
HORIZON = 5  # trading days each contract is followed


def _read(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write(path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _num(x: Any) -> float | None:
    try:
        x = float(x)
        return None if np.isnan(x) else x
    except (TypeError, ValueError):
        return None


def _trading_days(start: str, end: date) -> int:
    return int(np.busday_count(date.fromisoformat(start), end))


def _entry(row, info: dict[str, Any], now: str, source: str) -> dict[str, Any]:
    return {
        "source": source, "entry_date": now[:10], "entry_time": now, "ticker": row["ticker"],
        "type": row["type"], "strike": float(row["strike"]), "expiration": row["expiration"],
        "dte": int(row["dte"]), "ask": float(row["ask"]), "bid": float(row["bid"]), "spot": float(row["spot"]),
        "score": _num(row.get("score")), "score_control": _num(row.get("score_control")),
        "moneyness": _num(row.get("moneyness")),
        "factors": {k[2:]: _num(v) for k, v in row.items() if k.startswith("f_")},
        "read": _num(info.get("trend")), "bias": info.get("bias"),
        "methods": {m["key"]: m["score"] for m in info.get("methods", [])},
        "days": {},
    }


def record_scan(result: dict[str, Any], cfg: dict[str, Any]) -> dict[str, int]:
    """Add today's new candidates, update the ones being followed, and retire matured ones."""
    if not data.market_open():
        return {"added": 0, "updated": 0, "matured": 0}
    now = datetime.now().isoformat(timespec="seconds")
    today = date.today()
    tracked: dict[str, dict] = _read(OPEN_FILE, {})
    added = updated = matured = 0

    # Method scores for every ticker today (the latest scan of the day wins).
    reads_path = DIR / "reads" / f"{today.isoformat()}.json"
    reads = _read(reads_path, {})
    for ticker, info in result["tickers"].items():
        reads[ticker] = {"time": now, "spot": info["spot"], "read": info["trend"], "bias": info["bias"],
                         "methods": {m["key"]: m["score"] for m in info["methods"]}}
    _write(reads_path, reads)

    # New candidates: scanner picks, and earnings plays.
    def add(symbol: str, entry: dict[str, Any]) -> None:
        nonlocal added
        key = f"{symbol}|{entry['source']}"
        if key in tracked or entry["ask"] <= 0 or entry["bid"] <= 0:
            return
        tracked[key] = {"symbol": symbol, **entry}
        added += 1

    contracts = result["contracts"]
    for _, row in contracts[~contracts["stale"]].iterrows():
        add(row["contractSymbol"], _entry(row, result["tickers"][row["ticker"]], now, "scanner"))
    for ticker, info in result["tickers"].items():
        outlook = info.get("earnings_outlook")
        for play in (outlook or {}).get("plays", []):
            if len(play["legs"]) != 1:
                continue  # strangles are left out: one contract per journal entry
            leg = play["legs"][0]
            row = {"ticker": ticker, "type": leg["type"], "strike": leg["strike"], "expiration": outlook["expiry"],
                   "dte": (date.fromisoformat(outlook["expiry"]) - today).days, "ask": leg["ask"], "bid": leg["bid"],
                   "spot": outlook["spot"], "score": outlook["score"]}
            entry = _entry(row, info, now, "earnings")
            entry.update(report_date=outlook["date"], timing=outlook["timing"], lean=outlook["lean"])
            add(leg["symbol"], entry)

    # Follow every open entry: high, low and latest bid for each trading day after entry.
    for key, entry in list(tracked.items()):
        day = _trading_days(entry["entry_date"], today)
        expired = date.fromisoformat(entry["expiration"]) < today
        if day > HORIZON or expired:
            _retire(entry)
            del tracked[key]
            matured += 1
            continue
        if day == 0:
            continue
        quotes = data.cached_quotes(entry["ticker"])
        if quotes is None or entry["symbol"] not in quotes.index:
            continue
        bid = float(quotes.at[entry["symbol"], "bid"])
        if bid < 0:
            continue
        mark = entry["days"].setdefault(str(day), {"hi": bid, "lo": bid, "close": bid})
        mark.update(hi=max(mark["hi"], bid), lo=min(mark["lo"], bid), close=bid)
        updated += 1
    _write(OPEN_FILE, tracked)
    record_earnings(result)
    return {"added": added, "updated": updated, "matured": matured}


def _retire(entry: dict[str, Any]) -> None:
    path = DIR / "outcomes" / f"{entry['entry_date'][:7]}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def record_earnings(result: dict[str, Any]) -> None:
    """Keep the latest prediction for each upcoming report until the report is out."""
    stored = _read(EARNINGS_FILE, {})
    today = date.today().isoformat()
    for ticker, info in result["tickers"].items():
        o = info.get("earnings_outlook")
        if not o:
            continue
        key = f"{ticker}|{o['date']}"
        if o["date"] < today or (o["date"] == today and o["timing"] != "after close"):
            continue  # the report is out: keep the last prediction made before it
        stored[key] = {
            "ticker": ticker, "date": o["date"], "timing": o["timing"], "recorded": today, "lean": o["lean"],
            "score": o["score"], "beat_probability": o["beat_probability"], "implied_move": o["implied_move"],
            "historical_move": o["historical_move"], "spot": o["spot"],
            "factors": {f["name"]: f["points"] for f in o["factors"]},
        }
    _write(EARNINGS_FILE, stored)


def outcomes() -> list[dict[str, Any]]:
    """Every matured journal entry."""
    rows = []
    for path in sorted((DIR / "outcomes").glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    return rows


def open_entries() -> dict[str, Any]:
    return _read(OPEN_FILE, {})


def earnings_predictions() -> dict[str, Any]:
    return _read(EARNINGS_FILE, {})
