"""Sector rotation: which sectors are gaining or losing strength against the market.

Each SPDR sector ETF's price ratio to SPY gives its relative strength (RS). Over one and three
months, and with RS momentum (this month's change in RS against last month's), each sector
falls in one of four quadrants, as on a relative rotation graph:

    leading    strong and getting stronger      improving  weak but getting stronger
    weakening  strong but losing strength       lagging    weak and getting weaker

A stock's read adds its sector's quadrant and its own strength against that sector.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import requests

from . import data
from .config import STATE_DIR

SECTOR_ETFS = {
    "Technology": "XLK", "Finance": "XLF", "Energy": "XLE", "Health Care": "XLV", "Industrials": "XLI",
    "Consumer Discretionary": "XLY", "Consumer Staples": "XLP", "Utilities": "XLU", "Basic Materials": "XLB",
    "Real Estate": "XLRE", "Telecommunications": "XLC",
}
QUADRANT_POINTS = {"leading": 30, "improving": 15, "weakening": -15, "lagging": -30}
SECTOR_FILE = STATE_DIR / "sectors.json"


def sector_of() -> dict[str, str]:
    """Each listed stock's sector from Nasdaq's screener, fetched once a day."""
    today = date.today().isoformat()
    try:
        cached = json.loads(SECTOR_FILE.read_text(encoding="utf-8"))
        if cached.get("date") == today:
            return cached["sectors"]
    except (OSError, json.JSONDecodeError):
        cached = {}
    try:
        r = requests.get("https://api.nasdaq.com/api/screener/stocks", params={"tableonly": "true", "download": "true"},
                         headers=data.CBOE_HEADERS, timeout=30)
        r.raise_for_status()
        rows = r.json()["data"]["rows"]
        sectors = {row["symbol"]: row["sector"] for row in rows if row.get("sector") in SECTOR_ETFS}
    except Exception:
        return cached.get("sectors", {})  # keep yesterday's map if Nasdaq is unavailable
    STATE_DIR.mkdir(exist_ok=True)
    SECTOR_FILE.write_text(json.dumps({"date": today, "sectors": sectors}), encoding="utf-8")
    return sectors


def rotation() -> dict[str, dict[str, Any]]:
    """Relative strength, momentum and quadrant for every sector, keyed by sector name."""
    try:
        spy = data.history("SPY")["Close"]
    except Exception:
        return {}
    out = {}
    for sector, etf in SECTOR_ETFS.items():
        try:
            px = data.history(etf)["Close"]
        except Exception:
            continue
        ratio = (px / spy).dropna()
        if len(ratio) < 70:
            continue
        rs1 = float(ratio.iloc[-1] / ratio.iloc[-22] - 1)
        rs3 = float(ratio.iloc[-1] / ratio.iloc[-64] - 1)
        prior = float(ratio.iloc[-22] / ratio.iloc[-43] - 1)
        momentum = rs1 - prior
        quadrant = ("leading" if rs3 >= 0 and momentum >= 0 else "weakening" if rs3 >= 0 else
                    "improving" if momentum >= 0 else "lagging")
        out[sector] = {"etf": etf, "rs_1m": rs1, "rs_3m": rs3, "momentum": momentum, "quadrant": quadrant,
                       "return_1m": float(px.iloc[-1] / px.iloc[-22] - 1)}
    ranked = sorted(out, key=lambda s: out[s]["rs_1m"], reverse=True)
    for i, s in enumerate(ranked, 1):
        out[s]["rank"] = i
    return out


def read(ticker: str, sector: str | None, rot: dict[str, dict[str, Any]], daily: pd.DataFrame) -> dict[str, Any] | None:
    """Sector quadrant plus the stock's one-month strength against its sector ETF."""
    etf_sector = next((s for s, e in SECTOR_ETFS.items() if e == ticker), None)
    sector = sector or etf_sector
    if not sector or sector not in rot:
        return None
    row = rot[sector]
    score = float(QUADRANT_POINTS[row["quadrant"]])
    sign = "+" if score > 0 else "-"
    reasons = [f"{sign} {sector} ({row['etf']}) is {row['quadrant']}: {row['rs_3m']:+.1%} against SPY over three months, "
               f"momentum {row['momentum']:+.1%}; ranked {row['rank']} of {len(rot)} sectors this month"]
    if not etf_sector and len(daily) > 22:
        try:
            etf = data.history(row["etf"])["Close"]
            stock = daily["Close"]
            rel = float((stock.iloc[-1] / stock.iloc[-22]) / (etf.iloc[-1] / etf.iloc[-22]) - 1)
            if abs(rel) >= 0.03:
                score += 15 if rel > 0 else -15
                reasons.append(f"{'+' if rel > 0 else '-'} {'Outperforming' if rel > 0 else 'Underperforming'} its sector by "
                               f"{abs(rel):.1%} over the past month")
            else:
                reasons.append(f"= Moving with its sector over the past month ({rel:+.1%} relative)")
        except Exception:
            pass
    return {"score": float(np.clip(score, -100, 100)), "reasons": reasons, "sector": sector,
            "sector_quadrant": row["quadrant"]}
