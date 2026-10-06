"""Sector rotation: which sectors are gaining or losing strength against the market.

Each SPDR sector ETF's price ratio to SPY gives its relative strength (RS). Its RS ratio (strength
against its own recent average) and RS momentum (whether that strength is rising) place it in
one of four quadrants, as on a relative rotation graph:

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


RS_WEEKS = 10  # weeks of relative strength the ratio is measured against
MOMENTUM_WEEKS = 5  # weeks the momentum is measured against
TAIL_WEEKS = 8  # weekly points shown behind each sector on the rotation graph


def _quadrant(ratio: float, momentum: float) -> str:
    if ratio >= 100:
        return "leading" if momentum >= 100 else "weakening"
    return "improving" if momentum >= 100 else "lagging"


def rotation() -> dict[str, dict[str, Any]]:
    """Relative rotation for every sector, keyed by sector name.

    Weekly closes. RS ratio = 100 x (sector / SPY) / its RS_WEEKS-week average: above 100 the
    sector is stronger than the market by its recent standard. RS momentum = 100 x RS ratio / its
    MOMENTUM_WEEKS-week average: above 100 that strength is rising. Sectors circle clockwise
    through leading, weakening, lagging and improving. The latest week uses today's price.
    """
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
        rs = (px / spy).dropna()
        if len(rs) < 70:
            continue
        weekly = rs.resample("W-FRI").last().dropna()
        ratio = 100 * weekly / weekly.rolling(RS_WEEKS).mean()
        momentum = 100 * ratio / ratio.rolling(MOMENTUM_WEEKS).mean()
        path = pd.DataFrame({"ratio": ratio, "momentum": momentum}).dropna().tail(TAIL_WEEKS + 1)
        if path.empty:
            continue
        last = path.iloc[-1]
        prior = float(rs.iloc[-22] / rs.iloc[-43] - 1)
        rs1 = float(rs.iloc[-1] / rs.iloc[-22] - 1)
        out[sector] = {
            "etf": etf, "rs_1m": rs1, "rs_3m": float(rs.iloc[-1] / rs.iloc[-64] - 1), "momentum": rs1 - prior,
            "ratio": float(last["ratio"]), "rs_momentum": float(last["momentum"]),
            "quadrant": _quadrant(float(last["ratio"]), float(last["momentum"])),
            "return_1m": float(px.iloc[-1] / px.iloc[-22] - 1),
            "tail": [{"week": f"{d:%b} {d.day}", "ratio": float(r.ratio), "momentum": float(r.momentum)}
                     for d, r in path.iterrows()],
        }
    ranked = sorted(out, key=lambda s: out[s]["rs_1m"], reverse=True)
    for i, s in enumerate(ranked, 1):
        out[s]["rank"] = i
    return out


REGIME_TILT = 10  # points for a sector the current market regime favours (or against one it does not)


def read(ticker: str, sector: str | None, rot: dict[str, dict[str, Any]], daily: pd.DataFrame,
         regime: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Sector quadrant, the stock's one-month strength against its sector ETF, and a tilt toward
    sectors that have done best in the current market regime and are rotating the right way."""
    etf_sector = next((s for s, e in SECTOR_ETFS.items() if e == ticker), None)
    sector = sector or etf_sector
    if not sector or sector not in rot:
        return None
    row = rot[sector]
    score = float(QUADRANT_POINTS[row["quadrant"]])
    sign = "+" if score > 0 else "-"
    reasons = [f"{sign} {sector} ({row['etf']}) is {row['quadrant']} on the rotation graph (RS ratio "
               f"{row['ratio']:.1f}, momentum {row['rs_momentum']:.1f}); {row['rs_1m']:+.1%} against SPY this month, "
               f"ranked {row['rank']} of {len(rot)} sectors"]
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
    if regime:
        favoured = {f["sector"]: f for f in regime.get("favoured", [])}
        avoided = {f["sector"]: f for f in regime.get("avoid", [])}
        if sector in favoured:
            score += REGIME_TILT
            reasons.append(f"+ Favoured in the current regime ({regime['state']}): {sector} has beaten SPY by "
                           f"{favoured[sector]['edge']:+.1%} a month on average in this state since 2008")
        elif sector in avoided:
            score -= REGIME_TILT
            reasons.append(f"- Out of favour in the current regime ({regime['state']}): {sector} has trailed SPY by "
                           f"{abs(avoided[sector]['edge']):.1%} a month on average in this state since 2008")
    return {"score": float(np.clip(score, -100, 100)), "reasons": reasons, "sector": sector,
            "sector_quadrant": row["quadrant"]}
