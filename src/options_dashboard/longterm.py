"""Long-term research: S&P 500 companies scored for buy-and-hold, with buy areas and targets.

For a Roth IRA at Schwab, where fractional shares (Stock Slices) cover the S&P 500, so every
company here can be bought in any dollar amount from $5.

Once a week each company's last four annual reports, its dividends and its analyst targets are
refreshed; prices are refreshed every run (after the close each weekday). Each company gets four
scores, as percentiles against the rest of the index (0-100, higher is better):

  quality    return on invested capital (return on equity for financials), free-cash-flow and gross
             margins, steady operating margins, low debt
  growth     three-year growth in revenue, earnings per share and free cash flow, and analysts'
             expected earnings growth
  dividend   yield, five-year dividend growth, years of increases, and how well free cash flow
             covers it (only companies with a 0.5%+ yield and five years of payments)
  value      free-cash-flow yield, price against the company's own valuation history, and the
             expected yearly return to the stretch target

The shortlist is the best overall (quality counts most) among companies with a quality score of
60+, positive free cash flow and manageable debt, with at most four per sector.

Buy area: the nearest long-term support at or below the price (40-week average, one-year anchored
VWAP, one-year volume value area, 52-week low), with the next support below it as the floor.
Targets: per-share free cash flow (earnings for financials and real estate) grown at the company's
own rate (capped at 20% a year). The stretch target, three years out, values it at the median
multiple the company traded at over the last four fiscal years; the base target, one year out,
assumes the multiple closes only half the gap to that median.

    python -m options_dashboard.longterm      # refresh and save .state/longterm/latest.json
"""

from __future__ import annotations

import io
import json
import time
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from . import data
from .config import HOME, STATE_DIR

DIR = STATE_DIR / "longterm"
RAW_DIR = DIR / "raw"
LATEST = DIR / "latest.json"
NOTES = HOME / "research" / "longterm_notes.json"
FUNDAMENTALS_MAX_AGE = 7 * 86400
SHORTLIST = 25
PER_SECTOR = 4
EARNINGS_SECTORS = {"Financials", "Real Estate"}  # free cash flow means little for banks, insurers and REITs
INFO_KEYS = ("shortName", "longBusinessSummary", "sector", "industry", "marketCap", "trailingPE", "forwardPE",
             "trailingEps", "forwardEps", "dividendYield", "payoutRatio", "targetMeanPrice", "targetLowPrice",
             "targetHighPrice", "numberOfAnalystOpinions", "recommendationKey", "beta", "grossMargins",
             "operatingMargins", "returnOnEquity", "earningsGrowth", "revenueGrowth")
ROWS = {  # field: candidate row names in Yahoo's annual statements, first match wins
    "revenue": ("Total Revenue",), "gross_profit": ("Gross Profit",), "ebit": ("EBIT", "Operating Income"),
    "ebitda": ("EBITDA", "Normalized EBITDA"), "net_income": ("Net Income", "Net Income Common Stockholders"),
    "eps": ("Diluted EPS", "Basic EPS"), "fcf": ("Free Cash Flow",), "dividends_paid": ("Cash Dividends Paid",),
    "buybacks": ("Repurchase Of Capital Stock",), "shares": ("Ordinary Shares Number", "Share Issued"),
    "debt": ("Total Debt",), "cash": ("Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"),
    "invested_capital": ("Invested Capital",), "equity": ("Stockholders Equity", "Common Stock Equity"),
}


# ---------------------------------------------------------------- data

def sp500() -> pd.DataFrame:
    """Symbol, name, GICS sector and sub-industry for every S&P 500 member, refreshed weekly."""
    path = DIR / "sp500.json"
    try:
        if time.time() - path.stat().st_mtime < FUNDAMENTALS_MAX_AGE:
            return pd.read_json(path)
    except (OSError, ValueError):
        pass
    r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                     headers={"User-Agent": "options-dashboard research"}, timeout=30)
    r.raise_for_status()
    table = pd.read_html(io.StringIO(r.text))[0]
    df = pd.DataFrame({"symbol": table["Symbol"].str.replace(".", "-", regex=False), "name": table["Security"],
                       "sector": table["GICS Sector"], "industry": table["GICS Sub-Industry"]})
    DIR.mkdir(parents=True, exist_ok=True)
    df.to_json(path)
    return df


def _row(frame: pd.DataFrame, names: tuple[str, ...]) -> dict[str, float]:
    for name in names:
        if frame is not None and name in frame.index:
            return {str(c.date()): float(v) for c, v in frame.loc[name].items() if pd.notna(v)}
    return {}


def fundamentals(ticker: str) -> dict[str, Any]:
    """Company profile, four annual reports and dividends per calendar year, cached for a week."""
    path = RAW_DIR / f"{ticker}.json"
    try:
        if time.time() - path.stat().st_mtime < FUNDAMENTALS_MAX_AGE:
            return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    tk = yf.Ticker(ticker)
    info = data.call(lambda: tk.info) or {}
    statements = [data.call(lambda f=f: getattr(tk, f)) for f in ("income_stmt", "cashflow", "balance_sheet")]
    annual: dict[str, dict[str, float]] = {}
    for field, names in ROWS.items():
        for frame in statements:
            values = _row(frame, names)
            if values:
                for day, v in values.items():
                    annual.setdefault(day, {})[field] = v
                break
    divs = data.call(lambda: tk.dividends)
    by_year = {}
    if divs is not None and len(divs):
        s = divs.copy()
        s.index = pd.to_datetime(s.index).year
        by_year = {int(y): float(v) for y, v in s.groupby(level=0).sum().items()}
    raw = {"ticker": ticker, "fetched": datetime.now().isoformat(timespec="seconds"),
           "info": {k: info.get(k) for k in INFO_KEYS}, "annual": annual, "dividends": by_year}
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw), encoding="utf-8")
    return raw


def weekly_prices(tickers: list[str], cache: str = "weekly.pkl") -> dict[str, pd.DataFrame]:
    """Ten years of weekly closes and volume per ticker, downloaded in batches once a day."""
    path = DIR / cache
    try:
        if date.fromtimestamp(path.stat().st_mtime) == date.today():
            cached = pd.read_pickle(path)
            if set(tickers) <= set(cached):
                return cached
    except (OSError, ValueError, EOFError):
        pass
    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), 50):
        batch = tickers[i:i + 50]
        frame = data.call(yf.download, batch, period="10y", interval="1wk", auto_adjust=True, group_by="ticker",
                          progress=False, threads=False)
        for t in batch:
            try:
                df = frame[t][["Close", "Volume"]].dropna()
            except KeyError:
                continue
            if len(df) >= 60:
                out[t] = df
    DIR.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(out, path)
    return out


# ---------------------------------------------------------------- per-company metrics

def _cagr(series: list[float], years: int) -> float | None:
    if len(series) <= years or series[-1 - years] <= 0 or series[-1] <= 0:
        return None
    return (series[-1] / series[-1 - years]) ** (1 / years) - 1


def _price_on(prices: pd.DataFrame, day: str) -> float | None:
    s = prices["Close"][prices.index <= pd.Timestamp(day)]
    return float(s.iloc[-1]) if len(s) else None


def metrics(raw: dict[str, Any], prices: pd.DataFrame, sector: str) -> dict[str, Any] | None:
    years = sorted(raw["annual"])
    if len(years) < 2 or prices is None or prices.empty:
        return None
    a = [raw["annual"][y] for y in years]  # oldest first
    latest = a[-1]
    info = raw["info"]
    price = float(prices["Close"].iloc[-1])
    earnings_based = sector in EARNINGS_SECTORS

    def col(field: str) -> list[float]:
        return [r[field] for r in a if r.get(field) is not None]

    m: dict[str, Any] = {"price": price, "earnings_based": earnings_based}
    revenue, fcf, eps = col("revenue"), col("fcf"), col("eps")
    rev = latest.get("revenue")
    # Quality.
    if earnings_based:
        roes = [r["net_income"] / r["equity"] for r in a if r.get("net_income") is not None and r.get("equity", 0) > 0]
        m["return_on_capital"] = float(np.mean(roes)) if roes else None
    else:
        roics = [r["ebit"] * 0.79 / r["invested_capital"] for r in a
                 if r.get("ebit") is not None and r.get("invested_capital", 0) > 0]
        m["return_on_capital"] = float(np.mean(roics)) if roics else None
    m["fcf_margin"] = latest["fcf"] / rev if latest.get("fcf") is not None and rev else None
    m["gross_margin"] = latest["gross_profit"] / rev if latest.get("gross_profit") is not None and rev else None
    margins = [r["ebit"] / r["revenue"] for r in a if r.get("ebit") is not None and r.get("revenue")]
    m["margin_steadiness"] = -float(np.std(margins)) if len(margins) >= 3 else None  # higher is steadier
    ebitda = latest.get("ebitda")
    m["net_debt_ebitda"] = ((latest.get("debt", 0) - latest.get("cash", 0)) / ebitda
                            if ebitda and ebitda > 0 and not earnings_based else None)
    # Growth.
    span = min(3, len(a) - 1)
    m["revenue_growth"] = _cagr(revenue, span) if len(revenue) > span else None
    m["eps_growth"] = _cagr(eps, span) if len(eps) > span else None
    m["fcf_growth"] = _cagr(fcf, span) if len(fcf) > span else None
    fwd, trail = info.get("forwardEps"), info.get("trailingEps")
    m["expected_growth"] = (fwd / trail - 1) if fwd and trail and trail > 0 else None
    shares = col("shares")
    m["share_change"] = _cagr(shares, len(shares) - 1) if len(shares) >= 2 else None
    # Dividends: complete calendar years only.
    this_year = date.today().year
    divs = {int(y): v for y, v in raw["dividends"].items() if int(y) < this_year}
    m["dividend_yield"] = divs.get(this_year - 1, 0.0) / price  # last full year's dividends
    paid_years = sorted(y for y, v in divs.items() if v > 0)
    streak = 0
    for y in range(this_year - 1, (paid_years[0] if paid_years else this_year), -1):
        if divs.get(y, 0) > divs.get(y - 1, 0) * 1.001 and divs.get(y - 1, 0) > 0:
            streak += 1
        else:
            break
    m["dividend_streak"] = streak
    m["dividend_growth"] = ((divs[this_year - 1] / divs[this_year - 6]) ** 0.2 - 1
                            if divs.get(this_year - 1) and divs.get(this_year - 6) else None)
    paid = abs(latest.get("dividends_paid") or 0)
    m["fcf_payout"] = paid / latest["fcf"] if latest.get("fcf") and latest["fcf"] > 0 and paid else None
    m["pays_dividend"] = bool(m["dividend_yield"] >= 0.005 and len(paid_years) >= 5)

    # Valuation against the company's own history: price-to-free-cash-flow (price-to-earnings for
    # financials and real estate) at each fiscal year end, and now.
    multiples = []
    for y, r in zip(years, a):
        p = _price_on(prices, y)
        if p is None:
            continue
        if earnings_based and r.get("eps", 0) > 0:
            multiples.append(p / r["eps"])
        elif not earnings_based and r.get("fcf", 0) > 0 and r.get("shares"):
            multiples.append(p * r["shares"] / r["fcf"])
    per_share = (latest.get("eps") if earnings_based else
                 (latest["fcf"] / latest["shares"] if latest.get("fcf") and latest.get("shares") else None))
    m["per_share"], m["metric"] = per_share, "earnings" if earnings_based else "free cash flow"
    if not per_share or per_share <= 0 or not multiples:
        m.update(multiple_now=None, fair_multiple=None, fcf_yield=None, fair_value=None, base=None, stretch=None,
                 growth_used=None, expected_return=None)
        return m
    fair_multiple = float(np.clip(np.median(multiples), 8, 60))
    growths = [g for g in (m["fcf_growth"] if not earnings_based else m["eps_growth"], m["revenue_growth"],
                           m["expected_growth"]) if g is not None]
    g = float(np.clip(np.median(growths), 0.0, 0.20)) if growths else 0.03
    m["multiple_now"] = price / per_share
    m["fair_multiple"] = fair_multiple
    m["history_multiples"] = [round(x, 1) for x in multiples]
    m["fcf_yield"] = per_share / price
    m["growth_used"] = g
    m["fair_value"] = per_share * fair_multiple
    # One year out the multiple is assumed to close half the gap to its usual level; three years
    # out, all of it. A company that has re-rated for a reason should not be valued as if it will
    # snap back within a year.
    m["base_multiple"] = (m["multiple_now"] + fair_multiple) / 2
    m["base"] = per_share * (1 + g) * m["base_multiple"]
    m["stretch"] = per_share * (1 + g) ** 3 * fair_multiple
    m["expected_return"] = (m["stretch"] / price) ** (1 / 3) - 1 + m["dividend_yield"]
    return m


# ---------------------------------------------------------------- buy areas

def buy_area(prices: pd.DataFrame, fair_value: float | None) -> dict[str, Any]:
    """Nearest long-term support at or below the price, the next one below as the floor, and
    where the price sits against them."""
    year = prices.tail(52)
    close, vol = year["Close"], year["Volume"].clip(lower=1)
    price = float(close.iloc[-1])
    levels = {"40-week average": float(prices["Close"].tail(40).mean()),
              "1-year anchored VWAP": float((close * vol).sum() / vol.sum()),
              "52-week low": float(close.min())}
    hist, edges = np.histogram(close, bins=24, weights=vol)
    order = np.argsort(hist)[::-1]
    kept, total = [], hist.sum()
    for i in order:
        kept.append(i)
        if hist[kept].sum() >= 0.7 * total:
            break
    levels["1-year value area low"] = float(edges[min(kept)])
    levels["1-year point of control"] = float((edges[order[0]] + edges[order[0] + 1]) / 2)
    below = sorted(((v, k) for k, v in levels.items() if v <= price * 1.02), reverse=True)
    if below:
        (top, top_name), rest = below[0], below[1:]
        # The floor is the next support at least 4% lower (a narrower area is one price, not an
        # area), as long as it is within 15%; otherwise 5% under the top.
        under = [(v, k) for v, k in rest if top * 0.85 <= v <= top * 0.96]
        floor, floor_name = under[0] if under else (top * 0.95, "5% under it")
        if price > top * 1.02:
            status = "Near buy area" if price <= top * 1.07 else "Above: wait for a pullback"
        else:
            status = "In buy area"
    else:  # below every support: no floor has formed yet
        top, top_name = min((v, k) for k, v in levels.items())
        floor, floor_name = price * 0.95, "5% under the price"
        status = "Below support: let it base first"
    mid = (top + floor) / 2
    valuation = None
    if fair_value:
        valuation = ("expensive even here" if mid > fair_value * 1.2 else
                     "below its usual valuation" if mid < fair_value * 0.9 else "near its usual valuation")
    return {"low": round(floor, 2), "high": round(top, 2), "low_name": floor_name, "high_name": top_name,
            "status": status, "distance": price / top - 1, "valuation": valuation,
            "levels": {k: round(v, 2) for k, v in sorted(levels.items(), key=lambda kv: -kv[1])}}


# ---------------------------------------------------------------- scoring and shortlist

def _pct(s: pd.Series, higher_better: bool = True) -> pd.Series:
    return s.rank(pct=True, ascending=higher_better) * 100


def score(rows: pd.DataFrame) -> pd.DataFrame:
    """Four style scores, each the average of its components' percentile ranks across the index."""
    def blend(parts: list[pd.Series]) -> pd.Series:
        return pd.concat(parts, axis=1).mean(axis=1, skipna=True)

    n = rows
    for c in ("return_on_capital", "fcf_margin", "gross_margin", "margin_steadiness", "net_debt_ebitda", "revenue_growth",
              "eps_growth", "fcf_growth", "expected_growth", "dividend_yield", "dividend_growth", "dividend_streak",
              "fcf_payout", "multiple_now", "fair_multiple", "fcf_yield", "expected_return", "per_share", "base"):
        n[c] = pd.to_numeric(n[c], errors="coerce")
    n["pays_dividend"] = n["pays_dividend"].fillna(False).astype(bool)
    debt = _pct(n["net_debt_ebitda"].clip(lower=-5), higher_better=False)
    n["quality"] = blend([_pct(n["return_on_capital"]), _pct(n["fcf_margin"]), _pct(n["gross_margin"]),
                          _pct(n["margin_steadiness"]), debt])
    n["growth"] = blend([_pct(n["revenue_growth"]), _pct(n["eps_growth"]), _pct(n["fcf_growth"]),
                         _pct(n["expected_growth"])])
    payers = n["pays_dividend"]
    safety = _pct(n["fcf_payout"].where(n["fcf_payout"] < 1.0), higher_better=False)
    n["dividend"] = blend([_pct(n["dividend_yield"].where(payers)), _pct(n["dividend_growth"].where(payers)),
                           _pct(n["dividend_streak"].where(payers).astype(float)), safety.where(payers)])
    n.loc[~payers, "dividend"] = np.nan
    vs_history = n["multiple_now"] / n["fair_multiple"]
    n["value"] = blend([_pct(n["fcf_yield"]), _pct(vs_history, higher_better=False), _pct(n["expected_return"])])
    weights = {"quality": 0.45, "growth": 0.2, "value": 0.2, "dividend": 0.15}
    parts = pd.DataFrame({k: n[k] * w for k, w in weights.items()})
    present = pd.DataFrame({k: n[k].notna() * w for k, w in weights.items()}).sum(axis=1)
    n["overall"] = parts.sum(axis=1, skipna=True) / present
    styles = {"quality": "Quality compounder", "growth": "Growth", "dividend": "Dividend growth", "value": "Undervalued"}
    n["styles"] = [[label for k, label in styles.items() if pd.notna(r[k]) and r[k] >= 75
                    and (k == "quality" or r["quality"] >= 50)] for _, r in n.iterrows()]
    return n


def shortlist(scored: pd.DataFrame) -> pd.DataFrame:
    ok = scored[(scored["quality"] >= 60) & (scored["per_share"] > 0) & scored["base"].notna()
                & ~(scored["net_debt_ebitda"] > 3.5)].sort_values("overall", ascending=False)
    picked, per_sector = [], {}
    for t, r in ok.iterrows():
        if per_sector.get(r["sector"], 0) >= PER_SECTOR:
            continue
        per_sector[r["sector"]] = per_sector.get(r["sector"], 0) + 1
        picked.append(t)
        if len(picked) == SHORTLIST:
            break
    return ok.loc[picked]


# ---------------------------------------------------------------- cash machines

CM_MIN_CAP = 300_000_000
CM_MIN_YIELD = 0.12  # free cash flow as a share of market value, latest year


def cash_candidates() -> list[str]:
    """US stocks whose trailing free cash flow is large against their market value.

    Yahoo's screener filters on absolute free cash flow, not the ratio, so it is asked band by band
    (market value doubling each band) for cash flow of at least CM_MIN_YIELD of the band's floor.
    That is a superset: the exact yield comes from the annual reports afterwards."""
    from yfinance import EquityQuery
    found: set[str] = set()
    lo = CM_MIN_CAP
    while lo < 2e12:
        hi = lo * 2
        q = EquityQuery("and", [EquityQuery("eq", ["region", "us"]),
                                EquityQuery("is-in", ["exchange", "NMS", "NGM", "NCM", "NYQ", "ASE"]),
                                EquityQuery("btwn", ["intradaymarketcap", lo, hi]),
                                EquityQuery("gte", ["leveredfreecashflow.lasttwelvemonths", lo * CM_MIN_YIELD])])
        offset = 0
        while True:
            res = data.call(yf.screen, q, offset=offset, size=250, sortField="intradaymarketcap", sortAsc=True)
            quotes = res.get("quotes", [])
            found |= {x["symbol"] for x in quotes if x.get("quoteType") == "EQUITY" and "." not in x["symbol"]}
            offset += len(quotes)
            if not quotes or offset >= res.get("total", 0):
                break
        lo = hi
    return sorted(found)


def cash_machine(raw: dict[str, Any], prices: pd.DataFrame | None) -> dict[str, Any] | None:
    """How much free cash flow the company makes against what the market charges for all of it,
    and whether that looks durable or like the top of a cycle (the ZIM pattern)."""
    info = raw["info"]
    if (info.get("sector") or "") == "Financial Services" or prices is None or prices.empty:
        return None  # a bank's or insurer's cash flow statement does not measure free cash
    years = sorted(raw["annual"])
    a = [raw["annual"][y] for y in years]
    fcf = [r["fcf"] for r in a if r.get("fcf") is not None]
    shares = a[-1].get("shares")
    if len(fcf) < 3 or not shares or fcf[-1] <= 0:
        return None
    price = float(prices["Close"].iloc[-1])
    cap = price * shares
    latest, avg, worst = fcf[-1], float(np.mean(fcf)), min(fcf)
    prior = float(np.median(fcf[:-1]))
    returned = abs(a[-1].get("dividends_paid") or 0) + abs(a[-1].get("buybacks") or 0)
    net_cash = (a[-1].get("cash") or 0) - (a[-1].get("debt") or 0)
    if prior > 0 and latest > 2 * prior:
        verdict = "Peak-cycle risk: the latest year is more than double the usual"
    elif latest < 0.6 * avg:
        verdict = "Falling: the latest year is well below the average"
    elif worst > 0 and worst / cap >= 0.08:
        verdict = "Durable: every year above an 8% yield"
    elif worst > 0:
        verdict = "Steady: positive every year"
    else:
        verdict = "Uneven: at least one year burned cash"
    per_share_avg = avg / shares
    return {
        "price": price, "market_cap": cap, "fcf_yield": latest / cap, "avg_yield": avg / cap, "worst_yield": worst / cap,
        "years_to_pay": cap / avg if avg > 0 else None, "four_year_fcf_vs_cap": sum(fcf[-4:]) / cap,
        "fcf_history": [round(x / 1e6) for x in fcf], "trend": latest / avg if avg > 0 else None, "verdict": verdict,
        "returned_share": returned / latest, "net_cash_share": net_cash / cap,
        # Re-rating target: the price at which the average year's cash flow would yield 10%.
        "target": per_share_avg * 10 if per_share_avg > 0 else None,
    }


# ---------------------------------------------------------------- themes and emerging industries

THEMES = HOME / "research" / "themes.json"


def _rs(df: pd.DataFrame | None, spy: pd.DataFrame, weeks: int) -> float | None:
    """Return over `weeks` relative to SPY's: +0.10 means it beat SPY by 10%."""
    if df is None or len(df) <= weeks or len(spy) <= weeks:
        return None
    p, s = df["Close"], spy["Close"]
    return float((p.iloc[-1] / p.iloc[-1 - weeks]) / (s.iloc[-1] / s.iloc[-1 - weeks]) - 1)


def _median(xs: list[float | None]) -> float | None:
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def themes(scored: pd.DataFrame, in_index: set[str]) -> tuple[list[dict[str, Any]], pd.DataFrame | None]:
    """Each researched theme's tiers with live strength against SPY, and whether the outer tiers
    (suppliers and enablers) are lagging the leaders: the extrapolation the research is after."""
    try:
        doc = json.loads(THEMES.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [], None
    tickers = sorted({t for th in doc["themes"] for tier in th["tiers"] for t in tier["tickers"]} | {"SPY"})
    px = weekly_prices(tickers, "weekly_themes.pkl")
    spy = px.get("SPY")
    if spy is None:
        return [], None
    out = []
    for th in doc["themes"]:
        tiers = []
        for tier in th["tiers"]:
            members = []
            for t, why in tier["tickers"].items():
                df = px.get(t)
                q = scored["quality"].get(t) if t in scored.index else None
                members.append({"ticker": t, "why": why, "in_sp500": t in in_index,
                                "price": float(df["Close"].iloc[-1]) if df is not None else None,
                                "rs13": _rs(df, spy, 13), "rs52": _rs(df, spy, 52),
                                "quality": None if q is None or pd.isna(q) else float(q),
                                "status": buy_area(df, None)["status"] if df is not None and len(df) >= 52 else None})
            tiers.append({"tier": tier["tier"], "why": tier["why"], "members": members,
                          "rs13": _median([m["rs13"] for m in members]), "rs52": _median([m["rs52"] for m in members])})
        lead = tiers[0]
        signals = []
        for t in tiers[1:]:
            if lead["rs52"] is None or t["rs52"] is None:
                continue
            gap = lead["rs52"] - t["rs52"]
            turning = t["rs13"] is not None and lead["rs13"] is not None and t["rs13"] > lead["rs13"]
            if gap >= 0.10:
                signals.append(f"{t['tier']} has trailed the first order by {gap:.0%} over a year"
                               + (", and has started catching up over the last 3 months: the extrapolation is "
                                  "beginning to play out" if turning else ": room to catch up if the theme holds"))
            elif gap <= -0.10:
                signals.append(f"{t['tier']} has beaten the first order by {-gap:.0%} over a year: the market is "
                               "already paying for this link in the chain")
        r13 = _median([t["rs13"] for t in tiers])
        r52 = _median([t["rs52"] for t in tiers])
        if r13 is None or r52 is None:
            verdict = "Not enough history"
        elif r13 > 0 and r52 > 0:
            verdict = "Working: beating SPY over 3 months and a year"
        elif r13 > 0:
            verdict = "Turning up: beating SPY over 3 months after a weak year"
        elif r52 > 0:
            verdict = "Cooling: ahead over a year but behind SPY over 3 months"
        else:
            verdict = "Not working yet: behind SPY over 3 months and a year"
        out.append({k: th[k] for k in ("id", "name", "thesis", "evidence", "sources", "risks", "watch")}
                   | {"tiers": tiers, "signals": signals, "verdict": verdict, "rs13": r13, "rs52": r52})
    return out, spy


def emerging_industries(members: pd.DataFrame, prices: dict[str, pd.DataFrame], spy: pd.DataFrame) -> dict[str, list]:
    """S&P 500 sub-industries ranked by whether their strength against SPY is accelerating: the last
    three months against the pace of the last year. Themes often show up here before they get a name."""
    rows = []
    for industry, grp in members.groupby("industry"):
        rs = [(t, _rs(prices.get(t), spy, 13), _rs(prices.get(t), spy, 52)) for t in grp["symbol"]]
        rs = [(t, a, b) for t, a, b in rs if a is not None and b is not None]
        if len(rs) < 3:
            continue
        r13, r52 = float(np.median([a for _, a, _ in rs])), float(np.median([b for _, _, b in rs]))
        rows.append({"industry": industry, "sector": grp["sector"].iloc[0], "members": len(rs), "rs13": r13, "rs52": r52,
                     "acceleration": r13 - r52 * 13 / 52,
                     "leaders": [t for t, _, _ in sorted(rs, key=lambda x: -x[1])[:4]]})
    rows.sort(key=lambda r: -r["acceleration"])
    return {"gaining": [r for r in rows if r["rs13"] > 0][:10],
            "fading": [r for r in reversed(rows) if r["rs13"] < 0][:10]}


def _notes() -> dict[str, Any]:
    try:
        return json.loads(NOTES.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def refresh() -> dict[str, Any]:
    """Refresh data, score the index, build the shortlist with buy areas and targets, and save it."""
    members = sp500()
    tickers = members["symbol"].tolist()
    prices = weekly_prices(tickers)
    rows, failed = [], []
    for _, mem in members.iterrows():
        t = mem["symbol"]
        try:
            raw = fundamentals(t)
            m = metrics(raw, prices.get(t), mem["sector"])
        except Exception as exc:  # one company's missing statements must not sink the run
            failed.append(f"{t}: {type(exc).__name__}")
            continue
        if m is None:
            failed.append(f"{t}: not enough data")
            continue
        info = raw["info"]
        rows.append({"ticker": t, "name": mem["name"], "sector": mem["sector"], "industry": mem["industry"],
                     "summary": (info.get("longBusinessSummary") or "")[:700], "market_cap": info.get("marketCap"),
                     "analyst_mean": info.get("targetMeanPrice"), "analyst_low": info.get("targetLowPrice"),
                     "analyst_high": info.get("targetHighPrice"), "analysts": info.get("numberOfAnalystOpinions"),
                     "recommendation": info.get("recommendationKey"), "beta": info.get("beta"),
                     "fundamentals_date": raw["fetched"][:10], **m})
    scored = score(pd.DataFrame(rows).set_index("ticker"))
    picks = shortlist(scored)
    notes = _notes()
    out = []
    for t, r in picks.iterrows():
        area = buy_area(prices[t], r["fair_value"])
        out.append({"ticker": t, **{k: (None if isinstance(v, float) and np.isnan(v) else v)
                                    for k, v in r.items()}, "buy_area": area, "note": notes.get(t)})
    leaders = {style: [{"ticker": t, "name": r["name"], "score": round(float(r[style]), 0), "overall": round(float(r["overall"]), 0)}
                       for t, r in scored[scored["quality"] >= 50].sort_values(style, ascending=False).head(10).iterrows()]
               for style in ("quality", "growth", "dividend", "value")}
    # Cash machines: the whole US market, not just the S&P 500.
    in_index = set(tickers)
    candidates = cash_candidates()
    cm_prices = weekly_prices(candidates, "weekly_cash.pkl")
    machines = []
    for t in candidates:
        try:
            raw = fundamentals(t)
            m = cash_machine(raw, cm_prices.get(t))
        except Exception:
            continue
        if not m or m["fcf_yield"] < CM_MIN_YIELD:
            continue
        info = raw["info"]
        machines.append({"ticker": t, "name": info.get("shortName") or t, "sector": info.get("sector"),
                         "industry": info.get("industry"), "summary": (info.get("longBusinessSummary") or "")[:500],
                         "in_sp500": t in in_index, **m, "buy_area": buy_area(cm_prices[t], m["target"]),
                         "note": notes.get(t)})
    order = {"Durable": 0, "Steady": 1, "Uneven": 2, "Falling": 3, "Peak": 4}
    machines.sort(key=lambda r: (order[r["verdict"].split(":")[0].split("-")[0]], -r["avg_yield"]))
    result = {"built": datetime.now(data.NEW_YORK).isoformat(timespec="minutes"), "screened": len(scored),
              "members": len(members), "failed": failed, "shortlist": out, "leaders": leaders,
              "cash_machines": machines, "cash_candidates": len(candidates)}
    theme_list, spy = themes(scored, in_index)
    result["themes"] = theme_list
    try:
        result["themes_meta"] = {k: json.loads(THEMES.read_text(encoding="utf-8")).get(k) for k in ("date", "author")}
    except (OSError, json.JSONDecodeError):
        result["themes_meta"] = {}
    result["industries"] = emerging_industries(members, prices, spy) if spy is not None else {"gaining": [], "fading": []}
    DIR.mkdir(parents=True, exist_ok=True)
    LATEST.write_text(json.dumps(result, default=float), encoding="utf-8")
    return result


def load() -> dict[str, Any]:
    try:
        return json.loads(LATEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


if __name__ == "__main__":
    import sys

    from . import site

    if "--page-only" not in sys.argv:
        res = refresh()
        print(f"Screened {res['screened']} of {res['members']}; shortlisted {len(res['shortlist'])}; "
              f"{len(res.get('cash_machines', []))} cash machines; {len(res['failed'])} without enough data")
    site.OUT.with_name("invest.html").write_text(site.render_invest(load()), encoding="utf-8")
    if "--publish" in sys.argv:
        print(site.publish() or "Published")
