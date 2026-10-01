"""Market-structure reads: auction/market profile, VPA, Wyckoff and dealer gamma.

Each function returns {"score": -100..100, "reasons": [...], plus price levels}, or
None when there is not enough data. Positive scores are bullish. Reasons start with
"+" (bullish), "-" (bearish), "!" (warning) or "=" (neutral context).

These are rule-based approximations of discretionary methods built on free data:
the profile spreads each 30-minute bar's volume evenly across its range, and gamma
exposure assumes dealers are long calls and short puts.
"""

from __future__ import annotations

from datetime import date, datetime, time as clock
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import greeks

NEW_YORK = ZoneInfo("America/New_York")

# Share of the composite direction score each method contributes.
METHOD_WEIGHTS = {"auction": 0.25, "gamma": 0.20, "wyckoff": 0.15, "vpa": 0.15, "trend": 0.25}
METHOD_NAMES = {
    "auction": "Auction / market profile", "gamma": "Dealer gamma, vanna and charm", "wyckoff": "Wyckoff",
    "vpa": "Volume price analysis", "trend": "Trend and momentum",
}


def _out(score: float, reasons: list[str], **levels: Any) -> dict[str, Any]:
    return {"score": float(np.clip(score, -100, 100)), "reasons": reasons, **levels}


def completed_sessions(daily: pd.DataFrame) -> pd.DataFrame:
    """Drop today's bar while the market is still open: its volume is only partial."""
    now = datetime.now(NEW_YORK)
    if len(daily) and daily.index[-1].date() == now.date() and now.time() < clock(16, 0):
        return daily.iloc[:-1]
    return daily


# ---------------------------------------------------------------- auction / profile

def volume_profile(bars: pd.DataFrame, bins: int = 60) -> dict[str, float] | None:
    """Point of control and 70% value area from intraday bars."""
    lo, hi = float(bars["Low"].min()), float(bars["High"].max())
    if not hi > lo:
        return None
    volume = np.zeros(bins)
    scale = bins / (hi - lo)
    for low, high, vol in zip(bars["Low"], bars["High"], bars["Volume"]):
        if vol > 0:
            a = min(int((low - lo) * scale), bins - 1)
            b = min(int((high - lo) * scale), bins - 1)
            volume[a:b + 1] += vol / (b - a + 1)
    total = volume.sum()
    if total <= 0:
        return None
    poc = left = right = int(volume.argmax())
    inside = volume[poc]
    while inside < 0.7 * total and (left > 0 or right < bins - 1):
        below = volume[left - 1] if left > 0 else -1.0
        above = volume[right + 1] if right < bins - 1 else -1.0
        if above >= below:
            right += 1
            inside += above
        else:
            left -= 1
            inside += below
    width = (hi - lo) / bins
    return {"poc": lo + (poc + 0.5) * width, "val": lo + left * width, "vah": lo + (right + 1) * width}


def auction(bars: pd.DataFrame, atr: float) -> dict[str, Any] | None:
    """Where price is trading relative to established value, and where value is moving."""
    days = [frame for _, frame in bars.groupby(bars.index.date)]
    if len(days) < 7:
        return None
    prior = days[-11:-1]  # established value: the sessions before the current one
    half = len(prior) // 2
    area = volume_profile(pd.concat(prior))
    yesterday = volume_profile(days[-2])
    older, newer = volume_profile(pd.concat(prior[:half])), volume_profile(pd.concat(prior[half:]))
    if not (area and yesterday and older and newer):
        return None

    # Two consecutive 30-minute closes beyond a level count as acceptance, not a probe.
    closes = bars["Close"].iloc[-2:]
    price, hi2, lo2 = float(closes.iloc[-1]), float(closes.max()), float(closes.min())
    n = len(prior)
    score, reasons = 0.0, []

    if lo2 > area["vah"]:
        score += 40
        reasons.append(f"+ Accepted above the {n}-day value area high (${area['vah']:.2f}): buyers are initiating")
    elif hi2 < area["val"]:
        score -= 40
        reasons.append(f"- Accepted below the {n}-day value area low (${area['val']:.2f}): sellers are initiating")
    else:
        lean = 10 if price > area["poc"] else -10
        score += lean
        reasons.append(
            f"= Inside the {n}-day value area (${area['val']:.2f} to ${area['vah']:.2f}), "
            f"{'above' if lean > 0 else 'below'} the ${area['poc']:.2f} point of control: two-sided, balanced trade"
        )

    if lo2 > yesterday["vah"]:
        score += 30
        reasons.append(f"+ Holding above yesterday's value area high (${yesterday['vah']:.2f})")
    elif hi2 < yesterday["val"]:
        score -= 30
        reasons.append(f"- Holding below yesterday's value area low (${yesterday['val']:.2f})")
    else:
        reasons.append(f"= Rotating inside yesterday's value (${yesterday['val']:.2f} to ${yesterday['vah']:.2f})")

    shift = newer["poc"] - older["poc"]
    if shift > 0.5 * atr:
        score += 30
        reasons.append(f"+ Value is migrating higher (point of control ${older['poc']:.2f} to ${newer['poc']:.2f})")
    elif shift < -0.5 * atr:
        score -= 30
        reasons.append(f"- Value is migrating lower (point of control ${older['poc']:.2f} to ${newer['poc']:.2f})")
    else:
        reasons.append("= Value is overlapping from week to week: the market is in balance")

    return _out(score, reasons, poc=area["poc"], vah=area["vah"], val=area["val"])


# ---------------------------------------------------------------- volume price analysis

def vpa(daily: pd.DataFrame) -> dict[str, Any] | None:
    """Effort (volume) against result (spread and close) on the last five sessions."""
    d = completed_sessions(daily).tail(45)
    if len(d) < 30:
        return None
    spread = d["High"] - d["Low"]
    rel_spread = spread / spread.rolling(20).mean().shift()
    rel_vol = d["Volume"] / d["Volume"].rolling(20).mean().shift()
    close_pos = ((d["Close"] - d["Low"]) / spread.replace(0, np.nan)).fillna(0.5)
    up = d["Close"] > d["Close"].shift()
    run = d["Close"].shift() / d["Close"].shift(6) - 1  # move into the bar

    score, reasons = 0.0, []
    for back, weight in zip(range(1, 6), (1.0, 0.8, 0.6, 0.45, 0.3)):
        rs, rv, cp, is_up = rel_spread.iloc[-back], rel_vol.iloc[-back], close_pos.iloc[-back], up.iloc[-back]
        day = f"{d.index[-back]:%b} {d.index[-back].day}"
        vol = f"{rv:.1f}x volume"
        if rv >= 1.8 and not is_up and cp >= 0.55:
            pts, text = 35, f"+ {day}: stopping volume, heavy selling ({vol}) absorbed with a close off the lows"
        elif rv >= 1.8 and is_up and cp <= 0.45:
            pts, text = -35, f"- {day}: buying climax, heavy volume ({vol}) but a weak close"
        elif rv >= 1.5 and rs <= 0.7:
            pts = -20 if run.iloc[-back] > 0 else 20
            text = (f"{'-' if pts < 0 else '+'} {day}: high effort, little result ({vol}, narrow spread): "
                    f"{'supply is absorbing the rally' if pts < 0 else 'demand is absorbing the decline'}")
        elif is_up and rv >= 1.2 and rs >= 1.0 and cp >= 0.6:
            pts, text = 30, f"+ {day}: wide up bar on {vol} closing near the high, real demand"
        elif not is_up and rv >= 1.2 and rs >= 1.0 and cp <= 0.4:
            pts, text = -30, f"- {day}: wide down bar on {vol} closing near the low, real supply"
        elif is_up and rv <= 0.8 and rs <= 0.8:
            pts, text = -15, f"- {day}: no demand, a narrow up bar on {vol}"
        elif not is_up and rv <= 0.8 and rs <= 0.8:
            pts, text = 15, f"+ {day}: no supply, a narrow down bar on {vol}"
        else:
            continue
        score += pts * weight
        reasons.append(text)

    if not reasons:
        reasons.append("= No notable effort-versus-result signals in the last five sessions")
    return _out(score, reasons)


# ---------------------------------------------------------------- Wyckoff

def wyckoff(daily: pd.DataFrame) -> dict[str, Any] | None:
    """Trading-range events (spring, upthrust, breakout) or the trend phase outside a range."""
    d = completed_sessions(daily)
    if len(d) < 70:
        return None
    window, recent = d.iloc[-45:-5], d.iloc[-5:]
    top, bottom = float(window["High"].max()), float(window["Low"].min())
    close, atr = float(d["Close"].iloc[-1]), float(d["atr"].iloc[-1])
    width = top - bottom
    levels = {"range_high": top, "range_low": bottom}
    expanding = float((recent["Volume"] / d["Volume"].rolling(20).mean().shift().iloc[-5:]).max()) >= 1.3
    span = f"${bottom:.2f} to ${top:.2f}"

    # A random walk covers about 8 ATRs in 40 sessions, so anything tighter is a range.
    if width <= 7 * atr:
        if close > top:
            pts = 70 if expanding else 40
            return _out(pts, [f"+ Sign of strength: broke out above the {span} range"
                              f"{' on expanding volume' if expanding else ', but without a volume surge'}"],
                        phase="Breakout from range", **levels)
        if close < bottom:
            pts = -70 if expanding else -40
            return _out(pts, [f"- Sign of weakness: broke down below the {span} range"
                              f"{' on expanding volume' if expanding else ', but without a volume surge'}"],
                        phase="Breakdown from range", **levels)
        if float(recent["Low"].min()) < bottom:
            return _out(60, [f"+ Spring: undercut the ${bottom:.2f} range low and closed back inside, "
                             "a failed breakdown that traps sellers"], phase="Spring", **levels)
        if float(recent["High"].max()) > top:
            return _out(-60, [f"- Upthrust: poked above the ${top:.2f} range high and closed back inside, "
                              "a failed breakout that traps buyers"], phase="Upthrust", **levels)
        rising = window["Close"] > window["Close"].shift()
        balance = window["Volume"][rising].sum() / max(window["Volume"][~rising].sum(), 1)
        if balance > 1.15:
            return _out(25, [f"+ Trading range {span} with heavier volume on up days: reads as accumulation"],
                        phase="Accumulation", **levels)
        if balance < 0.87:
            return _out(-25, [f"- Trading range {span} with heavier volume on down days: reads as distribution"],
                        phase="Distribution", **levels)
        return _out(0, [f"= Trading range {span} with balanced volume: no edge until it resolves"],
                    phase="Trading range", **levels)

    where = (close - bottom) / width
    move = close / float(window["Close"].iloc[0]) - 1
    if close > top or (where >= 0.75 and move > 0):
        return _out(35, [f"+ Markup phase: trending higher, up {move:.1%} across the last 45 sessions"],
                    phase="Markup", **levels)
    if close < bottom or (where <= 0.25 and move < 0):
        return _out(-35, [f"- Markdown phase: trending lower, down {abs(move):.1%} across the last 45 sessions"],
                    phase="Markdown", **levels)
    return _out(0, ["= No clear Wyckoff structure: wide swings without a defined range or trend"],
                phase="Unclear", **levels)


# ---------------------------------------------------------------- dealer gamma

def _short(x: float) -> str:
    for size, unit in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= size:
            return f"${abs(x) / size:.1f}{unit}"
    return f"${abs(x):.0f}"


def signed_short(x: float) -> str:
    return ("+" if x >= 0 else "-") + _short(x)


FLOW_MIN = 0.002  # vanna flow per vol point below this share of average daily dollar volume is ignored
# Charm flow is positive for almost every stock (out-of-the-money options dominate and their
# delta decays toward zero), so it only counts when it is large.
CHARM_MIN = 0.01


def gamma(chain: pd.DataFrame, spot: float, rate: float, adv_dollars: float,
          vol_trend: int, vol_note: str) -> dict[str, Any] | None:
    """Dealer gamma, vanna and charm exposure from open interest.

    Gamma gives the walls, the flip level and whether hedging dampens or extends moves.
    Vanna and charm give the stock dealers must trade as implied volatility and time
    change their delta. `vol_trend` is -1 when implied volatility is falling, +1 when
    rising and 0 when unknown; `vol_note` says how that was judged.
    """
    # Contracts expiring today are left out: their gamma swamps the walls and is gone by the close.
    df = chain[
        (chain["openInterest"] > 0)
        & chain["strike"].between(spot * 0.75, spot * 1.25)
        & (pd.to_datetime(chain["expiration"]) > pd.Timestamp(date.today()))
    ]
    if len(df) < 10 or df["openInterest"].sum() < 500:
        return None
    iv = df["impliedVolatility"].to_numpy(dtype=float)
    valid = (iv > 0.03) & (iv < 5)
    iv = np.where(valid, iv, np.median(iv[valid]) if valid.any() else 0.3)
    dte = (pd.to_datetime(df["expiration"]) - pd.Timestamp(date.today())).dt.days
    T = np.maximum(dte.to_numpy(dtype=float), 0.5) / 365.0
    K = df["strike"].to_numpy(dtype=float)
    oi = df["openInterest"].to_numpy(dtype=float)
    is_call = (df["type"] == "call").to_numpy()
    sign = np.where(is_call, 1.0, -1.0)

    def exposure(price: float) -> np.ndarray:
        """Dollar gamma per 1% move, signed: calls positive, puts negative."""
        g = greeks.greeks(price, K, T, rate, iv, 0.0, is_call)["gamma"]
        return g * oi * 100 * price * price * 0.01 * sign

    gex = pd.Series(exposure(spot), index=K)
    net = float(gex.sum())
    call_wall = float(gex[is_call].groupby(level=0).sum().idxmax())
    put_wall = float(gex[~is_call].groupby(level=0).sum().idxmin())

    # Vanna and charm: how much stock dealers must trade as volatility and time change their delta.
    # Positive numbers are dollars of stock bought.
    second = greeks.greeks(spot, K, T, rate, iv, 0.0, is_call)
    vanna_flow = float((second["vanna"] * oi * 100 * spot * sign).sum())  # per 1-point fall in IV
    charm_flow = float(-(second["charm"] * oi * 100 * spot * sign).sum())  # per day of time decay

    grid = np.linspace(spot * 0.85, spot * 1.15, 61)
    nets = np.array([exposure(p).sum() for p in grid])
    crossings = np.nonzero(np.diff(np.sign(nets)))[0]
    flip = None
    if len(crossings):
        i = min(crossings, key=lambda j: abs(grid[j] - spot))
        flip = float(grid[i] - nets[i] * (grid[i + 1] - grid[i]) / (nets[i + 1] - nets[i]))

    score = 0.0
    reasons = [f"= Call wall ${call_wall:g}, put wall ${put_wall:g}, net gamma "
               f"{'+' if net >= 0 else '-'}{_short(net)} per 1% move"]
    if net >= 0:
        score += 30 if flip and spot > flip else 15
        level = f" (above the ${flip:.2f} gamma flip)" if flip and spot > flip else ""
        reasons.append(f"+ Positive gamma{level}: dealer hedging supports dips, but it also dampens big moves")
    else:
        score -= 30 if flip and spot < flip else 15
        level = f" (below the ${flip:.2f} gamma flip)" if flip and spot < flip else ""
        reasons.append(f"- Negative gamma{level}: dealer hedging chases price, so moves tend to extend")
    if 0 <= call_wall / spot - 1 <= 0.01:
        score -= 20
        reasons.append(f"! Pressing into the ${call_wall:g} call wall, which tends to act as resistance")
    if 0 <= 1 - put_wall / spot <= 0.01:
        score += 20
        reasons.append(f"! Sitting on the ${put_wall:g} put wall, which tends to act as support")

    # A flow only counts when it is a meaningful share of a normal day's trading.
    vanna_now = -vol_trend * vanna_flow  # falling IV (trend -1) realises the flow as written
    if vol_trend == 0 or abs(vanna_flow) < FLOW_MIN * adv_dollars:
        reasons.append(f"= Vanna: dealers would {'buy' if vanna_flow >= 0 else 'sell'} {_short(vanna_flow)} of stock "
                       f"per 1-point fall in implied volatility, too small or too unclear to lean on")
    else:
        score += 20 if vanna_now > 0 else -20
        reasons.append(
            f"{'+' if vanna_now > 0 else '-'} Vanna {'tailwind' if vanna_now > 0 else 'headwind'}: implied volatility "
            f"is {'falling' if vol_trend < 0 else 'rising'} ({vol_note}), which makes dealers "
            f"{'buy' if vanna_now > 0 else 'sell'} about {_short(vanna_flow)} of stock per point"
        )
    if abs(charm_flow) < CHARM_MIN * adv_dollars:
        reasons.append(f"= Charm: time decay has dealers {'buying' if charm_flow >= 0 else 'selling'} "
                       f"{_short(charm_flow)} of stock a day, too small to matter")
    else:
        score += 20 if charm_flow > 0 else -20
        reasons.append(
            f"{'+' if charm_flow > 0 else '-'} Charm {'tailwind' if charm_flow > 0 else 'headwind'}: as time passes, "
            f"dealers must {'buy' if charm_flow > 0 else 'sell'} about {_short(charm_flow)} of stock a day to stay hedged"
        )

    return _out(score, reasons, call_wall=call_wall, put_wall=put_wall, flip=flip, net_gex=net,
                vanna_flow=vanna_flow, charm_flow=charm_flow)


# ---------------------------------------------------------------- composite

def composite(parts: dict[str, dict[str, Any] | None]) -> float:
    """Weighted direction score over the methods that produced a reading."""
    used = {k: p for k, p in parts.items() if p is not None}
    weight = sum(METHOD_WEIGHTS[k] for k in used)
    return sum(METHOD_WEIGHTS[k] * p["score"] for k, p in used.items()) / weight if weight else 0.0


def gamma_fit(levels: dict[str, Any] | None, is_call: np.ndarray, breakeven: np.ndarray, spot: float) -> np.ndarray:
    """0..1 score for how well the gamma backdrop suits buying each contract.

    Long options want negative gamma (moves extend) and room to reach breakeven
    before the wall in the trade's direction.
    """
    if not levels:
        return np.full(len(breakeven), 0.5)
    regime = 1.0 if levels["net_gex"] < 0 else 0.35
    wall = np.where(is_call, levels["call_wall"], levels["put_wall"])
    needed = np.abs(breakeven - spot)
    available = np.where(is_call, wall - spot, spot - wall)
    room = np.clip(available / np.maximum(needed, 1e-9), 0.0, 1.0)
    room = np.where(available < 0, 1.0, room)  # price is already through the wall: nothing in the way
    return 0.5 * regime + 0.5 * room
