"""Black-Scholes-Merton pricing, greeks and implied volatility.

Yahoo does not publish greeks, so they are computed here. The model is European;
for the short-dated, near-the-money US equity options this tool looks at, the
early-exercise premium is small enough to ignore.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma, q):
    vol_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_t
    return d1, d1 - vol_t


def price(S, K, T, r, sigma, q=0.0, is_call=True):
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    call = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return np.where(is_call, call, put)


def greeks(S, K, T, r, sigma, q=0.0, is_call=True) -> dict[str, np.ndarray]:
    """Delta, gamma, theta (per calendar day), vega (per 1 vol point), rho (per 1% rate),
    vanna (delta change per 1 vol point) and charm (delta change per calendar day)."""
    S, K, T, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, sigma))
    is_call = np.asarray(is_call, dtype=bool)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_q, disc_r = np.exp(-q * T), np.exp(-r * T)
    pdf = norm.pdf(d1)

    delta = np.where(is_call, disc_q * norm.cdf(d1), -disc_q * norm.cdf(-d1))
    gamma = disc_q * pdf / (S * sigma * np.sqrt(T))
    decay = -S * disc_q * pdf * sigma / (2 * np.sqrt(T))
    theta_call = decay - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1)
    theta_put = decay + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1)
    rho = np.where(is_call, K * T * disc_r * norm.cdf(d2), -K * T * disc_r * norm.cdf(-d2))
    # Second-order: how delta moves with volatility (vanna) and with time (charm).
    vanna = -disc_q * pdf * d2 / sigma
    drift = disc_q * pdf * (2 * (r - q) * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))
    charm = np.where(is_call, q * disc_q * norm.cdf(d1) - drift, -q * disc_q * norm.cdf(-d1) - drift)

    return {
        "vanna": vanna / 100.0,
        "charm": charm / 365.0,
        "delta": delta,
        "gamma": gamma,
        "theta": np.where(is_call, theta_call, theta_put) / 365.0,
        "vega": S * disc_q * pdf * np.sqrt(T) / 100.0,
        "rho": rho / 100.0,
    }


def implied_vol(target, S, K, T, r, q=0.0, is_call=True) -> float:
    """Volatility that reprices the option to `target`; NaN when no solution exists."""
    if not (target > 0 and S > 0 and K > 0 and T > 0):
        return float("nan")
    try:
        return float(
            brentq(lambda v: float(price(S, K, T, r, v, q, is_call)) - target, 1e-3, 6.0, xtol=1e-5)
        )
    except ValueError:  # price below intrinsic or above the model's ceiling
        return float("nan")


def prob_above(S, level, T, r, sigma, q=0.0):
    """Risk-neutral probability the underlying finishes above `level` at expiry."""
    _, d2 = _d1_d2(S, level, T, r, sigma, q)
    return norm.cdf(d2)
