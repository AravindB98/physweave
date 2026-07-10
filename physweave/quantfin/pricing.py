"""Option pricing: Monte Carlo estimator validated against Black-Scholes."""

from __future__ import annotations

import math

import numpy as np

from physweave.quantfin.sde import simulate_gbm


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_call(s0: float, k: float, r: float, sigma: float, t: float) -> float:
    """Closed-form Black-Scholes price of a European call."""
    if t <= 0:
        return max(s0 - k, 0.0)
    d1 = (math.log(s0 / k) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    return s0 * _norm_cdf(d1) - k * math.exp(-r * t) * _norm_cdf(d2)


def mc_european_call(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    n_paths: int = 100_000,
    n_steps: int = 1,
    antithetic: bool = True,
    seed: int | None = None,
) -> tuple[float, float]:
    """Monte Carlo price of a European call under risk-neutral GBM.

    Returns (price, standard_error).
    """
    paths = simulate_gbm(s0, r, sigma, t, n_steps, n_paths, antithetic, seed)
    payoffs = np.maximum(paths[:, -1] - k, 0.0) * math.exp(-r * t)
    price = float(np.mean(payoffs))
    stderr = float(np.std(payoffs, ddof=1) / math.sqrt(n_paths))
    return price, stderr
