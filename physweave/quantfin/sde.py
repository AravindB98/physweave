"""SDE path simulation. The physics connection: geometric Brownian motion is
just a drifted diffusion — the same Langevin machinery as in statistical
mechanics, with volatility playing the role of temperature."""

from __future__ import annotations

import numpy as np


def simulate_gbm(
    s0: float,
    mu: float,
    sigma: float,
    t: float,
    n_steps: int,
    n_paths: int,
    antithetic: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate geometric Brownian motion via the exact log-space solution.

    dS = mu S dt + sigma S dW  =>  S_t = S_0 exp((mu - sigma^2/2) t + sigma W_t)

    With `antithetic=True`, half the paths use mirrored Brownian increments
    (a variance-reduction trick borrowed from importance sampling in MC
    physics). Returns paths of shape (n_paths, n_steps + 1).
    """
    rng = np.random.default_rng(seed)
    dt = t / n_steps
    if antithetic:
        half = (n_paths + 1) // 2
        z = rng.standard_normal((half, n_steps))
        z = np.concatenate([z, -z], axis=0)[:n_paths]
    else:
        z = rng.standard_normal((n_paths, n_steps))
    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(increments, axis=1)], axis=1
    )
    return s0 * np.exp(log_paths)
