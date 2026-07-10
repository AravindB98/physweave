"""Spectral (FFT-based) solver for the 2D heat equation on a periodic domain.

Serves as the exact ground truth that learned surrogates are benchmarked
against: du/dt = alpha * laplacian(u), x in [0, L)^2, periodic BCs.
"""

from __future__ import annotations

import numpy as np


def _laplacian_symbol(n: int, length: float) -> np.ndarray:
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=length / n)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    return -(kx**2 + ky**2)


def heat_step_spectral(u: np.ndarray, alpha: float, dt: float, length: float = 1.0) -> np.ndarray:
    """Advance the heat equation one step of size dt, exactly, in Fourier space."""
    n = u.shape[0]
    lap = _laplacian_symbol(n, length)
    u_hat = np.fft.fft2(u)
    u_hat *= np.exp(alpha * lap * dt)
    return np.real(np.fft.ifft2(u_hat))


def solve_heat_2d(
    u0: np.ndarray, alpha: float, dt: float, n_steps: int, length: float = 1.0
) -> np.ndarray:
    """Return trajectory of shape (n_steps + 1, n, n)."""
    traj = [u0]
    u = u0
    for _ in range(n_steps):
        u = heat_step_spectral(u, alpha, dt, length)
        traj.append(u)
    return np.stack(traj)


def random_initial_condition(n: int, n_modes: int = 4, seed: int | None = None) -> np.ndarray:
    """Smooth random periodic field built from a few low-frequency Fourier modes."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    u = np.zeros((n, n))
    for _ in range(n_modes):
        kx, ky = rng.integers(1, 4, size=2)
        amp, phase = rng.normal(), rng.uniform(0, 2 * np.pi)
        u += amp * np.sin(kx * xx + ky * yy + phase)
    return u / n_modes
