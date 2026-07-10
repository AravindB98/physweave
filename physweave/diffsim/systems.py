"""Canonical physical systems exposed as force functions."""

from __future__ import annotations

import numpy as np


class SpringMass:
    """1D (or component-wise ND) harmonic oscillator: F = -k (x - x_eq)."""

    def __init__(self, k: float = 1.0, x_eq: float = 0.0):
        self.k = k
        self.x_eq = x_eq

    def force(self, x: np.ndarray) -> np.ndarray:
        return -self.k * (x - self.x_eq)

    def energy(self, x: np.ndarray, v: np.ndarray, mass: float = 1.0) -> float:
        return float(0.5 * mass * np.sum(v**2) + 0.5 * self.k * np.sum((x - self.x_eq) ** 2))

    def analytic(self, x0: float, v0: float, t: float, mass: float = 1.0):
        """Closed-form solution used to validate integrators."""
        w = np.sqrt(self.k / mass)
        x = x0 * np.cos(w * t) + (v0 / w) * np.sin(w * t)
        v = -x0 * w * np.sin(w * t) + v0 * np.cos(w * t)
        return x, v


class NBodyGravity:
    """Pairwise Newtonian gravity with Plummer softening.

    Positions have shape (n_bodies, dim).
    """

    def __init__(self, masses: np.ndarray, G: float = 1.0, softening: float = 1e-3):
        self.masses = np.asarray(masses, dtype=float)
        self.G = G
        self.eps2 = softening**2

    def force(self, x: np.ndarray) -> np.ndarray:
        diff = x[None, :, :] - x[:, None, :]  # (n, n, dim), r_j - r_i
        dist2 = np.sum(diff**2, axis=-1) + self.eps2
        inv_d3 = dist2**-1.5
        np.fill_diagonal(inv_d3, 0.0)
        # F_i = G m_i sum_j m_j (r_j - r_i) / |r|^3
        acc = self.G * np.einsum("j,ijd,ij->id", self.masses, diff, inv_d3)
        return acc * self.masses[:, None]

    def energy(self, x: np.ndarray, v: np.ndarray) -> float:
        ke = 0.5 * np.sum(self.masses[:, None] * v**2)
        diff = x[None, :, :] - x[:, None, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1) + self.eps2)
        iu = np.triu_indices(len(self.masses), k=1)
        pe = -self.G * np.sum(self.masses[iu[0]] * self.masses[iu[1]] / dist[iu])
        return float(ke + pe)
