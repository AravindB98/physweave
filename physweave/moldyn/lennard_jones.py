"""Lennard-Jones molecular dynamics with periodic boundary conditions.

Reduced units (epsilon = sigma = m = 1). This classical baseline is the
ground truth that learned interatomic potentials (the `torch` extra) are
trained against and benchmarked on.
"""

from __future__ import annotations

import numpy as np


class LennardJones:
    """Pairwise 12-6 Lennard-Jones potential with minimum-image PBC."""

    def __init__(self, box: float, epsilon: float = 1.0, sigma: float = 1.0):
        self.box = box
        self.eps = epsilon
        self.sig = sigma

    def _displacements(self, x: np.ndarray) -> np.ndarray:
        diff = x[:, None, :] - x[None, :, :]
        return diff - self.box * np.round(diff / self.box)  # minimum image

    def force(self, x: np.ndarray) -> np.ndarray:
        diff = self._displacements(x)
        r2 = np.sum(diff**2, axis=-1)
        np.fill_diagonal(r2, np.inf)
        inv_r2 = self.sig**2 / r2
        inv_r6 = inv_r2**3
        # F = 24 eps (2 (sig/r)^12 - (sig/r)^6) / r^2 * r_vec
        coef = 24.0 * self.eps * (2.0 * inv_r6**2 - inv_r6) / r2
        return np.sum(coef[:, :, None] * diff, axis=1)

    def potential_energy(self, x: np.ndarray) -> float:
        diff = self._displacements(x)
        r2 = np.sum(diff**2, axis=-1)
        iu = np.triu_indices(x.shape[0], k=1)
        inv_r6 = (self.sig**2 / r2[iu]) ** 3
        return float(4.0 * self.eps * np.sum(inv_r6**2 - inv_r6))


def lattice_positions(n_per_side: int, box: float) -> np.ndarray:
    """Simple cubic lattice filling the box — a safe, non-overlapping start."""
    pts = (np.arange(n_per_side) + 0.5) * (box / n_per_side)
    grid = np.stack(np.meshgrid(pts, pts, pts, indexing="ij"), axis=-1)
    return grid.reshape(-1, 3)


def run_md(
    potential: LennardJones,
    x0: np.ndarray,
    v0: np.ndarray,
    dt: float,
    n_steps: int,
) -> dict:
    """Velocity-Verlet MD. Returns dict with final state and energy series."""
    x, v = x0.copy(), v0.copy()
    f = potential.force(x)
    energies = []
    for _ in range(n_steps):
        x = (x + v * dt + 0.5 * f * dt**2) % potential.box
        f_new = potential.force(x)
        v = v + 0.5 * (f + f_new) * dt
        f = f_new
        ke = 0.5 * float(np.sum(v**2))
        energies.append(ke + potential.potential_energy(x))
    return {"x": x, "v": v, "total_energy": np.array(energies)}
