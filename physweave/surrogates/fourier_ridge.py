"""A minimal learned PDE surrogate: ridge regression in Fourier space.

This is the "FNO-lite" baseline of PhysWeave: it learns a per-mode complex
multiplier that maps u(t) -> u(t + dt). For linear PDEs (heat, advection)
this hypothesis class contains the exact solution operator, which makes it a
sharp correctness test and a strong baseline that heavier neural operators
(FNO, transformers — see the `torch` extra) must beat on nonlinear problems.
"""

from __future__ import annotations

import numpy as np


class FourierRidgeSurrogate:
    """Learns a diagonal spectral propagator from (input, output) field pairs."""

    def __init__(self, ridge: float = 1e-8):
        self.ridge = ridge
        self.multiplier_: np.ndarray | None = None

    def fit(self, inputs: np.ndarray, outputs: np.ndarray) -> "FourierRidgeSurrogate":
        """inputs/outputs: (n_samples, n, n) real fields."""
        x_hat = np.fft.fft2(inputs, axes=(-2, -1))
        y_hat = np.fft.fft2(outputs, axes=(-2, -1))
        num = np.sum(np.conj(x_hat) * y_hat, axis=0)
        den = np.sum(np.abs(x_hat) ** 2, axis=0) + self.ridge
        self.multiplier_ = num / den
        return self

    def predict(self, u: np.ndarray) -> np.ndarray:
        if self.multiplier_ is None:
            raise RuntimeError("Call fit() before predict().")
        u_hat = np.fft.fft2(u, axes=(-2, -1))
        return np.real(np.fft.ifft2(u_hat * self.multiplier_, axes=(-2, -1)))

    def rollout(self, u0: np.ndarray, n_steps: int) -> np.ndarray:
        traj = [u0]
        u = u0
        for _ in range(n_steps):
            u = self.predict(u)
            traj.append(u)
        return np.stack(traj)
