"""Symplectic integrators written to be autodiff-friendly.

All functions are pure (no in-place mutation), so they can be traced by JAX
(`jax.grad`, `jax.jit`, `jax.vmap`) when `physweave[jax]` is installed, and
still run on plain NumPy otherwise.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

Array = np.ndarray
ForceFn = Callable[[Array], Array]


def velocity_verlet(
    force_fn: ForceFn,
    x0: Array,
    v0: Array,
    dt: float,
    n_steps: int,
    mass: float | Array = 1.0,
) -> Tuple[Array, Array]:
    """Integrate Newton's equations with velocity Verlet.

    Second-order, symplectic, time-reversible. Returns final (x, v).
    """
    x, v = x0, v0
    a = force_fn(x) / mass
    for _ in range(n_steps):
        x = x + v * dt + 0.5 * a * dt**2
        a_new = force_fn(x) / mass
        v = v + 0.5 * (a + a_new) * dt
        a = a_new
    return x, v


def leapfrog(
    force_fn: ForceFn,
    x0: Array,
    v0: Array,
    dt: float,
    n_steps: int,
    mass: float | Array = 1.0,
) -> Tuple[Array, Array]:
    """Kick-drift-kick leapfrog integrator. Returns final (x, v)."""
    x, v = x0, v0
    for _ in range(n_steps):
        v = v + 0.5 * dt * force_fn(x) / mass
        x = x + dt * v
        v = v + 0.5 * dt * force_fn(x) / mass
    return x, v


def rollout(
    force_fn: ForceFn,
    x0: Array,
    v0: Array,
    dt: float,
    n_steps: int,
    mass: float | Array = 1.0,
) -> Tuple[Array, Array]:
    """Like velocity_verlet but records the full trajectory.

    Returns arrays of shape (n_steps + 1, *x0.shape).
    """
    xs = [x0]
    vs = [v0]
    x, v = x0, v0
    a = force_fn(x) / mass
    for _ in range(n_steps):
        x = x + v * dt + 0.5 * a * dt**2
        a_new = force_fn(x) / mass
        v = v + 0.5 * (a + a_new) * dt
        a = a_new
        xs.append(x)
        vs.append(v)
    return np.stack(xs), np.stack(vs)
