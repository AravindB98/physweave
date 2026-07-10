"""Differentiable physics simulation (JAX-optional, NumPy fallback)."""

from physweave.diffsim.integrators import leapfrog, velocity_verlet
from physweave.diffsim.systems import NBodyGravity, SpringMass

__all__ = ["leapfrog", "velocity_verlet", "SpringMass", "NBodyGravity"]
