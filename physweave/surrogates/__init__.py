"""Neural/learned surrogates for PDEs, plus classical spectral solvers as ground truth."""

from physweave.surrogates.spectral import heat_step_spectral, solve_heat_2d
from physweave.surrogates.fourier_ridge import FourierRidgeSurrogate

__all__ = ["heat_step_spectral", "solve_heat_2d", "FourierRidgeSurrogate"]
