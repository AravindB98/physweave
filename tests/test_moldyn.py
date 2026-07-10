import numpy as np

from physweave.moldyn import LennardJones, run_md
from physweave.moldyn.lennard_jones import lattice_positions


def test_energy_conservation():
    box = 6.0
    lj = LennardJones(box)
    x0 = lattice_positions(3, box)  # 27 atoms
    rng = np.random.default_rng(0)
    v0 = rng.normal(scale=0.1, size=x0.shape)
    v0 -= v0.mean(axis=0)  # zero total momentum

    result = run_md(lj, x0, v0, dt=1e-3, n_steps=2000)
    e = result["total_energy"]
    drift = abs(e[-1] - e[0]) / abs(e[0])
    assert drift < 1e-3


def test_forces_are_conservative_gradient():
    # F = -dU/dx: check against finite differences for one coordinate.
    # Use a perturbed 3x3x3 lattice: atoms must NOT sit at exactly half the
    # box length apart, where the minimum-image convention is discontinuous.
    box = 6.0
    lj = LennardJones(box)
    rng = np.random.default_rng(3)
    x = lattice_positions(3, box) + rng.uniform(-0.05, 0.05, size=(27, 3))
    f = lj.force(x)
    h = 1e-6
    xp, xm = x.copy(), x.copy()
    xp[0, 0] += h
    xm[0, 0] -= h
    fd = -(lj.potential_energy(xp) - lj.potential_energy(xm)) / (2 * h)
    assert abs(f[0, 0] - fd) < 1e-5
