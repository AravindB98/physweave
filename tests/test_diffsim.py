import numpy as np

from physweave.diffsim import SpringMass, NBodyGravity, velocity_verlet, leapfrog


def test_spring_matches_analytic():
    sys = SpringMass(k=4.0)
    x0, v0, dt, n = np.array([1.0]), np.array([0.0]), 1e-3, 5000
    x, v = velocity_verlet(sys.force, x0, v0, dt, n)
    xa, va = sys.analytic(1.0, 0.0, dt * n)
    assert abs(x[0] - xa) < 1e-4
    assert abs(v[0] - va) < 1e-4


def test_leapfrog_energy_conservation():
    sys = SpringMass(k=1.0)
    x0, v0 = np.array([1.0]), np.array([0.5])
    e0 = sys.energy(x0, v0)
    x, v = leapfrog(sys.force, x0, v0, 1e-2, 10_000)
    assert abs(sys.energy(x, v) - e0) / e0 < 1e-3


def test_two_body_energy_conservation():
    masses = np.array([1.0, 1.0])
    grav = NBodyGravity(masses, softening=1e-2)
    x0 = np.array([[-0.5, 0.0], [0.5, 0.0]])
    v0 = np.array([[0.0, -0.5], [0.0, 0.5]])
    e0 = grav.energy(x0, v0)
    x, v = velocity_verlet(grav.force, x0, v0, 1e-3, 2000, mass=masses[:, None])
    assert abs(grav.energy(x, v) - e0) / abs(e0) < 1e-3
