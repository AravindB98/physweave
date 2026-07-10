import numpy as np

from physweave.surrogates import FourierRidgeSurrogate, heat_step_spectral
from physweave.surrogates.spectral import random_initial_condition


def test_heat_decays_high_modes():
    n = 32
    u0 = random_initial_condition(n, seed=0)
    u1 = heat_step_spectral(u0, alpha=0.1, dt=0.5, length=2 * np.pi)
    assert np.std(u1) < np.std(u0)  # diffusion smooths the field
    assert abs(np.mean(u1) - np.mean(u0)) < 1e-10  # mean is conserved


def test_surrogate_learns_heat_operator():
    n, alpha, dt = 32, 0.05, 0.1
    rng_seeds = range(20)
    inputs = np.stack([random_initial_condition(n, seed=s) for s in rng_seeds])
    outputs = np.stack(
        [heat_step_spectral(u, alpha, dt, length=2 * np.pi) for u in inputs]
    )
    model = FourierRidgeSurrogate().fit(inputs, outputs)

    u_test = random_initial_condition(n, seed=999)
    pred = model.predict(u_test)
    truth = heat_step_spectral(u_test, alpha, dt, length=2 * np.pi)
    rel_err = np.linalg.norm(pred - truth) / np.linalg.norm(truth)
    assert rel_err < 1e-6  # exact operator is inside the hypothesis class
