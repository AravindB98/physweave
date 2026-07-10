from physweave.quantfin import black_scholes_call, mc_european_call, simulate_gbm


def test_mc_matches_black_scholes():
    s0, k, r, sigma, t = 100.0, 105.0, 0.03, 0.2, 1.0
    bs = black_scholes_call(s0, k, r, sigma, t)
    mc, stderr = mc_european_call(s0, k, r, sigma, t, n_paths=200_000, seed=42)
    assert abs(mc - bs) < 4 * stderr


def test_antithetic_reduces_variance():
    s0, k, r, sigma, t = 100.0, 100.0, 0.03, 0.2, 1.0
    _, se_plain = mc_european_call(s0, k, r, sigma, t, 100_000, antithetic=False, seed=1)
    _, se_anti = mc_european_call(s0, k, r, sigma, t, 100_000, antithetic=True, seed=1)
    assert se_anti < se_plain


def test_gbm_martingale_property():
    import numpy as np

    # Under drift mu, E[S_t] = S_0 exp(mu t)
    paths = simulate_gbm(100.0, 0.05, 0.3, 1.0, 50, 200_000, seed=7)
    expected = 100.0 * np.exp(0.05)
    assert abs(np.mean(paths[:, -1]) - expected) / expected < 0.01
