# PhysWeave

**Weaving physics simulation and machine learning into one modular toolkit.**

PhysWeave unifies five pillars of computational physics — differentiable simulation, neural PDE surrogates, stochastic finance, quantum circuit simulation, and molecular dynamics — under one clean, tested, NumPy-first API with optional JAX/PyTorch acceleration.

[![CI](https://img.shields.io/badge/CI-pytest-blue)](/.github/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)

---

## ⭐ Support the project

If PhysWeave is useful to you, **please star ⭐ and fork 🍴 this repository** — it takes 5 seconds and directly helps the project grow:

1. Click **Star** (top-right of this page) so more physicists and ML engineers discover the project.
2. Click **Fork** to get your own copy — the first step to contributing (see below).

---

## Modules

| Module | What it does | Target domain |
|---|---|---|
| `physweave.diffsim` | Symplectic, autodiff-friendly integrators (velocity Verlet, leapfrog) + canonical systems (spring-mass, N-body gravity) | Robotics / ML research |
| `physweave.surrogates` | Spectral PDE solvers as ground truth + learned surrogates (Fourier-ridge baseline, neural operators via `torch` extra) | Scientific ML, climate, aerospace |
| `physweave.quantfin` | GBM/SDE path simulation with antithetic variance reduction; Monte Carlo option pricing validated against Black-Scholes | Quantitative finance |
| `physweave.quantum` | Dense statevector quantum circuit simulator (H, X, Y, Z, S, T, RX, RZ, controlled gates, sampling) | Quantum computing |
| `physweave.moldyn` | Lennard-Jones molecular dynamics with periodic boundaries and energy-conservation guarantees | Materials / pharma ML |

Every module ships with physics-grounded tests: integrators are checked against closed-form solutions, MC pricing against Black-Scholes, MD forces against finite-difference gradients, and surrogates against exact spectral operators.

## Installation

```bash
git clone https://github.com/<your-username>/physweave.git
cd physweave
pip install -e ".[dev]"        # core (NumPy only) + test tools
pip install -e ".[jax]"        # optional: autodiff/GPU via JAX
pip install -e ".[torch]"      # optional: neural operators via PyTorch
```

## Quickstart

```python
import numpy as np

# 1. Differentiable simulation — validate an integrator against theory
from physweave.diffsim import SpringMass, velocity_verlet
sys = SpringMass(k=4.0)
x, v = velocity_verlet(sys.force, np.array([1.0]), np.array([0.0]), 1e-3, 5000)

# 2. Learn a PDE solution operator from data
from physweave.surrogates import FourierRidgeSurrogate, heat_step_spectral
from physweave.surrogates.spectral import random_initial_condition
inputs = np.stack([random_initial_condition(32, seed=s) for s in range(20)])
outputs = np.stack([heat_step_spectral(u, 0.05, 0.1) for u in inputs])
model = FourierRidgeSurrogate().fit(inputs, outputs)

# 3. Price an option by Monte Carlo, with error bars
from physweave.quantfin import mc_european_call, black_scholes_call
price, stderr = mc_european_call(s0=100, k=105, r=0.03, sigma=0.2, t=1.0)
print(price, "vs closed form", black_scholes_call(100, 105, 0.03, 0.2, 1.0))

# 4. Entangle qubits
from physweave.quantum import StatevectorSimulator
bell = StatevectorSimulator(2).h(0).cnot(0, 1)
print(bell.sample(shots=1024))   # ~50/50 between '00' and '11'

# 5. Run molecular dynamics
from physweave.moldyn import LennardJones, run_md
from physweave.moldyn.lennard_jones import lattice_positions
lj = LennardJones(box=6.0)
result = run_md(lj, lattice_positions(3, 6.0), np.zeros((27, 3)), 1e-3, 1000)
```

## Running tests

```bash
pytest -q
```

## Roadmap

- [ ] JAX backend with `jax.grad` through full trajectories (inverse design demos)
- [ ] Fourier Neural Operator (FNO) and transformer surrogates + honest benchmark suite vs classical solvers
- [ ] GPU statevector simulation and tensor-network contraction for 30+ qubits
- [ ] Heston/local-vol models, American options via Longstaff-Schwartz
- [ ] Learned interatomic potentials trained on LJ/DFT datasets
- [ ] Documentation site with physics derivations for every module

## How to contribute (open source workflow)

New to open source? PhysWeave is a friendly place to start. Follow these steps:

1. **Star ⭐ and fork 🍴 the repository.** Starring helps others find the project; forking creates your own copy under your GitHub account to work in.
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/<your-username>/physweave.git
   cd physweave
   ```
3. **Set the upstream remote** so you can stay in sync:
   ```bash
   git remote add upstream https://github.com/<original-owner>/physweave.git
   git fetch upstream
   ```
4. **Create a feature branch** — never work on `main`:
   ```bash
   git checkout -b feat/my-improvement
   ```
5. **Set up the dev environment and make your changes:**
   ```bash
   pip install -e ".[dev]"
   ```
   Keep changes focused: one feature or fix per pull request.
6. **Add or update tests.** Every physics claim needs a test (energy conservation, closed-form validation, gradient checks). Run:
   ```bash
   pytest -q && ruff check .
   ```
7. **Commit with a clear message** using conventional commits:
   ```bash
   git commit -m "feat(quantum): add RY gate and controlled-phase"
   ```
8. **Push to your fork and open a Pull Request:**
   ```bash
   git push origin feat/my-improvement
   ```
   Then open a PR on GitHub against `main`, describing *what* you changed and *why*, with benchmark numbers if performance-related.
9. **Respond to review feedback** — maintainers may request changes; push follow-up commits to the same branch.
10. **Stay in sync** for your next contribution:
    ```bash
    git checkout main && git pull upstream main && git push origin main
    ```

Good first issues are labeled [`good first issue`](../../issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22). See [CONTRIBUTING.md](CONTRIBUTING.md) for code style and review standards.

## License

MIT — see [LICENSE](LICENSE).
