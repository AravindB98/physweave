# Contributing to PhysWeave

Thank you for considering a contribution! Before anything else: **star ⭐ and fork 🍴 the repo** — forking is step one of the workflow below.

## Ground rules

- Every physics claim must be backed by a test: closed-form validation, energy conservation, gradient checks against finite differences, or statistical bounds with error bars.
- Core code stays NumPy-only. JAX/PyTorch code lives behind the `jax`/`torch` extras and must degrade gracefully when not installed.
- Public functions need docstrings stating units, conventions, and shapes.
- Run `pytest -q` and `ruff check .` before opening a PR.

## Workflow

1. Fork → clone → `git remote add upstream <original repo>`.
2. Branch from `main`: `git checkout -b feat/<name>` or `fix/<name>`.
3. Make focused changes with tests.
4. Use conventional commit messages: `feat(scope): ...`, `fix(scope): ...`, `docs: ...`, `test: ...`, `perf(scope): ...`.
5. Push to your fork, open a PR against `main` with a clear description (and benchmarks for performance changes).
6. Address review comments with follow-up commits (no force-push during review).

## Where help is wanted

- JAX backends and end-to-end differentiability demos
- Neural operator implementations (FNO, DeepONet) + benchmark harness
- Quantum: more gates, noise channels, tensor-network backend
- Finance: exotic payoffs, Greeks via pathwise/likelihood-ratio methods
- MD: neighbor lists, thermostats (Langevin, Nosé-Hoover), learned potentials
- Docs, tutorials, and physics derivation notes

## Reporting bugs

Open an issue with a minimal reproducible example, expected vs actual behavior, and your Python/NumPy versions.
