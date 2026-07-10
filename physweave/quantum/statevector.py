"""Dense statevector quantum circuit simulator.

The state of n qubits is a complex vector of length 2^n reshaped to
(2,) * n; single- and two-qubit gates are applied with tensor contractions
(np.einsum-free axis manipulation), the same trick GPU simulators like
cuQuantum use at scale.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

_SQ2 = 1.0 / np.sqrt(2.0)

gates = SimpleNamespace(
    I=np.eye(2, dtype=complex),
    X=np.array([[0, 1], [1, 0]], dtype=complex),
    Y=np.array([[0, -1j], [1j, 0]], dtype=complex),
    Z=np.array([[1, 0], [0, -1]], dtype=complex),
    H=np.array([[_SQ2, _SQ2], [_SQ2, -_SQ2]], dtype=complex),
    S=np.array([[1, 0], [0, 1j]], dtype=complex),
    T=np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
)


def rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), -1j * np.sin(theta / 2)
    return np.array([[c, s], [s, c]], dtype=complex)


def rz(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex
    )


class StatevectorSimulator:
    """Simulate an n-qubit register initialised to |0...0>."""

    def __init__(self, n_qubits: int):
        if n_qubits < 1 or n_qubits > 24:
            raise ValueError("n_qubits must be in [1, 24] for a dense simulator.")
        self.n = n_qubits
        self.state = np.zeros((2,) * n_qubits, dtype=complex)
        self.state[(0,) * n_qubits] = 1.0

    # -- gate application -------------------------------------------------
    def apply(self, gate: np.ndarray, qubit: int) -> "StatevectorSimulator":
        """Apply a single-qubit gate to `qubit`."""
        psi = np.moveaxis(self.state, qubit, 0)
        psi = np.tensordot(gate, psi, axes=([1], [0]))
        self.state = np.moveaxis(psi, 0, qubit)
        return self

    def apply_controlled(
        self, gate: np.ndarray, control: int, target: int
    ) -> "StatevectorSimulator":
        """Apply a controlled single-qubit gate (e.g. CNOT = controlled-X)."""
        psi = np.moveaxis(self.state, (control, target), (0, 1))
        psi = psi.copy()
        psi[1] = np.tensordot(gate, psi[1], axes=([1], [0]))
        self.state = np.moveaxis(psi, (0, 1), (control, target))
        return self

    def h(self, q: int):
        return self.apply(gates.H, q)

    def x(self, q: int):
        return self.apply(gates.X, q)

    def cnot(self, control: int, target: int):
        return self.apply_controlled(gates.X, control, target)

    # -- measurement ------------------------------------------------------
    def probabilities(self) -> np.ndarray:
        """Probability of each computational basis state, length 2^n."""
        return (np.abs(self.state) ** 2).reshape(-1)

    def sample(self, shots: int = 1024, seed: int | None = None) -> dict[str, int]:
        """Sample measurement outcomes; keys are bitstrings like '01'."""
        rng = np.random.default_rng(seed)
        probs = self.probabilities()
        outcomes = rng.choice(len(probs), size=shots, p=probs / probs.sum())
        counts: dict[str, int] = {}
        for o in outcomes:
            key = format(o, f"0{self.n}b")
            counts[key] = counts.get(key, 0) + 1
        return counts
