import numpy as np

from physweave.quantum import StatevectorSimulator


def test_bell_state():
    sim = StatevectorSimulator(2)
    sim.h(0).cnot(0, 1)
    probs = sim.probabilities()
    assert abs(probs[0] - 0.5) < 1e-12  # |00>
    assert abs(probs[3] - 0.5) < 1e-12  # |11>
    assert probs[1] < 1e-12 and probs[2] < 1e-12


def test_ghz_sampling():
    sim = StatevectorSimulator(3)
    sim.h(0).cnot(0, 1).cnot(1, 2)
    counts = sim.sample(shots=4096, seed=0)
    assert set(counts) == {"000", "111"}
    assert abs(counts["000"] / 4096 - 0.5) < 0.05


def test_x_gate_flips():
    sim = StatevectorSimulator(1)
    sim.x(0)
    assert np.allclose(sim.probabilities(), [0.0, 1.0])


def test_state_normalized_after_circuit():
    sim = StatevectorSimulator(4)
    sim.h(0).h(1).cnot(0, 2).cnot(1, 3).h(2)
    assert abs(np.sum(sim.probabilities()) - 1.0) < 1e-12
