# -*- coding: utf-8 -*-
"""
Authored by team: PauliZee
"""
from helpers import error
import numpy as np
from naive import unitarize, make_circuit, simulate
from naive_trotter import construct_heisenberg
from helpers import error
from qiskit import IBMQ
import config as config
from qiskit.providers import provider

num_qubits = 3
qubits_neighbours = list(range(num_qubits))
time, r, noise = 2, 1000, np.random.uniform(-1, 1, 7)
noise = [0.0] * num_qubits
state = np.array([0, 1, 0, 0, 0, 0, 1, 0])
state = state / np.linalg.norm(state)


def authenticate() -> provider.ProviderV1:
    IBMQ.save_account(config.TOKEN)
    IBMQ.load_account()
    provider = IBMQ.get_provider(
        hub=config.HUB, group=config.GROUP, project=config.PROJECT
    )
    return provider


def get_naive_trotter_state() -> np.ndarray:
    circuit = construct_heisenberg(num_qubits, qubits_neighbours, time, r, noise, state)

    state_vector = simulate(circuit)
    return state_vector


def get_naive_state() -> np.ndarray:
    unitary = unitarize(time, num_qubits)
    circ = make_circuit(unitary, state, time, num_qubits)

    return simulate(circ)


if __name__ == "__main__":
    np.random.seed(42)
    naive_trotter_state = get_naive_trotter_state()
    naive_state = get_naive_state()

    print(f"naive trotter state: {naive_trotter_state}")
    print(f"naive classical state: {naive_state}")
    print(f"error: {error(naive_state, naive_trotter_state)}")
