# -*- coding: utf-8 -*-
"""
Authored by team: PauliZee
"""
from typing import List
from helpers import get_probs
import numpy as np
from naive import unitarize, make_circuit, simulate, simulate_measurement
from naive_trotter import construct_heisenberg
from helpers import error
from qiskit import IBMQ
from qiskit.compiler import transpile
import time
import config as config
from qiskit.providers import provider
from qiskit.visualization import plot_histogram
from matplotlib import pyplot as plt

state = []
num_qubits = 3

qubits_neighbours = [0, 1, 3, 5, 4]
t, r, noise = 2, 100, np.random.uniform(-1, 1, 7)
noise = [0.0] * max(7, num_qubits)
state = np.array([0, 1, 0, 0, 0, 0, 1, 0])
state = state / np.linalg.norm(state)


def authenticate() -> provider.ProviderV1:
    IBMQ.save_account(config.TOKEN)
    IBMQ.load_account()
    provider = IBMQ.get_provider(
        hub=config.HUB, group=config.GROUP, project=config.PROJECT
    )
    return provider


def initialize(n):
    global noise, qubits_neighbours, t, r, num_qubits, state
    num_qubits = n
    qubits_neighbours = list(range(num_qubits))
    noise = [0.0] * num_qubits
    state = [0] * (2**num_qubits)
    state[0] = 1
    state = np.array(state)
    state = state / np.linalg.norm(state)


def get_naive_trotter_state() -> np.ndarray:
    circuit = construct_heisenberg(num_qubits, qubits_neighbours, t, r, noise, state)

    state_vector = simulate(circuit)
    return state_vector


def simulate_naive_trotter(
    num_qubits, qubits_neighbours, t, r, noise, state
) -> np.ndarray:
    circuit = construct_heisenberg(num_qubits, qubits_neighbours, t, r, noise, state)

    counts = simulate_measurement(circuit)

    print(f"classical measurements:{counts}")

    probs = np.zeros(2**num_qubits)
    for key in counts.keys():
        key_val = int(key, 2)
        probs[key_val] = counts[key]

    print(f"classical {probs=}")
    return probs


def get_naive_state(state) -> np.ndarray:
    unitary = unitarize(t, num_qubits)
    circ = make_circuit(unitary, state, t, num_qubits)

    return simulate(circ)


def run_on_quantum_computer(
    num_qubits, qubits_neighbours, t, r, noise, state, backend_name, provider
) -> np.ndarray:
    backend = provider.get_backend(backend_name)

    circuit = construct_heisenberg(num_qubits, qubits_neighbours, t, r, noise, state)
    transpiled_circuit = transpile(circuit, backend, optimization_level=0)

    print("running circuit")
    result = provider.run_circuits(
        transpiled_circuit, backend_name=backend_name, optimization_level=0
    ).result()
    print("running completed")
    counts = result.get_counts()

    print(f"quantum_computer:{counts}")

    probs = np.zeros(2**num_qubits)
    for key in counts.keys():
        key_val = int(key, 2)
        probs[key_val] = counts[key]

    print(f"quantum {probs=}")
    return probs


def compare_on_quantum_computer(
    range_qubits: List[int], run_quantum=False
) -> np.ndarray:

    backend_name = "ibm_perth"
    provider = None
    if run_quantum:
        print("Queried for auth")
        provider = authenticate()
        print("Auth complete")

    all_probs_c = []
    state_lists = []
    for num_qubits in range_qubits:
        state = np.random.rand(2**num_qubits)
        state = state / np.linalg.norm(state)
        state_lists.append(state)

    for ind, state_vec in enumerate(state_lists):
        num_qubits = range_qubits[ind]
        # cur_neighbours = qubits_neighbours[:num_qubits]
        cur_neighbours = list(range(num_qubits))

        probs_c = simulate_naive_trotter(
            num_qubits, cur_neighbours, t, r, noise, state_vec
        )
        all_probs_c.append(probs_c)

    if not run_quantum:
        return np.array(all_probs_c)

    all_probs_q = []
    num_qubits = 7 # setting the number of qubits to the same as ibm_perth.
    for ind in range_qubits:
        print(f"running for {num_qubits=} on quantum computer")
        cur_neighbours = qubits_neighbours[:ind]
        state = np.random.rand(2**num_qubits)
        state = state / np.linalg.norm(state)
        probs_q = run_on_quantum_computer(
            num_qubits, cur_neighbours, t, r, noise, state, backend_name, provider
        )
        all_probs_q.append(probs_q)

    errors = []
    for ind, q in enumerate(all_probs_q):
        errors.append(error(q, all_probs_c[ind]))

    return np.array(errors)


def normalize(vec):
    if np.linalg.norm(vec) != 0:
        return vec / np.linalg.norm(vec)
    else:
        return 0


def performance(l, r, is_normalize=False):
    timestamps_naive = []
    timestamps_naivetrotter = []
    for n in range(l, r):
        initialize(n)
        start_time = time.time()
        naive_trotter_state = get_naive_trotter_state()
        timestamps_naivetrotter.append(time.time() - start_time)

        start_time = time.time()
        naive_state = get_naive_state(state)
        timestamps_naive.append(time.time() - start_time)

    if is_normalize:
        timestamps_naive = normalize(timestamps_naive)
        timestamps_naivetrotter = normalize(timestamps_naivetrotter)
    plt.plot(range(l, r), timestamps_naivetrotter, label="naive trotter")
    plt.plot(range(l, r), timestamps_naive, label="naive")
    plt.legend()
    plt.xlabel("Number of qubits")
    plt.ylabel("Time")
    plt.title("Hamiltonian Simulation")
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    compare_on_quantum_computer([3], run_quantum=True)
