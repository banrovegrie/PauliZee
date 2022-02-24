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
import time
# import config as config
from qiskit.providers import provider
from matplotlib import pyplot as plt

state = []
num_qubits = 3
qubits_neighbours = list(range(3))
t, r, noise = 2, 1000, np.random.uniform(-1, 1, 7)


def initialize(n):
    global noise, qubits_neighbours, t, r, num_qubits, state
    num_qubits = n
    qubits_neighbours = list(range(num_qubits))
    noise = [0.0] * num_qubits
    state = [0] * (2 ** num_qubits)
    state[0] = 1
    state = np.array(state)
    state = state / np.linalg.norm(state)


# def authenticate() -> provider.ProviderV1:
#     IBMQ.save_account(config.TOKEN)
#     IBMQ.load_account()
#     provider = IBMQ.get_provider(
#         hub=config.HUB, group=config.GROUP, project=config.PROJECT
#     )
#     return provider


def get_naive_trotter_state() -> np.ndarray:
    circuit = construct_heisenberg(
        num_qubits, qubits_neighbours, t, r, noise, state)

    state_vector = simulate(circuit)
    return state_vector


def get_naive_state() -> np.ndarray:
    unitary = unitarize(t, num_qubits)
    circ = make_circuit(unitary, state, t, num_qubits)

    return simulate(circ)


def normalize(vec):
    if np.linalg.norm(vec) != 0:
        return vec / np.linalg.norm(vec)
    else:
        return 0


def performance(l, r, normalize=False):
    timestamps_naive = []
    timestamps_naivetrotter = []
    for n in range(l, r):
        initialize(n)
        start_time = time.time()
        naive_trotter_state = get_naive_trotter_state()
        timestamps_naivetrotter.append(time.time() - start_time)

        start_time = time.time()
        naive_state = get_naive_state()
        timestamps_naive.append(time.time() - start_time)

    # if normalize:
    #     timestamps_naive = normalize(timestamps_naive)
    #     timestamps_naivetrotter = normalize(timestamps_naivetrotter)
    plt.plot(range(l, r), timestamps_naivetrotter, label="naive trotter")
    plt.plot(range(l, r), timestamps_naive, label="naive")
    plt.legend()
    plt.xlabel("Number of qubits")
    plt.ylabel("Time")
    plt.title("Hamiltonian Simulation")
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)

    performance(3, 11)
    # start_time = time.time()
    # naive_trotter_state = get_naive_trotter_state()
    # naive_trotter_time = time.time() - start_time

    # start_time = time.time()
    # naive_state = get_naive_state()
    # naive_time = time.time() - start_time

    # print(f"naive trotter state: {naive_trotter_state}")
    # print(f"naive classical state: {naive_state}")
    # print(f"error: {error(naive_state, naive_trotter_state)}")

    # print(f"Naive trotter time taken: {naive_trotter_time}")
    # print(f"Naive time taken: {naive_time}")
