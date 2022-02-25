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
from qiskit.visualization import plot_histogram

num_qubits = 3
qubits_neighbours = [0, 1, 2, 3, 5, 4]
time, r, noise = 2, 1000, np.random.uniform(-1, 1, 7)
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


def get_naive_trotter_state() -> np.ndarray:
    circuit = construct_heisenberg(num_qubits, qubits_neighbours, time, r, noise, state)

    state_vector = simulate(circuit)
    return state_vector


def get_naive_state(state) -> np.ndarray:
    unitary = unitarize(time, num_qubits)
    circ = make_circuit(unitary, state, time, num_qubits)

    return simulate(circ)

def run_on_quantum_computer(range_qubits: List[int]) -> np.ndarray:

    provider = authenticate()
    backend_name = 'ibm_perth'
    backend = provider.get_backend(backend_name)

    all_probs_q = []
    for num_qubits in range_qubits:
        cur_neighbours = qubits_neighbours[:num_qubits]
        state = np.random.rand(2 ** num_qubits)
        state = state / np.linalg.norm(state)

        # quantum circuit 
        circuit = construct_heisenberg(num_qubits, cur_neighbours, time, r, noise, state)
        result = provider.run_circuits(circuit, backend_name=backend_name, optimization_level=0).result()
        counts = result.get_counts()
        probs = np.zeros(2 ** num_qubits)
        for key in counts.keys():
            key_val = int(key, 2)
            probs[key_val] = counts[key]
        
        all_probs_q.append(probs)


    all_probs_c = []
    for num_qubits in range_qubits:
        naive_state = get_naive_state()

            


if __name__ == "__main__":
    np.random.seed(42)
    # naive_trotter_state = get_naive_trotter_state()
    naive_state = get_naive_state()

    # print(f"naive trotter state: {naive_trotter_state}")
    print(f"naive classical state: {naive_state}")
    # print(f"error: {error(naive_state, naive_trotter_state)}")
