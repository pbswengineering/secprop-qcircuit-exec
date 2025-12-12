"""
A 15-qubit benchmark circuit exported as a QuantumTape:
   - n_qubits: qubits number of the benchmark circuit
   - benchmark_tape: benchmark circuit encoded as a QuantumTape
   - benchmark_tape_cut: benchmark circuit encoded as a QuantumTape
     with WireCut operations
"""

import pennylane as qml
import numpy as np

n_qubits = 3

with qml.tape.QuantumTape() as benchmark_tape:
    qml.RX(0.531, wires=0)
    qml.RY(0.9, wires=1)
    qml.RX(0.3, wires=2)
    qml.CZ(wires=(0, 1))
    qml.RY(-0.4, wires=0)
    qml.CZ(wires=[1, 2])
    qml.RY(-0.4, wires=1)
    qml.RY(-0.4, wires=0)
    qml.expval(qml.pauli.string_to_pauli_word("Z" * n_qubits))

with qml.tape.QuantumTape() as benchmark_tape_cut:
    qml.RX(0.531, wires=0)
    qml.RY(0.9, wires=1)
    qml.RX(0.3, wires=2)
    qml.CZ(wires=(0, 1))
    qml.RY(-0.4, wires=0)
    qml.WireCut(wires=1)
    qml.CZ(wires=[1, 2])
    qml.RY(-0.4, wires=1)
    qml.RY(-0.4, wires=0)
    qml.expval(qml.pauli.string_to_pauli_word("Z" * n_qubits))
