import pennylane as qml
from pennylane.tape import QuantumTape

n_qubits = 15

with QuantumTape() as benchmark_tape:
    # Apply RX and RY gates to initialize a subset of qubits
    qml.RX(0.531, wires=0)
    qml.RY(0.9, wires=1)
    qml.RX(0.3, wires=2)
    qml.RX(0.5, wires=3)
    qml.RY(-0.2, wires=4)
    qml.RX(0.7, wires=5)

    # Add entanglement: cascading CZ gates across different groups of qubits
    qml.CZ(wires=(0, 1))
    qml.CZ(wires=(1, 2))
    qml.CZ(wires=(3, 4))
    qml.CZ(wires=(4, 5))

    # Deeper rotations on specific qubits
    qml.RY(-0.4, wires=0)
    qml.RX(0.8, wires=1)
    qml.RY(-0.6, wires=2)
    qml.RX(0.9, wires=3)

    # Add deeper entanglement involving qubits not yet touched
    qml.CZ(wires=(1, 6))
    qml.CZ(wires=(6, 7))
    qml.CZ(wires=(7, 8))
    qml.CZ(wires=(9, 10))
    qml.CZ(wires=(11, 12))

    # Another round of rotations
    qml.RY(0.2, wires=5)
    qml.RX(-0.7, wires=6)
    qml.RY(0.3, wires=7)
    qml.RX(-0.4, wires=8)
    qml.RY(0.9, wires=9)

    # Extend operations to cover all qubits
    for i in range(10, n_qubits):
        qml.RX(0.2 * i, wires=i)
        qml.RY(-0.3 * i, wires=i)
        if i > 1:
            qml.CZ(wires=(i - 1, i))

    # Final measurement
    qml.expval(qml.pauli.string_to_pauli_word("Z" * n_qubits))


with qml.tape.QuantumTape() as benchmark_tape_cut:
    # Apply RX and RY gates to initialize a subset of qubits
    qml.RX(0.531, wires=0)
    qml.RY(0.9, wires=1)
    qml.RX(0.3, wires=2)
    qml.RX(0.5, wires=3)
    qml.RY(-0.2, wires=4)
    qml.RX(0.7, wires=5)

    # Add entanglement: cascading CZ gates across different groups of qubits
    qml.CZ(wires=(0, 1))
    qml.CZ(wires=(1, 2))
    qml.CZ(wires=(3, 4))
    qml.CZ(wires=(4, 5))

    # Deeper rotations on specific qubits
    qml.RY(-0.4, wires=0)
    qml.RX(0.8, wires=1)
    qml.RY(-0.6, wires=2)
    qml.RX(0.9, wires=3)

    # Add WireCut operations
    qml.WireCut(wires=1)
    qml.WireCut(wires=3)

    # Add deeper entanglement involving qubits not yet touched
    qml.CZ(wires=(1, 6))
    qml.CZ(wires=(6, 7))
    qml.CZ(wires=(7, 8))
    qml.CZ(wires=(9, 10))
    qml.CZ(wires=(11, 12))

    # Another round of rotations
    qml.RY(0.2, wires=5)
    qml.RX(-0.7, wires=6)
    qml.RY(0.3, wires=7)
    qml.RX(-0.4, wires=8)
    qml.RY(0.9, wires=9)

    # Extend operations to cover all qubits
    for i in range(10, n_qubits):
        qml.RX(0.2 * i, wires=i)
        qml.RY(-0.3 * i, wires=i)
        if i > 1:
            qml.CZ(wires=(i - 1, i))

    # Final measurement
    qml.expval(qml.pauli.string_to_pauli_word("Z" * n_qubits))

# Alternative test circuit
with QuantumTape() as benchmark_tape_cut_alternative:
    for i in range(n_qubits):
        qml.RX(-0.1 * i, wires=i)  # RX rotation proportional to qubit index
    qml.WireCut(wires=6)
    for i in range(1, n_qubits, 4):
        qml.CNOT(wires=(i, (i + 1) % n_qubits))  # CNOT gates on every fourth qubit
    for i in range(2, n_qubits, 5):
        qml.RZ(0.6 * i, wires=i)  # RZ rotation for every fifth qubit
    qml.expval(qml.pauli.string_to_pauli_word("Z" * n_qubits))
