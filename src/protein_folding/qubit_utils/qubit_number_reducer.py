# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0.
# Modified to use SparsePauliOp with Qiskit 2.x.
"""Remove qubit registers that are irrelevant to the folding Hamiltonian."""

import numpy as np
from qiskit.quantum_info import PauliList, SparsePauliOp


def remove_unused_qubits(
    total_hamiltonian: SparsePauliOp,
) -> tuple[SparsePauliOp, list[int]]:
    """Compress identity-only qubits out of a diagonal Hamiltonian."""
    unused_qubits = _find_unused_qubits(total_hamiltonian)
    if not unused_qubits:
        return total_hamiltonian.simplify(), []

    keep = [
        index
        for index in range(total_hamiltonian.num_qubits)
        if index not in unused_qubits
    ]
    new_tables_z = [term.paulis.z[0][keep] for term in total_hamiltonian]
    new_tables_x = [term.paulis.x[0][keep] for term in total_hamiltonian]
    new_coeffs = [term.coeffs[0] for term in total_hamiltonian]

    paulis = PauliList.from_symplectic(new_tables_z, new_tables_x)
    compressed = SparsePauliOp(paulis, coeffs=new_coeffs).simplify()
    return compressed, unused_qubits


def _find_unused_qubits(total_hamiltonian: SparsePauliOp) -> list[int]:
    """Return qubits represented by identity in every Hamiltonian term."""
    used = np.any(total_hamiltonian.paulis.x | total_hamiltonian.paulis.z, axis=0)
    return np.flatnonzero(~used).tolist()
