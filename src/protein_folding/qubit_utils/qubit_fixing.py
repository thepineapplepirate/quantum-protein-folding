# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0.
# Modified to use SparsePauliOp with Qiskit 2.x.
"""Replace symmetry-fixed turn qubits with their predefined values."""

import numpy as np
from qiskit.quantum_info import PauliList, SparsePauliOp


def _fix_qubits(
    operator: SparsePauliOp | int,
    has_side_chain_second_bead: bool = False,
) -> SparsePauliOp | int:
    """Apply the tetrahedral model's symmetry-fixed turn-qubit values."""
    if not isinstance(operator, SparsePauliOp):
        return operator

    operator = operator.simplify()
    # Opflow's PauliOp path fixed the bits without applying eigenvalue signs;
    # preserve that historical behavior for a single Pauli term.
    if operator.size == 1:
        table_z = np.copy(operator.paulis.z[0])
        table_x = np.copy(operator.paulis.x[0])
        _preset_binary_vals(table_z, has_side_chain_second_bead)
        pauli = PauliList.from_symplectic([table_z], [table_x])
        return SparsePauliOp(pauli, coeffs=operator.coeffs)

    new_tables_x = []
    new_tables_z = []
    new_coeffs = []
    for term in operator:
        table_z = np.copy(term.paulis.z[0])
        table_x = np.copy(term.paulis.x[0])
        coefficient = _calc_updated_coeff(
            term.coeffs[0], table_z, has_side_chain_second_bead
        )
        _preset_binary_vals(table_z, has_side_chain_second_bead)
        new_tables_x.append(table_x)
        new_tables_z.append(table_z)
        new_coeffs.append(coefficient)

    paulis = PauliList.from_symplectic(new_tables_z, new_tables_x)
    return SparsePauliOp(paulis, coeffs=new_coeffs).simplify()


def _calc_updated_coeff(
    coefficient: complex,
    table_z: np.ndarray,
    has_side_chain_second_bead: bool,
) -> complex:
    """Account for qubits fixed to the one eigenstate before removing them."""
    if len(table_z) > 1 and table_z[1]:
        coefficient = -coefficient
    if not has_side_chain_second_bead and len(table_z) > 5 and table_z[5]:
        coefficient = -coefficient
    return coefficient


def _preset_binary_vals(
    table_z: np.ndarray, has_side_chain_second_bead: bool
) -> None:
    fixed_indices = [0, 1, 2, 3]
    if not has_side_chain_second_bead:
        fixed_indices.append(5)
    for index in fixed_indices:
        if index < len(table_z):
            table_z[index] = False
