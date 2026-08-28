"""Modern Qiskit operator aliases used by the tetrahedral folding model.

The original implementation used :mod:`qiskit.opflow`, which was removed in
Qiskit 1.0.  The folding Hamiltonian is diagonal and only needs sums,
compositions, and tensor products of Pauli operators, all of which are provided
directly by :class:`qiskit.quantum_info.SparsePauliOp`.
"""

from qiskit.quantum_info import SparsePauliOp


# Retain the historical type names inside the implementation to keep the
# scientific formulas readable while using a single current Qiskit type.
OperatorBase = SparsePauliOp
PauliOp = SparsePauliOp
PauliSumOp = SparsePauliOp

I = SparsePauliOp("I")
Z = SparsePauliOp("Z")


__all__ = ["I", "Z", "OperatorBase", "PauliOp", "PauliSumOp"]
