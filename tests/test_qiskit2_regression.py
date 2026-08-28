"""Scientific regression tests for the Qiskit 2.x tetrahedral port."""

import json
from pathlib import Path

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import NumPyMinimumEigensolver

from protein_folding.interactions.miyazawa_jernigan_interaction import (
    MiyazawaJerniganInteraction,
)
from protein_folding.penalty_parameters import PenaltyParameters
from protein_folding.peptide.peptide import Peptide
from protein_folding.protein_folding_problem import ProteinFoldingProblem
from protein_folding.utils.result_projection import project_bitstring_distribution


def test_modern_notebook_code_cells_compile() -> None:
    """Every generated code cell must remain valid standalone Python."""
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "protein_folding_qiskit2.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook:{cell['id']}", "exec")


def test_physical_measurements_project_to_logical_bitstrings() -> None:
    """Backend-width strings must aggregate onto the logical register."""
    probabilities = {
        "100101": 0.25,
        "000101": 0.50,
        "011000": 0.25,
    }
    # q0 -> physical bit 0, q1 -> bit 2, q2 -> bit 4.
    projected = project_bitstring_distribution(probabilities, [0, 2, 4])
    assert projected == {"011": 0.75, "100": 0.25}


def reference_problem() -> ProteinFoldingProblem:
    """Return the APRLRFY example from the archived notebook."""
    return ProteinFoldingProblem(
        Peptide("APRLRFY", [""] * 7),
        MiyazawaJerniganInteraction(),
        PenaltyParameters(10, 10, 10),
    )


def test_reference_hamiltonian_matches_archived_notebook() -> None:
    """The modern operator must retain the historical Hamiltonian exactly."""
    operator = reference_problem().qubit_op()

    assert isinstance(operator, SparsePauliOp)
    assert operator.num_qubits == 9
    assert operator.size == 77

    coefficients = dict(operator.to_list())
    expected = {
        "IIIIIIIII": 1613.5895,
        "IIIIIIZII": 487.5,
        "IIIIIIIZZ": -192.5,
        "IZIIIIIII": -904.2875,
        "ZIIIIIIII": -701.802,
        "ZIIIIIIIZ": 5.0,
    }
    for label, coefficient in expected.items():
        assert np.isclose(coefficients[label], coefficient)


def test_reference_ground_energy_and_interpretation() -> None:
    """The exact solver must reproduce the notebook's minimum energy."""
    problem = reference_problem()
    raw_result = NumPyMinimumEigensolver().compute_minimum_eigenvalue(
        problem.qubit_op()
    )

    assert np.isclose(raw_result.eigenvalue.real, -1.425)
    interpreted = problem.interpret(raw_result)
    assert len(interpreted.protein_shape_decoder.main_turns) == 6
    assert len(interpreted.protein_shape_file_gen.get_xyz_data()) == 7


def test_reference_fold_without_side_chains_can_be_plotted() -> None:
    """The default example must plot even when it has no side-chain beads."""
    problem = reference_problem()
    raw_result = NumPyMinimumEigensolver().compute_minimum_eigenvalue(
        problem.qubit_op()
    )
    figure = problem.interpret(raw_result).get_figure()
    assert len(figure.axes) == 1
