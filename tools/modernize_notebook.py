"""Generate the user-facing Qiskit 2 tetrahedral folding notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "docs" / "protein_folding_qiskit2.ipynb"


def _lines(source: str) -> list[str]:
    return dedent(source).strip("\n").splitlines(keepends=True)


def markdown(cell_id: str, source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": _lines(source),
    }


def code(cell_id: str, source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": _lines(source),
    }


cells = [
    markdown(
        "overview",
        """
        # Tetrahedral protein folding with Qiskit 2

        This notebook performs variational protein folding on the tetrahedral
        lattice using either Aer or IBM Quantum hardware. Run the simulator
        first to inspect the workflow and create a warm start; hardware mode
        then validates and reuses those optimized parameters.
        """,
    ),
    code(
        "configuration",
        """
        # Primary execution mode. Hardware mode submits real IBM Runtime jobs.
        EXECUTION_MODE = "simulator"  # "simulator" or "hardware"

        # Scientific model. Use one empty string per main-chain residue when no
        # explicit side-chain bead is required.
        PROTEIN_SEQUENCE = "APRLRFY"
        SIDE_CHAINS = [""] * len(PROTEIN_SEQUENCE)
        PENALTY_BACK = 10.0
        PENALTY_CHIRAL = 10.0
        PENALTY_CONTACT = 10.0

        # SamplingVQE configuration. Simulator mode creates the warm start;
        # hardware mode uses SPSA so the QPU evaluation count stays predictable.
        ANSATZ_REPS = 1
        CVAR_AGGREGATION = 0.1
        SIMULATOR_SHOTS = 5_000
        SIMULATOR_MAX_ITER = 50
        HARDWARE_SHOTS = 1_000
        HARDWARE_MAX_ITER = 2
        RANDOM_SEED = 23

        # Warm-start and output files are written beside the launched notebook.
        WARM_START_FILE = "tetrahedral_warm_start.npz"
        LOAD_WARM_START = EXECUTION_MODE == "hardware"
        SAVE_XYZ = True
        XYZ_FILE = "tetrahedral_fold.xyz"
        OVERWRITE_XYZ = True

        # IBM Quantum settings. None chooses the least-busy operational QPU
        # with enough qubits for this Hamiltonian.
        HARDWARE_BACKEND = None
        HARDWARE_OPTIMIZATION_LEVEL = 1
        """,
    ),
    code(
        "imports",
        """
        from pathlib import Path
        from time import perf_counter

        import matplotlib.pyplot as plt
        import numpy as np
        from qiskit.circuit.library import real_amplitudes
        from qiskit.quantum_info import Statevector
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_aer.primitives import SamplerV2 as AerSampler
        from qiskit_algorithms import SamplingVQE
        from qiskit_algorithms.optimizers import COBYLA, SPSA
        from qiskit_algorithms.utils import algorithm_globals
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as RuntimeSampler, Session

        from protein_folding.interactions.miyazawa_jernigan_interaction import (
            MiyazawaJerniganInteraction,
        )
        from protein_folding.penalty_parameters import PenaltyParameters
        from protein_folding.peptide.peptide import Peptide
        from protein_folding.protein_folding_problem import ProteinFoldingProblem
        from protein_folding.utils.result_projection import project_bitstring_distribution

        algorithm_globals.random_seed = RANDOM_SEED
        if EXECUTION_MODE not in {"simulator", "hardware"}:
            raise ValueError("EXECUTION_MODE must be 'simulator' or 'hardware'")
        if len(SIDE_CHAINS) != len(PROTEIN_SEQUENCE):
            raise ValueError("SIDE_CHAINS must contain one entry per main-chain residue")
        """,
    ),
    markdown(
        "build-heading",
        """
        ## Build the tetrahedral Hamiltonian

        The Miyazawa–Jernigan interaction and three penalty terms encode the
        folding objective as a diagonal `SparsePauliOp`.
        """,
    ),
    code(
        "build-problem",
        """
        peptide = Peptide(PROTEIN_SEQUENCE, SIDE_CHAINS)
        penalties = PenaltyParameters(
            PENALTY_CHIRAL, PENALTY_BACK, PENALTY_CONTACT
        )
        protein_folding_problem = ProteinFoldingProblem(
            peptide, MiyazawaJerniganInteraction(), penalties
        )
        qubit_op = protein_folding_problem.qubit_op()
        logical_ansatz = real_amplitudes(
            qubit_op.num_qubits, reps=ANSATZ_REPS, entanglement="linear"
        )
        print(
            f"Sequence: {PROTEIN_SEQUENCE}; Hamiltonian: {qubit_op.num_qubits} "
            f"qubits, {qubit_op.size} Pauli terms; ansatz: "
            f"{logical_ansatz.num_parameters} parameters"
        )
        """,
    ),
    code(
        "warm-start",
        """
        warm_start_path = Path(WARM_START_FILE)
        initial_point = None
        if LOAD_WARM_START:
            if not warm_start_path.exists():
                raise FileNotFoundError(
                    f"{warm_start_path} is missing. Run simulator mode first, or "
                    "set LOAD_WARM_START=False to start hardware optimization randomly."
                )
            with np.load(warm_start_path, allow_pickle=False) as saved:
                expected = {
                    "protein_sequence": PROTEIN_SEQUENCE,
                    "num_qubits": qubit_op.num_qubits,
                    "ansatz_reps": ANSATZ_REPS,
                }
                observed = {
                    "protein_sequence": str(saved["protein_sequence"]),
                    "num_qubits": int(saved["num_qubits"]),
                    "ansatz_reps": int(saved["ansatz_reps"]),
                }
                if observed != expected:
                    raise ValueError(
                        f"Warm-start metadata does not match this run: {observed} != {expected}"
                    )
                initial_point = np.asarray(saved["optimal_point"], dtype=float)
            if initial_point.size != logical_ansatz.num_parameters:
                raise ValueError("Warm-start parameter count does not match the ansatz")
            print(f"Loaded {initial_point.size} parameters from {warm_start_path}")
        """,
    ),
    markdown(
        "vqe-heading",
        """
        ## Run SamplingVQE

        Aer uses COBYLA for the longer warm-start optimization. Hardware uses
        fixed-rate SPSA; this avoids an automatic calibration phase and makes
        the configured QPU iteration budget easier to understand.
        """,
    ),
    code(
        "run-vqe",
        """
        evaluation_counts, energies = [], []

        def store_intermediate_result(eval_count, parameters, mean, metadata):
            evaluation_counts.append(int(eval_count))
            energies.append(float(mean))

        started = perf_counter()
        logical_to_physical = list(range(qubit_op.num_qubits))
        if EXECUTION_MODE == "simulator":
            optimizer = COBYLA(maxiter=SIMULATOR_MAX_ITER)
            sampler = AerSampler(default_shots=SIMULATOR_SHOTS, seed=RANDOM_SEED)
            vqe = SamplingVQE(
                sampler,
                logical_ansatz,
                optimizer,
                initial_point=initial_point,
                aggregation=CVAR_AGGREGATION,
                callback=store_intermediate_result,
            )
            raw_result = vqe.compute_minimum_eigenvalue(qubit_op)
        else:
            optimizer = SPSA(
                maxiter=HARDWARE_MAX_ITER,
                learning_rate=0.05,
                perturbation=0.1,
            )
            service = QiskitRuntimeService()
            backend = (
                service.backend(HARDWARE_BACKEND)
                if HARDWARE_BACKEND
                else service.least_busy(
                    operational=True,
                    simulator=False,
                    min_num_qubits=qubit_op.num_qubits,
                )
            )
            pass_manager = generate_preset_pass_manager(
                backend=backend,
                optimization_level=HARDWARE_OPTIMIZATION_LEVEL,
                seed_transpiler=RANDOM_SEED,
            )
            isa_ansatz = pass_manager.run(logical_ansatz)
            isa_operator = qubit_op.apply_layout(isa_ansatz.layout)
            logical_to_physical = isa_ansatz.layout.final_index_layout(
                filter_ancillas=True
            )
            print(
                f"Hardware mode: {backend.name}; {HARDWARE_SHOTS} shots; "
                f"{HARDWARE_MAX_ITER} SPSA iterations"
            )
            with Session(backend=backend) as session:
                sampler = RuntimeSampler(
                    mode=session, options={"default_shots": HARDWARE_SHOTS}
                )
                vqe = SamplingVQE(
                    sampler,
                    isa_ansatz,
                    optimizer,
                    initial_point=initial_point,
                    aggregation=CVAR_AGGREGATION,
                    callback=store_intermediate_result,
                )
                raw_result = vqe.compute_minimum_eigenvalue(isa_operator)

        eigenstate = raw_result.eigenstate
        if hasattr(eigenstate, "binary_probabilities"):
            probabilities = eigenstate.binary_probabilities()
        elif hasattr(eigenstate, "probabilities_dict"):
            probabilities = eigenstate.probabilities_dict()
        elif isinstance(eigenstate, dict):
            probabilities = eigenstate
        else:
            raise TypeError(
                "Unsupported eigensolver state representation: "
                f"{type(eigenstate).__name__}"
            )
        if EXECUTION_MODE == "hardware":
            physical_width = len(next(iter(probabilities), ""))
            probabilities = project_bitstring_distribution(
                probabilities, logical_to_physical
            )
            raw_result.eigenstate = probabilities
            print(
                f"Projected {physical_width}-bit hardware measurements onto "
                f"the {qubit_op.num_qubits}-bit logical conformer register"
            )

        elapsed = perf_counter() - started
        print(f"SamplingVQE completed in {elapsed:.2f} seconds")
        print(f"Optimal energy: {float(np.real(raw_result.eigenvalue)):.6f}")
        """,
    ),
    code(
        "save-warm-start",
        """
        if EXECUTION_MODE == "simulator":
            np.savez(
                warm_start_path,
                protein_sequence=np.array(PROTEIN_SEQUENCE),
                num_qubits=np.array(qubit_op.num_qubits),
                ansatz_reps=np.array(ANSATZ_REPS),
                optimal_point=np.asarray(raw_result.optimal_point, dtype=float),
            )
            print(f"Saved simulator warm start to {warm_start_path.resolve()}")
        """,
    ),
    markdown("outputs-heading", "## Optimization and folding results"),
    code(
        "convergence-plot",
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(evaluation_counts, energies, marker=".", linewidth=1)
        axes[0].set(title="SamplingVQE convergence", xlabel="Objective evaluation", ylabel="CVaR energy")
        zoom_start = max(0, len(energies) // 2)
        axes[1].plot(evaluation_counts[zoom_start:], energies[zoom_start:], marker=".", linewidth=1)
        axes[1].set(title="Second half of optimization", xlabel="Objective evaluation", ylabel="CVaR energy")
        fig.tight_layout()
        plt.show()
        """,
    ),
    code(
        "ranked-solutions",
        """
        # SamplingVQE returns the optimized bitstring distribution. Display the
        # five most probable conformations and their exact diagonal energies.
        ranked_solutions = sorted(
            probabilities.items(), key=lambda item: item[1], reverse=True
        )[:5]
        print("Top sampled conformations:")
        print("rank  bitstring    probability    energy")
        for rank, (bitstring, probability) in enumerate(ranked_solutions, start=1):
            energy = float(np.real(Statevector.from_label(bitstring).expectation_value(qubit_op)))
            print(f"{rank:>4}  {bitstring:<12} {probability:>10.4f}  {energy:>10.4f}")
        """,
    ),
    code(
        "decode-and-plot",
        """
        result = protein_folding_problem.interpret(raw_result)
        print("Best encoded turn sequence:", result.turn_sequence)
        print("Expanded expression:", result.get_result_binary_vector())
        print("Main-chain turns:", result.protein_shape_decoder.main_turns)
        print("Side-chain turns:", result.protein_shape_decoder.side_turns)
        print("\\nLattice coordinates:")
        print(result.protein_shape_file_gen.get_xyz_data())

        structure_figure = result.get_figure(
            title=f"Tetrahedral fold: {PROTEIN_SEQUENCE}", ticks=False, grid=True
        )
        structure_figure.get_axes()[0].view_init(10, 70)
        plt.show()
        """,
    ),
    code(
        "save-xyz",
        """
        if SAVE_XYZ:
            xyz_path = Path(XYZ_FILE)
            xyz_path.parent.mkdir(parents=True, exist_ok=True)
            result.save_xyz_file(
                name=xyz_path.stem,
                path=str(xyz_path.parent),
                comment=(
                    f"Tetrahedral lattice fold for {PROTEIN_SEQUENCE}; "
                    f"mode={EXECUTION_MODE}; energy={float(np.real(raw_result.eigenvalue)):.8f}"
                ),
                replace=OVERWRITE_XYZ,
            )
            print(f"Saved XYZ coordinates to {xyz_path.resolve()}")
        """,
    ),
    code(
        "versions",
        """
        from importlib.metadata import version

        for package in (
            "numpy", "qiskit", "qiskit-aer", "qiskit-algorithms", "qiskit-ibm-runtime"
        ):
            print(f"{package}: {version(package)}")
        """,
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

TARGET.write_text(json.dumps(notebook, indent=1) + "\n")
print(TARGET)
