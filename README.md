# Tetrahedral protein folding for Qiskit 2.x

This is a modernized port of the tetrahedral lattice model from the archived
[`qiskit-community/quantum-protein-folding`](https://github.com/qiskit-community/quantum-protein-folding)
and Qiskit Research projects. It implements the model described in
[*Resource-efficient quantum algorithm for protein folding*](https://www.nature.com/articles/s41534-021-00368-4).

The scientific model is retained while the removed `qiskit.opflow`,
`QuantumInstance`, `execute`, IBMQ Provider, and V1 primitive APIs are not.
Hamiltonians are represented directly by `SparsePauliOp`; algorithms come from
`qiskit-algorithms`; the example notebook uses Aer SamplerV2 locally and IBM
Runtime SamplerV2 on hardware.

## Supported environment

- Python 3.10–3.12
- NumPy 2.0–2.2
- Qiskit 2.0–2.4 (the shared image currently uses 2.2)
- Qiskit Algorithms 0.4.x
- Qiskit Aer 0.17.x
- Qiskit IBM Runtime 0.44.x

These constraints match the QBioCode environment used to build
`thepineapplepirate/qiskit_galaxy:2.0.0`.

## Install

```bash
python -m pip install -e .
```

The modern notebook is
[`docs/protein_folding_qiskit2.ipynb`](docs/protein_folding_qiskit2.ipynb).
The notebook includes simulator/hardware selection, validated warm starts,
convergence plots, ranked sampled conformations, a decoded 3D fold, and XYZ
export. The archived notebook remains in `docs/protein_folding.ipynb` solely
as a scientific and migration reference.

## Regression checks

```bash
python -m pytest -q tests/test_qiskit2_regression.py
```

For the archived `APRLRFY` example, the regression checks require the modern
implementation to reproduce:

- the same 9-qubit, 77-term Hamiltonian;
- representative Pauli coefficients from the archived executed notebook;
- the exact ground energy of `-1.425`;
- a seven-residue decoded structure.

## Resource note

The default example is the historical nine-qubit `APRLRFY` problem. Hardware
mode defaults to two fixed-rate SPSA iterations because each iteration requires
multiple Runtime evaluations. Review the configuration cell before submitting
QPU work.
