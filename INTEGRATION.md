# Galaxy image integration

## Shared baseline

Use QBioCode's pinned Qiskit stack as the single source of truth:

```text
qiskit==2.2.0
qiskit-aer==0.17.0
qiskit-algorithms==0.4.0
qiskit-ibm-runtime==0.44.0
qiskit-ibm-transpiler==0.11.0
qiskit-machine-learning==0.9.0
qiskit-nature==0.7.2
```

Install QBioCode first, then install this package normally. Its declared
requirements are compatible with that baseline.

## FCC lattice package

The FCC `qiskit-2-modernization` branch is compatible with this baseline and
NumPy 2. Ray remains an optional FCC extra; the Galaxy image should install
FCC as `.[ray]` when that accelerator is meant to be available.

## QTF

The inspected QTF dependency ranges are compatible with QBioCode's versions:

```text
qiskit>=1.0
qiskit-aer>=0.14
qiskit-ibm-runtime>=0.20
```

QTF 0.4.4 is compatible when installed with pheat 0.2.0. Its Galaxy
compatibility branch constrains pandas to the 2.x line required by bqplot.

## Suggested build order

1. Install QBioCode and its `apps` extra.
2. Install pheat and QTF from their local checkouts.
3. Install this tetrahedral package normally.
4. Install the FCC package with its `ray` extra.
5. Run `python -m pip check`.
6. Run each project's smoke/regression tests in the completed image.

Avoid cloning mutable `HEAD` during a release build. Pin every source project
to a commit and record the resulting container digest.
