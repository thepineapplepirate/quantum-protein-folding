# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Utilities for projecting hardware measurements onto logical qubits."""

from collections.abc import Mapping, Sequence


def project_bitstring_distribution(
    probabilities: Mapping[str, float],
    logical_to_physical: Sequence[int],
) -> dict[str, float]:
    """Project physical-register probabilities into logical Qiskit bit order.

    ``logical_to_physical[q]`` is the physical bit position containing logical
    qubit ``q``. Input and output strings use Qiskit's displayed, big-endian
    convention, where bit position zero is the rightmost character.
    Probabilities that differ only on nonlogical physical bits are aggregated.
    """
    positions = tuple(int(position) for position in logical_to_physical)
    if len(set(positions)) != len(positions) or any(position < 0 for position in positions):
        raise ValueError("Logical-to-physical positions must be unique and nonnegative")

    projected: dict[str, float] = {}
    for physical_bitstring, probability in probabilities.items():
        if set(physical_bitstring) - {"0", "1"}:
            raise ValueError("Measurement keys must be binary strings")
        if positions and max(positions) >= len(physical_bitstring):
            raise ValueError("A physical position exceeds the measurement width")
        logical_bitstring = "".join(
            physical_bitstring[-1 - positions[logical_qubit]]
            for logical_qubit in reversed(range(len(positions)))
        )
        projected[logical_bitstring] = (
            projected.get(logical_bitstring, 0.0) + float(probability)
        )
    return projected


__all__ = ["project_bitstring_distribution"]
