"""Parameterised quantum circuit (PQC) and its Qiskit-ML wrapper.

Circuit layout for ``n`` qubits and ``L`` layers::

    |0>-- Ry(x_i) --[ Ry(theta) Rz(phi) -- CNOT chain ] x L -- Ry(final) -- <Z_i>
        data encoding            trainable layer                 (optional)

Design note (found with the gradient check, see tests/):
With the layout ``Ry, Rz -> CNOT chain -> measure Z``, every <Z_i> pulls back
through the CNOT chain to a pure Z-string, which commutes with Rz. The Rz
parameters then have *exactly zero* gradient (dead parameters) and a single
layer is effectively a product-state circuit. A trainable ``Ry`` layer just
before measurement (``final_rotation=True``, the default) rotates the
measurement basis and makes every parameter live. ``final_rotation=False``
reproduces the literal textbook ansatz and is kept for the ablation study.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN


def n_weights(n_qubits: int, n_layers: int, final_rotation: bool = True) -> int:
    """Number of trainable circuit parameters."""
    return 2 * n_qubits * n_layers + (n_qubits if final_rotation else 0)


def build_circuit(
    n_qubits: int = 4, n_layers: int = 1, final_rotation: bool = True
) -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """Build the data-encoding + variational circuit.

    Returns ``(circuit, input_params, weight_params)``.
    """
    x = ParameterVector("x", n_qubits)
    w = ParameterVector("w", n_weights(n_qubits, n_layers, final_rotation))
    qc = QuantumCircuit(n_qubits, name="hybrid_pqc")

    for i in range(n_qubits):  # data encoding: Ry(x_i)|0>
        qc.ry(x[i], i)

    k = 0
    for _ in range(n_layers):
        qc.barrier()
        for i in range(n_qubits):  # trainable rotations
            qc.ry(w[k], i)
            qc.rz(w[k + 1], i)
            k += 2
        for i in range(n_qubits - 1):  # linear entanglement
            qc.cx(i, i + 1)

    if final_rotation:
        qc.barrier()
        for i in range(n_qubits):
            qc.ry(w[k], i)
            k += 1
    assert k == len(w)
    return qc, x, w


def z_observables(n_qubits: int) -> list[SparsePauliOp]:
    """One Pauli-Z observable per qubit: q_i = <Z_i> = P(0) - P(1)."""
    return [
        SparsePauliOp.from_sparse_list([("Z", [i], 1.0)], num_qubits=n_qubits)
        for i in range(n_qubits)
    ]


def build_qnn(
    n_qubits: int = 4,
    n_layers: int = 1,
    final_rotation: bool = True,
    precision: float = 0.0,
    estimator=None,
    seed: int | None = None,
) -> EstimatorQNN:
    """Wrap the circuit as a Qiskit-ML ``EstimatorQNN``.

    ``precision=0`` gives exact (ideal) expectation values. A positive
    ``precision`` eps emulates finite-shot estimation with about 1/eps^2 shots
    per expectation value, for forward passes AND parameter-shift gradients.

    NOTE 1: EstimatorQNN's own default precision is 1/64, i.e. it silently adds
    shot noise. We always pass the precision explicitly.

    NOTE 2: ``StatevectorEstimator`` models finite-shot noise as Gaussian with
    std = precision (a standard approximation of binomial shot noise). Given an
    *integer* seed it re-creates its RNG on every call, which would replay the
    identical noise pattern each time. We therefore hand it a persistent
    ``Generator`` so noise is independent between calls yet reproducible.
    """
    qc, x, w = build_circuit(n_qubits, n_layers, final_rotation)
    if estimator is None:
        rng = np.random.default_rng(seed) if seed is not None else None
        estimator = StatevectorEstimator(default_precision=precision, seed=rng)
    return EstimatorQNN(
        circuit=qc,
        estimator=estimator,
        observables=z_observables(n_qubits),
        input_params=list(x),
        weight_params=list(w),
        input_gradients=True,  # needed so the classical layer BEFORE the PQC gets gradients
        default_precision=precision,
    )


def shots_to_precision(shots: int) -> float:
    return 1.0 / float(np.sqrt(shots))
