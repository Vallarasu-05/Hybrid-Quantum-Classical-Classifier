"""Unit tests: circuit structure, analytic correctness, gradients, no data leakage."""
import numpy as np
import pytest
import torch

from src.data_preprocessing import load_binary_iris
from src.hybrid_model import ClassicalNN, HybridQNN, count_params
from src.quantum_circuit import build_circuit, build_qnn, n_weights


# ---------- circuit structure ----------
@pytest.mark.parametrize("n,L,fr", [(4, 1, True), (4, 2, True), (2, 1, True), (4, 1, False)])
def test_parameter_counts(n, L, fr):
    qc, x, w = build_circuit(n, L, fr)
    assert qc.num_qubits == n
    assert len(x) == n
    assert len(w) == n_weights(n, L, fr)
    assert qc.num_parameters == len(x) + len(w)


def test_cnot_chain_is_linear():
    qc, _, _ = build_circuit(4, 1)
    cx = [tuple(qc.find_bit(q).index for q in inst.qubits) for inst in qc.data if inst.operation.name == "cx"]
    assert cx == [(0, 1), (1, 2), (2, 3)]


# ---------- analytic correctness ----------
def test_matches_analytic_expectation_values():
    """With all trainable weights = 0 the circuit is Ry(x_i) then a CNOT chain.
    Pulling Z_i back through the chain gives <Z_i> = prod_{j<=i} cos(x_j)."""
    n = 4
    qnn = build_qnn(n, 1, True, precision=0.0)
    x = np.array([[0.3, -1.1, 2.0, 0.7]])
    out = qnn.forward(x, np.zeros(n_weights(n, 1, True)))[0]
    expected = np.cumprod(np.cos(x[0]))
    np.testing.assert_allclose(out, expected, atol=1e-9)


def test_ideal_simulation_is_deterministic():
    qnn = build_qnn(4, 1, True, precision=0.0)
    x = np.random.default_rng(0).uniform(-3, 3, (3, 4))
    w = np.random.default_rng(1).uniform(-3, 3, n_weights(4, 1, True))
    np.testing.assert_allclose(qnn.forward(x, w), qnn.forward(x, w))


def test_finite_shots_are_noisy_but_unbiased_ish():
    qnn = build_qnn(4, 1, True, precision=0.05, seed=0)
    x = np.array([[0.3, -1.1, 2.0, 0.7]])
    w = np.zeros(n_weights(4, 1, True))
    ideal = np.cumprod(np.cos(x[0]))
    runs = np.stack([qnn.forward(x, w)[0] for _ in range(200)])
    assert runs.std(axis=0).max() > 1e-3          # genuinely stochastic
    np.testing.assert_allclose(runs.mean(axis=0), ideal, atol=0.02)


# ---------- gradients (the plan's "critical" check) ----------
def _weight_grads(final_rotation, n_layers, seed):
    rng = np.random.default_rng(seed)
    qnn = build_qnn(4, n_layers, final_rotation, precision=0.0)
    x = rng.uniform(-2.5, 2.5, (6, 4))
    w = rng.uniform(-3, 3, n_weights(4, n_layers, final_rotation))
    _, wg = qnn.backward(x, w)           # (batch, outputs, n_weights)
    return np.abs(wg).sum(axis=(0, 1))


def test_every_parameter_has_gradient_with_final_rotation():
    g = _weight_grads(True, 1, seed=0)
    assert (g > 1e-6).all(), g


def test_literal_ansatz_has_dead_rz_parameters():
    """Documents the finding that motivates `final_rotation=True`."""
    g = _weight_grads(False, 1, seed=0)
    rz = g[1::2]                          # indices of Rz weights
    ry = g[0::2]
    assert (rz < 1e-9).all(), rz
    assert (ry > 1e-6).any()


def test_gradients_match_finite_differences():
    qnn = build_qnn(4, 2, True, precision=0.0)
    rng = np.random.default_rng(3)
    x = rng.uniform(-2, 2, (1, 4))
    w = rng.uniform(-2, 2, n_weights(4, 2, True))
    _, wg = qnn.backward(x, w)
    eps = 1e-5
    for k in [0, 5, 9, 15]:
        wp, wm = w.copy(), w.copy()
        wp[k] += eps
        wm[k] -= eps
        fd = (qnn.forward(x, wp) - qnn.forward(x, wm))[0] / (2 * eps)
        np.testing.assert_allclose(wg[0, :, k], fd, atol=1e-5)


def test_end_to_end_gradients_reach_all_three_parameter_groups():
    torch.manual_seed(0)
    model = HybridQNN(4, 1, True, seed=0)
    x = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,)).float()
    torch.nn.functional.binary_cross_entropy_with_logits(model(x), y).backward()
    for name, p in [("classical feature layer", model.pre.weight),
                    ("quantum circuit", model.qnn.weight),
                    ("classical output layer", model.head.weight)]:
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_one_optimizer_step_updates_classical_and_quantum_params():
    torch.manual_seed(0)
    model = HybridQNN(4, 1, True, seed=0)
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    x, y = torch.randn(8, 4), torch.randint(0, 2, (8,)).float()
    torch.nn.functional.binary_cross_entropy_with_logits(model(x), y).backward()
    opt.step()
    changed = {k: not torch.equal(before[k], v) for k, v in model.named_parameters()}
    assert all(changed.values()), changed


# ---------- data handling ----------
def test_split_is_stratified_and_sized():
    d = load_binary_iris("setosa_vs_versicolor", seed=0)
    assert len(d.X_train) == 80 and len(d.X_test) == 20
    assert d.y_train.sum() == 40 and d.y_test.sum() == 10


def test_scaler_fit_on_train_only():
    d = load_binary_iris("versicolor_vs_virginica", seed=1)
    assert torch.allclose(d.X_train.mean(0), torch.zeros(4), atol=1e-5)
    assert not torch.allclose(d.X_test.mean(0), torch.zeros(4), atol=1e-3)


def test_feature_subset_for_two_qubit_model():
    d = load_binary_iris("setosa_vs_versicolor", features=(2, 3))
    assert d.X_train.shape[1] == 2
    assert d.feature_names == ["petal length (cm)", "petal width (cm)"]


def test_model_shapes_and_param_counts():
    assert ClassicalNN(4, 4)(torch.randn(5, 4)).shape == (5,)
    assert HybridQNN(4, 1, True)(torch.randn(5, 4)).shape == (5,)
    assert count_params(ClassicalNN(4, 4)) == 25
    assert count_params(HybridQNN(4, 1, True)) == 20 + 12 + 5
