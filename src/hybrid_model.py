"""Model A (classical NN) and Model B (hybrid quantum-classical NN)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from qiskit_machine_learning.connectors import TorchConnector

from .quantum_circuit import build_qnn, n_weights


class ClassicalNN(nn.Module):
    """Model A: Linear -> tanh -> Linear (logit)."""

    def __init__(self, n_features: int = 4, hidden: int = 4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class HybridQNN(nn.Module):
    """Model B: classical feature layer -> PQC -> classical output layer.

    x --Linear--> z --pi*tanh--> angles --[Ry encode + PQC]--> <Z_i> --Linear--> logit

    The ``pi * tanh`` maps the classical layer's output into [-pi, pi] so it is a
    valid rotation-angle range. ``forward`` returns logits; apply a sigmoid for
    p(y=1|x) (the sigmoid is folded into ``BCEWithLogitsLoss`` for stability).
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 1,
        final_rotation: bool = True,
        precision: float = 0.0,
        seed: int = 0,
    ):
        super().__init__()
        self.n_qubits, self.n_layers = n_qubits, n_layers
        self.final_rotation = final_rotation
        self.pre = nn.Linear(n_qubits, n_qubits)
        self.qnn_core = build_qnn(n_qubits, n_layers, final_rotation, precision, seed=seed)
        rng = np.random.default_rng(seed)
        w0 = rng.uniform(-np.pi, np.pi, n_weights(n_qubits, n_layers, final_rotation))
        self.qnn = TorchConnector(self.qnn_core, initial_weights=w0)
        self.head = nn.Linear(n_qubits, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return np.pi * torch.tanh(self.pre(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.qnn(self.encode(x))  # (batch, n_qubits) of <Z_i>
        return self.head(q).squeeze(-1)

    @torch.no_grad()
    def forward_with_qnn(self, x: torch.Tensor, alt_qnn) -> torch.Tensor:
        """Inference using the trained weights but a DIFFERENT quantum back-end
        (finite shots, noisy simulator, ...). Used by the noise study."""
        angles = self.encode(x).numpy()
        q = alt_qnn.forward(angles, self.qnn.weight.detach().numpy())
        return self.head(torch.as_tensor(q, dtype=torch.float32)).squeeze(-1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
