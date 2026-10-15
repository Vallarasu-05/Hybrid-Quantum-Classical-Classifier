"""Shared training loop (identical protocol for classical and hybrid models)."""
from __future__ import annotations

import random
import time

import numpy as np
import torch
import torch.nn as nn

from .data_preprocessing import Split


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(
    model: nn.Module,
    data: Split,
    epochs: int = 20,
    lr: float = 0.05,
    batch_size: int = 16,
    seed: int = 0,
    verbose: bool = False,
) -> dict:
    """Train with Adam + binary cross-entropy. No early stopping / model
    selection on the test set: the model after the last epoch is reported.

    Test loss/accuracy are logged per epoch ONLY for plotting learning curves.
    """
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(data.X_train)
    hist = {"train_loss": [], "test_loss": [], "test_acc": []}

    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        running = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(data.X_train[idx]), data.y_train[idx])
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
        hist["train_loss"].append(running / n)

        model.eval()
        with torch.no_grad():
            logits = model(data.X_test)
            hist["test_loss"].append(loss_fn(logits, data.y_test).item())
            hist["test_acc"].append(((logits > 0).float() == data.y_test).float().mean().item())
        if verbose:
            print(f"epoch {epoch+1:3d} train_loss={hist['train_loss'][-1]:.4f} "
                  f"test_acc={hist['test_acc'][-1]:.3f}")
    hist["train_time_s"] = time.perf_counter() - t0
    return hist
