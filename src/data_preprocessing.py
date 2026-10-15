"""Dataset loading and preprocessing for binary Iris classification.

Two binary tasks are provided:

* ``setosa_vs_versicolor`` - the task suggested in the project plan. It is
  linearly separable, so almost any model reaches ~100 % accuracy.
* ``versicolor_vs_virginica`` - a harder, overlapping task where differences
  between models can actually show up.

The scaler is fitted on the TRAINING split only, so no test information leaks
into preprocessing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TASKS = {
    "setosa_vs_versicolor": (0, 1),
    "versicolor_vs_virginica": (1, 2),
}


@dataclass
class Split:
    X_train: torch.Tensor
    X_test: torch.Tensor
    y_train: torch.Tensor
    y_test: torch.Tensor
    scaler: StandardScaler
    feature_names: list[str]


def load_binary_iris(
    task: str = "setosa_vs_versicolor",
    features: tuple[int, ...] | None = None,
    test_size: float = 0.2,
    seed: int = 0,
) -> Split:
    """Return a stratified, standardised train/test split.

    Args:
        task: key of :data:`TASKS`.
        features: indices of the Iris features to keep (default: all four).
        test_size: fraction of samples held out for testing (default 0.2).
        seed: controls the stratified split.
    """
    class_a, class_b = TASKS[task]
    iris = load_iris()
    mask = np.isin(iris.target, [class_a, class_b])
    X = iris.data[mask]
    y = (iris.target[mask] == class_b).astype(np.float32)  # class_b -> label 1
    names = list(iris.feature_names)
    if features is not None:
        X = X[:, list(features)]
        names = [names[i] for i in features]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(X_tr)  # fit on train only -> no leakage
    to_t = lambda a: torch.as_tensor(a, dtype=torch.float32)
    return Split(
        to_t(scaler.transform(X_tr)),
        to_t(scaler.transform(X_te)),
        to_t(y_tr),
        to_t(y_te),
        scaler,
        names,
    )
