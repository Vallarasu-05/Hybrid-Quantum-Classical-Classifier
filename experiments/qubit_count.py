"""Experiment 3 - 2 qubits (petal length + petal width only) vs 4 qubits (all features)."""
from experiments._runner import run_grid

HARD = "versicolor_vs_virginica"


def main():
    petal = (2, 3)  # petal length, petal width
    run_grid([
        dict(task=HARD, model="classical", features=petal, n_qubits=2, hidden=2),
        dict(task=HARD, model="hybrid", features=petal, n_qubits=2),
    ], range(5))


if __name__ == "__main__":
    main()
