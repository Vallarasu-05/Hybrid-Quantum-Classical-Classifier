"""Experiment 2 - one vs two variational layers (hard task, everything else fixed)."""
from experiments._runner import run_grid

HARD = "versicolor_vs_virginica"


def main():
    run_grid([dict(task=HARD, model="hybrid", n_layers=2)], range(3))  # L=1 comes from Exp. 1


if __name__ == "__main__":
    main()
