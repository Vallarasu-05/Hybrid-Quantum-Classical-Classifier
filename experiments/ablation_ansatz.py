"""Ablation - literal textbook ansatz (no final rotation) vs our ansatz.

Unit tests show that without the final Ry layer the Rz parameters have exactly
zero gradient. This ablation measures what that costs in accuracy.
"""
from experiments._runner import run_grid

HARD = "versicolor_vs_virginica"


def main():
    run_grid([
        dict(task=HARD, model="hybrid", n_layers=1, final_rotation=False),
        dict(task=HARD, model="hybrid", n_layers=2, final_rotation=False),
    ], range(3))


if __name__ == "__main__":
    main()
